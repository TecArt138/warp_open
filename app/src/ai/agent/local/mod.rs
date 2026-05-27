mod anthropic;
mod openai;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use anyhow::anyhow;
use async_stream::stream;
use futures_util::StreamExt;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use serde_json::json;
use uuid::Uuid;
use warp_multi_agent_api as api;

use crate::ai::agent::AIAgentInput;
use crate::ai::agent::api::{LocalAgentConfig, RequestParams, ResponseStream};
use crate::server::server_api::AIApiError;

const RUN_SHELL_COMMAND_TOOL_NAME: &str = "run_shell_command";
const CALL_MCP_TOOL_NAME: &str = "call_mcp_tool";
const READ_MCP_RESOURCE_TOOL_NAME: &str = "read_mcp_resource";
const READ_SKILL_TOOL_NAME: &str = "read_skill";
const ANTHROPIC_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LocalProviderKind {
    OpenAICompatible,
    AnthropicMessages,
}

pub fn local_agent_stream(
    config: LocalAgentConfig,
    params: RequestParams,
    cancellation_rx: futures::channel::oneshot::Receiver<()>,
) -> ResponseStream {
    let request_id = Uuid::new_v4().to_string();
    let task_id = params
        .tasks
        .first()
        .map(|task| task.id.clone())
        .unwrap_or_else(|| "root-task".to_string());
    let chat_messages = extract_chat_messages(&params, &task_id);
    let provider_kind = local_provider_kind(&config.endpoint_url);

    let output_stream = stream! {
        yield Ok(stream_init_event(&request_id));

        let client = reqwest::Client::new();
        let mut full_text = String::new();
        let mut pending_tool_calls: BTreeMap<usize, PendingToolCall> = BTreeMap::new();

        match provider_kind {
            LocalProviderKind::OpenAICompatible => {
                let request = openai::ChatRequest {
                    model: config.model_name.clone(),
                    messages: chat_messages.clone(),
                    tools: local_tool_definitions(),
                    stream: true,
                };
                let mut sse_stream = openai::stream_chat_completions(
                    client.clone(),
                    &config.endpoint_url,
                    config.api_key.as_deref(),
                    &request,
                );

                while let Some(chunk_result) = sse_stream.next().await {
                    let data = match chunk_result {
                        Ok(data) => data,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };

                    let parsed_chunk = match openai::parse_chunk(&data) {
                        Ok(chunk) => chunk,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };

                    let Some(chunk) = parsed_chunk else {
                        break;
                    };

                    for choice in chunk.choices {
                        if let Some(content) = choice.delta.content {
                            full_text.push_str(&content);
                        }
                        for tool_delta in choice.delta.tool_calls {
                            merge_tool_call_delta(&mut pending_tool_calls, tool_delta);
                        }
                        if matches!(choice.finish_reason.as_deref(), Some("content_filter")) {
                            yield Err(Arc::new(AIApiError::Other(anyhow!(
                                "Local model response was blocked by a content filter.",
                            ))));
                            return;
                        }
                    }
                }
            }
            LocalProviderKind::AnthropicMessages => {
                let request = anthropic::MessagesRequest {
                    model: config.model_name.clone(),
                    messages: anthropic_messages_from_chat_messages(&chat_messages),
                    tools: anthropic_tool_definitions(),
                    max_tokens: ANTHROPIC_MAX_TOKENS,
                    stream: true,
                };
                let mut sse_stream = anthropic::stream_messages(
                    client,
                    &config.endpoint_url,
                    config.api_key.as_deref(),
                    &request,
                );

                while let Some(chunk_result) = sse_stream.next().await {
                    let data = match chunk_result {
                        Ok(data) => data,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };

                    let parsed_event = match anthropic::parse_stream_event(&data) {
                        Ok(event) => event,
                        Err(e) => {
                            yield Err(e);
                            return;
                        }
                    };

                    let Some(event) = parsed_event else {
                        continue;
                    };

                    match event {
                        anthropic::StreamEvent::TextDelta(delta) => {
                            full_text.push_str(&delta);
                        }
                        anthropic::StreamEvent::ToolUseStart { index, id, name, input } => {
                            let entry = pending_tool_calls.entry(index).or_default();
                            if !id.is_empty() {
                                entry.id = id;
                            }
                            if !name.is_empty() {
                                entry.name = name;
                            }
                            if let Some(input) = input {
                                entry.arguments = if input
                                    .as_object()
                                    .is_some_and(|object| object.is_empty())
                                {
                                    String::new()
                                } else {
                                    normalize_tool_input_json(input)
                                };
                            }
                        }
                        anthropic::StreamEvent::ToolUseInputDelta { index, partial_json } => {
                            let entry = pending_tool_calls.entry(index).or_default();
                            entry.arguments.push_str(&partial_json);
                        }
                        anthropic::StreamEvent::ToolUseStop { index } => {
                            if let Some(entry) = pending_tool_calls.get_mut(&index) {
                                if entry.arguments.trim().is_empty() {
                                    entry.arguments = "{}".to_string();
                                } else if let Some(normalized) =
                                    normalize_partial_tool_input(&entry.arguments)
                                {
                                    entry.arguments = normalized;
                                }
                            }
                        }
                        anthropic::StreamEvent::MessageStop => {
                            break;
                        }
                        anthropic::StreamEvent::ContentFiltered => {
                            yield Err(Arc::new(AIApiError::Other(anyhow!(
                                "Local model response was blocked by a content filter.",
                            ))));
                            return;
                        }
                        anthropic::StreamEvent::Error { kind, message } => {
                            let user_message = format!("Anthropic stream error ({kind}): {message}");
                            if kind == "overloaded_error" {
                                yield Err(Arc::new(AIApiError::ServerOverloaded));
                            } else {
                                yield Err(Arc::new(AIApiError::Other(anyhow!(user_message))));
                            }
                            return;
                        }
                    }
                }
            }
        }

        let mut output_messages = Vec::new();
        if !full_text.trim().is_empty() {
            output_messages.push(agent_output_message(&task_id, &request_id, full_text));
        }

        if !pending_tool_calls.is_empty() {
            for tool_call in pending_tool_calls.into_values() {
                let Some(message) = to_tool_call_message(tool_call, &task_id, &request_id)? else {
                    continue;
                };
                output_messages.push(message);
            }
        }

        if !output_messages.is_empty() {
            yield Ok(add_messages_event(&task_id, output_messages));
        }

        yield Ok(stream_finished_done_event());
    };

    let output_stream = output_stream.take_until(cancellation_rx);
    Box::pin(output_stream)
}

#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct RunShellCommandArgs {
    command: String,
    #[serde(default)]
    wait_until_complete: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct CallMcpToolArgs {
    name: String,
    #[serde(default)]
    args: Option<serde_json::Value>,
    #[serde(default)]
    server_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReadMcpResourceArgs {
    uri: String,
    #[serde(default)]
    server_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReadSkillArgs {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    skill_path: Option<String>,
    #[serde(default)]
    bundled_skill_id: Option<String>,
}

fn extract_chat_messages(
    params: &RequestParams,
    current_task_id: &str,
) -> Vec<openai::ChatMessage> {
    let mut messages = Vec::new();
    let current_task = params
        .tasks
        .iter()
        .find(|task| task.id == current_task_id)
        .or_else(|| params.tasks.first());

    if let Some(task) = current_task {
        let tool_names_by_id = tool_names_by_call_id(task);
        for message in &task.messages {
            match &message.message {
                Some(api::message::Message::UserQuery(user_query))
                    if !user_query.query.is_empty() =>
                {
                    messages.push(openai::ChatMessage::user(user_query.query.clone()));
                }
                Some(api::message::Message::AgentOutput(agent_output))
                    if !agent_output.text.trim().is_empty() =>
                {
                    messages.push(openai::ChatMessage::assistant(agent_output.text.clone()));
                }
                Some(api::message::Message::ToolCall(tool_call)) => {
                    if let Some(openai_tool_call) = to_openai_tool_call(tool_call) {
                        messages.push(openai::ChatMessage::assistant_tool_calls(vec![
                            openai_tool_call,
                        ]));
                    }
                }
                Some(api::message::Message::ToolCallResult(tool_result))
                    if !tool_result.tool_call_id.is_empty() =>
                {
                    let tool_name = tool_names_by_id
                        .get(tool_result.tool_call_id.as_str())
                        .map(|s| s.as_str());
                    messages.push(openai::ChatMessage::tool(
                        tool_result.tool_call_id.clone(),
                        format_tool_call_result(tool_result, tool_name),
                    ));
                }
                _ => {}
            }
        }
    }

    for input in &params.input {
        if let Some(chat_message) = to_chat_message_for_current_input(input) {
            if messages.last() != Some(&chat_message) {
                messages.push(chat_message);
            }
        }
    }

    if messages.is_empty() {
        messages.push(openai::ChatMessage::user("Continue".to_string()));
    }

    messages
}

fn to_chat_message_for_current_input(input: &AIAgentInput) -> Option<openai::ChatMessage> {
    if let Some(query) = input.user_query() {
        return Some(openai::ChatMessage::user(query));
    }

    input
        .action_result()
        .map(|result| openai::ChatMessage::tool(result.id.to_string(), result.result.to_string()))
}

fn tool_names_by_call_id(task: &api::Task) -> HashMap<&str, String> {
    task.messages
        .iter()
        .filter_map(|message| {
            let api::message::Message::ToolCall(tool_call) = message.message.as_ref()? else {
                return None;
            };
            let tool_name = openai_tool_name_from_api_tool(tool_call.tool.as_ref()?)?;
            Some((tool_call.tool_call_id.as_str(), tool_name.to_string()))
        })
        .collect()
}

fn openai_tool_name_from_api_tool(tool: &api::message::tool_call::Tool) -> Option<&'static str> {
    match tool {
        api::message::tool_call::Tool::RunShellCommand(_) => Some(RUN_SHELL_COMMAND_TOOL_NAME),
        api::message::tool_call::Tool::CallMcpTool(_) => Some(CALL_MCP_TOOL_NAME),
        api::message::tool_call::Tool::ReadMcpResource(_) => Some(READ_MCP_RESOURCE_TOOL_NAME),
        api::message::tool_call::Tool::ReadSkill(_) => Some(READ_SKILL_TOOL_NAME),
        _ => None,
    }
}

fn to_openai_tool_call(tool_call: &api::message::ToolCall) -> Option<openai::ChatToolCall> {
    let tool = tool_call.tool.as_ref()?;
    let id = if tool_call.tool_call_id.is_empty() {
        format!("local-tool-call-{}", Uuid::new_v4())
    } else {
        tool_call.tool_call_id.clone()
    };

    let (name, arguments) = match tool {
        api::message::tool_call::Tool::RunShellCommand(run_shell) => {
            let wait_until_complete = run_shell
                .wait_until_complete_value
                .as_ref()
                .and_then(|value| match value {
                    api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete(value) => {
                        Some(*value)
                    }
                });
            (
                RUN_SHELL_COMMAND_TOOL_NAME,
                json!({
                    "command": run_shell.command,
                    "wait_until_complete": wait_until_complete,
                }),
            )
        }
        api::message::tool_call::Tool::CallMcpTool(call_mcp_tool) => (
            CALL_MCP_TOOL_NAME,
            json!({
                "name": call_mcp_tool.name,
                "args": call_mcp_tool.args.as_ref().map(prost_struct_to_json),
                "server_id": (!call_mcp_tool.server_id.is_empty()).then_some(call_mcp_tool.server_id.clone()),
            }),
        ),
        api::message::tool_call::Tool::ReadMcpResource(read_mcp_resource) => (
            READ_MCP_RESOURCE_TOOL_NAME,
            json!({
                "uri": read_mcp_resource.uri,
                "server_id": (!read_mcp_resource.server_id.is_empty()).then_some(read_mcp_resource.server_id.clone()),
            }),
        ),
        api::message::tool_call::Tool::ReadSkill(read_skill) => {
            let (skill_path, bundled_skill_id) = match &read_skill.skill_reference {
                Some(api::message::tool_call::read_skill::SkillReference::SkillPath(path)) => {
                    (Some(path.clone()), None)
                }
                Some(api::message::tool_call::read_skill::SkillReference::BundledSkillId(id)) => {
                    (None, Some(id.clone()))
                }
                None => (None, None),
            };
            (
                READ_SKILL_TOOL_NAME,
                json!({
                    "name": (!read_skill.name.is_empty()).then_some(read_skill.name.clone()),
                    "skill_path": skill_path,
                    "bundled_skill_id": bundled_skill_id,
                }),
            )
        }
        _ => return None,
    };

    Some(openai::ChatToolCall {
        id,
        r#type: "function".to_string(),
        function: openai::ChatToolCallFunction {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    })
}

fn format_tool_call_result(
    result: &api::message::ToolCallResult,
    tool_name: Option<&str>,
) -> String {
    use api::message::tool_call_result::Result as ToolCallResult;
    match &result.result {
        Some(ToolCallResult::RunShellCommand(run_result)) => match &run_result.result {
            Some(api::run_shell_command_result::Result::CommandFinished(finished)) => {
                format!(
                    "Command finished with exit code {}:\n{}",
                    finished.exit_code, finished.output
                )
            }
            Some(api::run_shell_command_result::Result::LongRunningCommandSnapshot(_)) => {
                "Command is still running.".to_string()
            }
            Some(api::run_shell_command_result::Result::PermissionDenied(_)) => {
                "Command execution was denied.".to_string()
            }
            None => "Command result received.".to_string(),
        },
        Some(ToolCallResult::CallMcpTool(call_result)) => match &call_result.result {
            Some(api::call_mcp_tool_result::Result::Success(success)) => {
                format!(
                    "MCP tool succeeded with {} result item(s).",
                    success.results.len()
                )
            }
            Some(api::call_mcp_tool_result::Result::Error(error)) => {
                format!("MCP tool failed: {}", error.message)
            }
            None => "MCP tool result received.".to_string(),
        },
        Some(ToolCallResult::ReadMcpResource(read_result)) => match &read_result.result {
            Some(api::read_mcp_resource_result::Result::Success(success)) => {
                format!("Read {} MCP resource item(s).", success.contents.len())
            }
            Some(api::read_mcp_resource_result::Result::Error(error)) => {
                format!("MCP resource read failed: {}", error.message)
            }
            None => "MCP resource result received.".to_string(),
        },
        Some(ToolCallResult::ReadSkill(read_skill_result)) => match &read_skill_result.result {
            Some(api::read_skill_result::Result::Success(success)) => success
                .content
                .as_ref()
                .map(|content| content.content.clone())
                .filter(|content| !content.is_empty())
                .unwrap_or_else(|| "Skill read successfully.".to_string()),
            Some(api::read_skill_result::Result::Error(error)) => {
                format!("Skill read failed: {}", error.message)
            }
            None => "Skill result received.".to_string(),
        },
        Some(ToolCallResult::Cancel(())) => "Tool call was cancelled.".to_string(),
        _ => tool_name
            .map(|name| format!("Result received for tool `{name}`."))
            .unwrap_or_else(|| "Tool call result received.".to_string()),
    }
}

fn local_tool_definitions() -> Vec<openai::ToolDefinition> {
    vec![
        openai::ToolDefinition::function(
            RUN_SHELL_COMMAND_TOOL_NAME,
            "Run a shell command in the current terminal session.",
            json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute."
                    },
                    "wait_until_complete": {
                        "type": "boolean",
                        "description": "Whether to wait for command completion before responding."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        ),
        openai::ToolDefinition::function(
            CALL_MCP_TOOL_NAME,
            "Call an MCP tool by name with optional JSON arguments.",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "args": {"type": "object"},
                    "server_id": {"type": "string"}
                },
                "required": ["name"],
                "additionalProperties": false
            }),
        ),
        openai::ToolDefinition::function(
            READ_MCP_RESOURCE_TOOL_NAME,
            "Read an MCP resource by URI.",
            json!({
                "type": "object",
                "properties": {
                    "uri": {"type": "string"},
                    "server_id": {"type": "string"}
                },
                "required": ["uri"],
                "additionalProperties": false
            }),
        ),
        openai::ToolDefinition::function(
            READ_SKILL_TOOL_NAME,
            "Read a local or bundled skill.",
            json!({
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "skill_path": {"type": "string"},
                    "bundled_skill_id": {"type": "string"}
                },
                "additionalProperties": false
            }),
        ),
    ]
}

fn anthropic_tool_definitions() -> Vec<anthropic::ToolDefinition> {
    local_tool_definitions()
        .into_iter()
        .map(|tool| {
            anthropic::ToolDefinition::new(
                tool.function.name,
                tool.function.description,
                tool.function.parameters,
            )
        })
        .collect()
}

fn anthropic_messages_from_chat_messages(
    messages: &[openai::ChatMessage],
) -> Vec<anthropic::Message> {
    messages
        .iter()
        .filter_map(|message| match message.role.as_str() {
            "user" => {
                let content = message.content.clone().unwrap_or_default();
                if content.trim().is_empty() {
                    None
                } else {
                    Some(anthropic::Message::user(vec![
                        anthropic::ContentBlock::text(content),
                    ]))
                }
            }
            "assistant" => {
                let mut content_blocks = Vec::new();
                if let Some(text) = message
                    .content
                    .clone()
                    .filter(|text| !text.trim().is_empty())
                {
                    content_blocks.push(anthropic::ContentBlock::text(text));
                }
                if let Some(tool_calls) = message.tool_calls.as_ref() {
                    content_blocks.extend(tool_calls.iter().map(|tool_call| {
                        anthropic::ContentBlock::tool_use(
                            tool_call.id.clone(),
                            tool_call.function.name.clone(),
                            parse_openai_tool_arguments_to_value(&tool_call.function.arguments),
                        )
                    }));
                }
                if content_blocks.is_empty() {
                    None
                } else {
                    Some(anthropic::Message::assistant(content_blocks))
                }
            }
            "tool" => {
                let tool_use_id = message.tool_call_id.clone().unwrap_or_default();
                if tool_use_id.trim().is_empty() {
                    return None;
                }
                let content = message.content.clone().unwrap_or_default();
                Some(anthropic::Message::user(vec![
                    anthropic::ContentBlock::tool_result(tool_use_id, content),
                ]))
            }
            _ => None,
        })
        .collect()
}

fn parse_openai_tool_arguments_to_value(arguments: &str) -> Value {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| json!({}))
}

fn normalize_tool_input_json(value: Value) -> String {
    if value.is_null() {
        "{}".to_string()
    } else {
        value.to_string()
    }
}

fn normalize_partial_tool_input(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some("{}".to_string());
    }
    serde_json::from_str::<Value>(trimmed)
        .ok()
        .map(normalize_tool_input_json)
}

fn local_provider_kind(endpoint_url: &str) -> LocalProviderKind {
    if is_anthropic_messages_endpoint(endpoint_url) {
        LocalProviderKind::AnthropicMessages
    } else {
        LocalProviderKind::OpenAICompatible
    }
}

fn is_anthropic_messages_endpoint(endpoint_url: &str) -> bool {
    let trimmed = endpoint_url.trim_end_matches('/');
    if trimmed.ends_with("/v1/messages") || trimmed.ends_with("/messages") {
        return true;
    }

    if let Ok(parsed_url) = url::Url::parse(trimmed) {
        let host_is_anthropic = parsed_url
            .host_str()
            .is_some_and(|host| host.eq_ignore_ascii_case("api.anthropic.com"));
        let path_is_messages =
            parsed_url.path().ends_with("/v1/messages") || parsed_url.path().ends_with("/messages");
        return host_is_anthropic || path_is_messages;
    }

    false
}

fn merge_tool_call_delta(
    pending_tool_calls: &mut BTreeMap<usize, PendingToolCall>,
    delta: openai::ToolCallDelta,
) {
    let index = delta.index.unwrap_or(pending_tool_calls.len());
    let entry = pending_tool_calls.entry(index).or_default();

    if let Some(id) = delta.id.filter(|id| !id.is_empty()) {
        entry.id = id;
    }

    if let Some(function) = delta.function {
        if let Some(name) = function.name {
            entry.name.push_str(&name);
        }
        if let Some(arguments) = function.arguments {
            entry.arguments.push_str(&arguments);
        }
    }
}

fn to_tool_call_message(
    tool_call: PendingToolCall,
    task_id: &str,
    request_id: &str,
) -> Result<Option<api::Message>, Arc<AIApiError>> {
    let tool_name = tool_call.name.trim();
    if tool_name.is_empty() {
        return Ok(None);
    }

    let tool_call_id = if tool_call.id.trim().is_empty() {
        format!("local-tool-call-{}", Uuid::new_v4())
    } else {
        tool_call.id
    };

    let tool = match tool_name {
        RUN_SHELL_COMMAND_TOOL_NAME => {
            let args: RunShellCommandArgs = parse_tool_args(tool_name, &tool_call.arguments)?;
            Some(api::message::tool_call::Tool::RunShellCommand(
                api::message::tool_call::RunShellCommand {
                    command: args.command,
                    is_read_only: false,
                    uses_pager: false,
                    citations: vec![],
                    is_risky: false,
                    risk_category: api::RiskCategory::Unspecified as i32,
                    wait_until_complete_value: args.wait_until_complete.map(
                        api::message::tool_call::run_shell_command::WaitUntilCompleteValue::WaitUntilComplete,
                    ),
                },
            ))
        }
        CALL_MCP_TOOL_NAME => {
            let args: CallMcpToolArgs = parse_tool_args(tool_name, &tool_call.arguments)?;
            Some(api::message::tool_call::Tool::CallMcpTool(
                api::message::tool_call::CallMcpTool {
                    name: args.name,
                    args: json_value_to_prost_struct(args.args)?,
                    server_id: args.server_id.unwrap_or_default(),
                },
            ))
        }
        READ_MCP_RESOURCE_TOOL_NAME => {
            let args: ReadMcpResourceArgs = parse_tool_args(tool_name, &tool_call.arguments)?;
            Some(api::message::tool_call::Tool::ReadMcpResource(
                api::message::tool_call::ReadMcpResource {
                    uri: args.uri,
                    server_id: args.server_id.unwrap_or_default(),
                },
            ))
        }
        READ_SKILL_TOOL_NAME => {
            let args: ReadSkillArgs = parse_tool_args(tool_name, &tool_call.arguments)?;
            let skill_reference = if let Some(path) =
                args.skill_path.filter(|path| !path.is_empty())
            {
                Some(api::message::tool_call::read_skill::SkillReference::SkillPath(path))
            } else if let Some(id) = args.bundled_skill_id.filter(|id| !id.is_empty()) {
                Some(api::message::tool_call::read_skill::SkillReference::BundledSkillId(id))
            } else {
                return Err(Arc::new(AIApiError::Other(anyhow!(
                    "Tool `{READ_SKILL_TOOL_NAME}` requires either `skill_path` or `bundled_skill_id`.",
                ))));
            };

            Some(api::message::tool_call::Tool::ReadSkill(
                api::message::tool_call::ReadSkill {
                    name: args.name.unwrap_or_default(),
                    skill_reference,
                },
            ))
        }
        _ => {
            log::warn!("Ignoring unsupported local tool call: {}", tool_name);
            None
        }
    };

    Ok(tool.map(|tool| api::Message {
        id: format!("local-tool-call-msg-{}", Uuid::new_v4()),
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::ToolCall(api::message::ToolCall {
            tool_call_id,
            tool: Some(tool),
        })),
        request_id: request_id.to_string(),
        timestamp: None,
    }))
}

fn parse_tool_args<T: DeserializeOwned>(
    tool_name: &str,
    raw_arguments: &str,
) -> Result<T, Arc<AIApiError>> {
    let arguments = raw_arguments.trim();
    let arguments = if arguments.is_empty() {
        "{}"
    } else {
        arguments
    };
    serde_json::from_str(arguments).map_err(|e| {
        Arc::new(AIApiError::Other(anyhow!(
            "Failed to parse arguments for local tool `{tool_name}`: {e}. arguments={arguments}",
        )))
    })
}

fn json_value_to_prost_struct(
    value: Option<serde_json::Value>,
) -> Result<Option<prost_types::Struct>, Arc<AIApiError>> {
    let Some(value) = value else {
        return Ok(None);
    };
    match value {
        serde_json::Value::Object(fields) => {
            let fields: Result<BTreeMap<String, prost_types::Value>, Arc<AIApiError>> = fields
                .into_iter()
                .map(|(key, value)| serde_json_to_prost_value(value).map(|value| (key, value)))
                .collect();
            Ok(Some(prost_types::Struct { fields: fields? }))
        }
        other => Err(Arc::new(AIApiError::Other(anyhow!(
            "Expected MCP `args` to be a JSON object, got: {}",
            other
        )))),
    }
}

fn serde_json_to_prost_value(
    value: serde_json::Value,
) -> Result<prost_types::Value, Arc<AIApiError>> {
    use prost_types::value::Kind;

    let kind = match value {
        serde_json::Value::Null => Kind::NullValue(0),
        serde_json::Value::Bool(v) => Kind::BoolValue(v),
        serde_json::Value::Number(number) => {
            let Some(value) = number.as_f64() else {
                return Err(Arc::new(AIApiError::Other(anyhow!(
                    "Invalid JSON number for protobuf conversion: {number}",
                ))));
            };
            Kind::NumberValue(value)
        }
        serde_json::Value::String(v) => Kind::StringValue(v),
        serde_json::Value::Array(values) => Kind::ListValue(prost_types::ListValue {
            values: values
                .into_iter()
                .map(serde_json_to_prost_value)
                .collect::<Result<Vec<_>, _>>()?,
        }),
        serde_json::Value::Object(fields) => {
            let fields: Result<BTreeMap<String, prost_types::Value>, Arc<AIApiError>> = fields
                .into_iter()
                .map(|(key, value)| serde_json_to_prost_value(value).map(|value| (key, value)))
                .collect();
            Kind::StructValue(prost_types::Struct { fields: fields? })
        }
    };

    Ok(prost_types::Value { kind: Some(kind) })
}

fn prost_struct_to_json(value: &prost_types::Struct) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = value
        .fields
        .iter()
        .map(|(key, value)| (key.clone(), prost_value_to_json(value)))
        .collect();
    serde_json::Value::Object(map)
}

fn prost_value_to_json(value: &prost_types::Value) -> serde_json::Value {
    use prost_types::value::Kind;

    match value.kind.as_ref() {
        Some(Kind::NullValue(_)) => serde_json::Value::Null,
        Some(Kind::NumberValue(number)) => json!(number),
        Some(Kind::StringValue(string)) => serde_json::Value::String(string.clone()),
        Some(Kind::BoolValue(boolean)) => serde_json::Value::Bool(*boolean),
        Some(Kind::StructValue(struct_value)) => prost_struct_to_json(struct_value),
        Some(Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
        None => serde_json::Value::Null,
    }
}

fn stream_init_event(request_id: &str) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Init(
            api::response_event::StreamInit {
                request_id: request_id.to_string(),
                conversation_id: String::new(),
                run_id: String::new(),
            },
        )),
    }
}

fn agent_output_message(task_id: &str, request_id: &str, text: String) -> api::Message {
    api::Message {
        id: format!("local-agent-output-{}", Uuid::new_v4()),
        task_id: task_id.to_string(),
        server_message_data: String::new(),
        citations: vec![],
        message: Some(api::message::Message::AgentOutput(
            api::message::AgentOutput { text },
        )),
        request_id: request_id.to_string(),
        timestamp: None,
    }
}

fn add_messages_event(task_id: &str, messages: Vec<api::Message>) -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::ClientActions(
            api::response_event::ClientActions {
                actions: vec![api::ClientAction {
                    action: Some(api::client_action::Action::AddMessagesToTask(
                        api::client_action::AddMessagesToTask {
                            task_id: task_id.to_string(),
                            messages,
                        },
                    )),
                }],
            },
        )),
    }
}

fn stream_finished_done_event() -> api::ResponseEvent {
    api::ResponseEvent {
        r#type: Some(api::response_event::Type::Finished(
            api::response_event::StreamFinished {
                reason: Some(api::response_event::stream_finished::Reason::Done(
                    api::response_event::stream_finished::Done {},
                )),
                conversation_usage_metadata: None,
                token_usage: vec![],
                should_refresh_model_config: false,
                request_cost: None,
            },
        )),
    }
}
