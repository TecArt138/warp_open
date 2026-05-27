use std::sync::Arc;

use futures_util::StreamExt;
use futures_util::stream::BoxStream;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::EventSource;
use serde::Serialize;
use serde_json::Value;

use crate::server::server_api::AIApiError;

const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ANTHROPIC_VERSION_HEADER: &str = "anthropic-version";
const ANTHROPIC_API_KEY_HEADER: &str = "x-api-key";

#[derive(Debug, Clone, Serialize)]
pub struct MessagesRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: Vec<ContentBlock>,
}

impl Message {
    pub fn user(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

impl ContentBlock {
    pub fn text(text: String) -> Self {
        Self::Text { text }
    }

    pub fn tool_use(id: String, name: String, input: Value) -> Self {
        Self::ToolUse { id, name, input }
    }

    pub fn tool_result(tool_use_id: String, content: String) -> Self {
        Self::ToolResult {
            tool_use_id,
            content,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    #[serde(rename = "input_schema")]
    pub input_schema: Value,
}

impl ToolDefinition {
    pub fn new(name: String, description: String, input_schema: Value) -> Self {
        Self {
            name,
            description,
            input_schema,
        }
    }
}

#[derive(Debug, Clone)]
pub enum StreamEvent {
    TextDelta(String),
    ToolUseStart {
        index: usize,
        id: String,
        name: String,
        input: Option<Value>,
    },
    ToolUseInputDelta {
        index: usize,
        partial_json: String,
    },
    ToolUseStop {
        index: usize,
    },
    MessageStop,
    ContentFiltered,
    Error {
        kind: String,
        message: String,
    },
}

/// Sends a streaming Anthropic messages request and yields raw SSE data payloads.
pub fn stream_messages(
    client: reqwest::Client,
    endpoint_url: &str,
    api_key: Option<&str>,
    request: &MessagesRequest,
) -> BoxStream<'static, Result<String, Arc<AIApiError>>> {
    let url = messages_endpoint_url(endpoint_url);
    let mut builder = client
        .post(&url)
        .header(CONTENT_TYPE, "application/json")
        .header(ANTHROPIC_VERSION_HEADER, ANTHROPIC_API_VERSION);

    if let Some(key) = api_key {
        builder = builder
            .header(ANTHROPIC_API_KEY_HEADER, key)
            .header(AUTHORIZATION, format!("Bearer {key}"));
    }

    let body = match serde_json::to_string(request) {
        Ok(body) => body,
        Err(e) => {
            let err = Arc::new(AIApiError::Other(anyhow::anyhow!(
                "Failed to serialize anthropic local request: {e}"
            )));
            return futures_util::stream::once(async move { Err(err) }).boxed();
        }
    };

    let req = builder.body(body);
    let event_source = EventSource::new(req);

    match event_source {
        Ok(es) => {
            let mapped = es.map(|event| match event {
                Ok(reqwest_eventsource::Event::Open) => Ok(String::new()),
                Ok(reqwest_eventsource::Event::Message(msg)) => Ok(msg.data),
                Err(e) => Err(Arc::new(map_eventsource_error(e))),
            });

            futures_util::stream::StreamExt::filter(mapped, |result| {
                std::future::ready(result.as_ref().map(|s| !s.is_empty()).unwrap_or(true))
            })
            .boxed()
        }
        Err(e) => {
            let err = Arc::new(AIApiError::Other(anyhow::anyhow!(
                "Failed to connect to anthropic local endpoint: {e}"
            )));
            futures_util::stream::once(async move { Err(err) }).boxed()
        }
    }
}

pub fn parse_stream_event(data: &str) -> Result<Option<StreamEvent>, Arc<AIApiError>> {
    let trimmed = data.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" {
        return Ok(None);
    }

    let payload: Value = serde_json::from_str(trimmed).map_err(|e| {
        Arc::new(AIApiError::Other(anyhow::anyhow!(
            "Failed to parse anthropic SSE event: {e}; data={trimmed}"
        )))
    })?;

    let Some(event_type) = payload.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };

    let event = match event_type {
        "content_block_start" => {
            let index = payload
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or_default() as usize;
            let Some(content_block) = payload.get("content_block") else {
                return Ok(None);
            };
            let Some(content_type) = content_block.get("type").and_then(Value::as_str) else {
                return Ok(None);
            };

            match content_type {
                "text" => content_block
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| StreamEvent::TextDelta(text.to_string())),
                "tool_use" => {
                    let id = content_block
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let name = content_block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let input = content_block.get("input").cloned();
                    Some(StreamEvent::ToolUseStart {
                        index,
                        id,
                        name,
                        input,
                    })
                }
                _ => None,
            }
        }
        "content_block_delta" => {
            let index = payload
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or_default() as usize;
            let Some(delta) = payload.get("delta") else {
                return Ok(None);
            };
            let Some(delta_type) = delta.get("type").and_then(Value::as_str) else {
                return Ok(None);
            };

            match delta_type {
                "text_delta" => delta
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| StreamEvent::TextDelta(text.to_string())),
                "input_json_delta" => delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .filter(|partial| !partial.is_empty())
                    .map(|partial_json| StreamEvent::ToolUseInputDelta {
                        index,
                        partial_json: partial_json.to_string(),
                    }),
                _ => None,
            }
        }
        "content_block_stop" => {
            let index = payload
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or_default() as usize;
            Some(StreamEvent::ToolUseStop { index })
        }
        "message_delta" => {
            let stop_reason = payload
                .get("delta")
                .and_then(|delta| delta.get("stop_reason"))
                .and_then(Value::as_str);
            if matches!(stop_reason, Some("content_filter")) {
                Some(StreamEvent::ContentFiltered)
            } else {
                None
            }
        }
        "message_stop" => Some(StreamEvent::MessageStop),
        "error" => {
            let error = payload.get("error");
            let kind = error
                .and_then(|error| error.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("error")
                .to_string();
            let message = error
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown Anthropic streaming error.")
                .to_string();
            Some(StreamEvent::Error { kind, message })
        }
        _ => None,
    };

    Ok(event)
}

fn messages_endpoint_url(endpoint_url: &str) -> String {
    let trimmed = endpoint_url.trim_end_matches('/');
    if trimmed.ends_with("/v1/messages") || trimmed.ends_with("/messages") {
        trimmed.to_string()
    } else if trimmed.ends_with("/v1") {
        format!("{trimmed}/messages")
    } else {
        format!("{trimmed}/v1/messages")
    }
}

fn map_eventsource_error(e: reqwest_eventsource::Error) -> AIApiError {
    match e {
        reqwest_eventsource::Error::Transport(re) => AIApiError::Transport(re),
        reqwest_eventsource::Error::InvalidStatusCode(status, _) => {
            if status == http::StatusCode::TOO_MANY_REQUESTS {
                AIApiError::QuotaLimit {
                    user_display_message: None,
                }
            } else if status.is_server_error() {
                AIApiError::ServerOverloaded
            } else {
                AIApiError::ErrorStatus(status, format!("HTTP {}", status.as_u16()))
            }
        }
        other => AIApiError::Other(anyhow::anyhow!("Anthropic SSE error: {other}")),
    }
}
