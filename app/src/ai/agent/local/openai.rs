use std::sync::Arc;

use futures_util::stream::BoxStream;
use futures_util::StreamExt;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::EventSource;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::server::server_api::AIApiError;

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub tools: Vec<ToolDefinition>,
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant(content: String) -> Self {
        Self {
            role: "assistant".to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    pub fn assistant_tool_calls(tool_calls: Vec<ChatToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }

    pub fn tool(tool_call_id: String, content: String) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(content),
            tool_calls: None,
            tool_call_id: Some(tool_call_id),
            name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    pub r#type: String,
    pub function: ChatToolCallFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: ToolFunctionDefinition,
}

impl ToolDefinition {
    pub fn function(name: &str, description: &str, parameters: Value) -> Self {
        Self {
            r#type: "function".to_string(),
            function: ToolFunctionDefinition {
                name: name.to_string(),
                description: description.to_string(),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionChunk {
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChunkChoice {
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
pub struct Delta {
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ToolCallDelta {
    pub index: Option<usize>,
    pub id: Option<String>,
    pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Default, Deserialize)]
pub struct ToolCallFunctionDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// Sends a streaming chat-completions request and yields raw SSE data payloads.
pub fn stream_chat_completions(
    client: reqwest::Client,
    endpoint_url: &str,
    api_key: Option<&str>,
    request: &ChatRequest,
) -> BoxStream<'static, Result<String, Arc<AIApiError>>> {
    let url = format!("{}/chat/completions", endpoint_url.trim_end_matches('/'));
    let mut builder = client.post(&url).header(CONTENT_TYPE, "application/json");

    if let Some(key) = api_key {
        builder = builder.header(AUTHORIZATION, format!("Bearer {key}"));
    }

    let body = match serde_json::to_string(request) {
        Ok(body) => body,
        Err(e) => {
            let err = Arc::new(AIApiError::Other(anyhow::anyhow!(
                "Failed to serialize local chat request: {e}"
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
                "Failed to connect to local endpoint: {e}"
            )));
            futures_util::stream::once(async move { Err(err) }).boxed()
        }
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
        other => AIApiError::Other(anyhow::anyhow!("Local SSE error: {other}")),
    }
}

pub fn parse_chunk(data: &str) -> Result<Option<ChatCompletionChunk>, Arc<AIApiError>> {
    if data.trim() == "[DONE]" {
        return Ok(None);
    }

    serde_json::from_str::<ChatCompletionChunk>(data)
        .map(Some)
        .map_err(|e| {
            Arc::new(AIApiError::Other(anyhow::anyhow!(
                "Failed to parse local SSE chunk: {e}; data={data}"
            )))
        })
}
