//! OpenAI-compatible request/response types for the RLM server

use serde::{Deserialize, Serialize};

/// A chat message in OpenAI format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Request body for chat completions
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    /// The model to use (ignored - RLM uses its configured backend)
    #[serde(default)]
    pub model: String,

    /// The messages in the conversation
    pub messages: Vec<ChatMessage>,

    /// Sampling temperature (0-2)
    #[serde(default)]
    pub temperature: Option<f32>,

    /// Maximum tokens to generate
    #[serde(default)]
    pub max_tokens: Option<u32>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: Option<bool>,
}

/// A choice in the completion response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Token usage statistics
#[derive(Debug, Clone, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Response body for chat completions (non-streaming)
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: CompletionUsage,
}

/// A delta message for streaming responses
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessageDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

/// A choice in a streaming chunk
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatMessageDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// A streaming chunk response
#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChunkChoice>,
}

impl ChatCompletionResponse {
    /// Create a new completion response
    pub fn new(
        id: String,
        model: String,
        content: String,
        usage: CompletionUsage,
    ) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: "stop".to_string(),
            }],
            usage,
        }
    }
}

impl ChatCompletionChunk {
    /// Create a new chunk with role
    pub fn with_role(id: String, model: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        }
    }

    /// Create a new chunk with content
    pub fn with_content(id: String, model: String, content: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: Some(content),
                },
                finish_reason: None,
            }],
        }
    }

    /// Create a final chunk with finish_reason
    pub fn finished(id: String, model: String) -> Self {
        Self {
            id,
            object: "chat.completion.chunk".to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            model,
            choices: vec![ChatCompletionChunkChoice {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        }
    }
}
