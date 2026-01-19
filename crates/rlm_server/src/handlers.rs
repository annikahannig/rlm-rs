//! HTTP handlers for the RLM server

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use std::convert::Infallible;
use std::sync::Arc;
use uuid::Uuid;

use crate::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionUsage,
};
use rlm::{Message, PromptInput, Rlm, RlmConfig, Role};

/// Shared server state
pub struct AppState {
    pub model: String,
    pub backend_url: String,
    pub backend_key: Option<String>,
}

/// Convert OpenAI-style messages to RLM messages
fn convert_messages(messages: &[crate::types::ChatMessage]) -> Vec<Message> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role.as_str() {
                "system" => Role::System,
                "assistant" => Role::Assistant,
                _ => Role::User,
            };
            Message {
                role,
                content: m.content.clone(),
            }
        })
        .collect()
}

/// Handler for POST /v1/chat/completions
pub async fn create_chat_completion(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(false);

    if stream {
        handle_streaming_completion(state, req).await
    } else {
        handle_completion(state, req).await
    }
}

/// Handle non-streaming completion
async fn handle_completion(state: Arc<AppState>, req: ChatCompletionRequest) -> Response {
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());

    // Build RLM config
    let mut config = RlmConfig::new(&state.model);
    if let Some(temp) = req.temperature {
        config = config.with_temperature(temp);
    }
    if let Some(max_tokens) = req.max_tokens {
        config = config.with_max_tokens(max_tokens);
    }

    // Create RLM instance
    let rlm = match create_rlm(&state, config) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to create RLM: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Convert messages to RLM format
    let messages = convert_messages(&req.messages);
    let prompt = PromptInput::Messages(messages);

    // Run completion in a blocking task (RLM uses synchronous code)
    let result = tokio::task::spawn_blocking(move || rlm.completion(prompt)).await;

    match result {
        Ok(Ok(completion)) => {
            let response = ChatCompletionResponse::new(
                request_id,
                state.model.clone(),
                completion.response,
                CompletionUsage {
                    prompt_tokens: completion.usage.input_tokens,
                    completion_tokens: completion.usage.output_tokens,
                    total_tokens: completion.usage.total_tokens,
                },
            );
            (StatusCode::OK, Json(response)).into_response()
        }
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("RLM error: {}", e),
                    "type": "server_error"
                }
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": {
                    "message": format!("Task join error: {}", e),
                    "type": "server_error"
                }
            })),
        )
            .into_response(),
    }
}

/// Handle streaming completion
async fn handle_streaming_completion(state: Arc<AppState>, req: ChatCompletionRequest) -> Response {
    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = state.model.clone();

    // Build RLM config
    let mut config = RlmConfig::new(&state.model);
    if let Some(temp) = req.temperature {
        config = config.with_temperature(temp);
    }
    if let Some(max_tokens) = req.max_tokens {
        config = config.with_max_tokens(max_tokens);
    }

    // Create RLM instance
    let rlm = match create_rlm(&state, config) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": {
                        "message": format!("Failed to create RLM: {}", e),
                        "type": "server_error"
                    }
                })),
            )
                .into_response();
        }
    };

    // Convert messages to RLM format
    let messages = convert_messages(&req.messages);
    let prompt = PromptInput::Messages(messages);

    // Create a channel to stream results
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(100);

    // Spawn blocking task to run RLM
    let request_id_clone = request_id.clone();
    let model_clone = model.clone();
    tokio::task::spawn_blocking(move || {
        // Send initial role chunk
        let role_chunk = ChatCompletionChunk::with_role(request_id_clone.clone(), model_clone.clone());
        let _ = tx.blocking_send(Ok(Event::default()
            .data(serde_json::to_string(&role_chunk).unwrap())));

        // Run completion
        match rlm.completion(prompt) {
            Ok(completion) => {
                // Send content in chunks (split by words for more natural streaming)
                for word in completion.response.split_inclusive(' ') {
                    let content_chunk =
                        ChatCompletionChunk::with_content(request_id_clone.clone(), model_clone.clone(), word.to_string());
                    if tx
                        .blocking_send(Ok(Event::default()
                            .data(serde_json::to_string(&content_chunk).unwrap())))
                        .is_err()
                    {
                        return;
                    }
                    // Small delay for more natural streaming feel
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }

                // Send finish chunk
                let finish_chunk =
                    ChatCompletionChunk::finished(request_id_clone.clone(), model_clone.clone());
                let _ = tx.blocking_send(Ok(Event::default()
                    .data(serde_json::to_string(&finish_chunk).unwrap())));

                // Send [DONE]
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
            Err(e) => {
                // Send error as content
                let error_chunk = ChatCompletionChunk::with_content(
                    request_id_clone.clone(),
                    model_clone.clone(),
                    format!("Error: {}", e),
                );
                let _ = tx.blocking_send(Ok(Event::default()
                    .data(serde_json::to_string(&error_chunk).unwrap())));
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
        }
    });

    // Convert receiver to stream
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Create an RLM instance with the appropriate configuration
fn create_rlm(state: &AppState, config: RlmConfig) -> rlm::Result<Rlm> {
    match &state.backend_key {
        Some(key) => Rlm::with_base_url_and_key(config, &state.backend_url, key),
        None => Rlm::with_base_url(config, &state.backend_url),
    }
}

/// Handler for GET /v1/models
pub async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [
            {
                "id": "rlm",
                "object": "model",
                "created": 1700000000,
                "owned_by": "rlm"
            },
            {
                "id": state.model,
                "object": "model",
                "created": 1700000000,
                "owned_by": "rlm"
            }
        ]
    }))
}
