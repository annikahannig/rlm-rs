use thiserror::Error;

/// RLM error types
#[derive(Error, Debug)]
pub enum RlmError {
    #[error("OpenAI API error: {0}")]
    OpenAi(#[from] async_openai::error::OpenAIError),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Python execution error: {0}")]
    Python(String),

    #[error("PyO3 error: {0}")]
    PyO3(#[from] pyo3::PyErr),

    #[error("Tokio runtime error: {0}")]
    Runtime(#[from] std::io::Error),

    #[error("Max iterations reached ({0})")]
    MaxIterationsReached(u32),

    #[error("No API key found. Set OPENAI_API_KEY environment variable.")]
    MissingApiKey,

    #[error("Invalid configuration: {0}")]
    Config(String),
}

/// Result type alias for RLM operations
pub type Result<T> = std::result::Result<T, RlmError>;
