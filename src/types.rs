use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// LLM Backend provider
#[derive(Debug, Clone, Default)]
pub enum Backend {
    #[default]
    OpenAI,
    Anthropic,
}

/// Token usage statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
}

impl Usage {
    pub fn new(input: u64, output: u64) -> Self {
        Self {
            input_tokens: input,
            output_tokens: output,
            total_tokens: input + output,
        }
    }

    /// Accumulate usage from another instance
    pub fn add(&mut self, other: &Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.total_tokens += other.total_tokens;
    }
}

/// OpenAI-style message
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// Message role
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Prompt can be a string or message list
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PromptInput {
    Text(String),
    Messages(Vec<Message>),
}

impl From<String> for PromptInput {
    fn from(s: String) -> Self {
        PromptInput::Text(s)
    }
}

impl From<&str> for PromptInput {
    fn from(s: &str) -> Self {
        PromptInput::Text(s.to_string())
    }
}

impl From<Vec<Message>> for PromptInput {
    fn from(m: Vec<Message>) -> Self {
        PromptInput::Messages(m)
    }
}

impl std::fmt::Display for PromptInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptInput::Text(s) => write!(f, "{}", s),
            PromptInput::Messages(msgs) => {
                for msg in msgs {
                    writeln!(f, "[{:?}]: {}", msg.role, msg.content)?;
                }
                Ok(())
            }
        }
    }
}

/// Result of a single LM completion call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletion {
    pub prompt: PromptInput,
    pub response: String,
    pub usage: Usage,
    #[serde(with = "humantime_serde")]
    pub execution_time: Duration,
}

/// Result of code execution in REPL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplResult {
    pub stdout: String,
    pub stderr: String,
    pub locals: HashMap<String, String>,
    #[serde(with = "humantime_serde")]
    pub execution_time: Duration,
    pub llm_calls: Vec<ChatCompletion>,
    pub success: bool,
    pub error: Option<String>,
    /// Output from llm_output() call - signals iteration should stop
    pub llm_output: Option<String>,
}

impl ReplResult {
    /// Create a successful result
    pub fn success(stdout: String, locals: HashMap<String, String>, execution_time: Duration) -> Self {
        Self {
            stdout,
            stderr: String::new(),
            locals,
            execution_time,
            llm_calls: Vec::new(),
            success: true,
            error: None,
            llm_output: None,
        }
    }

    /// Create a failed result
    pub fn failure(error: String, stderr: String, execution_time: Duration) -> Self {
        Self {
            stdout: String::new(),
            stderr,
            locals: HashMap::new(),
            execution_time,
            llm_calls: Vec::new(),
            success: false,
            error: Some(error),
            llm_output: None,
        }
    }
}

/// Extracted code block with its execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeBlock {
    pub code: String,
    pub result: Option<ReplResult>,
    pub retry_count: u32,
}

/// Single iteration of the RLM loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmIteration {
    pub iteration: u32,
    pub response: String,
    pub code_blocks: Vec<CodeBlock>,
    pub final_answer: Option<String>,
    #[serde(with = "humantime_serde")]
    pub execution_time: Duration,
}

/// Final RLM completion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlmCompletion {
    pub prompt: PromptInput,
    pub response: String,
    pub iterations: Vec<RlmIteration>,
    pub usage: Usage,
    #[serde(with = "humantime_serde")]
    pub execution_time: Duration,
}

/// Configuration for RLM
#[derive(Debug, Clone)]
pub struct RlmConfig {
    pub model: String,
    pub max_iterations: u32,
    pub max_exec_retries: u32,
    pub temperature: f32,
    pub max_tokens: Option<u32>,
    pub verbose: bool,
    /// Show minimal execution progress (iterations, code exec, final)
    pub exec_log: bool,
    /// LLM backend provider
    pub backend: Backend,
    /// Base URL for API (optional, for custom endpoints)
    pub base_url: Option<String>,
    /// API key (optional, can use env vars)
    pub api_key: Option<String>,
}

impl Default for RlmConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o".to_string(),
            max_iterations: 20,
            max_exec_retries: 2,
            temperature: 0.0,
            max_tokens: None,
            verbose: false,
            exec_log: false,
            backend: Backend::default(),
            base_url: None,
            api_key: None,
        }
    }
}

impl RlmConfig {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n;
        self
    }

    pub fn with_max_exec_retries(mut self, n: u32) -> Self {
        self.max_exec_retries = n;
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }

    pub fn with_verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    pub fn with_exec_log(mut self, v: bool) -> Self {
        self.exec_log = v;
        self
    }

    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

/// humantime_serde module for Duration serialization
mod humantime_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{:?}", duration))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        // Parse "1.234s" format
        let s = s.trim_end_matches('s');
        let secs: f64 = s.parse().map_err(serde::de::Error::custom)?;
        Ok(Duration::from_secs_f64(secs))
    }
}
