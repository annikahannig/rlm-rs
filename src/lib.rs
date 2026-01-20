//! # RLM - Recursive Language Models
//!
//! An inference engine enabling LLMs to recursively decompose tasks
//! via REPL-based code execution.

pub mod error;
pub mod parsing;
pub mod types;

pub mod env;

mod prompts;
mod rlm;

// Re-exports
pub use error::{Result, RlmError};
pub use rlm::Rlm;
pub use types::{
    Backend, ChatCompletion, CodeBlock, Message, PromptInput, ReplResult, RlmCompletion, RlmConfig,
    RlmIteration, Role, Usage,
};
