//! RLM Agent - Tool-use agent harness
//!
//! Uses RLM as an opaque reasoning engine. The agent harness:
//! 1. Sends tasks to RLM
//! 2. Parses tool calls from RLM output
//! 3. Executes tools externally
//! 4. Feeds results back to RLM
//! 5. Repeats until task complete

pub mod tools;

use rlm::{Backend, Rlm, RlmConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Tool execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub success: bool,
    pub output: String,
    pub error: Option<String>,
}

impl ToolResult {
    pub fn ok(output: impl Into<String>) -> Self {
        Self {
            success: true,
            output: output.into(),
            error: None,
        }
    }

    pub fn err(error: impl Into<String>) -> Self {
        Self {
            success: false,
            output: String::new(),
            error: Some(error.into()),
        }
    }
}

/// Parsed tool call from model output
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub args: String,
}

/// Tool definition
pub trait Tool: Send + Sync {
    /// Tool name
    fn name(&self) -> &str;

    /// Tool description for the model
    fn description(&self) -> &str;

    /// Usage example
    fn usage(&self) -> &str;

    /// Execute the tool
    fn execute(&self, args: &str) -> ToolResult;
}

/// Registry of available tools
#[derive(Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    pub fn list(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Generate tool documentation for system prompt
    pub fn generate_docs(&self) -> String {
        let mut docs = String::new();

        for (name, tool) in &self.tools {
            docs.push_str(&format!("- {}: {}\n", name, tool.description()));
            docs.push_str(&format!("  Usage: {}\n", tool.usage()));
        }

        docs
    }

    /// Execute a tool by name
    pub fn execute(&self, name: &str, args: &str) -> ToolResult {
        match self.get(name) {
            Some(tool) => tool.execute(args),
            None => ToolResult::err(format!("Unknown tool: {}", name)),
        }
    }
}

/// Agent configuration
#[derive(Debug, Clone)]
pub struct AgentConfig {
    pub model: String,
    pub backend: Backend,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub max_iterations: u32,
    pub max_tool_rounds: u32,
    pub temperature: f32,
    pub verbose: bool,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            backend: Backend::Anthropic,
            base_url: None,
            api_key: None,
            max_iterations: 20,
            max_tool_rounds: 10,
            temperature: 0.7,
            verbose: false,
        }
    }
}

/// Parse tool calls from model output
/// Format: <tool:name>args</tool>
fn parse_tool_calls(text: &str) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut remaining = text;

    while let Some(start) = remaining.find("<tool:") {
        let after_prefix = &remaining[start + 6..];

        if let Some(name_end) = after_prefix.find('>') {
            let name = &after_prefix[..name_end];

            let content_start = name_end + 1;
            let end_tag = format!("</tool>");

            if let Some(end) = after_prefix[content_start..].find(&end_tag) {
                let args = &after_prefix[content_start..content_start + end];

                calls.push(ToolCall {
                    name: name.to_string(),
                    args: args.trim().to_string(),
                });

                remaining = &after_prefix[content_start + end + end_tag.len()..];
                continue;
            }
        }
        break;
    }

    calls
}

/// Check if response signals completion
fn is_complete(text: &str) -> bool {
    text.contains("<done>") || text.contains("</done>")
}

/// Extract final answer from response
fn extract_answer(text: &str) -> Option<String> {
    if let Some(start) = text.find("<answer>") {
        let after = &text[start + 8..];
        if let Some(end) = after.find("</answer>") {
            return Some(after[..end].trim().to_string());
        }
    }
    None
}

/// Tool-use Agent
pub struct Agent {
    config: AgentConfig,
    tools: ToolRegistry,
    rlm: Rlm,
}

impl Agent {
    /// Create a new agent
    pub fn new(config: AgentConfig, tools: ToolRegistry) -> rlm::Result<Self> {
        let mut rlm_config = RlmConfig::new(&config.model)
            .with_backend(config.backend.clone())
            .with_max_iterations(config.max_iterations)
            .with_temperature(config.temperature)
            .with_verbose(config.verbose)
            .with_exec_log(true);

        if let Some(ref url) = config.base_url {
            rlm_config = rlm_config.with_base_url(url);
        }
        if let Some(ref key) = config.api_key {
            rlm_config = rlm_config.with_api_key(key);
        }

        let rlm = Rlm::new(rlm_config)?;

        Ok(Self { config, tools, rlm })
    }

    /// Build context with tool docs and conversation
    fn build_context(&self, task: &str, history: &[(String, String)]) -> String {
        let tool_docs = self.tools.generate_docs();

        let mut context = format!(
            r#"You are an AI agent that completes tasks using tools.

AVAILABLE TOOLS:
{tool_docs}

TOOL CALL FORMAT:
<tool:tool_name>arguments</tool>

COMPLETION FORMAT:
When done, output: <answer>your final answer</answer><done>

RULES:
1. Use tools by outputting <tool:name>args</tool>
2. Wait for tool results before continuing
3. You can call multiple tools
4. End with <answer>...</answer><done> when task is complete

IMPORTANT: never simulate tool use.

TASK: {task}
"#,
            tool_docs = tool_docs,
            task = task
        );

        // Add conversation history
        for (role, content) in history {
            context.push_str(&format!("\n{}: {}", role, content));
        }

        context.push_str("\nAssistant: ");
        context
    }

    /// Run the agent on a task
    pub fn run(&self, task: &str) -> rlm::Result<String> {
        let mut history: Vec<(String, String)> = Vec::new();

        for round in 0..self.config.max_tool_rounds {
            if self.config.verbose {
                println!("══ Agent Round {} ══", round + 1);
            }

            // Build context and call RLM
            let context = self.build_context(task, &history);
            let result = self.rlm.completion_with_context(&context, None)?;
            let response = &result.response;

            if self.config.verbose {
                println!("Response: {}", response);
            }

            // Check for completion
            if is_complete(response) {
                if let Some(answer) = extract_answer(response) {
                    return Ok(answer);
                }
                return Ok(response.clone());
            }

            // Parse and execute tool calls
            let tool_calls = parse_tool_calls(response);

            if tool_calls.is_empty() {
                // No tools called, treat response as final
                history.push(("Assistant".to_string(), response.clone()));
                continue;
            }

            // Execute tools and collect results
            let mut tool_output = String::new();
            for call in &tool_calls {
                if self.config.verbose {
                    println!("  Tool: {}({})", call.name, call.args);
                }

                let result = self.tools.execute(&call.name, &call.args);

                if result.success {
                    tool_output
                        .push_str(&format!("[{}] Result:\n{}\n\n", call.name, result.output));
                } else {
                    tool_output.push_str(&format!(
                        "[{}] Error: {}\n\n",
                        call.name,
                        result.error.unwrap_or_default()
                    ));
                }
            }

            // Add to history
            history.push(("Assistant".to_string(), response.clone()));
            history.push(("Tool Results".to_string(), tool_output));
        }

        Err(rlm::RlmError::MaxIterationsReached(
            self.config.max_tool_rounds,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_tool_calls() {
        let text = "Let me read the file <tool:read_file>config.json</tool> and check.";
        let calls = parse_tool_calls(text);

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].args, "config.json");
    }

    #[test]
    fn test_parse_multiple_tools() {
        let text = "<tool:read_file>a.txt</tool> then <tool:read_file>b.txt</tool>";
        let calls = parse_tool_calls(text);

        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn test_is_complete() {
        assert!(is_complete("Here's the answer <answer>42</answer><done>"));
        assert!(!is_complete("Still working..."));
    }

    #[test]
    fn test_extract_answer() {
        let text = "Done! <answer>The result is 42</answer><done>";
        assert_eq!(extract_answer(text), Some("The result is 42".to_string()));
    }
}
