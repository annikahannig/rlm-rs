use anthropic_sdk::{Anthropic, ContentBlock, MessageCreateBuilder};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client as OpenAIClient,
};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::runtime::Runtime;

use crate::env::{execute_with_error_handling, LlmQueryFn, PyO3Repl, ReplEnvironment};
use crate::error::{Result, RlmError};
use crate::parsing::{extract_answer, extract_code_blocks, extract_final_answer_from_stdout};
use crate::prompts::{build_continue_prompt, build_initial_user_prompt, build_system_prompt};
use crate::types::{
    Backend, CodeBlock, Message, PromptInput, ReplResult, RlmCompletion, RlmConfig, RlmIteration,
    Role, Usage,
};

/// LLM client abstraction
enum LlmClient {
    OpenAI(OpenAIClient<OpenAIConfig>),
    Anthropic(Anthropic),
}

/// Truncate response after first ```repl``` or ```python``` block ends
/// Discards everything after the closing ``` to force step-by-step evaluation
fn truncate_after_first_repl_block(text: &str) -> String {
    // Find start of first repl/python block
    let block_start = text.find("```repl\n").or_else(|| text.find("```python\n"));

    let Some(start) = block_start else {
        return text.to_string(); // No block, return as-is
    };

    // Find the closing ``` after the block start
    let after_marker = start + 8; // skip past "```repl\n" or "```python"
    if let Some(end_offset) = text[after_marker..].find("\n```") {
        let end = after_marker + end_offset + 4; // include the closing ```
        text[..end].to_string()
    } else {
        text.to_string() // No closing, return as-is
    }
}

/// Format code execution result for history - simple REPL-style output
fn format_execution_result(result: &ReplResult) -> String {
    let mut output = String::new();

    if !result.stdout.is_empty() {
        output.push_str(&result.stdout);
    }

    if !result.stderr.is_empty() {
        if !output.is_empty() {
            output.push('\n');
        }
        output.push_str(&result.stderr);
    }

    if output.is_empty() {
        "(no output)".to_string()
    } else {
        output
    }
}

/// Main RLM orchestrator
pub struct Rlm {
    config: RlmConfig,
    client: LlmClient,
    runtime: Runtime,
}

impl Rlm {
    /// Create a new RLM instance from config
    ///
    /// Uses config.backend, config.base_url, and config.api_key to configure the client.
    /// Falls back to environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY) if no key provided.
    pub fn new(config: RlmConfig) -> Result<Self> {
        let runtime = Runtime::new()?;
        let client = Self::create_client(&config)?;
        Ok(Self {
            config,
            client,
            runtime,
        })
    }

    /// Create the appropriate LLM client based on config
    fn create_client(config: &RlmConfig) -> Result<LlmClient> {
        match config.backend {
            Backend::OpenAI => {
                let mut openai_config = OpenAIConfig::new();
                if let Some(ref url) = config.base_url {
                    openai_config = openai_config.with_api_base(url);
                }
                if let Some(ref key) = config.api_key {
                    openai_config = openai_config.with_api_key(key);
                } else if config.base_url.is_some() {
                    // For Ollama/local models without explicit key
                    openai_config = openai_config.with_api_key("ollama");
                }
                Ok(LlmClient::OpenAI(OpenAIClient::with_config(openai_config)))
            }
            Backend::Anthropic => {
                let anthropic = if let Some(ref key) = config.api_key {
                    Anthropic::new(key).map_err(|e| RlmError::Config(e.to_string()))?
                } else {
                    Anthropic::from_env().map_err(|e| RlmError::Config(e.to_string()))?
                };
                Ok(LlmClient::Anthropic(anthropic))
            }
        }
    }

    /// Create with explicit API key (legacy, prefer using config.with_api_key())
    pub fn with_api_key(config: RlmConfig, api_key: &str) -> Result<Self> {
        let config = RlmConfig {
            api_key: Some(api_key.to_string()),
            ..config
        };
        Self::new(config)
    }

    /// Create with custom base URL (for Ollama, local models, etc.)
    pub fn with_base_url(config: RlmConfig, base_url: &str) -> Result<Self> {
        let config = RlmConfig {
            base_url: Some(base_url.to_string()),
            ..config
        };
        Self::new(config)
    }

    /// Create with custom base URL and API key
    pub fn with_base_url_and_key(config: RlmConfig, base_url: &str, api_key: &str) -> Result<Self> {
        let config = RlmConfig {
            base_url: Some(base_url.to_string()),
            api_key: Some(api_key.to_string()),
            ..config
        };
        Self::new(config)
    }

    /// Run a completion with the given prompt
    ///
    /// The entire prompt (data + question) goes into the REPL `context` variable.
    /// The system prompt tells the model to examine `context` to find what to do.
    pub fn completion(&self, prompt: impl Into<PromptInput>) -> Result<RlmCompletion> {
        let prompt = prompt.into();
        let context_payload = match &prompt {
            PromptInput::Text(s) => s.clone(),
            PromptInput::Messages(msgs) => msgs
                .iter()
                .filter(|m| m.role == Role::User)
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
        };
        // Root prompt is optional - can be used to remind the model of the original question
        self.completion_with_context(&context_payload, None)
    }

    /// Run a completion with context payload and optional root prompt reminder
    ///
    /// RLM inference strategy:
    /// - `context_payload`: Goes into REPL as `context` variable (data + query combined)
    /// - `root_prompt`: Optional short reminder of the question (shown in user prompts)
    ///
    /// The model uses the REPL to examine `context` and recursively process it.
    pub fn completion_with_context(
        &self,
        context_payload: &str,
        _root_prompt: Option<&str>,
    ) -> Result<RlmCompletion> {
        let prompt = PromptInput::Text(context_payload.to_string());
        let start = Instant::now();

        // Build initial messages - system prompt includes context metadata
        let system_prompt = build_system_prompt(context_payload.len());

        // Initial user message - tells model to start examining context
        let initial_user_msg = build_initial_user_prompt();

        let mut history: Vec<Message> = vec![
            Message::system(system_prompt),
            Message::user(&initial_user_msg),
        ];

        let mut iterations: Vec<RlmIteration> = Vec::new();
        let mut total_usage = Usage::default();

        // Create REPL with callback that uses our backend config
        let backend_for_callback = self.config.backend.clone();
        let model_for_callback = self.config.model.clone();
        let temp_for_callback = self.config.temperature;
        let api_key_for_callback = self.config.api_key.clone();
        let base_url_for_callback = self.config.base_url.clone();

        // We need to track usage from sub-calls
        let sub_call_usage = Arc::new(Mutex::new(Usage::default()));
        let sub_call_usage_for_callback = sub_call_usage.clone();

        let query_fn: LlmQueryFn = Arc::new(move |prompt: &str| {
            // Create a new runtime for the callback (we're in a different thread context)
            let rt = match Runtime::new() {
                Ok(rt) => rt,
                Err(e) => return Err(format!("Runtime error: {}", e)),
            };

            rt.block_on(async {
                match backend_for_callback {
                    Backend::OpenAI => {
                        // Create OpenAI client for sub-call
                        let mut openai_config = OpenAIConfig::new();
                        if let Some(ref url) = base_url_for_callback {
                            openai_config = openai_config.with_api_base(url);
                        }
                        if let Some(ref key) = api_key_for_callback {
                            openai_config = openai_config.with_api_key(key);
                        } else if base_url_for_callback.is_some() {
                            openai_config = openai_config.with_api_key("ollama");
                        }
                        let client = OpenAIClient::with_config(openai_config);

                        let messages = vec![ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(prompt)
                                .build()
                                .map_err(|e| e.to_string())?,
                        )];

                        let request = CreateChatCompletionRequestArgs::default()
                            .model(&model_for_callback)
                            .messages(messages)
                            .temperature(temp_for_callback)
                            .build()
                            .map_err(|e| e.to_string())?;

                        let response = client
                            .chat()
                            .create(request)
                            .await
                            .map_err(|e| e.to_string())?;

                        // Track usage
                        if let Some(usage) = &response.usage {
                            let mut guard = sub_call_usage_for_callback.lock().unwrap();
                            guard.input_tokens += usage.prompt_tokens as u64;
                            guard.output_tokens += usage.completion_tokens as u64;
                            guard.total_tokens += usage.total_tokens as u64;
                        }

                        let content = response
                            .choices
                            .first()
                            .and_then(|c| c.message.content.clone())
                            .unwrap_or_default();

                        Ok(content)
                    }
                    Backend::Anthropic => {
                        // Create Anthropic client for sub-call
                        let client = if let Some(ref key) = api_key_for_callback {
                            Anthropic::new(key).map_err(|e| e.to_string())?
                        } else {
                            Anthropic::from_env().map_err(|e| e.to_string())?
                        };

                        let params = MessageCreateBuilder::new(&model_for_callback, 4096)
                            .user(prompt)
                            .build();

                        let response = client
                            .messages()
                            .create(params)
                            .await
                            .map_err(|e| e.to_string())?;

                        // Track usage
                        {
                            let mut guard = sub_call_usage_for_callback.lock().unwrap();
                            guard.input_tokens += response.usage.input_tokens as u64;
                            guard.output_tokens += response.usage.output_tokens as u64;
                            guard.total_tokens +=
                                (response.usage.input_tokens + response.usage.output_tokens) as u64;
                        }

                        // Extract text from content blocks
                        let content = response
                            .content
                            .iter()
                            .filter_map(|block| {
                                if let ContentBlock::Text { text } = block {
                                    Some(text.as_str())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("");

                        Ok(content)
                    }
                }
            })
        });

        let mut repl = PyO3Repl::new(query_fn)?;

        // Add context variable to REPL - this is the DATA to analyze, not instructions
        repl.add_context("context", context_payload)?;

        // Main iteration loop
        for iteration_num in 0..self.config.max_iterations {
            let iter_start = Instant::now();

            // Minimal progress log
            if self.config.exec_log && !self.config.verbose {
                println!("── iter {} ──", iteration_num + 1);
                let _ = io::stdout().flush();
            }

            if self.config.verbose {
                println!("┌─────────────────────────────────────────────────────────────┐");
                println!(
                    "│ ITERATION {:3}                                               │",
                    iteration_num + 1
                );
                println!("└─────────────────────────────────────────────────────────────┘");
                println!();
                println!("📥 LLM Query (message history):");
                println!("─────────────────────────────────────────────────────────────");
                for (i, msg) in history.iter().enumerate() {
                    let role_str = match msg.role {
                        Role::System => "SYSTEM",
                        Role::User => "USER",
                        Role::Assistant => "ASSISTANT",
                    };
                    let content_preview = if msg.content.len() > 10500 {
                        format!(
                            "{}...[truncated, {} chars total]",
                            &msg.content[..500],
                            msg.content.len()
                        )
                    } else {
                        msg.content.clone()
                    };
                    println!("[{}] {}: {}", i, role_str, content_preview);
                    println!();
                }
                println!("─────────────────────────────────────────────────────────────");
                println!();
                println!(
                    "📦 REPL context variable ({} chars):",
                    context_payload.len()
                );
                if context_payload.len() > 300 {
                    println!("{}...[truncated]", &context_payload[..300]);
                } else if context_payload.is_empty() {
                    println!("(empty)");
                } else {
                    println!("{}", context_payload);
                }
                println!("─────────────────────────────────────────────────────────────");
                let _ = io::stdout().flush();
            }

            // Call LLM
            let (raw_response, usage) = self.call_llm(&history)?;
            total_usage.add(&usage);

            // Truncate after first ```repl``` block ends - discard everything after
            let response_text = truncate_after_first_repl_block(&raw_response);

            if self.config.verbose {
                println!();
                println!("📤 LLM Response:");
                println!("─────────────────────────────────────────────────────────────");
                if response_text.len() > 2000 {
                    println!("{}...[truncated]", &response_text[..2000]);
                } else {
                    println!("{}", response_text);
                }
                println!("─────────────────────────────────────────────────────────────");
                let _ = io::stdout().flush();
            }

            // Add assistant response to history
            history.push(Message::assistant(&response_text));

            // Extract code blocks - only execute the FIRST one, throw away extras
            // This forces step-by-step evaluation
            let code_blocks = extract_code_blocks(&response_text);
            let mut executed_blocks: Vec<CodeBlock> = Vec::new();

            if self.config.verbose && code_blocks.is_empty() {
                println!("📝 No code blocks in this iteration");
                let _ = io::stdout().flush();
            } else if self.config.exec_log && !self.config.verbose && code_blocks.is_empty() {
                println!("   (no code)");
                let _ = io::stdout().flush();
            }

            // Only execute first code block (step-by-step)
            if let Some(code) = code_blocks.first() {
                if self.config.exec_log && !self.config.verbose {
                    // Show first line of code as preview
                    let preview: String =
                        code.lines().next().unwrap_or("").chars().take(50).collect();
                    println!(
                        "   ⚡ {}{}",
                        preview,
                        if code.len() > 50 { "..." } else { "" }
                    );
                    let _ = io::stdout().flush();
                }
                if self.config.verbose {
                    if code_blocks.len() > 1 {
                        println!(
                            "📝 Executing Code Block 1 of {} (others discarded):",
                            code_blocks.len()
                        );
                    } else {
                        println!("📝 Executing Code Block:");
                    }
                    println!("┌─────────────────────────────────────────────────────────────┐");
                    for line in code.lines() {
                        println!("│ {}", line);
                    }
                    println!("└─────────────────────────────────────────────────────────────┘");
                    let _ = io::stdout().flush();
                }

                let block_result =
                    self.execute_with_retry(&mut repl, code, &mut history, &mut total_usage)?;

                if self.config.exec_log && !self.config.verbose {
                    if let Some(ref res) = block_result.result {
                        if res.success {
                            print!("   → ✓");
                            if !res.stdout.is_empty() {
                                // Show first line of output
                                let out_preview: String = res
                                    .stdout
                                    .lines()
                                    .next()
                                    .unwrap_or("")
                                    .chars()
                                    .take(60)
                                    .collect();
                                print!(" {}", out_preview);
                                if res.stdout.lines().count() > 1 {
                                    print!(" (+{} lines)", res.stdout.lines().count() - 1);
                                }
                            }
                            println!();
                        } else {
                            println!("   → ✗ {}", res.error.as_deref().unwrap_or("error"));
                        }
                    }
                    let _ = io::stdout().flush();
                }
                if self.config.verbose {
                    if let Some(ref res) = block_result.result {
                        if res.success {
                            println!(
                                "✅ Execution SUCCESS (retries: {})",
                                block_result.retry_count
                            );
                            if !res.stdout.is_empty() {
                                println!("📤 Output:");
                                for line in res.stdout.lines() {
                                    println!("   {}", line);
                                }
                            }
                        } else {
                            println!(
                                "❌ Execution FAILED (retries: {})",
                                block_result.retry_count
                            );
                            if let Some(ref err) = res.error {
                                println!("   Error: {}", err);
                            }
                        }
                    }
                    let _ = io::stdout().flush();
                }

                executed_blocks.push(block_result);
            }

            // Check for final answer - llm_output() invocation is the primary signal
            // Also check FINAL patterns for backwards compatibility
            let locals = repl.get_locals();

            // Primary: Check if llm_output was called in code execution
            let llm_output_answer = executed_blocks
                .iter()
                .filter_map(|b| b.result.as_ref())
                .find_map(|r| r.llm_output.clone());

            // Fallback: Check stdout from code execution (FINAL_ANSWER: prefix)
            let final_from_code = executed_blocks
                .iter()
                .filter_map(|b| b.result.as_ref())
                .find_map(|r| extract_final_answer_from_stdout(&r.stdout));

            // Fallback: Check response text for FINAL patterns
            let final_answer = llm_output_answer
                .or(final_from_code)
                .or_else(|| extract_answer(&response_text, &locals));

            if self.config.exec_log && !self.config.verbose && final_answer.is_some() {
                println!("   🎯 FINAL");
                let _ = io::stdout().flush();
            }
            if self.config.verbose {
                println!("⏱️  Iteration time: {:?}", iter_start.elapsed());
                if final_answer.is_some() {
                    println!("🎯 FINAL answer detected!");
                }
                println!();
                let _ = io::stdout().flush();
            }

            iterations.push(RlmIteration {
                iteration: iteration_num,
                response: response_text.clone(),
                code_blocks: executed_blocks,
                final_answer: final_answer.clone(),
                execution_time: iter_start.elapsed(),
            });

            // If we found a final answer, we're done
            if let Some(answer) = final_answer {
                // Add sub-call usage
                let sub_usage = sub_call_usage.lock().unwrap();
                total_usage.add(&sub_usage);

                return Ok(RlmCompletion {
                    prompt,
                    response: answer,
                    iterations,
                    usage: total_usage,
                    execution_time: start.elapsed(),
                });
            }

            // Note: execution results already added to history in execute_with_retry

            // Add continue prompt to keep model on track
            let continue_msg = build_continue_prompt(iteration_num, self.config.max_iterations);
            history.push(Message::user(&continue_msg));
        }

        Err(RlmError::MaxIterationsReached(self.config.max_iterations))
    }

    /// Call the LLM with the current history
    fn call_llm(&self, history: &[Message]) -> Result<(String, Usage)> {
        match &self.client {
            LlmClient::OpenAI(client) => self.call_openai(client, history),
            LlmClient::Anthropic(client) => self.call_anthropic(client, history),
        }
    }

    /// Call OpenAI-compatible API
    fn call_openai(
        &self,
        client: &OpenAIClient<OpenAIConfig>,
        history: &[Message],
    ) -> Result<(String, Usage)> {
        let messages: Vec<ChatCompletionRequestMessage> = history
            .iter()
            .map(|m| match m.role {
                Role::System => ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(m.content.clone())
                        .build()
                        .unwrap(),
                ),
                Role::User => ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(m.content.clone())
                        .build()
                        .unwrap(),
                ),
                Role::Assistant => ChatCompletionRequestMessage::Assistant(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(m.content.clone())
                        .build()
                        .unwrap(),
                ),
            })
            .collect();

        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder
            .model(&self.config.model)
            .messages(messages)
            .temperature(self.config.temperature);

        if let Some(max_tokens) = self.config.max_tokens {
            request_builder.max_tokens(max_tokens);
        }

        let request = request_builder.build()?;

        let response = self
            .runtime
            .block_on(async { client.chat().create(request).await })?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        let usage = response
            .usage
            .map(|u| Usage::new(u.prompt_tokens as u64, u.completion_tokens as u64))
            .unwrap_or_default();

        Ok((content, usage))
    }

    /// Call Anthropic API
    fn call_anthropic(&self, client: &Anthropic, history: &[Message]) -> Result<(String, Usage)> {
        // Extract system message
        let system_content = history
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone());

        // Build request using builder pattern
        let max_tokens = self.config.max_tokens.unwrap_or(4096);
        let mut builder = MessageCreateBuilder::new(&self.config.model, max_tokens);

        // Add system prompt if present
        if let Some(system) = system_content {
            builder = builder.system(system);
        }

        // Add temperature if set
        if self.config.temperature > 0.0 {
            builder = builder.temperature(self.config.temperature);
        }

        // Add messages (skip system messages)
        for msg in history.iter().filter(|m| m.role != Role::System) {
            builder = match msg.role {
                Role::User => builder.user(msg.content.clone()),
                Role::Assistant => builder.assistant(msg.content.clone()),
                Role::System => builder, // shouldn't happen due to filter
            };
        }

        let params = builder.build();

        let response = self
            .runtime
            .block_on(async { client.messages().create(params).await })
            .map_err(|e| RlmError::Api(e.to_string()))?;

        // Extract text from content blocks
        let content = response
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        let usage = Usage::new(
            response.usage.input_tokens as u64,
            response.usage.output_tokens as u64,
        );

        Ok((content, usage))
    }

    /// Execute code with automatic retry on failure
    fn execute_with_retry(
        &self,
        repl: &mut PyO3Repl,
        code: &str,
        history: &mut Vec<Message>,
        total_usage: &mut Usage,
    ) -> Result<CodeBlock> {
        let mut retry_count = 0;
        let mut current_code = code.to_string();

        loop {
            let result = execute_with_error_handling(repl, &current_code)?;

            // Add execution result to history wrapped in ```result block
            let output = if result.success {
                if result.stdout.is_empty() {
                    "```result\n(no output)\n```".to_string()
                } else {
                    format!("```result\n{}\n```", result.stdout.trim())
                }
            } else {
                format!(
                    "```error\n{}\n```",
                    result
                        .error
                        .as_ref()
                        .unwrap_or(&"Unknown error".to_string())
                )
            };
            history.push(Message::user(&output));

            // If success or max retries reached, return
            if result.success || retry_count >= self.config.max_exec_retries {
                return Ok(CodeBlock {
                    code: current_code,
                    result: Some(result),
                    retry_count,
                });
            }

            // Ask LLM to fix the error
            retry_count += 1;

            let fix_prompt = "Please fix the code and try again. Provide the corrected code in a ```repl``` block.";
            history.push(Message::user(fix_prompt));

            // Call LLM for fix
            let (fix_response, usage) = self.call_llm(history)?;
            total_usage.add(&usage);

            history.push(Message::assistant(&fix_response));

            // Extract fixed code
            let fixed_blocks = extract_code_blocks(&fix_response);
            if let Some(fixed) = fixed_blocks.first() {
                current_code = fixed.clone();
            } else {
                // No code block in fix response, return with error
                return Ok(CodeBlock {
                    code: current_code,
                    result: Some(result),
                    retry_count,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rlm_config_default() {
        let config = RlmConfig::default();
        assert_eq!(config.model, "gpt-4o");
        assert_eq!(config.max_iterations, 20);
        assert_eq!(config.max_exec_retries, 2);
        assert_eq!(config.temperature, 0.0);
    }

    #[test]
    fn test_rlm_config_builder() {
        let config = RlmConfig::new("gpt-4o-mini")
            .with_max_iterations(10)
            .with_max_exec_retries(3)
            .with_temperature(0.5)
            .with_verbose(true);

        assert_eq!(config.model, "gpt-4o-mini");
        assert_eq!(config.max_iterations, 10);
        assert_eq!(config.max_exec_retries, 3);
        assert_eq!(config.temperature, 0.5);
        assert!(config.verbose);
    }
}
