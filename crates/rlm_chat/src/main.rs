//! RLM Chat - Interactive chat CLI
//!
//! RLM inference strategy:
//! - The ENTIRE prompt (context + query) goes into the REPL `context` variable
//! - The system prompt tells the model to examine `context` to find what to do
//! - The model uses the REPL to recursively process the context with sub-LLM calls

use clap::Parser;
use rlm::{Rlm, RlmConfig};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::io::{self, Write};
use std::path::PathBuf;

/// Chat message for history tracking
struct ChatMessage {
    role: &'static str,
    content: String,
}

/// Build the context payload for the REPL `context` variable
///
/// Simple chat format - just User/Assistant turns like normal LLM chat.
fn build_context_payload(
    file_context: Option<&str>,
    history: &[ChatMessage],
    current_query: &str,
) -> String {
    let mut payload = String::new();

    // System prompt
    payload.push_str("System: You are a super nice AI agent in conversation with User.\n\n");

    // File content (if any)
    if let Some(file_content) = file_context {
        payload.push_str(file_content);
        payload.push_str("\n\n");
    }

    // Prior conversation in simple chat format
    for msg in history.iter().take(history.len().saturating_sub(1)) {
        payload.push_str(msg.role);
        payload.push_str(": ");
        payload.push_str(&msg.content);
        payload.push_str("\n");
    }

    // Current query
    payload.push_str("User: ");
    payload.push_str(current_query);
    payload.push_str("\n");
    payload.push_str("Assistant: ");

    payload
}

#[derive(Parser, Debug)]
#[command(name = "rlm_chat")]
#[command(about = "Interactive chat CLI for RLM")]
struct Args {
    /// Model to use
    #[arg(short, long, default_value = "cogito:14b")]
    model: String,

    /// Backend LLM URL
    #[arg(short = 'u', long, default_value = "http://localhost:11434/v1")]
    backend_url: String,

    /// Backend API key
    #[arg(short = 'k', long)]
    backend_key: Option<String>,

    /// Temperature for sampling
    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    /// Verbose mode (show iterations)
    #[arg(short, long)]
    verbose: bool,

    /// Context file to load (large files supported)
    #[arg(short = 'c', long)]
    context_file: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    // Load context file if provided
    let file_context: Option<String> =
        args.context_file
            .as_ref()
            .map(|path| match std::fs::read_to_string(path) {
                Ok(content) => content,
                Err(e) => {
                    eprintln!("Failed to read context file '{}': {}", path.display(), e);
                    std::process::exit(1);
                }
            });

    // Configure RLM
    let config = RlmConfig::new(&args.model)
        .with_max_iterations(50)
        .with_max_exec_retries(3)
        .with_temperature(args.temperature)
        .with_verbose(true);

    // Create RLM instance
    let rlm = match create_rlm(&args, config) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to create RLM: {}", e);
            eprintln!("Make sure the backend is running");
            std::process::exit(1);
        }
    };

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                        RLM Chat                              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("Model:   {}", args.model);
    println!("Backend: {}", args.backend_url);
    if let Some(ref path) = args.context_file {
        let size = file_context.as_ref().map(|c| c.len()).unwrap_or(0);
        println!("Context: {} ({} bytes)", path.display(), size);
    }
    println!();
    println!("Type your message and press Enter. Use Ctrl+C or Ctrl+D to exit.");
    println!();

    // Chat history
    let mut history: Vec<ChatMessage> = Vec::new();

    // Setup readline
    let mut rl = match DefaultEditor::new() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to initialize readline: {}", e);
            std::process::exit(1);
        }
    };

    loop {
        let readline = rl.readline("You: ");

        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                // Add to readline history
                let _ = rl.add_history_entry(input);

                // Add user message to chat history
                history.push(ChatMessage {
                    role: "User",
                    content: input.to_string(),
                });

                // Build context payload - EVERYTHING goes into context (RLM inference strategy)
                let context_payload = build_context_payload(
                    file_context.as_deref(),
                    &history,
                    input, // Current query
                );

                if !args.verbose {
                    print!("Assistant: ");
                    io::stdout().flush().unwrap();
                }

                // Run completion - context_payload goes into REPL `context` variable
                match rlm.completion_with_context(&context_payload, None) {
                    Ok(result) => {
                        // Add assistant response to history
                        history.push(ChatMessage {
                            role: "Assistant",
                            content: result.response.clone(),
                        });

                        if args.verbose {
                            println!();
                            println!(
                                "─────────────────────────────────────────────────────────────"
                            );
                            println!("Assistant: {}", result.response);
                            println!(
                                "─────────────────────────────────────────────────────────────"
                            );
                            println!(
                                "({} iterations, {} tokens, {:?})",
                                result.iterations.len(),
                                result.usage.total_tokens,
                                result.execution_time
                            );
                        } else {
                            println!("{}", result.response);
                        }
                        println!();
                    }
                    Err(e) => {
                        eprintln!("\nError: {}", e);
                        // Remove the failed user message from history
                        history.pop();
                        println!();
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("\nInterrupted. Goodbye!");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("\nGoodbye!");
                break;
            }
            Err(e) => {
                eprintln!("Error: {}", e);
                break;
            }
        }
    }
}

fn create_rlm(args: &Args, config: RlmConfig) -> rlm::Result<Rlm> {
    match &args.backend_key {
        Some(key) => Rlm::with_base_url_and_key(config, &args.backend_url, key),
        None => Rlm::with_base_url(config, &args.backend_url),
    }
}
