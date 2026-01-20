//! RLM Agent CLI - Tool-use agent demo

use clap::Parser;
use rlm::Backend;
use rlm_agent::{tools, Agent, AgentConfig};
use rustyline::DefaultEditor;

#[derive(Debug, Clone, clap::ValueEnum)]
enum CliBackend {
    OpenAI,
    Anthropic,
}

#[derive(Parser)]
#[command(name = "rlm_agent")]
#[command(about = "Tool-use agent powered by RLM")]
struct Args {
    /// Model to use
    #[arg(short, long, default_value = "claude-sonnet-4-20250514")]
    model: String,

    /// Backend: openai or anthropic
    #[arg(short, long, value_enum, default_value = "anthropic")]
    backend: CliBackend,

    /// Backend API URL (for OpenAI-compatible)
    #[arg(short = 'u', long)]
    backend_url: Option<String>,

    /// API key (or use env vars)
    #[arg(short = 'k', long)]
    backend_key: Option<String>,

    /// Temperature for sampling
    #[arg(short, long, default_value = "0.7")]
    temperature: f32,

    /// Max tool execution rounds
    #[arg(long, default_value = "10")]
    max_rounds: u32,

    /// Max RLM iterations per round
    #[arg(long, default_value = "20")]
    max_iterations: u32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Single task (non-interactive mode)
    #[arg(short = 'T', long)]
    task: Option<String>,

    /// Allow all shell commands (dangerous!)
    #[arg(long)]
    allow_all_shell: bool,
}

fn main() {
    let args = Args::parse();

    // Build config
    let backend = match args.backend {
        CliBackend::OpenAI => Backend::OpenAI,
        CliBackend::Anthropic => Backend::Anthropic,
    };

    let mut config = AgentConfig {
        model: args.model.clone(),
        backend,
        base_url: args.backend_url.clone(),
        api_key: args.backend_key.clone(),
        max_iterations: args.max_iterations,
        max_tool_rounds: args.max_rounds,
        temperature: args.temperature,
        verbose: args.verbose,
    };

    // Default URL for OpenAI backend
    if matches!(args.backend, CliBackend::OpenAI) && config.base_url.is_none() {
        config.base_url = Some("http://localhost:11434/v1".to_string());
    }

    // Build tool registry
    let mut tools = tools::default_tools();

    // Replace shell tool if allow_all requested
    if args.allow_all_shell {
        tools.register(tools::ShellTool::allow_all());
    }

    // Create agent
    let agent = match Agent::new(config, tools) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Failed to create agent: {}", e);
            std::process::exit(1);
        }
    };

    println!("RLM Agent - Tool-use demo");
    println!("Model: {}", args.model);
    println!("Backend: {:?}", args.backend);
    println!();

    // Single task mode
    if let Some(task) = args.task {
        run_task(&agent, &task);
        return;
    }

    // Interactive mode
    let mut rl = match DefaultEditor::new() {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Failed to create readline: {}", e);
            std::process::exit(1);
        }
    };

    println!("Available tools: echo, read_file, write_file, list_dir, shell, calc");
    println!("Type 'exit' or Ctrl+D to quit.");
    println!();

    loop {
        let readline = rl.readline("task> ");

        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if line == "exit" || line == "quit" {
                    break;
                }

                let _ = rl.add_history_entry(line);
                run_task(&agent, line);
                println!();
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                println!("Goodbye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }
}

fn run_task(agent: &Agent, task: &str) {
    println!("─── Running task ───");
    println!();

    match agent.run(task) {
        Ok(result) => {
            println!();
            println!("─── Result ───");
            println!("{}", result);
        }
        Err(e) => {
            println!();
            println!("─── Error ───");
            println!("{}", e);
        }
    }
}
