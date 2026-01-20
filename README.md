# RLM - Recursive Language Models

LLM inference with interactive REPL-based reasoning. Models explore, analyze, and build responses through iterative Python code execution.

## What is RLM?

Based on: https://github.com/alexzhang13/rlm

RLM gives language models a Python REPL environment for performing text generation for chat completion: "You are a LLM."

## Features

- **Dual Backend Support** - OpenAI-compatible APIs (Ollama, vLLM, etc.) and Anthropic (Claude)
- **Recursive Sub-LLM Calls** - Models can spawn sub-queries for complex reasoning
- **Sandboxed Python REPL** - Safe code execution with PyO3
- **Dynamic Prompting** - Context-aware strategy hints (small/medium/large)
- **Iteration Tracking** - Usage stats, timing, and execution logs

## Installation

```bash
# Clone the repo
git clone https://github.com/annikahannig/rlm
cd rlm/rlm-rs

# Build
cargo build --release
```


## Quick Start

### With Ollama (default)

```bash
# Start Ollama
ollama serve

# Run RLM chat
cargo run -p rlm_chat -- -m ministral-3:14b -e
```

### With Anthropic (Claude)

```bash
# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run with Claude
cargo run -p rlm_chat -- -b anthropic -m claude-sonnet-4-5 -e
```

## CLI Options

```
rlm_chat [OPTIONS]

Options:
  -m, --model <MODEL>        Model to use [default: cogito:14b]
  -b, --backend <BACKEND>    Backend: openai or anthropic [default: openai]
  -u, --backend-url <URL>    API URL for OpenAI-compatible backends
                             [default: http://localhost:11434/v1]
  -k, --backend-key <KEY>    API key (or use env vars)
  -t, --temperature <TEMP>   Sampling temperature [default: 0.7]
  -v, --verbose              Show full iteration details
  -e, --exec-log             Show execution progress (recommended)
  -c, --context-file <FILE>  Load context from file
  -h, --help                 Print help
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                         RLM Loop                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. User prompt → stored in `context` variable              │
│                         ↓                                   │
│  2. LLM generates response with ```repl code block          │
│                         ↓                                   │
│  3. Python code executes in sandboxed REPL                  │
│                         ↓                                   │
│  4. Output shown to LLM → back to step 2                    │
│                         ↓                                   │
│  5. LLM calls llm_output(answer) → done!                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### REPL Functions

| Function | Description |
|----------|-------------|
| `print(value)` | Display output, continue reasoning |
| `llm_query(prompt)` | Query sub-LLM (isolated context!) |
| `llm_output(answer)` | Submit final answer, stop iteration |


## Library Usage

```rust
use rlm::{Rlm, RlmConfig, Backend};

// Configure for Anthropic
let config = RlmConfig::new("claude-sonnet-4-20250514")
    .with_backend(Backend::Anthropic)
    .with_max_iterations(30)
    .with_temperature(0.7);

// Create RLM instance
let rlm = Rlm::new(config)?;

// Run completion
let result = rlm.completion("What is the capital of France?")?;
println!("Response: {}", result.response);
println!("Iterations: {}", result.iterations.len());
```

## Project Structure

```
rlm-rs/
├── src/
│   ├── lib.rs          # Library exports
│   ├── rlm.rs          # Main orchestrator
│   ├── prompts.rs      # System prompts
│   ├── types.rs        # Data types
│   ├── parsing.rs      # Code block extraction
│   ├── error.rs        # Error types
│   └── env/
│       ├── mod.rs      # REPL traits
│       ├── pyo3_repl.rs    # Python REPL implementation
│       └── callback.rs     # LLM callback handlers
├── crates/
│   ├── rlm_chat/       # Interactive CLI
│   └── rlm_server/     # HTTP server (WIP)
└── Cargo.toml
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude models |
| `OPENAI_API_KEY` | OpenAI API key (if using OpenAI directly) |

## Supported Models

### Anthropic
- `claude-opus-4-20250514` - Most capable
- `claude-sonnet-4-20250514` - Balanced
- `claude-haiku-3-5-20241022` - Fast & efficient

### OpenAI-Compatible (Ollama, vLLM, etc.)
- Any model supporting the OpenAI chat completions API
- Works with: `cogito:14b`, `ministra-3:14b`

## Tips

1. **Use `-e` flag** - Shows iteration progress without full verbosity
2. **Start simple** - Test with small prompts first
3. **Check iterations** - If stuck in loops, model may need better prompting
4. **Sub-LLM context** - Remember `llm_query()` can't see your context!

## License

MIT

---

Built with Rust + PyO3 and Claude :)
