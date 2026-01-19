//! RLM Server - OpenAI-compatible API for RLM

mod handlers;
mod types;

use axum::{routing::{get, post}, Router};
use clap::Parser;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use handlers::{create_chat_completion, list_models, AppState};

/// RLM Server - OpenAI-compatible API for Recursive Language Models
#[derive(Parser, Debug)]
#[command(name = "rlm-server")]
#[command(about = "Run RLM as an OpenAI-compatible API server")]
struct Args {
    /// Port to listen on
    #[arg(short, long, default_value = "8080")]
    port: u16,

    /// Model to use for completions
    #[arg(short, long, default_value = "gpt-4o")]
    model: String,

    /// Backend LLM URL (e.g., http://localhost:11434/v1 for Ollama)
    #[arg(short = 'u', long, default_value = "https://api.openai.com/v1")]
    backend_url: String,

    /// Backend API key (optional, uses OPENAI_API_KEY env var if not provided)
    #[arg(short = 'k', long)]
    backend_key: Option<String>,
}

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    // Resolve API key from args or environment
    let backend_key = args.backend_key.or_else(|| std::env::var("OPENAI_API_KEY").ok());

    let state = Arc::new(AppState {
        model: args.model.clone(),
        backend_url: args.backend_url.clone(),
        backend_key,
    });

    // CORS configuration for browser clients
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/v1/chat/completions", post(create_chat_completion))
        .route("/v1/models", get(list_models))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!("RLM Server starting on {}", addr);
    tracing::info!("Model: {}", args.model);
    tracing::info!("Backend URL: {}", args.backend_url);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
