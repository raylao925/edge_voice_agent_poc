//! Voice agent MVP server (Rust).
//!
//! Pipeline: client mic -> FunASR Paraformer GGUF -> agent (tools) ->
//! OpenRouter LLM (SSE) -> per-sentence Piper TTS.
//! Speaks the same WebSocket protocol as the Python POC, so `www/index.html`
//! works unchanged.

mod agent;
mod config;
mod llm;
mod pipeline;
mod protocol;
mod stt;
mod tools;
mod tts;
mod ws;

use axum::{response::Html, routing::get, Router};
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::llm::LlmClient;
use crate::pipeline::AppContext;
use crate::stt::FunAsrSubprocess;
use crate::tts::UnimplementedTts;

async fn index(
    axum::extract::State(ctx): axum::extract::State<Arc<AppContext>>,
) -> Html<String> {
    let html = tokio::fs::read_to_string(ctx.config.www_dir.join("index.html"))
        .await
        .unwrap_or_else(|_| "<h1>www/index.html not found</h1>".to_string());
    Html(html)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let config = Config::load();
    info!("www_dir: {}", config.www_dir.display());
    info!("stt: {} {}", config.stt_bin.display(), config.stt_model.display());
    info!("llm model: {} (key configured: {})", config.openrouter_model, !config.openrouter_key.is_empty());

    let ctx = Arc::new(AppContext {
        llm: LlmClient::new(config.openrouter_key.clone(), config.openrouter_model.clone()),
        stt: Arc::new(FunAsrSubprocess::new(config.stt_bin.clone(), config.stt_model.clone())),
        tts: Arc::new(UnimplementedTts),
        config: config.clone(),
    });

    let app = Router::new()
        .route("/", get(index))
        .route("/index.html", get(index))
        .route("/ws", get(ws::ws_handler))
        .with_state(ctx);

    let addr = format!("127.0.0.1:{}", config.port);
    info!("[server] open http://{addr} in your browser");
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
