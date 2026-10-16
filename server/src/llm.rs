use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;

pub const SYSTEM_PROMPT: &str =
    "你是一个简洁的中文语音助手。用简体中文回答，控制在两句话以内，口语化，不要用列表或代码。";

/// Minimal OpenRouter client (OpenAI-compatible chat completions + SSE).
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
}

impl LlmClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    pub fn has_key(&self) -> bool {
        !self.api_key.is_empty()
    }

    /// Streams the assistant reply as text deltas into `tx`, then returns.
    /// Mirrors the Python POC's `llm_stream`.
    pub async fn stream_chat(
        &self,
        user_text: &str,
        context: &str,
        tx: mpsc::Sender<String>,
    ) -> Result<()> {
        if !self.has_key() {
            return self.mock_stream(user_text, tx).await;
        }

        let user_content = if context.is_empty() {
            user_text.to_string()
        } else {
            format!("{user_text}\n\n[工具结果]\n{context}")
        };

        let resp = self
            .http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "stream": true,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            }))
            .send()
            .await
            .context("openrouter request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("openrouter {status}: {}", &body[..body.len().min(300)]);
        }

        // --- SSE parsing: `data: {...}` lines, terminated by `data: [DONE]`
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("sse chunk")?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(pos) = buf.find('\n') {
                let line: String = buf.drain(..=pos).collect();
                let line = line.trim();
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(content) = v
                        .pointer("/choices/0/delta/content")
                        .and_then(|c| c.as_str())
                    {
                        if !content.is_empty() && tx.send(content.to_string()).await.is_err() {
                            return Ok(()); // receiver gone
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Fallback when no API key is configured (same as the Python POC).
    async fn mock_stream(&self, user_text: &str, tx: mpsc::Sender<String>) -> Result<()> {
        let mock = format!("我听到了，你说的是「{user_text}」。大语言模型还没接上，这是测试回复。");
        let _ = tx.send(mock).await;
        Ok(())
    }
}
