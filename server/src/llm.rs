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

    /// One-shot (non-streaming) chat completion, used for intent classification.
    /// Mirrors the Python POC's `_llm_json`.
    pub async fn chat_once(&self, prompt: &str) -> Result<String> {
        anyhow::ensure!(self.has_key(), "no openrouter key");
        let resp = self
            .http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "max_tokens": 120,
                "messages": [{"role": "user", "content": prompt}],
            }))
            .send()
            .await
            .context("openrouter request")?
            .error_for_status()
            .context("openrouter status")?;
        let v: serde_json::Value = resp.json().await.context("openrouter json")?;
        v.pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .context("bad chat response")
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

        let mut system = SYSTEM_PROMPT.to_string();
        if !context.is_empty() {
            // Mirror the Python POC: tool results go into the system prompt
            // so the reply stays grounded instead of hallucinated.
            let ctx: String = context.chars().take(3000).collect();
            system += &format!("\n以下是实时工具返回的参考资料，请基于它们回答，不要编造：\n{ctx}");
        }

        let resp = self
            .http
            .post("https://openrouter.ai/api/v1/chat/completions")
            .bearer_auth(&self.api_key)
            .json(&json!({
                "model": self.model,
                "stream": true,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_text},
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
