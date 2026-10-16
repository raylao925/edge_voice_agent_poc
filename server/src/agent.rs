//! Two-stage agent loop (mirrors `pipeline.py::agent_prepare`):
//! 1. detect intent + pick a tool (LLM JSON, or keyword fallback),
//! 2. run the tool, fold its result into `context` for the reply LLM call.

use anyhow::Result;
use serde::Deserialize;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::llm::LlmClient;
use crate::protocol::{Event, ToolCallInfo};
use crate::tools::find_tool;

const INTENT_PROMPT: &str = r#"判断用户这句话是否需要调用实时工具。只返回 JSON，不要其他文字。
可选工具：
- {"tool": "web_search", "query": "搜索关键词"} // 需要上网查资料、事实、解释
- {"tool": "news_search", "query": "新闻关键词"} // 问最新新闻、时事
- {"tool": "get_weather", "city": "城市名"} // 问天气
- {"tool": "none"} // 闲聊、打招呼、不需要实时信息

用户：{text}
JSON:"#;

pub struct AgentOutput {
pub context: String,
pub tool_calls: Vec<ToolCallInfo>,
}

#[derive(Debug, Deserialize)]
struct Intent {
tool: String,
#[serde(default)]
query: Option<String>,
#[serde(default)]
city: Option<String>,
#[serde(default)]
url: Option<String>,
}

fn round2(v: f64) -> f64 {
(v * 100.0).round() / 100.0
}

fn empty_output() -> AgentOutput {
AgentOutput {
context: String::new(),
tool_calls: vec![],
}
}

/// Intent detection + tool execution. Emits `Status` events for the frontend.
pub async fn agent_prepare(
user_text: &str,
llm: &LlmClient,
status_tx: &mpsc::Sender<Event>,
) -> Result<AgentOutput> {
let intent = detect_intent(user_text, llm).await;
let tool_name = intent.tool.as_str();
if tool_name == "none" {
return Ok(empty_output());
}
let Some(tool) = find_tool(tool_name) else {
return Ok(empty_output());
};

let arg = intent
.query
.clone()
.or(intent.city.clone())
.or(intent.url.clone())
.unwrap_or_default();
status_tx
.send(Event::Status {
text: format!("{}：{arg}", tool.label()),
})
.await
.ok();

let t0 = Instant::now();
let result = match tool_name {
"get_weather" => tool.run(intent.city.as_deref().unwrap_or("澳门")).await,
"fetch_page" => tool.run(intent.url.as_deref().unwrap_or("")).await,
_ => tool.run(intent.query.as_deref().unwrap_or(user_text)).await,
};
let latency = t0.elapsed().as_secs_f64();

match result {
Ok(text) => {
tracing::info!("[tool] {tool_name} done in {latency:.1}s");
Ok(AgentOutput {
context: format!("\n{text}"),
tool_calls: vec![ToolCallInfo {
tool: tool_name.to_string(),
arg,
latency: round2(latency),
}],
})
}
Err(e) => {
tracing::warn!("[tool] {tool_name} failed: {e:#}");
status_tx
.send(Event::Status {
text: "⚠️ 工具执行失败".to_string(),
})
.await
.ok();
Ok(AgentOutput {
context: format!("{e:#}"),
tool_calls: vec![],
})
}
}
}

async fn detect_intent(user_text: &str, llm: &LlmClient) -> Intent {
if llm.has_key() {
let prompt = INTENT_PROMPT.replace("{text}", user_text);
if let Ok(raw) = llm.chat_once(&prompt).await {
let clean = raw.replace("```json", "").replace("```", "");
if let Ok(intent) = serde_json::from_str::<Intent>(clean.trim()) {
if matches!(
intent.tool.as_str(),
"web_search" | "news_search" | "get_weather" | "fetch_page" | "none"
) {
return intent;
}
}
}
tracing::warn!("[intent] llm decision failed, falling back to keywords");
}
keyword_intent(user_text)
}

/// Fallback intent when no LLM key (or the LLM call fails):
/// simple keyword matching, mirrors `_keyword_intent`.
fn keyword_intent(text: &str) -> Intent {
let has = |ks: &[&str]| ks.iter().any(|k| text.contains(k));
let none = || Intent {
tool: "none".to_string(),
query: None,
city: None,
url: None,
};

if has(&["天气", "下雨", "气温", "几度"]) {
let mut city = "澳门";
for c in ["澳门", "香港", "台湾", "北京", "上海", "广州", "深圳", "珠海"] {
if text.contains(c) {
city = c;
break;
}
}
return Intent {
tool: "get_weather".to_string(),
city: Some(city.to_string()),
query: None,
url: None,
};
}
if has(&["新闻", "头条", "时事"]) {
let mut q = text.to_string();
for k in ["新闻", "头条", "时事", "最新", "今天", "的", "一下", "查", "搜"] {
q = q.replace(k, "");
}
let q = q.trim();
return Intent {
tool: "news_search".to_string(),
query: Some(if q.is_empty() {
"今日新闻".to_string()
} else {
q.to_string()
}),
city: None,
url: None,
};
}
if has(&["搜索", "搜一下", "查一下", "什么是", "介绍一下"]) {
return Intent {
tool: "web_search".to_string(),
query: Some(text.to_string()),
city: None,
url: None,
};
}
none()
}
