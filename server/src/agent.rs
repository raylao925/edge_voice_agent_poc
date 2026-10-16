//! Two-stage agent loop (mirrors `pipeline.py::agent_prepare`):
//! 1. detect intent + pick a tool (LLM JSON, or keyword fallback),
//! 2. run the tool, fold its result into `context` for the reply LLM call.

use anyhow::Result;
use serde::Deserialize;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::conversation::Turn;
use crate::llm::LlmClient;
use crate::protocol::{Event, ToolCallInfo};
use crate::timers::{TimerRegistry, MAX_TIMER_SECS};
use crate::tools::find_tool;

const INTENT_PROMPT: &str = r#"判断用户这句话是否需要调用实时工具。只返回 JSON，不要其他文字。
可选工具：
- {"tool": "web_search", "query": "搜索关键词"} // 需要上网查资料、事实、解释
- {"tool": "news_search", "query": "新闻关键词"} // 问最新新闻、时事
- {"tool": "get_weather", "city": "城市名"} // 问天气
- {"tool": "set_timer", "seconds": 300, "message": "喝水"} // 用户要定时提醒，如"5分钟后提醒我喝水"、"明天早上8点叫我起床"：把时间换算成从现在起的秒数填入 seconds
- {"tool": "list_timers"} // 用户问设了哪些提醒，如"我设了哪些定时器"
- {"tool": "cancel_timer", "id": "t1"} // 用户要取消定时器；说"取消所有"时 id 填 "all"
- {"tool": "none"} // 闲聊、打招呼、不需要实时信息

现在时间：{now}（UTC+8）。换算 seconds 时以此为准，不要瞎猜。

用户：{text}
JSON:"#;

/// Current local time (Asia/Macau = UTC+8, no DST) as "YYYY-MM-DD HH:MM".
/// No chrono dependency: days-to-civil by hand.
fn now_local_string() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
        + 8 * 3600;
    let days = (t / 86400) as i64;
    let sod = t % 86400;
    // Howard Hinnant's civil_from_days
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}",
        y,
        m,
        d,
        sod / 3600,
        (sod % 3600) / 60
    )
}

/// "5分钟后" -> "5分钟" (for stripping the time phrase from the message).
fn fmt_duration_cn(seconds: u64) -> String {
    if seconds % 86400 == 0 {
        format!("{}天", seconds / 86400)
    } else if seconds % 3600 == 0 {
        format!("{}小时", seconds / 3600)
    } else if seconds % 60 == 0 {
        format!("{}分钟", seconds / 60)
    } else {
        format!("{seconds}秒")
    }
}

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
#[serde(default)]
seconds: Option<u64>,
#[serde(default)]
message: Option<String>,
#[serde(default)]
id: Option<String>,
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
/// `last_turn` (if any) is included in the intent prompt so follow-ups like
/// 「那明天呢？」 resolve against the previous turn.
pub async fn agent_prepare(
user_text: &str,
llm: &LlmClient,
last_turn: Option<Turn>,
status_tx: &mpsc::Sender<Event>,
timers: &TimerRegistry,
) -> Result<AgentOutput> {
let intent = detect_intent(user_text, llm, last_turn).await;
let tool_name = intent.tool.as_str();
if tool_name == "none" {
return Ok(empty_output());
}
// Timer tools are handled locally: they need the per-connection registry,
// not the generic Tool trait.
if matches!(tool_name, "set_timer" | "list_timers" | "cancel_timer") {
return handle_timer_tool(&intent, timers, status_tx).await;
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

/// set_timer / list_timers / cancel_timer: local tools backed by the
/// per-connection TimerRegistry.
async fn handle_timer_tool(
intent: &Intent,
timers: &TimerRegistry,
status_tx: &mpsc::Sender<Event>,
) -> Result<AgentOutput> {
let tool_name = intent.tool.as_str();
match tool_name {
"set_timer" => {
let seconds = intent.seconds.unwrap_or(0);
let message = intent.message.clone().unwrap_or_default().trim().to_string();
if seconds < 1 || seconds > MAX_TIMER_SECS || message.is_empty() {
return Ok(AgentOutput {
context: "定时器设置失败：时间需在1秒到7天之间，且提醒内容不能为空。请换个说法再试一次。".to_string(),
tool_calls: vec![],
});
}
let (id, _) = timers.set(seconds, message.clone());
status_tx
.send(Event::Status {
text: format!("⏰ 已设置：{}后提醒「{message}」", fmt_duration_cn(seconds)),
})
.await
.ok();
Ok(AgentOutput {
context: format!("已设置定时器（id={id}）：{seconds} 秒后提醒「{message}」。时间到时我会自动语音提醒。请自然地向用户确认。"),
tool_calls: vec![ToolCallInfo {
tool: "set_timer".to_string(),
arg: message,
latency: 0.0,
}],
})
}
"list_timers" => {
let list = timers.list();
let context = if list.is_empty() {
"当前没有设置任何定时器。".to_string()
} else {
let lines: Vec<String> = list
.iter()
.map(|(id, msg, remain)| format!("- id={id}：{}后提醒「{msg}」", fmt_duration_cn(*remain)))
.collect();
format!("当前定时器：\n{}", lines.join("\n"))
};
Ok(AgentOutput {
context,
tool_calls: vec![ToolCallInfo {
tool: "list_timers".to_string(),
arg: String::new(),
latency: 0.0,
}],
})
}
"cancel_timer" => {
let id = intent.id.clone().unwrap_or_default();
let n = timers.cancel(&id);
let context = if n > 0 {
status_tx
.send(Event::Status {
text: format!("⏰ 已取消 {n} 个定时器"),
})
.await
.ok();
format!("已取消 {n} 个定时器。请自然地向用户确认。")
} else {
"没有找到对应的定时器（可能已经响过或 id 不对）。可以用 list_timers 先看看有哪些。".to_string()
};
Ok(AgentOutput {
context,
tool_calls: vec![ToolCallInfo {
tool: "cancel_timer".to_string(),
arg: id,
latency: 0.0,
}],
})
}
_ => Ok(empty_output()),
}
}

async fn detect_intent(user_text: &str, llm: &LlmClient, last_turn: Option<Turn>) -> Intent {
if llm.has_key() {
let mut prompt = String::new();
if let Some(t) = last_turn {
prompt += &format!("上轮对话：\n用户：{}\n助手：{}\n\n", t.user_text, t.reply);
}
prompt += &INTENT_PROMPT
.replace("{now}", &now_local_string())
.replace("{text}", user_text);
if let Ok(raw) = llm.chat_once(&prompt).await {
let clean = raw.replace("```json", "").replace("```", "");
if let Ok(intent) = serde_json::from_str::<Intent>(clean.trim()) {
if matches!(
intent.tool.as_str(),
"web_search" | "news_search" | "get_weather" | "fetch_page" | "set_timer"
| "list_timers" | "cancel_timer" | "none"
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
seconds: None,
message: None,
id: None,
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
seconds: None,
message: None,
id: None,
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
seconds: None,
message: None,
id: None,
};
}
if has(&["搜索", "搜一下", "查一下", "什么是", "介绍一下"]) {
return Intent {
tool: "web_search".to_string(),
query: Some(text.to_string()),
city: None,
url: None,
seconds: None,
message: None,
id: None,
};
}
// Timer / reminder fallback (no LLM): "5分钟后提醒我喝水".
if has(&["提醒", "定时", "闹钟"])
|| (text.contains("叫我") && has(&["分钟", "小时", "秒", "天", "半小时"]))
{
// Explicit "cancel everything" wins over everything else.
if has(&["取消", "删除"]) && has(&["所有", "全部"]) {
return Intent {
tool: "cancel_timer".to_string(),
query: None,
city: None,
url: None,
seconds: None,
message: None,
id: Some("all".to_string()),
};
}
// A duration + reminder words means "set a timer", even if the text
// also mentions 取消 (e.g. "一分钟后提醒我测试取消").
if let Some((seconds, phrase)) = parse_duration(text) {
let message = extract_reminder_message(text, &phrase);
if !message.is_empty() {
return Intent {
tool: "set_timer".to_string(),
query: None,
city: None,
url: None,
seconds: Some(seconds),
message: Some(message),
id: None,
};
}
}
if has(&["哪些", "几个", "列表", "查看", "多少个"]) {
return Intent {
tool: "list_timers".to_string(),
query: None,
city: None,
url: None,
seconds: None,
message: None,
id: None,
};
}
// Single cancel without an id, or anything else: no tool; the reply
// LLM handles it conversationally.
}
none()
}

/// "5分钟后提醒我喝水" -> Some((300, "5分钟")).
/// Handles arabic digits and common chinese numerals (五/十五/二十/一百).
fn parse_duration(text: &str) -> Option<(u64, String)> {
if text.contains("半小时") {
return Some((1800, "半小时".to_string()));
}
let chars: Vec<char> = text.chars().collect();
let mut i = 0;
while i < chars.len() {
if let Some((num, used)) = parse_number_at(&chars, i) {
let rest: String = chars[i + used..].iter().collect();
let (mult, unit_len) = if rest.starts_with("分钟") {
(60u64, 2usize)
} else if rest.starts_with("小时") || rest.starts_with("钟头") {
(3600, 2)
} else if rest.starts_with('秒') {
(1, 1)
} else if rest.starts_with('分') {
(60, 1)
} else if rest.starts_with('天') {
(86400, 1)
} else {
(0, 0)
};
if mult > 0 {
let phrase: String = chars[i..i + used + unit_len].iter().collect();
let seconds = num * mult;
if (1..=MAX_TIMER_SECS).contains(&seconds) {
return Some((seconds, phrase));
}
return None;
}
}
i += 1;
}
None
}

/// Parse an arabic ("15") or chinese ("十五", "二十") number at `chars[i..]`.
/// Returns (value, chars consumed).
fn parse_number_at(chars: &[char], i: usize) -> Option<(u64, usize)> {
let c = chars[i];
if c.is_ascii_digit() {
let mut j = i;
while j < chars.len() && chars[j].is_ascii_digit() {
j += 1;
}
let s: String = chars[i..j].iter().collect();
return s.parse::<u64>().ok().map(|n| (n, j - i));
}
let digit = |c: char| match c {
'一' => Some(1u64),
'二' | '两' => Some(2),
'三' => Some(3),
'四' => Some(4),
'五' => Some(5),
'六' => Some(6),
'七' => Some(7),
'八' => Some(8),
'九' => Some(9),
_ => None,
};
let mut j = i;
let mut value: u64 = 0;
let mut any = false;
if j + 1 < chars.len() {
if let (Some(d), '百') = (digit(chars[j]), chars[j + 1]) {
value += d * 100;
j += 2;
any = true;
}
}
if j < chars.len() {
if chars[j] == '十' {
value += 10;
j += 1;
any = true;
} else if j + 1 < chars.len() {
if let (Some(d), '十') = (digit(chars[j]), chars[j + 1]) {
value += d * 10;
j += 2;
any = true;
}
}
}
if j < chars.len() {
if let Some(d) = digit(chars[j]) {
value += d;
j += 1;
any = true;
}
}
if any && j > i {
Some((value, j - i))
} else {
None
}
}

/// Drop the duration phrase and trigger words, leaving the reminder text:
/// "5分钟后提醒我喝水" -> "喝水".
fn extract_reminder_message(text: &str, phrase: &str) -> String {
let mut m = text.replace(phrase, "");
for k in [
"后提醒我",
"后提醒",
"提醒我",
"提醒",
"定时",
"闹钟",
"叫我",
"帮我",
"一下",
"倒计时",
] {
m = m.replace(k, "");
}
m.trim()
.trim_start_matches(['的', '，', ',', ' ', '在'])
.trim()
.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_durations() {
        assert_eq!(
            parse_duration("5分钟后提醒我喝水"),
            Some((300, "5分钟".to_string()))
        );
        assert_eq!(
            parse_duration("十秒后提醒我喝水"),
            Some((10, "十秒".to_string()))
        );
        assert_eq!(
            parse_duration("一分钟后提醒我测试取消"),
            Some((60, "一分钟".to_string()))
        );
        assert_eq!(
            parse_duration("两小时后开会"),
            Some((7200, "两小时".to_string()))
        );
        assert_eq!(parse_duration("半小时后吃饭"), Some((1800, "半小时".to_string())));
        assert_eq!(parse_duration("取消我所有的定时器"), None);
        assert_eq!(parse_duration("你好"), None);
    }

    #[test]
    fn extract_messages() {
        assert_eq!(
            extract_reminder_message("5分钟后提醒我喝水", "5分钟"),
            "喝水"
        );
        assert_eq!(
            extract_reminder_message("十秒后提醒我", "十秒"),
            String::new()
        );
    }

    #[test]
    fn keyword_timer_intents() {
        let i = keyword_intent("5分钟后提醒我喝水");
        assert_eq!(i.tool, "set_timer");
        assert_eq!(i.seconds, Some(300));
        assert_eq!(i.message.as_deref(), Some("喝水"));

        let i = keyword_intent("取消我所有的定时器");
        assert_eq!(i.tool, "cancel_timer");
        assert_eq!(i.id.as_deref(), Some("all"));

        let i = keyword_intent("我设了哪些提醒");
        assert_eq!(i.tool, "list_timers");

        // 取消 must not hijack a set-timer request
        let i = keyword_intent("一分钟后提醒我测试取消");
        assert_eq!(i.tool, "set_timer");
        assert_eq!(i.seconds, Some(60));
    }
}
