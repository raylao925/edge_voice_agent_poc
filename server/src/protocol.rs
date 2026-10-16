use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Messages the browser/Tauri client sends us.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ClientMsg {
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "utterance")]
    Utterance {
        data: String, // base64-encoded recorded audio
        /// STT language hint: "auto" (default), "zh", "yue", "en".
        /// Optional for wire-compat with older clients.
        #[serde(default)]
        lang: Option<String>,
    },
    /// Clear conversation history and start fresh.
    #[serde(rename = "new_conversation")]
    NewConversation,
    /// Streaming 16kHz mono s16le PCM for wake-word detection (base64).
    /// Only fed to the keyword spotter when no turn is in flight.
    #[serde(rename = "audio_chunk")]
    AudioChunk { data: String, seq: u64 },
    /// Interrupt the in-flight turn (barge-in).
    #[serde(rename = "interrupt")]
    Interrupt,
}

/// Events we stream back. Wire-compatible with the Python POC (`server.py`),
/// so the existing `www/index.html` works unchanged.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    #[serde(rename = "pong")]
    Pong,
    #[serde(rename = "stt")]
    Stt {
        text: String,
        in_sec: f64,
        rtf_stt: f64,
        latency: f64,
    },
    #[serde(rename = "status")]
    Status { text: String },
    #[serde(rename = "text")]
    Text { delta: String },
    #[serde(rename = "audio")]
    Audio {
        seq: usize,
        text: String,
        audio: String, // base64-encoded wav chunk
        latency: f64,
    },
    #[serde(rename = "done")]
    Done(DoneSummary),
    #[serde(rename = "error")]
    Error { message: String },
    /// Wake word spotted in the audio_chunk stream.
    #[serde(rename = "wake")]
    Wake { keyword: String },
    /// The in-flight turn was cancelled by an interrupt.
    #[serde(rename = "interrupted")]
    Interrupted,
    /// A reminder timer fired. The reminder speech follows as normal
    /// `audio` events; the frontend shows the bubble itself.
    #[serde(rename = "timer_fired")]
    TimerFired { id: String, message: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct DoneSummary {
    pub text: String,
    pub reply: String,
    pub in_sec: f64,
    pub out_sec: f64,
    pub sentences: usize,
    pub ttfb: Option<f64>,
    pub timings: HashMap<String, f64>,
    pub rtf_stt: f64,
    pub tool_calls: Vec<ToolCallInfo>,
}

/// One executed tool call, shown in the frontend perf panel.
/// Wire-compatible with the Python POC: `{tool, arg, latency}`.
#[derive(Debug, Clone, Serialize)]
pub struct ToolCallInfo {
    pub tool: String,
    pub arg: String,
    pub latency: f64,
}
