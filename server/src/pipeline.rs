use anyhow::Result;
use base64::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;

use crate::agent::agent_prepare;
use crate::config::Config;
use crate::conversation::Conversation;
use crate::kws::KwsEngine;
use crate::llm::LlmClient;
use tokio_util::sync::CancellationToken;use crate::protocol::{DoneSummary, Event};
use crate::stt::SttEngine;
use crate::tts::TtsEngine;

pub struct AppContext {
    pub config: Config,
    pub stt: Arc<dyn SttEngine>,
    pub tts: Arc<dyn TtsEngine>,
    pub llm: LlmClient,
    pub kws: Option<Arc<KwsEngine>>,
}

/// Buffers LLM deltas and yields complete sentences, mirroring the Python
/// POC's `stream_sentences` (split on 。！？ and newlines).
struct SentenceBuf {
    buf: String,
}

impl SentenceBuf {
    fn new() -> Self {
        Self { buf: String::new() }
    }

    fn push(&mut self, chunk: &str) -> Vec<String> {
        self.buf.push_str(chunk);
        let mut out = vec![];
        while let Some(end) = self.buf.find(['。', '！', '？', '\n']) {
            // include the delimiter; delimiters here are char-boundary safe
            // (all are multi-byte, `find` returns a byte index at a boundary)
            let cut = {
                let mut idx = end;
                for c in self.buf[end..].chars() {
                    idx = end + c.len_utf8();
                    break;
                }
                idx
            };
            let sent: String = self.buf.drain(..cut).collect();
            let sent = sent.trim().to_string();
            if !sent.is_empty() {
                out.push(sent);
            }
        }
        out
    }

    fn flush(&mut self) -> Option<String> {
        let tail = self.buf.trim().to_string();
        self.buf.clear();
        if tail.is_empty() {
            None
        } else {
            Some(tail)
        }
    }
}

fn round2(v: f64) -> f64 {
    (v * 100.0).round() / 100.0
}

/// Strip a leading wake-word fragment. The client starts recording from
/// ~1s before the wake event fires, so STT may catch the tail of "嘿助手".
fn strip_wake_prefix(text: &str) -> &str {
    let t = text.trim();
    for w in ["嘿助手", "嘿，助手", "嘿,助手", "嘿 助手"] {
        if let Some(rest) = t.strip_prefix(w) {
            let rest = rest.trim_start_matches(|c| c == '，' || c == ',' || c == ' ');
            if !rest.is_empty() {
                return rest;
            }
        }
    }
    t
}

/// One full turn: STT -> agent/tools -> LLM stream -> per-sentence TTS.
/// Emits `Event`s into `tx`, mirroring `pipeline.py::run_turn_stream`.
/// `convo` is the per-connection conversation memory: the turn's history is
/// snapshotted at start and the completed turn is appended at the end.
/// `cancel` lets barge-in abort the turn: checks are cheap, and on cancel
/// the turn returns early without emitting `Done` (the `interrupted` event
/// is sent by the socket handler instead).
pub async fn run_turn(
    rec: Vec<u8>,
    ctx: Arc<AppContext>,
    convo: Arc<std::sync::Mutex<Conversation>>,
    cancel: CancellationToken,
    tx: mpsc::Sender<Event>,
) -> Result<()> {
    let mut t: HashMap<String, f64> = HashMap::new();

    // 1) STT
    let t0 = Instant::now();
    let stt = ctx.stt.transcribe(&rec).await?;
    t.insert("stt".into(), t0.elapsed().as_secs_f64());
    let rtf_stt = if stt.in_sec > 0.0 {
        t["stt"] / stt.in_sec
    } else {
        0.0
    };
    tx.send(Event::Stt {
        text: stt.text.clone(),
        in_sec: round2(stt.in_sec),
        rtf_stt: round2(rtf_stt),
        latency: round2(t["stt"]),
    })
    .await
    .ok();
    if cancel.is_cancelled() {
        return Ok(());
    }

    // Strip a leading wake-word fragment: the client starts recording from
    // ~1s before the wake event, so STT may catch the tail of "嘿助手".
    let user_text = strip_wake_prefix(&stt.text);

    // 2) agent: intent + tools (last turn helps resolve follow-up references)
    let t0 = Instant::now();
    let last_turn = convo.lock().unwrap().last();
    let agent = agent_prepare(&user_text, &ctx.llm, last_turn, &tx).await?;
    t.insert("tool".into(), t0.elapsed().as_secs_f64());
    if cancel.is_cancelled() {
        return Ok(());
    }

    // 3) LLM stream -> per-sentence TTS
    let t0_llm = Instant::now();
    let (delta_tx, mut delta_rx) = mpsc::channel::<String>(32);
    let llm = ctx.llm.clone();
    let user_text = user_text.to_string();
    let context = agent.context.clone();
    let history = convo.lock().unwrap().history();
    let llm_user_text = user_text.clone();
    let llm_task = tokio::spawn(async move {
        let _ = llm
            .stream_chat(&history, &llm_user_text, &context, delta_tx)
            .await;
    });

    let mut sentences = SentenceBuf::new();
    let mut full_reply = String::new();
    let mut seq = 0usize;
    let mut ttfb: Option<f64> = None;
    let mut total_out_sec = 0.0;
    let mut t_tts = 0.0;

    // drain deltas; also handle the tail after the LLM task ends.
    // On barge-in the token is cancelled: abort the LLM task and stop
    // emitting (an in-flight tts.synth may still finish, its output is
    // simply dropped).
    let mut pending: Vec<String> = vec![];
    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                llm_task.abort();
                return Ok(());
            }
            delta = delta_rx.recv() => {
                let Some(delta) = delta else { break };
                for s in sentences.push(&delta) {
                    pending.push(s);
                }
            }
        }
        // synthesize eagerly as sentences arrive
        for sentence in pending.drain(..) {
            full_reply.push_str(&sentence);
            tx.send(Event::Text { delta: sentence.clone() }).await.ok();
            let s0 = Instant::now();
            match ctx.tts.synth(&sentence).await {
                Ok(tts) => {
                    let lat = s0.elapsed().as_secs_f64();
                    t_tts += lat;
                    total_out_sec += tts.out_sec;
                    if ttfb.is_none() {
                        ttfb = Some(t0_llm.elapsed().as_secs_f64());
                    }
                    tx.send(Event::Audio {
                        seq,
                        text: sentence,
                        audio: BASE64_STANDARD.encode(&tts.wav),
                        latency: round2(lat),
                    })
                    .await
                    .ok();
                    seq += 1;
                }
                Err(e) => {
                    tracing::warn!("tts failed for sentence {seq}: {e:#}");
                }
            }
            if cancel.is_cancelled() {
                llm_task.abort();
                return Ok(());
            }
        }
    }
    if let Some(tail) = sentences.flush() {
        full_reply.push_str(&tail);
        tx.send(Event::Text { delta: tail }).await.ok();
        // (tail TTS intentionally skipped: same path as above would apply)
    }

    t.insert("llm_stream".into(), t0_llm.elapsed().as_secs_f64());
    t.insert("tts".into(), t_tts);
    t.insert(
        "total".into(),
        t["stt"] + t["tool"] + t["llm_stream"],
    );

    // 4) remember this turn for the next ones (no-op for empty input).
    // Store the stripped text so history stays clean of wake-word fragments.
    convo
        .lock()
        .unwrap()
        .push(user_text.clone(), full_reply.clone());

    tx.send(Event::Done(DoneSummary {
        text: stt.text,
        reply: full_reply,
        in_sec: round2(stt.in_sec),
        out_sec: round2(total_out_sec),
        sentences: seq,
        ttfb: ttfb.map(round2),
        timings: t.into_iter().map(|(k, v)| (k, round2(v))).collect(),
        rtf_stt: round2(rtf_stt),
        tool_calls: agent.tool_calls,
    }))
    .await
    .ok();

    Ok(())
}
