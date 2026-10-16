//! Per-connection timer/reminder registry.
//!
//! The agent's `set_timer` / `list_timers` / `cancel_timer` tools manipulate
//! the registry. A fired timer sends `TimerFired` and then speaks the
//! reminder through TTS, reusing the normal `Audio` event path so the
//! frontend just plays it.
//!
//! Timers are in-memory and scoped to one WebSocket connection: a server
//! restart (or disconnect) drops them. Documented MVP limitation.

use base64::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::protocol::Event;
use crate::tts::TtsEngine;

/// Longest timer the agent may set (7 days).
pub const MAX_TIMER_SECS: u64 = 7 * 24 * 3600;

struct TimerEntry {
    message: String,
    fire_at_unix: u64,
    handle: JoinHandle<()>,
}

#[derive(Clone)]
pub struct TimerRegistry {
    inner: Arc<Mutex<HashMap<String, TimerEntry>>>,
    /// Channel the ws loop forwards to the client (independent of any turn).
    tx: mpsc::Sender<Event>,
    tts: Arc<dyn TtsEngine>,
    next_id: Arc<Mutex<u64>>,
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl TimerRegistry {
    pub fn new(tx: mpsc::Sender<Event>, tts: Arc<dyn TtsEngine>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            tx,
            tts,
            next_id: Arc::new(Mutex::new(0)),
        }
    }

    /// Set a timer; returns `(id, fire_at_unix)`.
    pub fn set(&self, seconds_from_now: u64, message: String) -> (String, u64) {
        let id = {
            let mut n = self.next_id.lock().unwrap();
            *n += 1;
            format!("t{n}")
        };
        let fire_at = now_unix() + seconds_from_now;
        let this = self.clone();
        let id2 = id.clone();
        let msg2 = message.clone();
        // Insert before spawning so a racing cancel sees the entry.
        let handle = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(seconds_from_now)).await;
            this.fire(&id2, &msg2).await;
        });
        self.inner.lock().unwrap().insert(
            id.clone(),
            TimerEntry {
                message,
                fire_at_unix: fire_at,
                handle,
            },
        );
        tracing::info!("[timer] set {id} in {seconds_from_now}s");
        (id, fire_at)
    }

    /// (id, message, seconds_remaining), sorted by soonest first.
    pub fn list(&self) -> Vec<(String, String, u64)> {
        let now = now_unix();
        let mut v: Vec<_> = self
            .inner
            .lock()
            .unwrap()
            .iter()
            .map(|(id, e)| {
                (
                    id.clone(),
                    e.message.clone(),
                    e.fire_at_unix.saturating_sub(now),
                )
            })
            .collect();
        v.sort_by_key(|(_, _, remain)| *remain);
        v
    }

    /// Cancel one timer by id, or every timer with id `"all"`.
    /// Returns the number of timers cancelled.
    pub fn cancel(&self, id: &str) -> usize {
        let mut inner = self.inner.lock().unwrap();
        if id == "all" {
            let n = inner.len();
            for (_, e) in inner.drain() {
                e.handle.abort();
            }
            tracing::info!("[timer] cancelled all ({n})");
            return n;
        }
        match inner.remove(id) {
            Some(e) => {
                e.handle.abort();
                tracing::info!("[timer] cancelled {id}");
                1
            }
            None => 0,
        }
    }

    /// Abort every pending timer (called when the connection closes).
    pub fn shutdown(&self) {
        let mut inner = self.inner.lock().unwrap();
        for (_, e) in inner.drain() {
            e.handle.abort();
        }
    }

    async fn fire(&self, id: &str, message: &str) {
        // Remove first: a concurrent cancel wins the race, and then the
        // timer must stay silent.
        let existed = self.inner.lock().unwrap().remove(id).is_some();
        if !existed {
            return;
        }
        tracing::info!("[timer] fired {id}: {message}");
        self.tx
            .send(Event::TimerFired {
                id: id.to_string(),
                message: message.to_string(),
            })
            .await
            .ok();
        // Speak the reminder like a normal assistant utterance. No separate
        // Text event: the frontend shows the TimerFired bubble itself.
        match self.tts.synth(&format!("提醒一下：{message}")).await {
            Ok(tts) => {
                self.tx
                    .send(Event::Audio {
                        seq: 0,
                        text: message.to_string(),
                        audio: BASE64_STANDARD.encode(&tts.wav),
                        latency: 0.0,
                    })
                    .await
                    .ok();
            }
            Err(e) => tracing::warn!("[timer] tts failed: {e:#}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::TtsEngine;
    use anyhow::Result;

    /// TTS stub: instant empty wav.
    struct NullTts;
    #[async_trait::async_trait]
    impl TtsEngine for NullTts {
        async fn synth(&self, _text: &str) -> Result<crate::tts::TtsResult> {
            Ok(crate::tts::TtsResult {
                wav: vec![],
                out_sec: 0.0,
            })
        }
    }

    fn registry() -> (TimerRegistry, mpsc::Receiver<Event>) {
        let (tx, rx) = mpsc::channel(32);
        (TimerRegistry::new(tx, Arc::new(NullTts)), rx)
    }

    #[tokio::test]
    async fn timer_fires() {
        let (reg, mut rx) = registry();
        let (id, _) = reg.set(1, "喝水".to_string());
        // TimerFired arrives after ~1s.
        let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv())
            .await
            .expect("timed out")
            .expect("channel closed");
        assert!(
            matches!(ev, Event::TimerFired { id: ref got, .. } if got == &id),
            "expected TimerFired, got {ev:?}"
        );
        assert!(reg.list().is_empty(), "fired timer should be gone");
    }

    #[tokio::test]
    async fn timer_cancel() {
        let (reg, mut rx) = registry();
        let (id, _) = reg.set(60, "不應響".to_string());
        assert_eq!(reg.cancel(&id), 1);
        assert_eq!(reg.cancel(&id), 0, "double cancel");
        // Nothing should arrive within 2s.
        let res = tokio::time::timeout(Duration::from_secs(2), rx.recv()).await;
        assert!(res.is_err(), "cancelled timer fired!");
    }

    #[tokio::test]
    async fn timer_cancel_all() {
        let (reg, _rx) = registry();
        reg.set(60, "a".to_string());
        reg.set(60, "b".to_string());
        assert_eq!(reg.cancel("all"), 2);
        assert!(reg.list().is_empty());
    }
}
