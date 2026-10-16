//! Server-side keyword spotting (wake word) via the official `sherpa-onnx`
//! Rust crate — same dependency as the TTS engine.
//!
//! The `KeywordSpotter` is shared across connections (`Arc`); each
//! connection that streams `audio_chunk`s gets its own `KwsStream`
//! holding the per-stream detection state.

use anyhow::{Context, Result};
use sherpa_onnx::{
    KeywordSpotter, KeywordSpotterConfig, OnlineModelConfig, OnlineStream,
    OnlineTransducerModelConfig,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct KwsEngine {
    spotter: KeywordSpotter,
}

impl KwsEngine {
    pub fn new(model_dir: &Path, num_threads: i32) -> Result<Self> {
        let dir = model_dir.to_string_lossy();
        // int8 encoder/decoder/joiner: ~5MB total, fast on CPU
        let config = KeywordSpotterConfig {
            model_config: OnlineModelConfig {
                transducer: OnlineTransducerModelConfig {
                    encoder: Some(format!(
                        "{dir}/encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx"
                    )),
                    decoder: Some(format!(
                        "{dir}/decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx"
                    )),
                    joiner: Some(format!(
                        "{dir}/joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx"
                    )),
                },
                tokens: Some(format!("{dir}/tokens.txt")),
                num_threads,
                ..Default::default()
            },
            keywords_buf: Some("h ēi zh ù sh ǒu @嘿助手\n".to_string()),
            keywords_threshold: 0.25,
            ..Default::default()
        };
        let spotter =
            KeywordSpotter::create(&config).context("failed to create KeywordSpotter")?;
        tracing::info!("[kws] ready (keyword: 嘿助手)");
        Ok(Self { spotter })
    }

    pub fn create_stream(self: &Arc<Self>) -> KwsStream {
        KwsStream {
            engine: self.clone(),
            stream: Mutex::new(self.spotter.create_stream()),
        }
    }
}

/// Per-connection detection state.
pub struct KwsStream {
    engine: Arc<KwsEngine>,
    stream: Mutex<OnlineStream>,
}

impl KwsStream {
    /// Feed 16kHz mono f32 samples. Returns the spotted keyword, if any.
    /// Call is cheap (a few ms per chunk); the stream is reset after a hit
    /// so one utterance doesn't trigger repeatedly.
    pub fn accept(&self, samples: &[f32]) -> Option<String> {
        let stream = self.stream.lock().unwrap();
        stream.accept_waveform(16000, samples);
        while self.engine.spotter.is_ready(&stream) {
            self.engine.spotter.decode(&stream);
        }
        let result = self.engine.spotter.get_result(&stream)?;
        if result.keyword.is_empty() {
            return None;
        }
        self.engine.spotter.reset(&stream);
        Some(result.keyword)
    }
}
