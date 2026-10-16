use anyhow::Result;
use async_trait::async_trait;

pub struct TtsResult {
    pub wav: Vec<u8>,
    pub out_sec: f64,
}

/// Text-to-speech engine, shared across turns (`Arc<dyn TtsEngine>`).
#[async_trait]
pub trait TtsEngine: Send + Sync {
    async fn synth(&self, text: &str) -> Result<TtsResult>;
}

/// Placeholder until the sherpa-rs integration lands (see `sherpa-rs` crate,
/// `tts` feature flag: in-process Piper synthesis, no subprocess needed).
/// Every other part of the pipeline already works against this trait.
pub struct UnimplementedTts;

#[async_trait]
impl TtsEngine for UnimplementedTts {
    async fn synth(&self, _text: &str) -> Result<TtsResult> {
        anyhow::bail!("TTS not wired up yet: next step is sherpa-rs (tts feature)")
    }
}
