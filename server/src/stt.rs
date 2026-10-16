use anyhow::{Context, Result};
use async_trait::async_trait;
use std::path::PathBuf;
use tokio::process::Command;

pub struct SttResult {
    pub text: String,
    pub in_sec: f64,
}

/// Speech-to-text engine. Implementations must be cheap to share across
/// concurrent turns (`Arc<dyn SttEngine>`).
#[async_trait]
pub trait SttEngine: Send + Sync {
    async fn transcribe(&self, rec_bytes: &[u8]) -> Result<SttResult>;
}

/// Calls the FunASR llama.cpp binary as a subprocess, exactly like the Python
/// POC did: ffmpeg -> 16k mono wav -> `llama-funasr-paraformer -m <model> -a <wav>`.
pub struct FunAsrSubprocess {
    bin: PathBuf,
    model: PathBuf,
}

impl FunAsrSubprocess {
    pub fn new(bin: PathBuf, model: PathBuf) -> Self {
        Self { bin, model }
    }
}

#[async_trait]
impl SttEngine for FunAsrSubprocess {
    async fn transcribe(&self, rec_bytes: &[u8]) -> Result<SttResult> {
        let dir = tempfile::tempdir().context("create tempdir")?;
        let src = dir.path().join("in.audio");
        let wav = dir.path().join("in.wav");
        tokio::fs::write(&src, rec_bytes).await?;

        // 1) normalize to 16kHz mono s16 wav
        let status = Command::new("ffmpeg")
            .args([
                "-y", "-v", "error", "-i",
                src.to_str().unwrap(),
                "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
                wav.to_str().unwrap(),
            ])
            .status()
            .await
            .context("run ffmpeg")?;
        anyhow::ensure!(status.success(), "ffmpeg failed");

        // 2) audio duration from the wav header (we forced the format above,
        //    so data starts at byte 44 and each second is 32000 bytes)
        let wav_bytes = tokio::fs::read(&wav).await?;
        anyhow::ensure!(wav_bytes.len() > 44 && &wav_bytes[0..4] == b"RIFF", "bad wav");
        let in_sec = (wav_bytes.len() - 44) as f64 / 32000.0;

        // 3) run the recognizer; transcription is stdout
        let out = Command::new(&self.bin)
            .args([
                "-m", self.model.to_str().unwrap(),
                "-a", wav.to_str().unwrap(),
            ])
            .output()
            .await
            .context("run funasr binary")?;
        let text = String::from_utf8_lossy(&out.stdout).trim().to_string();

        Ok(SttResult { text, in_sec })
    }
}
