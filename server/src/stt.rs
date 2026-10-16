use anyhow::{Context, Result};
use async_trait::async_trait;
use sherpa_onnx::{
    OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSenseVoiceModelConfig,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::process::Command;

pub struct SttResult {
    pub text: String,
    pub in_sec: f64,
}

/// Speech-to-text engine. Implementations must be cheap to share across
/// concurrent turns (`Arc<dyn SttEngine>`).
///
/// `language`: BCP-ish hint for the utterance — "auto" (default), "zh",
/// "yue", "en". Engines that can't use it (FunASR/Paraformer) ignore it.
#[async_trait]
pub trait SttEngine: Send + Sync {
    async fn transcribe(&self, rec_bytes: &[u8], language: &str) -> Result<SttResult>;
}

/// Normalize recorded audio (webm/opus/wav from the client) to 16kHz mono
/// f32 samples. Returns (samples, duration_seconds).
async fn rec_to_mono16(rec_bytes: &[u8]) -> Result<(Vec<f32>, f64)> {
    let dir = tempfile::tempdir().context("create tempdir")?;
    let src = dir.path().join("in.audio");
    let wav = dir.path().join("in.wav");
    tokio::fs::write(&src, rec_bytes).await?;

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

    // We forced the format above: data starts at byte 44, 32000 bytes/sec.
    let wav_bytes = tokio::fs::read(&wav).await?;
    anyhow::ensure!(wav_bytes.len() > 44 && &wav_bytes[0..4] == b"RIFF", "bad wav");
    let in_sec = (wav_bytes.len() - 44) as f64 / 32000.0;
    let samples: Vec<f32> = wav_bytes[44..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    Ok((samples, in_sec))
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
    async fn transcribe(&self, rec_bytes: &[u8], _language: &str) -> Result<SttResult> {
        // Paraformer is Mandarin-only; the language hint is ignored.
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

/// SenseVoice-Small (zh/en/ja/ko/yue) via the official `sherpa-onnx` crate —
/// in-process, no subprocess. `decode` is blocking, so it runs on Tokio's
/// blocking thread pool (same pattern as the Piper TTS).
///
/// One recognizer per language is cached (`"auto"` = language ID); the
/// language is a model-level config in sherpa-onnx, not per-stream.
pub struct SenseVoiceStt {
    model_dir: PathBuf,
    num_threads: i32,
    recognizers: Mutex<HashMap<String, Arc<Mutex<OfflineRecognizer>>>>,
}

impl SenseVoiceStt {
    pub fn new(model_dir: PathBuf, num_threads: i32) -> Result<Self> {
        for f in ["model.int8.onnx", "tokens.txt"] {
            let p = model_dir.join(f);
            anyhow::ensure!(p.is_file(), "missing SenseVoice file: {}", p.display());
        }
        Ok(Self {
            model_dir,
            num_threads,
            recognizers: Mutex::new(HashMap::new()),
        })
    }

    fn create_recognizer(&self, language: &str) -> Result<OfflineRecognizer> {
        let s = |p: PathBuf| p.to_string_lossy().into_owned();
        let mut config = OfflineRecognizerConfig::default();
        config.model_config = OfflineModelConfig {
            sense_voice: OfflineSenseVoiceModelConfig {
                model: Some(s(self.model_dir.join("model.int8.onnx"))),
                language: Some(language.to_string()),
                use_itn: true,
            },
            tokens: Some(s(self.model_dir.join("tokens.txt"))),
            num_threads: self.num_threads,
            provider: Some("cpu".to_string()),
            ..Default::default()
        };
        OfflineRecognizer::create(&config).context("create SenseVoice recognizer")
    }

    fn recognizer_for(&self, language: &str) -> Result<Arc<Mutex<OfflineRecognizer>>> {
        let lang = if language.trim().is_empty() || language == "auto" {
            "auto"
        } else {
            language.trim()
        };
        let mut map = self.recognizers.lock().unwrap();
        if let Some(r) = map.get(lang) {
            return Ok(r.clone());
        }
        let r = Arc::new(Mutex::new(self.create_recognizer(lang)?));
        tracing::info!("[stt] SenseVoice recognizer ready (language={lang})");
        map.insert(lang.to_string(), r.clone());
        Ok(r)
    }
}

#[async_trait]
impl SttEngine for SenseVoiceStt {
    async fn transcribe(&self, rec_bytes: &[u8], language: &str) -> Result<SttResult> {
        let (samples, in_sec) = rec_to_mono16(rec_bytes).await?;
        let rec = self.recognizer_for(language)?;
        let text = tokio::task::spawn_blocking(move || {
            let rec = rec.lock().map_err(|_| anyhow::anyhow!("stt mutex poisoned"))?;
            let stream = rec.create_stream();
            stream.accept_waveform(16000, &samples);
            rec.decode(&stream);
            let result = stream.get_result().context("no recognition result")?;
            Ok::<String, anyhow::Error>(result.text.trim().to_string())
        })
        .await??;
        Ok(SttResult { text, in_sec })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<PathBuf> {
        let d = PathBuf::from(concat!(
            env!("HOME"),
            "/workspace/voice-agent/models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
        ));
        if d.join("model.int8.onnx").is_file() {
            Some(d)
        } else {
            None
        }
    }

    /// SenseVoice smoke: needs the model dir; skipped otherwise.
    /// Uses the tarball's own test wavs (zh/yue/en, all 16kHz).
    #[tokio::test]
    async fn sense_voice_smoke() {
        let Some(dir) = model_dir() else {
            eprintln!("SKIP: SenseVoice model not present");
            return;
        };
        let stt = SenseVoiceStt::new(dir.clone(), 2).unwrap();
        let cases = [
            // (wav, lang, must_contain)
            ("zh.wav", "auto", "开放时间"),
            ("yue.wav", "auto", "唔到"),
            ("yue.wav", "yue", "嘅"),
            ("en.wav", "en", "chieftain"),
        ];
        for (name, lang, needle) in cases {
            let bytes = std::fs::read(dir.join("test_wavs").join(name)).unwrap();
            let t0 = std::time::Instant::now();
            let r = stt.transcribe(&bytes, lang).await.unwrap();
            let rtf = t0.elapsed().as_secs_f64() / r.in_sec;
            println!("[{name} lang={lang}] {:?} rtf={rtf:.2}", r.text);
            assert!(
                r.text.contains(needle),
                "expected {needle:?} in {name} transcript"
            );
            assert!(rtf < 1.0, "STT RTF {rtf:.2} >= 1.0 for {name}");
        }
    }
}
