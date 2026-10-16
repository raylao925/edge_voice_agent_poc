use anyhow::{Context, Result};
use async_trait::async_trait;
use sherpa_onnx::{
    GenerationConfig, OfflineTts, OfflineTtsConfig, OfflineTtsModelConfig,
    OfflineTtsVitsModelConfig,
};
use std::path::Path;
use std::sync::{Arc, Mutex};

pub struct TtsResult {
    pub wav: Vec<u8>,
    pub out_sec: f64,
}

/// Text-to-speech engine, shared across turns (`Arc<dyn TtsEngine>`).
#[async_trait]
pub trait TtsEngine: Send + Sync {
    async fn synth(&self, text: &str) -> Result<TtsResult>;
}

/// Piper VITS TTS via the official `sherpa-onnx` Rust crate — in-process,
/// no subprocess. Mirrors the Python POC's `init_tts`/`synth_wav`
/// (same model dir, `lexicon.txt`, 2 threads, cpu provider).
///
/// `OfflineTts::generate_*` is blocking, so synthesis runs on Tokio's
/// blocking thread pool to keep the async runtime responsive.
pub struct SherpaPiperTts {
    tts: Arc<Mutex<OfflineTts>>,
}

impl SherpaPiperTts {
    pub fn new(model_dir: &Path, num_threads: i32) -> Result<Self> {
        for f in ["zh_CN-chaowen-medium.onnx", "tokens.txt", "lexicon.txt"] {
            let p = model_dir.join(f);
            anyhow::ensure!(p.is_file(), "missing TTS file: {}", p.display());
        }
        let s = |p: std::path::PathBuf| p.to_string_lossy().into_owned();
        let config = OfflineTtsConfig {
            model: OfflineTtsModelConfig {
                vits: OfflineTtsVitsModelConfig {
                    model: Some(s(model_dir.join("zh_CN-chaowen-medium.onnx"))),
                    tokens: Some(s(model_dir.join("tokens.txt"))),
                    lexicon: Some(s(model_dir.join("lexicon.txt"))),
                    ..Default::default()
                },
                num_threads,
                debug: false,
                provider: Some("cpu".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let tts = OfflineTts::create(&config).context("create OfflineTts")?;
        tracing::info!("[tts] ready, sample_rate={}", tts.sample_rate());

        // warm up, like the Python POC
        let gen_config = GenerationConfig {
            sid: 0,
            speed: 1.0,
            ..Default::default()
        };
        tts.generate_with_config("你好", &gen_config, Some(|_: &[f32], _: f32| true))
            .context("tts warmup")?;

        Ok(Self {
            tts: Arc::new(Mutex::new(tts)),
        })
    }
}

#[async_trait]
impl TtsEngine for SherpaPiperTts {
    async fn synth(&self, text: &str) -> Result<TtsResult> {
        let tts = self.tts.clone();
        let text = text.to_owned();
        tokio::task::spawn_blocking(move || {
            let tts = tts.lock().map_err(|_| anyhow::anyhow!("tts mutex poisoned"))?;
            let gen_config = GenerationConfig {
                sid: 0,
                speed: 1.0,
                ..Default::default()
            };
            let audio = tts
                .generate_with_config(&text, &gen_config, Some(|_: &[f32], _: f32| true))
                .context("tts generate")?;
            let sample_rate = audio.sample_rate();
            let samples = audio.samples();
            let wav = encode_wav_mono16(samples, sample_rate);
            let out_sec = samples.len() as f64 / sample_rate as f64;
            Ok(TtsResult { wav, out_sec })
        })
        .await?
    }
}

/// Encode f32 mono samples as a 16-bit PCM WAV (mirrors `synth_wav`).
fn encode_wav_mono16(samples: &[f32], sample_rate: i32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&(sample_rate as u32).to_le_bytes());
    wav.extend_from_slice(&((sample_rate * 2) as u32).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        wav.extend_from_slice(&v.to_le_bytes());
    }
    wav
}
