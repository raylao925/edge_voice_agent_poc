use std::path::PathBuf;

/// Server configuration. Values come from the environment, with repo-relative
/// defaults so the existing `bin/`, `models/`, `www/` layout keeps working.
#[derive(Debug, Clone)]
pub struct Config {
    pub port: u16,
    pub www_dir: PathBuf,
    pub openrouter_key: String,
    pub openrouter_model: String,
    pub stt_bin: PathBuf,
    pub stt_model: PathBuf,
    pub tts_model_dir: PathBuf,
    pub kws_model_dir: PathBuf,
}

impl Config {
    pub fn load() -> Self {
        // Repo root = parent of the `server/` crate directory.
        let root: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("server/ must live inside the repo")
            .to_path_buf();

        let port: u16 = std::env::var("VOICE_PORT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8765);

        let openrouter_key = std::env::var("OPENROUTER_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                std::fs::read_to_string(root.join(".keys").join("openrouter.key"))
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_default();

        let openrouter_model = std::env::var("OPENROUTER_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "openrouter/free".to_string());

        Self {
            port,
            www_dir: root.join("www"),
            openrouter_key,
            openrouter_model,
            stt_bin: root.join("bin").join("llama-funasr-paraformer"),
            stt_model: root.join("models").join("paraformer-q8.gguf"),
            tts_model_dir: root
                .join("tts")
                .join("vits-piper-zh_CN-chaowen-medium-int8"),
            kws_model_dir: root
                .join("kws")
                .join("sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"),
        }
    }
}
