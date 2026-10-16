# Voice Agent MVP (Rust + Tauri)

> **Branch `mvp-rust`.** Full rewrite of the Python POC (`main` branch) in Rust.
> The POC stays untouched on `main` as the reference implementation.

## Architecture

```
┌──────────────┐   WebSocket    ┌───────────────────────────────────┐
│ Tauri client │ ◄────────────► │ Rust server (single binary, axum) │
│ (Rust, later)│  same protocol │                                   │
└──────────────┘   as POC       │  STT: FunASR Paraformer (subproc) │
┌──────────────┐                │  Agent: OpenRouter + tools        │
│ www/index.html│ ◄────────────► │  TTS: sherpa-rs Piper (next)      │
│ (works today)│                └───────────────────────────────────┘
└──────────────┘
```

The WebSocket protocol is wire-compatible with the Python POC, so the
existing browser frontend (`www/`) works unchanged against the Rust server.

## Layout

- `server/` — Rust server crate (`voice-agent-server`)
  - `src/main.rs` — axum HTTP + WebSocket server
  - `src/protocol.rs` — wire protocol types (serde)
  - `src/pipeline.rs` — turn orchestration: STT → agent → LLM SSE → per-sentence TTS
  - `src/stt.rs` — `SttEngine` trait + `FunAsrSubprocess` (real, calls the llama.cpp binary)
  - `src/tts.rs` — `TtsEngine` trait (stub; sherpa-rs integration is next)
  - `src/llm.rs` — OpenRouter SSE client (real)
  - `src/agent.rs`, `src/tools.rs` — agent loop + tool registry (stubs; port `tools.py` next)
  - `src/ws.rs` — WebSocket handler
  - `src/config.rs` — env/file config
- `client/` — Tauri native client (next: scaffold, reuse `www/`)
- `www/` — shared web frontend (unchanged from POC)
- `bin/`, `models/`, `tts/` — reused binaries/models (see `download_models.sh`)

## Build & run

```bash
cd server
cargo run
# serves on 127.0.0.1:8765 (VOICE_PORT to override)
# open http://127.0.0.1:8765 in a browser
```

Config via env (or `<repo>/.keys/openrouter.key`):

```bash
OPENROUTER_API_KEY=... OPENROUTER_MODEL=openrouter/free cargo run
```

## Roadmap

1. ✅ Project scaffold, wire protocol, STT subprocess, OpenRouter SSE client
2. ⬜ sherpa-rs TTS integration (`tts` feature) — replaces the stub
3. ⬜ Port `tools.py` (web/news search, weather, fetch_page) + two-stage agent loop
4. ⬜ Multi-turn conversation memory (biggest POC gap)
5. ⬜ Tauri client scaffold (tray + global hotkey), reuse `www/`
6. ⬜ Wake word (openWakeWord) / barge-in

## Reference

The Python POC on `main` remains the behavioral reference: same event
protocol, same timings semantics (`stt`/`status`/`text`/`audio`/`done`).
