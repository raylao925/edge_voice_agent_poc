#!/usr/bin/env python3
"""Shared voice pipeline: STT (FunASR Paraformer GGUF via llama.cpp binary)
-> LLM (Anthropic API or mock) -> TTS (sherpa-onnx Piper chaowen int8)."""
import io
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

import sherpa_onnx

BASE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(BASE, "bin", "llama-funasr-paraformer")
STT_MODEL = os.path.join(BASE, "models", "paraformer-q8.gguf")
TTS_DIR = os.path.join(BASE, "tts", "vits-piper-zh_CN-chaowen-medium-int8")

_tts = None


def init_tts():
    global _tts
    vits_cfg = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=os.path.join(TTS_DIR, "zh_CN-chaowen-medium.onnx"),
        tokens=os.path.join(TTS_DIR, "tokens.txt"),
        lexicon=os.path.join(TTS_DIR, "lexicon.txt"),
    )
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=vits_cfg, num_threads=2, debug=False, provider="cpu"),
    )
    _tts = sherpa_onnx.OfflineTts(cfg)
    _tts.generate("你好", sid=0, speed=1.0)  # warm up
    print("[pipeline] tts ready", flush=True)


def synth_wav(text: str):
    """Returns (wav_bytes, audio_seconds)."""
    audio = _tts.generate(text, sid=0, speed=1.0)
    pcm = (np.clip(np.asarray(audio.samples, dtype=np.float32), -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(audio.sample_rate)
        w.writeframes(pcm.tobytes())
    wav = buf.getvalue()
    return wav, len(pcm) / audio.sample_rate


def wav_to_opus(wav_bytes: bytes) -> bytes:
    with tempfile.TemporaryDirectory() as td:
        iw, ow = os.path.join(td, "in.wav"), os.path.join(td, "out.webm")
        with open(iw, "wb") as f:
            f.write(wav_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", iw,
             "-c:a", "libopus", "-b:a", "48k", ow],
            check=True, timeout=60)
        with open(ow, "rb") as f:
            return f.read()


def transcribe(rec_bytes: bytes):
    """rec_bytes: browser-recorded audio (webm/mp4/whatever ffmpeg reads).
    Returns (text, audio_seconds)."""
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "in.audio")
        wav = os.path.join(td, "in.wav")
        with open(src, "wb") as f:
            f.write(rec_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src,
             "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", wav],
            check=True, timeout=60)
        with wave.open(wav, "rb") as w:
            audio_sec = w.getnframes() / w.getframerate()
        r = subprocess.run([BIN, "-m", STT_MODEL, "-a", wav],
                           capture_output=True, text=True, timeout=180)
        out = (r.stdout or "").strip()
        if not out:
            print("[stt] empty stdout; stderr tail:", (r.stderr or "")[-400:], flush=True)
        return out, audio_sec


def _read_key_file(name: str) -> str:
    p = os.path.join(BASE, ".keys", name)
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return ""


def llm_reply(user_text: str):
    """Returns (reply_text, info_dict with model/latency/token info)."""
    import time
    import urllib.request
    system = "你是一个简洁的中文语音助手。用简体中文回答，控制在两句话以内，口语化，不要用列表或代码。"
    # 1) OpenRouter (OpenAI-compatible)
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or _read_key_file("openrouter.key")
    if or_key:
        model = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()
        payload = json.dumps({
            "model": model,
            "max_tokens": 200,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
        }).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions", data=payload,
            headers={"Authorization": "Bearer " + or_key,
                     "Content-Type": "application/json",
                     "HTTP-Referer": "https://muse.ai/",
                     "X-Title": "voice-agent-poc"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        latency = time.perf_counter() - t0
        info = {"backend": "openrouter", "model": model, "latency": latency}
        try:
            usage = data.get("usage", {})
            info["prompt_tokens"] = usage.get("prompt_tokens")
            info["completion_tokens"] = usage.get("completion_tokens")
            reply = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            reply = "抱歉，我刚才没想好怎么回答。"
        return reply, info
    # 2) Anthropic direct
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip() or _read_key_file("anthropic.key")
    if key:
        payload = json.dumps({
            "model": "claude-sonnet-4-6",
            "max_tokens": 200,
            "system": system,
            "messages": [{"role": "user", "content": user_text}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        latency = time.perf_counter() - t0
        info = {"backend": "anthropic", "model": "claude-sonnet-4-6", "latency": latency}
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        return ("".join(parts).strip() or "抱歉，我刚才没想好怎么回答。"), info
    # 3) mock
    return ("我听到了，你说的是「" + user_text + "」。大语言模型还没接上，这是测试回复。",
            {"backend": "mock", "model": "mock", "latency": 0.0})


def run_turn(rec_bytes: bytes) -> dict:
    """Full STT -> LLM -> TTS turn with per-stage timings and RTF.

    RTF (real-time factor) = processing_time / audio_duration.
    RTF < 1 means faster than realtime.
    """
    import time
    t = {}
    t0 = time.perf_counter()
    text, in_sec = transcribe(rec_bytes)
    t["stt"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reply, llm_info = llm_reply(text)
    t["llm"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    wav, out_sec = synth_wav(reply)
    t["tts"] = time.perf_counter() - t0

    t["total"] = t["stt"] + t["llm"] + t["tts"]
    return {
        "text": text,
        "reply": reply,
        "wav": wav,
        "in_sec": round(in_sec, 2),
        "out_sec": round(out_sec, 2),
        "timings": {k: round(v, 2) for k, v in t.items()},
        "rtf_stt": round(t["stt"] / in_sec, 3) if in_sec > 0 else 0,
        "rtf_tts": round(t["tts"] / out_sec, 3) if out_sec > 0 else 0,
        "llm": llm_info,
    }
