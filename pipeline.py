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


def llm_reply(user_text: str, context: str = ""):
    """Returns (reply_text, info_dict with model/latency/token info).
    If context (tool results) is given, it is injected for grounding."""
    import time
    import urllib.request
    system = "你是一个简洁的中文语音助手。用简体中文回答，控制在两句话以内，口语化，不要用列表或代码。"
    if context:
        system += f"\n以下是实时工具返回的参考资料，请基于它们回答，不要编造：\n{context[:3000]}"
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


# ---------------------------------------------------------------- agent loop
_INTENT_PROMPT = """判断用户这句话是否需要调用实时工具。只返回 JSON，不要其他文字。
可选工具：
- {"tool": "web_search", "query": "搜索关键词"}  // 需要上网查资料、事实、解释
- {"tool": "news_search", "query": "新闻关键词"}  // 问最新新闻、时事
- {"tool": "get_weather", "city": "城市名"}       // 问天气
- {"tool": "none"}                               // 闲聊、打招呼、不需要实时信息

用户：{text}
JSON:"""

_TOOL_LABEL = {
    "web_search": "🔍 正在搜索",
    "news_search": "📰 正在查新闻",
    "get_weather": "🌤 正在查天气",
    "fetch_page": "📄 正在读网页",
}


def _llm_json(prompt: str):
    """Call the LLM for a JSON intent decision. Returns dict or None."""
    import urllib.request
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or _read_key_file("openrouter.key")
    if not or_key:
        return None
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()
    payload = json.dumps({
        "model": model,
        "max_tokens": 120,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + or_key,
                 "Content-Type": "application/json",
                 "HTTP-Referer": "https://muse.ai/",
                 "X-Title": "voice-agent-poc"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        raw = data["choices"][0]["message"]["content"].strip()
        # tolerate markdown fences
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        print("[intent] llm failed:", repr(e)[:120], flush=True)
        return None


def _keyword_intent(text: str):
    """Fallback intent when no LLM key: simple keyword matching."""
    if any(k in text for k in ("天气", "下雨", "气温", "几度")):
        city = "澳门"
        for c in ("澳门", "香港", "台湾", "北京", "上海", "广州", "深圳", "珠海"):
            if c in text:
                city = c
                break
        return {"tool": "get_weather", "city": city}
    if any(k in text for k in ("新闻", "头条", "时事")):
        q = text
        for k in ("新闻", "头条", "时事", "最新", "今天", "的", "一下", "查", "搜"):
            q = q.replace(k, "")
        return {"tool": "news_search", "query": q.strip() or "今日新闻"}
    if any(k in text for k in ("搜索", "搜一下", "查一下", "什么是", "介绍一下")):
        return {"tool": "web_search", "query": text}
    return {"tool": "none"}


def detect_intent(user_text: str) -> dict:
    intent = _llm_json(_INTENT_PROMPT.replace("{text}", user_text))
    if isinstance(intent, dict) and intent.get("tool") in (
            "web_search", "news_search", "get_weather", "fetch_page", "none"):
        return intent
    return _keyword_intent(user_text)


def agent_prepare(user_text: str, status_cb=None):
    """Intent detection + tool execution. Returns (context, tool_calls, latency)."""
    import time
    from tools import TOOLS

    tool_calls = []
    context = ""
    t_tool = 0.0

    intent = detect_intent(user_text)
    tool = intent.get("tool", "none")
    if tool != "none" and tool in TOOLS:
        label = _TOOL_LABEL.get(tool, "⚙️")
        arg = intent.get("query") or intent.get("city") or intent.get("url") or ""
        if status_cb:
            status_cb(f"{label}：{arg}")
        t0 = time.perf_counter()
        try:
            if tool == "get_weather":
                result = TOOLS[tool](intent.get("city", "澳门"))
            elif tool == "fetch_page":
                result = TOOLS[tool](intent.get("url", ""))
            else:
                result = TOOLS[tool](intent.get("query", user_text))
            t_tool = time.perf_counter() - t0
            tool_calls.append({"tool": tool, "arg": arg,
                               "latency": round(t_tool, 2)})
            context = f"【{tool} 结果】\n{result}"
            print(f"[tool] {tool} done in {t_tool:.1f}s", flush=True)
        except Exception as e:
            t_tool = time.perf_counter() - t0
            print(f"[tool] {tool} failed: {repr(e)[:150]}", flush=True)
            context = f"【{tool} 执行失败】{str(e)[:100]}"
            if status_cb:
                status_cb(f"⚠️ 工具执行失败")
    return context, tool_calls, t_tool


def agent_reply(user_text: str, status_cb=None):
    """Two-stage agent: intent -> tool -> grounded answer.
    Returns (reply_text, info_dict). status_cb(str) gets progress updates."""
    import time
    context, tool_calls, t_tool = agent_prepare(user_text, status_cb=status_cb)
    t0 = time.perf_counter()
    reply, llm_info = llm_reply(user_text, context=context)
    t_llm = time.perf_counter() - t0
    llm_info["tool_calls"] = tool_calls
    llm_info["tool_latency"] = round(t_tool, 2)
    return reply, llm_info


# ---------------------------------------------------------------- streaming
def llm_stream(user_text: str, context: str = ""):
    """Yield LLM reply text chunks (SSE). Mock mode yields sentence chunks."""
    import urllib.request
    system = "你是一个简洁的中文语音助手。用简体中文回答，控制在两句话以内，口语化，不要用列表或代码。"
    if context:
        system += f"\n以下是实时工具返回的参考资料，请基于它们回答，不要编造：\n{context[:3000]}"
    or_key = os.environ.get("OPENROUTER_API_KEY", "").strip() or _read_key_file("openrouter.key")
    if not or_key:
        mock = "我听到了，你说的是「" + user_text + "」。大语言模型还没接上，这是测试回复。"
        for sent in _split_sentences(mock):
            yield sent
        return
    model = os.environ.get("OPENROUTER_MODEL", "openrouter/free").strip()
    payload = json.dumps({
        "model": model,
        "max_tokens": 200,
        "stream": True,
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
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0]["delta"].get("content", "")
            except Exception:
                continue
            if delta:
                yield delta


def _split_sentences(text: str):
    """Split Chinese text into sentences on 。！？\n (keeps delimiter)."""
    import re
    parts = re.split(r"([。！？\n]+)", text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        s = (parts[i] + parts[i + 1]).strip()
        if s:
            sentences.append(s)
    tail = parts[-1].strip() if len(parts) % 2 == 1 else ""
    if tail:
        sentences.append(tail)
    return sentences


def stream_sentences(text_iter):
    """Assemble streamed text chunks into complete sentences."""
    import re
    buf = ""
    for chunk in text_iter:
        buf += chunk
        while True:
            m = re.search(r"^.*?[。！？\n]", buf, re.DOTALL)
            if not m:
                break
            sent = m.group(0).strip()
            buf = buf[m.end():]
            if sent:
                yield sent
    tail = buf.strip()
    if tail:
        yield tail


def run_turn_stream(rec_bytes: bytes, on_event) -> dict:
    """Streaming pipeline: STT -> agent tools -> LLM stream -> per-sentence TTS.

    Calls on_event(dict) for each event:
      {"type": "stt", ...}, {"type": "status", "text": ...},
      {"type": "text", "delta": ...}, {"type": "audio", "seq": N, ...},
      {"type": "done", ...}
    Returns summary dict.
    """
    import base64
    import time
    t = {}

    # 1) STT
    t0 = time.perf_counter()
    text, in_sec = transcribe(rec_bytes)
    t["stt"] = time.perf_counter() - t0
    rtf_stt = round(t["stt"] / in_sec, 3) if in_sec > 0 else 0
    on_event({"type": "stt", "text": text,
              "in_sec": round(in_sec, 2), "rtf_stt": rtf_stt,
              "latency": round(t["stt"], 2)})

    # 2) agent: intent + tools
    def _status(s):
        on_event({"type": "status", "text": s})
    t0 = time.perf_counter()
    context, tool_calls, t_tool = agent_prepare(text, status_cb=_status)
    t["tool"] = time.perf_counter() - t0

    # 3) LLM stream -> per-sentence TTS
    t0_llm = time.perf_counter()
    full_reply = ""
    seq = 0
    ttfb = None  # time to first audio
    total_out_sec = 0.0
    t_tts = 0.0
    for sentence in stream_sentences(llm_stream(text, context=context)):
        full_reply += sentence
        on_event({"type": "text", "delta": sentence})
        s0 = time.perf_counter()
        wav, out_sec = synth_wav(sentence)
        t_tts += time.perf_counter() - s0
        total_out_sec += out_sec
        if ttfb is None:
            ttfb = round(time.perf_counter() - t0_llm, 2)
        on_event({"type": "audio", "seq": seq, "text": sentence,
                  "audio": base64.b64encode(wav).decode(),
                  "latency": round(time.perf_counter() - s0, 2)})
        seq += 1
    t["llm_stream"] = round(time.perf_counter() - t0_llm, 2)
    t["tts"] = round(t_tts, 2)
    t["total"] = round(t["stt"] + t["tool"] + t["llm_stream"], 2)

    summary = {
        "text": text,
        "reply": full_reply,
        "in_sec": round(in_sec, 2),
        "out_sec": round(total_out_sec, 2),
        "sentences": seq,
        "ttfb": ttfb,
        "timings": {k: round(v, 2) for k, v in t.items()},
        "rtf_stt": rtf_stt,
        "tool_calls": tool_calls,
    }
    on_event({"type": "done", **summary})
    return summary


def run_turn(rec_bytes: bytes, status_cb=None) -> dict:
    """Full STT -> agent (LLM + tools) -> TTS turn with timings and RTF.

    RTF (real-time factor) = processing_time / audio_duration.
    RTF < 1 means faster than realtime.
    """
    import time
    t = {}
    t0 = time.perf_counter()
    text, in_sec = transcribe(rec_bytes)
    t["stt"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    reply, llm_info = agent_reply(text, status_cb=status_cb)
    t["llm"] = time.perf_counter() - t0
    t["tool"] = llm_info.get("tool_latency", 0.0)

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
