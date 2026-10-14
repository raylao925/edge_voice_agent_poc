#!/usr/bin/env python3
"""Voice agent POC - browser demo server.

Pipeline: browser mic -> FunASR Paraformer GGUF (llama.cpp) -> LLM -> Piper TTS.
Single port: HTTP serves the demo page, WebSocket handles the voice loop.

Usage:
    python server.py              # serves on 127.0.0.1:8765
    VOICE_PORT=9000 python server.py
"""
import asyncio
import base64
import json
import os

import websockets

from pipeline import init_tts, run_turn

WWW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "www")
PORT = int(os.environ.get("VOICE_PORT", "8765"))


async def handle(ws):
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("type") == "ping":
            await ws.send(json.dumps({"type": "pong"}))
        elif msg.get("type") == "utterance":
            try:
                rec = base64.b64decode(msg["data"])
                print(f"[ws] utterance: {len(rec)} bytes", flush=True)
                res = await asyncio.to_thread(run_turn, rec)
                print(f"[stt] {res['text']} "
                      f"(rtf={res['rtf_stt']})", flush=True)
                print(f"[llm] {res['reply'][:60]}... "
                      f"({res['timings']['llm']}s)", flush=True)
                print(f"[tts] rtf={res['rtf_tts']} "
                      f"total={res['timings']['total']}s", flush=True)
                await ws.send(json.dumps({
                    "type": "result",
                    "stt": res["text"],
                    "reply": res["reply"],
                    "audio": base64.b64encode(res["wav"]).decode(),
                    "timings": res["timings"],
                    "in_sec": res["in_sec"],
                    "out_sec": res["out_sec"],
                    "rtf_stt": res["rtf_stt"],
                    "rtf_tts": res["rtf_tts"],
                    "llm_model": res["llm"].get("model"),
                }))
            except Exception as e:
                print("[err]", repr(e), flush=True)
                await ws.send(json.dumps({"type": "error", "message": str(e)[:200]}))


async def process_request(path, headers):
    if path in ("/", "/index.html"):
        with open(os.path.join(WWW, "index.html"), "rb") as f:
            body = f.read()
        return (200, [("Content-Type", "text/html; charset=utf-8")], body)
    return None


async def main():
    init_tts()
    async with websockets.serve(handle, "127.0.0.1", PORT,
                                process_request=process_request,
                                max_size=32 * 1024 * 1024):
        print(f"[server] open http://127.0.0.1:{PORT} in your browser", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
