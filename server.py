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

from pipeline import init_tts, run_turn_stream

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

                loop = asyncio.get_running_loop()
                q: asyncio.Queue = asyncio.Queue()

                def on_event(ev):
                    # called from worker thread
                    loop.call_soon_threadsafe(q.put_nowait, ev)

                async def _worker():
                    await asyncio.to_thread(run_turn_stream, rec, on_event)
                    await q.put({"type": "__end__"})

                wt = asyncio.create_task(_worker())
                while True:
                    ev = await q.get()
                    if ev.get("type") == "__end__":
                        break
                    if ev.get("type") == "done":
                        print(f"[done] ttfb={ev.get('ttfb')}s "
                              f"sentences={ev.get('sentences')} "
                              f"total={ev.get('timings', {}).get('total')}s", flush=True)
                    await ws.send(json.dumps(ev))
                await wt
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
