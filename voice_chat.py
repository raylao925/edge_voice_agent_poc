#!/usr/bin/env python3
"""In-chat voice loop: input voice note -> FunASR STT -> OpenRouter LLM -> Piper TTS -> mp3.
Usage: voice_chat.py <input-audio> <output-mp3>
Prints: STT text, reply text, per-stage timings and RTF."""
import os
import subprocess
import sys

from pipeline import init_tts, run_turn


def main():
    src, out_mp3 = sys.argv[1], sys.argv[2]
    init_tts()
    with open(src, "rb") as f:
        rec = f.read()
    print(f"[chat] got {len(rec)} bytes", flush=True)
    res = run_turn(rec)
    print(f"STT::{res['text']}", flush=True)
    print(f"REPLY::{res['reply']}", flush=True)
    t = res["timings"]
    print(f"TIME::stt={t['stt']}s (rtf={res['rtf_stt']}) "
          f"llm={t['llm']}s tts={t['tts']}s (rtf={res['rtf_tts']}) "
          f"total={t['total']}s", flush=True)
    tmp = out_mp3 + ".tmp.wav"
    with open(tmp, "wb") as f:
        f.write(res["wav"])
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp,
         "-c:a", "libmp3lame", "-b:a", "96k", out_mp3],
        check=True, timeout=60)
    os.remove(tmp)
    print(f"[chat] wrote {out_mp3}", flush=True)


if __name__ == "__main__":
    main()
