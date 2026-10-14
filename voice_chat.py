#!/usr/bin/env python3
"""In-chat voice loop: input voice note -> FunASR STT -> OpenRouter LLM -> Piper TTS -> mp3.
Usage: voice_chat.py <input-audio> <output-mp3>
Prints: STT text, reply text (for the chat message)."""
import os
import subprocess
import sys

from pipeline import init_tts, llm_reply, synth_wav, transcribe


def main():
    src, out_mp3 = sys.argv[1], sys.argv[2]
    init_tts()
    with open(src, "rb") as f:
        rec = f.read()
    print(f"[chat] got {len(rec)} bytes", flush=True)
    text = transcribe(rec)
    print(f"STT::{text}", flush=True)
    reply = llm_reply(text)
    print(f"REPLY::{reply}", flush=True)
    wav = synth_wav(reply)
    tmp = out_mp3 + ".tmp.wav"
    with open(tmp, "wb") as f:
        f.write(wav)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", tmp,
         "-c:a", "libmp3lame", "-b:a", "96k", out_mp3],
        check=True, timeout=60)
    os.remove(tmp)
    print(f"[chat] wrote {out_mp3}", flush=True)


if __name__ == "__main__":
    main()
