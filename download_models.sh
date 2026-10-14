#!/usr/bin/env bash
# Download all binaries and models for the voice-agent POC.
#   STT: FunASR llama.cpp prebuilt binary + Paraformer GGUF (q8) + FSMN-VAD
#   TTS: sherpa-onnx Piper zh_CN-chaowen-medium-int8
set -euo pipefail
cd "$(dirname "$0")"

VER="runtime-llamacpp-v0.1.9"
GH="https://github.com/modelscope/FunASR/releases/download/${VER}"

OS="$(uname -s)"; ARCH="$(uname -m)"
case "${OS}-${ARCH}" in
  Linux-x86_64)
    if grep -q avx2 /proc/cpuinfo 2>/dev/null; then ASSET="funasr-llamacpp-linux-x64-avx2.tar.gz";
    else ASSET="funasr-llamacpp-linux-x64.tar.gz"; fi ;;
  Linux-aarch64|Linux-arm64) ASSET="funasr-llamacpp-linux-arm64.tar.gz" ;;
  Darwin-arm64)              ASSET="funasr-llamacpp-macos-arm64.tar.gz" ;;
  Darwin-x86_64)             echo "No macOS x64 binary; use the arm64 build under Rosetta." >&2; exit 1 ;;
  MINGW*|MSYS*|CYGWIN*)      echo "On Windows, download ${GH}/funasr-llamacpp-windows-x64-avx2.zip manually." >&2; exit 1 ;;
  *) echo "Unsupported platform: ${OS}-${ARCH}" >&2; exit 1 ;;
esac

mkdir -p bin models tts
echo "== STT binary: ${ASSET}"
curl -sL --retry 3 -o bin/funasr-llamacpp.tgz "${GH}/${ASSET}"
tar --no-same-owner -xzf bin/funasr-llamacpp.tgz -C bin/
rm bin/funasr-llamacpp.tgz

echo "== STT models (Paraformer q8 + FSMN-VAD)"
curl -sL --retry 3 -o models/paraformer-q8.gguf \
  https://huggingface.co/FunAudioLLM/Paraformer-GGUF/resolve/main/paraformer-q8.gguf
curl -sL --retry 3 -o models/fsmn-vad.gguf \
  https://huggingface.co/FunAudioLLM/fsmn-vad-GGUF/resolve/main/fsmn-vad.gguf

echo "== TTS model (Piper zh_CN-chaowen-medium-int8)"
curl -sL --retry 3 -o tts/chaowen-int8.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-zh_CN-chaowen-medium-int8.tar.bz2
tar --no-same-owner -xjf tts/chaowen-int8.tar.bz2 -C tts/
rm tts/chaowen-int8.tar.bz2

echo
echo "Done. Layout:"
ls bin/llama-funasr-paraformer models/paraformer-q8.gguf \
   tts/vits-piper-zh_CN-chaowen-medium-int8/zh_CN-chaowen-medium.onnx
