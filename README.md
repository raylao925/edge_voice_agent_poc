# Voice Agent POC

一个跑在 CPU 上的中文语音助手原型：浏览器里直接说话，实时得到语音回答。

```
浏览器麦克风
    │  WebSocket (音频)
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ STT         │────▶│ LLM          │────▶│ TTS         │
│ FunASR      │     │ OpenRouter   │     │ Piper       │
│ Paraformer  │     │ (openrouter/ │     │ chaowen     │
│ GGUF (q8)   │     │  free)       │     │ int8        │
│ llama.cpp   │     │              │     │ sherpa-onnx │
└─────────────┘     └──────────────┘     └─────────────┘
    │                                         │  WebSocket (音频)
    ▼                                         ▼
  文本                                      浏览器播放
```

## 特性

- **全 CPU 运行**：STT 和 TTS 都是量化模型，无需 GPU（2 核 / 8GB 内存实测可跑）
- **中文优化**：Paraformer 在中文识别上比 Whisper 误字率更低；Piper `chaowen` 中文音色
- **两种玩法**：浏览器实时对话（WebSocket）/ 命令行单轮对话
- **LLM 可换**：默认 OpenRouter 免费模型，一行环境变量切换

## 实测性能（2 vCPU，无 GPU）

| 环节 | 耗时 | RTF |
|---|---|---|
| STT（2.5 秒音频） | ~0.9 秒 | ~0.36 |
| LLM（OpenRouter free） | ~1.4 秒 | — |
| TTS（约 7 秒语音） | ~1.5–3 秒 | ~0.3 |
| 单轮端到端 | ~7–12 秒 | — |

> **RTF**（Real-Time Factor，实时系数）= 处理耗时 / 音频时长。
> RTF < 1 表示比实时更快，是评估 STT/TTS 性能的核心指标。
> 浏览器 demo 每轮对话后都会显示各阶段耗时和 RTF（绿色 < 0.5，黄色 < 1.0，红色 ≥ 1.0）。

## 快速开始

### 1. 环境要求

- Python 3.10+
- ffmpeg（`apt install ffmpeg` / `brew install ffmpeg`）

### 2. 安装依赖

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. 下载模型和二进制

```bash
bash download_models.sh
```

这会下载（约 300MB）：

| 内容 | 来源 | 落盘位置 |
|---|---|---|
| FunASR llama.cpp 预编译二进制（按系统自动选 avx2/arm64/macos 版） | modelscope/FunASR releases | `bin/` |
| Paraformer GGUF q8（~237MB）+ FSMN-VAD | HuggingFace `FunAudioLLM` | `models/` |
| Piper `zh_CN-chaowen-medium-int8`（~18MB） | sherpa-onnx releases | `tts/` |

### 4. 配置 LLM

```bash
# 方式一：环境变量（推荐）
export OPENROUTER_API_KEY="sk-or-v1-..."

# 方式二：写文件（权限 600）
mkdir -p .keys && echo "sk-or-v1-..." > .keys/openrouter.key && chmod 600 .keys/openrouter.key
```

没有 key 也能跑，此时 LLM 返回测试回复（pipeline 照样全通）。

去 [openrouter.ai](https://openrouter.ai) 免费注册拿 key。

### 5. 运行

**浏览器实时对话：**

```bash
python server.py
# 打开 http://127.0.0.1:8765
```

点麦克风说话，停顿约 1.5 秒自动结束，稍等即听到语音回答。
局域网/公网给别人玩：把 `server.py` 里的 `127.0.0.1` 改成 `0.0.0.0`（注意：demo 未做鉴权，仅本地/可信网络使用）。

**命令行单轮对话：**

```bash
python voice_chat.py input.m4a reply.mp3
# 输出：STT 文本、LLM 回复文本，并生成 reply.mp3
```

## 配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `OPENROUTER_MODEL` | `openrouter/free` | OpenRouter 模型 ID，可换成任意付费/免费模型 |
| `ANTHROPIC_API_KEY` | — | 备选：Anthropic 官方 key（无 OpenRouter key 时使用） |
| `VOICE_PORT` | `8765` | `server.py` 监听端口 |

## 文件结构

```
├── pipeline.py          # 核心：transcribe() / llm_reply() / synth_wav()
├── server.py            # WebSocket 服务 + 浏览器 demo 页（单端口）
├── voice_chat.py        # 命令行：音频文件 in → 回复 mp3 out
├── www/index.html       # 浏览器 demo 前端（录音 + VAD + 播放）
├── download_models.sh   # 一键下载二进制和模型
└── requirements.txt
```

## 工作原理

1. **录音**：浏览器 `MediaRecorder` 录音，前端用 `AnalyserNode` 做简单 VAD（静音 1.5 秒自动截断）
2. **STT**：音频经 ffmpeg 转 16k 单声道 WAV，调 `llama-funasr-paraformer` 子进程识别
3. **LLM**：文本调 OpenRouter（OpenAI 兼容接口），system prompt 要求简短口语化中文回答
4. **TTS**：`sherpa-onnx` 加载 int8 量化 Piper 模型合成 22050Hz 音频，回传浏览器播放

## 已知限制

- 中文普通话场景优化，粤语识别效果未验证
- TTS 会跳过词表外的字符（如 emoji、部分英文缩写）
- `server.py` 无鉴权，不要直接暴露到公网
- macOS 仅提供 arm64 二进制（Intel Mac 可用 Rosetta 运行 arm64 版）

## 致谢

- [FunASR](https://github.com/modelscope/FunASR)（llama.cpp / GGUF runtime）
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- [Piper TTS](https://github.com/rhasspy/piper)
- [OpenRouter](https://openrouter.ai)

## License

MIT
