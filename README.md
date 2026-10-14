# Voice Agent POC

一个跑在 CPU 上的中文语音助手原型：浏览器里直接说话，实时得到语音回答。

```
浏览器麦克风
    │  WebSocket (音频)
    ▼
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│ STT         │────▶│ Agent                │────▶│ TTS         │
│ FunASR      │     │ 1. 意图判断 (LLM)     │     │ Piper       │
│ Paraformer  │     │ 2. 工具: 搜索/新闻/天气 │     │ chaowen     │
│ GGUF (q8)   │     │ 3. 生成回答 (LLM)     │     │ int8        │
│ llama.cpp   │     │                      │     │ sherpa-onnx │
└─────────────┘     └──────────────────────┘     └─────────────┘
    │                                                   │  WebSocket (音频)
    ▼                                                   ▼
  文本                                                浏览器播放
```

## 特性

- **全 CPU 运行**：STT 和 TTS 都是量化模型，无需 GPU（2 核 / 8GB 内存实测可跑）
- **中文优化**：Paraformer 在中文识别上比 Whisper 误字率更低；Piper `chaowen` 中文音色
- **实时工具**：可上网查资料、查新闻、查天气（见下）
- **两种玩法**：浏览器实时对话（WebSocket）/ 命令行单轮对话
- **LLM 可换**：默认 OpenRouter 免费模型，一行环境变量切换

## 实时工具

Agent 采用两段式：LLM 先判断意图 → 执行工具 → LLM 基于工具结果生成口语回答。
（不用 LLM 原生 function calling，因为免费模型对 tool calling 支持不稳定。）

| 工具 | 实现 | 说明 |
|---|---|---|
| `web_search` | DuckDuckGo（`ddgs`） | 免 key；想更准可换 Tavily（有免费额度） |
| `news_search` | DuckDuckGo news（`ddgs`） | 免 key |
| `get_weather` | Open-Meteo | 免 key、无严格限额，自动 geocoding |
| `fetch_page` | Jina Reader | 免 key，把网页转干净文本 |

无 LLM key 时，意图判断降级为关键词匹配（天气/新闻/搜索），工具照样跑。

试试问它：「澳门今天天气怎么样」「最新 AI 新闻」「帮我搜一下横琴人工智能大赛」

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

### 2. 安装依赖（推荐 uv）

先装 [uv](https://docs.astral.sh/uv/)（如果还没装）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

然后一键同步环境（自动建 venv、按 `uv.lock` 装精确版本）：

```bash
uv sync
```

> 不想用 uv 的话，传统方式也行：`python -m venv venv && source venv/bin/activate && pip install -r requirements.txt`

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
uv run server.py
# 打开 http://127.0.0.1:8765
```

点麦克风说话，停顿约 1.5 秒自动结束，稍等即听到语音回答。
局域网/公网给别人玩：把 `server.py` 里的 `127.0.0.1` 改成 `0.0.0.0`（注意：demo 未做鉴权，仅本地/可信网络使用）。

**命令行单轮对话：**

```bash
uv run voice_chat.py input.m4a reply.mp3
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
├── pipeline.py          # 核心：transcribe() / agent_reply() / synth_wav() / run_turn()
├── tools.py             # 实时工具：web_search / news_search / get_weather / fetch_page
├── server.py            # WebSocket 服务 + 浏览器 demo 页（单端口）
├── voice_chat.py        # 命令行：音频文件 in → 回复 mp3 out
├── www/index.html       # 浏览器 demo 前端（录音 + VAD + 播放 + 性能面板）
├── download_models.sh   # 一键下载二进制和模型
├── pyproject.toml       # uv / pip 依赖定义
├── uv.lock              # 锁定的精确版本
└── requirements.txt     # pip 备用
```

## 工作原理

1. **录音**：浏览器 `MediaRecorder` 录音，前端用 `AnalyserNode` 做简单 VAD（静音 1.5 秒自动截断）
2. **STT**：音频经 ffmpeg 转 16k 单声道 WAV，调 `llama-funasr-paraformer` 子进程识别
3. **Agent**：LLM 判断意图 → 按需调用工具（搜索/新闻/天气）→ LLM 生成回答
4. **TTS streaming**：LLM 用 SSE 流式输出，按句子切分后逐句调 Piper 合成，音频 chunk 经 WebSocket 逐段发送，浏览器顺序播放——**首音延迟约 1.6 秒**（对比非流式约 6 秒）

## STT 模型對比

在 2 vCPU 上實測三個 FunASR llama.cpp 模型（6 段測試音頻，CER 已去除標點差異）：

| 模型 | 大小 | 平均 RTF | 平均 CER | 特點 |
|---|---|---|---|---|
| **Paraformer q8**（預設） | 226MB | **0.37** | 1.4% | 最快，準確度夠用 |
| SenseVoiceSmall q8 | 243MB | 0.45 | 4.0% | 自帶標點，支援粵語（yue） |
| FunASR-Nano q4km | 910MB | 2.07 | **0.0%** | 最準，但 CPU 上比實時慢，不適合即時對話 |

> **結論**：即時 voice agent 繼續用 Paraformer（速度/準確度平衡最好）。
> Nano 準確度最高但 RTF > 1（Qwen3-0.6B 自回歸解碼在 CPU 上太慢），適合離線轉寫；
> SenseVoice 速度接近 Paraformer，自帶標點＋粵語支援，有興趣可試。
> 測試細節見 `stt-bench/`（已 gitignore，不在 repo 內）。

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
