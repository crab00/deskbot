# Deskbot — 桌面语音 AI 机器人

唤醒即用的桌面语音助手：喊「小桌」→ 说话 → 云端大模型回答 → 扬声器播报。支持**免唤醒连续对话**、**抢话打断**、**多音字正确读音**、**长期记忆**与**优化闭环**。

> 同一套代码跑在 **macOS Apple Silicon（Mac，主）** 与 **Jetson Nano（Nano，保持可用）** 上。LLM 生成走 DeepSeek 线上 API，本地 llama-server 只做 RAG 嵌入。

## 架构

```
[麦克风] → VAD(静音检测) → ASR(SenseVoice) → [RAG 记忆检索] → LLM(DeepSeek) → TTS(VITS/piper) → [扬声器]
    ↑                          ↑                                                  ↑
[唤醒词 KWS]（trigger=wake）  [免唤醒会话超时]                              [多音字消歧]
```

- **触发方式**：唤醒词「小桌」或键盘（config `trigger: wake|keyboard`）
- **持续对话**：唤醒后无需每次重喊，静默超过 `conversation_timeout` 才重新待命
- **打断**：回答播放中直接说话即打断（barge-in）
- **LLM**：DeepSeek（`deepseek-chat`）线上生成；本地 Qwen3-0.6B 仅作 RAG 嵌入（1024 维）
- **RAG**：对话记忆向量检索，最新记忆优先注入（时间加权）
- **优化闭环**：记忆 / 反馈 / 规则自改进 / 评估 / 对话考核

## 快速开始

### Mac（Apple Silicon，主）

```bash
# 1) 环境（sherpa-onnx 1.13.6 / onnxruntime / sounddevice；brew 装 llama.cpp + portaudio）
python3 -m venv .venv-mac && source .venv-mac/bin/activate
pip install -e .[mac]
bash scripts/mac-download.sh            # 下载模型（SenseVoice/VITS/KWS/VAD/嵌入）

# 2) 冒烟测试
python -m deskbot.main --smoke

# 3) 语音问答（配好 USB 麦克风）
python -m deskbot.main --voice          # 喊「小桌」唤醒，连续对话免唤醒
```

### Jetson Nano（保持可用）

```bash
./scripts/setup_jetson.sh               # 一键部署（venv/依赖/模型/swap）
./scripts/dev-sync.sh crab@192.168.31.202   # 从 Mac 同步代码（排除 config.yaml）
# Nano 用自己的 config.yaml（keyboard 触发 + piper TTS + CPU 推理）
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `python -m deskbot.main --voice` | 语音问答（唤醒词「小桌」+ 免唤醒连续对话） |
| `python -m deskbot.main --text` | 键盘问答（无音频，调试） |
| `python -m deskbot.main --smoke` | 冒烟测试（LLM/记忆/反馈/导出） |
| `python -m deskbot.main --eval` | 评估集打分（当前 21/22, 96%） |
| `python -m deskbot.main --dialogue` | 自动化对话考核（12 场景文字层断言） |
| `python -m deskbot.main --tune` | 规则自改进（扫描反馈→调 prompt→回归） |
| `python -m deskbot.main --export` | 导出微调训练集 |
| `python -m deskbot.main --memory-add "..."` | 直接记住一句话 |
| `python -m deskbot.main --say "你好"` | 仅合成播报一句话 |

## 核心能力

### 免唤醒连续对话（会话超时）
- 喊「小桌」唤醒后，**持续对话无需再唤醒**；每说一句话自动续期
- 静默超过 `wake.conversation_timeout`（默认 30s）且无回答在播 → 回到待命
- 小桌回答播放期间**不计入静默**（回答中续期）

### 抢话打断（barge-in）
- 回答播放中直接说话 → 立即停止 TTS + 取消当前回答任务，处理新语音

### 多音字正确读音（静态消歧）
- sherpa-onnx 的 text→phoneme 在 C++ 内部（jieba 分词 → lexicon 查音），单字多音字默认音易读错
- `deskbot/tts_poly.py` 在启动时把正确注音（Zhuyin）词条追加进 lexicon，让 jieba 匹配整词读对
- 已修复：归还/还价/地壳/解数/亲家/中意/着重/目的/的确/行李
- piper（Nano）无 lexicon 自动跳过

### 长期记忆 + 优化闭环
- 说「记住我喜欢吃火锅」→ 直接入库；后续回答自动用（RAG 检索 + 时间加权）
- 反馈（"回答很好/不对"）→ 规则自改进 → 评估回归
- 离机微调：`--export` → GPU 机 `fine_tune.sh` → `deploy_model.sh`

## 配置

所有配置在 `config.yaml`（Mac）与 `config.nano.yaml`（Nano）：

| 段 | 关键项 |
|---|---|
| `trigger` / `wake` | 触发方式、唤醒词、会话超时 |
| `audio` | 麦克风/扬声器设备、VAD、barge_in |
| `asr` | SenseVoice 模型、线程数 |
| `tts` | 模型目录、音色 `speaker_id`、FST、多音字消歧 `polyphone` |
| `llm` | DeepSeek provider、API key、本地嵌入 server |
| `rag` | 向量库维度、检索条数、记忆抽取 |
| `optimize` | 反馈关键词、评估集、规则文件 |
| `vision` | 视觉（默认关闭，需 YOLO 模型） |

## 目录

```
deskbot/
├── deskbot/
│   ├── main.py            # 入口、唤醒/会话状态机、_handle 分支
│   ├── audio/             # 麦克风 / VAD / 唤醒词 KWS / 扬声器(打断)
│   ├── asr.py tts.py      # SenseVoice 识别 / VITS 合成
│   ├── tts_poly.py        # 多音字静态消歧
│   ├── geo.py             # 位置（口述优先 + IP 兜底）
│   ├── llm/               # llama-server 管理 + OpenAI 兼容客户端
│   ├── rag/               # 嵌入 / 向量库 / 记忆
│   ├── optimize/          # 反馈 / 规则自改进 / 评估 / 对话考核 / 数据集
│   ├── vision/            # 摄像头 / YOLO 检测（默认关）
│   └── utils/             # 配置 / 日志 / 状态
├── scripts/               # 部署 / 同步 / 下载 / 微调
├── data/                  # 日志 / 反馈 / 记忆 / 数据集 / 评估报告
└── models/                # 模型文件（脚本下载）
```

## 技术文档

详见 [docs/TECHNICAL.md](docs/TECHNICAL.md)：架构细节、模块职责、关键设计、已知坑、配置对照。

## 注意事项

- **Mac**：需 USB 麦克风（蓝牙耳机拾音增益过低不可用）；系统代理开着时本地 llama-server 调用走 `trust_env=False`（见技术文档坑列表）。
- **Nano**：CPU 推理，TTS 合成是延迟瓶颈；GPU 不可行（CUDA 10.2 过旧）。
- **模型**：`.env` 配 `DEEPSEEK_API_KEY`；模型文件不入 git，用脚本下载。
