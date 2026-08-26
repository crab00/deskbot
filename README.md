# Deskbot — Jetson Nano 本地语音 AI 桌面机器人

把一台 **Jetson Nano（4GB）** 变成能对话的桌面机器人：麦克风提问 → 本地大模型回答 → 扬声器播报，完全离线；并内置「持续优化」闭环（记忆、反馈、规则自改进、微调）。

## 架构

```
[麦克风] → VAD → ASR(sherpa-onnx Paraformer) → [RAG 记忆检索] → LLM(llama.cpp+Qwen1.5B) → TTS(sherpa-onnx VITS) → [扬声器]
                                                    ↑                                                        ↓
                                         [摄像头] → YOLO 物体描述              [优化闭环] → 反馈/记忆/评估/微调
```

## 快速开始

```bash
# 1) 在 Jetson Nano 上一键部署（Python≥3.8、依赖、llama.cpp、模型、swap）
./scripts/setup_jetson.sh

# 2) 冒烟测试（验证各模块）
source .venv/bin/activate
python -m deskbot.main --smoke

# 3) 键盘问答（无音频，先跑通逻辑）
python -m deskbot.main --text

# 4) 语音问答（接好 USB 麦克风+扬声器）
python -m deskbot.main --voice
```

开发机(Mac)改完代码后同步到 Nano：

```bash
./scripts/dev-sync.sh crab@192.168.31.202   # 或直接 ./scripts/dev-sync.sh
```

## 常用命令

| 命令 | 作用 |
|---|---|
| `python -m deskbot.main --voice` | 语音问答（按 Enter 后说话） |
| `python -m deskbot.main --text` | 键盘问答 |
| `python -m deskbot.main --smoke` | 冒烟测试 |
| `python -m deskbot.main --eval` | 跑评估集并打分 |
| `python -m deskbot.main --tune` | 规则自改进（扫描反馈→调 prompt→回归） |
| `python -m deskbot.main --export` | 导出微调训练集 |
| `python -m deskbot.main --memory-add "我喜欢喝咖啡"` | 直接记住一句话 |
| `python -m deskbot.main --say "你好"` | 仅合成播报一句话 |

## 持续优化闭环

```
对话日志 ──→ 记忆抽取 ──→ RAG 向量库（下次回答自动用）
    │
    └──→ 用户反馈(语音"回答很好/不对"或文字) ──→ 优质数据
            ├─→ 规则自改进：--tune 自动提炼规则→注入 prompt→评估回归→失败回滚
            └─→ 训练集导出：--export → 离机 QLoRA 微调 → 部署 + 评估对比
```

### 微调流程（离机，需带独显的 PC 或云 GPU）

```bash
# 1) Nano 上导出数据
python -m deskbot.main --export
# 2) 把 data/datasets/train.jsonl 传到 GPU 机器
scp data/datasets/train.jsonl user@gpu-host:~/
# 3) GPU 机器上微调（产物为 Q4_K_M GGUF）
./scripts/fine_tune.sh train.jsonl
# 4) 回传 Nano 并部署（自动评估对比，分数低可回滚）
./scripts/deploy_model.sh models/llm/finetuned/model.gguf
```

## 配置

所有配置在 `config.yaml`：
- `llm.model` / `llm.system_prompt`：模型路径与人格
- `rag.*`：记忆库（嵌入模型、检索条数）
- `optimize.*`：反馈关键词、规则/变更记录文件、评估集
- `audio.*` / `asr.*` / `tts.*`：音频、识别、合成
- `vision.*`：摄像头与 YOLO

## 目录

```
deskbot/
├── deskbot/               # 主代码
│   ├── main.py            # 入口与流水线编排
│   ├── audio/             # 麦克风 / VAD / 扬声器
│   ├── asr.py tts.py      # 语音识别 / 合成（sherpa-onnx）
│   ├── llm/               # llama.cpp server 管理 + 客户端
│   ├── rag/               # 嵌入 / 向量库 / 记忆
│   ├── vision/            # 摄像头 / YOLO 检测
│   └── optimize/          # 反馈 / 规则自改进 / 评估 / 数据集
├── scripts/               # 部署 / 同步 / 微调 / 发布脚本
├── data/                  # 运行数据（日志 / 反馈 / 记忆 / 数据集）
└── models/                # 模型文件（脚本下载）
```

## 注意事项

- **性能**：Nano 4GB 上 1.5B Q4 推理约 3~8 token/s，回答已限制 256 token；CPU 推理最稳（CUDA 10.2 过旧）。
- **内存**：建议 4GB swap（setup 脚本已配置）；ASR/TTS 模型按需加载。
- **供电散热**：满负载务必 5V/4A 直流供电 + 散热风扇。
- **模型下载**：需要联网一次；嵌入模型由 fastembed 首次运行时下载。
