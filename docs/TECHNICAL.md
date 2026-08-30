# Deskbot 技术文档

面向继续开发者的架构、模块职责、关键设计与已知坑。项目：桌面语音 AI 机器人，macOS Apple Silicon 为主 + Jetson Nano 兼容。

---

## 1. 系统架构

### 1.1 数据流（语音问答）

```
麦克风(16k mono)
  → audio/mic.py Mic       阻塞式采集，后台线程收块(0.1s)
  → audio/vad.py SpeechSegmenter
       SileroVAD(sherpa x/h/c) / EnergyVAD(兜底)  静音判定
  → (trigger=wake) audio/kws.py WakeWordSpotter   唤醒词「小桌」
  → audio/speaker.py Speaker + barge-in           抢话打断
  → asr.py ASR             SenseVoice(sherpa-onnx, int8)
  → main.py _handle        意图分支(见 2)
  → llm/client.py          DeepSeek / 本地 llama-server
  → tts.py TTS             VITS-zh-ll / piper
  → speaker.py 播放
```

### 1.2 触发与状态机

`main.py` 的 `_start_listener` 后台线程维护唤醒状态：

```
待命(idle) ──命中「小桌」──→ 会话中(in_session)
    ↑                            │ 每说一句话续期
    │ 静默>conversation_timeout  │ (回答播放中不超时)
    └────────────────────────────┘
```

- `trigger: wake`：未唤醒只跑 KWS；唤醒后 VAD 片段正常入队，**不再每句锁定**。
- `trigger: keyboard`：常听直说（Nano 默认），不涉及唤醒。

### 1.3 双平台差异

| | Mac (config.yaml) | Nano (config.nano.yaml) |
|---|---|---|
| 触发 | wake「小桌」 | keyboard |
| TTS | **Qwen3-TTS**（mlx-audio，24kHz，最自然）；Kokoro `engine: sherpa` 可选 | **matcha-zh-baker**（合成 2.4s/句，RTF 0.69 实测） |
| 嵌入 | llama-server Metal (ngl 999) | llama-server CPU |
| 多音字消歧 | VITS/Matcha 启用（lexicon.txt 体系）；Kokoro 走拼音原生消歧 / Qwen3 无需 | matcha pinyin 前端 lexicon 自带，Zhuyin 词表不适用 → 关 |
| 麦克风 | USB（蓝牙不可用） | USB |

代码改动用**安全默认**隔离（缺失配置键自动取缺省），因此同一代码可跑双平台。`scripts/dev-sync.sh` 排除 `config.yaml`，防止 Mac 配置覆盖 Nano。

---

## 2. 意图分支（main.py `_handle`）

单轮文字入口，分支顺序（命中即返回）：

```
A. 退出    "退出/再见/quit/exit/拜拜" → KeyboardInterrupt
B. 位置    "我在杭州"(_LOCATION_RE) → 记录+固定句，绕过LLM
C. 记忆    "记住XXX"/"我喜欢XXX"(feedback.py analyze) → 入库+固定句，绕过LLM
D. 反馈    "回答很好/不对"(_only_feedback) → 固定夸奖/抱歉句，绕过LLM
E. 正常问答 → RAG检索+规则+视觉 → LLM(ask/ask_stream) → 记忆抽取
```

- 位置正则 `_LOCATION_RE`：`(?:我|我现在)(?:现在)?在([一-龥]{2,6}?)(?:方位词)?`，拒绝疑问句/占位词。
- 反馈关键词来自 `config.yaml optimize.*`，**"对"是"回答不对"子串**（双重匹配），场景测试用无歧义词。
- 记忆类输入**不进 LLM history**，而是 RAG 注入（多轮断言查 `llm_rag_has` 而非 history）。

## 3. 关键设计

### 3.1 免唤醒连续对话（会话超时）

`wake.conversation_timeout`（默认 30s）。唤醒后 `in_session=True`；每个 VAD 片段续期 `session_deadline`；到期时若 `_current_task` 未完成（回答播放中）则续期等待。实现于 `_start_listener` 的 `run()` 闭包。

### 3.2 多音字消歧（tts_poly.py）

**背景**：sherpa-onnx 的 text→phoneme 全在 C++ 内部——cppjieba 分词 → `lexicon.txt` 最大匹配查音 → Zhuyin 音素。`lexicon.txt` 中**单字多音字只有默认读音**（还→hái），无词条词（归还/地壳）会被 jieba 拆单字 → 读错。

**机制**：
- `OfflineTts` **无热重载**（lexicon 构造时固定）→ 必须在启动时一次性写入。
- `TTS._resolve_lexicon()` 用 `PolyphoneResolver.augment_lexicon()` 把正确注音词条追加到 `lexicon.session.txt`，构建时加载。
- 注音用 **Zhuyin**（`ㄏ ㄨ ㄢ ˊ`，声调 ˉˊˇˋ˙ 独立 token，tokens.txt 44-48），非 pinyin。
- 词条需验证 token 全在 tokens.txt；piper 无 lexicon 自动跳过（`can_support=False`）。

**选词原则**：合成 → SenseVoice ASR 回环确认真读错才加。已修复：归还/还价/地壳/解数/亲家/中意/着重/目的/的确/行李。

**验证方法**：`TTS.synthesize(text)` → `ASR.transcribe(audio)` 读回听音，对比基线 vs 消歧。

### 3.3 自动化对话考核（dialogue.py）

`python -m deskbot.main --dialogue` 驱动 `_handle` 分支逻辑，12 场景断言预期行为：

- `FakeLlm`（记录 ask 调用 + 罐头答案）+ `HashingEmbedder`（确定性离线嵌入）→ **hermetic，无需真实服务器/.env**。
- 每场景前 `_reset_state`（store/llm/feedback 隔离）；多轮场景故意复用状态。
- 插话打断（barge_in）在**音频层**，`--dialogue` 只测 `_barge_in` 机制（stub speaker 计 stop + 取消 fake 任务），标注 `scope=audio-manual`。
- 报告追加 `data/datasets/dialogue_report.jsonl`，含 `baseline_passed` 回归保护。

### 3.4 评估（eval.py）

- `data/datasets/eval_set.jsonl`：22 题 `{"q":..., "keywords":[...]}`，任一关键词出现即过。
- 走 `llm.complete()` 直测（**不走 `_handle`**，只测 LLM 事实输出）。
- 当前基线 **21/22 (96%)**，报告 `eval_report.jsonl`。

## 4. 模块清单

| 模块 | 职责 |
|---|---|
| `main.py` | CLI、唤醒/会话状态机、`_handle` 分支、流式 TTS 流水线 |
| `audio/mic.py` | 阻塞式采集，后台线程收块 |
| `audio/vad.py` | SileroVAD(sherpa x/h/c)/EnergyVAD + 语音分段器 |
| `audio/kws.py` | 唤醒词（sherpa KeywordSpotter） |
| `audio/speaker.py` | 播放 + barge-in（`sd.play`/`sd.stop`） |
| `asr.py` | SenseVoice（`from_sense_voice`，`language="zh"`） |
| `tts.py` | VITS/piper 合成，FST 规则，lexicon 扩展 |
| `tts_poly.py` | 多音字静态词表 |
| `geo.py` | 位置（口述优先 + ipinfo.io IP 兜底） |
| `llm/server.py` | llama-server 子进程管理（跨平台杀端口） |
| `llm/client.py` | DeepSeek/本地 OpenAI 兼容客户端、流式、历史 |
| `rag/embedder.py` | 嵌入（llama-server / HashingEmbedder 降级） |
| `rag/store.py` | 向量库（numpy npz + jsonl，时间加权检索） |
| `rag/memory.py` | 记忆抽取/写入/检索 |
| `optimize/feedback.py` | 意图识别（记忆/反馈）+ 落盘 |
| `optimize/eval.py` | 评估集打分 + 回归 |
| `optimize/dialogue.py` | 对话考核 |
| `optimize/prompt_tuner.py` | 规则自改进 |
| `optimize/dataset.py` | 训练集导出 |
| `vision/` | 摄像头 + YOLO（默认关，模型未下载） |

## 5. 已知坑（非显而易见）

1. **macOS 系统代理**：本机开着 Clash 类代理（`127.0.0.1:7890`），httpx 默认 `trust_env=True` 会转发本地请求 → **502**。所有访问本地 llama-server 的 httpx 调用加 `trust_env=False`；DeepSeek 远程保留 `trust_env=True`。
2. **VAD 坏模型**：`silero_vad.onnx` 必须用 **sherpa 官方版**（628KB，输入 `x/h/c`）。若启动日志显示 `格式=official, 输入=['input','state','sr']` 即坏模型（带 sr 的 v5，语音概率≈0），换回 sherpa 版。症状：唤醒词能命中但唤醒后说话永远判静音。
3. **蓝牙耳机当麦克风拾音增益极低**：真实语音 peak≈0.005，KWS/ASR 全识别不了。需 USB 麦克风（Quark2 类，peak≈0.06）。诊断：`sd.rec` 录真实语音看 peak。
4. **sherpa-onnx 无热重载**：`OfflineTts`/lexicon 构造时固定 → 任何 lexicon 改动需启动时写入 + 重建。
5. **KeywordSpotter.get_result() 返回 str**（sherpa 1.13.6）：`getattr(r,"keyword")` 永远 None，直接 `if r: return str(r)`。
6. **Qwen3-0.6B 只有 Q8_0 量化**（官方仓库无 Q4_K_M），文件 `Qwen3-0.6B-Q8_0.gguf`（610MB）。
7. **TTS FST OOV 警告**：LLM 输出含全角括号/英文（`（`、`LA`）时 `character-lexicon.cc: Ignore OOV`（stderr，无碍运行）。
8. **`HashingEmbedder` 无 `embed_one`**：测试里需手动补（`hashing.embed([text])[0]`）。
9. **`cancel()` 后需 `await sleep(0)`** 让 asyncio 处理取消，`task.cancelled()` 才生效。
10. **Nano 用 Python 3.8**：语法需兼容（`from __future__ import annotations` 已用）。

## 6. 配置对照（Mac vs Nano 关键差异）

| 键 | Mac | Nano |
|---|---|---|
| `trigger` | `wake` | `keyboard` |
| `wake.keyword` | 小桌 | — |
| `wake.conversation_timeout` | 30 | — |
| `tts.model_dir` | vits-zh-ll | piper-xiao_ya |
| `tts.speaker_id` | 2 | 0 |
| `tts.enable_fst` | true | false |
| `tts.polyphone.enabled` | true | false |
| `llm.n_gpu_layers` | 999 (Metal) | 缺省 0 (CPU) |
| `audio.barge_in` | true | 缺省 true |
| `asr.engine` | sense_voice | sense_voice |

缺失键由代码安全默认兜底（barge_in 缺省 true、enable_fst 缺省 false、ngl 缺省 0），故单配置即可跑双平台。

## 7. 验证与回归

```bash
python -m pytest tests/            # 6/6 单元测试
python -m deskbot.main --smoke     # 冒烟
python -m deskbot.main --eval      # 21/22 (96%)
python -m deskbot.main --dialogue  # 12/12 对话考核
```

改动 LLM 提示词 / 分支逻辑后跑 `--eval` + `--dialogue` 对比基线（`eval_report.jsonl` / `dialogue_report.jsonl` 有 `baseline_passed` 回归保护）。

## 8. 后续方向（未实现）

- **vits-zh-hf-fanchen-C**：187 音色 VITS，drop-in（改 `tts.model_dir` + `speaker_id`）。音质好但 Nano CPU RTF 1.6-4.3 → 慢，仅在 Mac 可选。
- **语音换音色命令**：拦截"换个音色/音色N号"，运行时改 `self.tts.speaker_id`（`generate` 每调传 sid，即时生效，无需重建）。试听工具已就绪：`--say "你好" --say-speaker <id>`。
- **视觉 M5**：`vision.enabled` 默认关，YOLO 模型未下载；代码完整待启用。
- **M4 微调**：`finetune_lora.py` + `fine_tune.sh` 已写但 train.jsonl 样本不足。

## 9. TTS 模型矩阵与决策记录

| 模型 | 音质 | 采样率 | 权重 | Mac RTF | Nano CPU | 结论 |
|---|---|---|---|---|---|---|
| **Qwen3-TTS-0.6B**（mlx-audio） | ⭐⭐⭐⭐⭐ 最自然 | 24kHz | ~1.3GB(8bit) | RTF ~0.5（Apple Silicon 实测） | — | **Mac 主用**（`tts.engine: qwen3`），常驻 Python 子进程 |
| **Kokoro v1.0**（zh+en） | ⭐⭐⭐ | 24kHz | ~81MB onnx | ~0.3 | RTF 3-8（不可行） | Mac 备选（`engine: sherpa`），53 音色 |
| **VITS-zh-ll** | ⭐⭐ | 16kHz | ~115MB | ~0.25 | 6-13s/句 | Mac 兜底 / Nano 慢 |
| **piper xiao_ya int8** | ⭐ | 16kHz | ~100MB | — | 1.6-3.9s/句 | Nano 现状，最快 |
| **matcha-icefall-zh-baker** | ⭐⭐⭐ | 22.05kHz | ~96MB + vocoder 52MB | RTF 0.03（Mac 实测） | **RTF 0.69 实测**（A57，合成 2.4s/句，加载 9.1s） | **Nano 主用**，drop-in（`tts.py` Matcha 分支已修复 vocoder） |

**Nano GPU（Maxwell 128 核）结论：不追。** 原因：sherpa-onnx TTS 走自编译的 onnxruntime CPU C API，无 Python 钩子切 GPU provider；CUDA 10.2 无配套 aarch64 wheel（sherpa-onnx CUDA 在 Nano 实测不可运行）；换 onnxruntime-gpu 只影响 ORTC 路径，理论增益 ~1.3-1.8x 不抵安装风险。

### Qwen3-TTS（Mac）接入说明

- **运行时**：Python `mlx-audio` 0.4.3 参考实现（mlx Metal），Apple Silicon 专属，Nano 不可用。
  > 起初用官方 Swift 移植（`AtomGradient/swift-qwen3-tts`）——其 AR 生成有 bug：官方 `Qwen3TTSDemo`
  > 与本项目服务在任意型号/语言/采样下都产出**退化解码音频**（常 6s 满帧、无 EOS、87% 削波）。
  > 已定位为 Swift 移植的 AR 循环问题，改为 Python 参考实现后稳定（干净语音、时长随文本合理、0% 削波）。
- **模型**：必须用 **unpruned 版本** `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit`。
  > `AtomGradient/*-pruned-vocab-lite` 在 Swift/Python 两种推理路径都不稳定（同句时长 0.9-5s 漂移、
  > 偶发空输出）。unpruned-8bit 在 `mlx-audio.tts.utils.load(strict=False)` 下稳定（实测同句 2.7-3.7s）。
- **后端**：`deskbot/tts_qwen3.py` 常驻子进程（`subprocess.Popen` + stdin/stdout 行协议），
  `deskbot/tts_qwen3_server.py` 在独立 venv（Python 3.11 + mlx-audio 0.4.3）跑，模型只加载一次（~1.6s）。
  解释器路径：`QWEN3_PYTHON` 环境变量，或默认 `~/projects/deskbot-mlx-audio/.venv/bin/python`。
- **配置**：`tts.engine: qwen3`（Mac 默认）+ `tts.speaker`（Vivian/Ryan/Aiden 等 CustomVoice 音色）+ `tts.language: zh`。
- **集成**：server 输出 WAV → Python 读回 → `Speaker.play`。`--say` / 预热 / barge-in 全走同一 `synthesize()` 接口。

### Matcha vocoder 修复（相对上个版本）

`sherpa-onnx` 的 `OfflineTtsMatchaModelConfig` **必须**传 `vocoder`（如 `vocos-22khz-univ.onnx`，官网单独下载放模型目录 `vocoder-*.onnx`）。旧 `tts.py` 的 Matcha 分支漏传导致 TypeError——已修复。
