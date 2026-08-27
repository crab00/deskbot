"""Deskbot 主程序（asyncio 编排）。

用法：
  python -m deskbot.main --voice      # 语音问答（麦克风→ASR→LLM→TTS→扬声器）
  python -m deskbot.main --text       # 键盘问答（无音频，开发/调试用）
  python -m deskbot.main --smoke      # 冒烟测试（各模块自检 + 一次 LLM 调用）
  python -m deskbot.main --eval       # 跑一次评估集
  python -m deskbot.main --tune       # 扫描反馈并执行一次规则自改进
  python -m deskbot.main --export     # 导出训练集
  python -m deskbot.main --memory-add "我家有只猫叫咪咪"   # 直接记住一句话
  python -m deskbot.main --say "你好"  # 只说一句话（合成并播报，验证 TTS）

按键说话（--voice）：听到提示音后按 Enter 开始说话，说完停顿自动结束。
内置口令：退出/再见 → 退出；回答很好/不对 → 反馈。
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

from .utils.config import Config, load_config
from .utils.async_utils import to_thread
from .utils.logging_setup import get_logger, setup_logging

log = get_logger("main")

# 延迟导入（让 --smoke/--export 等子命令在缺少音频依赖时也能跑）
from .audio.mic import Mic, AudioUnavailable as MicUnavailable  # noqa: E402
from .audio.speaker import Speaker  # noqa: E402
from .audio.vad import EnergyVAD, SileroVAD, SpeechSegmenter  # noqa: E402

VAD_MODEL = "models/vad/silero_vad.onnx"

# 用户口述位置：口语化"我在杭州"/"我现在在上海"，后缀方位词（这边/那儿）剔除
_LOCATION_RE = re.compile(r"(?:我|我现在)(?:现在)?在([一-龥]{2,6}?)(?:这里|这儿|那边|这边)?[。！？]?$")
# 方位词占位符（"我在这里"不是位置），剔除
_PLACEHOLDER_LOCS = {"这里", "这儿", "那边", "这边", "哪里"}

# 纯标点/符号（VAD 误触发抓到的噪音常被 ASR 识别成这些）
_NOISE_RE = re.compile(r"^[\s\.。·…、,，!！?？~～\-—_—（）()【】\[\]\"'“”‘’]+$")
# 常见无意义语气词/单字（无语义，只会触发小模型坍缩）
_FILLERS = {"嗯", "啊", "哦", "呃", "诶", "唔", "哈", "唉", "哟", "呀", "哼", "喔",
            "哦哦", "嗯嗯", "啊哈", "呵呵", "嘿嘿", "对哦", "是哦"}


def _is_noise(text: str) -> bool:
    """判定一段 ASR 文本是否为无意义噪声（纯标点/语气词），应忽略而非送 LLM。"""
    t = (text or "").strip()
    if not t:
        return True
    if _NOISE_RE.match(t):
        return True
    core = re.sub(r"[\s\.。·…、,，!！?？~～\-—_—（）()【】\[\]\"'“”‘’]+", "", t)
    return core in _FILLERS


class DeskBot:
    def __init__(self, cfg: Config, voice: bool = True):
        self.cfg = cfg
        self.voice = voice
        self._start_ts = time.time()
        self.trigger = str(self.cfg.get("trigger", "keyboard")).lower()
        self._wake_spotter = None

        # ---- LLM ----
        from .llm.client import LlmClient
        from .llm.server import LlamaServer
        self.llm_server = LlamaServer(cfg)
        self.llm = LlmClient(cfg)

        # ---- ASR / TTS（可选加载）----
        self.asr = None
        self.tts = None
        self.mic: Optional[Mic] = None
        self.speaker: Optional[Speaker] = None
        if self.voice:
            self._init_audio()

        # ---- RAG ----
        from .rag.embedder import Embedder
        from .rag.memory import MemoryService
        from .rag.store import VectorStore
        self.embedder = Embedder(cfg)
        store_path = cfg.path("paths.memories") / "vectors"
        self.store = VectorStore(store_path, dim=self.embedder.dim)
        self.memory = MemoryService(cfg, self.embedder, self.store, self.llm)

        # ---- 优化闭环 ----
        from .optimize.eval import EvalRunner
        from .optimize.feedback import FeedbackCollector
        from .optimize.prompt_tuner import PromptTuner, RulesStore
        self.collector = FeedbackCollector(cfg)
        self.rules = RulesStore(cfg)
        self.eval_runner = EvalRunner(cfg, self.llm)
        self.tuner = PromptTuner(cfg, self.llm, self.rules, self.eval_runner)

        # ---- 视觉（可选）----
        self.vision = None
        if cfg.get("vision.enabled", False):
            self._init_vision()

        self.conv_log_path = cfg.path("paths.logs") / time.strftime("conv_%Y%m%d.jsonl")
        self._turn_timings: dict = {}   # 本轮语音流水线分阶段耗时（VAD/ASR/LLM/TTS）

    # ================= 组件初始化 =================
    def _init_audio(self) -> None:
        a = self.cfg.section("audio")
        try:
            self.mic = Mic(device=a.get("mic_device"), sample_rate=a.get("sample_rate", 16000))
            self.speaker = Speaker(device=a.get("speaker_device"))
        except MicUnavailable as e:
            log.warning("音频不可用，降级为键盘模式: %s", e)
            self.voice = False
            return
        try:
            from .asr import ASR
            self.asr = ASR(self.cfg.path("asr.model_dir"),
                           engine=str(self.cfg.get("asr.engine", "auto")),
                           num_threads=int(self.cfg.get("asr.num_threads", 2)))
        except Exception as e:
            log.warning("ASR 加载失败，语音输入不可用: %s", e)
            self.asr = None
        try:
            from .tts import TTS
            poly_entries = None
            if self.cfg.get("tts.polyphone.enabled", False):
                poly_entries = self.cfg.get("tts.polyphone.entries", {}) or {}
            self.tts = TTS(self.cfg.path("tts.model_dir"),
                           num_threads=int(self.cfg.get("tts.num_threads", 2)),
                           speaker_id=int(self.cfg.get("tts.speaker_id", 0)),
                           speed=float(self.cfg.get("tts.speed", 1.0)),
                           enable_fst=bool(self.cfg.get("tts.enable_fst", False)),
                           poly_entries=poly_entries)
        except Exception as e:
            log.warning("TTS 加载失败，语音输出不可用: %s", e)
            self.tts = None
        if self.asr is None and self.tts is None:
            self.voice = False
        if self.voice:
            self._init_wake()

    def _init_vision(self) -> None:
        try:
            from .vision.camera import Camera
            from .vision.detector import YoloDetector
            self.vision = {
                "camera": Camera(int(self.cfg.get("vision.camera_index", 0))),
                "detector": YoloDetector(self.cfg.path("vision.model"),
                                         confidence=float(self.cfg.get("vision.confidence", 0.35))),
            }
            log.info("视觉模块就绪")
        except Exception as e:
            log.warning("视觉模块加载失败: %s", e)
            self.vision = None

    async def _init_geo(self) -> None:
        """IP 兜底定位：仅当用户没有口述位置时才写 IP 推断（source=geo_location）。

        用户口述"我在杭州"（source=user_location）优先，覆盖 IP 推断。
        失败不阻塞启动。
        """
        try:
            if self.store.has_source("user_location"):
                log.info("已有用户口述位置，跳过 IP 兜底")
                return
            from .geo import detect_location
            loc = await to_thread(detect_location)
            if not loc:
                log.warning("未能定位用户位置，跳过地理记忆")
                return
            country, region, city, ip = loc
            parts = [p for p in (country, region, city) if p]
            text = "用户所在位置：" + " ".join(parts)
            # 去重：删旧 geo 记忆（含早期 source="geo"）再写新
            removed = (self.store.delete_by_source("geo_location")
                       + self.store.delete_by_source("geo"))
            if removed:
                self.store.persist()
                log.info("刷新 IP 位置记忆（删除旧 %d 条）", removed)
            self.memory.add_direct(text, source="geo_location")
        except Exception as e:
            log.warning("地理定位初始化失败: %s", e)

    @staticmethod
    def _extract_location(raw: str) -> Optional[str]:
        """从口述中提取用户位置（"我在杭州"→"杭州"）。非口语短句/疑问句返回 None。"""
        t = (raw or "").strip()
        if not t or t.endswith(("？", "?", "吗", "呢", "吧", "么")):
            return None
        m = _LOCATION_RE.search(t)
        if not m:
            return None
        loc = m.group(1).strip()
        if len(loc) < 2 or loc in _PLACEHOLDER_LOCS:
            return None
        return loc

    def _set_user_location(self, loc: str) -> None:
        """用户口述位置 → 覆盖所有旧位置记忆（IP 推断 + 之前口述）。"""
        for src in ("geo_location", "geo", "user_location"):
            if self.store.delete_by_source(src):
                self.store.persist()
        self.memory.add_direct(f"用户所在位置：{loc}", source="user_location")
        log.info("已记住用户位置: %s", loc)

    def _vad(self):
        vad_path = self.cfg.resolve_path(VAD_MODEL)
        if self.cfg.get("audio.vad_enabled", True) and vad_path.exists():
            try:
                return SileroVAD(vad_path)
            except Exception as e:
                log.warning("Silero VAD 加载失败，降级能量 VAD: %s", e)
        return EnergyVAD()

    def _init_wake(self) -> None:
        """trigger=wake 时加载唤醒词检测器；失败则降级为常听（无唤醒词）。"""
        if self.trigger != "wake":
            return
        try:
            from .audio.kws import WakeWordSpotter
            self._wake_spotter = WakeWordSpotter(
                self.cfg.path("wake.model_dir"),
                self.cfg.path("wake.keywords_file"),
                num_threads=int(self.cfg.get("wake.num_threads", 2)),
            )
            log.info("唤醒词模式就绪（%s）", self.cfg.get("wake.keyword", "小桌"))
        except Exception as e:
            log.warning("唤醒词加载失败，降级为常听模式: %s", e)
            self._wake_spotter = None

    def _beep(self, dur: float = 0.15, freq: float = 880.0, sr: int = 16000) -> None:
        """唤醒提示音（短促正弦波）。listener 线程内同步播放，时长极短。"""
        if not self.speaker:
            return
        try:
            import numpy as np
            t = np.arange(int(sr * dur), dtype=np.float32) / sr
            tone = (0.35 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            self.speaker.play(tone, sr)
        except Exception as e:
            log.debug("提示音播放失败: %s", e)

    # ================= 主循环 =================
    async def run(self) -> None:
        await self.llm_server.start()
        await self._init_geo()
        # TTS 预热：触发 jieba/拼音/音素一次性初始化（省首个回答 ~0.5-1s）。
        # 同步等待，避免与首个真实合成并发竞争同一 OfflineTts 实例。
        if self.voice and self.tts:
            try:
                await to_thread(self.tts.synthesize, "你好。")
                log.info("TTS 预热完成")
            except Exception as e:
                log.warning("TTS 预热失败: %s", e)
        self._banner()
        try:
            if self.voice:
                await self._loop_voice()
            else:
                await self._loop_text()
        except KeyboardInterrupt:
            pass
        finally:
            await self.llm_server.stop()
            if self.vision:
                self.vision["camera"].release()
            self.store.persist()

    def _banner(self) -> None:
        mode = "语音" if self.voice else "键盘(无音频)"
        provider = getattr(self.llm, "provider", "local")
        print(f"\n=== Deskbot 就绪（{mode}模式 / LLM: {provider}）===")
        if self.voice:
            if self.trigger == "wake":
                print(f"喊“{self.cfg.get('wake.keyword', '小桌')}”唤醒后说话；说“退出/再见”结束。\n")
            else:
                print("直接对着麦克风说话即可；说“退出/再见”结束。\n")
        else:
            print("输入 '退出' 或 '再见' 结束。\n")

    # ---- 语音循环（常开聆听 + 追加队列）----
    async def _loop_voice(self) -> None:
        """麦克风常开：说话自动识别；回答期间继续聆听，新内容排队追加处理（不打断）。"""
        if not self.asr or not self.mic:
            log.warning("无麦克风/ASR，退回键盘模式")
            self.voice = False
            await self._loop_text()
            return
        self._seg_q: "asyncio.Queue[np.ndarray]" = asyncio.Queue()
        self._listening = True
        self._current_task: Optional[asyncio.Task] = None
        self._exit_requested = False
        self._start_listener()
        if self.trigger == "wake":
            print("\r🎙️ 待命中…（喊“小桌”唤醒；回答期间继续听）", flush=True)
        else:
            print("\r🎙️ 聆听中…（直接说话；回答期间继续听，会排队追加）", flush=True)
        try:
            while True:
                item = await self._seg_q.get()
                if item is None or self._exit_requested:
                    break
                seg, vad_ms = item
                # 抢话打断：上一轮还在处理时，新语音立即取消它
                if self.cfg.get("audio.barge_in", True) and self._current_task is not None:
                    self._barge_in()
                print(f"\r🎤 检测到语音 {len(seg)/16000:.1f}s，识别中…", flush=True)
                self._turn_timings = {"vad_ms": vad_ms, "audio_s": len(seg) / 16000}
                t0 = time.monotonic()
                try:
                    text = await to_thread(self.asr.transcribe, seg)
                except Exception as e:
                    log.warning("识别失败: %s", e)
                    continue
                finally:
                    self._turn_timings["asr_ms"] = (time.monotonic() - t0) * 1000
                if not text.strip():
                    print("\r（没听清，请再说一遍）", flush=True)
                    continue
                if _is_noise(text):
                    print(f"\r（忽略噪声：{text!r}）", flush=True)
                    continue
                print(f"你说：{text}")
                # 用可取消任务处理（不 await）：循环回到队列等下一片段，
                # 新语音到达时 _barge_in 会取消本任务实现打断。
                self._current_task = asyncio.create_task(self._handle_guard(text))
                self._current_task.add_done_callback(self._clear_task)
        finally:
            self._listening = False
            if getattr(self, "_listen_thread", None):
                self._listen_thread.join(timeout=2)

    def _barge_in(self) -> None:
        """打断：停止当前 TTS 播放并取消进行中的回答任务。"""
        if self.speaker:
            try:
                self.speaker.stop()
            except Exception:
                pass
        if self._current_task is not None and not self._current_task.done():
            self._current_task.cancel()

    async def _handle_guard(self, text: str) -> None:
        """包装 _handle：吞掉普通异常；放行打断 CancelledError；处理退出。"""
        try:
            await self._handle(text)
        except asyncio.CancelledError:
            log.info("上一轮回答被新语音打断")
            raise
        except KeyboardInterrupt:
            # "退出/再见"：设标志并投递哨兵唤醒主循环退出
            self._exit_requested = True
            if getattr(self, "_seg_q", None) is not None:
                self._seg_q.put_nowait(None)
        except Exception as e:
            log.warning("回答处理异常: %s", e)

    def _clear_task(self, task: asyncio.Task) -> None:
        if getattr(self, "_current_task", None) is task:
            self._current_task = None

    def _start_listener(self) -> None:
        """后台线程：常开麦克风 + VAD，语音片段追加进队列（不打断当前回答）。

        trigger=wake：未唤醒时只跑 KWS；命中唤醒词 → 提示音 + 打开命令窗口，
        窗口内 VAD 片段正常入队，说完锁回待命，超时回 idle。
        """
        import threading

        loop = asyncio.get_running_loop()
        segmenter = SpeechSegmenter(self._vad(),
                                    after_silence=self.cfg.get("audio.vad_after_silence", 0.8),
                                    max_seconds=self.cfg.get("audio.vad_max_seconds", 20))
        wake_mode = self.trigger == "wake" and self._wake_spotter is not None
        # 会话超时（s）：唤醒后持续可说话，静默超过该时长（且无回答在播）才回待命
        session_timeout = float(self.cfg.get("wake.conversation_timeout",
                                             self.cfg.get("wake.timeout", 30)))
        woken = False
        in_session = False
        session_deadline = 0.0

        def run() -> None:
            nonlocal woken, in_session, session_deadline
            try:
                self.mic.start()
            except Exception as e:
                log.warning("麦克风启动失败: %s", e)
                return
            while self._listening:
                block = self.mic.read_block(timeout=0.2)
                if block is None:
                    continue
                # 唤醒词模式：未唤醒 → 只跑 KWS，不产 VAD 片段
                if wake_mode and not woken:
                    try:
                        if self._wake_spotter.feed(block):
                            self._beep()
                            segmenter.reset()          # 丢弃提示音回声
                            woken = True
                            in_session = True
                            session_deadline = time.monotonic() + session_timeout
                            print("\r🔔 唤醒，请说话…", flush=True)
                    except Exception as e:
                        log.warning("KWS 处理异常: %s", e)
                    continue
                # 会话超时 → 回待命（回答播放中不超时，续期等待）
                if wake_mode and in_session and time.monotonic() > session_deadline:
                    cur = getattr(self, "_current_task", None)
                    if cur is not None and not cur.done():
                        session_deadline = time.monotonic() + session_timeout
                        continue
                    in_session = False
                    woken = False
                    segmenter.reset()
                    print("\r⏰ 长时间未说话，重新待命…", flush=True)
                    continue
                try:
                    seg = segmenter.feed(block)
                except Exception as e:
                    log.warning("VAD 处理异常: %s", e)
                    continue
                if seg is not None and seg.size >= 1600:  # >=0.1s 视为有效语音
                    loop.call_soon_threadsafe(self._seg_q.put_nowait,
                                              (seg, segmenter.last_seg_ms))
                    if wake_mode:
                        # 说话续期：持续对话不需重新唤醒
                        session_deadline = time.monotonic() + session_timeout
            self.mic.stop()

        self._listen_thread = threading.Thread(target=run, daemon=True)
        self._listen_thread.start()

    # ---- 键盘循环 ----
    async def _loop_text(self) -> None:
        while True:
            try:
                raw = await to_thread(input, "你：")
            except EOFError:
                break
            await self._handle(raw)

    # ================= 单轮处理 =================
    async def _handle(self, raw: str) -> None:
        raw = (raw or "").strip()
        if not raw:
            return
        if raw in ("退出", "再见", "quit", "exit", "拜拜"):
            print("再见！")
            raise KeyboardInterrupt

        # 0) 用户口述位置（"我在杭州"）→ 记录并覆盖 IP 推断，不走 LLM
        loc = self._extract_location(raw)
        if loc:
            self._set_user_location(loc)
            answer = f"好的，我记住你在{loc}。"
            await self._respond(answer, raw, feedback="", bypass_llm=True)
            return

        # 1) 意图识别（记忆 + 反馈）
        actions = self.collector.analyze(raw, "")
        memory_acts = [a for a in actions if a["type"] == "memory"]
        fb_acts = [a for a in actions if a["type"] == "feedback"]

        # 2) 记忆类输入 → 直接入库，不走 LLM（快）
        if memory_acts:
            for a in memory_acts:
                self.memory.add_direct(a["text"], source="user_request")
                self.collector.record_memory(a["text"], raw)
            answer = "好的，我记住了：" + "；".join(a["text"] for a in memory_acts)
            await self._respond(answer, raw, feedback="", bypass_llm=True)
            return

        # 3) 纯反馈输入 → 简短回应，不走 LLM
        if fb_acts and self._only_feedback(raw, fb_acts):
            fb = fb_acts[0]
            answer = ("谢谢夸奖，我会继续保持！" if fb["rating"] == "good"
                      else "抱歉，我记住了，下次会改进。")
            self.collector.record_feedback(fb["rating"], raw, answer, fb["reason"])
            await self._respond(answer, raw, feedback=fb["rating"], bypass_llm=True)
            return

        # 4) 正常问答：检索记忆 + 规则 + 视觉 → LLM
        rag_ctx = self.memory.retrieve(raw)
        rules = self.rules.format()
        vision_desc = await self._maybe_vision(raw)
        if self.voice and self.tts and self.speaker:
            # 语音场景：流式生成 + 逐句 TTS 播放（感知提速）
            answer = await self._respond_stream(raw, rag_ctx=rag_ctx, rules=rules,
                                                vision_desc=vision_desc)
        else:
            answer = await self.llm.ask(raw, rag_context=rag_ctx,
                                        learned_rules=rules, vision_desc=vision_desc)
            await self._respond(answer, raw, feedback="")
        fb = ""
        if fb_acts:
            fb = fb_acts[0]["rating"]
            self.collector.record_feedback(fb, raw, answer, fb_acts[0]["reason"])

    @staticmethod
    def _only_feedback(text: str, fb_acts: List[dict]) -> bool:
        """去掉常见反馈壳子后剩余内容很短，且不是问句 → 纯反馈。

        判定不准时返回 False（走 LLM），宁可多调一次模型也别答非所问。
        """
        t = text.strip()
        if not fb_acts or t.endswith(("？", "?")):
            return False
        for kw in ("这个回答很好", "这个回答不错", "回答很好", "回答不错",
                   "回答太棒了", "回答对了", "回答错了", "这个回答不对",
                   "回答不对", "回答错误", "很好", "真棒", "太棒了", "不错",
                   "答对了", "记住了", "错了", "不对", "错误", "谢谢", "不错不错"):
            t = t.replace(kw, "")
        return len(t.strip(" ，,。.!！~、")) <= 4

    # ---- 回应（TTS + 日志 + 记忆抽取）----
    async def _respond(self, answer: str, user_text: str,
                       feedback: str = "", bypass_llm: bool = False) -> None:
        print(f"小桌：{answer}")
        self._log_conv(user_text, answer, feedback)

        synth = None
        if self.voice and self.tts and self.speaker:
            synth = await self._speak_async(answer)

        # 记忆抽取（后台，不增加用户等待；与 TTS 合成并行）
        if not bypass_llm and self.cfg.get("rag.memory_extract_enabled", True):
            try:
                await self.memory.extract_and_store(user_text, answer)
            except Exception as e:
                log.warning("记忆抽取失败: %s", e)

        if synth is not None:
            await self._play_synth(synth[0], 0.0, 0.0)

    async def _respond_stream(self, user_text: str, rag_ctx: str = "",
                              rules: str = "", vision_desc: str = "") -> str:
        """流式回答：LLM 生成一句 → TTS 播一句（播 A 时生成 B，感知提速）。"""
        q: "asyncio.Queue[Optional[str]]" = asyncio.Queue(maxsize=4)

        async def producer() -> None:
            try:
                async for sent in self.llm.ask_stream(user_text, rag_context=rag_ctx,
                                                      learned_rules=rules,
                                                      vision_desc=vision_desc):
                    await q.put(sent)
            finally:
                await q.put(None)

        prod = asyncio.create_task(producer())
        parts: List[str] = []
        tts_total = play_total = 0.0
        print("小桌：", end="", flush=True)
        prev = None  # (合成task, 文本) 上一句，用于合成与生成并行
        try:
            while True:
                sent = await q.get()
                if sent is None:
                    break
                parts.append(sent)
                print(sent, end="", flush=True)
                # 启动本句合成（后台），同时等待上一句合成完成并播放 → 流水线并行
                cur = await self._speak_async(sent)
                if prev is not None:
                    tts_total, play_total = await self._play_synth(prev[0], tts_total, play_total)
                prev = cur
            if prev is not None:
                tts_total, play_total = await self._play_synth(prev[0], tts_total, play_total)
            print()
            answer = "".join(parts).strip()
            await prod
        finally:
            # 被打断时确保取消 LLM 流式 producer，避免后台连接泄漏
            if not prod.done():
                prod.cancel()
        self._summarize_timings(tts_total, play_total)
        self._log_conv(user_text, answer, "")
        # 记忆抽取（后台，不增加用户等待）
        if self.cfg.get("rag.memory_extract_enabled", True):
            try:
                await self.memory.extract_and_store(user_text, answer)
            except Exception as e:
                log.warning("记忆抽取失败: %s", e)
        return answer

    async def _speak_async(self, text: str) -> tuple:
        """启动 TTS 合成（后台线程），立即返回 (合成 future, 文本)。

        播放由调用方 await 合成 future 后再播，使合成与 LLM 生成并行。
        """
        synth = asyncio.create_task(to_thread(self.tts.synthesize, text))
        return synth, text

    async def _play_synth(self, synth_task, tts_total: float, play_total: float) -> tuple:
        """等合成完成并播放，返回累加后的 (tts_total, play_total)。"""
        try:
            t0 = time.monotonic()
            audio, sr = await synth_task
            tts_ms = (time.monotonic() - t0) * 1000
            t0 = time.monotonic()
            await to_thread(self.speaker.play, audio, sr)
            play_ms = (time.monotonic() - t0) * 1000
            return tts_total + tts_ms, play_total + play_ms
        except Exception as e:
            log.warning("语音播报失败: %s", e)
            return tts_total, play_total

    def _summarize_timings(self, tts_ms: float, play_ms: float) -> None:
        """打印本轮语音流水线各阶段耗时（VAD断句→ASR→LLM→TTS→发声）。"""
        t = self._turn_timings
        m = getattr(self.llm, "last_metrics", None) or {}
        tok = m.get("completion_tokens")
        tok_s = f"{int(tok)}" if tok is not None else "?"
        s = lambda ms: f"{ms / 1000:.2f}"  # noqa: E731
        print(f"⏱  VAD断句 {s(t.get('vad_ms', 0))}s → ASR {s(t.get('asr_ms', 0))}s → "
              f"LLM首字 {s(m.get('ttft_ms', 0))}s / 生成 {s(m.get('gen_ms', 0))}s({tok_s}tok) → "
              f"TTS {s(tts_ms)}s / 播放 {s(play_ms)}s", flush=True)

    def _log_conv(self, user_text: str, answer: str, feedback: str = "") -> None:
        from .utils.state import append_jsonl, now_ts
        append_jsonl(self.conv_log_path, {
            "ts": now_ts(), "user": user_text, "answer": answer,
            "feedback": feedback or None,
            "duration_ms": int((time.time() - self._start_ts) * 1000),
        })

    # ---- 视觉 ----
    async def _maybe_vision(self, raw: str) -> str:
        if not self.vision:
            return ""
        if not self.cfg.get("vision.inject_on_ask", True):
            return ""
        if not any(k in raw for k in ("桌面", "桌上", "看到", "看见", "周围", "有什么")):
            return ""
        try:
            img = await to_thread(self.vision["camera"].capture)
            if img is None:
                return ""
            return await to_thread(self.vision["detector"].describe, img)
        except Exception as e:
            log.warning("视觉检测失败: %s", e)
            return ""


# ================= 子命令 =================
async def _smoke(cfg: Config, voice: bool) -> None:
    bot = DeskBot(cfg, voice=False)
    await bot.llm_server.start()
    print("[1/4] LLM 调用…")
    r = await bot.llm.complete("你是测试助手", "用一句话回答：1+1=？", max_tokens=30)
    print(f"  -> {r}")
    print("[2/4] 记忆写入/检索…")
    bot.memory.add_direct("冒烟测试记忆", source="smoke")
    ctx = bot.memory.retrieve("冒烟测试")
    print(f"  -> 检索: {ctx or '(空)'}")
    print("[3/4] 反馈记录…")
    bot.collector.record_feedback("good", "测试", "好的")
    print("  -> 已写入 feedback.jsonl")
    print("[4/4] 数据集导出…")
    from .optimize.dataset import export_training_set
    n = export_training_set(cfg)
    print(f"  -> 导出 {n} 条")
    await bot.llm_server.stop()
    print("\n冒烟测试通过 ✓")


async def _say(cfg: Config, text: str) -> None:
    bot = DeskBot(cfg, voice=True)
    if not bot.tts or not bot.speaker:
        print("TTS/扬声器不可用")
        return
    await bot.llm_server.start()
    audio, sr = await to_thread(bot.tts.synthesize, text)
    await to_thread(bot.speaker.play, audio, sr)
    await bot.llm_server.stop()
    print(f"已播报: {text}")


# ================= 入口 =================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deskbot", description="Deskbot 桌面语音机器人")
    p.add_argument("--config", default=None, help="配置文件路径（默认 config.yaml）")
    p.add_argument("--voice", action="store_true", help="语音问答模式")
    p.add_argument("--text", action="store_true", help="键盘问答模式")
    p.add_argument("--smoke", action="store_true", help="冒烟测试")
    p.add_argument("--eval", action="store_true", help="跑评估集")
    p.add_argument("--tune", action="store_true", help="规则自改进")
    p.add_argument("--export", action="store_true", help="导出训练集")
    p.add_argument("--memory-add", default=None, metavar="TEXT", help="直接记住一句话")
    p.add_argument("--say", default=None, metavar="TEXT", help="合成并播报一句话")
    p.add_argument("--dialogue", action="store_true", help="自动化对话考核（文字层）")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg)

    if args.smoke:
        asyncio.run(_smoke(cfg, voice=args.voice))
        return
    if args.eval:
        bot = DeskBot(cfg, voice=False)
        asyncio.run(_run_eval(bot))
        return
    if args.dialogue:
        asyncio.run(_run_dialogue(cfg))
        return
    if args.tune:
        bot = DeskBot(cfg, voice=False)
        asyncio.run(_run_tune(bot))
        return
    if args.export:
        from .optimize.dataset import export_training_set
        n = export_training_set(cfg)
        print(f"已导出 {n} 条训练样本")
        return
    if args.memory_add:
        bot = DeskBot(cfg, voice=False)
        bot.memory.add_direct(args.memory_add, source="cli")
        print(f"已记住: {args.memory_add}")
        return
    if args.say:
        asyncio.run(_say(cfg, args.say))
        return

    voice = args.voice and not args.text
    if args.text:
        voice = False
    bot = DeskBot(cfg, voice=voice)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n已退出。")


async def _run_eval(bot) -> None:
    await bot.llm_server.start()
    try:
        report = await bot.eval_runner.run(learned_rules=bot.rules.format())
        for d in report["details"]:
            mark = "✓" if d["passed"] else "✗"
            print(f"{mark} {d['q']} → {d['answer']}")
        print(f"\n总分: {report['passed']}/{report['total']} ({report['score']:.0%})"
              f"，平均长度 {report['avg_len']} 字，基线 {report['baseline_passed']}")
    finally:
        await bot.llm_server.stop()


async def _run_tune(bot) -> None:
    await bot.llm_server.start()
    try:
        report = await bot.tuner.improve_once(bot.collector)
        print(f"新增规则 {report['added']} 条，当前共 {report['rules_total']} 条，"
              f"回滚: {report.get('rolled_back', False)}")
    finally:
        await bot.llm_server.stop()


async def _run_dialogue(cfg) -> None:
    """自动化对话考核（文字层，不依赖音频/TTS）。"""
    from .optimize.dialogue import run_dialogue
    report = await run_dialogue(cfg)
    for d in report["details"]:
        mark = "✓" if d["passed"] else "✗"
        scope = f" [{d.get('notes')}]" if d.get("notes") else ""
        print(f"{mark} {d['name']}: {d.get('expected', '')}{scope}")
    print(f"\n对话考核: {report['passed']}/{report['total']} 通过 ({report['score']:.0%})"
          f"，基线 {report['baseline_passed']}")


if __name__ == "__main__":
    main()
