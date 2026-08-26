"""语音活动检测（VAD）。

优先使用 Silero VAD（onnxruntime 加载 models/vad/silero_vad.onnx），
模型缺失时自动降级为基于能量的简单 VAD —— 保证任何环境都能跑。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..utils.logging_setup import get_logger

log = get_logger("audio.vad")

try:
    import onnxruntime as ort
    _HAS_ORT = True
    # 压掉模型加载时的 initializer 清理警告（Silero 有 ~286 个未用 STFT 节点，
    # 每次会话创建都刷屏）。全局设置，对 ASR 的 sherpa-onnx 会话同样生效。
    try:
        ort.set_default_logger_severity(3)
    except Exception:  # pragma: no cover
        pass
except Exception:  # pragma: no cover
    ort = None
    _HAS_ORT = False


class EnergyVAD:
    """能量阈值 VAD（兜底，无需任何模型文件）。"""

    def __init__(self, threshold_db: float = -35.0, sample_rate: int = 16000):
        self.threshold = 10.0 ** (threshold_db / 20.0)
        self.sample_rate = sample_rate

    def frame_speech(self, frame: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(frame.astype(np.float32) ** 2))) if frame.size else 0.0
        return rms > self.threshold

    def reset(self) -> None:
        pass


class SileroVAD:
    """Silero VAD（onnx）。samples 需为 16k float32 mono。

    自动适配两种常见导出格式：
      - sherpa 版：输入 [x(1,512), h(2,1,64), c(2,1,64)]，输出 [prob, new_h, new_c]
      - 官方版  ：输入 [input, state(2,1,128)]，部分 v5 还要求 [sr]
    """

    def __init__(self, model_path: Path, threshold: float = 0.5, sample_rate: int = 16000):
        if not _HAS_ORT:
            raise RuntimeError("未安装 onnxruntime")
        so = ort.SessionOptions()
        so.log_severity_level = 3  # 0=verbose..3=error：压掉 STFT initializer 清理警告刷屏
        self.sess = ort.InferenceSession(str(model_path), sess_options=so,
                                         providers=["CPUExecutionProvider"])
        self._in_names = [i.name for i in self.sess.get_inputs()]
        if "x" in self._in_names and "h" in self._in_names:
            self._fmt = "sherpa"
        else:
            self._fmt = "official"
            ndim = 3
            for i in self.sess.get_inputs():
                if i.name == "input":
                    ndim = len(i.shape)
            self._input_ndim = ndim if ndim in (2, 3) else 3
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.reset()
        log.info("Silero VAD 就绪（格式=%s, 输入=%s）", self._fmt, self._in_names)

    def reset(self) -> None:
        if self._fmt == "sherpa":
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        else:
            # 官方 v4/v5 状态：(2, 1, 128) 的 [h; c]
            self._state = np.zeros((2, 1, 128), dtype=np.float32)

    @staticmethod
    def _pad512(x: np.ndarray) -> np.ndarray:
        if x.shape[0] == 512:
            return x
        if x.shape[0] < 512:
            return np.pad(x, (0, 512 - x.shape[0]))
        return x[:512]

    def frame_speech(self, frame: np.ndarray) -> bool:
        """对一块（≤512 样本）返回是否语音。"""
        if frame.size == 0:
            return False
        x = self._pad512(frame).astype(np.float32)
        try:
            if self._fmt == "sherpa":
                prob, nh, nc = self.sess.run(None, {"x": x[None, :], "h": self._h, "c": self._c})
                self._h, self._c = np.asarray(nh), np.asarray(nc)
            else:
                if self._input_ndim == 3:
                    feed = {"input": x[None, :, None], "state": self._state}
                else:
                    feed = {"input": x[None, :], "state": self._state}
                if "sr" in self._in_names:
                    feed["sr"] = np.asarray(self.sample_rate, dtype=np.int64)
                prob, state = self.sess.run(None, feed)
                self._state = np.asarray(state)
            return float(np.asarray(prob).reshape(-1)[0]) > self.threshold
        except Exception:
            return False


class SpeechSegmenter:
    """把音频块流切分成完整的语音片段。

    状态机：静音 → 检测到语音开始缓冲 → 尾部静音超过 after_silence 或超
    过 max_seconds 则结束一个片段。前导静音不保留。
    """

    def __init__(self, vad, sample_rate: int = 16000, chunk_size: int = 512,
                 after_silence: float = 0.8, max_seconds: float = 20.0):
        self.vad = vad
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.after_silence = int(after_silence * sample_rate / chunk_size)
        self.max_frames = int(max_seconds * sample_rate / chunk_size)
        self._frames_buf: List[np.ndarray] = []   # 待切帧的原始块
        self._speech_start_ts: Optional[float] = None
        self.last_seg_ms: float = 0.0            # 最近一个片段：从检测到语音到断句完成的时长
        self.reset()

    def reset(self) -> None:
        self._frames_buf = []
        self._speech_buf: List[np.ndarray] = []
        self._silence = 0
        self._frames = 0
        self._in_speech = False
        self._speech_start_ts = None
        if hasattr(self.vad, "reset"):
            self.vad.reset()

    @property
    def in_speech(self) -> bool:
        """是否正在说话中（供上层判断超时）。"""
        return self._in_speech

    def _take_frame(self) -> Optional[np.ndarray]:
        """从原始块缓冲中切出一个固定大小帧（不足则截断后返回）。"""
        if not self._frames_buf:
            return None
        # 简单起见：把缓冲拼接后取头部 chunk_size
        acc = np.concatenate(self._frames_buf)
        if acc.size < self.chunk_size:
            self._frames_buf = [acc]
            return None
        frame, rest = acc[:self.chunk_size], acc[self.chunk_size:]
        self._frames_buf = [rest] if rest.size else []
        return frame

    def feed(self, block: np.ndarray) -> Optional[np.ndarray]:
        """喂入一块音频；若恰好完成一个语音片段则返回该片段（否则 None）。"""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None
        self._frames_buf.append(block)

        while True:
            frame = self._take_frame()
            if frame is None:
                break
            if frame.size < self.chunk_size:
                frame = np.pad(frame, (0, self.chunk_size - frame.size))
            speech = self.vad.frame_speech(frame)
            self._frames += 1

            if speech:
                if not self._in_speech:
                    self._speech_start_ts = time.monotonic()
                self._in_speech = True
                self._silence = 0
                self._speech_buf.append(frame)
            elif self._in_speech:
                self._silence += 1
                self._speech_buf.append(frame)
                if self._silence >= self.after_silence or self._frames >= self.max_frames:
                    return self._finish()
        return None

    def _finish(self) -> np.ndarray:
        if self._speech_start_ts is not None:
            self.last_seg_ms = (time.monotonic() - self._speech_start_ts) * 1000
        else:
            self.last_seg_ms = 0.0
        self._speech_start_ts = None
        seg = np.concatenate(self._speech_buf) if self._speech_buf else np.zeros(0, dtype=np.float32)
        self._speech_buf = []
        self._silence = 0
        self._frames = 0
        self._in_speech = False
        return seg
