"""麦克风采集。

底层用 sounddevice（PortAudio）。无麦克风的环境（如纯文本测试/无 USB 声卡）
会抛出 AudioUnavailable，上层可优雅降级到键盘输入。
"""
from __future__ import annotations

import queue
import threading
from typing import List, Optional

import numpy as np

try:
    import sounddevice as sd
    _HAS_SD = True
except Exception:  # pragma: no cover
    sd = None
    _HAS_SD = False


class AudioUnavailable(Exception):
    """麦克风不可用（未装 sounddevice / 无设备 / 无权限）。"""


class Mic:
    """阻塞式采集：start() 后在后台线程收块，stop() 停止。"""

    def __init__(self, device: Optional[str] = None,
                 sample_rate: int = 16000, channels: int = 1,
                 blocksize: int = 1600):
        if not _HAS_SD:
            raise AudioUnavailable("未安装 sounddevice（pip install sounddevice）")
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize  # 0.1s @16k
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    def _probe_default(self) -> int:
        """探测可用的默认输入设备。"""
        try:
            dev = sd.query_devices(kind="input")
            return dev["index"] if isinstance(dev, dict) else dev
        except Exception:
            raise AudioUnavailable("无可用麦克风设备")

    def start(self) -> None:
        if self._stream:
            return
        try:
            device_id = self.device or self._probe_default()
            self._stream = sd.InputStream(
                device=device_id,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=self.blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            raise AudioUnavailable(f"打开麦克风失败: {e}") from e

    def _callback(self, indata, frames, time_info, status):
        self._queue.put(indata[:, 0].copy())

    def read_block(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """取一块音频；超时返回 None。"""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    @staticmethod
    def list_devices() -> List[str]:
        if not _HAS_SD:
            return []
        try:
            return [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
        except Exception:
            return []
