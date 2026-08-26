"""扬声器播放（TTS 结果）与打断（barge-in）支持。

用 sounddevice 的非阻塞 play/stop；play 跑在后台线程里，主循环不被阻塞。
任何时刻调用 stop() 会立即打断当前播放 —— 用户开始说话时用它抢麦。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import sounddevice as sd
    _HAS_SD = True
except Exception:  # pragma: no cover
    sd = None
    _HAS_SD = False

from ..utils.logging_setup import get_logger

log = get_logger("audio.speaker")


class AudioUnavailable(Exception):
    pass


class Speaker:
    def __init__(self, device: Optional[str] = None, sample_rate: int = 24000):
        self.device = device
        self.sample_rate = sample_rate
        self._playing = False

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """播放一段音频（阻塞当前线程直到播完或被打断）。"""
        if not _HAS_SD:
            raise AudioUnavailable("未安装 sounddevice")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        try:
            self._playing = True
            sd.play(audio, samplerate=int(sample_rate), device=self.device)
            sd.wait()
        except Exception as e:  # 含被打断时的 AbortError
            log.debug("播放结束/被打断: %s", e)
        finally:
            self._playing = False

    def stop(self) -> None:
        """打断当前播放。"""
        if _HAS_SD and self._playing:
            try:
                sd.stop()
            except Exception:
                pass
            self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing
