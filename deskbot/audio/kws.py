"""唤醒词检测（KWS）：sherpa-onnx KeywordSpotter。

用于 trigger=wake 的免按键唤醒。中文唤醒词用 WenetSpeech KWS 模型
（sherpa-onnx-kws-zipformer-wenetspeech-3.3M），关键词由 pinyin token
定义在 keywords.txt 里（sherpa-onnx-cli text2token 生成）。

模型文件按后缀自动发现（*encoder*.onnx / *decoder*.onnx / *joiner*.onnx
+ tokens.txt），与 asr.py / tts.py 一致，不依赖精确文件名。
模型或 sherpa-onnx 缺失时抛 KwsUnavailable，上层降级为无唤醒词模式。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import sherpa_onnx
    _HAS_SHERPA = True
except Exception:  # pragma: no cover
    sherpa_onnx = None
    _HAS_SHERPA = False

from ..utils.logging_setup import get_logger

log = get_logger("audio.kws")


class KwsUnavailable(Exception):
    pass


def _find(model_dir: Path, pattern: str) -> Optional[Path]:
    hits = sorted(model_dir.glob(pattern))
    return hits[0] if hits else None


class WakeWordSpotter:
    """流式关键词检测：喂 16k float32 音频块，命中返回关键词文本。"""

    def __init__(self, model_dir: Path, keywords_file: Path,
                 num_threads: int = 2, provider: str = "cpu"):
        if not _HAS_SHERPA:
            raise KwsUnavailable("未安装 sherpa-onnx")
        self.model_dir = Path(model_dir)
        self.keywords_file = Path(keywords_file)

        tokens = self.model_dir / "tokens.txt"
        encoder = _find(self.model_dir, "*encoder*.onnx")
        decoder = _find(self.model_dir, "*decoder*.onnx")
        joiner = _find(self.model_dir, "*joiner*.onnx")
        if not (tokens.exists() and encoder and decoder and joiner):
            raise KwsUnavailable(
                f"KWS 模型不完整: {self.model_dir} 缺 encoder/decoder/joiner/tokens")
        if not self.keywords_file.exists():
            raise KwsUnavailable(f"KWS keywords 文件不存在: {self.keywords_file}")

        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=num_threads,
            keywords_file=str(self.keywords_file),
            keywords_score=1.5,          # 短词适当提权，提高命中率
            keywords_threshold=0.25,
            num_trailing_blanks=1,
            provider=provider,
        )
        self._stream = self._spotter.create_stream()
        log.info("KWS 唤醒词就绪（%s）", self.keywords_file)

    def feed(self, block: np.ndarray) -> Optional[str]:
        """喂入一块 16k float32 音频；命中唤醒词则返回关键词文本，否则 None。"""
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return None
        self._stream.accept_waveform(16000, block)
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            r = self._spotter.get_result(self._stream)
            if r:  # get_result 返回 str（关键词文本），非空即命中
                self._spotter.reset_stream(self._stream)  # 命中后必须重置
                return str(r)
        return None

    def reset(self) -> None:
        self._spotter.reset_stream(self._stream)
