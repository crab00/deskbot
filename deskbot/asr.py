"""语音识别（ASR）：sherpa-onnx 离线模型。

支持两种引擎（config asr.engine 指定）：
  - paraformer  ：两段式 Paraformer（encoder.onnx + decoder.onnx + tokens.txt）
  - sense_voice ：SenseVoice 单文件（model*.onnx + tokens.txt，中文效果好）
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

from .utils.logging_setup import get_logger

log = get_logger("asr")

SAMPLE_RATE = 16000


class ASRUnavailable(Exception):
    pass


class ASR:
    def __init__(self, model_dir: Path, engine: str = "auto", num_threads: int = 2):
        if not _HAS_SHERPA:
            raise ASRUnavailable("未安装 sherpa-onnx")
        self.model_dir = Path(model_dir)
        self.engine = engine
        self.num_threads = num_threads
        self._recognizer = self._build()

    # ---- 加载 ----
    def _build(self):
        tokens = self.model_dir / "tokens.txt"
        if not tokens.exists():
            raise ASRUnavailable(f"缺少 tokens.txt: {tokens}")

        enc, dec = self.model_dir / "encoder.onnx", self.model_dir / "decoder.onnx"
        model_file = next((self.model_dir / n for n in
                           ("model.onnx", "model.int8.onnx", "paraformer.onnx")
                           if (self.model_dir / n).exists()), None)
        engine = self.engine
        if engine == "auto":
            if enc.exists() and dec.exists():
                engine = "paraformer"
            elif model_file is not None:
                engine = "sense_voice"
            else:
                raise ASRUnavailable(f"ASR 模型不完整: {self.model_dir}")

        if engine == "paraformer":
            if not (enc.exists() and dec.exists()):
                raise ASRUnavailable("paraformer 引擎需要 encoder.onnx + decoder.onnx")
            log.info("加载两段式 Paraformer-zh（%s）", self.model_dir)
            return sherpa_onnx.OfflineRecognizer.from_paraformer(
                encoder=str(enc), decoder=str(dec),
                tokens=str(tokens), num_threads=self.num_threads,
                sample_rate=SAMPLE_RATE, feature_dim=80,
                decoding_method="greedy_search")

        if engine == "sense_voice":
            if model_file is None:
                raise ASRUnavailable("sense_voice 引擎需要 model*.onnx")
            log.info("加载 SenseVoice（%s）", model_file)
            try:
                # language="zh" 强制中文输出，避免多语种模型混出日文/英文
                # （此前出现过 "はい"、"不什は？" 等混语识别结果）
                return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model_file), tokens=str(tokens),
                    num_threads=self.num_threads,
                    sample_rate=SAMPLE_RATE, feature_dim=80,
                    use_itn=True, debug=False, language="zh")
            except TypeError:
                # 老版本参数差异（不支持 language 等参数）
                return sherpa_onnx.OfflineRecognizer.from_sense_voice(
                    model=str(model_file), tokens=str(tokens),
                    num_threads=self.num_threads)

        raise ASRUnavailable(f"未知 ASR 引擎: {engine}")

    # ---- 推理 ----
    def transcribe(self, samples: np.ndarray) -> str:
        """输入 16k float32 mono 波形，返回中文文本。"""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return ""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        return (stream.result.text or "").strip()

    def __call__(self, samples: np.ndarray) -> str:
        return self.transcribe(samples)
