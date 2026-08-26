"""语音合成（TTS）：sherpa-onnx 离线 VITS 中文音色。

模型目录（models/tts/*/）应包含：
  - *.onnx + tokens.txt          （VITS：model.onnx；piper：zh_CN-*.onnx）
  - 可选 lexicon.txt / dict/ / *.fst —— 存在才加载，缺省不阻塞
  - piper 模型（单说话人 pinyin 词法）比 VITS 快数倍，Nano 首选
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import sherpa_onnx
    _HAS_SHERPA = True
except Exception:  # pragma: no cover
    sherpa_onnx = None
    _HAS_SHERPA = False

from .utils.logging_setup import get_logger

log = get_logger("tts")


class TTSUnavailable(Exception):
    pass


class TTS:
    def __init__(self, model_dir: Path, num_threads: int = 2,
                 speaker_id: int = 0, speed: float = 1.0):
        if not _HAS_SHERPA:
            raise TTSUnavailable("未安装 sherpa-onnx")
        self.model_dir = Path(model_dir)
        self.speaker_id = speaker_id
        self.speed = speed
        self._tts = self._build(num_threads)

    def _build(self, num_threads: int):
        tokens = self.model_dir / "tokens.txt"
        models = sorted(self.model_dir.glob("*.onnx"))
        if not models or not tokens.exists():
            raise TTSUnavailable(f"TTS 模型不完整: {self.model_dir} 缺 *.onnx 或 tokens.txt")
        model = models[0]

        # 可选的 lexicon / dict / FST。VITS/piper 加载 dict（jieba 分词）；piper 不需 dict。
        # FST 规则在 Nano CPU 上前处理极慢（日期带FST 11s vs 不带 3.7s）→ 不加载。
        lexicon = self.model_dir / "lexicon.txt"
        dict_dir = self.model_dir / "dict"
        data_dir = self.model_dir / "espeak-ng-data"
        lex = str(lexicon) if lexicon.exists() else ""
        ddict = str(dict_dir) if dict_dir.is_dir() else ""
        ddir = str(data_dir) if data_dir.is_dir() else ""

        # Matcha-TTS：文件名形如 model-steps-3.onnx，用 OfflineTtsMatchaModelConfig
        # （注意 OfflineTtsModelConfig 的字段是 vits= / matcha=，不是 model=）
        if "steps" in model.name or "matcha" in model.name.lower():
            log.info("加载 Matcha-TTS（%s）", model.name)
            mcfg = sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(model), tokens=str(tokens),
                lexicon=lex, dict_dir=ddict, data_dir=ddir)
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(matcha=mcfg, num_threads=num_threads)
        else:
            vits_cfg = sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(model), tokens=str(tokens),
                lexicon=lex, dict_dir=ddict, data_dir=ddir)
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(vits=vits_cfg, num_threads=num_threads)

        try:
            config = sherpa_onnx.OfflineTtsConfig(model=model_cfg)
        except TypeError:
            config = sherpa_onnx.OfflineTtsConfig(model=model_cfg)

        log.info("加载 TTS（%s → %s）", self.model_dir, model.name)
        return sherpa_onnx.OfflineTts(config)

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """合成中文语音，返回 (float32 波形, 采样率)。"""
        if not text:
            return np.zeros(0, dtype=np.float32), 0
        audio = self._tts.generate(text, sid=self.speaker_id, speed=self.speed)
        samples = audio.samples if hasattr(audio, "samples") else audio[0]
        sr = audio.sample_rate if hasattr(audio, "sample_rate") else audio[1]
        return np.asarray(samples, dtype=np.float32).reshape(-1), int(sr)

    def __call__(self, text: str) -> Tuple[np.ndarray, int]:
        return self.synthesize(text)
