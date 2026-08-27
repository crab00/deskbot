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
                 speaker_id: int = 0, speed: float = 1.0,
                 enable_fst: bool = False,
                 poly_entries: Optional[dict] = None):
        if not _HAS_SHERPA:
            raise TTSUnavailable("未安装 sherpa-onnx")
        self.model_dir = Path(model_dir)
        self.speaker_id = speaker_id
        self.speed = speed
        self.enable_fst = enable_fst
        self.poly_entries = poly_entries
        self._tts = self._build(num_threads)

    def _build(self, num_threads: int):
        tokens = self.model_dir / "tokens.txt"
        models = sorted(self.model_dir.glob("*.onnx"))
        if not models or not tokens.exists():
            raise TTSUnavailable(f"TTS 模型不完整: {self.model_dir} 缺 *.onnx 或 tokens.txt")
        model = models[0]

        # 可选的 lexicon / dict / FST。VITS/piper 加载 dict（jieba 分词）；piper 不需 dict。
        # FST 规则在 Nano CPU 上前处理极慢（日期带FST 11s vs 不带 3.7s）→ 不加载。
        dict_dir = self.model_dir / "dict"
        data_dir = self.model_dir / "espeak-ng-data"
        lex = self._resolve_lexicon()
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

        # FST 规则（date/number/phone/new_heteronym）——数字/日期读准。
        # Nano CPU 上前处理极慢故默认关；Mac 上 enable_fst=true 开启。
        rule_fsts = ""
        if self.enable_fst:
            fsts = [str(p) for p in sorted(self.model_dir.glob("*.fst"))]
            if fsts:
                rule_fsts = ",".join(fsts)
                log.info("加载 FST 规则: %s", rule_fsts)

        kwargs = {}
        if rule_fsts:
            kwargs["rule_fsts"] = rule_fsts
        try:
            config = sherpa_onnx.OfflineTtsConfig(model=model_cfg, **kwargs)
        except TypeError:
            # 旧版 sherpa-onnx 无 rule_fsts 参数，回退不加载 FST
            config = sherpa_onnx.OfflineTtsConfig(model=model_cfg)

        log.info("加载 TTS（%s → %s）", self.model_dir, model.name)
        return sherpa_onnx.OfflineTts(config)

    def _resolve_lexicon(self) -> str:
        """lexicon 路径解析：若配置了多音字词表且模型有 lexicon.txt，
        生成扩展后的会话级临时 lexicon 并返回其路径；否则返回原 lexicon。
        sherpa 无热重载（lexicon 在 OfflineTts 构造时固定），故启动时一次写入。
        """
        lexicon = self.model_dir / "lexicon.txt"
        if not lexicon.exists():
            return ""
        if not self.poly_entries:
            return str(lexicon)
        try:
            from .tts_poly import PolyphoneResolver
            resolver = PolyphoneResolver(self.poly_entries)
            if not resolver.can_support(self.model_dir):
                return str(lexicon)
            tokens_txt = self.model_dir / "tokens.txt"
            base = lexicon.read_text(encoding="utf-8")
            augmented = resolver.augment_lexicon(base, tokens_txt)
            if augmented == base:
                return str(lexicon)
            tmp = self.model_dir / "lexicon.session.txt"
            tmp.write_text(augmented, encoding="utf-8")
            log.info("多音字词表已扩展 lexicon → %s", tmp)
            return str(tmp)
        except Exception as e:
            log.warning("多音字词表扩展失败，用原始 lexicon: %s", e)
            return str(lexicon)

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
