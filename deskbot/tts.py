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
        ddict = str(dict_dir) if dict_dir.is_dir() else ""
        ddir = str(data_dir) if data_dir.is_dir() else ""

        # Kokoro：目录含 voices.bin 即判定（VITS/Matcha/piper 均无此文件）。
        # 多说话人 boxed-voice 模型，lexicon 是逗号分隔多文件（lexicon-us-en.txt,lexicon-zh.txt），
        # 且需 espeak-ng-data（文本归一化）。中文前端用拼音+jieba 原生消歧，不走 lexicon.txt
        # 多音字词表（跳过 _resolve_lexicon）。
        voices = self.model_dir / "voices.bin"
        if voices.exists():
            if not (self.model_dir / "espeak-ng-data").is_dir():
                log.warning("Kokoro 模型缺 espeak-ng-data/，中文文本归一化可能异常")
            # 只取 us-en + zh：中文前端用拼音+zh 词表；gb-en 与 us-en 大量重复且无中文词，
            # 一起传会触发 C++ 端 "Duplicated word" 去重日志噪音。
            lex_list = [str(p) for p in sorted(self.model_dir.glob("lexicon-*-en.txt"))
                        if "gb-" not in p.name] + [str(self.model_dir / "lexicon-zh.txt")]
            lex_str = ",".join(p for p in lex_list if Path(p).exists())
            kokoro_cfg = sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(model), voices=str(voices), tokens=str(tokens),
                lexicon=lex_str, data_dir=ddir, dict_dir=ddict)
            model_cfg = sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro_cfg, num_threads=num_threads)
            log.info("加载 Kokoro（%s，lexicon=%s）", self.model_dir, lex_str)
        else:
            # VITS/Matcha/piper 用单个 lexicon.txt + 多音字词表扩展
            lex = self._resolve_lexicon()
            if "steps" in model.name or "matcha" in model.name.lower():
                # Matcha-TTS：acoustic model 加独立 vocoder（vocos/hifigan，官网单独下载）。
                # 目录内约定 vocoder-*.onnx；缺 vocoder 会 TypeError，显式报错更清晰。
                # （注意 OfflineTtsModelConfig 的字段是 vits= / matcha=，不是 model=）
                vocoders = sorted(self.model_dir.glob("vocoder-*.onnx"))
                if not vocoders:
                    log.error("Matcha 模型缺 vocoder-*.onnx，请下载（如 vocos-22khz-univ.onnx）放入模型目录")
                log.info("加载 Matcha-TTS（%s, vocoder=%s）", model.name,
                         vocoders[0].name if vocoders else "缺失!")
                # MatchaModelConfig 的绑定签名随轮次不同：某些 aarch64 轮（Nano cp38）
                # 把 lexicon(带默认值) 排在 tokens(必填) 之前，按关键字传 tokens= 会被拒；
                # Mac arm64 轮接受关键字。try/except 双保险，位置参数按 C++ 声明顺序。
                try:
                    mcfg = sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model=str(model), vocoder=str(vocoders[0]) if vocoders else "",
                        tokens=str(tokens), lexicon=lex, dict_dir=ddict, data_dir=ddir)
                except TypeError:
                    # C++ 顺序：acoustic_model, vocoder, lexicon, tokens, data_dir, dict_dir
                    mcfg = sherpa_onnx.OfflineTtsMatchaModelConfig(
                        str(model), str(vocoders[0]) if vocoders else "",
                        lex, str(tokens), ddir, ddict)
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
