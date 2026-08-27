"""TTS 多音字静态消歧：扩展 lexicon.txt 让 sherpa-onnx VITS 读对多音字。

背景：sherpa-onnx 的 text→phoneme 在 C++ 内部：cppjieba 分词 → lexicon.txt
最大匹配查音。lexicon.txt 里单字多音字只有默认读音（如 还→hái、乐→lè），
而「归还/行李/音乐厅」等词若无词条，会被 jieba 拆成单字 → 用默认音读错。

方案：把这些「上下文无关」词的**正确注音词条**在 TTS 启动时追加进 lexicon，
让 jieba 匹配到整词 → 读对。运行时零 LLM 调用、零重建（sherpa 无热重载，
lexicon 在 OfflineTts 构造时固定，故必须启动时一次写入）。

注音格式：Zhuyin 符号 + 独立声调标记（ˉ ˊ ˇ ˋ ˙），与 lexicon.txt 一致。
piper（Nano）无 lexicon.txt，本模块对其自动跳过（can_support=False）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

log = logging.getLogger("tts_poly")

# 声调标记（tokens.txt 中的独立 token），校验用
_TONE_MARKS = {"ˉ", "ˊ", "ˇ", "ˋ", "˙"}

# 静态多音字词表：词 → 正确注音（Zhuyin）。
# 只放「上下文无关」的多音字词：读音不随语境变，仅因 lexicon 无词条而读错。
# 选词依据：ASR 回环验证（合成→SenseVoice 识别）确认确实读错。
# 上下文相关字（还/乐/行/长的多音靠语境）不属于本表 —— 那是 LLM 方案的边界。
HETERO: Dict[str, str] = {
    "归还": "ㄏ ㄨ ㄢ ˊ",          # 动词 huán，非副词 hái（ASR 实测：无词条读成"柜还"）
    "行李": "ㄒ ㄧ ㄥ ˊ ㄌ ㄧ ˙",   # xíngli，li 轻声（保险加词条，无害）
    # 注意：lexicon 已有"音乐 ㄧ ㄣ ˉ ㄩ ㄝ ˋ"词条，"音乐厅"自动切"音乐+厅"读对，
    # 无需单独加词条（加了反而可能干扰 jieba 分词）。同理"觉得"jieba 拆"觉+得"默认对。
}


class PolyphoneResolver:
    """静态多音字词表：TTS 启动时把正确注音追加进 lexicon。

    仅提供 lexicon 文本扩展能力，不参与运行时合成。piper 等无 lexicon 的
    模型用 can_support() 判 False，调用方跳过。
    """

    def __init__(self, entries: Optional[Dict[str, str]] = None):
        self.entries = dict(HETERO)
        if entries:
            self.entries.update(entries)

    @staticmethod
    def can_support(model_dir: Path) -> bool:
        """该模型目录是否有 lexicon.txt（VITS 注音体系）。piper 无 → False。"""
        return (Path(model_dir) / "lexicon.txt").exists()

    def valid_entries(self, tokens_txt: Path) -> Dict[str, str]:
        """过滤出注音 token 全部存在于 tokens.txt 的词条（防拼写错误）。"""
        if not tokens_txt.exists():
            log.warning("tokens.txt 缺失，多音字词表禁用")
            return {}
        token_set = {line.split()[0] for line in tokens_txt.read_text().splitlines() if line.strip()}
        ok: Dict[str, str] = {}
        for word, zhuyin in self.entries.items():
            syms = [s for s in zhuyin.split() if s]
            if all(s in token_set or s in _TONE_MARKS for s in syms):
                ok[word] = zhuyin
            else:
                bad = [s for s in syms if s not in token_set and s not in _TONE_MARKS]
                log.warning("多音字词条 %s 注音含未知 token %s，丢弃", word, bad)
        return ok

    def augment_lexicon(self, base: str, tokens_txt: Path) -> str:
        """在 lexicon 原文末尾追加缺失词条，返回扩展后文本。已存在的跳过。"""
        entries = self.valid_entries(tokens_txt)
        if not entries:
            return base
        existing = {line.split()[0] for line in base.splitlines() if line.strip()}
        add_lines = [f"{w} {z}" for w, z in entries.items() if w not in existing]
        if not add_lines:
            return base
        log.info("多音字词表追加 %d 条: %s", len(add_lines), ", ".join(e.split()[0] for e in add_lines))
        return base.rstrip("\n") + "\n" + "\n".join(add_lines) + "\n"
