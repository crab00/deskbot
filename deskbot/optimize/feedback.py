"""偏好反馈采集。

识别用户输入中的反馈意图（语音口令/文字），产出两类动作：
  - memory   ：用户要求记住某件事 → 直接写入记忆库
  - feedback ：对上一轮回答的评价（good/bad）→ 供规则自改进与数据集使用
所有反馈落盘到 data/feedback/feedback.jsonl。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..utils.config import Config
from ..utils.logging_setup import get_logger
from ..utils.state import append_jsonl, now_ts, read_jsonl

log = get_logger("optimize.feedback")


class FeedbackCollector:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.feedback_path = cfg.path("paths.feedback") / "feedback.jsonl"
        opt = cfg.section("optimize")
        self.good_kw = opt.get("feedback_keywords_good", [])
        self.bad_kw = opt.get("feedback_keywords_bad", [])
        self.memory_kw = opt.get("memory_keywords", [])
        self._memory_re = re.compile("|".join(map(re.escape, self.memory_kw)))

    # ---- 意图识别 ----
    def analyze(self, user_text: str, answer: str) -> List[Dict]:
        """返回动作列表；每个动作 {'type': 'memory'|'feedback', ...}。"""
        if not user_text:
            return []
        actions: List[Dict] = []

        # 1) 记住类（疑问句不算，避免把问题误当记忆请求）
        m = self._memory_re.search(user_text)
        if m and not self._looks_like_question(user_text):
            text = self._extract_memory(user_text, m.end())
            if text:
                actions.append({"type": "memory", "text": text, "kw": m.group(0)})

        # 2) 评价类
        if any(kw in user_text for kw in self.good_kw):
            actions.append({"type": "feedback", "rating": "good", "reason": user_text[:120]})
        if any(kw in user_text for kw in self.bad_kw):
            actions.append({"type": "feedback", "rating": "bad", "reason": user_text[:120]})
        return actions

    @staticmethod
    def _looks_like_question(text: str) -> bool:
        t = text.strip()
        if t.endswith(("？", "?", "吗", "呢", "么", "吧")):
            return True
        return t.startswith(("你能", "你可以", "能不能", "可不可以", "会记住", "记得"))

    @staticmethod
    def _extract_memory(user_text: str, start: int) -> Optional[str]:
        text = user_text[start:].strip()
        # 去掉常见的语气/疑问尾缀
        text = re.sub(r"[？?。!！，,；;]+$", "", text)
        for suf in ("了吗", "吧", "啊", "哦", "哈", "好不好", "好的"):
            if text.endswith(suf):
                text = text[: -len(suf)]
        text = text.strip("，,。.!！?？ ")
        if len(text) < 2:
            return None
        return text

    # ---- 落盘 ----
    def record(self, record: Dict) -> None:
        record = {"ts": now_ts(), **record}
        append_jsonl(self.feedback_path, record)

    def record_feedback(self, rating: str, user: str, answer: str, reason: str = "") -> None:
        self.record({"type": "feedback", "rating": rating, "user": user,
                     "answer": answer, "reason": reason})

    def record_memory(self, text: str, user: str) -> None:
        self.record({"type": "memory", "text": text, "user": user})

    def recent(self, limit: int = 50, only: str = "") -> List[Dict]:
        rows = read_jsonl(self.feedback_path, limit=limit * 10)
        if only:
            rows = [r for r in rows if r.get("type") == only]
        return rows[-limit:]
