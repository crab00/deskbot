"""评估（eval）：固定中文问答集打分 + 回归对比。

评估集格式（data/datasets/eval_set.jsonl，每行一条）：
  {"q": "问题", "keywords": ["期望出现的词", ...]}
keywords 为空数组表示只考核回答长度（简洁性）。判定标准：任一期望词出现即通过（keywords 为空则只看简洁性）。
每次运行结果追加到 data/datasets/eval_report.jsonl，用于前后对比与回归保护。
"""
from __future__ import annotations

from typing import Dict, List

from ..utils.config import Config
from ..utils.logging_setup import get_logger
from ..utils.state import append_jsonl, read_jsonl

log = get_logger("optimize.eval")

_EVAL_SYSTEM_TAIL = (
    "\n（评估模式）请直接回答下面的问题，不要寒暄，不要 Markdown，尽量简短。"
)


class EvalRunner:
    def __init__(self, cfg: Config, llm):
        self.cfg = cfg
        self.llm = llm
        self.eval_set_path = cfg.path("optimize.eval_set")
        self.eval_report_path = cfg.path("optimize.eval_report")

    def load_eval_set(self) -> List[Dict]:
        rows = read_jsonl(self.eval_set_path)
        if not rows:
            raise FileNotFoundError(f"评估集为空: {self.eval_set_path}")
        return rows

    def _system_prompt(self, learned_rules: str = "") -> str:
        base = str(self.cfg.get("llm.system_prompt", ""))
        if learned_rules:
            base += "\n【已学习到的规则】\n" + learned_rules
        return base + _EVAL_SYSTEM_TAIL

    async def run(self, learned_rules: str = "") -> Dict:
        items = self.load_eval_set()
        total, passed, char_sum = len(items), 0, 0
        details = []
        sys_prompt = self._system_prompt(learned_rules)
        for it in items:
            answer = await self.llm.complete(sys_prompt, it["q"], max_tokens=120)
            answer = answer.strip()
            char_sum += len(answer)
            kws = [k for k in (it.get("keywords") or []) if k]
            ok = any(k in answer for k in kws) if kws else True
            if ok:
                passed += 1
            details.append({"q": it["q"], "expected": kws, "answer": answer[:120], "passed": ok})
        avg_len = round(char_sum / max(total, 1), 1)
        baseline = self._last_passed()
        report = {
            "ts": self._ts(),
            "total": total, "passed": passed,
            "score": round(passed / max(total, 1), 3),
            "avg_len": avg_len,
            "baseline_passed": baseline,
            "details": details,
        }
        append_jsonl(self.eval_report_path, {k: v for k, v in report.items() if k != "details"})
        log.info("评估完成: %d/%d 通过 (%.0f%%), 平均长度 %.1f, 基线 %s",
                 passed, total, 100 * report["score"], avg_len, baseline)
        return report

    def _last_passed(self) -> int:
        rows = read_jsonl(self.eval_report_path)
        for r in reversed(rows):
            if "passed" in r:
                return int(r["passed"])
        return -1  # 无基线

    @staticmethod
    def _ts() -> str:
        import time
        return time.strftime("%Y-%m-%dT%H:%M:%S")
