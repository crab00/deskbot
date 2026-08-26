"""规则自改进（prompt_tuner）。

思路：扫描用户负面反馈 → 让 LLM 提炼一条可执行的改进规则 → 追加到
data/rules.yaml，并注入 system prompt → 跑评估集回归，若分数下降则自动
回滚。所有变更记入 data/changelog.jsonl，可手工回滚。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..utils.config import Config
from ..utils.logging_setup import get_logger
from ..utils.state import append_jsonl, now_ts, read_jsonl

log = get_logger("optimize.prompt_tuner")

_RULE_SYSTEM = (
    "你是规则提炼助手。根据用户对助手回答的负面反馈，提炼出一条具体的、"
    "可执行的改进规则，让助手以后避免同样的错误。规则用一句中文祈使句表达，"
    "不超过 40 字，例如『用户叫小明，回答时必须用『小明』称呼他』。"
    "只输出规则本身，不要解释、不要加引号。"
)


class RulesStore:
    MAX_RULES = 30

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rules_file: Path = cfg.path("optimize.rules_file")
        self.changelog_file: Path = cfg.path("optimize.changelog_file")
        self.rules: List[str] = self._load()

    def _load(self) -> List[str]:
        if not self.rules_file.exists():
            return []
        with open(self.rules_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or []
        return [str(x).strip() for x in data if str(x).strip()]

    def save(self) -> None:
        self.rules_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.rules_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.rules, f, allow_unicode=True, sort_keys=False)

    def format(self) -> str:
        if not self.rules:
            return ""
        return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(self.rules))

    def add_rule(self, rule: str, source: str) -> bool:
        rule = rule.strip().strip("\"'“”")
        if not rule or len(rule) < 4:
            return False
        for r in self.rules:  # 去重（包含或被包含）
            if rule in r or r in rule:
                return False
        self.rules.append(rule)
        if len(self.rules) > self.MAX_RULES:
            self.rules = self.rules[-self.MAX_RULES:]
        self.save()
        append_jsonl(self.changelog_file,
                     {"ts": now_ts(), "action": "add", "rule": rule, "source": source})
        log.info("新增规则: %s（来自 %s）", rule, source)
        return True

    def remove_rule(self, rule: str) -> bool:
        if rule in self.rules:
            self.rules.remove(rule)
            self.save()
            append_jsonl(self.changelog_file,
                         {"ts": now_ts(), "action": "remove", "rule": rule, "source": "manual"})
            return True
        return False

    def rollback_last(self) -> Optional[str]:
        """撤销最近一次 add 的规则（评估回归失败时用）。"""
        rows = read_jsonl(self.changelog_file)
        for row in reversed(rows):
            if row.get("action") == "add" and row.get("rule") in self.rules:
                self.rules.remove(row["rule"])
                self.save()
                append_jsonl(self.changelog_file,
                             {"ts": now_ts(), "action": "rollback", "rule": row["rule"],
                              "source": "eval_regression"})
                log.info("回滚规则: %s", row["rule"])
                return row["rule"]
        return None


class PromptTuner:
    def __init__(self, cfg: Config, llm, rules: Optional[RulesStore] = None,
                 eval_runner=None):
        self.cfg = cfg
        self.llm = llm
        self.rules = rules or RulesStore(cfg)
        self.eval_runner = eval_runner

    async def improve_once(self, collector, max_rules: int = 5) -> Dict:
        """扫描反馈并应用改进，返回报告。"""
        bads = [f for f in collector.recent(limit=30, only="feedback")
                if f.get("rating") == "bad"]
        added = 0
        for fb in bads[:max_rules]:
            rule = await self._propose_rule(fb)
            if rule and self.rules.add_rule(rule, source=f"feedback:{fb.get('ts', '')}"):
                added += 1

        report: Dict = {"added": added, "rolled_back": False, "rules_total": len(self.rules.rules)}
        if added and self.eval_runner:
            new = await self.eval_runner.run(learned_rules=self.rules.format())
            baseline = new.get("baseline_passed", 0)
            if new.get("passed", 0) < baseline:
                for _ in range(added):
                    self.rules.rollback_last()
                report["rolled_back"] = True
            report["eval"] = new
        return report

    async def _propose_rule(self, fb: Dict) -> str:
        try:
            resp = await self.llm.complete(
                _RULE_SYSTEM,
                f"用户说：{fb.get('user', '')}\n助手答：{fb.get('answer', '')}\n"
                f"用户反馈：{fb.get('reason', '')}",
                max_tokens=80,
            )
        except Exception as e:
            log.warning("规则提炼失败: %s", e)
            return ""
        return resp.strip().strip("\"'“”")
