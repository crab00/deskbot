"""训练数据集导出。

把【对话日志 + 正面反馈对 + 记忆】导出为 chat 格式 JSONL（messages 字段），
供离机 QLoRA 微调使用（scripts/fine_tune.sh 读取）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..utils.config import Config
from ..utils.logging_setup import get_logger
from ..utils.state import read_jsonl, write_json

log = get_logger("optimize.dataset")


def _system_prompt(cfg: Config) -> str:
    return str(cfg.get("llm.system_prompt", ""))


def _conversation_records(cfg: Config) -> List[Dict]:
    """读取 data/logs/conv_*.jsonl 的全部对话记录。"""
    logs_dir = cfg.path("paths.logs")
    out: List[Dict] = []
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("conv_*.jsonl")):
            out.extend(read_jsonl(p))
    return out


def export_training_set(cfg: Config, out_path: Optional[Path] = None,
                        include_unrated: bool = True,
                        include_feedback_only: bool = True) -> int:
    """导出训练集，返回写入条数。默认路径 data/datasets/train.jsonl。"""
    out_path = out_path or cfg.path("paths.datasets") / "train.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys_prompt = _system_prompt(cfg)

    examples: List[Dict] = []
    seen = set()

    def add(user: str, answer: str, src: str) -> None:
        key = (user.strip(), answer.strip())
        if not user.strip() or not answer.strip() or key in seen:
            return
        seen.add(key)
        examples.append({
            "source": src,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
        })

    # 1) 对话日志
    for rec in _conversation_records(cfg):
        rating = rec.get("feedback")
        if rating == "good" or include_unrated:
            add(rec.get("user", ""), rec.get("answer", ""), f"conv:{rec.get('ts', '')}")

    # 2) 正面反馈对（feedback.jsonl 中 rating=good 且带完整问答）
    fb_path = cfg.path("paths.feedback") / "feedback.jsonl"
    for fb in read_jsonl(fb_path):
        if fb.get("type") == "feedback" and fb.get("rating") == "good":
            add(fb.get("user", ""), fb.get("answer", ""), f"feedback:{fb.get('ts', '')}")

    # 写训练样本（纯 chat 格式 JSONL，每行一条）
    _write_chat_jsonl(out_path, examples)
    # 写清单（供 fine_tune.sh 读取 prompt 与规模）
    write_json(out_path.with_suffix(".meta.json"),
               {"prompt": sys_prompt, "num_examples": len(examples)})
    log.info("导出训练集 %d 条 → %s", len(examples), out_path)
    return len(examples)


def _write_chat_jsonl(path: Path, examples: List[Dict]) -> None:
    import json
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
