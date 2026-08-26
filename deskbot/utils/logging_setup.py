"""日志配置：控制台 + 可选文件输出。"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from .config import Config

_CONFIGURED = False


def setup_logging(cfg: Config) -> logging.Logger:
    """按 config 初始化根日志器，返回 deskbot 命名空间 logger。"""
    global _CONFIGURED
    level_name = (cfg.get("logging.level", "INFO") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("deskbot")
    if not _CONFIGURED:
        root.setLevel(level)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

        if cfg.get("logging.console", True):
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(fmt)
            root.addHandler(ch)

        log_file = cfg.get("logging.file")
        if log_file:
            fp = cfg.resolve_path(log_file)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(fp, encoding="utf-8")
            fh.setFormatter(fmt)
            root.addHandler(fh)
        _CONFIGURED = True
    else:
        root.setLevel(level)
    return root


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger(f"deskbot.{name}" if name else "deskbot")
