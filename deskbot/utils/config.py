"""配置加载。

约定：所有相对路径均以「项目根目录」为基准（即 config.yaml 所在目录的上级，
等价于 paths.root 指向的目录）。路径解析集中在这里，避免各处重复拼接。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # deskbot/../ → 项目根


class Config:
    """yaml 配置 + 路径解析。"""

    def __init__(self, data: Dict[str, Any], root: Path = PROJECT_ROOT):
        self._data = data
        self._root = root

    # ---- 便捷访问 ----
    def get(self, key: str, default: Any = None) -> Any:
        """点分路径取值，如 'llm.port'。"""
        cur: Any = self._data
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def section(self, name: str) -> Dict[str, Any]:
        return self._data.get(name, {}) or {}

    def path(self, key: str) -> Path:
        """取一个『相对项目根的路径』并解析为绝对路径。"""
        p = self.get(key)
        if p is None:
            raise KeyError(f"配置项 {key} 不存在")
        return self.resolve_path(str(p))

    def resolve_path(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute():
            return p
        return (self._root / p).resolve()

    # ---- 数据目录 ----
    @property
    def data_dir(self) -> Path:
        return self.path("paths.logs").parent

    # ---- 序列化 ----
    def to_dict(self) -> Dict[str, Any]:
        return self._data

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        cur = self._data
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value


def load_config(path: Optional[os.PathLike] = None) -> Config:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(data, root=cfg_path.resolve().parent)
