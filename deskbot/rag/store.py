"""轻量向量库：numpy 内存 + 磁盘持久化（npz 向量 + jsonl 元数据）。

个人记忆规模通常在数千条以内，余弦相似度全量扫描毫秒级完成，
无需引入 ChromaDB 等重量级依赖（后者在 aarch64 上编译成本高）。
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..utils.logging_setup import get_logger

log = get_logger("rag.store")


class VectorStore:
    def __init__(self, path: Path, dim: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._meta: List[Dict[str, Any]] = []
        self._load()

    # ---- 持久化 ----
    def _load(self) -> None:
        vec_file, meta_file = self.path.with_suffix(".npz"), self.path.with_suffix(".jsonl")
        if vec_file.exists() and meta_file.exists():
            try:
                with np.load(vec_file, allow_pickle=True) as d:
                    vecs = d["vectors"]
                # 维度与当前模型不符（换模型后）→ 丢弃旧向量重建，避免检索崩溃
                if vecs.shape[1] != self.dim:
                    log.warning("向量库维度 %d ≠ 当前嵌入维度 %d（换模型了？），重建",
                                vecs.shape[1], self.dim)
                    self._reset()
                    return
                self._vectors = vecs
                self._meta = []
                with open(meta_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self._meta.append(json.loads(line))
                log.info("载入向量库 %s（%d 条）", self.path.name, len(self._meta))
            except Exception as e:
                log.warning("向量库载入失败，重建: %s", e)
                self._reset()

    def persist(self) -> None:
        np.savez(self.path.with_suffix(".npz"), vectors=self._vectors)
        with open(self.path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
            for m in self._meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def _reset(self) -> None:
        self._vectors = np.zeros((0, self.dim), dtype=np.float32)
        self._meta = []

    # ---- 写入 ----
    def add(self, text: str, embedding: np.ndarray,
            meta: Optional[Dict[str, Any]] = None) -> str:
        e = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        e = e / (np.linalg.norm(e) + 1e-9)
        record_id = str(uuid.uuid4())[:8]
        ts = (meta or {}).get("ts") or time.time()   # 自动补时间戳（检索时间加权用）
        self._vectors = np.vstack([self._vectors, e])
        self._meta.append({"id": record_id, "text": text,
                           **(meta or {}), "ts": ts})
        return record_id

    def add_batch(self, texts: List[str], embeddings: np.ndarray,
                  metas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        metas = metas or [None] * len(texts)
        return [self.add(t, embeddings[i], metas[i]) for i, t in enumerate(texts)]

    # ---- 检索（余弦相似度 + 时间加权）----
    def search(self, embedding: np.ndarray, top_k: int = 4,
               min_score: float = 0.0, recency_weight: float = 0.3) -> List[Dict[str, Any]]:
        """检索 top-k 记忆。

        recency_weight∈[0,1]：>0 时按记忆写入时间加权（越新越优先），
        解决"最新语音/记忆没被优先注入 LLM"的问题——相关度相近时最新记忆胜出。
        """
        if len(self._meta) == 0:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        q = q / (np.linalg.norm(q) + 1e-9)
        scores = (self._vectors @ q.T).reshape(-1).copy()
        if recency_weight > 0:
            now = time.time()
            ages = np.array([max(0.0, now - float(m.get("ts") or 0.0)) for m in self._meta])
            max_age = float(ages.max()) if ages.size else 0.0
            if max_age > 0:
                recency = 1.0 - ages / max_age          # 最新=1, 最旧≈0
                scores = scores * (1 - recency_weight) + recency * recency_weight
        idx = np.argsort(-scores)[:top_k]
        out = []
        for i in idx:
            s = float(scores[i])
            if s < min_score:
                continue
            out.append({"score": round(s, 4), "text": self._meta[i].get("text", ""),
                        "meta": {k: v for k, v in self._meta[i].items() if k not in ("id", "text")}})
        return out

    def delete(self, record_id: str) -> bool:
        for i, m in enumerate(self._meta):
            if m.get("id") == record_id:
                self._meta.pop(i)
                self._vectors = np.delete(self._vectors, i, axis=0)
                return True
        return False

    def delete_by_source(self, source: str) -> int:
        """删除所有指定来源的记忆（如 geo 定位，启动时刷新去重）。返回删除条数。"""
        n = 0
        for i in range(len(self._meta) - 1, -1, -1):
            if self._meta[i].get("source") == source:
                self._meta.pop(i)
                self._vectors = np.delete(self._vectors, i, axis=0)
                n += 1
        return n

    def has_source(self, source: str) -> bool:
        """是否已存在指定来源的记忆（用于口述位置优先于 IP 推断）。"""
        return any(m.get("source") == source for m in self._meta)

    def __len__(self) -> int:
        return len(self._meta)

    def all_texts(self) -> List[str]:
        return [m.get("text", "") for m in self._meta]
