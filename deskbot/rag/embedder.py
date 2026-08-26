"""文本嵌入。

默认复用本地 llama-server 的 /v1/embeddings 接口（零额外模型/依赖），
用同一份 Qwen 大模型产出 1536 维嵌入做 RAG 检索。
server 未就绪/调用失败时降级为确定性 hash 嵌入（仅保证能跑、能测，
会在日志里明确警告）。
"""
from __future__ import annotations

from typing import List

import httpx
import numpy as np

from ..utils.config import Config
from ..utils.logging_setup import get_logger

log = get_logger("rag.embedder")

DEFAULT_DIM = 1536  # Qwen2.5-1.5B hidden size


class HashingEmbedder:
    """确定性字符哈希嵌入（降级用，dim 可配）。"""

    def __init__(self, dim: int = DEFAULT_DIM):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t:
                h = (ord(ch) * 2654435761) & 0xFFFFFFFF
                idx = h % self.dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                out[i, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class LlamaEmbedder:
    """通过 llama-server /v1/embeddings 接口嵌入（OpenAI 兼容）。"""

    def __init__(self, base_url: str, dim: int = DEFAULT_DIM):
        self.base_url = base_url
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # 每次调用批量嵌入（llama.cpp 支持批量 input）
        # trust_env=False：本机 llama-server 不走系统代理
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            r = client.post(f"{self.base_url}/v1/embeddings",
                            json={"input": texts, "model": "deskbot"})
            r.raise_for_status()
            data = r.json()
        rows = sorted(data["data"], key=lambda x: x["index"])
        out = np.array([row["embedding"] for row in rows], dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class Embedder:
    """统一入口：优先 llama-server 嵌入，退化到 hash。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dim = int(cfg.get("rag.embed_dim", DEFAULT_DIM))
        self._impl: object = None
        self._init()

    def _init(self) -> None:
        host = self.cfg.get("llm.host", "127.0.0.1")
        port = self.cfg.get("llm.port", 8080)
        try:
            self._impl = LlamaEmbedder(f"http://{host}:{port}", dim=self.dim)
            log.info("嵌入器就绪：复用 llama-server /v1/embeddings (dim=%d)", self.dim)
        except Exception as e:  # pragma: no cover
            log.warning("嵌入器初始化失败，降级 hash: %s", e)
            self._impl = HashingEmbedder(self.dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        try:
            return self._impl.embed(texts)
        except Exception as e:
            log.warning("llama 嵌入失败(%s)，降级 hash 嵌入", e)
            return HashingEmbedder(self.dim).embed(texts)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
