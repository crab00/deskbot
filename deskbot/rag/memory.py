"""记忆服务：对话后抽取可记忆事实入库，提问时检索相关记忆拼入 prompt。

这是「持续优化」闭环的第一步 —— 让机器人在长期对话中记住用户。
"""
from __future__ import annotations

import json
import re
from typing import List

from ..utils.config import Config
from ..utils.logging_setup import get_logger
from .embedder import Embedder
from .store import VectorStore

log = get_logger("rag.memory")

_EXTRACT_SYSTEM = (
    "你是记忆抽取助手。从下面这段对话中提取【值得长期记住的事实】，"
    "例如用户的姓名、称呼偏好、兴趣爱好、个人物品、家庭信息、生活习惯、"
    "以及用户明确要求记住的事情。只输出一个 JSON 字符串数组，"
    '形如 ["事实1", "事实2"]。没有可记住的内容就输出 []。'
    "不要输出其他任何文字。"
)


class MemoryService:
    def __init__(self, cfg: Config, embedder: Embedder,
                 store: VectorStore, llm):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = int(cfg.get("rag.top_k", 4))
        self.max_chars = int(cfg.get("rag.max_memory_chars", 400))
        self.min_score = float(cfg.get("rag.min_score", 0.0))

    # ---- 写入 ----
    def add_direct(self, text: str, source: str = "user_request") -> None:
        """用户明确说『记住XXX』时直接入库。"""
        emb = self.embedder.embed_one(text)
        self.store.add(text, emb, {"source": source})
        self.store.persist()
        log.info("已记住: %s", text)

    async def extract_and_store(self, user_text: str, answer: str) -> List[str]:
        """对话后由 LLM 抽取事实并入库，返回入库数量对应的事实。"""
        if not self.cfg.get("rag.memory_extract_enabled", True):
            return []
        try:
            facts = await self._extract(user_text, answer)
        except Exception as e:
            log.warning("记忆抽取失败: %s", e)
            return []
        if not facts:
            return []
        for f in facts:
            emb = self.embedder.embed_one(f)
            self.store.add(f, emb, {"source": "auto_extract"})
        self.store.persist()
        log.info("抽取并记住 %d 条: %s", len(facts), facts)
        return facts

    async def _extract(self, user_text: str, answer: str) -> List[str]:
        resp = await self.llm.complete(
            _EXTRACT_SYSTEM,
            f"用户说：{user_text}\n助手答：{answer}",
            max_tokens=160,
        )
        return self._parse_json_array(resp)

    @staticmethod
    def _parse_json_array(text: str) -> List[str]:
        """从模型输出中稳健解析 JSON 字符串数组。"""
        text = text.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                if isinstance(data, list):
                    return [str(x).strip() for x in data if str(x).strip()]
            except json.JSONDecodeError:
                pass
        # 退路：抓取引号内的中文/任意字符串
        return re.findall(r'"([^"]{2,})"', text)

    # ---- 检索 ----
    def retrieve(self, query: str) -> str:
        """返回拼好的记忆上下文文本（为空串表示无相关记忆）。"""
        if len(self.store) == 0:
            return ""
        emb = self.embedder.embed_one(query)
        # recency_weight>0：最新写入的记忆优先（用户最新语音/口述应最先注入 LLM）
        results = self.store.search(emb, top_k=self.top_k, min_score=self.min_score,
                                    recency_weight=0.3)
        if not results:
            return ""
        lines, used = [], 0
        for r in results:
            t = r["text"]
            if used + len(t) > self.max_chars:
                break
            lines.append(f"- {t}")
            used += len(t)
        return "\n".join(lines)
