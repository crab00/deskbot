"""OpenAI 兼容 LLM 客户端。

支持两种提供方（config llm.provider）：
  - local    ：对接本地 llama.cpp server（host:port，无认证）
  - deepseek ：对接 DeepSeek 线上 API（api_base + Bearer key + remote_model）
二者都走 OpenAI 兼容 /v1/chat/completions，流式 SSE 格式一致。

维护滚动对话历史；每次提问可注入 RAG 检索到的记忆与学习到的规则，
最终拼进 system prompt。
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import httpx

from ..utils.config import PROJECT_ROOT, Config
from ..utils.logging_setup import get_logger

log = get_logger("llm.client")

# 断句（流式 TTS 用）：仅按句末标点（。！？；换行）切分，逗号/顿号/冒号不再切。
# 每次 TTS 合成都有固定前处理开销（jieba 分词+拼音+音素，与文本长度近似无关），
# 切 N 块 = N 倍固定开销 → 块太碎会让总 TTS 耗时虚高且持续占 CPU（Nano 卡顿主因）。
# 只按句末标点切 → 一次回答从 6-12 小块降到 2-4 大块。
_SENT_RE = re.compile(r"[^。！？!?；\n]*[。！？!?；\n]")
# 无标点但攒够此字数 → 硬切一块，避免整段无标点久等（可用 llm.chunk_max 覆盖）
_CHUNK_MAX = 24
# 首块用小阈值：第一个非空块尽快切出让 TTS 尽早开口，之后用大阈值合并摊薄开销
_FIRST_CHUNK_MAX = 10

# DeepSeek 本地模型（Qwen3）默认开 thinking，工具类调用需 /no_think 关闭
_NO_THINK = "/no_think"


class LlmClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.provider = str(cfg.get("llm.provider", "local")).lower()
        self.max_tokens = int(cfg.get("llm.max_tokens", 256))
        self.temperature = float(cfg.get("llm.temperature", 0.7))
        self.system_prompt = str(cfg.get("llm.system_prompt", ""))
        self.max_history = int(cfg.get("llm.max_history", 10))
        self.chunk_max = int(cfg.get("llm.chunk_max", _CHUNK_MAX))  # 无标点硬切阈值
        self.first_chunk_max = int(cfg.get("llm.first_chunk_max", _FIRST_CHUNK_MAX))  # 首块小阈值
        self.history: List[Dict[str, str]] = []
        # 最近一次生成的分段耗时统计（ask_stream 写入，供上层展示）
        self.last_metrics: Dict[str, Optional[float]] = {}
        self._init_endpoint(cfg)

    @staticmethod
    def _read_env_file(path: Optional[os.PathLike], name: str) -> str:
        """从 .env 文件读取单个键值（轻量解析，无 python-dotenv 依赖）。"""
        try:
            p = Path(path)
            if not p.exists():
                return ""
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() == name:
                    return v.strip().strip('"').strip("'")
        except Exception:
            return ""
        return ""

    def _init_endpoint(self, cfg: Config) -> None:
        if self.provider == "deepseek":
            self.base_url = str(cfg.get("llm.api_base", "https://api.deepseek.com")).rstrip("/")
            env_name = str(cfg.get("llm.api_key_env", "DEEPSEEK_API_KEY"))
            # key 优先读环境变量，其次项目根 .env 文件（部署机常放 ~/deskbot/.env）
            key = os.environ.get(env_name, "") or self._read_env_file(PROJECT_ROOT / ".env", env_name)
            if not key:
                raise RuntimeError(
                    f"LLM provider=deepseek 但未找到 API key：环境变量 {env_name} 或 {PROJECT_ROOT}/.env "
                    "均未设置。请 export DEEPSEEK_API_KEY=sk-xxx 或写入 .env")
            self.headers = {"Authorization": f"Bearer {key}"}
            self.model = str(cfg.get("llm.remote_model", "deepseek-chat"))
            log.info("LLM 提供方: deepseek (%s)", self.model)
        else:
            self.base_url = f"http://{cfg.get('llm.host', '127.0.0.1')}:{cfg.get('llm.port', 8080)}"
            self.headers = {}
            self.model = "deskbot"
            log.info("LLM 提供方: local llama-server (%s)", self.base_url)

    def reset(self) -> None:
        self.history = []

    # ---- 上下文拼装 ----
    @staticmethod
    def _today_line() -> str:
        import time
        now = time.localtime()
        week = "一二三四五六日"[now.tm_wday]
        return f"今天是{now.tm_year}年{now.tm_mon}月{now.tm_mday}日，星期{week}。"

    def build_system_prompt(self, rag_context: str = "",
                            learned_rules: str = "", vision_desc: str = "") -> str:
        parts = [self._today_line(), self.system_prompt]
        if learned_rules:
            parts.append("\n【已学习到的规则】\n" + learned_rules)
        if rag_context:
            parts.append("\n【我的记忆，回答相关问题时参考】\n" + rag_context)
        if vision_desc:
            parts.append("\n【我当前看到的桌面】\n" + vision_desc)
        return "\n".join(parts)

    def _trim_history(self) -> None:
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

    # ---- 生成 ----
    async def ask(self, user_text: str, rag_context: str = "",
                  learned_rules: str = "", vision_desc: str = "",
                  stream: bool = False) -> str:
        system = self.build_system_prompt(rag_context, learned_rules, vision_desc)
        messages = [{"role": "system", "content": system},
                    *self.history,
                    {"role": "user", "content": user_text}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                r = await client.post(f"{self.base_url}/v1/chat/completions",
                                      json=payload, headers=self.headers)
                r.raise_for_status()
                data = r.json()
            answer = data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            log.error("LLM 调用失败: %s", e)
            raise
        # 更新历史（含本轮）
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})
        self._trim_history()
        return answer

    # ---- 流式问答（按句产出，配合流式 TTS）----
    async def ask_stream(self, user_text: str, rag_context: str = "",
                         learned_rules: str = "", vision_desc: str = "") -> Iterator[str]:
        """流式生成，按句（。！？;换行）切分产出文本；同时维护对话历史。

        用于语音场景：生成一句就交给 TTS 播一句，不等完整回答。
        """
        system = self.build_system_prompt(rag_context, learned_rules, vision_desc)
        messages = [{"role": "system", "content": system},
                    *self.history,
                    {"role": "user", "content": user_text}]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }
        t_start = time.monotonic()
        first_token_at: Optional[float] = None
        usage: Optional[dict] = None
        full, buf = "", ""
        first_chunk = True   # 首个非空块用小阈值硬切，保证 TTS 首声快
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                     json=payload, headers=self.headers) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    if obj.get("usage"):
                        usage = obj["usage"]
                    try:
                        delta = obj["choices"][0]["delta"].get("content") or ""
                    except Exception:
                        continue
                    if delta and first_token_at is None:
                        first_token_at = time.monotonic()
                    full += delta
                    buf += delta
                    # 1) 句末标点边界 → 切块（一次回答通常 2-4 块）
                    matched = 0
                    for m in _SENT_RE.finditer(buf):
                        yield m.group(0).strip()
                        matched = m.end()
                    if matched:
                        buf = buf[matched:]
                        first_chunk = False
                    # 2) 无标点硬切：首块小阈值（保首声快），后续大阈值（合并摊薄开销）
                    elif len(buf) >= (self.first_chunk_max if first_chunk else self.chunk_max):
                        n = self.first_chunk_max if first_chunk else self.chunk_max
                        yield buf[:n].strip()
                        buf = buf[n:]
                        first_chunk = False
        end = time.monotonic()
        self.last_metrics = {
            "ttft_ms": round((first_token_at - t_start) * 1000, 1) if first_token_at else round((end - t_start) * 1000, 1),
            "gen_ms": round((end - first_token_at) * 1000, 1) if first_token_at else 0.0,
            "total_ms": round((end - t_start) * 1000, 1),
            "completion_tokens": usage.get("completion_tokens") if usage else None,
            "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        }
        # 尾部未闭合标点的余句
        tail = buf.strip()
        if tail:
            yield tail
        # 更新历史（含本轮）
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full.strip()})
        self._trim_history()

    # ---- 工具/一次性问答（不污染对话历史）----
    async def complete(self, system: str, user: str, max_tokens: Optional[int] = None) -> str:
        # 仅本地 Qwen3 需要 /no_think 关 thinking（否则 token 耗在思考上 content 为空）；
        # DeepSeek 无 thinking 机制，追加会污染 system prompt。
        if self.provider == "local" and _NO_THINK not in system:
            system = system + "\n" + _NO_THINK
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self._today_line() + "\n" + system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
        }
        async with httpx.AsyncClient(timeout=300.0) as client:
            r = await client.post(f"{self.base_url}/v1/chat/completions",
                                  json=payload, headers=self.headers)
            r.raise_for_status()
            data = r.json()
        return data["choices"][0]["message"]["content"].strip()
