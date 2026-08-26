"""llama.cpp server（llama-server）子进程管理。

启动/健康检查/停止都由本模块负责；默认 CPU 推理（-ngl 0），
适配 Jetson Nano 无新 CUDA 的环境。server 路径可用 config 指定。
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import httpx

from ..utils.async_utils import to_thread
from ..utils.config import Config
from ..utils.logging_setup import get_logger

log = get_logger("llm.server")


class LlamaServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        model = cfg.path("llm.model")
        self.model = model
        self.host = cfg.get("llm.host", "127.0.0.1")
        self.port = int(cfg.get("llm.port", 8080))
        self.ctx = int(cfg.get("llm.context_size", 2048))
        self.threads = int(cfg.get("llm.num_threads", 4))
        server_path = cfg.get("llm.server_path", "llama-server")
        self.server_bin = self._resolve_server(server_path)
        self._proc: Optional[subprocess.Popen] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _resolve_server(server_path: str) -> str:
        p = Path(server_path)
        if p.exists():
            return str(p)
        found = shutil.which(server_path)
        if found:
            return found
        return server_path  # 让子进程报错，信息更明确

    def _args(self) -> list:
        return [
            self.server_bin,
            "--model", str(self.model),
            "--host", self.host,
            "--port", str(self.port),
            "-c", str(self.ctx),
            "-t", str(self.threads),
            "-ngl", "0",               # 纯 CPU，避免 Nano 老 CUDA 崩溃
            "--embeddings",            # 复用本模型做 RAG 嵌入
            "--pooling", "mean",
            "--no-warmup",
            "--alias", "deskbot",
        ]

    async def start(self, timeout: float = 180.0) -> None:
        """启动 server 并等待就绪。已在运行且模型一致则复用，不一致则替换。"""
        if await self.healthy():
            served = await self._served_model()
            if served and Path(served).resolve() != self.model.resolve():
                log.warning("端口 %s 上已有 llama-server 在跑 %s，与配置 %s 不一致，替换为配置模型",
                            self.port, served, self.model)
                await self._stop_existing()
            else:
                log.info("llama-server 已在运行: %s", self.base_url)
                return
        if not self.model.exists():
            raise FileNotFoundError(
                f"LLM 模型不存在: {self.model}（先运行 scripts/download_models.sh）")
        if not Path(self._resolve_server(self.server_bin)).exists() and not shutil.which(self.server_bin):
            raise FileNotFoundError(
                f"找不到 llama-server: {self.server_bin}（先运行 scripts/setup_jetson.sh 构建）")

        log.info("启动 llama-server: %s", " ".join(self._args()))
        env = os.environ.copy()
        # 关键：不继承外部 LD_LIBRARY_PATH。否则 shell 里 export 的新版
        # llama 库（如 models/llm/lib）会污染加载，Ubuntu18.04 的
        # GLIBCXX/GLIBC 版本不够直接崩溃。二进制用自己的 RUNPATH 库即可。
        env.pop("LD_LIBRARY_PATH", None)
        # 1) 配置中显式声明的额外库目录（可选，如 third_party/openssl3）
        lib_dirs = [str(self.cfg.resolve_path(p.strip()))
                    for p in str(self.cfg.get("llm.lib_paths", "")).split(",") if p.strip()]
        # 2) 二进制同目录旁的 lib/（如 conda 预编译版）
        bin_path = Path(self._resolve_server(self.server_bin))
        if os.path.isdir(bin_path.parent / "lib"):
            lib_dirs.append(str(bin_path.parent / "lib"))
        if lib_dirs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(lib_dirs)
            log.info("LD_LIBRARY_PATH 设为: %s", os.pathsep.join(lib_dirs))
        # stderr 写日志文件，便于排查启动失败
        err_log = self.cfg.path("paths.logs") / "llama-server.log"
        err_log.parent.mkdir(parents=True, exist_ok=True)
        self._err_fh = open(err_log, "a", encoding="utf-8")
        self._proc = subprocess.Popen(
            self._args(),
            stdout=subprocess.DEVNULL, stderr=self._err_fh,
            env=env,
        )
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(f"llama-server 提前退出，返回码 {self._proc.returncode}")
            if await self.healthy():
                log.info("llama-server 就绪")
                return
            await asyncio.sleep(2.0)
        raise TimeoutError(f"llama-server 在 {timeout}s 内未就绪")

    async def healthy(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(f"{self.base_url}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def _served_model(self) -> Optional[str]:
        """询问现有服务器的模型路径（/v1/models）。失败返回 None。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{self.base_url}/v1/models")
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    if data:
                        return str(data[0].get("id", ""))
        except Exception:
            return None
        return None

    async def _stop_existing(self) -> None:
        """停掉端口上模型不一致的陈旧 llama-server：先优雅 /shutdown，失败按端口强杀。"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{self.base_url}/shutdown")
        except Exception:
            pass
        for _ in range(12):  # 最多等 ~6s
            if not await self.healthy():
                return
            await asyncio.sleep(0.5)
        log.warning("优雅停止旧 llama-server 失败，按端口 %s 强杀", self.port)
        await to_thread(self._force_kill_port, self.port)
        for _ in range(20):
            if not await self.healthy():
                return
            await asyncio.sleep(0.5)
        raise RuntimeError(f"端口 {self.port} 上的旧 llama-server 未能停止")

    @staticmethod
    def _force_kill_port(port: int) -> None:
        """按 TCP 端口杀掉占用进程（fuser）。用于替换端口上模型不一致的旧服务器。"""
        import subprocess
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=10)
        except Exception:
            pass

    async def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(to_thread(self._proc.wait), 5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
            log.info("llama-server 已停止")
        self._proc = None
