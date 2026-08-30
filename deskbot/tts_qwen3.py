"""Qwen3-TTS 后端（Mac，Apple Silicon）：Python mlx-audio 常驻子进程。

背景：AtomGradient/Qwen3-TTS-0.6B-CustomVoice-bf16-pruned-vocab-lite 是专为
Apple Silicon 优化的 Qwen3-TTS（MLX，macOS 专用）。早期尝试用官方 Swift 移植
（swift-qwen3-tts/Sources/Qwen3TTSServer）集成，但其 AR 生成有 bug —— 官方
Qwen3TTSDemo 与我们的服务在任意型号/语言/采样下都产出退化解码音频（常 6s 满帧、
无 EOS、87% 削波）。实测 mlx-audio 0.4.3 Python 参考实现对该模型家族产出干净自然语音。

方案：Deskbot 用 subprocess 常驻本包的 tts_qwen3_server.py（Python mlx-audio），
stdin/stdout 行协议对话（模型只加载一次，进程常驻）：
  发: <text>        合成句子
  收: LOADED ...    启动握手（模型加载完成）
  收: OK <wav> <sr> <samples> <gen_sec> <rtf> / DONE     成功
  收: ERR <msg> / DONE                                          失败
  发: __PING__     探活 → 收 PONG
  发: __EXIT__     退出

解释器用独立 mlx-audio venv（默认 ~/projects/deskbot-mlx-audio/.venv，Python 3.11 +
mlx-audio 0.4.3 + mlx Metal），与 Deskbot 主 venv（Python 3.9）隔离，Nano 不受影响。
外部可通过环境变量 QWEN3_PYTHON 指定解释器路径。
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .utils.logging_setup import get_logger

log = get_logger("tts.qwen3")


class Qwen3TTSUnavailable(Exception):
    pass


def _read_wav_mono(path: str) -> Tuple[np.ndarray, int]:
    """读单声道 16bit PCM WAV，返回 (float32 波形, 采样率)。"""
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0, int(sr)


def _resolve_python() -> str:
    """定位 mlx-audio Python 解释器。优先级：QWEN3_PYTHON → 默认 venv → PATH。"""
    env = os.environ.get("QWEN3_PYTHON")
    if env and Path(env).exists():
        return env
    # 默认：~/projects/deskbot-mlx-audio/.venv/bin/python
    for cand in [
        Path.home() / "projects" / "deskbot-mlx-audio" / ".venv" / "bin" / "python",
        Path("/Users/crab/projects/deskbot-mlx-audio/.venv/bin/python"),
    ]:
        if cand.exists():
            return str(cand)
    import shutil
    found = shutil.which("python3")
    if found:
        return found
    raise Qwen3TTSUnavailable("找不到 mlx-audio 解释器（请设置 QWEN3_PYTHON 或建 venv）")


def _find_server() -> str:
    """定位 tts_qwen3_server.py。"""
    # 与后端同目录
    cand = Path(__file__).with_name("tts_qwen3_server.py")
    if cand.exists():
        return str(cand)
    raise Qwen3TTSUnavailable(f"找不到 tts_qwen3_server.py（{cand}）")


class Qwen3TTSBackend:
    """Qwen3-TTS 常驻子进程后端（Python mlx-audio），接口对齐 sherpa-onnx 的 TTS.synthesize。"""

    def __init__(self, model_dir: Path, speaker: str = "Vivian",
                 language: str = "zh", out_dir: Optional[Path] = None,
                 python: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.speaker = speaker
        self.language = language
        self.out_dir = Path(out_dir) if out_dir else Path("/tmp")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # 进程串行，防并发写冲突
        self._proc: Optional[subprocess.Popen] = None
        self.sample_rate = 24000
        self._spawn(python)

    # ---- 进程管理 ----
    def _spawn(self, python: Optional[str]) -> None:
        py = python or _resolve_python()
        server = _find_server()
        cmd = [py, server,
               "--model", str(self.model_dir),
               "--speaker", self.speaker,
               "--language", self.language,
               "--out-dir", str(self.out_dir)]
        log.info("启动 Qwen3-TTS(mlx-audio): %s", " ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1)
        except Exception as e:
            raise Qwen3TTSUnavailable(f"mlx-audio 服务启动失败: {e}") from e
        # 握手：等 LOADED 行（含加载耗时）
        deadline = time.time() + 180
        while time.time() < deadline:
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read()[-1500:]
                raise Qwen3TTSUnavailable(f"mlx-audio 服务在握手前退出: {err}")
            line = line.strip()
            if line.startswith("LOADED"):
                log.info("Qwen3-TTS 就绪: %s", line)
                if "sr=" in line:
                    try:
                        self.sample_rate = int(line.split("sr=")[1].split()[0])
                    except Exception:
                        pass
                return
            if line.startswith("ERR"):
                raise Qwen3TTSUnavailable(f"mlx-audio 服务初始化失败: {line}")
            log.debug("Qwen3 启动输出: %s", line)
        raise Qwen3TTSUnavailable("Qwen3-TTS 服务加载超时(180s)")

    # ---- 合成 ----
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """合成一句中文，返回 (float32 波形, 采样率)。线程安全。"""
        if not text:
            return np.zeros(0, dtype=np.float32), self.sample_rate
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                raise Qwen3TTSUnavailable("Qwen3-TTS 服务已退出")
            assert self._proc.stdin and self._proc.stdout
            # 写入时把换行去掉（句子可能含 \n）；协议见模块 docstring
            self._proc.stdin.write(text.replace("\n", " ") + "\n")
            self._proc.stdin.flush()
            # 等 OK ... / DONE（或 ERR）
            ok_line = None
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise Qwen3TTSUnavailable("Qwen3-TTS stdout 意外关闭")
                line = line.strip()
                if line.startswith("OK "):
                    ok_line = line
                    continue
                if line == "DONE":
                    break
                if line.startswith("ERR"):
                    raise Qwen3TTSUnavailable(f"Qwen3 合成失败: {line}")
                log.debug("Qwen3 输出: %s", line)
            if ok_line is None:
                raise Qwen3TTSUnavailable("Qwen3 未返回 OK")
            parts = ok_line.split()
            # OK <wav> <sr> <samples> <gen_sec> <rtf>
            wav_path = parts[1]
            audio, sr = _read_wav_mono(wav_path)
            # 清理临时 wav（optional）
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass
            return audio, int(sr)

    # ---- 生命周期 ----
    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                assert self._proc.stdin
                self._proc.stdin.write("__EXIT__\n")
                self._proc.stdin.flush()
                self._proc.wait(timeout=15)
            except Exception:
                self._proc.kill()
        if self._proc and self._proc.poll() is None:
            self._proc.kill()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass