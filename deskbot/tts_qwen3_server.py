#!/usr/bin/env python3
"""Qwen3-TTS 常驻服务进程（Python mlx-audio 参考实现，Apple Silicon）。

由 Deskbot 的 Qwen3TTSBackend 通过 subprocess 拉起，stdin/stdout 行协议：
  发: <text>        合成句子（UTF-8，整行）
  收: OK <wav_path> <sr> <samples> <gen_sec> <rtf>   成功（wav 已写好，16bit PCM mono）
  收: ERR <message>                                   失败
  收: DONE                                            当前句结束标记
  发: __PING__     → 收 PONG
  发: __EXIT__     退出

模型只加载一次（示例 ~2-4s 取决于磁盘/型号），避免每句冷启动。
依赖独立 venv（Python 3.11 + mlx-audio 0.4.3 + mlx Metal），与 Deskbot 主 venv 隔离。
背景：AtomGradient swift-qwen3-tts（Swift+MLX）AR 生成有 bug 产出退化解码音频，
Python 参考实现（mlx-audio 0.4.3 qwen3_tts.TTS）对该模型家族产出干净自然语音（已实测）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid


def main() -> int:
    ap = argparse.ArgumentParser(description="Qwen3-TTS 常驻服务（mlx-audio Python 参考实现）")
    ap.add_argument("--model", required=True, help="模型目录（config.json + safetensors）")
    ap.add_argument("--speaker", default="Vivian", help="说话人（CustomVoice，如 Vivian/Ryan）")
    ap.add_argument("--language", default="zh", help="语言：zh/en/auto 等")
    ap.add_argument("--out-dir", default="/tmp", help="临时 wav 输出目录")
    ap.add_argument("--max-tokens", type=int, default=2048)
    args = ap.parse_args()

    # mlx-audio 可能打印大量日志，重定向到 stderr 不污染协议
    import logging
    logging.getLogger("mlx_audio").setLevel(logging.WARNING)

    import numpy as np
    import mlx.core as mx
    from mlx_audio.tts.utils import load as tts_load

    model_dir = os.path.abspath(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    def ensure_dict(d):
        return d or {}

    # ---- 加载模型（一次）----
    load_start = time.time()
    # 非严格加载：pruned vocab 模型含 text_token_map 等额外权重，strict 会报错
    model = tts_load(model_dir, strict=False)
    sample_rate = 24000
    load_sec = time.time() - load_start
    print(f"LOADED model={model_dir} sr={sample_rate} load={load_sec:.2f}s", flush=True)

    # 说话人合法性检查（失败不致命，跑不起来再报）
    lang = args.language
    spk = args.speaker
    max_tokens = args.max_tokens

    seq = 0
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "__EXIT__":
            break
        if text == "__PING__":
            print("PONG", flush=True)
            continue

        seq += 1
        gen_start = time.time()
        out_path = os.path.join(args.out_dir, f"qwen3_{seq}_{uuid.uuid4().hex[:8]}.wav")
        try:
            gen = model.generate(
                text=text,
                voice=spk,
                temperature=0.9,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.05,
                max_tokens=max_tokens,
            )
            parts = []
            for item in gen:
                audio = getattr(item, "audio", None)
                if audio is not None:
                    mx.eval(audio)
                    arr = np.array(audio.reshape(-1)).astype(np.float32)
                    parts.append(arr)
            if not parts:
                raise RuntimeError("generate() 未返回音频")
            samples = np.concatenate(parts)
            samples = np.clip(samples, -1.0, 1.0)
            pcm = (samples * 32767.0).astype(np.int16)

            # 写 16bit PCM mono WAV
            import wave
            with wave.open(out_path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sample_rate)
                w.writeframes(pcm.tobytes())

            gen_sec = time.time() - gen_start
            duration = len(samples) / float(sample_rate)
            rtf = duration / gen_sec if gen_sec > 0 else 0.0
            print(
                f"OK {out_path} {sample_rate} {len(samples)} {gen_sec:.2f} {rtf:.2f}",
                flush=True,
            )
        except Exception as e:
            print(f"ERR {type(e).__name__}: {e}".replace("\n", " "), flush=True)
        finally:
            print("DONE", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())