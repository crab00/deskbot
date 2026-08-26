#!/usr/bin/env bash
# ============================================================
# 下载 Deskbot 所需的全部离线模型（ASR / TTS / VAD / LLM / 视觉）
# 用法：./scripts/download_models.sh
# 模型保存到 models/ 目录（可重复执行，已存在则跳过）
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MODELS="$ROOT/models"

mkdir -p "$MODELS"/{asr,tts,vad,llm,vision}

dl() { # dl <url> <outfile>
  local url="$1" out="$2"
  if [ -f "$out" ]; then echo "  [跳过] $out"; return; fi
  echo "  [下载] $url"
  if command -v wget >/dev/null; then
    wget -q --show-progress -O "$out" "$url"
  else
    curl -fL --progress-bar -o "$out" "$url"
  fi
}

untar() { # untar <archive> <destdir>
  mkdir -p "$2"
  tar -xjf "$1" -C "$2"
}

echo "== 1/4 语音识别 ASR（Paraformer-zh，~230MB）=="
ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2"
ASR_ARC="$MODELS/asr/paraformer.tar.bz2"
if [ ! -d "$MODELS/asr/paraformer-zh/encoder.onnx" ] && [ ! -d "$MODELS/asr/paraformer-zh" ]; then
  dl "$ASR_URL" "$ASR_ARC"
  untar "$ASR_ARC" "$MODELS/asr"
  rm -f "$ASR_ARC"
else
  echo "  [跳过] ASR 已就绪"
fi

echo "== 2/4 语音合成 TTS（VITS 中文，~130MB）=="
TTS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2"
TTS_ARC="$MODELS/tts/vits.tar.bz2"
if [ ! -d "$MODELS/tts/vits-zh-ll" ]; then
  dl "$TTS_URL" "$TTS_ARC"
  untar "$TTS_ARC" "$MODELS/tts"
  rm -f "$TTS_ARC"
else
  echo "  [跳过] TTS 已就绪"
fi

echo "== 3/4 语音检测 VAD（Silero，~2MB）=="
dl "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx" \
   "$MODELS/vad/silero_vad.onnx"

echo "== 4/4 大模型 LLM（Qwen2.5-1.5B Q4_K_M，~1GB）=="
dl "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf" \
   "$MODELS/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"

echo "== 视觉 YOLOv8n（可选，~13MB）=="
if [ -f config.yaml ] && grep -q "vision:.*enabled: true" config.yaml 2>/dev/null; then
  dl "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx" \
     "$MODELS/vision/yolov8n.onnx"
else
  echo "  [跳过] 视觉未启用（config.yaml 中 vision.enabled=true 后下载）"
fi

echo
echo "模型下载完成："
du -sh "$MODELS"/* 2>/dev/null || true
echo
echo "提示：嵌入模型（bge-small-zh）由 fastembed 首次运行时自动下载到 ~/.cache/fastembed。"
