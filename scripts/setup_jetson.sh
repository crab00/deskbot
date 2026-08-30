#!/usr/bin/env bash
# ============================================================
# Deskbot 一键环境部署（在 Jetson Nano 上运行）
# 用法：./scripts/setup_jetson.sh
#
# 前置：已在 Mac 跑过 scripts/mac-download.sh + dev-sync.sh，
#       models/ 与 third_party/llama.cpp.tar.gz 已随同步到达本机。
#
# 流程：
#   1) 安装/选用 Python ≥ 3.8（JetPack 4.6 默认 3.6，需升级）
#   2) 安装系统依赖（PortAudio、构建工具等）
#   3) 创建 venv 并安装 Python 依赖
#   4) 用本地源码包构建 llama.cpp（CPU 版 llama-server）
#   5) 校验模型文件完整性
# 全部幂等，可重复执行。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

log() { echo -e "\n\033[1;32m== $* ==\033[0m"; }
SUDO=""; [ "$(id -u)" = "0" ] || SUDO="sudo"

# ---------- 1) Python ≥ 3.8 ----------
PY=""
for cand in python3.11 python3.10 python3.9 python3.8 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    v=$("$cand" -c 'import sys;print(sys.version_info[:2])' 2>/dev/null || echo "0 0")
    major=${v% *}; minor=${v#* }
    if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then PY="$cand"; break; fi
  fi
done

if [ -z "$PY" ]; then
  log "Python < 3.8，安装 python3.8（deadsnakes PPA）"
  $SUDO apt-get update -y
  $SUDO apt-get install -y software-properties-common || true
  $SUDO add-apt-repository -y ppa:deadsnakes/ppa || true
  $SUDO apt-get update -y
  $SUDO apt-get install -y python3.8 python3.8-dev python3.8-venv
  PY=python3.8
fi
echo "使用 Python: $PY ($($PY --version))"

# ---------- 2) 系统依赖 ----------
log "安装系统依赖"
$SUDO apt-get install -y \
  build-essential cmake git wget curl \
  portaudio19-dev libasound2-dev \
  libopenblas-dev libomp-dev \
  python3-dev python3-pip

# ---------- 3) venv + Python 依赖 ----------
log "创建虚拟环境并安装依赖"
if [ ! -d "$ROOT/.venv" ]; then
  "$PY" -m venv "$ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip wheel setuptools
# 国内镜像优先（清华），失败退回官方 pypi
PIP_MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
pip install $PIP_MIRROR -r "$ROOT/requirements.txt" 2>/dev/null || pip install -r "$ROOT/requirements.txt" || {
  echo "部分包安装失败，逐个重试（忽略可选依赖）…"
  grep -v '^#' "$ROOT/requirements.txt" | grep -v '^$' | while read -r p; do
    pip install $PIP_MIRROR "$p" 2>/dev/null || pip install "$p" 2>/dev/null || echo "  [跳过] $p"
  done
}

# ---------- 4) llama-server（conda-forge 预编译二进制 + 动态库，免编译） ----------
log "部署 llama-server（预编译 linux-aarch64 CPU 版 + 运行库）"
SRC_BIN="$ROOT/third_party/llama-server.linux-aarch64"
SRC_LIBS="$ROOT/third_party/llama-server-libs"
LLAMA_BIN="$ROOT/models/llm/llama-server"
mkdir -p "$ROOT/models/llm"
if [ ! -x "$LLAMA_BIN" ]; then
  if [ ! -x "$SRC_BIN" ]; then
    echo "⚠️  缺少预编译二进制 $SRC_BIN，请先在 Mac 运行 scripts/mac-download.sh"
    echo "     （若坚持源码编译，Nano 需能访问 GitHub 源码包）"
  else
    cp "$SRC_BIN" "$LLAMA_BIN"
    # 动态库放 models/llm/lib/，与 llama-server 的 $ORIGIN/../lib 约定一致
    if [ -d "$SRC_LIBS" ]; then
      rm -rf "$ROOT/models/llm/lib"
      cp -r "$SRC_LIBS" "$ROOT/models/llm/lib"
    fi
    echo "已部署: $LLAMA_BIN (+ lib/)"
  fi
else
  echo "llama-server 已存在，跳过"
fi

# ---------- 5) 校验模型 ----------
log "校验模型文件"
MISSING=0
check() { [ -s "$1" ] || { echo "  缺失: $1"; MISSING=1; }; }
check "$ROOT/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
check "$ROOT/models/asr/sense-voice-zh/tokens.txt"
check "$ROOT/models/asr/sense-voice-zh/model.int8.onnx"
check "$ROOT/models/tts/vits-zh-ll/model.onnx"
check "$ROOT/models/tts/vits-zh-ll/tokens.txt"
check "$ROOT/models/tts/vits-zh-ll/lexicon.txt"
check "$ROOT/models/tts/matcha-icefall-zh-baker/model-steps-3.onnx"
check "$ROOT/models/tts/matcha-icefall-zh-baker/vocoder-vocos.onnx"
check "$ROOT/models/vad/silero_vad.onnx"
if [ "$MISSING" = "1" ]; then
  echo "⚠️  有模型缺失，请先在 Mac 上运行 scripts/mac-download.sh 再同步。"
else
  echo "模型完整 ✓"
fi

log "环境就绪"
echo "下一步："
echo "  source .venv/bin/activate"
echo "  python -m deskbot.main --smoke   # 冒烟测试"
echo "  python -m deskbot.main --text    # 键盘问答"
echo "  python -m deskbot.main --voice   # 语音问答"
