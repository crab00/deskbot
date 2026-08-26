#!/usr/bin/env bash
# ============================================================
# 开发机(Mac) → Jetson Nano 同步（代码 + 模型 + llama.cpp 源码包）
# 用法：./scripts/dev-sync.sh [user@host]   （默认 crab@192.168.31.202）
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

HOST="${1:-crab@192.168.31.202}"
DEST="$HOST:~/deskbot/"

echo "同步到 $DEST ..."
rsync -avz --progress \
  --exclude '.git/' \
  --exclude 'data/' \
  --exclude '.venv/' \
  --exclude '.venv-train/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  ./ "$DEST"

echo "完成。接下来在 Nano 上："
echo "  cd ~/deskbot && bash scripts/setup_jetson.sh"
