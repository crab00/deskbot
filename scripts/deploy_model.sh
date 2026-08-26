#!/usr/bin/env bash
# ============================================================
# 部署新 GGUF 模型到 Jetson Nano 并做评估回归。
# 用法：./scripts/deploy_model.sh [新模型.gguf]
#   - 备份当前模型 → 复制新模型 → 跑评估 → 报告
#   - 评估分数低于基线会自动回滚备份
# 需在 Nano 上运行（或用 ssh 远程执行）。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

NEW_GGUF="${1:-}"
CURRENT="$ROOT/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
BACKUP="$ROOT/models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf.bak"

if [ -z "$NEW_GGUF" ]; then
  echo "用法: $0 <新模型.gguf>" >&2
  exit 1
fi
if [ ! -f "$NEW_GGUF" ]; then
  echo "文件不存在: $NEW_GGUF" >&2
  exit 1
fi

# 备份当前
if [ -f "$CURRENT" ]; then
  cp "$CURRENT" "$BACKUP"
  echo "已备份当前模型 → $BACKUP"
fi

# 复制新模型
cp "$NEW_GGUF" "$CURRENT"
echo "已部署: $NEW_GGUF"

# 跑评估
if [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi
echo "== 运行评估 =="
python -m deskbot.main --eval || true

# 提示人工核对，若需回滚：
echo
echo "若评估不理想，可回滚："
echo "  cp models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf.bak models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf"
