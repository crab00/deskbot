#!/usr/bin/env bash
# ============================================================
# Mac 专用 Qwen3-TTS（mlx-audio Python 后端）环境搭建
# 建立独立 venv（Python 3.11 + mlx-audio 0.4.3 + mlx Metal），
# 与 Deskbot 主 venv（Python 3.9，sherpa）隔离。
# 用法：bash scripts/setup-mac-qwen3.sh
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# 项目外的专用 venv（apple silicon 专属；Nano 不需要）
VENV_DIR="${QWEN3_VENV:-$HOME/projects/deskbot-mlx-audio/.venv}"

echo "== Qwen3-TTS(mlx-audio) venv: $VENV_DIR =="
if [ -x "$VENV_DIR/bin/python" ]; then
  echo "  [跳过] $VENV_DIR 已存在"
else
  echo "== 创建 venv（Python 3.11）=="
  uv venv --python=3.11 "$VENV_DIR"
fi

echo "== 安装 mlx-audio 0.4.3 =="
uv pip install --python "$VENV_DIR/bin/python" "mlx-audio==0.4.3"

echo
echo "完成。Deskbot 通过配置文件 + QWEN3_PYTHON 使用此环境："
echo "  export QWEN3_PYTHON=$VENV_DIR/bin/python"
echo "  # 或后端默认探测 ~/projects/deskbot-mlx-audio/.venv/bin/python，无需 export"
echo
echo "模型（unpruned 8bit，~1.3GB）需另下（见 TECHNICAL.md §9）："
echo "  mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit"