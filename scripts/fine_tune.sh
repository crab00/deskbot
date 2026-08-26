#!/usr/bin/env bash
# ============================================================
# 离机 QLoRA 微调入口 —— 在【带独显的 PC 或云 GPU】上运行，
# 不要在 Jetson Nano 上跑（显存/算力不够）。
#
# 用法：
#   1) 在 Nano 上导出训练集：python -m deskbot.main --export
#   2) 把 data/datasets/train.jsonl 传到这台机器
#   3) ./scripts/fine_tune.sh data/datasets/train.jsonl [基础模型]
#   4) 产物 models/llm/finetuned/model.gguf，再用 deploy_model.sh 回传 Nano
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

DATASET="${1:-$ROOT/data/datasets/train.jsonl}"
BASE_MODEL="${2:-Qwen/Qwen2.5-1.5B-Instruct}"
OUT="$ROOT/models/llm/finetuned"

if [ ! -f "$DATASET" ]; then
  echo "找不到数据集: $DATASET（先到 Nano 上跑 python -m deskbot.main --export）" >&2
  exit 1
fi

echo "== 微调参数 =="
echo "数据集: $DATASET（$(wc -l < "$DATASET") 条）"
echo "基础模型: $BASE_MODEL"
echo "输出: $OUT"

if [ ! -d "$ROOT/.venv-train" ]; then
  python3 -m venv "$ROOT/.venv-train"
fi
# shellcheck disable=SC1091
source "$ROOT/.venv-train/bin/activate"
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
  "peft" "transformers" "trl" "bitsandbytes" "sentencepiece" "protobuf" "accelerate"

python "$ROOT/scripts/finetune_lora.py" \
  --dataset "$DATASET" \
  --base "$BASE_MODEL" \
  --out "$OUT"

echo
echo "完成！把 $OUT/model.gguf 传到 Nano 后执行："
echo "  ./scripts/deploy_model.sh models/llm/finetuned/model.gguf"
