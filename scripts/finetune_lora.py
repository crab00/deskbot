#!/usr/bin/env python3
"""离机 QLoRA 微调脚本（在带独显的 PC / 云 GPU 上运行）。

用法：
  python scripts/finetune_lora.py --dataset data/datasets/train.jsonl \
      --base Qwen/Qwen2.5-1.5B-Instruct --out models/llm/finetuned

依赖（fine_tune.sh 会自动安装）：
  pip install unsloth peft trl bitsandbytes transformers datasets accelerate

数据集格式：每行一条 chat 记录 {"messages":[{"role":...},...]}，
由 `python -m deskbot.main --export` 生成。

产物：
  --out/model.gguf        Q4_K_M GGUF，llama-server 可直接加载
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def load_chat_dataset(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"数据集为空: {path}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="models/llm/finetuned")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--max-seq-len", type=int, default=1024)
    args = ap.parse_args()

    dataset = load_chat_dataset(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"加载 {len(dataset)} 条训练样本，基础模型 {args.base}")

    try:
        from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError:
        raise SystemExit("未安装 unsloth，请先运行 ./scripts/fine_tune.sh 安装依赖")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base,
        max_seq_length=args.max_seq_len,
        dtype=None,  # 自动选择
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=32, lora_dropout=0.05, bias="none", use_gradient_checkpointing=True,
    )

    from datasets import Dataset as HFDataset
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    # chat 模板格式化
    def fmt(row: dict) -> str:
        msgs = row["messages"]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)

    hf_ds = HFDataset.from_list(dataset)
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template="<|im_start|>user",
        response_template="<|im_start|>assistant",
        tokenizer=tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=TrainingArguments(
            output_dir=str(out / "checkpoints"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            save_strategy="no",
            report_to=[],
        ),
        train_dataset=hf_ds,
        formatting_func=fmt,
        data_collator=collator,
        max_seq_length=args.max_seq_len,
        dataset_text_field="messages",  # 占位，formatting_func 生效
    )
    trainer.train()

    # 合并 + 导出 GGUF（Q4_K_M）
    print("合并 LoRA 权重…")
    model.save_pretrained_merged(str(out / "merged"), tokenizer, save_method="merged_16bit")
    try:
        print("导出 GGUF (Q4_K_M)…")
        model.save_pretrained_gguf(str(out), tokenizer, quantization_method="q4_k_m")
        print(f"完成！GGUF 位于 {out / 'model.gguf'}")
    except Exception as e:  # llama.cpp 转换可能缺依赖
        print(f"GGUF 导出失败（{e}），已保存 16bit 合并权重：{out / 'merged'}")
        print("可手动用 llama.cpp 的 convert_hf_to_gguf.py 转换后部署。")


if __name__ == "__main__":
    main()
