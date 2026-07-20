#!/usr/bin/env bash
set -euo pipefail
python scripts/train_adapter_h200_lora.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --train_file data/inventive/inventive_train.jsonl \
  --val_file data/inventive/inventive_val.jsonl \
  --output_dir models/qwen32b_inventive_h200_bf16_lora \
  --max_seq_length 4096 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --eval_steps 500 \
  --save_steps 500 \
  --gradient_checkpointing unsloth \
  --optim adamw_torch
