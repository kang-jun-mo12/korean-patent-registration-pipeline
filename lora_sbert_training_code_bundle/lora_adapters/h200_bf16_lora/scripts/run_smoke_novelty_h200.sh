#!/usr/bin/env bash
set -euo pipefail
python scripts/train_adapter_h200_lora.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --train_file data/novelty/novelty_train.jsonl \
  --val_file data/novelty/novelty_val.jsonl \
  --output_dir models/smoke_novelty_h200_bf16_lora \
  --max_seq_length 4096 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 0.01 \
  --learning_rate 1e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --eval_steps 50 \
  --save_steps 50 \
  --gradient_checkpointing unsloth \
  --optim adamw_torch
