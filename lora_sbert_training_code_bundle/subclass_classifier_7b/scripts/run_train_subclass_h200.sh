#!/usr/bin/env bash
set -euo pipefail
python scripts/train_subclass_h200.py \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --train_file data/subclass_train.jsonl \
  --val_file data/subclass_val.jsonl \
  --output_dir models/qwen7b_subclass_h200_grounded_lora \
  --max_seq_length 4096 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 5 \
  --learning_rate 2e-5 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --eval_steps 50 \
  --save_steps 50
