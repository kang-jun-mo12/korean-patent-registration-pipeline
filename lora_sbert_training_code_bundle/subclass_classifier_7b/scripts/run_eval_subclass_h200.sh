#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate_subclass_h200.py \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --adapter_dir models/qwen7b_subclass_h200_grounded_lora \
  --val_file data/subclass_val.jsonl \
  --test_file data/subclass_test.jsonl \
  --max_seq_length 4096 \
  --batch_size 4 \
  --output eval_subclass_result.json
