#!/usr/bin/env bash
set -euo pipefail
python scripts/train_sbert_h200.py \
  --train_pairs data/train_pairs.jsonl \
  --eval_pairs data/eval_pairs.jsonl \
  --base_model jhgan/ko-sroberta-multitask \
  --output_dir models/sbert_patent_h200 \
  --max_seq_length 512 \
  --batch_size 128 \
  --epochs 3 \
  --lr 2e-5 \
  --evaluation_steps 500 \
  --eval_queries 500 \
  --num_workers 4
