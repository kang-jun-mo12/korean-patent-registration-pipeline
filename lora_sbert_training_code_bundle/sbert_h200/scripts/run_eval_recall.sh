#!/usr/bin/env bash
set -euo pipefail
python scripts/evaluate_recall.py \
  --model_dir models/sbert_patent_h200 \
  --eval_pairs data/eval_pairs.jsonl \
  --corpus data/corpus_unique.jsonl \
  --batch_size 512 \
  --query_batch_size 128
