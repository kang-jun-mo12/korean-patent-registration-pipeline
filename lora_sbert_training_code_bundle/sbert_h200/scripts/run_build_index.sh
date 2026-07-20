#!/usr/bin/env bash
set -euo pipefail
python scripts/build_sbert_index.py \
  --model_dir models/sbert_patent_h200 \
  --corpus data/corpus_unique.jsonl \
  --output_dir index/sbert_patent_h200 \
  --batch_size 512
