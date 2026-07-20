#!/usr/bin/env python3
import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


LOG = logging.getLogger("build_sbert_index")


def parse_args():
    parser = argparse.ArgumentParser(description="Build SBERT corpus_embeddings.npy from corpus_meta.jsonl.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--corpus-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def row_text(row):
    for key in ["text", "target_text", "document", "content", "prior_text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("Corpus row has no text-like field.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.corpus_file))
    texts = [row_text(row) for row in rows]
    LOG.info("Encoding %d corpus texts with %s", len(texts), args.model_dir)
    model = SentenceTransformer(args.model_dir)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(output_dir / "corpus_embeddings.npy", emb)
    shutil.copyfile(args.corpus_file, output_dir / "corpus_meta.jsonl")
    LOG.info("Saved index to %s", output_dir)


if __name__ == "__main__":
    main()
