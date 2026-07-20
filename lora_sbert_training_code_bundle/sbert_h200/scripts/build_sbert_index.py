#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/sbert_patent_h200")
    p.add_argument("--corpus", default="data/corpus_unique.jsonl")
    p.add_argument("--output_dir", default="index/sbert_patent_h200")
    p.add_argument("--batch_size", type=int, default=512)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = list(read_jsonl(args.corpus))
    texts = [r["text"] for r in rows]

    model = SentenceTransformer(args.model_dir)
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    np.save(out / "corpus_embeddings.npy", emb)
    with (out / "corpus_meta.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            meta = {k: v for k, v in r.items() if k != "text"}
            meta["text"] = r["text"]
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print({"rows": len(rows), "embedding_shape": emb.shape, "output_dir": str(out)}, flush=True)


if __name__ == "__main__":
    main()
