#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict

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
    p.add_argument("--eval_pairs", default="data/eval_pairs.jsonl")
    p.add_argument("--corpus", default="data/corpus_unique.jsonl")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--query_batch_size", type=int, default=128)
    p.add_argument("--max_queries", type=int, default=0, help="0 means all eval queries")
    args = p.parse_args()

    corpus_rows = list(read_jsonl(args.corpus))
    corpus_ids = [r["prior_id"] for r in corpus_rows]
    corpus_texts = [r["text"] for r in corpus_rows]
    id_to_idx = {pid: i for i, pid in enumerate(corpus_ids)}

    query_text = {}
    relevant = defaultdict(set)
    for row in read_jsonl(args.eval_pairs):
        qid = row["query_id"]
        query_text.setdefault(qid, row["query_text"])
        if row["positive_id"] in id_to_idx:
            relevant[qid].add(row["positive_id"])

    qids = [qid for qid in query_text if relevant[qid]]
    if args.max_queries and len(qids) > args.max_queries:
        qids = qids[:args.max_queries]

    model = SentenceTransformer(args.model_dir)
    corpus_emb = model.encode(
        corpus_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    hits = {k: 0 for k in [1, 2, 3, 5, 10, 20, 50]}
    total = 0
    for start in range(0, len(qids), args.query_batch_size):
        batch_qids = qids[start:start + args.query_batch_size]
        q_emb = model.encode(
            [query_text[qid] for qid in batch_qids],
            batch_size=args.query_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        scores = q_emb @ corpus_emb.T
        top = np.argpartition(-scores, kth=min(50, scores.shape[1] - 1), axis=1)[:, :50]
        row_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-row_scores, axis=1)
        top = np.take_along_axis(top, order, axis=1)
        for i, qid in enumerate(batch_qids):
            total += 1
            found = [corpus_ids[idx] for idx in top[i]]
            truth = relevant[qid]
            for k in hits:
                if any(pid in truth for pid in found[:k]):
                    hits[k] += 1

    print(json.dumps({
        "queries": total,
        "corpus": len(corpus_rows),
        **{f"recall@{k}": round(hits[k] / total, 6) if total else 0 for k in hits},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
