#!/usr/bin/env python3
import argparse
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def clean_claim_1(claim):
    claim = claim or ""
    claim = re.sub(r"제\d+항에\s*있어서[,\s]*", "", claim)
    claim = re.sub(r"을\s*특징으로\s*하는", "", claim)
    return re.sub(r"\s+", " ", claim).strip()


def query_from_json(path):
    with open(path, encoding="utf-8") as f:
        r = json.load(f)
    title = r.get("invention_title") or r.get("title") or ""
    abstract = r.get("abstract") or ""
    claims = r.get("claims") or []
    claim_1 = clean_claim_1(claims[0] if claims else r.get("claim_1", ""))
    return "\n".join(part for part in [
        f"[발명의 명칭]\n{title}" if title else "",
        f"[초록]\n{abstract}" if abstract else "",
        f"[청구항]\n{claim_1}" if claim_1 else "",
    ] if part).strip()


def read_meta(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="models/sbert_patent_h200")
    p.add_argument("--index_dir", default="index/sbert_patent_h200")
    p.add_argument("--query_text", default="")
    p.add_argument("--query_json", default="")
    p.add_argument("--top_k", type=int, default=2)
    args = p.parse_args()

    query = args.query_text.strip()
    if args.query_json:
        query = query_from_json(args.query_json)
    if not query:
        raise SystemExit("Provide --query_text or --query_json")

    model = SentenceTransformer(args.model_dir)
    emb = np.load(f"{args.index_dir}/corpus_embeddings.npy")
    meta = read_meta(f"{args.index_dir}/corpus_meta.jsonl")

    qe = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores = (qe @ emb.T)[0]
    top = np.argsort(-scores)[:args.top_k]
    results = []
    for rank, idx in enumerate(top, 1):
        item = meta[int(idx)].copy()
        item["rank"] = rank
        item["score"] = float(scores[idx])
        results.append(item)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
