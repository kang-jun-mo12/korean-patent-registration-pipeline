#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


LOG = logging.getLogger("mine_sbert_hardneg")


def parse_args():
    parser = argparse.ArgumentParser(description="Mine hard negatives for reconstructed SBERT stage2 training.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--train-file", required=True, help="Positive pair JSONL.")
    parser.add_argument("--corpus-file", required=True, help="Corpus JSONL with id/text fields.")
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--rank-start", type=int, default=20)
    parser.add_argument("--rank-end", type=int, default=200)
    parser.add_argument("--num-negatives", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def first_value(row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
    return ""


def row_text(row, keys):
    return first_value(row, keys)


def row_ids(row):
    ids = set()
    for key in ["positive_id", "prior_id", "prior_uid", "document_id", "d1_id", "d2_id"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    for key in ["positive_ids", "gold_prior_ids", "prior_ids"]:
        value = row.get(key)
        if isinstance(value, list):
            ids.update(str(x).strip() for x in value if str(x).strip())
    return ids


def load_corpus(path):
    rows = list(read_jsonl(path))
    ids = []
    texts = []
    for i, row in enumerate(rows):
        corpus_id = first_value(
            row,
            ["prior_id", "prior_uid", "id", "document_id", "resolved_application_number", "application_number"],
        ) or f"corpus:{i}"
        text = row_text(row, ["text", "target_text", "document", "content", "prior_text"])
        if not text:
            raise ValueError(f"Corpus row {i + 1} has no text field.")
        ids.append(corpus_id)
        texts.append(text)
    return rows, np.array(ids, dtype=object), texts


def positive_pair(row):
    anchor = row_text(row, ["anchor", "query", "target_text", "target", "text1"])
    positive = row_text(row, ["positive", "positive_text", "prior_text", "document", "text2"])
    if not anchor or not positive:
        raise ValueError("Each train row needs anchor/query/target_text and positive/prior_text.")
    return anchor, positive


def batched(items, size):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.rank_start < 1 or args.rank_end < args.rank_start:
        raise ValueError("--rank-start must be >= 1 and --rank-end must be >= rank-start")

    model = SentenceTransformer(args.model_dir)
    corpus_rows, corpus_ids, corpus_texts = load_corpus(args.corpus_file)
    LOG.info("Encoding corpus rows=%d", len(corpus_texts))
    corpus_emb = model.encode(
        corpus_texts,
        batch_size=args.encode_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    train_rows = list(read_jsonl(args.train_file))
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for _, batch_rows in tqdm(list(batched(train_rows, args.batch_size)), desc="mine"):
            anchors = [positive_pair(row)[0] for row in batch_rows]
            q_emb = model.encode(
                anchors,
                batch_size=args.encode_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            scores = q_emb @ corpus_emb.T
            top_n = min(args.rank_end, scores.shape[1])
            candidate_idx = np.argpartition(-scores, kth=top_n - 1, axis=1)[:, :top_n]

            for row, row_scores, idxs in zip(batch_rows, scores, candidate_idx):
                anchor, positive = positive_pair(row)
                excluded_ids = row_ids(row)
                positive_text = positive.strip()
                ordered = idxs[np.argsort(-row_scores[idxs])]

                hard_negs = []
                hard_neg_ids = []
                for corpus_idx in ordered[args.rank_start - 1 : args.rank_end]:
                    corpus_id = str(corpus_ids[corpus_idx])
                    corpus_text = corpus_texts[int(corpus_idx)]
                    if corpus_id in excluded_ids or corpus_text.strip() == positive_text:
                        continue
                    hard_negs.append(corpus_text)
                    hard_neg_ids.append(corpus_id)
                    if len(hard_negs) >= args.num_negatives:
                        break

                if len(hard_negs) < args.num_negatives:
                    continue

                new_row = dict(row)
                new_row["anchor"] = anchor
                new_row["positive"] = positive
                new_row["hard_negatives"] = hard_negs
                new_row["hard_negative_ids"] = hard_neg_ids
                out.write(json.dumps(new_row, ensure_ascii=False) + "\n")

    LOG.info("Wrote %s", output_path)


if __name__ == "__main__":
    main()
