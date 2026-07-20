#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_eval(eval_pairs_path, max_queries=500):
    query_text = {}
    corpus = {}
    relevant = defaultdict(set)
    for row in read_jsonl(eval_pairs_path):
        qid = row["query_id"]
        pid = row["positive_id"]
        if qid not in query_text:
            query_text[qid] = row["query_text"]
        corpus[pid] = row["positive_text"]
        relevant[qid].add(pid)

    selected_qids = list(query_text)[:max_queries]
    queries = {qid: query_text[qid] for qid in selected_qids}
    rel = {qid: relevant[qid] for qid in selected_qids}
    used_pids = set().union(*rel.values()) if rel else set()
    eval_corpus = {pid: corpus[pid] for pid in used_pids}
    return queries, eval_corpus, rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_pairs", default="data/train_pairs.jsonl")
    p.add_argument("--eval_pairs", default="data/eval_pairs.jsonl")
    p.add_argument("--base_model", default="jhgan/ko-sroberta-multitask")
    p.add_argument("--output_dir", default="models/sbert_patent_h200")
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--evaluation_steps", type=int, default=500)
    p.add_argument("--eval_queries", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    examples = []
    for row in read_jsonl(args.train_pairs):
        examples.append(InputExample(texts=[row["query_text"], row["positive_text"]]))

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length

    loader = DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    train_loss = losses.MultipleNegativesRankingLoss(model)

    queries, corpus, relevant = build_eval(args.eval_pairs, max_queries=args.eval_queries)
    evaluator = evaluation.InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant,
        accuracy_at_k=[1, 2, 3, 5, 10],
        precision_recall_at_k=[1, 2, 3, 5, 10],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        name="patent_cited_eval",
        show_progress_bar=False,
    )

    warmup_steps = math.ceil(len(loader) * args.epochs * 0.10)
    print({
        "train_examples": len(examples),
        "batch_size": args.batch_size,
        "steps_per_epoch": len(loader),
        "epochs": args.epochs,
        "total_steps": len(loader) * args.epochs,
        "warmup_steps": warmup_steps,
        "max_seq_length": args.max_seq_length,
        "base_model": args.base_model,
    }, flush=True)

    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        evaluator=evaluator,
        evaluation_steps=args.evaluation_steps,
        output_path=args.output_dir,
        save_best_model=True,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    main()
