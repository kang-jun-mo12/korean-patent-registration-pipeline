#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


LABELS = ["결합류", "차이류", "설계변경", "공지기술부가"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


class SubclassDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        enc = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        enc["labels"] = torch.tensor(row["multihot"], dtype=torch.float32)
        return enc


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def collect_logits(model, loader, device):
    logits = []
    labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            y = batch.pop("labels").numpy()
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch).logits.float().cpu().numpy()
            logits.append(out)
            labels.append(y)
    return np.concatenate(logits, axis=0), np.concatenate(labels, axis=0)


def metrics(y_true, probs, thresholds):
    y_pred = np.zeros_like(y_true, dtype=np.int32)
    for i, label in enumerate(LABELS):
        y_pred[:, i] = (probs[:, i] >= thresholds[label]).astype(np.int32)
    per = {}
    f1s = []
    for i, label in enumerate(LABELS):
        yt = y_true[:, i].astype(bool)
        yp = y_pred[:, i].astype(bool)
        tp = int(np.logical_and(yt, yp).sum())
        fp = int(np.logical_and(~yt, yp).sum())
        fn = int(np.logical_and(yt, ~yp).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(yt.sum()), "tp": tp, "fp": fp, "fn": fn}
        f1s.append(f1)
    subset = float((y_pred == y_true).all(axis=1).mean())
    hit1 = hit2 = 0
    for p, y in zip(probs, y_true):
        gold = set(np.where(y == 1)[0].tolist())
        if not gold:
            continue
        order = np.argsort(-p)
        hit1 += int(order[0] in gold)
        hit2 += int(any(i in gold for i in order[:2]))
    return {
        "macro_f1": float(np.mean(f1s)),
        "subset_accuracy": subset,
        "hit_at_1": hit1 / max(len(y_true), 1),
        "hit_at_2": hit2 / max(len(y_true), 1),
        "per_label": per,
        "thresholds": thresholds,
    }


def calibrate_thresholds(y_true, probs):
    thresholds = {}
    for i, label in enumerate(LABELS):
        best_t = 0.5
        best_f1 = -1
        for t in np.arange(0.10, 0.91, 0.05):
            yp = probs[:, i] >= t
            yt = y_true[:, i].astype(bool)
            tp = np.logical_and(yt, yp).sum()
            fp = np.logical_and(~yt, yp).sum()
            fn = np.logical_and(yt, ~yp).sum()
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(round(t, 2))
        thresholds[label] = best_t
    return thresholds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--adapter_dir", default="models/qwen7b_subclass_h200_grounded_lora")
    p.add_argument("--val_file", default="data/subclass_val.jsonl")
    p.add_argument("--test_file", default="data/subclass_test.jsonl")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--output", default="eval_subclass_result.json")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    model = PeftModel.from_pretrained(base, args.adapter_dir)
    device = next(model.parameters()).device

    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    val_loader = DataLoader(SubclassDataset(args.val_file, tokenizer, args.max_seq_length), batch_size=args.batch_size, collate_fn=collator)
    test_loader = DataLoader(SubclassDataset(args.test_file, tokenizer, args.max_seq_length), batch_size=args.batch_size, collate_fn=collator)

    val_logits, val_y = collect_logits(model, val_loader, device)
    val_probs = sigmoid(val_logits)
    thresholds = calibrate_thresholds(val_y, val_probs)

    test_logits, test_y = collect_logits(model, test_loader, device)
    test_probs = sigmoid(test_logits)
    result = {
        "val_thresholds": thresholds,
        "val": metrics(val_y, val_probs, thresholds),
        "test": metrics(test_y, test_probs, thresholds),
    }
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.adapter_dir, "thresholds.json").write_text(json.dumps(thresholds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
