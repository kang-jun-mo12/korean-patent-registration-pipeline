#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LABELS = ["결합류", "차이류", "설계변경", "공지기술부가"]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


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
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = torch.tensor(row["multihot"], dtype=torch.float32)
        return enc


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def f1_stats(y_true, y_pred):
    per = {}
    f1s = []
    for i, label in enumerate(LABELS):
        yt = y_true[:, i].astype(bool)
        yp = y_pred[:, i].astype(bool)
        tp = np.logical_and(yt, yp).sum()
        fp = np.logical_and(~yt, yp).sum()
        fn = np.logical_and(yt, ~yp).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[label] = {"precision": precision, "recall": recall, "f1": f1}
        f1s.append(f1)
    return float(np.mean(f1s)), per


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(np.int32)
    y_true = labels.astype(np.int32)
    macro_f1, per = f1_stats(y_true, preds)
    subset_acc = float((preds == y_true).all(axis=1).mean())
    hit1 = 0
    hit2 = 0
    for p, y in zip(probs, y_true):
        gold = set(np.where(y == 1)[0].tolist())
        if not gold:
            continue
        order = np.argsort(-p)
        if order[0] in gold:
            hit1 += 1
        if any(i in gold for i in order[:2]):
            hit2 += 1
    denom = max(len(y_true), 1)
    return {
        "macro_f1": macro_f1,
        "subset_accuracy": subset_acc,
        "hit_at_1": hit1 / denom,
        "hit_at_2": hit2 / denom,
        **{f"f1_{label}": stats["f1"] for label, stats in per.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--train_file", default="data/subclass_train.jsonl")
    p.add_argument("--val_file", default="data/subclass_val.jsonl")
    p.add_argument("--output_dir", default="models/qwen7b_subclass_h200_grounded_lora")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.id2label = {i: label for i, label in enumerate(LABELS)}
    model.config.label2id = {label: i for i, label in enumerate(LABELS)}
    model.config.use_cache = False

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=TARGET_MODULES,
        modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = SubclassDataset(args.train_file, tokenizer, args.max_seq_length)
    val_ds = SubclassDataset(args.val_file, tokenizer, args.max_seq_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size // 2),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print({
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "labels": LABELS,
        "max_seq_length": args.max_seq_length,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "load_in_4bit": args.load_in_4bit,
    }, flush=True)

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "label_config.json").write_text(
        json.dumps({"labels": LABELS, "threshold_default": 0.5}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
