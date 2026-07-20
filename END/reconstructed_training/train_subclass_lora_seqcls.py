#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LOG = logging.getLogger("train_subclass_lora_seqcls")


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstructed Qwen 7B LoRA multi-label subclass classifier.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--label-names", default="결합류,차이류,설계변경,공지기술부가")
    parser.add_argument("--thresholds", default="0.35,0.3,0.3,0.3")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
    if not rows:
        raise ValueError(f"No rows loaded from {path}")
    return rows


def row_text(row):
    for key in ["text", "target_text", "input", "prompt", "document", "content"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = []
        for msg in messages:
            if msg.get("role") != "assistant":
                parts.append(str(msg.get("content", "")))
        text = "\n\n".join(part for part in parts if part.strip())
        if text:
            return text
    raise ValueError("Subclass rows need text, target_text, input, prompt, or messages fields.")


def row_labels(row, label_names):
    for key in ["labels", "label_ids", "multihot", "subclass_multihot"]:
        value = row.get(key)
        if isinstance(value, list) and len(value) == len(label_names) and all(isinstance(x, (int, float, bool)) for x in value):
            return [float(x) for x in value]

    names = None
    for key in ["labels", "label_names", "subclass_labels", "subclass_gold_labels"]:
        value = row.get(key)
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            names = set(value)
            break
        if isinstance(value, str) and value.strip():
            names = {x.strip() for x in value.split(",") if x.strip()}
            break
    if names is None:
        value = row.get("label")
        if isinstance(value, str) and value.strip():
            names = {value.strip()}
    if names is None:
        raise ValueError("Subclass rows need multi-hot labels or label name list.")

    return [1.0 if name in names else 0.0 for name in label_names]


class MultiLabelDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, label_names):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_names = label_names

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        encoded = self.tokenizer(
            row_text(row),
            truncation=True,
            max_length=self.max_length,
        )
        encoded["labels"] = row_labels(row, self.label_names)
        return encoded


class MultiLabelCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer=tokenizer)

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.base(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


def f1_score_np(y_true, y_pred, average):
    eps = 1e-12
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)
    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        return float((2 * tp) / (2 * tp + fp + fn + eps))
    per_label = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(per_label))


def build_metrics(thresholds):
    thresholds = np.array(thresholds, dtype=np.float32)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= thresholds).astype(np.float32)
        labels = labels.astype(np.float32)

        top2 = np.argsort(-probs, axis=1)[:, :2]
        hit_count = 0
        valid_count = 0
        for true_row, pred_top2 in zip(labels, top2):
            true_idx = set(np.where(true_row > 0.5)[0].tolist())
            if not true_idx:
                continue
            valid_count += 1
            if true_idx.intersection(set(pred_top2.tolist())):
                hit_count += 1

        return {
            "micro_f1": f1_score_np(labels, preds, "micro"),
            "macro_f1": f1_score_np(labels, preds, "macro"),
            "hit_at_2": float(hit_count / valid_count) if valid_count else 0.0,
        }

    return compute_metrics


def training_args_kwargs(args):
    kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "bf16": bool(args.bf16 and not args.fp16),
        "fp16": bool(args.fp16),
        "optim": "adamw_torch",
        "report_to": "none",
        "remove_unused_columns": False,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    kwargs["save_strategy"] = "steps"
    return kwargs


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    label_names = [x.strip() for x in args.label_names.split(",") if x.strip()]
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    if len(label_names) != len(thresholds):
        raise ValueError("--label-names and --thresholds must have the same length.")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=len(label_names),
        problem_type="multi_label_classification",
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.id2label = {i: name for i, name in enumerate(label_names)}
    model.config.label2id = {name: i for i, name in enumerate(label_names)}
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=["score", "classifier"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = MultiLabelDataset(args.train_file, tokenizer, args.max_seq_length, label_names)
    eval_dataset = MultiLabelDataset(args.validation_file, tokenizer, args.max_seq_length, label_names)
    LOG.info("Loaded train=%d validation=%d labels=%s", len(train_dataset), len(eval_dataset), label_names)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MultiLabelCollator(tokenizer),
        compute_metrics=build_metrics(thresholds),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    threshold_map = {name: threshold for name, threshold in zip(label_names, thresholds)}
    with open(Path(args.output_dir) / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(threshold_map, f, ensure_ascii=False, indent=2)
    LOG.info("Saved subclass adapter to %s", args.output_dir)


if __name__ == "__main__":
    main()
