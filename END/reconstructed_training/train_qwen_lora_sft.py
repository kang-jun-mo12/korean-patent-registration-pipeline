#!/usr/bin/env python3
import argparse
import inspect
import json
import logging
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


LOG = logging.getLogger("train_qwen_lora_sft")


def parse_args():
    parser = argparse.ArgumentParser(description="Reconstructed response-only SFT for novelty/inventive Qwen LoRA.")
    parser.add_argument("--task", choices=["novelty", "inventive"], required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-32B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-train-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
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


def row_messages(row):
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    if row.get("prompt") and row.get("completion"):
        return [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": str(row["completion"])},
        ]
    if row.get("input") and row.get("output"):
        return [
            {"role": "user", "content": str(row["input"])},
            {"role": "assistant", "content": str(row["output"])},
        ]
    raise ValueError("SFT rows need messages, prompt/completion, or input/output fields.")


def apply_template(tokenizer, messages, add_generation_prompt):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    rendered = []
    for msg in messages:
        rendered.append(f"{msg.get('role', 'user')}: {msg.get('content', '')}")
    if add_generation_prompt:
        rendered.append("assistant:")
    return "\n".join(rendered)


class ResponseOnlyDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        messages = row_messages(self.rows[idx])
        if messages[-1].get("role") != "assistant":
            raise ValueError("The last message must be an assistant completion for response-only SFT.")

        prompt_messages = messages[:-1]
        full_text = apply_template(self.tokenizer, messages, add_generation_prompt=False)
        prompt_text = apply_template(self.tokenizer, prompt_messages, add_generation_prompt=True)

        full = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        prompt = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = list(input_ids)
        mask_len = min(len(prompt["input_ids"]), len(labels))
        labels[:mask_len] = [-100] * mask_len
        if all(label == -100 for label in labels) and labels:
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class CausalCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        labels = [feature.pop("labels") for feature in features]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        padded_labels = []
        for label in labels:
            padded_labels.append(label + [-100] * (max_len - len(label)))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


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
        "lr_scheduler_type": args.lr_scheduler_type,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.bfloat16 if args.bf16 and not args.fp16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    target_modules = [x.strip() for x in args.target_modules.split(",") if x.strip()]
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset = ResponseOnlyDataset(args.train_file, tokenizer, args.max_seq_length)
    eval_dataset = ResponseOnlyDataset(args.validation_file, tokenizer, args.max_seq_length)
    LOG.info("Loaded train=%d validation=%d for task=%s", len(train_dataset), len(eval_dataset), args.task)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args_kwargs(args)),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    LOG.info("Saved LoRA adapter to %s", args.output_dir)


if __name__ == "__main__":
    main()
