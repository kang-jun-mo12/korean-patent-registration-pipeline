#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from sentence_transformers import InputExample, SentenceTransformer, losses


LOG = logging.getLogger("train_sbert")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reconstructed SBERT training: stage1 MNRL and stage2 hard-negative MNRL."
    )
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    parser.add_argument("--train-file", required=True, help="JSONL train file.")
    parser.add_argument("--base-model", default="jhgan/ko-sroberta-multitask")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--mini-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-save-steps", type=int, default=0)
    parser.add_argument("--checkpoint-path", default="")
    return parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc


def first_text(row, keys):
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def hard_negative_texts(row):
    values = []
    hard_negatives = row.get("hard_negatives") or row.get("hard_negative_texts")
    if isinstance(hard_negatives, list):
        values.extend(str(x).strip() for x in hard_negatives if str(x).strip())
    for key in ["hard_negative1", "hard_negative2", "hard_negative3", "negative", "negative_text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def row_to_example(row, stage):
    anchor = first_text(row, ["anchor", "query", "target_text", "target", "text1"])
    positive = first_text(row, ["positive", "positive_text", "prior_text", "document", "text2"])
    if not anchor or not positive:
        raise ValueError("Each SBERT row needs anchor/query/target_text and positive/prior_text fields.")

    texts = [anchor, positive]
    if stage == "stage2":
        hard_negs = hard_negative_texts(row)
        if not hard_negs:
            raise ValueError("Stage2 rows need hard_negatives or hard_negative1..3 fields.")
        texts.extend(hard_negs)
    return InputExample(texts=texts)


def load_examples(path, stage):
    examples = []
    for row in read_jsonl(path):
        examples.append(row_to_example(row, stage))
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples


def build_loss(model, stage, mini_batch_size):
    if stage == "stage1":
        return losses.MultipleNegativesRankingLoss(model)
    loss_cls = getattr(losses, "CachedMultipleNegativesRankingLoss", None)
    if loss_cls is None:
        LOG.warning("CachedMultipleNegativesRankingLoss not available; falling back to MultipleNegativesRankingLoss.")
        return losses.MultipleNegativesRankingLoss(model)
    return loss_cls(model, mini_batch_size=mini_batch_size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    batch_size = args.batch_size if args.batch_size is not None else 256
    epochs = args.epochs if args.epochs is not None else (3 if args.stage == "stage1" else 2)
    lr = args.lr if args.lr is not None else (2e-5 if args.stage == "stage1" else 1e-5)

    LOG.info("Loading model: %s", args.base_model)
    model = SentenceTransformer(args.base_model)
    try:
        model.max_seq_length = args.max_seq_length
    except Exception:
        LOG.warning("Could not set max_seq_length on this SentenceTransformer version.")

    LOG.info("Loading examples from %s", args.train_file)
    examples = load_examples(args.train_file, args.stage)
    train_loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    train_loss = build_loss(model, args.stage, args.mini_batch_size)

    fit_kwargs = {
        "train_objectives": [(train_loader, train_loss)],
        "epochs": epochs,
        "warmup_steps": args.warmup_steps,
        "optimizer_params": {"lr": lr},
        "output_path": args.output_dir,
        "show_progress_bar": True,
    }
    if args.checkpoint_save_steps > 0 and args.checkpoint_path:
        fit_kwargs["checkpoint_save_steps"] = args.checkpoint_save_steps
        fit_kwargs["checkpoint_path"] = args.checkpoint_path

    LOG.info("Training stage=%s examples=%d batch=%d epochs=%d lr=%g", args.stage, len(examples), batch_size, epochs, lr)
    model.fit(**fit_kwargs)
    model.save(args.output_dir)
    LOG.info("Saved SBERT model to %s", args.output_dir)


if __name__ == "__main__":
    main()
