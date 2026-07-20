#!/usr/bin/env python3
import argparse
import collections
import inspect
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


NOVELTY_SUFFIX = """[작업]
심사대상 청구항의 핵심 구성요소가 D1 또는 D2 중 하나에 실질적으로 모두 개시되어 신규성이 부족한지 판단하라.

[판단 절차]
1. 심사대상 청구항의 핵심 구성요소와 구성 간 관계를 파악한다.
2. D1과 D2를 각각 따로 비교한다.
3. 신규성 판단에서는 D1과 D2를 조합하지 않는다.
4. 하나의 선행문헌에 핵심 구성요소 전부가 명시적 또는 실질적으로 개시되면 신규성 부족이다.
5. 선행문헌 하나만으로는 차이점이 남거나 여러 문헌의 조합이 필요하면 신규성 만족이다.

[출력 규칙]
- JSON 객체 하나만 출력한다.
- novelty는 \"만족\" 또는 \"부족\" 중 하나다.
- 설명, 근거 문장, markdown, 코드블록은 출력하지 않는다.

[출력 형식]
{\"novelty\":\"만족|부족\"}"""


INVENTIVE_SUFFIX = """[작업]
D1,D2를 근거로 심사대상 청구항의 차이점이 통상의 기술자가 쉽게 도출할 수 있는지 판단해 진보성 만족/부족을 분류하라. 단순 유사도나 주제 유사성만으로 부족 판단하지 않는다.

[판단 절차]
1. 심사대상 청구항의 필수 구성요소, 구성 간 관계, 해결과제, 작용효과를 분리한다.
2. D1을 주선행으로 보고 공통 구성과 남는 차이점을 찾는다. D2가 그 차이점을 보완하는지 확인한다.
3. 부족: 차이점이 단순결합, 설계변경, 균등치환, 공지기술부가, 통상최적화에 가깝고 D1-D2의 기술분야, 해결과제, 기능이 맞아 결합 동기나 암시가 있으며 효과가 예측 가능하다.
4. 만족: 핵심 차이점이 선행문헌에 없거나, 결합 동기가 약하거나, 결합 시 구조/기능 충돌이 있거나, 효과가 예상 밖의 상승효과에 가깝다.
5. 신규성처럼 한 문헌에 전부 개시되는지만 보지 말고, 남는 차이점의 쉬운 도출 여부를 중심으로 판단한다.
6. 입력 문헌만 근거로 판단하고, 판단 근거가 더 강한 쪽의 라벨 하나를 선택한다.

[출력 규칙]
- JSON 객체 하나만 출력한다.
- inventive_step은 \"만족\" 또는 \"부족\" 중 하나다.
- 신규성 부족 전용 샘플은 이 어댑터 학습에 사용하지 않는다.
- 설명, 근거 문장, markdown, 코드블록은 출력하지 않는다.

[출력 형식]
{\"inventive_step\":\"만족|부족\"}"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run novelty and inventive LoRA adapters, combine predictions with AND, and score registration/rejection."
    )
    parser.add_argument("--base-model", default="", help="Defaults to the novelty adapter base_model_name_or_path.")
    parser.add_argument("--novelty-adapter", default="checkpoint-1500")
    parser.add_argument("--inventive-adapter", default="checkpoint-2000")
    parser.add_argument("--novelty-test", default="patent30000_h200_bf16_lora/data/novelty/novelty_test.jsonl")
    parser.add_argument("--inventive-test", default="patent30000_h200_bf16_lora/data/inventive/inventive_test.jsonl")
    parser.add_argument(
        "--inventive-fill-files",
        nargs="*",
        default=[
            "patent30000_h200_bf16_lora/data/inventive/inventive_val.jsonl",
            "patent30000_h200_bf16_lora/data/inventive/inventive_train.jsonl",
        ],
        help="Used only if inventive_test has fewer than 1000 ids not overlapping novelty_test.",
    )
    parser.add_argument("--per-task", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--write-chunk-size", type=int, default=25)
    parser.add_argument("--output", default="and_eval_predictions_nonoverlap_2000.jsonl")
    parser.add_argument("--summary-output", default="and_eval_summary_nonoverlap_2000.json")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    return parser.parse_args()


def adapter_base_model(adapter_dir):
    config_path = Path(adapter_dir) / "adapter_config.json"
    with config_path.open(encoding="utf-8") as f:
        return json.load(f).get("base_model_name_or_path", "")


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            yield line_number, json.loads(line)


def add_eval_fields(row, source_file, source_line, source_task, selection_index):
    row = dict(row)
    source_name = Path(source_file).stem
    row["_source_file"] = Path(source_file).as_posix()
    row["_source_line"] = source_line
    row["_source_task"] = source_task
    row["_selection_index"] = selection_index
    row["_eval_id"] = f"{source_task}:{source_name}:{source_line}:{row['dataset_id']}"
    return row


def select_nonoverlap_rows(novelty_test, inventive_test, inventive_fill_files, per_task):
    rows = []
    used_dataset_ids = set()
    novelty_count = 0

    for line_number, row in iter_jsonl(novelty_test):
        if novelty_count >= per_task:
            break
        rows.append(add_eval_fields(row, novelty_test, line_number, "novelty", len(rows)))
        used_dataset_ids.add(row["dataset_id"])
        novelty_count += 1

    if novelty_count < per_task:
        raise RuntimeError(f"Only selected {novelty_count} novelty rows from {novelty_test}; need {per_task}.")

    inventive_count = 0
    candidate_files = [inventive_test, *inventive_fill_files]
    for source_file in candidate_files:
        for line_number, row in iter_jsonl(source_file):
            if inventive_count >= per_task:
                break
            if row["dataset_id"] in used_dataset_ids:
                continue
            rows.append(add_eval_fields(row, source_file, line_number, "inventive", len(rows)))
            used_dataset_ids.add(row["dataset_id"])
            inventive_count += 1
        if inventive_count >= per_task:
            break

    if inventive_count < per_task:
        raise RuntimeError(
            f"Only selected {inventive_count} inventive rows without overlap; need {per_task}. "
            f"Checked: {candidate_files}"
        )

    return rows


def document_block(row):
    user_content = row["messages"][1]["content"]
    marker = "\n[작업]\n"
    if marker not in user_content:
        raise ValueError(f"Cannot split prompt for {row.get('dataset_id')}: missing [작업] marker")
    return user_content.split(marker, 1)[0].rstrip()


def build_messages(row, task):
    suffix = NOVELTY_SUFFIX if task == "novelty" else INVENTIVE_SUFFIX
    return [
        row["messages"][0],
        {"role": "user", "content": f"{document_block(row)}\n\n{suffix}"},
    ]


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def normalize_binary(value, generated):
    if value in {"만족", "부족"}:
        return value
    match = re.search(r"(만족|부족)", generated)
    if match:
        return match.group(1)
    return "INVALID"


def input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_processed(path):
    processed = {}
    if not path.exists():
        return processed
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            processed[row["eval_id"]] = row
    return processed


def torch_dtype(value):
    if value == "auto":
        return "auto"
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[value]


def model_from_pretrained(base_model, dtype, device_map):
    kwargs = {
        "device_map": device_map,
        "trust_remote_code": True,
    }
    dtype_value = torch_dtype(dtype)
    if dtype_value != "auto":
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        kwargs["torch_dtype" if "torch_dtype" in signature.parameters else "dtype"] = dtype_value
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def generate_for_task(model, tokenizer, rows, task, batch_size, max_input_tokens, max_new_tokens):
    model.set_adapter(task)
    results = {}
    device = input_device(model)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(build_messages(row, task), tokenize=False, add_generation_prompt=True)
            for row in batch_rows
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for row, ids in zip(batch_rows, output_ids):
            generated = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
            parsed = extract_json(generated)
            key = "novelty" if task == "novelty" else "inventive_step"
            results[row["_eval_id"]] = {
                f"{task}_raw": generated,
                f"{task}_pred": normalize_binary(parsed.get(key), generated),
            }
    return results


def summarize(predictions):
    confusion = collections.Counter()
    gold_counts = collections.Counter()
    pred_counts = collections.Counter()
    class_counts = collections.Counter()
    source_counts = collections.Counter()
    invalid_counts = collections.Counter()

    for row in predictions:
        gold = row["gold_decision"]
        pred = row["pred_decision"]
        gold_counts[gold] += 1
        pred_counts[pred] += 1
        class_counts[row["class_label"]] += 1
        source_counts[row["source_file"]] += 1
        confusion[f"gold={gold}|pred={pred}"] += 1
        if row["novelty_pred"] == "INVALID":
            invalid_counts["novelty"] += 1
        if row["inventive_pred"] == "INVALID":
            invalid_counts["inventive"] += 1

    total = len(predictions)
    correct = sum(row["gold_decision"] == row["pred_decision"] for row in predictions)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "gold_counts": dict(gold_counts),
        "pred_counts": dict(pred_counts),
        "class_counts": dict(class_counts),
        "source_counts": dict(source_counts),
        "confusion": dict(confusion),
        "invalid_counts": dict(invalid_counts),
        "rule": "pred_decision = 등록 if novelty_pred == 만족 and inventive_pred == 만족 else 거절",
    }


def main():
    args = parse_args()
    output_path = Path(args.output)
    summary_path = Path(args.summary_output)
    rows = select_nonoverlap_rows(
        args.novelty_test,
        args.inventive_test,
        args.inventive_fill_files,
        args.per_task,
    )

    base_model = args.base_model or adapter_base_model(args.novelty_adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.novelty_adapter, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = model_from_pretrained(base_model, args.dtype, args.device_map)
    model = PeftModel.from_pretrained(base, args.novelty_adapter, adapter_name="novelty")
    model.load_adapter(args.inventive_adapter, adapter_name="inventive")
    model.eval()

    processed = {} if args.no_resume else load_processed(output_path)
    pending = [row for row in rows if row["_eval_id"] not in processed]

    if pending:
        mode = "w" if args.no_resume else "a"
        with output_path.open(mode, encoding="utf-8") as out:
            for chunk_start in range(0, len(pending), args.write_chunk_size):
                chunk = pending[chunk_start : chunk_start + args.write_chunk_size]
                novelty = generate_for_task(
                    model,
                    tokenizer,
                    chunk,
                    "novelty",
                    args.batch_size,
                    args.max_input_tokens,
                    args.max_new_tokens,
                )
                inventive = generate_for_task(
                    model,
                    tokenizer,
                    chunk,
                    "inventive",
                    args.batch_size,
                    args.max_input_tokens,
                    args.max_new_tokens,
                )

                for row in chunk:
                    eval_id = row["_eval_id"]
                    novelty_pred = novelty[eval_id]["novelty_pred"]
                    inventive_pred = inventive[eval_id]["inventive_pred"]
                    pred_decision = "등록" if novelty_pred == "만족" and inventive_pred == "만족" else "거절"
                    record = {
                        "eval_id": eval_id,
                        "dataset_id": row["dataset_id"],
                        "source_file": row["_source_file"],
                        "source_line": row["_source_line"],
                        "source_task": row["_source_task"],
                        "selection_index": row["_selection_index"],
                        "class_label": row["class_label"],
                        "gold_decision": row["decision_label"],
                        "gold_novelty": row["novelty_label"],
                        "gold_inventive_step": row["inventive_step_label"],
                        "novelty_pred": novelty_pred,
                        "inventive_pred": inventive_pred,
                        "pred_decision": pred_decision,
                        "correct": pred_decision == row["decision_label"],
                        **novelty[eval_id],
                        **inventive[eval_id],
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed[eval_id] = record

                out.flush()
                partial = [processed[r["_eval_id"]] for r in rows if r["_eval_id"] in processed]
                print(json.dumps(summarize(partial), ensure_ascii=False), flush=True)

    ordered_predictions = [processed[row["_eval_id"]] for row in rows if row["_eval_id"] in processed]
    summary = summarize(ordered_predictions)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
