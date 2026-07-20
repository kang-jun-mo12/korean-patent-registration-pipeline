#!/usr/bin/env python3
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--task", choices=["novelty", "inventive"], required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return {}


def main():
    args = parse_args()
    key = "novelty" if args.task == "novelty" else "inventive_step"

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    total = 0
    correct = 0
    with open(args.test_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row["messages"][:-1]
            gold = json.loads(row["messages"][-1]["content"])[key]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=0.0,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
            pred = extract_json(generated).get(key)

            total += 1
            correct += int(pred == gold)
            if args.limit and total >= args.limit:
                break

    print(json.dumps({
        "task": args.task,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
