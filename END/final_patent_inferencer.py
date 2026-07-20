#!/usr/bin/env python3
import argparse
import gc
import inspect
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from evaluate_and_registration import INVENTIVE_SUFFIX, NOVELTY_SUFFIX, adapter_base_model

LABELS = ["결합류", "차이류", "설계변경", "공지기술부가"]
SUBCLASS_BASE = "Qwen/Qwen2.5-7B-Instruct"
CLASSIFIER_SYSTEM = {
    "role": "system",
    "content": "당신은 한국 특허 심사 보조 모델입니다. 심사대상 문헌과 선행문헌 D1,D2의 발명의 명칭, 초록, 청구항만 근거로 판단합니다. 허용된 라벨만 사용하고 JSON 객체 하나만 출력합니다.",
}
EXPLAINER_SYSTEM = {
    "role": "system",
    "content": "당신은 한국 특허 심사 결과를 사용자가 이해하기 쉽게 설명하는 특허 심사 보조자입니다. 제공된 심사대상 문헌, 검색된 선행문헌, 모델 판정값만 근거로 설명하고 없는 사실은 만들지 않습니다.",
}


def parse_args():
    p = argparse.ArgumentParser(description="Final patent inference pipeline: trained SBERT TOP2 + Qwen 32B adapters + optional subclass + 32B explanation.")
    input_group = p.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input-json", help="JSON input with target_text or title/abstract/claim_1 fields.")
    input_group.add_argument("--text-file", help="Text file containing target document/claim text.")
    input_group.add_argument("--target-text", help="Raw target text. If no section markers are present it is treated as claim 1.")
    p.add_argument("--title", default="", help="Invention title, used with --abstract/--claim-1.")
    p.add_argument("--abstract", default="", help="Abstract, used with --title/--claim-1.")
    p.add_argument("--claim-1", default="", help="Claim 1 text, used with --title/--abstract.")
    p.add_argument("--input-id", default="manual_input")

    p.add_argument("--sbert-model", default="patent30000_sbert_h200/models/sbert_patent_h200_hardneg_stage2")
    p.add_argument("--sbert-index", default="patent30000_sbert_h200/index/sbert_patent_h200_hardneg_stage2")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--novelty-adapter", default="checkpoint-1500")
    p.add_argument("--inventive-adapter", default="checkpoint-2000")
    p.add_argument("--subclass-adapter", default="handoff_to_friend/subclass_adapter")
    p.add_argument("--base-model", default="")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")

    p.add_argument("--max-input-tokens", type=int, default=6144)
    p.add_argument("--classification-max-new-tokens", type=int, default=32)
    p.add_argument("--subclass-max-length", type=int, default=3264)
    p.add_argument("--explanation-max-input-tokens", type=int, default=8192)
    p.add_argument("--explanation-max-new-tokens", type=int, default=900)
    p.add_argument("--sample-explanation", action="store_true", help="Use light sampling for the final explanation instead of greedy decoding.")
    p.add_argument("--skip-subclass", action="store_true")
    p.add_argument("--skip-explanation", action="store_true")
    p.add_argument("--output", default="final_inference_output.json")
    return p.parse_args()


def torch_dtype(value):
    if value == "auto":
        return "auto"
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[value]


def model_from_pretrained(base_model, dtype, device_map):
    kwargs = {"device_map": device_map, "trust_remote_code": True}
    dtype_value = torch_dtype(dtype)
    if dtype_value != "auto":
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        kwargs["torch_dtype" if "torch_dtype" in signature.parameters else "dtype"] = dtype_value
    return AutoModelForCausalLM.from_pretrained(base_model, **kwargs)


def input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def maybe_read_stdin():
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
        if text:
            return text
    return ""


def format_target(title, abstract, claim_1):
    parts = ["[심사대상 문헌]"]
    if title:
        parts.append("[발명의 명칭]\n" + title.strip())
    if abstract:
        parts.append("[초록]\n" + abstract.strip())
    if claim_1:
        parts.append("[청구항]\n" + claim_1.strip())
    return "\n\n".join(parts).strip()


def strip_task_and_priors(text):
    text = (text or "").strip()
    if "\n[작업]\n" in text:
        text = text.split("\n[작업]\n", 1)[0].rstrip()
    if "\n[선행문헌 D1]" in text:
        text = text.split("\n[선행문헌 D1]", 1)[0].rstrip()
    return text


def normalize_target_text(text):
    text = strip_task_and_priors(text)
    if not text:
        raise ValueError("Empty target text.")
    if "[심사대상 문헌]" not in text:
        if any(marker in text for marker in ["[발명의 명칭]", "[초록]", "[청구항]", "[청구항 전체]"]):
            text = "[심사대상 문헌]\n" + text
        else:
            text = format_target("", "", text)
    return text.strip()


def target_from_messages(data):
    messages = data.get("messages") or []
    for msg in messages:
        if msg.get("role") == "user":
            return normalize_target_text(msg.get("content", ""))
    raise ValueError("JSON messages did not contain a user message.")


def load_input(args):
    raw = {}
    if args.input_json:
        with open(args.input_json, encoding="utf-8") as f:
            raw = json.load(f)
        input_id = raw.get("id") or raw.get("dataset_id") or args.input_id
        if raw.get("messages"):
            return input_id, target_from_messages(raw), raw
        for key in ["target_text", "target_document", "text"]:
            if raw.get(key):
                return input_id, normalize_target_text(raw[key]), raw
        title = raw.get("title") or raw.get("invention_title") or raw.get("발명의 명칭") or ""
        abstract = raw.get("abstract") or raw.get("초록") or ""
        claim_1 = raw.get("claim_1") or raw.get("claim") or raw.get("청구항1") or raw.get("청구항") or ""
        return input_id, normalize_target_text(format_target(title, abstract, claim_1)), raw
    if args.text_file:
        text = Path(args.text_file).read_text(encoding="utf-8")
        return args.input_id, normalize_target_text(text), {"text_file": args.text_file}
    if args.target_text:
        return args.input_id, normalize_target_text(args.target_text), {"target_text": args.target_text}
    if args.title or args.abstract or args.claim_1:
        return args.input_id, normalize_target_text(format_target(args.title, args.abstract, args.claim_1)), {
            "title": args.title,
            "abstract": args.abstract,
            "claim_1": args.claim_1,
        }
    stdin_text = maybe_read_stdin()
    if stdin_text:
        return args.input_id, normalize_target_text(stdin_text), {"stdin": True}
    raise SystemExit("No input provided. Use --input-json, --text-file, --target-text, --claim-1, or pipe text on stdin.")


def clean_claim_1(claim):
    claim = claim or ""
    claim = re.sub(r"제\d+항에\s*있어서[,\s]*", "", claim)
    claim = re.sub(r"을\s*특징으로\s*하는", "", claim)
    return re.sub(r"\s+", " ", claim).strip()


def between(text, start, end=None):
    if start not in text:
        return ""
    s = text.split(start, 1)[1]
    if end and end in s:
        s = s.split(end, 1)[0]
    return s.strip()


def target_parts(target_text):
    title = between(target_text, "[발명의 명칭]", "[초록]")
    abstract = between(target_text, "[초록]", "[청구항]") or between(target_text, "[초록]", "[청구항 전체]")
    claims = between(target_text, "[청구항]") or between(target_text, "[청구항 전체]")
    claim_1 = claims
    split_markers = ["\n청구항2:", "\n청구항 2:", "\n청구항3:", "\n청구항 3:", "\n2.", "\n제 2 항"]
    cuts = [claim_1.find(m) for m in split_markers if claim_1.find(m) > 50]
    if cuts:
        claim_1 = claim_1[: min(cuts)]
    return {"title": title, "abstract": abstract, "claim_1": clean_claim_1(claim_1)}


def load_corpus_meta(index_dir):
    with open(Path(index_dir) / "corpus_meta.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def retrieve_sbert_topk(target_text, sbert_model_dir, sbert_index_dir, top_k):
    meta = load_corpus_meta(sbert_index_dir)
    emb = np.load(Path(sbert_index_dir) / "corpus_embeddings.npy").astype("float32")
    sbert = SentenceTransformer(sbert_model_dir)
    q_emb = sbert.encode(
        [target_text],
        batch_size=1,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype("float32")
    scores = (q_emb @ emb.T)[0]
    top = np.argpartition(-scores, kth=top_k - 1)[:top_k]
    top = top[np.argsort(-scores[top])]
    retrieved = []
    for rank, corpus_idx in enumerate(top, 1):
        item = meta[int(corpus_idx)].copy()
        item["rank"] = rank
        item["score"] = float(scores[int(corpus_idx)])
        retrieved.append(item)
    del sbert, emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return retrieved


def compact_reference(item):
    return {
        "rank": item.get("rank"),
        "score": item.get("score"),
        "prior_id": item.get("prior_id"),
        "prior_document_number": item.get("prior_document_number", ""),
        "invention_title": item.get("invention_title", ""),
        "text": item.get("text", ""),
    }


def build_augmented_document(target_text, retrieved):
    parts = [target_text]
    for idx, item in enumerate(retrieved, 1):
        parts.append(f"[선행문헌 D{idx}]\n{item.get('text', '').strip()}")
    return "\n\n".join(parts).rstrip()


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


def classifier_messages(augmented_document, task):
    suffix = NOVELTY_SUFFIX if task == "novelty" else INVENTIVE_SUFFIX
    return [CLASSIFIER_SYSTEM, {"role": "user", "content": f"{augmented_document}\n\n{suffix}"}]


def generate_classifier(model, tokenizer, augmented_document, task, max_input_tokens, max_new_tokens):
    model.set_adapter(task)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    prompt = tokenizer.apply_chat_template(classifier_messages(augmented_document, task), tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(input_device(model))
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    parsed = extract_json(generated)
    key = "novelty" if task == "novelty" else "inventive_step"
    return {"raw": generated, "pred": normalize_binary(parsed.get(key), generated), "parsed": parsed}


def decision_from_preds(novelty_pred, inventive_pred):
    return "등록" if novelty_pred == "만족" and inventive_pred == "만족" else "거절"


def class_from_preds(novelty_pred, inventive_pred):
    if novelty_pred == "부족":
        return "거절-신규성 부족"
    if novelty_pred == "만족" and inventive_pred == "부족":
        return "거절-진보성 부족"
    if novelty_pred == "만족" and inventive_pred == "만족":
        return "등록"
    return "INVALID"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def multihot(labels):
    labels = labels or []
    return [int(label in labels) for label in LABELS]


def subclass_input_text(target_text, retrieved):
    parts = target_parts(target_text)
    ref_text = retrieved[0].get("text", "")[:900] if retrieved else ""
    return "[claim_1]\n{}\n\n[abstract]\n{}\n\n[refs_top_K]\n[1] {}".format(
        parts["claim_1"][:1500],
        parts["abstract"][:600],
        ref_text,
    )


def load_subclass_model(adapter_dir, device_map):
    thresholds = json.loads((Path(adapter_dir) / "thresholds.json").read_text(encoding="utf-8"))
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForSequenceClassification.from_pretrained(
        SUBCLASS_BASE,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.id2label = {i: lab for i, lab in enumerate(LABELS)}
    base.config.label2id = {lab: i for i, lab in enumerate(LABELS)}
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer, thresholds


def predict_subclass(model, tokenizer, thresholds, target_text, retrieved, max_length):
    tokenizer.padding_side = "right"
    text = subclass_input_text(target_text, retrieved)
    batch = tokenizer(
        [text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        pad_to_multiple_of=8,
    ).to(input_device(model))
    with torch.inference_mode():
        logits = model(**batch).logits.float().cpu().numpy()[0]
    probs = sigmoid(logits)
    pred = [lab for i, lab in enumerate(LABELS) if probs[i] >= thresholds[lab]]
    fallback = False
    if not pred:
        top2 = np.argsort(-probs)[:2]
        pred = [LABELS[i] for i in top2]
        fallback = True
    ranked = sorted(
        [{"label": LABELS[i], "probability": float(probs[i])} for i in range(len(LABELS))],
        key=lambda x: -x["probability"],
    )
    return {
        "triggered": True,
        "labels": pred,
        "multihot": multihot(pred),
        "ranked": ranked,
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
        "fallback_top2": fallback,
        "input_format": "trained SBERT TOP1 README concise format",
    }


def disabled_adapter_context(model):
    if hasattr(model, "disable_adapter"):
        return model.disable_adapter()
    return nullcontext()


def explanation_prompt(target_text, retrieved, result_core):
    refs = []
    for item in retrieved:
        text = item.get("text", "")[:1800]
        refs.append(
            "D{rank} | score={score:.4f} | prior_id={prior_id}\n{title}\n{text}".format(
                rank=item.get("rank"),
                score=float(item.get("score", 0.0)),
                prior_id=item.get("prior_id", ""),
                title=item.get("invention_title", ""),
                text=text,
            )
        )
    subclass = result_core.get("subclass") or {}
    subclass_labels = subclass.get("labels") or []
    return """아래는 특허 심사 보조 파이프라인의 결과입니다. 사용자가 이해하기 쉬운 한국어 설명으로 정리하세요.

[심사대상]
{target}

[SBERT 검색 선행문헌 TOP{top_k}]
{refs}

[모델 판정]
- 신규성: {novelty}
- 진보성: {inventive}
- 최종 등록/거절: {decision}
- 세부 분류: {class_label}
- 진보성 부족 서브라벨: {subclass_labels}

[작성 지침]
1. 첫 문단에 결론을 분명히 말하세요.
2. 신규성과 진보성을 나누어 설명하세요.
3. 등록으로 나온 경우에는 "현재 검색된 TOP2 선행문헌 기준"이라는 한계를 분명히 쓰세요.
4. 거절로 나온 경우에는 어떤 판단값 때문에 거절인지 쉽게 설명하세요.
5. 서브라벨이 있으면 그 의미를 한두 문장으로 풀어주세요.
6. 과도하게 단정하지 말고, 모델 기반 예비 판단임을 마지막에 짧게 덧붙이세요.
""".format(
        target=target_text[:3000],
        top_k=len(retrieved),
        refs="\n\n".join(refs),
        novelty=result_core["novelty"]["prediction"],
        inventive=result_core["inventive_step"]["prediction"],
        decision=result_core["decision"],
        class_label=result_core["class_label"],
        subclass_labels=", ".join(subclass_labels) if subclass_labels else "없음/미실행",
    )


def generate_explanation(model, tokenizer, target_text, retrieved, result_core, max_input_tokens, max_new_tokens, sample):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    messages = [EXPLAINER_SYSTEM, {"role": "user", "content": explanation_prompt(target_text, retrieved, result_core)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(input_device(model))
    input_len = inputs["input_ids"].shape[-1]
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if sample:
        generate_kwargs.update({"do_sample": True, "temperature": 0.25, "top_p": 0.9})
    else:
        generate_kwargs.update({"do_sample": False})
    with torch.inference_mode():
        try:
            with disabled_adapter_context(model):
                output_ids = model.generate(**inputs, **generate_kwargs)
        except Exception:
            output_ids = model.generate(**inputs, **generate_kwargs)
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def load_32b_with_adapters(args):
    base_model = args.base_model or adapter_base_model(args.novelty_adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.novelty_adapter, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = model_from_pretrained(base_model, args.dtype, args.device_map)
    model = PeftModel.from_pretrained(base, args.novelty_adapter, adapter_name="novelty")
    model.load_adapter(args.inventive_adapter, adapter_name="inventive")
    model.eval()
    return model, tokenizer, base_model


def main():
    args = parse_args()
    input_id, target_text, raw_input = load_input(args)

    retrieved = retrieve_sbert_topk(target_text, args.sbert_model, args.sbert_index, args.top_k)
    augmented_document = build_augmented_document(target_text, retrieved)

    model32, tokenizer32, base_model = load_32b_with_adapters(args)
    novelty = generate_classifier(
        model32,
        tokenizer32,
        augmented_document,
        "novelty",
        args.max_input_tokens,
        args.classification_max_new_tokens,
    )
    inventive = generate_classifier(
        model32,
        tokenizer32,
        augmented_document,
        "inventive",
        args.max_input_tokens,
        args.classification_max_new_tokens,
    )

    novelty_pred = novelty["pred"]
    inventive_pred = inventive["pred"]
    decision = decision_from_preds(novelty_pred, inventive_pred)
    class_label = class_from_preds(novelty_pred, inventive_pred)

    result = {
        "input_id": input_id,
        "target_text": target_text,
        "retrieval": {
            "sbert_model": args.sbert_model,
            "sbert_index": args.sbert_index,
            "top_k": args.top_k,
            "references": [compact_reference(item) for item in retrieved],
        },
        "novelty": {"prediction": novelty_pred, "raw": novelty["raw"], "parsed": novelty["parsed"]},
        "inventive_step": {"prediction": inventive_pred, "raw": inventive["raw"], "parsed": inventive["parsed"]},
        "decision": decision,
        "class_label": class_label,
        "rule": "등록 if novelty=만족 and inventive_step=만족 else 거절; class priority is 신규성 부족 first, then 진보성 부족, then 등록.",
        "subclass": {"triggered": False, "labels": [], "reason": "inventive_step prediction is not 부족"},
        "model_paths": {
            "base_model": base_model,
            "novelty_adapter": args.novelty_adapter,
            "inventive_adapter": args.inventive_adapter,
            "subclass_adapter": args.subclass_adapter,
        },
        "raw_input": raw_input,
    }

    if inventive_pred == "부족" and not args.skip_subclass:
        subclass_model, subclass_tokenizer, thresholds = load_subclass_model(args.subclass_adapter, args.device_map)
        result["subclass"] = predict_subclass(
            subclass_model,
            subclass_tokenizer,
            thresholds,
            target_text,
            retrieved,
            args.subclass_max_length,
        )
        del subclass_model, subclass_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    if not args.skip_explanation:
        result["explanation"] = generate_explanation(
            model32,
            tokenizer32,
            target_text,
            retrieved,
            result,
            args.explanation_max_input_tokens,
            args.explanation_max_new_tokens,
            args.sample_explanation,
        )

    output_text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output_text + "\n", encoding="utf-8")
    print(output_text, flush=True)

    del model32, tokenizer32
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
