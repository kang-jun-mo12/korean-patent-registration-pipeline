#!/usr/bin/env python3
import argparse
import json
import re
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlparse

import gc
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from final_patent_inferencer import (
    build_augmented_document,
    class_from_preds,
    compact_reference,
    decision_from_preds,
    disabled_adapter_context,
    format_target,
    generate_classifier,
    input_device,
    load_32b_with_adapters,
    load_corpus_meta,
    load_subclass_model,
    normalize_target_text,
    predict_subclass,
)


HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Patent Inference</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --panel-alt: #f9fafb;
      --line: #d9dee5;
      --text: #17202a;
      --muted: #5d6877;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
      --success: #166534;
      --warn: #92400e;
      --shadow: 0 16px 38px rgba(20, 31, 46, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button, textarea, input {
      font: inherit;
      letter-spacing: 0;
    }

    header {
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }

    .topbar {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }

    .brand {
      display: flex;
      flex-direction: column;
      gap: 2px;
      min-width: 0;
    }

    h1 {
      margin: 0;
      font-size: 21px;
      line-height: 1.2;
      font-weight: 760;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.35;
    }

    .status-pill {
      border: 1px solid var(--line);
      background: var(--panel-alt);
      border-radius: 999px;
      padding: 7px 11px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(360px, 0.9fr) minmax(460px, 1.1fr);
      gap: 18px;
      align-items: start;
    }

    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }

    .section-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }

    h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.25;
      font-weight: 740;
    }

    .body {
      padding: 18px;
    }

    label {
      display: block;
      margin: 0 0 7px;
      color: #26313f;
      font-size: 13px;
      font-weight: 680;
    }

    input, textarea {
      width: 100%;
      border: 1px solid #cbd3dd;
      border-radius: 7px;
      background: #ffffff;
      color: var(--text);
      outline: none;
      transition: border-color 120ms ease, box-shadow 120ms ease;
    }

    input {
      height: 42px;
      padding: 9px 11px;
    }

    textarea {
      min-height: 156px;
      resize: vertical;
      padding: 11px;
      line-height: 1.48;
    }

    textarea.claim {
      min-height: 250px;
    }

    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }

    .field {
      margin-bottom: 16px;
    }

    .actions {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: flex-end;
      margin-top: 18px;
    }

    button {
      border: 1px solid transparent;
      border-radius: 7px;
      height: 42px;
      padding: 0 16px;
      background: var(--accent);
      color: #ffffff;
      font-weight: 720;
      cursor: pointer;
      min-width: 118px;
    }

    button:hover {
      background: var(--accent-strong);
    }

    button:disabled {
      opacity: 0.58;
      cursor: wait;
    }

    .secondary {
      background: #ffffff;
      color: #2f3b4a;
      border-color: var(--line);
      min-width: 92px;
    }

    .secondary:hover {
      background: #f3f5f7;
    }

    .status-box {
      border: 1px solid var(--line);
      background: var(--panel-alt);
      border-radius: 8px;
      padding: 13px;
      display: grid;
      gap: 8px;
    }

    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      font-size: 13px;
      color: var(--muted);
    }

    .status-row strong {
      color: #26313f;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }

    .metric {
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 8px;
      padding: 12px;
      min-height: 86px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 9px;
    }

    .metric .name {
      color: var(--muted);
      font-size: 12px;
      font-weight: 680;
    }

    .metric .value {
      font-size: 19px;
      line-height: 1.18;
      font-weight: 800;
      word-break: keep-all;
    }

    .value.register {
      color: var(--success);
    }

    .value.reject, .value.lack {
      color: var(--danger);
    }

    .value.warn {
      color: var(--warn);
    }

    .explanation {
      white-space: pre-wrap;
      line-height: 1.6;
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 8px;
      padding: 15px;
      min-height: 180px;
      color: #202b38;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 10px 0 14px;
    }

    .chip {
      border: 1px solid #b9c4d2;
      background: #f7f9fb;
      color: #26313f;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
    }

    .refs {
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }

    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #ffffff;
      overflow: hidden;
    }

    summary {
      cursor: pointer;
      padding: 12px 13px;
      color: #26313f;
      font-weight: 720;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      list-style: none;
    }

    summary::-webkit-details-marker {
      display: none;
    }

    .ref-meta {
      color: var(--muted);
      font-weight: 650;
      font-size: 12px;
      white-space: nowrap;
    }

    .ref-text {
      border-top: 1px solid var(--line);
      padding: 13px;
      white-space: pre-wrap;
      color: #253141;
      line-height: 1.55;
      max-height: 320px;
      overflow: auto;
      background: #fbfcfd;
      font-size: 13px;
    }

    .empty {
      color: var(--muted);
      border: 1px dashed #cbd3dd;
      border-radius: 8px;
      padding: 18px;
      min-height: 180px;
      display: flex;
      align-items: center;
      justify-content: center;
      text-align: center;
    }

    .error {
      border: 1px solid #f1a8a0;
      background: #fff4f2;
      color: #8f1d14;
      border-radius: 8px;
      padding: 12px;
      line-height: 1.45;
      display: none;
      margin-top: 12px;
    }

    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
        padding: 16px;
      }

      .topbar {
        padding: 15px 16px;
        align-items: flex-start;
        flex-direction: column;
      }

      .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }

    @media (max-width: 560px) {
      .metrics {
        grid-template-columns: 1fr;
      }

      .actions {
        flex-direction: column-reverse;
      }

      button {
        width: 100%;
      }
    }
  </style>
</head>
<body>
  <header>
    <div class="topbar">
      <div class="brand">
        <h1>특허 최종 추론기</h1>
        <p class="subtitle">선행기술 검색 · 신규성 판단 · 진보성 판단 · 세부 분류 · 설명 생성</p>
      </div>
      <div id="serverState" class="status-pill">대기</div>
    </div>
  </header>

  <main>
    <section>
      <div class="section-head">
        <h2>입력</h2>
      </div>
      <div class="body">
        <form id="patentForm">
          <div class="field">
            <label for="title">발명의 명칭</label>
            <input id="title" name="title" autocomplete="off">
          </div>
          <div class="field">
            <label for="abstract">초록</label>
            <textarea id="abstract" name="abstract"></textarea>
          </div>
          <div class="field">
            <label for="claim">청구항 1</label>
            <textarea id="claim" name="claim_1" class="claim"></textarea>
          </div>
          <div class="status-box">
            <div class="status-row"><span>상태</span><strong id="jobStatus">대기</strong></div>
            <div class="status-row"><span>단계</span><strong id="jobStage">입력 대기</strong></div>
          </div>
          <div id="errorBox" class="error"></div>
          <div class="actions">
            <button type="button" id="clearButton" class="secondary">초기화</button>
            <button type="submit" id="submitButton">추론 실행</button>
          </div>
        </form>
      </div>
    </section>

    <section>
      <div class="section-head">
        <h2>결과</h2>
      </div>
      <div class="body">
        <div id="emptyResult" class="empty">결과 없음</div>
        <div id="resultPanel" style="display:none">
          <div class="metrics">
            <div class="metric">
              <div class="name">최종</div>
              <div id="decisionValue" class="value">-</div>
            </div>
            <div class="metric">
              <div class="name">신규성</div>
              <div id="noveltyValue" class="value">-</div>
            </div>
            <div class="metric">
              <div class="name">진보성</div>
              <div id="inventiveValue" class="value">-</div>
            </div>
            <div class="metric">
              <div class="name">분류</div>
              <div id="classValue" class="value">-</div>
            </div>
          </div>

          <div id="subclassChips" class="chips"></div>
          <div id="explanation" class="explanation"></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const form = document.getElementById("patentForm");
    const submitButton = document.getElementById("submitButton");
    const clearButton = document.getElementById("clearButton");
    const errorBox = document.getElementById("errorBox");
    const jobStatus = document.getElementById("jobStatus");
    const jobStage = document.getElementById("jobStage");
    const serverState = document.getElementById("serverState");
    const emptyResult = document.getElementById("emptyResult");
    const resultPanel = document.getElementById("resultPanel");
    const decisionValue = document.getElementById("decisionValue");
    const noveltyValue = document.getElementById("noveltyValue");
    const inventiveValue = document.getElementById("inventiveValue");
    const classValue = document.getElementById("classValue");
    const subclassChips = document.getElementById("subclassChips");
    const explanation = document.getElementById("explanation");
    let pollTimer = null;

    function showError(message) {
      errorBox.textContent = message;
      errorBox.style.display = message ? "block" : "none";
    }

    function setBusy(busy) {
      submitButton.disabled = busy;
      submitButton.textContent = busy ? "진행 중" : "추론 실행";
      serverState.textContent = busy ? "실행 중" : "대기";
    }

    function setValue(el, value, kind) {
      el.textContent = value || "-";
      el.className = "value";
      if (kind === "decision") {
        el.classList.add(value === "등록" ? "register" : "reject");
      } else if (kind === "binary") {
        el.classList.add(value === "만족" ? "register" : "lack");
      } else if (kind === "class") {
        if (value === "등록") el.classList.add("register");
        else if (value && value.includes("신규성")) el.classList.add("reject");
        else if (value && value.includes("진보성")) el.classList.add("warn");
      }
    }

    function renderResult(result) {
      emptyResult.style.display = "none";
      resultPanel.style.display = "block";
      setValue(decisionValue, result.decision, "decision");
      setValue(noveltyValue, result.novelty && result.novelty.prediction, "binary");
      setValue(inventiveValue, result.inventive_step && result.inventive_step.prediction, "binary");
      setValue(classValue, result.class_label, "class");

      subclassChips.innerHTML = "";
      const labels = result.subclass && result.subclass.labels ? result.subclass.labels : [];
      if (labels.length) {
        labels.forEach(label => {
          const chip = document.createElement("span");
          chip.className = "chip";
          chip.textContent = label;
          subclassChips.appendChild(chip);
        });
      }

      explanation.textContent = result.explanation || "설명 생성 결과 없음";
    }

    async function pollJob(jobId) {
      const res = await fetch(`/api/jobs/${jobId}`);
      if (!res.ok) throw new Error(`작업 상태 조회 실패: ${res.status}`);
      const job = await res.json();
      jobStatus.textContent = job.status || "-";
      jobStage.textContent = job.stage || "-";
      if (job.status === "done") {
        clearInterval(pollTimer);
        pollTimer = null;
        setBusy(false);
        renderResult(job.result);
      } else if (job.status === "error") {
        clearInterval(pollTimer);
        pollTimer = null;
        setBusy(false);
        showError(job.error || "추론 중 오류가 발생했습니다.");
      }
    }

    form.addEventListener("submit", async event => {
      event.preventDefault();
      showError("");
      const payload = {
        title: document.getElementById("title").value,
        abstract: document.getElementById("abstract").value,
        claim_1: document.getElementById("claim").value,
      };
      if (!payload.abstract.trim() && !payload.claim_1.trim()) {
        showError("초록 또는 청구항 1을 입력하세요.");
        return;
      }

      setBusy(true);
      jobStatus.textContent = "queued";
      jobStage.textContent = "작업 등록";
      try {
        const res = await fetch("/api/jobs", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.error || `요청 실패: ${res.status}`);
        if (pollTimer) clearInterval(pollTimer);
        await pollJob(body.id);
        pollTimer = setInterval(() => pollJob(body.id).catch(err => {
          clearInterval(pollTimer);
          pollTimer = null;
          setBusy(false);
          showError(err.message);
        }), 2000);
      } catch (err) {
        setBusy(false);
        showError(err.message);
      }
    });

    clearButton.addEventListener("click", () => {
      form.reset();
      showError("");
      jobStatus.textContent = "대기";
      jobStage.textContent = "입력 대기";
      emptyResult.style.display = "flex";
      resultPanel.style.display = "none";
    });
  </script>
</body>
</html>
"""


WEB_EXPLAINER_SYSTEM = {
    "role": "system",
    "content": "당신은 한국 특허 심사 결과를 사용자가 이해하기 쉽게 설명하는 특허 심사 보조자입니다. 제공된 심사대상 문헌, 검색된 선행기술, 모델 판정값만 근거로 설명하고 없는 사실은 만들지 않습니다.",
}


def web_explanation_prompt(target_text, retrieved, result_core):
    refs = []
    for idx, item in enumerate(retrieved, 1):
        text = (item.get("text", "") or "")[:1800]
        refs.append(
            "선행기술 {rank}\n{title}\n{text}".format(
                rank=idx,
                title=item.get("invention_title", ""),
                text=text,
            )
        )
    subclass = result_core.get("subclass") or {}
    subclass_labels = subclass.get("labels") or []
    return """아래는 특허 심사 보조 파이프라인의 결과입니다. 사용자가 이해하기 쉬운 한국어 설명으로 정리하세요.

[심사대상]
{target}

[검색된 선행기술]
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
3. 등록으로 나온 경우에는 "현재 검색된 선행기술 기준"이라는 한계를 분명히 쓰세요.
4. 거절로 나온 경우에는 어떤 판단값 때문에 거절인지 쉽게 설명하세요.
5. 서브라벨이 있으면 그 의미를 한두 문장으로 풀어주세요.
6. 과도하게 단정하지 말고, 모델 기반 예비 판단임을 마지막에 짧게 덧붙이세요.
""".format(
        target=target_text[:3000],
        refs="\n\n".join(refs),
        novelty=result_core["novelty"]["prediction"],
        inventive=result_core["inventive_step"]["prediction"],
        decision=result_core["decision"],
        class_label=result_core["class_label"],
        subclass_labels=", ".join(subclass_labels) if subclass_labels else "없음/미실행",
    )


def sanitize_explanation(text):
    text = (text or "").strip()
    replacements = [
        (r"TOP\s*2\s*선행\s*문헌", "검색된 선행기술"),
        (r"TOP\s*1\s*선행\s*문헌", "검색된 선행기술"),
        (r"TOP\s*2\s*선행기술", "검색된 선행기술"),
        (r"TOP\s*1\s*선행기술", "검색된 선행기술"),
        (r"TOP\s*2", "검색된 선행기술"),
        (r"TOP\s*1", "검색된 선행기술"),
        (r"D\s*1\s*또는\s*D\s*2", "검색된 선행기술"),
        (r"D\s*1\s*과\s*D\s*2", "검색된 선행기술"),
        (r"D\s*1\s*,\s*D\s*2", "검색된 선행기술"),
        (r"D\s*1\s*/\s*D\s*2", "검색된 선행기술"),
        (r"\bD\s*1\b", "선행기술"),
        (r"\bD\s*2\b", "선행기술"),
        (r"선행문헌", "선행기술"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"검색된\s+검색된\s+선행기술", "검색된 선행기술", text)
    text = re.sub(r"선행기술\s+선행기술", "선행기술", text)
    return text.strip()


def generate_web_explanation(model, tokenizer, target_text, retrieved, result_core, max_input_tokens, max_new_tokens, sample):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    messages = [WEB_EXPLAINER_SYSTEM, {"role": "user", "content": web_explanation_prompt(target_text, retrieved, result_core)}]
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
    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return sanitize_explanation(generated)


def format_web_target(title, abstract, claim_1):
    return "[심사대상 문헌]\n[발명의 명칭]\n{title}\n\n[초록]\n{abstract}\n\n[청구항]\n{claim_1}".format(
        title=(title or "").strip(),
        abstract=(abstract or "").strip(),
        claim_1=(claim_1 or "").strip(),
    ).strip()


def now_ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def make_config(args):
    return SimpleNamespace(
        sbert_model=args.sbert_model,
        sbert_index=args.sbert_index,
        top_k=args.top_k,
        novelty_adapter=args.novelty_adapter,
        inventive_adapter=args.inventive_adapter,
        subclass_adapter=args.subclass_adapter,
        base_model=args.base_model,
        device_map=args.device_map,
        dtype=args.dtype,
        max_input_tokens=args.max_input_tokens,
        classification_max_new_tokens=args.classification_max_new_tokens,
        subclass_max_length=args.subclass_max_length,
        explanation_max_input_tokens=args.explanation_max_input_tokens,
        explanation_max_new_tokens=args.explanation_max_new_tokens,
        sample_explanation=args.sample_explanation,
        skip_subclass=args.skip_subclass,
        skip_explanation=args.skip_explanation,
    )


class PatentPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.meta = None
        self.embeddings = None
        self.sbert = None
        self.model32 = None
        self.tokenizer32 = None
        self.base_model = None
        self.subclass_model = None
        self.subclass_tokenizer = None
        self.subclass_thresholds = None

    def load_retriever(self, update):
        if self.sbert is not None:
            return
        update("선행기술 검색 준비")
        self.meta = load_corpus_meta(self.cfg.sbert_index)
        self.embeddings = np.load(f"{self.cfg.sbert_index}/corpus_embeddings.npy").astype("float32")
        update("검색 모델 로딩")
        self.sbert = SentenceTransformer(self.cfg.sbert_model)

    def retrieve(self, target_text, update):
        self.load_retriever(update)
        update("선행기술 검색")
        q_emb = self.sbert.encode(
            [target_text],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        scores = (q_emb @ self.embeddings.T)[0]
        top_k = min(max(1, int(self.cfg.top_k)), len(scores))
        top = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        top = top[np.argsort(-scores[top])]
        refs = []
        for rank, corpus_idx in enumerate(top, 1):
            item = self.meta[int(corpus_idx)].copy()
            item["rank"] = rank
            item["score"] = float(scores[int(corpus_idx)])
            refs.append(item)
        return refs

    def load_32b(self, update):
        if self.model32 is not None:
            return
        update("판정 모델 로딩")
        self.model32, self.tokenizer32, self.base_model = load_32b_with_adapters(self.cfg)

    def load_subclass(self, update):
        if self.subclass_model is not None:
            return
        update("세부 분류 모델 로딩")
        self.subclass_model, self.subclass_tokenizer, self.subclass_thresholds = load_subclass_model(
            self.cfg.subclass_adapter,
            self.cfg.device_map,
        )

    def run(self, payload, update):
        update("입력 정리")
        title = (payload.get("title") or "").strip()
        abstract = (payload.get("abstract") or "").strip()
        claim_1 = (payload.get("claim_1") or payload.get("claim") or "").strip()
        target_text = normalize_target_text(format_web_target(title, abstract, claim_1))

        retrieved = self.retrieve(target_text, update)
        augmented_document = build_augmented_document(target_text, retrieved)

        self.load_32b(update)
        update("신규성 판정")
        novelty = generate_classifier(
            self.model32,
            self.tokenizer32,
            augmented_document,
            "novelty",
            self.cfg.max_input_tokens,
            self.cfg.classification_max_new_tokens,
        )
        update("진보성 판정")
        inventive = generate_classifier(
            self.model32,
            self.tokenizer32,
            augmented_document,
            "inventive",
            self.cfg.max_input_tokens,
            self.cfg.classification_max_new_tokens,
        )

        novelty_pred = novelty["pred"]
        inventive_pred = inventive["pred"]
        result = {
            "input_id": payload.get("input_id") or "web_input",
            "target_text": target_text,
            "retrieval": {
                "sbert_model": self.cfg.sbert_model,
                "sbert_index": self.cfg.sbert_index,
                "top_k": self.cfg.top_k,
                "references": [compact_reference(item) for item in retrieved],
            },
            "novelty": {"prediction": novelty_pred, "raw": novelty["raw"], "parsed": novelty["parsed"]},
            "inventive_step": {
                "prediction": inventive_pred,
                "raw": inventive["raw"],
                "parsed": inventive["parsed"],
            },
            "decision": decision_from_preds(novelty_pred, inventive_pred),
            "class_label": class_from_preds(novelty_pred, inventive_pred),
            "rule": "등록 if novelty=만족 and inventive_step=만족 else 거절.",
            "subclass": {"triggered": False, "labels": [], "reason": "inventive_step prediction is not 부족"},
            "model_paths": {
                "base_model": self.base_model,
                "novelty_adapter": self.cfg.novelty_adapter,
                "inventive_adapter": self.cfg.inventive_adapter,
                "subclass_adapter": self.cfg.subclass_adapter,
            },
        }

        if inventive_pred == "부족" and not self.cfg.skip_subclass:
            self.load_subclass(update)
            update("진보성 부족 서브라벨 분류")
            result["subclass"] = predict_subclass(
                self.subclass_model,
                self.subclass_tokenizer,
                self.subclass_thresholds,
                target_text,
                retrieved,
                self.cfg.subclass_max_length,
            )

        if not self.cfg.skip_explanation:
            update("설명 생성")
            result["explanation"] = generate_web_explanation(
                self.model32,
                self.tokenizer32,
                target_text,
                retrieved,
                result,
                self.cfg.explanation_max_input_tokens,
                self.cfg.explanation_max_new_tokens,
                self.cfg.sample_explanation,
            )

        update("완료")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result


class AppState:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.pipeline_lock = threading.Lock()

    def create_job(self, payload):
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id,
            "status": "queued",
            "stage": "GPU 작업 대기",
            "created_at": now_ts(),
            "updated_at": now_ts(),
            "input": payload,
            "result": None,
            "error": "",
        }
        with self.jobs_lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def update_job(self, job_id, **fields):
        with self.jobs_lock:
            job = self.jobs[job_id]
            job.update(fields)
            job["updated_at"] = now_ts()

    def get_job(self, job_id):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return json.loads(json.dumps(job, ensure_ascii=False))

    def _run_job(self, job_id):
        try:
            self.update_job(job_id, status="queued", stage="GPU 작업 대기")
            with self.pipeline_lock:
                self.update_job(job_id, status="running", stage="시작")

                def update(stage):
                    self.update_job(job_id, stage=stage)

                payload = self.get_job(job_id)["input"]
                result = self.pipeline.run(payload, update)
                self.update_job(job_id, status="done", stage="완료", result=result)
        except Exception as exc:
            self.update_job(
                job_id,
                status="error",
                stage="실패",
                error=str(exc),
                traceback=traceback.format_exc(),
            )


APP_STATE = None


class Handler(BaseHTTPRequestHandler):
    server_version = "PatentInferenceHTTP/1.0"

    def log_message(self, fmt, *args):
        print("[%s] %s - %s" % (now_ts(), self.address_string(), fmt % args), flush=True)

    def send_bytes(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, data):
        self.send_bytes(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_bytes(200, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
            return
        if path == "/api/health":
            self.send_json(200, {"ok": True, "time": now_ts()})
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = APP_STATE.get_job(job_id)
            if not job:
                self.send_json(404, {"error": "job not found"})
                return
            self.send_json(200, job)
            return
        self.send_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "invalid content length"})
            return
        if length <= 0 or length > 4 * 1024 * 1024:
            self.send_json(400, {"error": "invalid request size"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self.send_json(400, {"error": "invalid json"})
            return

        title = str(payload.get("title") or "")
        abstract = str(payload.get("abstract") or "")
        claim_1 = str(payload.get("claim_1") or payload.get("claim") or "")
        if not abstract.strip() and not claim_1.strip():
            self.send_json(400, {"error": "초록 또는 청구항 1을 입력하세요."})
            return

        job = APP_STATE.create_job({"title": title, "abstract": abstract, "claim_1": claim_1})
        self.send_json(202, {"id": job["id"], "status": job["status"], "stage": job["stage"]})


def parse_args():
    p = argparse.ArgumentParser(description="Local web UI for the final patent inference pipeline.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
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
    p.add_argument("--sample-explanation", action="store_true")
    p.add_argument("--skip-subclass", action="store_true")
    p.add_argument("--skip-explanation", action="store_true")
    return p.parse_args()


def main():
    global APP_STATE
    args = parse_args()
    pipeline = PatentPipeline(make_config(args))
    APP_STATE = AppState(pipeline)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Patent inference web server running on http://{args.host}:{args.port}", flush=True)
    print("Models are loaded lazily on the first inference request.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
