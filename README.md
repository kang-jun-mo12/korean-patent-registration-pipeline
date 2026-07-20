# 특허나침반 — 한국어 특허 심사 보조 AI

한국어 특허 문헌을 입력받아 선행기술을 검색하고 신규성·진보성·거절 세부 유형을 추론하는 연구용 파이프라인입니다.

파이프라인은 다음 구성요소를 사용합니다.

1. 학습된 SBERT로 약 29,794건의 선행기술에서 유사 문헌 2건 검색
2. Qwen2.5-32B LoRA 어댑터로 신규성 판단
3. Qwen2.5-32B LoRA 어댑터로 진보성 판단
4. 진보성 부족일 때 Qwen2.5-7B 멀티라벨 LoRA로 세부 유형 분류
5. LoRA를 비활성화한 32B 기반 모델로 결과 설명 생성

> 이 프로젝트는 연구 및 심사 보조용입니다. 결과는 법률 자문이나 특허 등록 가능성의 보증이 아닙니다.

## 데이터셋 수집 및 구축

### 확인된 원천과 범위

남아 있는 학습 기록에서 확인되는 원본은 `patent_30000_with_2_prior_art_claims_fixed` 데이터셋입니다. 총 30,000개의 심사대상 특허로 구성되며, 각 샘플에는 심사대상 문헌 1건과 연결된 선행문헌 D1·D2 두 건이 포함되어 있습니다.

모델 입력에는 다음 텍스트를 사용했습니다.

```text
심사대상 문헌: 발명의 명칭 + 초록 + 청구항
선행문헌 D1:  발명의 명칭 + 초록 + 청구항
선행문헌 D2:  발명의 명칭 + 초록 + 청구항
```

문헌번호는 모델이 번호 패턴을 정답의 지름길로 학습하지 않도록 분류 모델 입력에서 제외했습니다. SBERT 검색 학습은 긴 문헌 전체 대신 `발명의 명칭 + 초록 + 청구항 1`을 중심으로 구성했습니다.

원본 30,000건의 클래스 구성은 균형을 맞췄습니다.

| 원본 클래스 | 건수 | 신규성 라벨 | 진보성 라벨 |
|---|---:|---|---|
| 등록 | 10,000 | 만족 | 만족 |
| 거절-신규성 부족 | 10,000 | 부족 | 학습 제외 |
| 거절-진보성 부족 | 10,000 | 만족 | 부족 |
| 합계 | **30,000** |  |  |

> 현재 보관된 자료에는 최초 특허 원문을 어느 사이트·API·검색 조건·수집 기간으로 확보했는지까지는 남아 있지 않습니다. 따라서 이 저장소는 원천 수집 과정 전체가 재현된다고 주장하지 않으며, 확인 가능한 30,000건 이후의 정제·라벨 구성·학습 과정을 문서화합니다.

### 태스크별 데이터 구성

하나의 30,000건 데이터에서 모델별 목적에 맞는 학습셋을 따로 만들었습니다.

| 태스크 | 구축 방법 | 학습/검증/평가 규모 |
|---|---|---|
| SBERT 선행기술 검색 | target마다 D1·D2를 positive pair로 전개해 60,000 pair 생성 | train 54,000 pair, eval target 3,000건 |
| 신규성 분류 | 신규성 부족 10,000건 + 등록 5,000건 + 진보성 부족 5,000건 | train 18,000 / validation 1,000 / test 1,000 |
| 진보성 분류 | 진보성 부족 10,000건 + 등록 10,000건, 신규성 부족은 제외 | train 18,000 / validation 1,000 / test 1,000 |
| 진보성 세부 유형 | 진보성 부족 10,000건을 기존 subtype과 매칭하고 근거 라벨이 있는 샘플만 사용 | 매칭 4,215 / 사용 가능 3,318 / train 2,654 / validation 332 / test 332 |
| 최종 파이프라인 평가 | 등록·거절 균형 평가셋에서 검색부터 최종 판정까지 실행 | 총 1,000건: 등록 500 / 신규성 부족 250 / 진보성 부족 250 |

SBERT용 선행문헌은 D1·D2를 모두 펼치면 60,000행이며, 중복을 제거한 실제 검색 corpus는 **29,794건**입니다.

신규성 만족 데이터는 등록 문헌만 사용하지 않고 `등록 5,000 + 진보성 부족 5,000`으로 구성했습니다. 모델이 등록/거절 자체를 외우는 대신 신규성이라는 별도 판단 기준을 학습하게 하기 위한 구성입니다. 진보성 학습에서는 단일 문헌만으로 이미 신규성이 부족한 10,000건을 제외했습니다.

진보성 세부 유형은 `subtypes_cleaned.jsonl`의 기존 subtype을 출원번호 기준으로 매칭한 grounded-only 데이터입니다. pseudo-label은 최종 기준 모델 학습에서 제외하고 다음 네 개 멀티라벨로 통합했습니다.

```text
결합류       = 결합용이성 + 단순결합
차이류       = 차이없음 + 시너지부정
설계변경     = 설계변경
공지기술부가 = 공지기술부가 + 단순부가
```

support가 매우 작은 `수치한정`, `균등치환`, `기능표현`은 학습 라벨에서 제외했습니다.

## 모델 학습

모든 최종 학습은 H200 환경을 기준으로 진행했습니다. 신규성과 진보성을 하나의 등록/거절 모델로 합치지 않고 독립된 어댑터로 학습한 뒤 마지막에 규칙으로 결합했습니다.

### SBERT 선행기술 검색

기반 모델은 `jhgan/ko-sroberta-multitask`입니다.

1. **Stage 1:** 54,000 positive pair에 `MultipleNegativesRankingLoss`를 적용해 3 epoch 학습했습니다. 최종 보고서 기준 batch size 256, learning rate `2e-5`입니다.
2. **Hard-negative mining:** Stage 1 모델로 전체 corpus를 검색하고, 정답 D1·D2가 아닌 결과 중 rank 20~200 구간에서 query당 hard negative 3개를 선택했습니다.
3. **Stage 2:** positive와 hard negative를 함께 사용해 `CachedMultipleNegativesRankingLoss`로 2 epoch 추가 학습했습니다. batch size 256, mini batch size 32, learning rate `1e-5`입니다.
4. **인덱싱:** 최종 SBERT로 29,794건 corpus 임베딩을 생성해 cosine similarity 기반 top-k 검색 인덱스를 만들었습니다.

### 신규성·진보성 Qwen2.5-32B LoRA

두 분류기는 `unsloth/Qwen2.5-32B-Instruct`를 기반으로 별도 학습했습니다.

| 설정 | 값 |
|---|---|
| 학습 방식 | supervised fine-tuning, response-only |
| 정밀도 | BF16 일반 LoRA, 4bit QLoRA 아님 |
| 입력 길이 | 최대 4,096 tokens |
| batch | per-device 8, gradient accumulation 2, effective 16 |
| epochs | 최대 2 |
| learning rate | `1e-4` |
| scheduler | cosine, warmup ratio 0.03 |
| LoRA | rank 32, alpha 64, dropout 0 |
| target modules | q/k/v/o projection + gate/up/down projection |
| 모델 선택 | validation loss가 가장 낮은 checkpoint |

베이스 모델은 freeze하고 LoRA 파라미터만 업데이트했습니다. assistant 정답 부분만 loss에 반영했으며 출력은 설명문 없이 다음 JSON 라벨 하나만 생성하도록 학습했습니다.

```json
{"novelty": "만족|부족"}
{"inventive_step": "만족|부족"}
```

최종 신규성 어댑터는 step 1,500에서 best validation loss 0.0336, 진보성 어댑터는 step 2,000에서 best validation loss 0.0255를 기록한 checkpoint를 사용했습니다.

### 진보성 세부 유형 Qwen2.5-7B LoRA

`Qwen/Qwen2.5-7B-Instruct`에 4-label sequence-classification head를 붙이고 PEFT LoRA로 학습했습니다.

| 설정 | 값 |
|---|---|
| 구조 | `AutoModelForSequenceClassification`, 4 labels |
| loss | `BCEWithLogitsLoss` |
| 정밀도 | BF16 |
| batch | per-device 8, gradient accumulation 2, effective 16 |
| epochs | 최대 5 |
| learning rate | `2e-5` |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| 모델 선택 | validation macro F1, early stopping patience 5 |

validation set으로 라벨별 sigmoid threshold를 보정했으며, threshold를 넘는 라벨이 없으면 확률 상위 2개를 fallback으로 제시합니다.

## 평가 방법

최종 성능은 정답 선행문헌을 모델에 직접 제공하는 gold-prior 조건이 아닙니다. 사용자가 입력한 문헌으로 SBERT가 29,794건에서 top-2를 직접 검색하고, 검색된 문헌을 신규성·진보성 어댑터에 전달한 **end-to-end retrieval 조건**입니다.

최종 판정 규칙은 다음과 같습니다.

```text
등록 = 신규성 만족 AND 진보성 만족
거절 = 그 외 모든 경우
```

`타이트 정확도`는 등록/거절 판정뿐 아니라 진보성 부족 샘플의 세부 유형까지 맞아야 정답으로 인정합니다.

SBERT는 두 기준을 구분해 평가했습니다.

- **Strict recall:** 검색 결과의 문헌 ID가 라벨 D1·D2와 정확히 일치
- **Soft Hit:** 다른 문헌이더라도 정답 선행과의 임베딩 유사도가 random prior pair 분포의 지정 분위보다 높으면 인정

soft 지표는 의미적으로 가까운 대체 선행문헌을 반영하지만 strict recall을 대체하는 동일 지표는 아닙니다.

## 최종 성능

| 평가 항목 | 최종 성능 | 평가 의미 |
|---|---:|---|
| 최종 등록/거절 정확도 | **87.5%** | 1,000건 end-to-end 평가 |
| 타이트 정확도 | **86.3%** | 최종 판정 + 진보성 세부 유형 |
| 신규성 어댑터 정확도 | **84.0%** | 신규성 test 1,000건 |
| 진보성 어댑터 정확도 | **91.0%** | 진보성 test 1,000건 |
| SBERT strict recall@1 | **29.37%** | 정확한 D1·D2 ID 기준 |
| SBERT strict recall@2 | **40.67%** | 정확한 D1·D2 ID 기준 |
| SBERT strict recall@5 | **57.97%** | 정확한 D1·D2 ID 기준 |
| SBERT soft Hit@2 | **85.1%** | random similarity p99.5 기준 |
| SBERT top-2 exact-or-same-IPC-class | **88.73%** | 동일 IPC class를 포함한 보조 지표 |
| 세부 유형 Hit@1 | **73.5%** | 최상위 라벨 적중 |
| 세부 유형 Hit@2 | **92.7%** | 상위 2개 중 정답 라벨 포함 |
| 세부 유형 Macro F1 | **62.3%** | 4개 라벨 평균 F1 |

87.5%는 검색 단계의 오류까지 포함한 전체 시스템 성능입니다. 다만 평가셋이 등록 500건과 거절 500건으로 균형 구성되어 실제 출원 분포와 다를 수 있고, soft retrieval 및 IPC 지표는 법적인 선행기술 타당성을 의미하지 않습니다.

더 자세한 실험 설정과 수치는 [최종 파이프라인 보고서](END/FINAL_PATENT_PIPELINE_REPORT.md), [SBERT 추가 설명](END/SBERT성능추가설명.txt), [32B LoRA 학습 설명](lora_sbert_training_code_bundle/lora_adapters/h200_bf16_lora/PATENT30000_H200_ADAPTER_TRAINING_EXPLANATION.md), [서브라벨 학습 가이드](lora_sbert_training_code_bundle/subclass_classifier_7b/TRAINING_GUIDE.md)에서 확인할 수 있습니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `END/final_patent_inferencer.py` | 명령행 추론기 |
| `END/final_patent_web_server.py` | 로컬 웹 UI와 API |
| `END/reconstructed_training` | 남아 있는 산출물을 바탕으로 재구성한 학습 코드 |
| `lora_sbert_training_code_bundle` | 별도 보관되어 있던 원본 학습·평가 스크립트와 문서 |
| `END/FINAL_PATENT_PIPELINE_REPORT.md` | 파이프라인 및 평가 상세 보고서 |
| `MODEL_CARD.md` | 모델 구성, 성능, 한계 |

이 공개 GitHub 저장소에는 코드와 문서만 포함합니다. 다음 로컬 모델·데이터 경로는 재배포 권한을 확인할 때까지 `.gitignore`로 제외되어 있습니다.

- `END/checkpoint-1500`: 신규성 Qwen2.5-32B LoRA
- `END/checkpoint-2000`: 진보성 Qwen2.5-32B LoRA
- `END/handoff_to_friend/subclass_adapter`: 진보성 부족 세부 유형 Qwen2.5-7B LoRA
- `END/patent30000_sbert_h200/models`: 학습된 SBERT 검색 모델
- `END/patent30000_sbert_h200/index`: 검색 corpus와 임베딩 인덱스
- `END/final_eval_sample_lookup.jsonl`: 문서 원문이 포함된 샘플별 평가 결과

따라서 저장소를 clone한 것만으로는 전체 추론 파이프라인이 실행되지 않습니다. 모델 자산을 별도로 내려받아 위 기본 경로에 배치하거나 실행 인자로 경로를 지정해야 합니다.

## 필요한 기반 모델

추론에는 이 프로젝트에서 학습된 LoRA·SBERT 산출물과 다음 기반 모델이 모두 필요합니다. 기반 모델은 경로가 올바르면 첫 실행 시 Hugging Face에서 자동으로 내려받지만, 프로젝트 학습 산출물은 현재 공개 저장소에서 제공하지 않습니다.

- [`unsloth/Qwen2.5-32B-Instruct`](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct)
- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

인터넷 연결과 기반 모델을 저장할 충분한 디스크 공간이 필요합니다.

## 하드웨어

기본 실행 설정은 BF16입니다. 32B 기반 모델 자체가 약 64GB의 파라미터 메모리를 사용하며, 진보성 부족 세부 분류 단계에서는 7B 모델도 추가로 로드됩니다.

- 권장: H200급 단일 GPU 또는 합산 메모리가 충분한 다중 GPU 환경
- 80GB GPU: 설정과 메모리 오버헤드에 따라 빠듯하거나 부족할 수 있음
- CPU offload: `device_map=auto`가 사용할 수 있지만 매우 느릴 수 있음
- 세부 분류를 생략하려면 `--skip-subclass` 사용

## 설치

Git을 설치한 다음 저장소를 clone합니다.

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

가상환경을 만들고 패키지를 설치합니다.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA 환경에 맞는 PyTorch 빌드가 필요하면 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에 따라 PyTorch를 먼저 설치한 뒤 나머지 패키지를 설치하세요.

## 웹 UI 실행

모델과 검색 인덱스를 먼저 기본 경로에 배치한 다음 저장소 루트에서 실행합니다.

```powershell
Set-Location .\END
python .\final_patent_web_server.py --host 127.0.0.1 --port 7860
```

브라우저에서 `http://127.0.0.1:7860`을 엽니다. 모델은 서버 시작 시점이 아니라 첫 추론 요청 때 로드됩니다.

Linux:

```bash
cd END
python final_patent_web_server.py --host 127.0.0.1 --port 7860
```

## 명령행 실행

`END` 폴더에서:

```powershell
python .\final_patent_inferencer.py `
  --title "발명의 명칭" `
  --abstract "발명의 초록" `
  --claim-1 "청구항 1의 내용" `
  --output .\final_inference_output.json
```

텍스트 파일 또는 JSON 입력도 지원합니다.

```powershell
python .\final_patent_inferencer.py --text-file .\input.txt
python .\final_patent_inferencer.py --input-json .\input.json
```

JSON 입력 예:

```json
{
  "id": "sample-001",
  "title": "발명의 명칭",
  "abstract": "발명의 초록",
  "claim_1": "청구항 1"
}
```

## 평가와 학습 코드

- 전체 보고서: [`END/FINAL_PATENT_PIPELINE_REPORT.md`](END/FINAL_PATENT_PIPELINE_REPORT.md)
- SBERT 성능 상세: [`END/SBERT성능추가설명.txt`](END/SBERT성능추가설명.txt)
- 재구성 학습 코드 사용법: [`END/reconstructed_training/README.md`](END/reconstructed_training/README.md)
- 보관된 학습 코드 묶음: [`lora_sbert_training_code_bundle/README_BUNDLE.md`](lora_sbert_training_code_bundle/README_BUNDLE.md)

원본 train/validation/test 데이터 전체는 포함되어 있지 않습니다. 재학습에는 문서에 설명된 형식의 데이터가 별도로 필요합니다.

## GitHub 게시

대용량 ZIP, 모델 가중치, 검색 corpus·인덱스, 샘플별 평가 데이터는 `.gitignore`에서 제외했습니다. 저장소 생성과 push 절차 및 공개 범위는 `docs/PUBLISHING.md`를 참고하세요.

## 라이선스

이 저장소에는 아직 프로젝트 라이선스가 선택되지 않았습니다. 공개 배포 전에 코드, 학습 산출물, 검색 corpus를 어떤 조건으로 재사용할 수 있는지 정한 뒤 루트에 `LICENSE`를 추가해야 합니다.

확인 시점의 Hugging Face 모델 페이지에서 두 Qwen 기반 모델은 Apache-2.0으로 표시됩니다. [`jhgan/ko-sroberta-multitask`](https://huggingface.co/jhgan/ko-sroberta-multitask) 페이지에는 라이선스 표기가 보이지 않으므로 SBERT 파생 가중치를 공개하기 전에 재배포 조건을 별도로 확인해야 합니다.
