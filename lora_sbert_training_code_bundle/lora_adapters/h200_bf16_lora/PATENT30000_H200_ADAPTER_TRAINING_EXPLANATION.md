# H200 3만 특허 데이터셋 기반 신규성·진보성 어댑터 훈련 설명

이 문서는 원래 H200 학습 계획에 맞춰, `등록 10,000건 + 진보성 부족 거절 10,000건 + 신규성 부족 거절 10,000건`으로 구성한 3만 특허 데이터셋을 사용해 신규성 Adapter와 진보성 Adapter를 각각 학습한 과정을 정리한다.

이 문서의 기준은 `patent_30000_with_2_prior_art_claims_fixed` 데이터셋이다. 각 샘플에는 심사대상 문헌과 선행문헌 D1,D2 두 개가 포함되어 있다.

## 1. 전체 구조

하나의 모델이 등록/거절을 바로 출력하게 하지 않고, 특허 판단을 두 단계로 나누어 학습한다.

```text
심사대상 문헌 + 선행문헌 D1,D2
  -> 신규성 Adapter
  -> 진보성 Adapter
  -> 두 결과를 결합해 최종 등록/거절 판단
```

각 Adapter는 서로 다른 이진 분류를 담당한다.

```text
신규성 Adapter:
D1 또는 D2 중 하나의 단일 선행문헌이 심사대상 청구항의 핵심 구성요소를 실질적으로 모두 개시하는지 판단

진보성 Adapter:
D1,D2를 근거로 심사대상 청구항의 차이점이 통상의 기술자가 쉽게 도출 가능한지 판단
```

## 2. 원본 3만 데이터셋 구성

원본 데이터셋은 총 30,000건이다.

| 구분 | 건수 | 신규성 라벨 | 진보성 라벨 | 설명 |
| --- | ---: | --- | --- | --- |
| 등록 데이터 | 10,000 | 만족 | 만족 | 신규성도 만족, 진보성도 만족 |
| 진보성 부족 거절 | 10,000 | 만족 | 부족 | 신규성은 통과했지만 진보성 부족 |
| 신규성 부족 거절 | 10,000 | 부족 | 진보성 학습 제외 | 단일 선행문헌 기준 신규성 부족 |
| 합계 | 30,000 | - | - | H200 원래 학습 계획 기준 데이터 |

각 원본 샘플에는 다음 정보가 들어간다.

```text
심사대상 문헌:
  발명의 명칭
  초록
  청구항

선행문헌 D1:
  발명의 명칭
  초록
  청구항

선행문헌 D2:
  발명의 명칭
  초록
  청구항
```

문헌번호는 모델이 번호 패턴에 의존하지 않도록 입력에서 제외한다.

## 3. 신규성 Adapter 데이터 구성

신규성 Adapter는 3만 원본 중 20,000건을 사용한다.

| 라벨 | 구성 | 건수 |
| --- | --- | ---: |
| 부족 | 신규성 부족 거절 | 10,000 |
| 만족 | 등록 5,000 + 진보성 부족 거절 5,000 | 10,000 |
| 합계 | 균형 데이터 | 20,000 |

split은 균형을 유지해 나눈다.

| split | 만족 | 부족 | 합계 |
| --- | ---: | ---: | ---: |
| train | 9,000 | 9,000 | 18,000 |
| validation | 500 | 500 | 1,000 |
| test | 500 | 500 | 1,000 |
| 합계 | 10,000 | 10,000 | 20,000 |

신규성 Adapter의 출력은 다음 둘 중 하나다.

```json
{"novelty":"만족"}
```

```json
{"novelty":"부족"}
```

## 4. 진보성 Adapter 데이터 구성

진보성 Adapter도 3만 원본 중 20,000건을 사용한다.

| 라벨 | 구성 | 건수 |
| --- | --- | ---: |
| 부족 | 진보성 부족 거절 | 10,000 |
| 만족 | 등록 데이터 | 10,000 |
| 합계 | 균형 데이터 | 20,000 |

신규성 부족 거절 10,000건은 진보성 Adapter 학습에서 제외한다. 신규성 부족 샘플은 단일 문헌에 이미 핵심 구성이 개시된 케이스라, 진보성의 결합 용이성 판단을 학습시키는 데이터로는 적합하지 않기 때문이다.

split은 균형을 유지해 나눈다.

| split | 만족 | 부족 | 합계 |
| --- | ---: | ---: | ---: |
| train | 9,000 | 9,000 | 18,000 |
| validation | 500 | 500 | 1,000 |
| test | 500 | 500 | 1,000 |
| 합계 | 10,000 | 10,000 | 20,000 |

진보성 Adapter의 출력은 다음 둘 중 하나다.

```json
{"inventive_step":"만족"}
```

```json
{"inventive_step":"부족"}
```

## 5. 공통 System Prompt

원래 H200 3만 데이터셋 학습에서는 두 Adapter 모두 같은 system prompt 구조를 사용한다.

```text
당신은 한국 특허 심사 보조 모델입니다. 심사대상 문헌과 선행문헌 D1,D2의 발명의 명칭, 초록, 청구항만 근거로 판단합니다. 허용된 라벨만 사용하고 JSON 객체 하나만 출력합니다.
```

판단 기준과 작업 지시문은 user prompt 안에 들어간다. assistant 출력에는 판단 기준이나 설명 문장을 넣지 않고 정답 JSON 하나만 넣는다.

## 6. 신규성 Adapter User Prompt

```text
[심사대상 문헌]
[발명의 명칭]
{심사대상 발명의 명칭}

[초록]
{심사대상 초록}

[청구항]
{심사대상 청구항}

[선행문헌 D1]
[발명의 명칭]
{D1 발명의 명칭}

[초록]
{D1 초록}

[청구항]
{D1 청구항}

[선행문헌 D2]
[발명의 명칭]
{D2 발명의 명칭}

[초록]
{D2 초록}

[청구항]
{D2 청구항}

[작업]
심사대상 청구항의 핵심 구성요소가 D1 또는 D2 중 하나에 실질적으로 모두 개시되어 신규성이 부족한지 판단하라.

[판단 절차]
1. 심사대상 청구항의 핵심 구성요소와 구성 간 관계를 파악한다.
2. D1과 D2를 각각 따로 비교한다.
3. 신규성 판단에서는 D1과 D2를 조합하지 않는다.
4. 하나의 선행문헌에 핵심 구성요소 전부가 명시적 또는 실질적으로 개시되면 신규성 부족이다.
5. 선행문헌 하나만으로는 차이점이 남거나 여러 문헌의 조합이 필요하면 신규성 만족이다.

[출력 규칙]
- JSON 객체 하나만 출력한다.
- novelty는 "만족" 또는 "부족" 중 하나다.
- 설명, 근거 문장, markdown, 코드블록은 출력하지 않는다.

[출력 형식]
{"novelty":"만족|부족"}
```

## 7. 진보성 Adapter User Prompt

```text
[심사대상 문헌]
[발명의 명칭]
{심사대상 발명의 명칭}

[초록]
{심사대상 초록}

[청구항]
{심사대상 청구항}

[선행문헌 D1]
[발명의 명칭]
{D1 발명의 명칭}

[초록]
{D1 초록}

[청구항]
{D1 청구항}

[선행문헌 D2]
[발명의 명칭]
{D2 발명의 명칭}

[초록]
{D2 초록}

[청구항]
{D2 청구항}

[작업]
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
- inventive_step은 "만족" 또는 "부족" 중 하나다.
- 신규성 부족 전용 샘플은 이 어댑터 학습에 사용하지 않는다.
- 설명, 근거 문장, markdown, 코드블록은 출력하지 않는다.

[출력 형식]
{"inventive_step":"만족|부족"}
```

## 8. H200 학습 설정

원래 H200 계획의 기본 학습 설정은 다음과 같다.

```yaml
model_name: Qwen/Qwen2.5-32B-Instruct
training_type: bf16 일반 LoRA
load_in_4bit: false
max_seq_length: 4096

batch_size: 8
gradient_accumulation_steps: 2
effective_batch_size: 16
epochs: 2
learning_rate: 1e-4
optimizer: adamw_torch
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01

lora_r: 32
lora_alpha: 64
lora_dropout: 0
bias: none
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

bf16: true
fp16: false
gradient_checkpointing: unsloth
save_total_limit: 2
early_stopping_patience: 4
eval_steps: 500
save_steps: 500
logging_steps: 10
load_best_model_at_end: true
best_metric: eval_loss
greater_is_better: false
```

H200에서는 QLoRA가 아니라 BF16 일반 LoRA로 학습한다. 32B base model은 BF16으로 로드하고, 베이스 모델 가중치는 freeze한 상태에서 LoRA adapter 파라미터만 업데이트한다.

각 Adapter의 train rows는 18,000건이고 effective batch size는 16이므로, 2 epochs 기준 optimizer step은 다음과 같다.

```text
18,000 / 16 * 2 = 약 2,250 optimizer steps
```

## 9. 신규성 Adapter 학습 명령

```bash
python scripts/train_adapter_h200_lora.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --train_file data/novelty/novelty_train.jsonl \
  --val_file data/novelty/novelty_val.jsonl \
  --output_dir models/qwen32b_novelty_h200_bf16_lora \
  --max_seq_length 4096 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --eval_steps 500 \
  --save_steps 500 \
  --gradient_checkpointing unsloth \
  --optim adamw_torch
```

## 10. 진보성 Adapter 학습 명령

```bash
python scripts/train_adapter_h200_lora.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --train_file data/inventive/inventive_train.jsonl \
  --val_file data/inventive/inventive_val.jsonl \
  --output_dir models/qwen32b_inventive_h200_bf16_lora \
  --max_seq_length 4096 \
  --batch_size 8 \
  --grad_accum 2 \
  --epochs 2 \
  --learning_rate 1e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --eval_steps 500 \
  --save_steps 500 \
  --gradient_checkpointing unsloth \
  --optim adamw_torch
```

## 11. 평가 방법

각 Adapter는 자기 역할만 평가한다.

신규성 Adapter 평가는 다음 기준을 사용한다.

```text
예측 novelty == 정답 novelty -> correct
예측 novelty != 정답 novelty -> incorrect
```

진보성 Adapter 평가는 다음 기준을 사용한다.

```text
예측 inventive_step == 정답 inventive_step -> correct
예측 inventive_step != 정답 inventive_step -> incorrect
```

JSON 파싱 실패, 허용 라벨 외 출력, 빈 출력은 오답으로 처리한다.

평가 명령은 다음과 같다.

```bash
python scripts/evaluate_adapter_h200.py \
  --base_model Qwen/Qwen2.5-32B-Instruct \
  --adapter models/qwen32b_novelty_h200_bf16_lora \
  --test_file data/novelty/novelty_test.jsonl \
  --task novelty
```

```bash
python scripts/evaluate_adapter_h200.py \
  --base_model Qwen/Qwen2.5-32B-Instruct \
  --adapter models/qwen32b_inventive_h200_bf16_lora \
  --test_file data/inventive/inventive_test.jsonl \
  --task inventive
```

## 12. 최종 추론 결합 방식

최종 등록/거절 판단은 두 Adapter 결과를 순차적으로 결합한다.

```text
1. 심사대상 문헌에 선행문헌 D1,D2를 붙인다.
2. 신규성 Adapter를 호출한다.
3. novelty="부족"이면 신규성 부족 거절로 판단한다.
4. novelty="만족"이면 진보성 Adapter를 호출한다.
5. inventive_step="부족"이면 진보성 부족 거절로 판단한다.
6. novelty="만족"이고 inventive_step="만족"이면 등록 가능으로 판단한다.
```

최종 해석은 다음과 같다.

```text
신규성 부족 -> 거절
신규성 만족 + 진보성 부족 -> 거절
신규성 만족 + 진보성 만족 -> 등록 가능
```

## 13. 최종 정리

이 문서의 기준은 다음과 같다.

```text
원본 데이터: 30,000건
  등록 10,000건
  진보성 부족 거절 10,000건
  신규성 부족 거절 10,000건

신규성 Adapter 데이터:
  만족 10,000건
  부족 10,000건
  train 18,000 / val 1,000 / test 1,000

진보성 Adapter 데이터:
  만족 10,000건
  부족 10,000건
  train 18,000 / val 1,000 / test 1,000

H200 학습:
  Qwen/Qwen2.5-32B-Instruct
  BF16 일반 LoRA
  max_seq_length 4096
  batch 8
  grad_accum 2
  epochs 2
  optimizer adamw_torch
  eval/save 500 steps
```
