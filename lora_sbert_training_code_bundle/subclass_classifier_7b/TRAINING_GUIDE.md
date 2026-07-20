# H200 세부분류 어댑터 훈련 방법: 1번 Grounded-only

## 1. 목표

진보성 Adapter가 `inventive_step="부족"`으로 판단한 샘플에 대해,
부족 사유를 4개 큰 그룹으로 multi-label 분류한다.

```text
입력:
  system prompt
  user prompt
  심사대상 문헌: 발명의 명칭 + 초록 + 청구항 전체
  선행문헌 D1: 발명의 명칭 + 초록 + 청구항 전체
  선행문헌 D2: 발명의 명칭 + 초록 + 청구항 전체
  작업 지시 + 판단 기준 + 허용 라벨

출력:
  sigmoid 확률 4개
  결합류 / 차이류 / 설계변경 / 공지기술부가
```

## 2. 모델 구조

```text
Qwen/Qwen2.5-7B-Instruct
  + AutoModelForSequenceClassification(num_labels=4)
  + PEFT LoRA
  + BCEWithLogitsLoss
```

생성형 JSON 모델이 아니라 sequence classification head가 붙은 multi-label classifier다.

## 3. 데이터 사용 정책

이 패키지는 1번 방법이므로 pseudo-label을 사용하지 않는다. 입력은 system prompt와 user prompt를 포함하며, 심사대상 문헌과 D1,D2의 발명의 명칭, 초록, 청구항 전체를 넣는다.

```text
train: data/subclass_train.jsonl  2,654
val:   data/subclass_val.jsonl      332
test:  data/subclass_test.jsonl     332
```

row마다 다음 필드가 있다.

```text
text
labels
multihot
source_subtypes
```

## 4. 권장 실험 순서

1. 기본 설정으로 H200에서 학습한다.
2. `run_eval_subclass_h200.sh`로 val threshold를 calibration한다.
3. test macro F1, Hit@1, Hit@2, label별 precision/recall을 확인한다.
4. 라벨별 support가 작으므로 `공지기술부가` precision/recall을 따로 본다.

## 5. 왜 grounded-only를 먼저 쓰는가

pseudo-label 포함 10,000건은 데이터 양은 많지만 라벨 노이즈가 들어간다.
grounded-only 3,318건은 양은 적어도 기존 subtype 근거가 있으므로,
첫 세부분류 어댑터의 기준 성능을 확인하기에 더 안전하다.
