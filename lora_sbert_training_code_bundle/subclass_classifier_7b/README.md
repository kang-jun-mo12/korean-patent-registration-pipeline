# Patent30000 진보성 세부분류 H200 Grounded-only LoRA 패키지

이 폴더는 3만 특허 데이터셋 중 `거절-진보성 부족` 샘플에서,
기존 `subtypes_cleaned.jsonl`과 app_no가 매칭되어 근거 라벨을 확인할 수 있는 데이터만 사용해
4-class multi-label 세부분류 어댑터를 학습하기 위한 H200용 패키지다.

현재 이 폴더의 기본 방식은 **1번 방법: 근거 라벨만 사용**이다.
pseudo-label 데이터는 기본 학습에 사용하지 않는다.

## 라벨

```text
결합류
차이류
설계변경
공지기술부가
```

원본 subtype 매핑:

```text
결합류       = 결합용이성 + 단순결합
차이류       = 차이없음 + 시너지부정
설계변경     = 설계변경
공지기술부가 = 공지기술부가 + 단순부가
```

## 데이터

```text
전체 grounded 라벨: 3,318
train: 2,654
val:     331
test:    333
```

이 데이터는 내가 임의로 추측한 라벨이 아니라, 기존 `subtypes_cleaned.jsonl`의 subtype을 4-class로 매핑한 근거 라벨이다.

## 실행

```bash
cd patent30000_subclass_h200_pseudo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_train_subclass_h200.sh
bash scripts/run_eval_subclass_h200.sh
```

## 기본 학습 설정

```yaml
base_model: Qwen/Qwen2.5-7B-Instruct
task_type: sequence classification
problem_type: multi_label_classification
max_seq_length: 4096
precision: bf16
LoRA r: 16
LoRA alpha: 32
LoRA dropout: 0.05
batch_size: 8
grad_accum: 2
epochs: 5
learning_rate: 2e-5
optimizer: adamw_torch_fused
scheduler: cosine
eval_steps: 50
save_steps: 50
```

## 예상 시간

H200 기준 대략:

```text
본학습: 5~15분
평가/threshold calibration 포함: 10~25분
```

## 주의

데이터가 3,318건으로 크지 않으므로, test set의 macro F1, Hit@1, Hit@2와 실제 샘플 검토를 같이 봐야 한다.
pseudo-label을 쓰지 않는 대신 라벨 노이즈는 낮아지는 방향이다.
