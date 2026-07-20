# H200 BF16 일반 LoRA 학습 패키지

이 폴더는 `patent_30000_with_2_prior_art_claims_fixed` 데이터로 신규성 어댑터와 진보성 어댑터를 H200에서 빠르게 학습하기 위한 패키지다.

H200은 NVIDIA 공식 기준 141GB HBM3e 메모리와 4.8TB/s 메모리 대역폭을 제공한다. 따라서 RTX PRO 6000용 QLoRA 설정보다 `Qwen/Qwen2.5-32B-Instruct`를 bf16 일반 LoRA로 학습하는 구성이 더 적합하다.

공식 사양 출처: https://www.nvidia.com/en-us/data-center/h200/

## 1. 폴더 구조

```text
raw/
  patent_30000_with_2_prior_art_claims_fixed.zip
data/
  novelty/
    novelty_train.jsonl
    novelty_val.jsonl
    novelty_test.jsonl
  inventive/
    inventive_train.jsonl
    inventive_val.jsonl
    inventive_test.jsonl
scripts/
  train_adapter_h200_lora.py
  evaluate_adapter_h200.py
  run_train_novelty_h200.sh
  run_train_inventive_h200.sh
  run_smoke_novelty_h200.sh
DATASET_SUMMARY.json
H200_BF16_LORA_TRAINING_GUIDE.md
```

## 2. 데이터 구성

원본:

```text
등록: 10,000건
거절-진보성 부족: 10,000건
거절-신규성 부족: 10,000건
각 샘플마다 선행문헌 D1,D2 2개
```

신규성 어댑터:

```text
train: 18,000
val: 1,000
test: 1,000
라벨: 만족 9,000 / 부족 9,000 in train
```

진보성 어댑터:

```text
train: 18,000
val: 1,000
test: 1,000
라벨: 만족 9,000 / 부족 9,000 in train
신규성 부족 샘플은 진보성 학습에서 제외
```

요약 원본:

```json
{
  "source_rows": 30000,
  "source_class_counts": {
    "등록": 10000,
    "거절-진보성 부족": 10000,
    "거절-신규성 부족": 10000
  },
  "novelty": {
    "policy": "balanced: 부족 10000 + 만족 10000(등록 5000, 진보성부족 5000)",
    "train_rows": 18000,
    "val_rows": 1000,
    "test_rows": 1000,
    "train_label_counts": {
      "부족": 9000,
      "만족": 9000
    },
    "val_label_counts": {
      "부족": 500,
      "만족": 500
    },
    "test_label_counts": {
      "부족": 500,
      "만족": 500
    }
  },
  "inventive": {
    "policy": "balanced: 부족 10000(진보성부족) + 만족 10000(등록), 신규성부족은 판정제외로 미사용",
    "train_rows": 18000,
    "val_rows": 1000,
    "test_rows": 1000,
    "train_label_counts": {
      "부족": 9000,
      "만족": 9000
    },
    "val_label_counts": {
      "부족": 500,
      "만족": 500
    },
    "test_label_counts": {
      "만족": 500,
      "부족": 500
    }
  },
  "seed": 42
}
```

## 3. H200 기본 학습 설정

1순위 추천 설정:

```yaml
model_name: Qwen/Qwen2.5-32B-Instruct
training_type: bf16 일반 LoRA
load_in_4bit: false
max_seq_length: 4096
batch_size: 8
gradient_accumulation_steps: 2
effective_batch_size: 16
lora_r: 32
lora_alpha: 64
lora_dropout: 0
epochs: 2
learning_rate: 1e-4
optimizer: adamw_torch
bf16: true
gradient_checkpointing: unsloth
eval_steps: 500
save_steps: 500
save_total_limit: 2
early_stopping_patience: 4
```

이 설정은 QLoRA가 아니다. H200의 VRAM을 활용해서 32B base를 bf16으로 올리고 LoRA만 학습한다.

## 4. 왜 QLoRA가 아니라 일반 LoRA인가

RTX 6000이나 16GB 로컬 GPU에서는 VRAM 때문에 `bnb-4bit` QLoRA가 필요했다.

H200에서는 32B 모델을 bf16으로 올릴 수 있으므로 다음 장점이 있다.

```text
4bit 양자화/역양자화 오버헤드 감소
bf16 Tensor Core 연산을 더 직접적으로 사용
학습 손실과 출력 안정성 측면에서 약간 유리할 가능성
batch_size를 키워 throughput 향상 가능
```

## 5. 환경 설치

vast.ai 또는 클라우드 H200 이미지에서 실행 예시:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install unsloth transformers datasets trl peft accelerate bitsandbytes
```

PyTorch/CUDA가 이미 설치된 이미지라면 PyTorch는 다시 설치하지 않는 편이 안전하다.

## 6. 스모크 테스트

먼저 0.01 epoch만 돌려서 모델 로드, 데이터 로드, VRAM, 저장 경로를 확인한다.

```bash
bash scripts/run_smoke_novelty_h200.sh
```

정상 확인 항목:

```text
Qwen/Qwen2.5-32B-Instruct 로드 성공
load_in_4bit=False 출력 확인
loss가 출력됨
OOM 없음
models/smoke_novelty_h200_bf16_lora 생성
```

## 7. 신규성 어댑터 학습

```bash
bash scripts/run_train_novelty_h200.sh
```

직접 실행 명령:

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

예상 optimizer step:

```text
18,000 rows / batch 8 / grad_accum 2 * 2 epochs
= 약 2,250 optimizer steps
```

## 8. 진보성 어댑터 학습

```bash
bash scripts/run_train_inventive_h200.sh
```

직접 실행 명령:

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

## 9. 더 빠르게 해보고 싶을 때

스모크 테스트에서 VRAM이 충분히 남으면 checkpointing off를 시험할 수 있다.

```bash
python scripts/train_adapter_h200_lora.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --train_file data/inventive/inventive_train.jsonl \
  --val_file data/inventive/inventive_val.jsonl \
  --output_dir models/qwen32b_inventive_h200_bf16_lora_no_gc_trial \
  --max_seq_length 4096 \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 0.05 \
  --learning_rate 1e-4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --eval_steps 100 \
  --save_steps 100 \
  --gradient_checkpointing false \
  --optim adamw_torch
```

no checkpointing에서 OOM이 없고 step time이 더 빠르면 본훈련에 적용한다. OOM이 나면 기본 설정인 `--gradient_checkpointing unsloth`로 돌아가면 된다.

## 10. 평가

신규성:

```bash
python scripts/evaluate_adapter_h200.py \
  --base_model Qwen/Qwen2.5-32B-Instruct \
  --adapter models/qwen32b_novelty_h200_bf16_lora \
  --test_file data/novelty/novelty_test.jsonl \
  --task novelty
```

진보성:

```bash
python scripts/evaluate_adapter_h200.py \
  --base_model Qwen/Qwen2.5-32B-Instruct \
  --adapter models/qwen32b_inventive_h200_bf16_lora \
  --test_file data/inventive/inventive_test.jsonl \
  --task inventive
```

## 11. 예상 시간

정확한 시간은 이미지, 드라이버, PyTorch/FlashAttention, 디스크 속도에 따라 달라진다.

현실적인 첫 예상:

```text
신규성 어댑터: 약 1~3시간
진보성 어댑터: 약 1~3시간
둘 다 합계: 약 2~6시간
```

스모크 테스트의 실제 `s/it`를 보면 더 정확히 계산할 수 있다.

## 12. 주의점

```text
Qwen/Qwen2.5-32B-Instruct는 bnb-4bit 모델이 아니다.
Hugging Face에서 base model을 다운로드해야 하므로 네트워크와 디스크 여유가 필요하다.
디스크 여유는 최소 250GB 이상 권장한다.
처음에는 기본 설정으로 성공 확인 후 batch/checkpointing을 조정한다.
```
