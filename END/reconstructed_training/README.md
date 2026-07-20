# Reconstructed Training Code

이 폴더는 현재 남아 있는 모델 산출물, adapter config, `FINAL_PATENT_PIPELINE_REPORT.md`,
`SBERT성능추가설명.txt`를 근거로 재구성한 학습 코드입니다.

중요: 원본 학습 코드를 복원한 것이 아니라, 문서와 모델 설정으로 추정한 재학습용 코드입니다.
현재 폴더에는 원본 train/validation/test 데이터가 없으므로, 아래 형식의 데이터 파일을 준비해야
실제로 재학습할 수 있습니다.

## 현재 모델별 추정값

| 모델 | 베이스 | 방식 | 주요 설정 |
|---|---|---|---|
| SBERT 검색 | `jhgan/ko-sroberta-multitask` | Stage1 MNRL, Stage2 hard negative MNRL | stage1: batch 256, epoch 3, lr 2e-5 / stage2: batch 256, mini batch 32, epoch 2, lr 1e-5 |
| 신규성 LoRA | `unsloth/Qwen2.5-32B-Instruct` | response-only SFT | LoRA r 32, alpha 64, dropout 0, bf16, max len 4096, batch 8, grad accum 2, lr 1e-4 |
| 진보성 LoRA | `unsloth/Qwen2.5-32B-Instruct` | response-only SFT | 신규성과 동일, 최종 checkpoint는 2000 step 부근 |
| 서브라벨 LoRA | `Qwen/Qwen2.5-7B-Instruct` | sequence classification multi-label | LoRA r 16, alpha 32, dropout 0.05, bf16, max len 4096, batch 8, grad accum 2, lr 2e-5 |

## 필요한 데이터

현재 실제 학습 데이터 폴더는 없습니다. 리포트상 원래 경로는 대략 아래였던 것으로 보입니다.

- 신규성: `patent30000_h200_bf16_lora/data/novelty/{novelty_train,novelty_val,novelty_test}.jsonl`
- 진보성: `patent30000_h200_bf16_lora/data/inventive/{inventive_train,inventive_val,inventive_test}.jsonl`
- SBERT hard negative: `data/train_pairs_hardneg_stage2.jsonl`
- 서브라벨: `patent30000_subclass_h200_pseudo/data/subclass_grounded_only.jsonl`

## SBERT 학습

Stage 1 positive pair 학습:

```powershell
python .\reconstructed_training\train_sbert.py `
  --stage stage1 `
  --train-file .\data\sbert_positive_train.jsonl `
  --base-model jhgan/ko-sroberta-multitask `
  --output-dir .\models\sbert_patent_h200
```

Hard negative mining:

```powershell
python .\reconstructed_training\mine_sbert_hardneg.py `
  --model-dir .\models\sbert_patent_h200 `
  --train-file .\data\sbert_positive_train.jsonl `
  --corpus-file .\patent30000_sbert_h200\index\sbert_patent_h200_hardneg_stage2\corpus_meta.jsonl `
  --output-file .\data\train_pairs_hardneg_stage2.jsonl
```

Stage 2 hard negative 학습:

```powershell
python .\reconstructed_training\train_sbert.py `
  --stage stage2 `
  --train-file .\data\train_pairs_hardneg_stage2.jsonl `
  --base-model .\models\sbert_patent_h200 `
  --output-dir .\models\sbert_patent_h200_hardneg_stage2
```

검색 인덱스 생성:

```powershell
python .\reconstructed_training\build_sbert_index.py `
  --model-dir .\models\sbert_patent_h200_hardneg_stage2 `
  --corpus-file .\patent30000_sbert_h200\index\sbert_patent_h200_hardneg_stage2\corpus_meta.jsonl `
  --output-dir .\index\sbert_patent_h200_hardneg_stage2
```

SBERT JSONL은 다음 키 중 하나를 쓰면 됩니다.

```json
{"anchor": "심사대상 문헌", "positive": "정답 선행문헌", "positive_id": "combined:123"}
```

Stage 2 hard negative 파일은 아래 형식도 지원합니다.

```json
{"anchor": "...", "positive": "...", "hard_negatives": ["...", "...", "..."]}
```

## 신규성/진보성 LoRA 학습

학습 파일은 `messages` 형식이 가장 안전합니다.

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"{\"novelty\":\"만족\"}"}]}
```

신규성:

```powershell
python .\reconstructed_training\train_qwen_lora_sft.py `
  --task novelty `
  --train-file .\data\novelty_train.jsonl `
  --validation-file .\data\novelty_val.jsonl `
  --output-dir .\checkpoint-novelty-retrain
```

진보성:

```powershell
python .\reconstructed_training\train_qwen_lora_sft.py `
  --task inventive `
  --train-file .\data\inventive_train.jsonl `
  --validation-file .\data\inventive_val.jsonl `
  --output-dir .\checkpoint-inventive-retrain
```

## 서브라벨 LoRA 학습

멀티라벨 데이터 예:

```json
{"text":"판단 대상 문헌과 선행문헌...", "labels":["결합류","차이류"]}
```

또는 multi-hot도 가능합니다.

```json
{"text":"...", "labels":[1,1,0,0]}
```

학습:

```powershell
python .\reconstructed_training\train_subclass_lora_seqcls.py `
  --train-file .\data\subclass_train.jsonl `
  --validation-file .\data\subclass_val.jsonl `
  --output-dir .\subclass_adapter_retrain
```

## 설치 패키지

```powershell
pip install -r .\reconstructed_training\requirements.txt
```

32B LoRA 학습은 큰 GPU 메모리가 필요합니다. 원 리포트는 4bit가 아닌 bf16 일반 LoRA라고 되어 있으므로,
단일 소비자 GPU에서는 그대로 재현하기 어렵습니다.
