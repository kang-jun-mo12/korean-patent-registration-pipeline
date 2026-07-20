# Patent30000 SBERT H200 Training Package

이 폴더는 3만 특허 데이터셋의 `심사대상 1건 + 선행기술 2건` 구조를 이용해,
새 특허 입력에 대해 선행기술 corpus에서 가장 유사한 선행사례 top-2를 찾는 SBERT 검색 모델을 H200에서 학습하기 위한 패키지다.

## 데이터 요약

```text
원본 target row: 30,000건
positive pair: 60,000개
train pair: 54,000개
eval pair: 6,000개
corpus_all: 60,000개
corpus_unique: 29,794개
```

각 target마다 선행기술 2개가 붙어 있고, 이 target-prior 관계를 positive pair로 사용한다.

## 추천 실행 순서

```bash
cd patent30000_sbert_h200
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

bash scripts/run_train_sbert_h200.sh
bash scripts/run_build_index.sh
bash scripts/run_eval_recall.sh
```

## 학습 목표

```text
입력: 새 특허의 발명의 명칭 + 초록 + 청구항 1
출력: 선행기술 corpus 중 의미적으로 가장 가까운 top-2 문헌
```

## 기본 H200 학습 설정

```yaml
base_model: jhgan/ko-sroberta-multitask
max_seq_length: 512
loss: MultipleNegativesRankingLoss
batch_size: 128
epochs: 3
learning_rate: 2e-5
warmup_ratio: 0.1
```

H200에서는 이 학습이 Qwen 32B LoRA보다 훨씬 가볍다. 보통 SBERT 본학습은 5~20분 안쪽, 인덱스 생성과 recall 평가까지 포함하면 20~60분 정도를 예상한다.

## 산출물

```text
models/sbert_patent_h200/
  fine-tuned SBERT model

index/sbert_patent_h200/
  corpus_meta.jsonl
  corpus_embeddings.npy
```

## 주의

`corpus_all_60000.jsonl`은 target별 선행기술 2개를 모두 펼친 파일이다.
같은 선행문헌이 여러 target에 반복될 수 있으므로, 실제 검색 기본값은 `corpus_unique.jsonl`을 사용한다.
