# H200 SBERT 선행기술 검색 모델 훈련 가이드

## 1. 목적

3만 특허 데이터셋에는 각 심사대상 문헌마다 선행기술 D1,D2가 붙어 있다.
이 관계를 이용해 SBERT를 fine-tune하면, 사용자가 새 특허를 입력했을 때 선행기술 corpus에서 기존 D1,D2와 같은 역할을 할 수 있는 유사 선행사례 top-2를 검색할 수 있다.

## 2. 학습 데이터

```text
target: 30,000건
target-prior positive pair: 60,000개
corpus_all: 60,000개
corpus_unique: 중복 제거 선행문헌
```

SBERT 입력 텍스트는 target과 prior 모두 같은 방식으로 만든다.

```text
[발명의 명칭]
...
[초록]
...
[청구항]
claim 1
```

모든 청구항을 넣지 않고 claim 1 중심으로 학습하는 이유는 SBERT가 검색용 모델이기 때문이다.
신규성/진보성 최종 판단은 Qwen Adapter가 하고, SBERT는 유사 선행기술 후보를 잘 가져오는 데 집중한다.

## 3. 학습 방식

```yaml
model: jhgan/ko-sroberta-multitask
loss: MultipleNegativesRankingLoss
max_seq_length: 512
batch_size: 128
epochs: 3
learning_rate: 2e-5
warmup_ratio: 0.1
```

`MultipleNegativesRankingLoss`는 같은 batch 안의 다른 positive들을 negative로 사용한다.
batch가 클수록 in-batch negative가 많아져 retrieval 학습에 유리하다.

## 4. H200 예상 시간

```text
train pair 약 54,000개
batch 128
epochs 3
optimizer step 약 1,266 steps
```

H200 기준 대략 예상:

```text
SBERT 본학습: 5~20분
corpus embedding 생성: 1~5분
recall@K 평가: 5~20분
전체: 20~60분
```

## 5. 평가 지표

핵심 지표는 recall@2다.

```text
query별 정답 선행기술 2개 중 하나라도 top-K 안에 들어오면 hit
recall@1, recall@2, recall@3, recall@5, recall@10 확인
```

실제 downstream에서는 top-2 또는 top-3를 Qwen Adapter 입력에 붙인다.
