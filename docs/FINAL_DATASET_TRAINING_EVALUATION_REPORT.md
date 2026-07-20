# 특허나침반 데이터 수집·학습·평가 최종 설명서

이 문서는 특허나침반 프로젝트의 KIPRIS Plus 데이터 수집, 데이터셋 정제와 라벨링, 모델별 학습, 최종 end-to-end 평가를 하나의 흐름으로 정리한 통합 설명서다.

## 1. 문서의 근거와 해석 원칙

이 설명서는 다음 자료를 교차 확인해 작성했다.

1. 프로젝트 내부 데이터 수집 프로세스 기록: KIPRIS Plus API, 검색 범위, 필터, 라벨 도출 방법
2. [`END/FINAL_PATENT_PIPELINE_REPORT.md`](../END/FINAL_PATENT_PIPELINE_REPORT.md): 최종 모델 구성, H200 학습 데이터, 평가 방법과 성능
3. [`END/SBERT성능추가설명.txt`](../END/SBERT성능추가설명.txt): SBERT 2-stage 학습 및 strict/soft 검색 평가
4. [`lora_sbert_training_code_bundle`](../lora_sbert_training_code_bundle): 모델별 학습 설정, 데이터 요약과 스크립트
5. 로컬 adapter config와 최종 산출물: 기반 모델, LoRA rank, alpha, dropout 등 최종 설정 확인

내부 수집 문서에서는 **수집 방법만** 가져왔다. 최종 수량은 실제 프로젝트 기준인 `균형 어댑터 데이터 30,000건 + 검색용 약 30,000건 = 전체 약 60,000건`으로 통일했다.

## 2. 프로젝트 목표

이 프로젝트는 단순한 등록/거절 이진 분류기가 아니다. 사용자가 발명의 명칭, 초록, 청구항을 입력하면 다음 순서로 처리한다.

```text
사용자 특허 문헌
  -> SBERT 선행기술 검색
  -> 신규성 Qwen2.5-32B LoRA
  -> 진보성 Qwen2.5-32B LoRA
  -> AND 규칙으로 등록/거절 예비 판정
  -> 진보성 부족이면 Qwen2.5-7B 세부 유형 분류
  -> 32B 기반 모델로 사용자용 설명 생성
```

최종 판정 규칙은 다음과 같다.

```text
등록 = 신규성 만족 AND 진보성 만족
거절 = 그 외 모든 경우
```

## 3. 법적 판단 범위

학습 대상은 특허법 제29조의 신규성·진보성 판단으로 제한했다.

| 판단 대상 | 법적 기준 | 모델 태스크 |
|---|---|---|
| 신규성 부족 | 특허법 제29조 제1항 | `novelty` |
| 진보성 부족 | 특허법 제29조 제2항 | `inventive` |

다음 거절 사유만 있는 문서는 모델 범위에서 제외했다.

- 명세서 기재불비
- 청구범위 기재불비 또는 불명확
- 선출원 또는 확대된 선원
- 불특허 대상
- 절차상 거절

제29조 사유와 제42조 기재불비가 함께 있는 경우에는 제29조 신규성·진보성 사유를 우선했다.

## 4. KIPRIS Plus 기반 수집

### 4.1 인증정보 관리

KIPRIS Plus REST access key는 코드나 문서에 직접 넣지 않고 `.env`의 `KIPRIS_ACCESS_KEY`로 관리했다.

```text
KIPRIS_ACCESS_KEY=발급받은_REST_ACCESS_KEY
```

저장소의 `.env`는 Git 추적 대상에서 제외되어 있다.

### 4.2 사용 API

| 목적 | 서비스/엔드포인트 |
|---|---|
| 자유어 후보 검색 | `freeSearchInfo` |
| CPC 검색 | `cpcSearchInfo` |
| IPC 검색 | `ipcSearchInfo` |
| 청구항 조회 | `patentClaimInfo` |
| 선행기술 조사문헌 | `patentPriorArtDocumentsInfo` |
| 심사결과 | `examineResultInfo` |
| 거절결정서 | `rejectDecisionInfo` |
| 추가 거절정보 | `additionRejectInfo` |
| 거절사유 검색 | `rejectionContent` |

모든 결과는 출원번호를 중심 키로 연결했다.

```text
검색 결과
  -> 출원번호 정규화와 중복 제거
  -> 청구항 조회
  -> 등록/거절 상태 확인
  -> 거절문서 조회와 법조항 라벨링
  -> 선행기술 조사문헌 조회
  -> 본원·선행문헌 텍스트 필터
  -> 모델별 데이터셋 생성
```

### 4.3 기술 분야

초기 수집은 IT·소프트웨어·데이터 처리·정보통신을 중심으로 진행했다.

주요 CPC/IPC 계열:

```text
G06, G06F, G06N, G06Q, G06T, G06V,
G10L, H04L, H04W, H04N, G11C, H01L, G16H
```

CPC/IPC 검색만으로 충분한 거절 사례를 빠르게 확보하기 어려워 실제 대량 수집은 `freeSearchInfo` 키워드 검색을 중심으로 수행했다. 사용 키워드는 인공지능, 데이터 처리, 네트워크, 통신, 보안, 영상·음성·자연어 처리, 반도체, 메모리, 센서, 제어, 로봇 등 IT 핵심어와 장치·방법·시스템 같은 일반 기술어를 포함했다.

### 4.4 후보 저장과 중복 제거

검색 단계에서는 다음 메타데이터를 저장했다.

```json
{
  "application_number": "출원번호",
  "invention_title": "발명의 명칭",
  "abstract": "초록",
  "register_status": "등록/거절 상태",
  "application_date": "출원일",
  "open_number": "공개번호",
  "register_number": "등록번호",
  "source_keyword": "검색 키워드"
}
```

동일 특허가 여러 키워드에서 반복 검색될 수 있으므로 출원번호 기준으로 중복 제거했다. 등록 데이터와 거절 데이터 사이에도 같은 출원번호가 겹치지 않도록 검사했다.

## 5. 청구항과 문헌 길이 필터

검색 결과만으로는 청구항이 충분하지 않아 각 출원번호에 `patentClaimInfo`를 호출했다.

기본 입력 후보는 다음과 같이 구성했다.

```text
claim_input = abstract + "\n" + claims
```

기준 특허와 선행문헌 모두 `초록 + 청구항`이 1,500자 이하인 문서만 사용했다.

```text
len(abstract + claims) <= 1500
```

이 제한은 심사대상 문헌, 선행문헌 2건, 지시문과 출력 형식을 Qwen 입력 길이 안에 함께 넣기 위한 운영상 필터다. 수집 문서는 최대 4,092 tokens를 기준으로 설명하고, 최종 학습 설정은 최대 4,096 tokens로 기록되어 있다.

## 6. 등록·거절 후보 선정

### 6.1 등록 데이터

등록 샘플은 다음 조건을 모두 통과해야 했다.

1. 특허이며 실용신안이 아닐 것
2. 등록 상태가 명확할 것
3. 초록과 청구항이 존재할 것
4. 본원의 `초록 + 청구항`이 1,500자 이하일 것
5. KIPRIS 선행기술 조사문헌이 2건 이상 연결될 것
6. 연결된 선행문헌도 동일한 길이 조건을 통과할 것
7. 출원번호가 중복되지 않을 것

등록 문헌은 신규성·진보성 모두 `pass`로 사용했다.

### 6.2 거절 데이터

거절 샘플도 동일한 기술 분야와 텍스트·선행문헌 조건을 적용했다. 거절 상태만으로 신규성·진보성 라벨을 확정하지 않고, 중간서류 API에서 실제 거절결정서와 심사결과 원문을 추가 수집했다.

추출 정보:

```json
{
  "application_number": "출원번호",
  "document_type": "rejectDecisionInfo",
  "official_text": "거절결정서 원문",
  "detected_statutes": ["특허법 제29조 제2항"],
  "detected_keywords": ["진보성", "통상의 기술자", "용이하게 발명"]
}
```

## 7. 거절 사유 라벨링

거절문서에서 법조항과 판단 문구를 함께 탐지했다.

### 7.1 신규성 부족

주요 기준:

- 특허법 제29조 제1항
- 신규성, 신규하지 아니하다
- 공지, 공연히 실시
- 반포된 간행물
- 인용문헌에 청구항 구성이 모두 개시

### 7.2 진보성 부족

주요 기준:

- 특허법 제29조 제2항
- 진보성
- 통상의 기술자
- 쉽게 또는 용이하게 발명·도출
- 인용발명들의 결합

### 7.3 혼합 거절

제29조 제1항과 제2항이 함께 감지되면 `mixed_novelty_inventive`로 분류했다. 혼합 사례는 신규성 어댑터의 `fail`에는 포함하지만 진보성 어댑터에서는 제외했다.

## 8. 선행기술 조사문헌 연결

각 기준 특허에 `patentPriorArtDocumentsInfo`를 호출하고 다음 순서로 처리했다.

1. 선행기술 조사문헌 번호 수집
2. 문헌번호 정규화
3. KIPRIS 검색 API로 선행문헌 재조회
4. 선행문헌의 명칭·초록·청구항 수집
5. `초록 + 청구항 <= 1,500자` 필터
6. 조건을 통과한 문헌이 2건 이상인 기준 특허만 유지
7. 최종 D1·D2 연결 저장

신규성은 단일 선행문헌에 모든 구성이 개시되는지를 판단하지만, 진보성은 복수 문헌의 결합 가능성도 보아야 한다. 두 태스크가 동일한 입력 구조를 사용할 수 있도록 최소 두 건의 선행문헌을 요구했다.

## 9. 최종 데이터 규모

### 9.1 균형 어댑터 데이터

신규성·진보성 판단과 SBERT query 구성에 사용하는 기준 특허는 총 30,000건이다.

| 클래스 | 건수 |
|---|---:|
| 등록 | 10,000 |
| 신규성 부족 | 10,000 |
| 진보성 부족 | 10,000 |
| 합계 | **30,000** |

### 9.2 검색용 선행문헌

각 target에 연결된 D1·D2를 모두 펼치면 60,000개의 target-prior positive pair가 만들어진다. 선행문헌 ID를 기준으로 중복 제거한 실제 검색 corpus는 29,794건이며 프로젝트 수량에서는 검색용 약 30,000건으로 집계한다.

```text
균형 어댑터 데이터 30,000건
+ 검색용 선행문헌 29,794건
= 전체 실문서 약 60,000건
```

60,000 positive pair는 별도로 수집한 60,000개 문서가 아니라 `30,000 target × D1·D2`로 파생한 학습 관계다.

### 9.3 모델별 분할

| 태스크 | train | validation | test |
|---|---:|---:|---:|
| 신규성 LoRA | 18,000 | 1,000 | 1,000 |
| 진보성 LoRA | 18,000 | 1,000 | 1,000 |
| SBERT | 54,000 positive pair | eval target 3,000 | corpus 29,794 |
| 세부 유형 LoRA | 2,654 | 332 | 332 |

## 10. 모델 학습

### 10.1 SBERT 검색 모델

기반 모델: `jhgan/ko-sroberta-multitask`

```text
Stage 1:
  54,000 positive pair
  MultipleNegativesRankingLoss
  batch 256, epoch 3, lr 2e-5

Hard-negative mining:
  전체 corpus 검색
  정답이 아닌 rank 20~200 후보에서 query당 3건 선택

Stage 2:
  CachedMultipleNegativesRankingLoss
  batch 256, mini batch 32, epoch 2, lr 1e-5
```

최종 검색 corpus는 60,000개의 D1·D2 행을 중복 제거한 29,794건이다.

### 10.2 신규성·진보성 32B LoRA

기반 모델: `unsloth/Qwen2.5-32B-Instruct`

| 설정 | 값 |
|---|---|
| 학습 | response-only supervised fine-tuning |
| 방식 | BF16 일반 LoRA, 4bit 아님 |
| max sequence length | 4,096 |
| batch | 8 × gradient accumulation 2 = effective 16 |
| epochs | 최대 2 |
| learning rate | `1e-4` |
| scheduler | cosine |
| LoRA | r=32, alpha=64, dropout=0 |
| target | q/k/v/o, gate/up/down projection |

assistant JSON 정답 부분에만 loss를 적용했다. 신규성과 진보성은 서로 다른 adapter로 학습하고 validation loss가 가장 낮은 checkpoint를 선택했다.

- 신규성: step 1,500, best eval loss 0.0336
- 진보성: step 2,000, best eval loss 0.0255

### 10.3 진보성 세부 유형 7B LoRA

기반 모델: `Qwen/Qwen2.5-7B-Instruct`

```text
AutoModelForSequenceClassification
4-label multi-label classification
BCEWithLogitsLoss
BF16 PEFT LoRA
r=16, alpha=32, dropout=0.05
```

진보성 부족 10,000건 중 기존 subtype과 매칭된 4,215건에서 4-class로 사용할 수 있는 grounded-only 3,318건을 선별했다. pseudo-label은 최종 기준 모델에서 제외했다.

## 11. 최종 평가

최종 평가는 정답 선행문헌을 직접 제공하지 않았다. SBERT가 29,794건 corpus에서 top-2를 검색하고, 검색 결과를 32B 어댑터에 전달하는 end-to-end retrieval 조건이다.

평가셋:

```text
등록 500
거절 500
  - 신규성 부족 250
  - 진보성 부족 250
합계 1,000
```

## 12. 최종 성능

| 항목 | 성능 |
|---|---:|
| 최종 등록/거절 정확도 | **87.5%** |
| 타이트 정확도 | **86.3%** |
| 신규성 어댑터 정확도 | **84.0%** |
| 진보성 어댑터 정확도 | **91.0%** |
| SBERT strict recall@1 | **29.37%** |
| SBERT strict recall@2 | **40.67%** |
| SBERT strict recall@5 | **57.97%** |
| SBERT soft Hit@2, random p99.5 | **85.1%** |
| SBERT top-2 exact-or-same-IPC-class | **88.73%** |
| 세부 유형 Hit@1 | **73.5%** |
| 세부 유형 Hit@2 | **92.7%** |
| 세부 유형 Macro F1 | **62.3%** |

최종 87.5%에는 검색 실패와 신규성·진보성 판단 오류가 모두 반영되어 있다. soft Hit와 IPC class 지표는 의미적으로 가까운 대체 후보를 설명하는 보조 지표이며, 법적인 선행기술 적합성을 보증하지 않는다.

## 13. 데이터 검증 항목

수집 데이터는 다음을 검사하도록 설계했다.

1. 출원번호 중복 여부
2. 초록·청구항 누락 여부
3. 본원과 선행문헌의 1,500자 제한
4. 등록/거절 상태 일치
5. 신규성 only, 진보성 only, 혼합 라벨 분리
6. 범위 밖 거절 사유 제외
7. 신규성 혼합 사례의 fail 포함
8. 진보성 데이터에서 신규성 only·혼합 제외
9. API key가 결과 파일과 문서에 없는지 확인

## 14. 공개 범위와 한계

- 공개 GitHub 저장소에는 코드와 문서만 포함한다.
- KIPRIS 원문, 검색 corpus, 평가 샘플, 모델 가중치는 공개하지 않는다.
- KIPRIS 데이터 이용·재배포 조건과 기반 모델 라이선스를 확인한 뒤 모델 배포 범위를 결정해야 한다.
- 1,500자 필터는 긴 청구항을 제외해 실제 특허 분포에 선택 편향을 만들 수 있다.
- 균형 평가셋 1,000건의 성능은 실제 출원 분포에서 동일하게 재현된다는 보장이 없다.
- 모든 결과는 연구용 심사 보조 신호이며 법률 자문이나 등록 가능성의 보증이 아니다.
