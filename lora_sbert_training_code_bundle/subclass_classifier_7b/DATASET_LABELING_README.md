# Patent30000 진보성 세부분류 4-class 라벨 데이터

이 폴더는 3만 특허 데이터셋 중 `거절-진보성 부족` 샘플에 대해 기존 `subtypes_cleaned.jsonl`을 매칭하고, 친구 패키지의 4-class multi-label 체계로 변환한 데이터다.

## 라벨 체계

```text
결합류       = 결합용이성 + 단순결합
차이류       = 차이없음 + 시너지부정
설계변경     = 설계변경
공지기술부가 = 공지기술부가 + 단순부가
```

데이터가 너무 적은 `수치한정`, `균등치환`, `기능표현`은 학습 라벨에서 제외했다.

## 수량

```text
진보성 부족 샘플 전체: 10,000
subtypes_cleaned 매칭: 4,215
4-class 학습 가능: 3,318
학습 제외/미매칭: 6,682
```

split:

```text
train: 2,654
val:   331
test:  333
```

## 파일

```text
subclass_all_usable.jsonl
subclass_train.jsonl
subclass_val.jsonl
subclass_test.jsonl
subclass_unusable_or_missing.jsonl
subclass_label_summary.json
```

각 row에는 `text`, `labels`, `multihot`, `source_subtypes`가 들어 있다.
`text`는 친구 패키지의 세부분류 입력 형식과 맞췄다.

```text
[claim_1]
...

[abstract]
...

[refs_top_K]
[1] ...
[2] ...
```
