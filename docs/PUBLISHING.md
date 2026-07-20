# GitHub 게시 범위와 절차

이 공개 GitHub 저장소는 **코드와 문서만** 게시합니다. 모델 가중치와 특허 문헌 데이터는 포함하지 않습니다.

## 공개되는 항목

- 명령행 추론기와 로컬 웹 UI 소스
- 평가 및 등록 판정 코드
- 재구성 학습 코드
- 별도 코드 번들에서 추출한 학습·평가 스크립트
- 파이프라인 보고서, 모델 카드, 학습 설명 문서
- 패키지 의존성 및 가벼운 문법 검사 CI

## 공개되지 않는 항목

- 신규성·진보성 Qwen2.5-32B LoRA 어댑터
- 7B 거절 세부 유형 LoRA 어댑터
- SBERT 파생 모델
- 검색 corpus와 임베딩 인덱스
- 문서 원문과 샘플별 예측이 포함된 평가 JSONL
- 위 파일을 중복 포함하는 대용량 ZIP

루트 `.gitignore`에 해당 경로가 명시되어 있습니다. 로컬 파일은 삭제되지 않으며 Git 기록에만 포함되지 않습니다.

## 게시 전 확인

```powershell
git status
git ls-files | Select-String 'safetensors|corpus_embeddings|corpus_meta|final_eval_sample_lookup'
python -m compileall -q .\END .\lora_sbert_training_code_bundle
```

두 번째 명령은 아무 결과도 출력하지 않아야 합니다.

## 최초 게시

```powershell
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

## 모델 배포

모델은 재배포 권한을 확인한 뒤 Hugging Face의 비공개 저장소에 먼저 올리는 방식을 권장합니다. 구성요소별 저장소 분리 예:

- `korean-patent-novelty-qwen2.5-32b-lora`
- `korean-patent-inventive-qwen2.5-32b-lora`
- `korean-patent-subclass-qwen2.5-7b-lora`
- `korean-patent-sbert-retriever`

특허 원문 corpus와 검색 인덱스는 모델 저장소와 분리하고, 데이터 재배포 권한을 확인하기 전에는 공개하지 않습니다.

## 라이선스

공개 저장소에 프로젝트 라이선스가 아직 지정되지 않았습니다. 라이선스를 추가하기 전에는 저장소를 볼 수 있더라도 명시적인 재사용 권한은 부여되지 않습니다.
