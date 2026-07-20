# 특허나침반 — 한국어 특허 심사 보조 AI

한국어 특허 문헌을 입력받아 선행기술을 검색하고 신규성·진보성·거절 세부 유형을 추론하는 연구용 파이프라인입니다.

파이프라인은 다음 구성요소를 사용합니다.

1. 학습된 SBERT로 약 29,794건의 선행기술에서 유사 문헌 2건 검색
2. Qwen2.5-32B LoRA 어댑터로 신규성 판단
3. Qwen2.5-32B LoRA 어댑터로 진보성 판단
4. 진보성 부족일 때 Qwen2.5-7B 멀티라벨 LoRA로 세부 유형 분류
5. LoRA를 비활성화한 32B 기반 모델로 결과 설명 생성

> 이 프로젝트는 연구 및 심사 보조용입니다. 결과는 법률 자문이나 특허 등록 가능성의 보증이 아닙니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `END/final_patent_inferencer.py` | 명령행 추론기 |
| `END/final_patent_web_server.py` | 로컬 웹 UI와 API |
| `END/reconstructed_training` | 남아 있는 산출물을 바탕으로 재구성한 학습 코드 |
| `lora_sbert_training_code_bundle` | 별도 보관되어 있던 원본 학습·평가 스크립트와 문서 |
| `END/FINAL_PATENT_PIPELINE_REPORT.md` | 파이프라인 및 평가 상세 보고서 |
| `MODEL_CARD.md` | 모델 구성, 성능, 한계 |

이 공개 GitHub 저장소에는 코드와 문서만 포함합니다. 다음 로컬 모델·데이터 경로는 재배포 권한을 확인할 때까지 `.gitignore`로 제외되어 있습니다.

- `END/checkpoint-1500`: 신규성 Qwen2.5-32B LoRA
- `END/checkpoint-2000`: 진보성 Qwen2.5-32B LoRA
- `END/handoff_to_friend/subclass_adapter`: 진보성 부족 세부 유형 Qwen2.5-7B LoRA
- `END/patent30000_sbert_h200/models`: 학습된 SBERT 검색 모델
- `END/patent30000_sbert_h200/index`: 검색 corpus와 임베딩 인덱스
- `END/final_eval_sample_lookup.jsonl`: 문서 원문이 포함된 샘플별 평가 결과

따라서 저장소를 clone한 것만으로는 전체 추론 파이프라인이 실행되지 않습니다. 모델 자산을 별도로 내려받아 위 기본 경로에 배치하거나 실행 인자로 경로를 지정해야 합니다.

## 필요한 기반 모델

추론에는 이 프로젝트에서 학습된 LoRA·SBERT 산출물과 다음 기반 모델이 모두 필요합니다. 기반 모델은 경로가 올바르면 첫 실행 시 Hugging Face에서 자동으로 내려받지만, 프로젝트 학습 산출물은 현재 공개 저장소에서 제공하지 않습니다.

- [`unsloth/Qwen2.5-32B-Instruct`](https://huggingface.co/unsloth/Qwen2.5-32B-Instruct)
- [`Qwen/Qwen2.5-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)

인터넷 연결과 기반 모델을 저장할 충분한 디스크 공간이 필요합니다.

## 하드웨어

기본 실행 설정은 BF16입니다. 32B 기반 모델 자체가 약 64GB의 파라미터 메모리를 사용하며, 진보성 부족 세부 분류 단계에서는 7B 모델도 추가로 로드됩니다.

- 권장: H200급 단일 GPU 또는 합산 메모리가 충분한 다중 GPU 환경
- 80GB GPU: 설정과 메모리 오버헤드에 따라 빠듯하거나 부족할 수 있음
- CPU offload: `device_map=auto`가 사용할 수 있지만 매우 느릴 수 있음
- 세부 분류를 생략하려면 `--skip-subclass` 사용

## 설치

Git을 설치한 다음 저장소를 clone합니다.

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_DIRECTORY>
```

가상환경을 만들고 패키지를 설치합니다.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CUDA 환경에 맞는 PyTorch 빌드가 필요하면 [PyTorch 설치 안내](https://pytorch.org/get-started/locally/)에 따라 PyTorch를 먼저 설치한 뒤 나머지 패키지를 설치하세요.

## 웹 UI 실행

모델과 검색 인덱스를 먼저 기본 경로에 배치한 다음 저장소 루트에서 실행합니다.

```powershell
Set-Location .\END
python .\final_patent_web_server.py --host 127.0.0.1 --port 7860
```

브라우저에서 `http://127.0.0.1:7860`을 엽니다. 모델은 서버 시작 시점이 아니라 첫 추론 요청 때 로드됩니다.

Linux:

```bash
cd END
python final_patent_web_server.py --host 127.0.0.1 --port 7860
```

## 명령행 실행

`END` 폴더에서:

```powershell
python .\final_patent_inferencer.py `
  --title "발명의 명칭" `
  --abstract "발명의 초록" `
  --claim-1 "청구항 1의 내용" `
  --output .\final_inference_output.json
```

텍스트 파일 또는 JSON 입력도 지원합니다.

```powershell
python .\final_patent_inferencer.py --text-file .\input.txt
python .\final_patent_inferencer.py --input-json .\input.json
```

JSON 입력 예:

```json
{
  "id": "sample-001",
  "title": "발명의 명칭",
  "abstract": "발명의 초록",
  "claim_1": "청구항 1"
}
```

## 평가와 학습 코드

- 전체 보고서: `END/FINAL_PATENT_PIPELINE_REPORT.md`
- 재구성 학습 코드 사용법: `END/reconstructed_training/README.md`
- 보관된 학습 코드 묶음: `lora_sbert_training_code_bundle/README_BUNDLE.md`

원본 train/validation/test 데이터 전체는 포함되어 있지 않습니다. 재학습에는 문서에 설명된 형식의 데이터가 별도로 필요합니다.

## GitHub 게시

대용량 ZIP, 모델 가중치, 검색 corpus·인덱스, 샘플별 평가 데이터는 `.gitignore`에서 제외했습니다. 저장소 생성과 push 절차 및 공개 범위는 `docs/PUBLISHING.md`를 참고하세요.

## 라이선스

이 저장소에는 아직 프로젝트 라이선스가 선택되지 않았습니다. 공개 배포 전에 코드, 학습 산출물, 검색 corpus를 어떤 조건으로 재사용할 수 있는지 정한 뒤 루트에 `LICENSE`를 추가해야 합니다.

확인 시점의 Hugging Face 모델 페이지에서 두 Qwen 기반 모델은 Apache-2.0으로 표시됩니다. [`jhgan/ko-sroberta-multitask`](https://huggingface.co/jhgan/ko-sroberta-multitask) 페이지에는 라이선스 표기가 보이지 않으므로 SBERT 파생 가중치를 공개하기 전에 재배포 조건을 별도로 확인해야 합니다.
