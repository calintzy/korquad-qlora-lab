# korquad-qlora-lab

Qwen2.5-1.5B-Instruct를 KorQuAD 1.0으로 QLoRA 파인튜닝하고, 한국어 추출형 QA에서 EM/F1이 실제로 오르는지 before/after로 측정하는 실습.

## 왜

<!-- TODO(자리표시자): 파인튜닝 학습을 직접 손으로 돌려본 경험을 만들기 위한 실습이라는 동기를, 구체적 사건 중심으로 1문단 산문으로 채운다. -->

## 방법

- **모델**: `Qwen/Qwen2.5-1.5B-Instruct` (T4 무료 티어에서 4bit로 로드)
- **데이터**: `KorQuAD/squad_kor_v1` (한국어 추출형 QA)
- **QLoRA**: 4bit(nf4, double-quant, compute dtype fp16) 베이스 + LoRA(r=16, alpha=32, dropout=0.05), 어텐션·MLP 프로젝션 전체를 타깃
- **학습**: `trl` SFTTrainer, conversational 포맷 + `assistant_only_loss`, cosine 스케줄, paged 8bit optimizer
- **평가**: KorQuAD 1.0 공식 문자 단위 EM/F1 채점기(`src/evaluate_korquad.py`)로 채점. before(제로샷)와 after(어댑터)는 어댑터 유무만 다른 동일 경로

### 동일성 계약

프롬프트 문구는 `src/data.py`의 `build_prompt` 한 곳에만 존재한다. `train.py`와 `eval_run.py`가 모두 이 함수를 호출하므로, before/after 비교의 변수는 "어댑터 유무" 하나뿐이다.

## 결과

| 구분 | EM | F1 | n |
|------|----|----|---|
| before (제로샷) | TBD | TBD | TBD |
| after (QLoRA)   | TBD | TBD | TBD |

<!-- TODO(자리표시자): Colab 노트북 S2/S4 실행 후 실제 수치로 교체. -->

## 정직한 한계

<!-- TODO(자리표시자): 서브셋 평가 여부, 1 에폭 학습, 1.5B 소형 모델, greedy 디코딩 등 결과 해석 시 감안할 점을 채운다. -->

## 고지

- 학습에는 KorQuAD 1.0(CC BY-ND)을 사용했으나, **데이터셋 자체는 이 리포에 재배포하지 않는다.**
- 코드와 학습된 어댑터 가중치는 Apache-2.0으로 배포한다.

## 재현 방법

Colab 무료 T4에서 `notebooks/qlora_finetune.ipynb`를 순서대로 실행한다.

- **S0**: GPU 확인 → 의존성 설치 → import·4bit 로딩·1-step 학습 스모크
- **S1**: 채점기 자가검증 + 데이터 로드 확인
- **S2**: 제로샷 평가(before)
- **S3**: QLoRA 본 학습(+ 재개 셀)
- **S4**: 파인튜닝 후 평가(after) + before/after 표 출력

노트북 내 로직은 최소화되어 있고, 단일 진실은 `src/` 스크립트다.
