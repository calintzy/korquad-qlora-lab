"""데이터 로드 + 단일 프롬프트 포맷터.

동일성 계약(계획서 §4)의 핵심: 학습(train.py)과 평가(eval_run.py)가
반드시 이 모듈의 build_prompt 하나만 호출한다. 프롬프트 문구가 한 곳에만
존재해야 before/after 비교가 "어댑터 유무"라는 단일 변수만 갖는다.
"""

import json
import os
import random

from datasets import load_dataset

# 프롬프트 지시문 — 정확 문구는 여기 한 곳에만 존재한다(동일성 계약).
INSTRUCTION = "다음 지문을 읽고 질문에 짧게 답하라. 지문에 있는 표현 그대로 답하라."


def load_korquad():
    """KorQuAD v1.0 데이터셋을 로드한다(train/validation split 포함)."""
    return load_dataset("KorQuAD/squad_kor_v1")


def build_prompt(context: str, question: str) -> list[dict]:
    """system + user 2-메시지 대화형 프롬프트를 만든다.

    학습·평가 공용. assistant 메시지는 호출자가 필요 시 부착한다.
    """
    return [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": f"지문:\n{context}\n\n질문: {question}"},
    ]


def select_eval_subset(dataset, subset_file, n=None, seed=42):
    """평가 서브셋 선택. 모드 감지 규칙: subset_file 존재 = 서브셋 모드.

    - subset_file 존재: 저장된 인덱스를 로드해 그대로 적용(재현성 보장).
    - subset_file 없고 n 지정: 고정 시드로 n개 샘플 후 인덱스를 subset_file에 저장.
    - 둘 다 없음: 전체 반환.
    """
    if subset_file and os.path.exists(subset_file):
        with open(subset_file) as f:
            indices = json.load(f)
        return dataset.select(indices)

    if n is not None:
        rng = random.Random(seed)
        indices = sorted(rng.sample(range(len(dataset)), min(n, len(dataset))))
        if subset_file:
            os.makedirs(os.path.dirname(subset_file) or ".", exist_ok=True)
            with open(subset_file, "w") as f:
                json.dump(indices, f)
        return dataset.select(indices)

    return dataset
