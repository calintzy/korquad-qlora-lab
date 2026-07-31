"""KorQuAD 평가 스크립트(src/evaluate_korquad.py) 자가검증 테스트.

실행: python3 tests/test_scorer.py
pytest 불필요 — 순수 stdlib(assert) 기반. 리포 루트 기준 상대 import를 위해
sys.path에 리포 루트를 추가한다.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.evaluate_korquad import (  # noqa: E402
    normalize_answer,
    f1_score,
    exact_match_score,
    evaluate_pairs,
)

failures = []


def check(name, condition, expected=None, actual=None):
    if condition:
        print(f"PASS {name}")
    else:
        detail = ""
        if expected is not None or actual is not None:
            detail = f" (expected={expected!r}, actual={actual!r})"
        print(f"FAIL {name}{detail}")
        failures.append(name)


# --- ISC-1.2a: 예측 == 정답 -> EM 100.0, F1 100.0 ---
result_a = evaluate_pairs(
    predictions={"q1": "대한민국 임시정부"},
    references={"q1": ["대한민국 임시정부"]},
)
check(
    "ISC-1.2a exact-match-em",
    result_a["exact_match"] == 100.0,
    100.0,
    result_a["exact_match"],
)
check(
    "ISC-1.2a exact-match-f1",
    result_a["f1"] == 100.0,
    100.0,
    result_a["f1"],
)

# --- ISC-1.2b: 완전 오답(문자 겹침 없음) -> EM 0.0, F1 0.0 ---
result_b = evaluate_pairs(
    predictions={"q2": "모차르트"},
    references={"q2": ["베토벤"]},
)
check(
    "ISC-1.2b wrong-answer-em",
    result_b["exact_match"] == 0.0,
    0.0,
    result_b["exact_match"],
)
check(
    "ISC-1.2b wrong-answer-f1",
    result_b["f1"] == 0.0,
    0.0,
    result_b["f1"],
)

# --- ISC-1.3: 문자 단위 F1 증명 ---
# 정답 "베토벤"(3자: 베,토,벤), 예측 "베토벤이"(4자: 베,토,벤,이)
# 공통 문자 수 = 3 (베,토,벤). precision = 3/4 = 0.75, recall = 3/3 = 1.0
# f1 = 2*precision*recall / (precision+recall) = 2*0.75*1.0/1.75 = 1.5/1.75 = 6/7 ≈ 0.857142857
# 단어 단위 F1이었다면 "베토벤"과 "베토벤이"는 공통 단어가 0개라 F1=0이 됐을 케이스.
em_josa = exact_match_score("베토벤이", "베토벤")
f1_josa = f1_score("베토벤이", "베토벤")
expected_f1_josa = 6.0 / 7.0

check(
    "ISC-1.3 josa-em-is-zero",
    em_josa is False,
    False,
    em_josa,
)
check(
    "ISC-1.3 josa-f1-char-level",
    f1_josa >= 0.85 and abs(f1_josa - expected_f1_josa) < 1e-9,
    f"f1>=0.85 (검산값={expected_f1_josa:.6f})",
    f1_josa,
)

# --- 정규화 케이스: 따옴표/괄호 제거 확인 ---
norm_quoted = normalize_answer('"베토벤"')
norm_plain = normalize_answer("베토벤")
check(
    "normalize-quote-strip",
    norm_quoted == norm_plain,
    norm_plain,
    norm_quoted,
)

if failures:
    print(f"{len(failures)} FAILED")
    sys.exit(1)
else:
    print("ALL PASS")
