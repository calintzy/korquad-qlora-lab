"""제로샷/파인튜닝 후 평가 러너 (before/after 공용).

before/after의 유일한 차이는 어댑터 유무다: 4bit 베이스 로딩·양자화 설정·
프롬프트(data.build_prompt)·greedy 생성·채점(evaluate_korquad)이 모두 동일하다.
--adapter를 주면 after(파인튜닝), 안 주면 before(제로샷)다.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# 스크립트 디렉토리(src/)를 경로에 추가해 로컬 모듈을 import한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402
from evaluate_korquad import evaluate_pairs  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="KorQuAD 평가 러너")
    parser.add_argument("--adapter", default=None, help="어댑터 경로(있으면 after, 없으면 zero-shot)")
    parser.add_argument("--output", required=True, help="결과 JSON 경로")
    parser.add_argument("--subset-file", default="results/subset_indices.json")
    parser.add_argument("--subset-n", type=int, default=None, help="최초 서브셋 생성용 표본 수")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--predictions-out", default=None, help="id→예측 저장 경로(선택)")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # before/after 동일 양자화 — 유일한 차이는 어댑터 유무.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    mode = "zero-shot(before)"
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
        mode = f"adapter(after): {args.adapter}"
    model.eval()

    dataset = data.load_korquad()
    dev = dataset["validation"]
    dev = data.select_eval_subset(dev, args.subset_file, n=args.subset_n)

    # 평가 배치 생성은 left padding으로 전환(학습은 right 기본).
    tokenizer.padding_side = "left"
    # 우측 절단은 생성 프롬프트(끝의 질문·generation 마커)를 잘라 사고를 낸다.
    # 좌측 절단으로 전환하면 넘칠 때 지문 앞부분만 희생돼 프롬프트 형태가 보존된다.
    tokenizer.truncation_side = "left"

    ids = dev["id"]
    contexts = dev["context"]
    questions = dev["question"]
    answers = dev["answers"]
    total = len(dev)
    print(f"[eval] mode={mode} n={total} batch_size={args.batch_size}")

    predictions = {}
    references = {}
    bs = args.batch_size

    for start in range(0, total, bs):
        end = min(start + bs, total)
        prompts = []
        for c, q in zip(contexts[start:end], questions[start:end]):
            messages = data.build_prompt(c, q)
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
            )

        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j in range(end - start):
            qid = ids[start + j]
            # 후처리: strip 후 첫 개행 이전만 사용.
            pred = decoded[j].strip().split("\n")[0].strip()
            predictions[qid] = pred
            references[qid] = list(answers[start + j]["text"])

        print(f"[eval] {end}/{total}")

    metrics = evaluate_pairs(predictions, references)
    result = {
        "em": metrics["exact_match"],
        "f1": metrics["f1"],
        "n": metrics["total"],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    if args.predictions_out:
        os.makedirs(os.path.dirname(args.predictions_out) or ".", exist_ok=True)
        with open(args.predictions_out, "w") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"em={result['em']:.2f} f1={result['f1']:.2f} n={result['n']}")


if __name__ == "__main__":
    main()
