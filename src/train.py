"""QLoRA SFT 학습 러너 (Qwen2.5-1.5B-Instruct, T4 4bit).

주의(리서치 발견 — 미확정 리스크):
    SFTTrainer 1.9.2가 양자화 모델의 LoRA 파라미터를 bf16으로 자동
    캐스팅한다는 보고가 있다. 여기서 강제하는 fp16(SFTConfig fp16=True,
    bf16=False)과 T4에서 충돌할 가능성은 미확인이다. 노트북 S0의 1-step
    스모크(`--max-steps 1`)가 이 리스크의 실측 프로브다 — dtype 크래시
    없이 1스텝이 통과하면 리스크 해소로 본다.
"""

import argparse
import glob
import os
import sys

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# 스크립트 디렉토리(src/)를 경로에 추가해 로컬 모듈을 import한다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data  # noqa: E402

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="QLoRA SFT 학습")
    parser.add_argument("--output-dir", required=True, help="어댑터 저장 경로(Drive 권장)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1, help="스모크용(-1이면 미사용)")
    parser.add_argument("--resume", action="store_true", help="체크포인트에서 재개")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print(
        f"[train] model={MODEL_NAME} output={args.output_dir} "
        f"epochs={args.epochs} max_steps={args.max_steps} "
        f"bs={args.batch_size} grad_accum={args.grad_accum} "
        f"lr={args.lr} max_length={args.max_length}"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # transformers 5.x: load_in_4bit 단축 경로 제거 — 객체 필수. compute_dtype은 T4 고정 fp16.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    dataset = data.load_korquad()
    train_split = dataset["train"]

    def keep_example(example):
        # KorQuAD v1은 전 문항 정답이 존재한다는 전제 — 방어적으로 빈 answers 제외.
        texts = example["answers"]["text"]
        if len(texts) == 0:
            return False
        # max_length 우측 절단이 긴 예제의 assistant 정답을 잘라 학습 신호를 0으로
        # 만드는 문제 방어: chat template 적용 토큰 길이가 max_length 초과면 제외한다
        # (CPU 토크나이즈만 — GPU 불필요).
        messages = data.build_prompt(example["context"], example["question"])
        messages.append({"role": "assistant", "content": texts[0]})
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return len(tokenizer(text)["input_ids"]) <= args.max_length

    before_n = len(train_split)
    train_split = train_split.filter(keep_example)
    print(f"filtered {before_n - len(train_split)} examples exceeding max_length")

    def to_messages(example):
        # 학습 예제 = build_prompt 결과 + assistant(정답) 부착 (동일성 계약, 단일 경로).
        messages = data.build_prompt(example["context"], example["question"])
        messages.append({"role": "assistant", "content": example["answers"]["text"][0]})
        return {"messages": messages}

    train_ds = train_split.map(
        to_messages,
        remove_columns=train_split.column_names,
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        save_strategy="steps",
        save_steps=500,
        logging_steps=20,
        packing=False,
        # Qwen2.5는 trl이 generation 마커 있는 학습용 템플릿으로 자동 교체.
        assistant_only_loss=True,
        # T4는 bf16 미지원 — fp16 반드시 명시(미명시 시 SFTConfig가 bf16 기본값을 강제해 크래시).
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        # kbit+PEFT 조합 권장값 — non-reentrant 체크포인팅.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=sft_config,
        train_dataset=train_ds,
        quantization_config=bnb_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # T4 실측 크래시 대응(2026-08-01): trl이 양자화 모델의 LoRA 파라미터를
    # bf16으로 자동 캐스팅하는데, T4의 fp16 GradScaler는 bf16 그래디언트를
    # unscale하는 CUDA 커널이 없다("_amp_foreach_non_finite_check_and_unscale_cuda"
    # not implemented for 'BFloat16'). 학습 파라미터를 fp32로 되돌린다
    # (fp16 AMP + fp32 LoRA 파라미터가 QLoRA 표준 조합).
    n_cast = 0
    for p in trainer.model.parameters():
        if p.requires_grad and p.dtype == torch.bfloat16:
            p.data = p.data.float()
            n_cast += 1
    print(f"[train] upcast {n_cast} trainable bf16 params to fp32 (T4 fp16 AMP fix)")

    if args.resume:
        # resume 가드: 체크포인트가 실제로 있을 때만 재개(없으면 크래시 방지).
        ckpts = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        if ckpts:
            trainer.train(resume_from_checkpoint=True)
        else:
            print("[train] --resume 지정됐으나 checkpoint-* 없음 — 처음부터 학습")
            trainer.train()
    else:
        trainer.train()

    trainer.save_model(args.output_dir)
    print(f"[train] done. adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
