import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils import build_sft_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--file_path", type=str, default="")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--warmup-steps", type=float, default=0.02)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    grad_accum = 16 // args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    ds = build_sft_dataset(args.file_path)
    train_ds = ds["train"]
    dev_ds = ds["test"]
    print(train_ds[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="kernels-community/flash-attn2",
    )

    print(f"[sft] train={len(train_ds)} dev={len(dev_ds)} data={args.file_path}")

    data_tag = Path(args.file_path.rstrip("/")).stem
    output_dir = args.output_dir or f"./qwen-{data_tag}"
    training_args = SFTConfig(
        output_dir=output_dir,
        run_name=f"{args.model.split('/')[-1]}-sft-{data_tag}",
        project="sense-sft",
        completion_only_loss=True,
        max_length=1024,
        packing=True,
        use_liger_kernel=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved final model → {output_dir}")


if __name__ == "__main__":
    main()
