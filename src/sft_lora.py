import argparse
from pathlib import Path

import torch
from datasets import DatasetDict
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lora-r", type=int, default=256, help="LoRA rank.")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling.")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    ap.add_argument(
        "--data",
        default="data/sft_wic",
        help="Prepared dataset dir written by prepare_data.py (a DatasetDict with "
        "train/dev splits of {prompt, completion} examples). Build it first with "
        "`uv run python src/prepare_data.py ...` so the data can be inspected before "
        "training and isn't re-processed every run.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Where to save checkpoints/adapter. Defaults to ./qwen-lora-<data-stem>.",
    )
    ap.add_argument(
        "--merged-dir",
        default=None,
        help="Where to write the adapter merged back into the base weights. Defaults "
        "to '<output-dir>-merged'. This merged model is what grpo_lora.py consumes: "
        "GRPO then starts a fresh LoRA on top of the warm-started weights instead of "
        "continuing the SFT adapter. Pass '' to skip merging.",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="kernels-community/flash-attn2",
    )

    dataset = DatasetDict.load_from_disk(args.data)
    print(f"[sft] train={len(dataset['train'])} dev={len(dataset['dev'])} data={args.data}")
    print(dataset["train"][0])

    data_tag = Path(args.data.rstrip("/")).stem
    output_dir = args.output_dir or f"./qwen-lora-{data_tag}"
    training_args = SFTConfig(
        output_dir=output_dir,
        completion_only_loss=True,
        max_length=1024,
        packing=True,
        use_liger_kernel=True,
        dataset_num_proc=8,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=0.03,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"qwen-lora-sft-{data_tag}",
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=training_args,
        peft_config=peft_config,
    )

    last = None
    out = Path(output_dir)
    if out.exists():
        cks = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if cks:
            last = str(cks[-1])
            print(f"Resuming from checkpoint: {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(output_dir)
    print(f"Saved final adapter → {output_dir}")

    merged_dir = args.merged_dir if args.merged_dir is not None else f"{output_dir}-merged"
    if merged_dir:
        # Fold the adapter into the base weights and save a plain HF model, so the
        # downstream GRPO run attaches a *fresh* LoRA to the warm-started policy
        # (and its adapter-disabled base is the correct KL reference).
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model → {merged_dir}")


if __name__ == "__main__":
    main()
