import argparse
from pathlib import Path

import torch
from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate. 1e-4 suits the 0.6B; scale it down for anything "
        "bigger (2e-5 for 2B) or the loss climbs through warmup and never recovers.",
    )
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
        help="Where to save checkpoints/adapter. Defaults to ./qwen-<data-stem>.",
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
    output_dir = args.output_dir or f"./qwen-{data_tag}"
    training_args = SFTConfig(
        output_dir=output_dir,
        completion_only_loss=True,
        max_length=2048,
        packing=True,
        use_liger_kernel=True,
        weight_decay=0.1,
        dataset_num_proc=8,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=0.03,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"qwen-sft-{data_tag}",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved final adapter → {output_dir}")


if __name__ == "__main__":
    main()
