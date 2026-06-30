"""SFT warm-start for the sense-modeling ablations.

  --mode direct   -> config 1: single-usage definition generation
  --mode triplet  -> config 2: anchor/positive/negative contrastive glosses

"""

import argparse
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import sense_data as sd


def format_example(rec, tokenizer, mode):
    msgs = (sd.direct_messages if mode == "direct" else sd.triplet_messages)(
        rec, with_target=True
    )
    return {"text": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}


def _load_or_build(mode):
    try:
        train = sd.load_split(mode, "train")
        dev = sd.load_split(mode, "dev")
    except FileNotFoundError:
        print("Splits not found; building them via sense_data ...")
        sd.save_dataset(sd.build_dataset())
        train, dev = sd.load_split(mode, "train"), sd.load_split(mode, "dev")
    return DatasetDict({"train": Dataset.from_list(train), "dev": Dataset.from_list(dev)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--mode", choices=["direct", "triplet"], required=True)
    ap.add_argument("--epochs", type=int, default=2)
    args = ap.parse_args()

    dataset = _load_or_build(args.mode)
    print(f"[{args.mode}] train={len(dataset['train'])} dev={len(dataset['dev'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    fmt = partial(format_example, tokenizer=tokenizer, mode=args.mode)
    cols = dataset["train"].column_names
    dataset = dataset.map(fmt, remove_columns=cols)
    print(dataset["train"][0]["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )

    output_dir = f"./qwen-sense-sft-{args.mode}"
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=64,
        num_train_epochs=args.epochs,
        warmup_steps=100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"qwen-sense-sft-{args.mode}",
    )
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer,
        train_dataset=dataset["train"], eval_dataset=dataset["dev"], args=training_args,
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


if __name__ == "__main__":
    main()
