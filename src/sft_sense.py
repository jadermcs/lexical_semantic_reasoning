import argparse
import random
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import sense_data as sd


def format_example(rec):
    # Conversational prompt/completion format: TRL renders the chat template and
    # masks the prompt so loss is computed on the assistant turn only.
    msgs = sd.wic_messages(rec, with_target=True)
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def load_dataset(wic_data, strategy="first", dev_frac=0.05, seed=42):
    """Distilled traces, split into train/dev.

    The distillation source is a single teacher predictions file (no prebuilt
    train/dev splits), so a small held-out dev set is carved off a deterministic
    shuffle.
    """
    recs = sd.load_teacher_traces(wic_data, strategy=strategy)
    random.Random(seed).shuffle(recs)
    n_dev = max(1, int(len(recs) * dev_frac))
    dev, train = recs[:n_dev], recs[n_dev:]
    return DatasetDict(
        {"train": Dataset.from_list(train), "dev": Dataset.from_list(dev)}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument(
        "--data",
        default="data/mcl_semcor.json",
        help="Teacher predictions file written by call_api.py.",
    )
    ap.add_argument(
        "--reasoning-select",
        choices=["first", "longest"],
        default="first",
        help="Which distilled teacher trace to keep per pair: first majority-voting "
        "sample, longest CoT, or the highest model predictive entropy",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    dataset = load_dataset(args.data, strategy=args.reasoning_select)
    print(
        f"[wic] train={len(dataset['train'])} dev={len(dataset['dev'])} "
        f"reasoning-select={args.reasoning_select}"
    )

    cols = dataset["train"].column_names
    dataset = dataset.map(format_example, remove_columns=cols)
    print(dataset["train"][0])

    # Tag runs with the ablation so different strategies get separate output dirs /
    # wandb runs and never resume from each other's checkpoints.
    data_tag = Path(args.data).stem
    tag = f"wic-{args.reasoning_select}"
    output_dir = f"./qwen-sense-sft-{tag}-{data_tag}"
    training_args = SFTConfig(
        output_dir=output_dir,
        completion_only_loss=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        warmup_steps=100,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=150,
        save_steps=150,
        save_total_limit=2,
        load_best_model_at_end=True,
        # Select by greedy-decode proxy (argmax correctness) rather than NLL,
        # which can rise from overconfidence while decode behavior still improves.
        metric_for_best_model="eval_mean_token_accuracy",
        greater_is_better=True,
        report_to="wandb",
        run_name=f"qwen-sense-sft-{tag}-{data_tag}",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=training_args,
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
