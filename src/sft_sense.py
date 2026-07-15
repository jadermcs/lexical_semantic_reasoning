"""SFT warm-start for the WiC task, distilled from teacher reasoning traces.

Trains on the pairs a teacher model got right (``sense_data.load_teacher_traces`` over the
``call_api.py`` output), so the policy learns both the <think> reasoning and the
JSON answer contract before GRPO tightens the verdict.

    uv run python src/sft_sense.py --reasoning-select longest
"""

import argparse
import random
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import sense_data as sd


def format_example(rec, tokenizer):
    msgs = sd.wic_messages(rec, with_target=True)
    return {
        "text": tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )
    }


@torch.no_grad()
def _make_entropy_scorer(model, tokenizer, max_cont_tokens=2048):
    """Score each candidate trace by the model's mean predictive entropy over it.

    The trace is scored in context: the wic prompt (no target) is the prefix, the
    candidate ``<think>`` block the continuation, and we average the Shannon entropy
    of the model's next-token distribution across the continuation positions. A
    higher score means the model is more uncertain reading that trace — the signal
    the ``entropy`` reasoning-select ablation maximises. Returns a per-candidate
    list aligned with ``cands``.
    """
    model.eval()

    def scorer(rec, cands):
        prompt_ids = tokenizer.apply_chat_template(
            sd.wic_messages(rec, with_target=False),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        scores = []
        for c in cands:
            cont = tokenizer(
                f"<think>\n{c['think']}\n</think>",
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=max_cont_tokens,
            ).input_ids.to(model.device)
            ids = torch.cat([prompt_ids, cont], dim=1)
            logits = model(ids).logits[0]  # [T, V]
            # logits[t] predicts token t+1; positions [len(prompt)-1, T-1) emit the
            # continuation tokens, so their next-token entropy is what we average.
            start = prompt_ids.shape[1]
            logp = torch.log_softmax(logits[start - 1 : -1], dim=-1)
            ent = -(logp.exp() * logp).sum(dim=-1)  # [len(cont)]
            scores.append(ent.mean().item())
        return scores

    return scorer


def load_dataset(wic_data, strategy="first", scorer=None, dev_frac=0.05, seed=42):
    """Distilled traces, split into train/dev.

    The distillation source is a single teacher predictions file (no prebuilt
    train/dev splits), so a small held-out dev set is carved off a deterministic
    shuffle.
    """
    recs = sd.load_teacher_traces(wic_data, strategy=strategy, scorer=scorer)
    random.Random(seed).shuffle(recs)
    n_dev = max(1, int(len(recs) * dev_frac))
    dev, train = recs[:n_dev], recs[n_dev:]
    return DatasetDict(
        {"train": Dataset.from_list(train), "dev": Dataset.from_list(dev)}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--wic-data",
        default="data/mcl_semcor.json",
        help="Teacher predictions file written by call_api.py.",
    )
    ap.add_argument(
        "--reasoning-select",
        choices=["first", "longest", "entropy"],
        default="first",
        help="Which distilled teacher trace to keep per pair: first majority-voting "
        "sample, longest CoT, or the highest model predictive entropy",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model before building the dataset: the entropy reasoning-select
    # strategy scores candidate traces with a forward pass, so it needs the model.
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    scorer = None
    if args.reasoning_select == "entropy":
        scorer = _make_entropy_scorer(model, tokenizer)

    dataset = load_dataset(args.wic_data, strategy=args.reasoning_select, scorer=scorer)
    print(
        f"[wic] train={len(dataset['train'])} dev={len(dataset['dev'])} "
        f"reasoning-select={args.reasoning_select}"
    )

    fmt = partial(format_example, tokenizer=tokenizer)
    cols = dataset["train"].column_names
    dataset = dataset.map(fmt, remove_columns=cols)
    print(dataset["train"][0]["text"])

    # Tag runs with the ablation so different strategies get separate output dirs /
    # wandb runs and never resume from each other's checkpoints.
    tag = f"wic-{args.reasoning_select}"
    output_dir = f"./qwen-sense-sft-{tag}"
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        warmup_steps=100,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=f"qwen-sense-sft-{tag}",
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
