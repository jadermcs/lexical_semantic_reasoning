"""GRPO for the sense-modeling ablations, warm-started from an SFT checkpoint.

Train on any subset of tasks at once (multitask) via ``--tasks``:

  direct   -> config 3: RL on single-usage definition generation
  triplet  -> config 4: RL on anchor/positive/negative contrastive glosses
  wic      -> reason about the gloss of each of two usages, then classify the pair
              as the same sense or a different sense (verifiable label reward)

The reward functions themselves live in ``sense_rewards`` (importable without the
training stack, and unit-tested in ``tests/test_sense_rewards.py``). When more than
one task is given, each task's reward functions only score that task's completions
(see ``sense_rewards.mask_by_task``).
"""

import argparse
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import sense_data as sd
import wic_task
from sense_rewards import KEEP_COLS, REWARDS, make_trace_saver, mask_by_task


TASKS = ("direct", "triplet", "wic", "supersense")

# Message builder per task; format_prompt renders the prompt column from it.
_MSG_FN = {
    "direct": sd.direct_messages,
    "triplet": sd.triplet_messages,
    "wic": sd.wic_messages,
    "supersense": sd.supersense_messages,
}


# --------------------------------------------------------------------------- #
# Prompt formatting (keeps gold columns so reward fns can read them via kwargs)
# --------------------------------------------------------------------------- #
def format_prompt(rec, tokenizer, task):
    msgs = _MSG_FN[task](rec, with_target=False)
    return {"prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}


def _load_split(task, split):
    """Load one task/split, building all datasets on first use if missing.

    ``wic`` trains on the gold MCL-WiC benchmark (verifiable same/different label,
    no glosses), read from the dumped ``data/wic_task.<split>.jsonl`` so the
    examples rolled out against can be inspected before the run (see
    ``wic_task.py``); the WordNet-built ``direct``/``triplet`` splits are generated
    on first use if absent.
    """
    if task == "wic":
        return wic_task.load(split)
    try:
        return sd.load_split(task, split)
    except FileNotFoundError:
        sd.save_dataset(sd.build_dataset())
        return sd.load_split(task, split)


def _task_dataset(task, split, tokenizer, dev_cap=None):
    """Rendered prompts + kept gold columns + a ``task`` tag for one task/split."""
    recs = _load_split(task, split)
    ds = Dataset.from_list(recs)
    if dev_cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(dev_cap, len(ds))))
    fmt = partial(format_prompt, tokenizer=tokenizer, task=task)
    drop = [c for c in ds.column_names if c not in KEEP_COLS[task]]
    ds = ds.map(fmt, remove_columns=drop)
    return ds.add_column("task", [task] * len(ds))


def _combine(tasks, split, tokenizer, dev_cap=None):
    """Concatenate per-task datasets, padding each to the shared column union.

    Different tasks keep different gold columns; ``concatenate_datasets`` needs a
    single schema, so missing columns are filled with "" (all kept columns are
    strings). ``mask_by_task`` filters these padded rows back out per reward fn.
    """
    parts = [_task_dataset(t, split, tokenizer, dev_cap) for t in tasks]
    all_cols = sorted({c for p in parts for c in p.column_names})
    padded = []
    for p in parts:
        for c in all_cols:
            if c not in p.column_names:
                p = p.add_column(c, [""] * len(p))
        padded.append(p.select_columns(all_cols))
    return concatenate_datasets(padded).shuffle(seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        required=True,
        help="One or more tasks to train on jointly, e.g. --tasks direct triplet wic.",
    )
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument(
        "--distill-out",
        default=None,
        help="If set, append completions with fidelity >= --distill-threshold to "
        "'<path>.rank<N>.jsonl' for self-distillation. Does not affect training.",
    )
    ap.add_argument("--distill-threshold", type=float, default=0.5)
    args = ap.parse_args()

    # De-duplicate while preserving order so the run name is stable.
    tasks = list(dict.fromkeys(args.tasks))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = _combine(tasks, "train", tokenizer)
    dev_ds = _combine(tasks, "dev", tokenizer, dev_cap=200)
    print(train_ds[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )

    vllm_kwargs = {}
    if args.vllm_server_host:
        vllm_kwargs = dict(
            use_vllm=True, vllm_mode="server",
            vllm_server_host=args.vllm_server_host, vllm_server_port=args.vllm_server_port,
        )
    else:
        vllm_kwargs = dict()

    run_name = f"qwen-sense-grpo-{'-'.join(tasks)}"
    output_dir = f"./{run_name}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=8,
        max_completion_length=1024,
        optim="paged_adamw_8bit",
        temperature=1.0,
        top_p=0.95,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=50,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        log_completions=True,
        num_completions_to_print=8,
        report_to="wandb",
        run_name=run_name,
        use_liger_kernel=True,
        **vllm_kwargs,
    )

    # Each task's reward fns (and optional trace saver) are masked to their own
    # task so they never perturb the other tasks' groups.
    reward_funcs = []
    for task in tasks:
        funcs = list(REWARDS[task])
        if args.distill_out:
            funcs.append(make_trace_saver(task, args.distill_out, args.distill_threshold))
        reward_funcs.extend(mask_by_task(task, fn) for fn in funcs)
    if args.distill_out:
        print(
            f"Self-distillation: saving completions with fidelity >= "
            f"{args.distill_threshold} to {args.distill_out}.rank*.jsonl"
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
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
