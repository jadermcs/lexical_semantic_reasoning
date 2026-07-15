"""GRPO on the WiC task, warm-started from an SFT checkpoint.

The policy reasons about the gloss of each of two usages, then classifies the pair
as the same sense or a different sense. The label is gold (MCL-WiC), so the reward
is verifiable — see ``sense_rewards``, which is importable without the training
stack and unit-tested in ``tests/test_sense_rewards.py``.

The examples rolled out against are read from ``data/wic_task.<split>.jsonl`` (built
on first use by ``wic_task.py``), so they can be inspected before a run.

    uv run python src/grpo_sense.py --model ./qwen-sense-sft-wic-longest
"""

import argparse
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_repetition_penalty_reward

import sense_data as sd
import wic_task
from sense_rewards import KEEP_COLS, REWARDS, make_trace_saver


def format_prompt(rec, tokenizer):
    """Render the prompt column; gold columns are kept so reward fns see them as kwargs."""
    msgs = sd.wic_messages(rec, with_target=False)
    return {"prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}


def build_dataset(split, tokenizer, cap=None):
    ds = Dataset.from_list(wic_task.load(split))
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(partial(format_prompt, tokenizer=tokenizer), remove_columns=drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument(
        "--distill-out",
        default=None,
        help="If set, append correct completions to '<path>.rank<N>.jsonl' for "
        "self-distillation. Does not affect training.",
    )
    ap.add_argument("--distill-threshold", type=float, default=0.5)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset("train", tokenizer)
    dev_ds = build_dataset("dev", tokenizer, cap=200)
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

    run_name = "qwen-sense-grpo-wic"
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

    # trl ships a token-id-based repetition penalty; use it instead of hand-rolling
    # one. It guards against the degenerate-repetition reward hacking the
    # format/length shaping can invite (see reward_wic_json). max_penalty is kept
    # small so it stays well under the +/-1 accuracy term — being right dominates.
    reward_funcs = list(REWARDS) + [get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.2)]
    if args.distill_out:
        reward_funcs.append(make_trace_saver(args.distill_out, args.distill_threshold))
        print(
            f"Self-distillation: saving completions scoring >= "
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
