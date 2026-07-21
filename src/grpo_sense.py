import argparse
import json
import random
from functools import partial
from pathlib import Path

import torch
torch._dynamo.config.disable = True

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_repetition_penalty_reward

import sense_data as sd
from sense_rewards import KEEP_COLS, REWARDS, make_trace_saver


def format_prompt(rec, tokenizer):
    """Render the prompt column; gold columns are kept so reward fns see them as kwargs."""
    msgs = sd.wic_messages(rec, with_target=False)
    return {"prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}


def load_exclude_pairs(path):
    """Read a prepare_data ``*.sft_pairs.json`` manifest into a set of pair keys."""
    if path is None:
        return None
    return {tuple(k) for k in json.loads(Path(path).read_text())}


def balance_labels(recs, seed=42):
    """Down-sample the majority same/different class so the set is 50/50.

    The teacher-failed complement fed to GRPO via ``--exclude-pairs`` is heavily
    skewed toward ``same_sense=true`` (the teacher under-predicts "same", so the
    pairs it gets wrong are mostly the "same" ones). On a skewed rollout set the
    cheapest reward gradient is to collapse to the majority class, which tanks
    accuracy on the balanced eval/test splits. Balancing removes that shortcut.
    """
    same = [r for r in recs if r["label"]]
    diff = [r for r in recs if not r["label"]]
    n = min(len(same), len(diff))
    rng = random.Random(seed)
    rng.shuffle(same)
    rng.shuffle(diff)
    out = same[:n] + diff[:n]
    rng.shuffle(out)
    return out


def build_dataset(split, tokenizer, cap=None, exclude_pairs=None, balance=False):
    recs = sd.load_mclwic(split, exclude_pairs=exclude_pairs)
    if balance:
        before = len(recs)
        recs = balance_labels(recs)
        print(f"[{split}] balanced labels: {before} → {len(recs)} pairs (50/50)")
    ds = Dataset.from_list(recs)
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
    ap.add_argument(
        "--max-completion-length",
        type=int,
        default=640,
        help="Max generated tokens per rollout. Main VRAM knob — lower it to fit. "
        "640 clears the p99.7 of distilled WiC traces (~520 tokens); truncated "
        "rollouts are masked out (mask_truncated_completions), so a tight cap costs "
        "wasted rollout compute, not wrong gradients.",
    )
    ap.add_argument(
        "--exclude-pairs",
        default=None,
        help="prepare_data '<out>.sft_pairs.json' manifest of WiC pairs already "
        "distilled into the SFT set; those pairs are held out of the GRPO train "
        "rollout set so RL only sees pairs the warm-start never saw.",
    )
    ap.add_argument(
        "--balance-labels",
        action="store_true",
        help="Down-sample the majority class so the train rollout set is 50/50 "
        "same/different. Strongly recommended with --exclude-pairs: the "
        "teacher-failed complement skews heavily toward same_sense=true, and the "
        "policy otherwise collapses to the majority class.",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="KL coefficient anchoring the policy to the SFT reference. >0 loads a "
        "reference model and stops the policy drifting off the warm-start init "
        "(set 0 to disable).",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    exclude = load_exclude_pairs(args.exclude_pairs)
    train_ds = build_dataset("train", tokenizer, exclude_pairs=exclude, balance=args.balance_labels)
    # dev stays as-is: it's the held-out eval signal and should reflect the true
    # (balanced) label distribution rather than a re-balanced subsample.
    dev_ds = build_dataset("dev", tokenizer, cap=200, exclude_pairs=exclude)
    if exclude:
        print(f"Excluding {len(exclude)} SFT-consumed pairs from GRPO train set → {len(train_ds)} rollout pairs")
    print(train_ds[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="kernels-community/flash-attn2",
    )

    if args.vllm_server_host:
        vllm_kwargs = dict(
            use_vllm=True, vllm_mode="server",
            vllm_server_host=args.vllm_server_host, vllm_server_port=args.vllm_server_port,
        )
    else:
        prompt_headroom = 512
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_max_model_length=prompt_headroom + args.max_completion_length,
        )

    run_name = "qwen-sense-grpo-wic"
    output_dir = f"./{run_name}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=8,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
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
        num_completions_to_print=3,
        report_to="wandb",
        run_name=run_name,
        use_liger_kernel=True,
        **vllm_kwargs,
    )

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
