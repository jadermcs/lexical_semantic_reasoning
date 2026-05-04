"""
Self-verification GRPO for WiC, following Chen et al. 2026
("Learning to Self-Verify Makes Language Models Better Reasoners", arXiv:2602.07594).

Two integration strategies are supported:
  * verify-init  : pure self-verification training, then standard generation GRPO.
  * verify-alter : alternate generation GRPO and self-verification GRPO every N steps.

Self-verification: given a (query, candidate-answer) pair, the model judges whether
the candidate's final answer is correct. Reward = 1 if judgment matches the
rule-based correctness label c ∈ {0,1}, else -1 (-0.5 if malformed).
"""

import argparse
import random
import re
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from grpo_finetune import (
    SYSTEM_PROMPT,
    _extract_answer,
    format_prompt,
    mark_target,
    reward_correctness,
    reward_format,
    reward_reasoning_quality,
)
from utils import load_data

VERIFY_SYSTEM_PROMPT = (
    "You are a linguistic expert specializing in word sense disambiguation. "
    "You will be shown a Word-in-Context question and a candidate answer with "
    "its reasoning trace. Your task is to judge whether the candidate's final "
    "answer is correct.\n\n"
    "Reason step by step inside <think> tags:\n"
    "  1. Restate each gloss the candidate proposed.\n"
    "  2. Check whether the glosses are accurate for the marked uses.\n"
    "  3. Decide whether the candidate's final true/false answer follows.\n"
    "Then output exactly one of "
    "<verdict>correct</verdict> or <verdict>incorrect</verdict>."
)


# ---------------------------------------------------------------------------
# Verification prompt formatting
# ---------------------------------------------------------------------------


def format_verify_prompt(example, candidate_completion: str, tokenizer) -> str:
    s1 = mark_target(example["sentence1"], example["word1"])
    s2 = mark_target(example["sentence2"], example["word2"])
    messages = [
        {"role": "system", "content": VERIFY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Word: {example['lemma']} ({example['pos']})\n"
                f"Sentence 1: {s1}\n"
                f"Sentence 2: {s2}\n\n"
                f"Candidate answer:\n{candidate_completion}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _extract_verdict(text: str) -> int | None:
    m = re.search(r"<verdict>\s*(correct|incorrect)\s*</verdict>", text, re.IGNORECASE)
    if m:
        return 1 if m.group(1).lower() == "correct" else 0
    # fallback: trailing token
    tokens = re.findall(r"\b(correct|incorrect)\b", text, re.IGNORECASE)
    if tokens:
        return 1 if tokens[-1].lower() == "correct" else 0
    return None


# ---------------------------------------------------------------------------
# Verification reward
# ---------------------------------------------------------------------------


def reward_verification(completions: list[str], **kwargs) -> list[float]:
    """+1 if verdict matches correctness label, -1 if wrong, -0.5 if malformed."""
    correctness = kwargs["correctness"]
    rewards = []
    for completion, c in zip(completions, correctness):
        pred = _extract_verdict(completion)
        if pred is None:
            rewards.append(-0.5)
        else:
            rewards.append(1.0 if pred == int(c) else -1.0)
    return rewards


def reward_verify_format(completions: list[str], **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", completion, re.DOTALL):
            r += 0.1
        if re.search(r"<verdict>(correct|incorrect)</verdict>", completion, re.IGNORECASE):
            r += 0.1
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# On-policy verification dataset construction (paper §3.1)
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_candidates(
    model,
    tokenizer,
    dataset: Dataset,
    n_queries: int,
    group_size: int,
    max_new_tokens: int = 512,
    temperature: float = 0.6,
    top_p: float = 0.95,
    seed: int = 0,
) -> list[dict]:
    """Sample G candidate completions per query; return raw triplets (x, y, c)."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(n_queries, len(dataset)))

    triplets = []
    model.eval()
    device = next(model.parameters()).device

    for idx in indices:
        ex = dataset[idx]
        # Build the generation prompt the same way training does.
        gen_prompt = format_prompt(ex, tokenizer)["prompt"]
        inputs = tokenizer(gen_prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            num_return_sequences=group_size,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        for seq in outputs:
            completion = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
            pred = _extract_answer(completion)
            expected = "true" if ex["label"] else "false"
            if pred is None:
                continue  # malformed, drop
            c = 1 if pred == expected else 0
            triplets.append(
                {
                    "query_idx": idx,
                    "example": ex,
                    "completion": completion,
                    "correctness": c,
                }
            )

    return triplets


def build_verify_dataset(
    triplets: list[dict],
    tokenizer,
    max_per_query: int = 2,
    seed: int = 0,
) -> Dataset:
    """Apply the paper's post-processing: filtering, diversity, balancing."""
    rng = random.Random(seed)

    # --- Filtering: drop queries where all sampled answers share the same label
    #     (no signal for verification) ---
    by_query: dict[int, list[dict]] = defaultdict(list)
    for t in triplets:
        by_query[t["query_idx"]].append(t)
    kept_per_query: dict[int, list[dict]] = {}
    for qid, ts in by_query.items():
        labels = {t["correctness"] for t in ts}
        if len(labels) < 2:
            # All-correct or all-incorrect: paper drops these.
            continue
        kept_per_query[qid] = ts

    # --- Diversity: cap samples per query ---
    pos, neg = [], []
    for qid, ts in kept_per_query.items():
        rng.shuffle(ts)
        per_query_pos = [t for t in ts if t["correctness"] == 1][: max_per_query]
        per_query_neg = [t for t in ts if t["correctness"] == 0][: max_per_query]
        pos.extend(per_query_pos)
        neg.extend(per_query_neg)

    # --- Balancing: equal counts of c=0 and c=1 ---
    n = min(len(pos), len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    selected = pos[:n] + neg[:n]
    rng.shuffle(selected)

    rows = []
    for t in selected:
        prompt = format_verify_prompt(t["example"], t["completion"], tokenizer)
        rows.append({"prompt": prompt, "correctness": t["correctness"]})
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Trainer factories
# ---------------------------------------------------------------------------


def make_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )


def make_grpo_config(output_dir: str, max_steps: int, run_name: str) -> GRPOConfig:
    return GRPOConfig(
        output_dir=output_dir,
        num_generations=8,
        max_completion_length=512,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        weight_decay=0.001,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=max_steps,
        warmup_steps=min(50, max_steps // 10),
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        bf16=True,
        save_strategy="steps",
        save_steps=max(50, max_steps // 4),
        save_total_limit=2,
        logging_steps=10,
        report_to="wandb",
        run_name=run_name,
        use_liger_kernel=True,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def run_verify_init(
    model, tokenizer, raw_train, gen_train, eval_dataset, args, peft_config
):
    """Stage-wise: pure self-verification, then standard generation GRPO."""

    # --- Stage 1: self-verification ---
    print(f"[verify-init] Sampling {args.verify_queries} on-policy queries...")
    triplets = sample_candidates(
        model, tokenizer, raw_train,
        n_queries=args.verify_queries,
        group_size=args.group_size,
    )
    verify_ds = build_verify_dataset(triplets, tokenizer, max_per_query=args.max_per_query)
    print(f"[verify-init] Verification dataset: {len(verify_ds)} balanced samples")

    v_args = make_grpo_config(
        output_dir=str(Path(args.output_dir) / "stage1_verify"),
        max_steps=args.verify_steps,
        run_name=f"{args.run_name}-verify",
    )
    v_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_verification, reward_verify_format],
        args=v_args,
        train_dataset=verify_ds,
        peft_config=peft_config,
    )
    v_trainer.train()
    model = v_trainer.model  # carry adapters/weights forward

    # --- Stage 2: generation GRPO ---
    g_args = make_grpo_config(
        output_dir=str(Path(args.output_dir) / "stage2_generate"),
        max_steps=args.generate_steps,
        run_name=f"{args.run_name}-generate",
    )
    g_args.eval_strategy = "steps"
    g_args.eval_steps = max(100, args.generate_steps // 5)
    g_trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_correctness, reward_format, reward_reasoning_quality],
        args=g_args,
        train_dataset=gen_train,
        eval_dataset=eval_dataset,
        peft_config=None,  # already attached by stage 1
    )
    g_trainer.train()
    return g_trainer


def run_verify_alter(
    model, tokenizer, raw_train, gen_train, eval_dataset, args, peft_config
):
    """Alternate: N steps generation, then a verification phase, repeat."""
    n_cycles = args.total_steps // (args.alter_n + args.verify_steps_per_cycle)
    if n_cycles == 0:
        raise ValueError("total_steps too small for one cycle")

    attached_peft = peft_config  # only used on the very first trainer
    last_trainer = None

    for cycle in range(n_cycles):
        # --- Generation phase ---
        g_args = make_grpo_config(
            output_dir=str(Path(args.output_dir) / f"cycle{cycle}_gen"),
            max_steps=args.alter_n,
            run_name=f"{args.run_name}-c{cycle}-gen",
        )
        g_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_correctness, reward_format, reward_reasoning_quality],
            args=g_args,
            train_dataset=gen_train,
            eval_dataset=eval_dataset,
            peft_config=attached_peft,
        )
        g_trainer.train()
        model = g_trainer.model
        attached_peft = None
        last_trainer = g_trainer

        # --- Verification phase (uses on-policy answers from updated model) ---
        triplets = sample_candidates(
            model, tokenizer, raw_train,
            n_queries=args.verify_queries,
            group_size=args.group_size,
            seed=cycle,
        )
        verify_ds = build_verify_dataset(
            triplets, tokenizer, max_per_query=args.max_per_query, seed=cycle
        )
        if len(verify_ds) == 0:
            print(f"[verify-alter] cycle {cycle}: empty verification set, skipping")
            continue
        print(f"[verify-alter] cycle {cycle}: verification set size {len(verify_ds)}")
        v_args = make_grpo_config(
            output_dir=str(Path(args.output_dir) / f"cycle{cycle}_verify"),
            max_steps=args.verify_steps_per_cycle,
            run_name=f"{args.run_name}-c{cycle}-verify",
        )
        v_trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[reward_verification, reward_verify_format],
            args=v_args,
            train_dataset=verify_ds,
            peft_config=None,
        )
        v_trainer.train()
        model = v_trainer.model
        last_trainer = v_trainer

    return last_trainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="mcl-wic")
    parser.add_argument(
        "--strategy", choices=["verify-init", "verify-alter"], default="verify-init"
    )
    parser.add_argument("--output-dir", type=str, default="./qwen-wic-self-verify")
    parser.add_argument("--run-name", type=str, default="qwen-wic-self-verify")
    # verify-init schedule (paper: 400 / 600)
    parser.add_argument("--verify-steps", type=int, default=400)
    parser.add_argument("--generate-steps", type=int, default=600)
    # verify-alter schedule
    parser.add_argument("--total-steps", type=int, default=1000)
    parser.add_argument("--alter-n", type=int, default=100,
                        help="generation steps between verification phases")
    parser.add_argument("--verify-steps-per-cycle", type=int, default=50)
    # on-policy sampling
    parser.add_argument("--verify-queries", type=int, default=512)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--max-per-query", type=int, default=2)
    args = parser.parse_args()

    dataset = DatasetDict(
        {split: load_data(args.dataset, split=split) for split in ("train", "dev")}
    )
    eval_dataset = dataset["dev"].shuffle(seed=42).select(range(200))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Generation dataset is mapped through format_prompt; verification dataset is
    # built lazily from the *raw* generation dataset (we need lemma/word/sentence
    # fields to build verify prompts), so keep a raw copy.
    raw_train = dataset["train"]
    partial_format = partial(format_prompt, tokenizer=tokenizer)
    gen_train = raw_train.map(
        partial_format,
        remove_columns=["lemma", "word1", "word2", "pos", "sentence1", "sentence2"],
    )
    gen_eval = eval_dataset.map(
        partial_format,
        remove_columns=["lemma", "word1", "word2", "pos", "sentence1", "sentence2"],
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora, is_trainable=True)
        print(f"Loaded LoRA adapter from: {args.lora}")
    peft_config = None if args.lora else make_lora_config()

    # NOTE: on-policy sampling needs the raw fields, so pass raw_train (not gen_train).
    if args.strategy == "verify-init":
        run_verify_init(
            model, tokenizer, raw_train, gen_train, gen_eval, args, peft_config
        )
    else:
        run_verify_alter(
            model, tokenizer, raw_train, gen_train, gen_eval, args, peft_config
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
