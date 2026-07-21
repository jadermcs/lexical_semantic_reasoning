"""SDPO (self-distillation policy optimization) on the WiC task, LoRA edition.

Same rollout set, rewards and LoRA warm start as ``grpo_lora.py`` — the trainer is
what changes. GRPO learns only from *reward variance inside a group*: when all 8
rollouts for a pair are wrong (or all right), the group advantage is zero and the
step teaches nothing. On the teacher-failed complement fed to RL (see
``--exclude-pairs``) that is a large slice of the batch, since those pairs are hard
by construction.

SDPO adds a second signal for exactly those groups. It re-prompts the *same* model
with privileged context — a sibling rollout that scored well, and/or textual
feedback we supply — and distills that feedback-conditioned next-token distribution
back into the policy on the *original* prompt. The teacher is the model itself, so
there is no second model to train, and the student never sees the privileged
context at inference.

The privileged context we supply here is the gold same/different verdict
(``--gold-feedback``, on by default). That is sound: it conditions the *teacher*
only, and the student is optimized on the unhinted prompt, so it is distillation of
a hint-informed distribution, not label leakage into the student's inputs.

Usage mirrors grpo_lora.py::

    uv run python src/sdpo_lora.py --model ./qwen-lora-<stem>-merged \
        --exclude-pairs data/sft_wic.sft_pairs.json --balance-labels
"""

import argparse
from functools import partial, wraps
from pathlib import Path

import torch
torch._dynamo.config.disable = True

from datasets import Dataset
from transformers import AutoTokenizer
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.rewards import get_repetition_penalty_reward

import sense_data as sd
from grpo_lora import balance_labels, load_exclude_pairs, load_policy
from sense_rewards import KEEP_COLS, REWARDS, make_trace_saver


# The teacher reprompt: SDPO rebuilds the last user turn as
# ``reprompt_template.format(prompt=<original user text>, solution=..., feedback=...)``
# and keeps the system turn, so the answer contract in ``sd.WIC_SYSTEM`` still holds
# and the teacher's tokens stay comparable to the student's.
REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n"
    "Now answer the original question, reasoning in <think> tags and ending with the "
    "required JSON object.\n"
)
SOLUTION_TEMPLATE = (
    "\nHere is a correct answer to this pair from an earlier attempt:\n\n"
    "{successful_previous_attempt}\n\n"
)
FEEDBACK_TEMPLATE = "\n{feedback_raw}\n"


def gold_feedback(rec):
    """The privileged context: the gold verdict, stated as a hint to the teacher.

    Only ever conditions the reprompted teacher — never the student's own prompt.
    Without it, a group where every rollout is wrong has no successful sibling to
    distill from and SDPO degenerates to GRPO on that group; with it, the hardest
    pairs are exactly the ones that still produce a learning signal.
    """
    same = bool(rec["label"])
    verdict = "the same sense" if same else "different senses"
    return (
        f'Hint: the two sentences use "{rec["lemma"]}" in {verdict}, so the correct '
        f'verdict is "same_sense": {"true" if same else "false"}. Do not mention this '
        "hint; reason to it from the sentences themselves."
    )


def format_prompt(rec, with_feedback=True):
    """Render the prompt column; gold columns are kept so reward fns see them as kwargs.

    Unlike ``grpo_lora.format_prompt`` this leaves the prompt *conversational* (a
    message list) instead of pre-rendering the chat template. SDPO needs the turn
    structure: it reconstructs the teacher prompt as ``system + rewritten user``,
    and a pre-rendered string would splice the demonstration inside the assistant
    turn instead.
    """
    out = {"prompt": sd.wic_messages(rec, with_target=False)}
    if with_feedback:
        out["privileged_context"] = gold_feedback(rec)
    return out


def as_text_reward(fn):
    """Adapt a ``sense_rewards`` fn (plain-string completions) to conversational ones.

    With a conversational prompt TRL hands reward fns
    ``[{"role": "assistant", "content": ...}]`` per completion; every reward in
    ``sense_rewards`` parses raw text. Flatten instead of duplicating the rewards.
    """
    @wraps(fn)
    def wrapper(completions, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return fn(texts, **kwargs)

    return wrapper


def build_dataset(split, cap=None, exclude_pairs=None, balance=False, with_feedback=True):
    recs = sd.load_mclwic(split, exclude_pairs=exclude_pairs)
    if balance:
        before = len(recs)
        recs = balance_labels(recs)
        print(f"[{split}] balanced labels: {before} → {len(recs)} pairs (50/50)")
    ds = Dataset.from_list(recs)
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(partial(format_prompt, with_feedback=with_feedback), remove_columns=drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Plain HF model to attach a fresh SDPO LoRA to — normally the merged "
        "SFT warm start written by `sft_lora.py --merged-dir` "
        "(./qwen-lora-<data-stem>-merged).",
    )
    ap.add_argument("--lora-r", type=int, default=32, help="LoRA rank.")
    ap.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha scaling.")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument(
        "--distill-out",
        default=None,
        help="If set, append correct completions to '<path>.rank<N>.jsonl' for "
        "offline self-distillation (the SFT loop). Does not affect training.",
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
        "distilled into the SFT set; those pairs are held out of the train rollout "
        "set so RL only sees pairs the warm-start never saw.",
    )
    ap.add_argument(
        "--balance-labels",
        action="store_true",
        help="Down-sample the majority class so the train rollout set is 50/50 "
        "same/different. Strongly recommended with --exclude-pairs.",
    )
    # ---- SDPO-specific knobs ----
    ap.add_argument(
        "--distillation-weight",
        type=float,
        default=0.5,
        help="Convex blend: loss = (1-w)*policy_grad + w*self_distillation. 1.0 is "
        "pure SDPO, 0.0 collapses to GRPO. 0.5 keeps the verifiable reward driving "
        "the update while the teacher densifies the zero-variance groups.",
    )
    ap.add_argument(
        "--teacher-kind",
        default="ema",
        choices=["base", "live", "ema"],
        help="Which model plays teacher under the privileged context: the frozen "
        "warm start ('base'), the current policy ('live'), or an EMA of it ('ema').",
    )
    ap.add_argument("--teacher-update-rate", type=float, default=0.05, help="EMA teacher rate.")
    ap.add_argument(
        "--success-reward-threshold",
        type=float,
        default=1.0,
        help="Minimum total reward for a rollout to be reused as a demonstration. "
        "Our shaping terms sit on top of the ±1 accuracy term, so 1.0 means "
        "'correct verdict and reasonably well formed'.",
    )
    ap.add_argument(
        "--no-gold-feedback",
        action="store_true",
        help="Do not supply the gold verdict as privileged context; teach only from "
        "successful sibling rollouts. Groups where every rollout fails then carry no "
        "distillation signal (the GRPO failure mode SDPO is here to fix).",
    )
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    with_feedback = not args.no_gold_feedback
    exclude = load_exclude_pairs(args.exclude_pairs)
    train_ds = build_dataset(
        "train", exclude_pairs=exclude, balance=args.balance_labels, with_feedback=with_feedback
    )
    dev_ds = build_dataset("dev", cap=200, exclude_pairs=exclude, with_feedback=with_feedback)
    if exclude:
        print(f"Excluding {len(exclude)} SFT-consumed pairs from the SDPO train set → {len(train_ds)} rollout pairs")
    print(train_ds[0])

    model, peft_config = load_policy(
        args.model,
        dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
    )

    prompt_headroom = 512
    if args.vllm_server_host:
        vllm_kwargs = dict(
            use_vllm=True, vllm_mode="server",
            vllm_server_host=args.vllm_server_host, vllm_server_port=args.vllm_server_port,
        )
    else:
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_max_model_length=prompt_headroom + args.max_completion_length,
        )

    run_name = "qwen-lora-sdpo-wic"
    output_dir = f"./{run_name}"
    training_args = SDPOConfig(
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
        # --- self-distillation ---
        # sampled_token is the paper's mode and pins distillation_alpha=1.0 (reverse
        # KL); use_liger_kernel is deliberately off, since the fused JSD path only
        # supports full_logits with distillation_weight=1.0.
        distillation_mode="sampled_token",
        distillation_alpha=1.0,
        distillation_weight=args.distillation_weight,
        teacher_model_kind=args.teacher_kind,
        teacher_update_rate=args.teacher_update_rate,
        use_successful_as_teacher=True,
        success_reward_threshold=args.success_reward_threshold,
        dont_reprompt_on_self_success=True,
        include_environment_feedback=with_feedback,
        reprompt_template=REPROMPT_TEMPLATE,
        solution_template=SOLUTION_TEMPLATE,
        feedback_template=FEEDBACK_TEMPLATE,
        # Teacher prompt = original prompt + one demonstration completion + the hint,
        # so it needs roughly twice the student's budget.
        max_reprompt_len=prompt_headroom + 2 * args.max_completion_length,
        **vllm_kwargs,
    )

    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_funcs.append(get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.2))
    if args.distill_out:
        reward_funcs.append(as_text_reward(make_trace_saver(args.distill_out, args.distill_threshold)))
        print(
            f"Offline self-distillation: saving completions scoring >= "
            f"{args.distill_threshold} to {args.distill_out}.rank*.jsonl"
        )

    trainer = SDPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
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


if __name__ == "__main__":
    main()
