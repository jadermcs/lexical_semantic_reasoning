"""SDPO (self-distillation policy optimization) on the WiC task, full fine-tune.

Same rollout set, rewards, prompts and privileged context as ``sdpo_lora.py`` — see
that module's docstring for what SDPO does and why the gold verdict is a sound
teacher hint. Only the optimized parameter set changes: every weight, no adapter.

Dropping LoRA changes two things that matter on one GPU:

**The teacher stops being free.** ``SDPOTrainer._setup_teacher_model`` resolves
``teacher_model_kind`` against the PEFT state: under LoRA, ``base`` is the student
with the adapter disabled and ``ema`` is a second small adapter, both sharing one
set of weights. With no adapter to toggle, both instead load a *frozen second copy
of the whole model* (plus a per-step EMA sync). So this script defaults to
``--teacher-kind live``, which reuses the student in every configuration. That
costs nothing in signal here: the teacher is better because it is *hinted*
(``gold_feedback``), not because its weights differ.

**Precision starts to matter.** LoRA tolerates pure-bf16 weights because the
adapter is small and takes a large effective step; a full fine-tune at ~1e-6 does
not — a bf16 weight has ~8 mantissa bits, so an update that small against a ~1e-2
weight rounds away to nothing. The model is therefore instantiated by TRL from the
path in float32 with ``bf16=True`` autocast on top, i.e. fp32 master weights and
bf16 math. That is the reason ``--model`` is handed to the trainer as a string
rather than pre-loaded: ``model_init_kwargs`` then applies uniformly to the
student and to any reference/teacher copy TRL builds.

Usage mirrors sdpo_lora.py, but points at a full SFT checkpoint (``sft_sense.py``
output) rather than a merged adapter::

    uv run python src/sdpo_sense.py --model ./qwen-sense-sft-sft_wic_filtered \
        --exclude-pairs data/sft_wic_filtered.sft_pairs.json --balance-labels
"""

import argparse
from pathlib import Path

import torch
torch._dynamo.config.disable = True

from transformers import AutoTokenizer
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.rewards import get_repetition_penalty_reward

from grpo_lora import load_exclude_pairs
from sdpo_lora import (
    FEEDBACK_TEMPLATE,
    REPROMPT_TEMPLATE,
    SOLUTION_TEMPLATE,
    as_text_reward,
    build_dataset,
)
from sense_rewards import REWARDS, make_trace_saver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Plain HF model to fully fine-tune — normally the SFT warm start from "
        "`sft_sense.py` (./qwen-sense-sft-<data-stem>).",
    )
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument(
        "--vllm-gpu-mem",
        type=float,
        default=0.22,
        help="Fraction of the GPU handed to the colocated vLLM engine (weights + KV "
        "cache). Ignored in server mode. A full fine-tune needs far more trainer-side "
        "memory than LoRA (fp32 weights + grads + optimizer ≈ 5x the weights), so vLLM "
        "gets a correspondingly smaller slice; 0.22 fits a 0.6B policy on 16GB.",
    )
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
        help="Max generated tokens per rollout. 640 clears the p99.7 of distilled WiC "
        "traces (~520 tokens); truncated rollouts are masked out "
        "(mask_truncated_completions), so a tight cap costs wasted rollout compute, "
        "not wrong gradients.",
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
    ap.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient anchoring the policy to the SFT reference. Unlike "
        "grpo_sense.py this defaults to 0: with no adapter to disable, beta>0 loads a "
        "second full copy of the model as the reference. Raise it (and budget the "
        "VRAM) if the policy drifts off the warm start.",
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
        default="live",
        choices=["base", "live", "ema"],
        help="Which model plays teacher under the privileged context. Without LoRA, "
        "'base' and 'ema' each load a full frozen second copy of the model (and 'ema' "
        "syncs it every step); 'live' reuses the student and is the default here.",
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

    if (Path(args.model) / "adapter_config.json").is_file():
        raise SystemExit(
            f"{args.model} is a LoRA adapter dir. Full fine-tuning starts from plain "
            f"weights — use sft_sense.py's output, or sdpo_lora.py for the adapter path."
        )
    if args.teacher_kind != "live":
        print(
            f"--teacher-kind {args.teacher_kind} without LoRA loads a second full copy of "
            f"the model; budget ~2x the weights in VRAM."
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
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_mem,
            vllm_max_model_length=prompt_headroom + args.max_completion_length,
        )

    run_name = "qwen-sense-sdpo-wic"
    output_dir = f"./{run_name}"
    training_args = SDPOConfig(
        output_dir=output_dir,
        num_generations=8,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        optim="paged_adamw_8bit",
        beta=args.beta,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_train_epochs=2,
        warmup_steps=50,
        learning_rate=1e-6,
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
        model=args.model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved final model → {output_dir}")


if __name__ == "__main__":
    main()
