import argparse
import json
import os
from functools import partial, wraps
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment

import semcor_pairs
import sense_data as sd
from sense_rewards import GLOSS_COLS, KEEP_COLS, REWARDS

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def format_prompt(rec):
    """Render the prompt column, plus the gold-gloss columns reward_wic_gloss reads.

    The gloss fields are always written, empty on MCL-WiC rows: Arrow needs one schema
    across the concatenated sources, and an empty gold gloss is how that reward is told
    this pair is not gloss-scoreable.
    """
    out = {"prompt": sd.wic_messages(rec, with_target=False)}
    for col, default in GLOSS_COLS.items():
        out[col] = rec.get(col) or default
    return out


def as_text_reward(fn):
    @wraps(fn)
    def wrapper(completions, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return fn(texts, **kwargs)

    return wrapper


def load_policy(path, lora_kwargs):
    """Load the warm-started weights and attach a fresh LoRA.

    ``--model`` is the *merged* SFT model (``sft_lora.py --merged-dir``), i.e. a
    plain HF checkpoint with the SFT adapter already folded in. GRPO trains a new
    adapter on top of it, so the frozen merged weights are exactly the right KL
    reference — TRL gets it by disabling the adapter, no second model loaded.
    """
    if (Path(path) / "adapter_config.json").is_file():
        raise SystemExit(
            f"{path} is a LoRA adapter dir, not a merged model. Run sft_lora.py so it "
            f"writes '<output-dir>-merged' and point --model at that."
        )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="kernels-community/flash-attn2",
    )
    return model, LoraConfig(
        task_type="CAUSAL_LM", target_modules="all-linear", **lora_kwargs
    )


def _prepare(recs, cap=None):
    ds = Dataset.from_list(recs)
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(partial(format_prompt), remove_columns=drop)


def build_dataset(split, cap=None, semcor_path=None):
    """Rollout set for one split, optionally mixing in the saved SemCor pairs.

    ``semcor_path`` points at the file ``semcor_pairs.py`` writes. Those pairs carry a
    gold gloss per usage, which is the only thing ``reward_wic_gloss`` can score; MCL-WiC
    contributes the label and nothing else. Mixing rather than replacing keeps the
    rollout distribution anchored on the benchmark the run is evaluated against.
    """
    parts = [_prepare(sd.load_mclwic(split), cap=cap)]
    if semcor_path:
        sc = semcor_pairs.load_pairs(semcor_path)
        print(f"[{split}] semcor: +{len(sc)} gloss-annotated pairs")
        parts.append(_prepare(sc, cap=cap))
    if len(parts) == 1:
        return parts[0]
    # Interleave the sources: each prompt forms its own rollout group, but leaving them
    # in blocks would make every optimizer step see one source only.
    return concatenate_datasets(parts).shuffle(seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Backward micro-batch, in completions. This is the knob that OOMs, and "
        "the ONLY thing it controls is peak activation memory — see --generation-batch "
        "for the one that sets how many prompts each gradient averages over.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Rollouts per prompt (the GRPO group size the advantage is centred on).",
    )
    ap.add_argument(
        "--generation-batch",
        type=int,
        default=256,
        help="Completions per optimizer step; grad-accum is derived from it as "
        "(this // --batch-size). Prompt groups per step = this // --num-generations, "
        "and that is what the gradient averages over. Deriving grad-accum this way "
        "keeps the group count fixed when you trade micro-batch size for memory — the "
        "previous 'gradient_accumulation_steps=32//batch_size' pinned the generation "
        "batch at 32, so --batch-size bought OOM and left the group count at 2.",
    )
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.4)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument(
        "--semcor-pairs",
        nargs="?",
        const=str(semcor_pairs.DEFAULT_PAIRS),
        default=None,
        help="Mix the saved SemCor pair set into the training rollouts (bare flag uses "
        f"{semcor_pairs.DEFAULT_PAIRS}). These are the only pairs carrying a gold gloss "
        "per usage, so they are the only ones reward_wic_gloss can score; build the file "
        "with 'uv run python src/semcor_pairs.py'. The dev set stays pure MCL-WiC so the "
        "eval curve remains comparable across runs.",
    )
    ap.add_argument(
        "--gloss-reward-weight",
        type=float,
        default=1.0,
        help="Weight on reward_wic_gloss. At 1.0 the term spans ±0.15, which is what "
        "the shape/accuracy invariant in tests/test_sense_rewards.py was checked "
        "against; raising it eats that headroom.",
    )
    # --- Dr. GRPO / DAPO -------------------------------------------------- #
    # The two papers disagree only on how token losses are aggregated, so that axis
    # is a flag; every other term below is shared and on by default.
    ap.add_argument(
        "--loss-type",
        default="dr_grpo",
        choices=["dr_grpo", "dapo", "grpo", "bnpo"],
        help="Token-loss aggregation. 'dr_grpo' normalises by the constant "
        "--max-completion-length (Dr. GRPO); 'dapo' normalises by the active-token "
        "count of the whole accumulated batch. Both remove the length bias that "
        "makes plain 'grpo' prefer short positive-advantage completions; they are "
        "mutually exclusive, hence the choice.",
    )
    ap.add_argument(
        "--scale-rewards",
        default="none",
        choices=["none", "group", "batch"],
        help="Dr. GRPO's second correction: dividing the advantage by the group std "
        "('group', TRL's default) up-weights prompts the policy already answers "
        "consistently, which is a question-level difficulty bias. 'none' keeps the "
        "mean-centred advantage only.",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Lower PPO clip bound.",
    )
    ap.add_argument(
        "--epsilon-high",
        type=float,
        default=0.28,
        help="Upper PPO clip bound (DAPO 'clip-higher'). Decoupling it from "
        "--epsilon gives low-probability tokens room to grow, which is what keeps "
        "the policy from collapsing to a single sampled mode; 0.28 is the paper's "
        "value. Pass the same value as --epsilon for symmetric clipping.",
    )
    ap.add_argument(
        "--soft-punish-cache",
        type=int,
        default=128,
        help="DAPO overlong reward shaping (their Eq. 13): completions in the last "
        "N tokens before --max-completion-length take a linearly ramped penalty "
        "instead of being truncated and silently discarded. Set 0 to fall back to "
        "overlong *filtering* (mask_truncated_completions) instead — the two are "
        "alternatives in the paper, not a stack.",
    )
    ap.add_argument(
        "--overlong-penalty",
        type=float,
        default=0.2,
        help="Weight on the overlong term. The raw reward bottoms out at -1.0, which "
        "would blow the shape/accuracy invariant in sense_rewards (0.4 of headroom "
        "against a 1.5 accuracy gap, already spent down to 0.2 by the repetition "
        "penalty), so it is scaled to match the repetition penalty's magnitude.",
    )
    ap.add_argument("--resume", nargs="?", const=True, default=None)
    args = ap.parse_args()

    # Fail loudly at launch rather than silently training on 2 prompt groups.
    if args.generation_batch % args.batch_size:
        ap.error(
            f"--generation-batch ({args.generation_batch}) must be divisible by --batch-size ({args.batch_size})"
        )
    if args.generation_batch % args.num_generations:
        # SDPOConfig enforces this too (grpo_config.py:1090); catching it here reports
        # the prompt-group count that actually motivates the constraint.
        ap.error(
            f"--generation-batch ({args.generation_batch}) must be divisible by "
            f"--num-generations ({args.num_generations})"
        )
    grad_accum = args.generation_batch // args.batch_size
    prompt_groups = args.generation_batch // args.num_generations
    print(
        f"batching: micro={args.batch_size} x grad_accum={grad_accum} = "
        f"{args.generation_batch} completions/step / {args.num_generations} rollouts "
        f"= {prompt_groups} prompt groups averaged per optimizer step"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset("train", semcor_path=args.semcor_pairs)
    dev_ds = build_dataset("dev", cap=200)
    print(train_ds[0])

    model, peft_config = load_policy(
        args.model,
        dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
    )

    prompt_headroom = 512
    if args.vllm_server_host:
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
        )
    else:
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_mem,
            vllm_max_model_length=prompt_headroom + args.max_completion_length,
        )

    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_weights = [
        args.gloss_reward_weight if f.__name__ == "reward_wic_gloss" else 1.0
        for f in REWARDS
    ]
    reward_funcs.append(get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.2))
    reward_weights.append(1.0)
    if args.soft_punish_cache > 0:
        # Not wrapped in as_text_reward: this one scores completion_ids, not text.
        reward_funcs.append(
            get_soft_overlong_punishment(
                max_completion_len=args.max_completion_length,
                soft_punish_cache=args.soft_punish_cache,
            )
        )
        reward_weights.append(args.overlong_penalty)
    # Overlong shaping and overlong filtering are the two alternatives DAPO offers
    # for truncated rollouts; masking on top of the shaping would zero the loss on
    # exactly the completions the penalty is meant to teach from.
    mask_truncated = args.soft_punish_cache == 0
    print(
        f"objective: loss_type={args.loss_type} scale_rewards={args.scale_rewards} "
        f"clip=[{args.epsilon}, {args.epsilon_high}] beta={args.beta} "
        + (
            f"overlong=shape(cache={args.soft_punish_cache}, w={args.overlong_penalty})"
            if not mask_truncated
            else "overlong=mask"
        )
    )

    run_name = "qwen-lora-grpo-wic"
    output_dir = f"./{run_name}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=mask_truncated,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        reward_weights=reward_weights,
        beta=args.beta,
        disable_dropout=True,
        optim="paged_adamw_8bit",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=2,
        warmup_steps=0.03,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        logging_steps=25,
        report_to="wandb",
        run_name=run_name,
        log_completions=True,
        num_completions_to_print=3,
        **vllm_kwargs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        peft_config=peft_config,
    )

    if args.resume:
        ckpt = (
            args.resume
            if isinstance(args.resume, str)
            else get_last_checkpoint(output_dir)
        )
        if ckpt is None:
            ap.error(f"--resume given but no checkpoint-* found in {output_dir}")
        state = json.loads((Path(ckpt) / "trainer_state.json").read_text())
        print(
            f"Resuming from {ckpt} at step {state['global_step']}/{state.get('max_steps')}"
        )
    else:
        ckpt = None

    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(output_dir)
    print(f"Saved final adapter → {output_dir}")


if __name__ == "__main__":
    main()
