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
import os
from functools import partial, wraps
from pathlib import Path

# Rollout lengths vary batch to batch, so the caching allocator fragments; the OOMs on
# a 16GB card were failing with 1.5-1.9GB reserved-but-unallocated.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
torch._dynamo.config.disable = True

from datasets import Dataset
from transformers import AutoTokenizer
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment

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
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Backward micro-batch, in completions. This is the knob that OOMs, and "
        "the ONLY thing it controls is peak activation memory — see --generation-batch "
        "for the one that sets how many prompts each gradient averages over.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Rollouts per prompt (the GRPO group size the advantage is centred on).",
    )
    ap.add_argument(
        "--generation-batch",
        type=int,
        default=64,
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
    ap.add_argument(
        "--vllm-gpu-mem",
        type=float,
        default=0.3,
        help="Colocate-mode share of the card reserved for the vLLM KV cache. 0.3 is "
        "sized for the observed rollouts (~240 tokens mean, 291 max); the old 0.55 "
        "handed vLLM half a 16GB card and starved the training step.",
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
        default=512,
        help="Rollout cap. Measured max_terminated_length never passed 291 and the "
        "clipped ratio sat at ~0.2%%, so 768 was paying KV cache for headroom nothing "
        "used. Do not drop below 512 without also cutting --soft-punish-cache: the "
        "penalty ramp starts at (cap - soft_punish_cache), and a 384 cap would put it "
        "at 256, below the ~240-token mean.",
    )
    ap.add_argument(
        "--soft-punish-cache",
        type=int,
        default=128,
        help="Width of the DAPO soft-overlong band below the cap: completions longer "
        "than (max_completion_length - this) are penalised on a 0 → -1 ramp. Truncated "
        "rollouts are dropped from the loss entirely (mask_truncated_completions=True), "
        "so this band is the only length pressure left — it grades the *approach* to the "
        "cap on completions that do terminate, giving the policy a slope to descend "
        "before it falls off the edge. 0 disables it.",
    )
    ap.add_argument(
        "--beta",
        type=float,
        default=0.0,
        help="KL coefficient toward the warm start. INERT in SDPOTrainer: it assigns "
        "self.beta (trl/experimental/sdpo/sdpo_trainer.py:606) and never reads it — "
        "_compute_policy_loss has no KL term, no reference logprobs are gathered, and "
        "no reference model is built, which is why no 'kl' metric is ever logged. "
        "Defaulted to 0.0 so the config stops implying an anchor that is not applied; "
        "note this also means runs here are NOT comparable to a stock GRPOTrainer run "
        "with beta>0, where it does bite.",
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
        default=0.3,
        help="Convex blend: loss = (1-w)*policy_grad + w*self_distillation. 1.0 is "
        "pure SDPO, 0.0 collapses to GRPO. Keep it a minority of the loss so the "
        "verifiable reward drives the update and the teacher only densifies the "
        "zero-variance groups: at 0.55 distillation outvoted the reward, and since the "
        "distillation term is masked by completion_mask alone (no reward gating — see "
        "sdpo_trainer._compute_self_distillation_loss) it kept training on runaway "
        "rollouts that the reward had already condemned.",
    )
    ap.add_argument(
        "--teacher-kind",
        default="base",
        choices=["base", "live", "ema"],
        help="Which model plays teacher under the privileged context: the frozen "
        "warm start ('base'), the current policy ('live'), or an EMA of it ('ema'). "
        "'base' is the only one that is an *anchor*: 'live' and 'ema' make the "
        "distillation target track the policy, so there is no fixed point and any "
        "drift certifies itself as the new target. An 'ema' run at rate 0.05 was stable "
        "for ~700 steps and then collapsed into runaway generation in under 200.",
    )
    ap.add_argument("--teacher-update-rate", type=float, default=0.05, help="EMA teacher rate.")
    ap.add_argument(
        "--success-reward-threshold",
        type=float,
        default=1.0,
        help="Minimum total reward for a rollout to be reused as a demonstration. "
        "The shaping terms sit on top of the accuracy term (+1.0 correct / -0.5 wrong "
        "/ -1.0 no verdict), whose ceiling puts a perfectly-formed wrong answer at "
        "0.0, so 1.0 means 'correct verdict and reasonably well formed' and no "
        "incorrect rollout can qualify.",
    )
    ap.add_argument(
        "--no-gold-feedback",
        action="store_true",
        help="Do not supply the gold verdict as privileged context; teach only from "
        "successful sibling rollouts. Groups where every rollout fails then carry no "
        "distillation signal (the GRPO failure mode SDPO is here to fix).",
    )
    args = ap.parse_args()

    # Fail loudly at launch rather than silently training on 2 prompt groups.
    if args.generation_batch % args.batch_size:
        ap.error(f"--generation-batch ({args.generation_batch}) must be divisible by --batch-size ({args.batch_size})")
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
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_mem,
            vllm_max_model_length=prompt_headroom + args.max_completion_length,
        )

    run_name = "qwen-lora-sdpo-wic"
    output_dir = f"./{run_name}"
    training_args = SDPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        beta=args.beta,
        optim="paged_adamw_8bit",
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=2,
        # transformers reads a fractional warmup_steps as a *ratio*
        # (TrainingArguments.get_warmup_steps: `int(x) if x >= 1 else ceil(n * x)`),
        # so this is 3% of the run, not 0.03 steps. The previous 0.2/0.3 meant 20-30%
        # of an RL run crawled at near-zero LR before cosine started decaying it.
        warmup_steps=0.03,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        # ~8x fewer optimizer steps now that each averages 8 prompt groups instead of
        # 2, so eval/save cadence scales down with it to keep the dev curve readable.
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        logging_steps=25,
        report_to="wandb",
        run_name=run_name,
        scale_rewards="none",
        distillation_mode="sampled_token",
        distillation_alpha=1.0,
        distillation_weight=args.distillation_weight,
        teacher_model_kind=args.teacher_kind,
        teacher_update_rate=args.teacher_update_rate,
        use_successful_as_teacher=True,
        success_reward_threshold=args.success_reward_threshold,
        dont_reprompt_on_self_success=True,
        include_environment_feedback=with_feedback,
        # Restrict the gold hint to groups with no successful sibling — the case SDPO
        # exists for. Without this the hint is always available, so the reprompt mask
        # in SuccessfulRolloutTeacherContextBuilder.build is 1.0 for *every* sample
        # (dont_reprompt_on_self_success only bars a sample from being its own demo,
        # it does not gate reprompting), and the distillation term fires uniformly
        # including on the ~90% of groups that already had plenty of reward variance.
        environment_feedback_only_without_solution=True,
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
    if args.soft_punish_cache > 0:
        # Not wrapped in as_text_reward: this one scores completion_ids, not text.
        reward_funcs.append(
            get_soft_overlong_punishment(
                max_completion_len=args.max_completion_length,
                soft_punish_cache=args.soft_punish_cache,
            )
        )
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

    trainer.train()
    trainer.save_model(output_dir)
    print(f"Saved final adapter → {output_dir}")


if __name__ == "__main__":
    main()
