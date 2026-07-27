import argparse
import json
import os
from functools import partial, wraps
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment

import sense_data as sd
from grpo_lora import load_policy
from sense_rewards import KEEP_COLS, REWARDS

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


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
    out = {"prompt": sd.wic_messages(rec, with_target=False)}
    if with_feedback:
        out["privileged_context"] = gold_feedback(rec)
    return out


def as_text_reward(fn):
    @wraps(fn)
    def wrapper(completions, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return fn(texts, **kwargs)

    return wrapper


def build_dataset(split, cap=None, with_feedback=True):
    recs = sd.load_mclwic(split)
    ds = Dataset.from_list(recs)
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(
        partial(format_prompt, with_feedback=with_feedback), remove_columns=drop
    )


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
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.3)
    ap.add_argument("--distill-threshold", type=float, default=0.5)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.0)
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
    ap.add_argument("--teacher-kind", default="base", choices=["base", "live", "ema"])
    ap.add_argument(
        "--teacher-update-rate", type=float, default=0.05, help="EMA teacher rate."
    )
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
    ap.add_argument("--resume", nargs="?", const=True, default=None)
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

    with_feedback = not args.no_gold_feedback
    train_ds = build_dataset("train", with_feedback=with_feedback)
    dev_ds = build_dataset("dev", cap=200, with_feedback=with_feedback)
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

    run_name = "qwen-lora-sdpo-wic"
    output_dir = f"./{run_name}"
    training_args = SDPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=1,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
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
        environment_feedback_only_without_solution=True,
        reprompt_template=REPROMPT_TEMPLATE,
        solution_template=SOLUTION_TEMPLATE,
        feedback_template=FEEDBACK_TEMPLATE,
        max_reprompt_len=prompt_headroom + 2 * args.max_completion_length,
        **vllm_kwargs,
    )

    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_funcs.append(get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.2))

    trainer = SDPOTrainer(
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
