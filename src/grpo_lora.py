import argparse
import json
import os
from functools import wraps
from pathlib import Path

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment

from sense_rewards import REWARDS
from utils import build_grpo_dataset

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def as_text_reward(fn):
    @wraps(fn)
    def wrapper(completions, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return fn(texts, **kwargs)

    return wrapper


def load_policy(path, lora_kwargs):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--file-path", default="data/xl-lexeme.json")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--generation-batch",
        type=int,
        default=256,
    )
    ap.add_argument(
        "--num-generations-eval",
        type=int,
        default=1,
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
        "--loss-type",
        default="dr_grpo",
        choices=["dr_grpo", "dapo", "grpo", "bnpo"],
    )
    ap.add_argument(
        "--scale-rewards",
        default="none",
        choices=["none", "group", "batch"],
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
        help="Upper PPO clip bound (DAPO 'clip-higher').",
    )
    ap.add_argument(
        "--soft-punish-cache",
        type=int,
        default=128,
    )
    ap.add_argument(
        "--overlong-penalty",
        type=float,
        default=0.2,
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Rollout temperature. 0.6/top-k 20 is Qwen3's *inference* recipe; it "
        "caps rollout diversity and manufactures unanimous groups (zero advantage). "
        "1.0 with --top-k 0 is the usual RL setting.",
    )
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    ap.add_argument(
        "--vllm-is-mode",
        default="sequence_mask",
        choices=["sequence_mask", "sequence_truncate", "token_mask", "token_truncate"],
        help="How the vLLM/trainer logprob mismatch is corrected. The sequence_* "
        "modes exponentiate a *sum* over tokens, so the correction decays "
        "exponentially in completion length; the token_* modes do not.",
    )
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-steps", type=float, default=200)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-name", default="qwen-lora-grpo-wic")
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument(
        "--save-strategy", default="steps", choices=["steps", "epoch", "no"]
    )
    ap.add_argument("--resume", nargs="?", const=True, default=None)
    ap.add_argument(
        "--experiment",
        default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "wic-grpo"),
        help="MLflow experiment to group runs under. MLflowCallback reads this "
        "from MLFLOW_EXPERIMENT_NAME only -- TrainingArguments.project is "
        "Trackio-only and is ignored by the mlflow integration.",
    )
    args = ap.parse_args()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = args.experiment

    if args.generation_batch % args.batch_size:
        ap.error(
            f"--generation-batch ({args.generation_batch}) must be divisible by --batch-size ({args.batch_size})"
        )
    if args.generation_batch % args.num_generations:
        ap.error(
            f"--generation-batch ({args.generation_batch}) must be divisible by "
            f"--num-generations ({args.num_generations})"
        )
    if args.batch_size % args.num_generations_eval:
        ap.error(
            f"--batch-size ({args.batch_size}) is the eval batch and must be divisible "
            f"by --num-generations-eval ({args.num_generations_eval})"
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

    ds = build_grpo_dataset(args.file_path)
    train_ds = ds["train"]
    dev_ds = ds["test"]
    print(train_ds[0])

    model, peft_config = load_policy(
        args.model,
        {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
    )

    prompt_headroom = 512
    if args.vllm_server_host:
        vllm_kwargs = {
            "use_vllm": True,
            "vllm_mode": "server",
            "vllm_server_host": args.vllm_server_host,
            "vllm_server_port": args.vllm_server_port,
        }
    else:
        vllm_kwargs = {
            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": args.vllm_gpu_mem,
            "vllm_max_model_length": prompt_headroom + args.max_completion_length,
        }

    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_weights = [1.0 for f in REWARDS]
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
    mask_truncated = args.soft_punish_cache == 0
    print(
        f"objective: loss_type={args.loss_type} scale_rewards={args.scale_rewards} "
        f"clip=[{args.epsilon}, {args.epsilon_high}] beta={args.beta} "
        f"sampling=T{args.temperature}/top_p{args.top_p}/top_k{args.top_k} "
        f"vllm_is={args.vllm_is_mode} "
        + (
            f"overlong=shape(cache={args.soft_punish_cache}, w={args.overlong_penalty})"
            if not mask_truncated
            else "overlong=mask"
        )
    )

    run_name = args.run_name
    output_dir = args.output_dir or f"./{run_name}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
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
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        vllm_importance_sampling_mode=args.vllm_is_mode,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=2,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=100,
        save_total_limit=6,
        logging_steps=args.logging_steps,
        report_to="mlflow",
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
