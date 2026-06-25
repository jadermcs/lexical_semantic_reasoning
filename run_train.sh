#!/bin/bash
# Training machine launcher.
# Single unified env (train + vLLM) shared with the inference machine; vLLM is
# present here so server-mode GRPO can sync policy weights to `trl vllm-serve`.
set -euo pipefail
cd "$(dirname "$0")"

export WANDB_PROJECT="wic-grpo"
# export WANDB_MODE=disabled
# export HF_HOME=/scratch/$USER/.cache/huggingface

uv sync                        # idempotent; resolves from the shared uv.lock
# uv run src/grpo_self_verify.py --strategy verify-init --vllm-server-host isp-cap-n10 --vllm-server-port 8000
uv run src/grpo_finetune.py --vllm-server-host isp-cap-n10 --vllm-server-port 8000
