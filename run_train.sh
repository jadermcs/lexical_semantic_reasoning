#!/bin/bash
# Training machine launcher.
# Uses a training-only uv environment so it never clobbers the inference
# machine's torch/torchvision in the shared project folder.
set -euo pipefail
cd "$(dirname "$0")"

export UV_PROJECT_ENVIRONMENT=".venv-train"
export WANDB_PROJECT="wic-grpo"
# export WANDB_MODE=disabled
# export HF_HOME=/scratch/$USER/.cache/huggingface

uv sync --extra train          # idempotent; resolves from the shared uv.lock
# uv run src/grpo_self_verify.py --strategy verify-init --vllm-server-host isp-cap-n10 --vllm-server-port 8000
uv run src/grpo_finetune.py verify-init --vllm-server-host isp-cap-n10 --vllm-server-port 8000
