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
uv run accelerate launch --num_processes 2 src/grpo_self_verify.py
