#!/bin/bash
# Training machine launcher.
# Single unified env (train + vLLM) shared with the inference machine; vLLM is
# present here so server-mode GRPO can sync policy weights to `trl vllm-serve`.
set -euo pipefail
cd "$(dirname "$0")"

export WANDB_PROJECT="wic-grpo"
export TORCHDYNAMO_DISABLE=1
# export WANDB_MODE=disabled
# export HF_HOME=/scratch/$USER/.cache/huggingface

uv sync                        # idempotent; resolves from the shared uv.lock
uv run src/grpo_sense.py --mode direct --vllm-server-host isp-cap-n10 --vllm-server-port 8000
