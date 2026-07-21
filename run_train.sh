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

uv sync
uv run src/prepare_data.py \
    --data data/mcl_train_dev_filtered.json \
    --reasoning-select longest \
    --out data/sft_wic_filtered \
    --balance-labels

uv run src/sft_sense.py \
    --data data/sft_wic_filtered

uv run src/eval_sense.py \
    --model ./qwen-sft_wic_filtered \
    --split test

uv run src/grpo_sense.py \
    --model ./qwen-sft_wic_filtered \
    --exclude-pairs data/sft_wic_filtered.sft_pairs.json
    --balance-labels
