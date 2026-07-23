#!/bin/bash
# Training machine launcher.
# Single unified env (train + vLLM) shared with the inference machine; vLLM is
# present here so server-mode GRPO can sync policy weights to `trl vllm-serve`.
set -euo pipefail
cd "$(dirname "$0")"

export WANDB_PROJECT="wic-reasoning"
export TORCHDYNAMO_DISABLE=1
# export WANDB_MODE=disabled
# export HF_HOME=/scratch/$USER/.cache/huggingface

uv sync

uv run src/filter_reasoning.py \
    --data data/mcl_train_dev.json \
    --out data/mcl_train_dev_scored.json \
    --emit-filtered data/mcl_train_dev_filtered.json \
    --strict-gloss True

uv run src/prepare_data.py \
    --data data/mcl_train_dev_filtered.json \
    --reasoning-select longest \
    --out data/sft_wic_filtered \

uv run src/sft_sense.py \
    --data data/sft_wic_filtered

uv run src/eval_sense.py \
    --model ./qwen-sft_wic_filtered \
    --split test
