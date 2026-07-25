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

# The `wn` package is a project dependency, but the lexicon it reads is a separate
# one-time download that `uv sync` knows nothing about. Both `filter_reasoning
# --strict-gloss` and `sense_rewards.reward_wic_wordnet` open it, so fetch it here
# rather than failing partway through a stage. Idempotent, and deliberately routed
# through gloss_wordnet so the check exercises the exact path the rewards use
# (including a WN_LEXICON override).
uv run python -c "
import sys; sys.path.insert(0, 'src')
import gloss_wordnet as gw
try:
    gw.synsets_for('bank', 'noun')
    print(f'WordNet lexicon {gw.LEXICON} present')
except LookupError:
    import wn
    print(f'downloading WordNet lexicon {gw.LEXICON} ...')
    wn.download(gw.LEXICON)
"

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
