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
# one-time download that `uv sync` knows nothing about. `filter_reasoning
# --strict-gloss` opens it, so fetch it here rather than failing partway through a
# stage. Idempotent, and deliberately routed through gloss_wordnet so the check
# exercises the exact path that module uses (including a WN_LEXICON override).
# NB this is Open English WordNet; the gold-gloss reward reads NLTK's WordNet 3.0
# instead (the inventory SemCor's annotations refer to) — never mix the two.
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

# Gold-sense WiC pairs. `data/semcor_en.json.gz` is an external
# input (produced outside this repo); this stage only reads it and materialises the
# sampled pairs, so a run is reproducible and NLTK stays off the trainer's path.
# Needs NLTK's WordNet 3.0 corpus — a separate one-time download, like the lexicon above.
uv run python -c "import nltk; nltk.download('wordnet', quiet=True)"
uv run src/semcor_pairs.py --out data/semcor_wic.json

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
