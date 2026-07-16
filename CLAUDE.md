# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Word-in-context (WiC) sense discrimination via RLVR: a small Qwen3 policy reasons about a target word's meaning in two sentences, then emits a JSON verdict on whether the senses match. The verdict is a gold label, so the RL reward is verifiable. See README.md for the full pipeline write-up.

## Commands

Always run Python through uv (`uv run python ...`), never bare `python`/`python3`.

```bash
uv run pytest                                # reward unit tests (CPU, fast)
uv run pytest tests/test_sense_rewards.py -k <name>   # single test

# Pipeline stages, in order:
uv run python src/call_api.py -f data/pairs.json -m deepseek/deepseek-v4-flash  # teacher traces (needs OPENROUTER_API_KEY; -r resumes)
uv run python src/filter_reasoning.py --data data/mcl_semcor.json               # quality filter (stage 2 needs a GPU + vLLM)
uv run python src/sft_sense.py --reasoning-select longest --data data/mcl_semcor_filtered.json
uv run python src/grpo_sense.py --model ./qwen-sense-sft-wic-longest
uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic --force-json
```

The heavy stack (torch, trl, vLLM) runs on the servers via `run_train.sh` (GRPO trainer) and `run_infer.sh` (`trl vllm-serve` rollout server); data prep and tests run locally. Both scripts `uv sync` from the shared `uv.lock`, and torch/vLLM resolve from the cu130 wheel index pinned in `pyproject.toml` to keep CUDA ABIs compatible.

## Architecture

**The teacher schema is the interchange format of the whole repo.** One JSON record per WiC pair with `lemma/pos/sentence1/sentence2/label` plus per-sample `votes`, `answers`, `reasonings`, and a majority-vote `prediction`. `call_api.py` produces it, `filter_reasoning.py` annotates it in place, `sft_sense.py` trains from it, and `eval_sense.py` writes its own predictions back in it — that last fact is what closes the self-distillation loop (eval the trained policy on the train split, feed the output straight back to `--data` of `sft_sense.py`).

Pipeline flow: `call_api.py` (teacher self-consistency, k=3) → `filter_reasoning.py` (stage 1: CPU regex/consistency rules; stage 2: local Gemma judge on vLLM scoring english/coherent/faithful/consistent) → `sft_sense.py` (trains only on pairs where teacher vote == gold; `--reasoning-select first|longest|entropy` is the ablation axis for which trace to keep) → `grpo_sense.py` (GRPO on the verifiable label, warm-started from SFT) → `eval_sense.py` (greedy decode; `--force-json` constrains the answer region with xgrammar after free `<think>` reasoning).

`src/sense_data.py` holds record loading, the shared prompt, and answer parsing used by everything downstream. Scripts import siblings as top-level modules (`import sense_data`), so `pythonpath = ["src"]` in pyproject makes tests resolve the same way.

**Reward invariant** (`src/sense_rewards.py`, importable without torch/trl): the accuracy term (±1.0) dominates the shaping terms (JSON validity, format, think-length), so being right always beats being tidy. A test in `tests/test_sense_rewards.py` pins this — don't rescale terms without checking it.

Model output contract: `<think>...</think>` followed by `{"sense1": ..., "sense2": ..., "same_sense": bool}`. Only `same_sense` is scored; the glosses force per-usage commitment.

## Conventions

- SFT output dirs are named `./qwen-sense-sft-wic-<strategy>-<data-stem>` and training resumes from the latest checkpoint in that dir; wandb logging is on by default.
- For constrained/structured generation use xgrammar, not lm-format-enforcer.
- `data/mcl-wic.test.json` is held out for evaluation only — never train on it.
