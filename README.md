# lexical_semantic_reasoning

Cluster-grounded definition generation via RLVR. A small policy (Qwen3) is taught
to generate WordNet-style sense definitions and is then optimised with a
verifiable, gloss-grounded reward. Methodology: `ch05_implementation_plan.md`.

This README covers the **sense-modeling ablation pipeline**: building data,
distilling reasoning traces, training the four configurations, and evaluating
them on BLEU.

> **Running things.** Always use `uv` (`uv run python ...`). The heavy training
> stack (`torch`, `trl`, vLLM) runs on the servers; data generation and
> evaluation scaffolding run locally.

---

## The four ablation configurations

| # | Config | Script | Output dir |
|---|--------|--------|-----------|
| 1 | SFT — direct (single-usage definition) | `sft_sense.py --mode direct` | `qwen-sense-sft-direct` |
| 2 | SFT — triplet (anchor/positive/negative) | `sft_sense.py --mode triplet` | `qwen-sense-sft-triplet` |
| 3 | GRPO — direct, warm-started from #1 | `grpo_sense.py --mode direct` | `qwen-sense-grpo-direct` |
| 4 | GRPO — triplet, warm-started from #2 | `grpo_sense.py --mode triplet` | `qwen-sense-grpo-triplet` |

- **direct** sees one usage and generates its gloss.
- **triplet** sees an anchor + positive (same sense) and a negative (different
  sense of the same lemma) and generates the contrastive reasoning + three glosses.
- All four are evaluated by **BLEU** of the generated gloss vs. the WordNet gold
  definition (triplet scores the *anchor* gloss only, for a clean per-usage
  comparison with direct).

---

## 1. Build the WordNet ablation splits

`sense_data.py` builds the train/dev/test data for both task framings from Open
English WordNet (usages = synset example sentences, gold = synset definitions).
The split is **lemma-disjoint** (no lemma appears in two splits).

```bash
uv run python src/sense_data.py
```

Writes `data/sense_{direct,triplet}.{train,dev,test}.json` and prints stats:

```
train  direct= 30911  triplet=  5286  lemmas=17001
dev    direct=  3934  triplet=   638  lemmas=2125
test   direct=  3853  triplet=   607  lemmas=2125
lemma overlap across splits: 0 0 0 (must be 0,0,0)
```

The split is reproducible from `(lexicon, seed=42)` and is reused everywhere via
`sense_data.lemma_splits()`.

---

## 2. (Optional) Distil ChatGPT reasoning traces from SemCor

`gen_reasoning_data.py` builds reasoning-distillation data for the Phase-3
warm-start. Usages come from **SemCor** (gold sense-tagged); definitions are
resolved through `wn` (SemCor WN3.0 sense key → OEWN 2024 synset). ChatGPT writes
a concise forward-reasoning `<think>` trace; the `<answer>` holds the gold
gloss(es).

```bash
# build prompts only (no API calls)
uv run python src/gen_reasoning_data.py --mode both

# build + annotate with a teacher model (needs OPENAI_API_KEY + `openai`)
uv run python src/gen_reasoning_data.py --mode both --annotate --model gpt-5-nano
```

Outputs `data/semcor_distill_{direct,triplet}.jsonl`. Each record carries the
`prompt`, the `sft_answer` block, and (after `--annotate`) the `argument`, the
assembled `sft_target` = `<think>…</think><answer>…</answer>`, and (when filtering
is on) the per-record `bertscore`.

### Quality control (annotate-only)

The teacher occasionally refuses or drifts off the target sense; those traces are
poison for the warm-start. Two guards run under `--annotate`:

- **Empty/refused completions** are always dropped (a `None`/blank completion is
  skipped, not written).
- **`--min-bertscore`** drops traces whose reasoning is semantically far from the
  gold definition it is meant to arrive at. Each `argument` is scored against its
  gold gloss (`definition` for direct, `definition_same` for triplet) with
  **BERTScore**, baseline-rescaled so the threshold is interpretable (~0 unrelated
  → ~1). Default `0.0` = off; `~0.15` is a reasonable start. Needs the
  `bert-score` package (`uv add bert-score`); the default `roberta-large` model
  (~1.4 GB) downloads on first use, so run this on a server.

```bash
# annotate, then drop traces that wander off the gold sense
uv run python src/gen_reasoning_data.py --mode both --annotate --model gpt-5-nano \
    --min-bertscore 0.15
```

Every kept record stores its `{precision, recall, f1}` under `bertscore`, so you
can build the JSONL once with `--min-bertscore 0` and inspect the distribution
before committing to a threshold.

| Filter flag | Default | Effect |
|-------------|---------|--------|
| `--min-bertscore F` | `0.0` (off) | drop traces scoring below `F` vs. the gold gloss |
| `--bertscore-metric {f1,recall,precision}` | `f1` | which component to threshold; `recall` rewards covering the gloss and is lenient on the argument's extra reasoning |
| `--bertscore-model NAME` | `roberta-large` (via `lang=en`) | override the scoring model; a custom model skips baseline rescaling |

### Leakage control: `--lemma-split`

Distillation data is *training* data, so it must not contain any dev/test lemma.
The filter reuses the **same** split as the ablation (`sense_data.lemma_splits`,
seed 42).

| `--lemma-split` | Keeps | ~records (direct / triplet) | When |
|-----------------|-------|------------------------------|------|
| `non-eval` *(default)* | every lemma except dev/test | 38.9k / 9.4k | maximise warm-start data |
| `train` | strictly the ablation's train lemmas | 24.4k / 8.3k | warm-start universe must equal train |
| `dev` / `test` | only those eval lemmas | — | inspection / analysis |
| `all` | no filtering | — | leakage not a concern |

Both `non-eval` and `train` are leakage-safe (verified: 0 dev and 0 test lemmas
in the output). `non-eval` additionally keeps SemCor lemmas that have no WordNet
example sentence (never assigned to any split, so they can never appear in eval).

```bash
# strict train-lemma warm-start
uv run python src/gen_reasoning_data.py --mode both --lemma-split train --annotate --model gpt-5-nano
```

Useful flags: `--mode {direct,triplet,both}`, `--max-per-lemma N`,
`--max-examples N` (cap for a quick sample), `--out PATH` (single mode only). See
[Quality control](#quality-control-annotate-only) above for the `--annotate`
filtering flags.

---

## 3. Train the four configurations

```bash
# config 1 — SFT direct
uv run python src/sft_sense.py --mode direct

# config 2 — SFT triplet
uv run python src/sft_sense.py --mode triplet

# config 3 — GRPO direct, warm-started from the SFT checkpoint
uv run python src/grpo_sense.py --mode direct  --model ./qwen-sense-sft-direct

# config 4 — GRPO triplet, warm-started from the SFT checkpoint
uv run python src/grpo_sense.py --mode triplet --model ./qwen-sense-sft-triplet
```

GRPO ablation (RL from base instead of the warm-started model) — just omit the
SFT checkpoint:

```bash
uv run python src/grpo_sense.py --mode direct      # RL from base
```

Both scripts auto-build the WordNet splits if `data/sense_*` is missing, resume
from the latest checkpoint in their output dir, and log to wandb.

### GRPO reward

Verifiable reward = **gold-gloss similarity** + a small format term
(`sense_data.gloss_similarity`, = mean of token-F1 and BLEU-2 — both stay smooth
on short glosses, unlike BLEU-4). For triplet, fidelity is averaged over the
three generated glosses against their gold definitions. vLLM rollout offload is
available via `--vllm-server-host/--vllm-server-port`.

---

## 4. Evaluate (BLEU)

`eval_sense.py` greedily generates a gloss per held-out test usage and reports
corpus BLEU-4 against the WordNet gold (+ mean similarity, + empty-output count).

```bash
# evaluate each config (point --model at the trained checkpoint)
uv run python src/eval_sense.py --mode direct  --model ./qwen-sense-sft-direct
uv run python src/eval_sense.py --mode direct  --model ./qwen-sense-grpo-direct
uv run python src/eval_sense.py --mode triplet --model ./qwen-sense-sft-triplet
uv run python src/eval_sense.py --mode triplet --model ./qwen-sense-grpo-triplet

# zero-shot base-model baseline
uv run python src/eval_sense.py --mode direct
```

Prints e.g. `[direct] n=3853  BLEU=… mean_sim=… empty=…` and saves predictions to
`predictions_sense_{mode}.json`. Add `--bertscore` to also report BERTScore F1
(downloads `roberta-large`; server-only), `--split dev` for the dev set, and
`--max-samples N` for a quick check.

---

## File map

| File | Role |
|------|------|
| `src/sense_data.py` | WordNet lemma-disjoint splits, prompts/targets, BLEU + reward metrics |
| `src/gen_reasoning_data.py` | SemCor → ChatGPT reasoning-distillation traces (train-filtered) |
| `src/sft_sense.py` | SFT for `--mode direct\|triplet` (configs 1 & 2) |
| `src/grpo_sense.py` | GRPO for `--mode direct\|triplet` (configs 3 & 4) |
| `src/eval_sense.py` | Generate on test split, report BLEU |
| `ch05_implementation_plan.md` | Methodology / phase plan |

Data caches: WordNet via `wn` (`~/.wn_data`); SemCor auto-downloads to
`.cache/semcor/` (gitignored).
