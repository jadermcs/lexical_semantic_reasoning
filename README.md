# lexical_semantic_reasoning

Word-in-context (WiC) sense discrimination via RLVR. A small policy (Qwen3) reasons
about what a target word means in each of two sentences, then decides whether the
two uses share the same sense. The verdict is a **gold label**, so the RL reward is
verifiable rather than an estimate.

The pipeline, in order:

1. **Generate** distilled reasoning traces from a teacher (`deepseek-v4-flash`,
   via OpenRouter) over MCL-WiC and SemCor-derived pairs.
2. **Filter** the traces for quality (cheap rules, then a local LLM judge).
3. **SFT** warm-start the policy on the traces the teacher got right.
4. **GRPO** on the verifiable same/different label.
5. **Self-distillation** — generate traces with the trained policy itself
   (`eval_sense.py`) and feed its correct ones back into SFT.

Evaluation is accuracy / F1 on the held-out MCL-WiC test split.

> **Running things.** Always use `uv` (`uv run python ...`). The heavy training
> stack (`torch`, `trl`, vLLM) runs on the servers; data prep and the reward unit
> tests run locally.

---

## The task

Given two sentences using the same target word (marked with `<t>` tags), the model
emits its reasoning and then a single JSON verdict:

```
<think>...reasoning about what the word means in each sentence...</think>
{"sense1": "a financial institution", "sense2": "sloping land beside a river", "same_sense": false}
```

`sense1`/`sense2` make the model commit to a gloss per usage before judging; only
`same_sense` is scored for correctness.

---

## 1. Data generation

### Sources

| File | What it is |
|------|------------|
| `data/mcl-wic.{train,dev,test}.json` | The gold MCL-WiC benchmark. Two sentences + a same/different label, no glosses. The test split is held out for evaluation only. |
| `data/mcl.json` | Teacher predictions over the MCL-WiC training pairs (8 000 pairs). |
| `data/semcor.json` | Teacher predictions over WiC pairs constructed from SemCor's sense annotations (~10 000 pairs; built outside this repo). |
| `data/mcl_semcor.json` | The two merged and label-balanced: 17 002 pairs, exactly 8 501 *same* / 8 501 *different*. Default SFT corpus. |

### Teacher sampling (`call_api.py`)

Each pair is sent to the teacher **k = 3 times** (self-consistency). Every record
stores the per-sample chains of thought and JSON answers plus a majority vote:

```json
{"lemma": "...", "pos": "...", "sentence1": "...", "sentence2": "...", "label": 0,
 "prediction": false, "confidence": 1.0, "votes": [false, false, false],
 "answers": ["{\"sense1\": ..., \"sense2\": ..., \"same_sense\": ...}", "..."],
 "reasonings": ["<CoT sample 1>", "..."]}
```

This is the **teacher schema** — the interchange format of the whole repo: the
filter annotates it, SFT trains from it, and `eval_sense.py` writes its own
predictions back in it (which is what closes the self-distillation loop, §5).

```bash
# needs OPENROUTER_API_KEY; streams one JSONL line per pair, resumable with -r
uv run python src/call_api.py -f data/pairs.json -m deepseek/deepseek-v4-flash
```

### Quality filtering (`filter_reasoning.py`)

Many raw traces are unusable (null, Chinese, repetition loops, stubs, or arguing
for the wrong answer). Filtering runs in two stages:

* **Stage 1 — rules (CPU, free).** Regex/statistical checks (script, length,
  type-token ratio) plus two exact checks: drop any sample whose own vote
  disagrees with the gold label, and any whose sense glosses contradict the label
  (identical glosses under *different*, differing glosses under *same*). Roughly
  half of all slots die here.
* **Stage 2 — LLM judge (GPU).** Survivors are graded by a local Gemma 4 12B
  (W4A16, vLLM, structured output) on four boolean axes: `english`, `coherent`,
  `faithful` (conclusion matches gold), `consistent` (prose supports its paired
  JSON). A trace is kept only if all required axes pass.

The script **annotates rather than deletes** — verdicts land in
`*_scored.json` (and a resumable checkpoint), so the accept threshold can be
retuned without re-judging. `--emit-filtered` writes the pruned corpus:

```bash
# runs in the isolated vLLM env (torch ABI conflict with the training env)
VIRTUAL_ENV=/path/to/vllm-env uv run --active python src/filter_reasoning.py \
    --data data/mcl_semcor.json \
    --out data/mcl_semcor_scored.json \
    --emit-filtered data/mcl_semcor_filtered.json
```

Filtered corpora in the repo: `data/mcl_filtered.json` (5 121 records),
`data/mcl_semcor_filtered.json` (13 023 records) — each record retains at least
one judge-approved trace.

---

## 2. SFT warm-start

Trains on the pairs the teacher got **right** (vote == gold label), so no
confidently-wrong reasoning is distilled. Of the samples that voted with the
majority, `--reasoning-select` decides which trace to keep — this is the ablation:

| `--reasoning-select` | Keeps |
|----------------------|-------|
| `first` *(default)* | the earliest majority-voting sample |
| `longest` | the longest chain of thought |
| `entropy` | the trace the model is most uncertain about (mean predictive entropy) |

`--data` picks the corpus (raw vs. filtered vs. merged — the second ablation axis):

```bash
uv run python src/sft_sense.py --reasoning-select longest --data data/mcl_semcor_filtered.json
```

Writes `./qwen-sense-sft-wic-<strategy>-<data-stem>` (e.g.
`qwen-sense-sft-wic-longest-mcl_semcor_filtered`), resumes from the latest
checkpoint in that dir, logs to wandb. A 5 % dev split is carved off a
deterministic shuffle for best-checkpoint selection.

---

## 3. GRPO

Warm-start from the SFT checkpoint and optimise the verifiable label:

```bash
uv run python src/grpo_sense.py --model ./qwen-sense-sft-wic-longest

# RL from base instead of the warm-started model (ablation)
uv run python src/grpo_sense.py
```

Writes `./qwen-sense-grpo-wic`. vLLM rollout offload via
`--vllm-server-host/--vllm-server-port` (see `run_train.sh` for the training
machine and `run_infer.sh` for the vLLM server machine).
`--distill-out PATH` additionally logs the policy's own correct rollouts
without affecting training.

### Reward

Defined in `sense_rewards.py` (importable without torch/trl, unit-tested in
`tests/test_sense_rewards.py`):

| Term | Range | What it buys |
|------|-------|--------------|
| `reward_wic_accuracy` | ±1.0 | the verdict is right (exact — the gold label is known) |
| `reward_wic_json` | −0.2 … +0.3 | a parseable JSON object, exactly the three keys, a real boolean verdict |
| `reward_wic_format` | 0 … +0.2 | a `<think>` block and an extractable verdict |
| `reward_think_length` | −0.3 … 0 | punishes a stubbed, missing or unclosed `<think>` |

The shape terms are capped well below the accuracy term, so **being right always
beats being tidy** — a test pins that invariant.

---

## 4. Evaluate

```bash
uv run python src/eval_sense.py --model Qwen/Qwen3-0.6B      # zero-shot baseline
uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic --force-json
```

Greedy-decodes the test split and prints accuracy + same/different P/R/F1
(completions with no extractable verdict are counted as `empty` and excluded from
P/R/F1). With `--force-json` the answer region is constrained to schema-valid
JSON via xgrammar (two-phase decode: free reasoning up to `</think>`, then a
grammar-guided continuation), so `empty` drops to 0.

Predictions are saved to `predictions_sense_wic_<split>.json` **in the teacher
schema** (one greedy sample per pair: single-element `answers`/`reasonings`), so
the file plugs straight into `sft_sense.py --data`. Metrics are printed, not
saved.

---

## 5. Self-distillation

Because `eval_sense.py` writes the teacher schema, the loop closes with the two
existing scripts — no glue code:

```bash
# 1. generate traces with the trained policy over the training pairs
uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic \
    --split train --output data/self_traces.json

# 2. SFT on them — load_teacher_traces keeps only the pairs the policy got right
uv run python src/sft_sense.py --data data/self_traces.json
```

`load_teacher_traces` applies the same teacher-correct filter as in §2
(prediction == gold), so wrong or unparseable generations are dropped at load
time and the usable train set is smaller than the split.

There is also a during-training collection path: `grpo_sense.py --distill-out`
appends every correct rollout to per-rank JSONL files. That path uses a raw
rollout schema (`prompt`/`completion`/`score`) rather than the teacher schema,
so it would need a small conversion before SFT — the eval-based loop above is
the wired one.

---

## File map

| File | Role |
|------|------|
| `src/sense_data.py` | Record loading (MCL-WiC + teacher traces), the shared prompt, answer parsing |
| `src/call_api.py` | Teacher self-consistency sampling over the WiC pairs |
| `src/filter_reasoning.py` | Quality-filter the distilled traces (rules + LLM judge) |
| `src/sft_sense.py` | SFT warm-start on the distilled traces |
| `src/grpo_sense.py` | GRPO on the verifiable label |
| `src/sense_rewards.py` | The reward functions |
| `src/eval_sense.py` | Test-split generation + accuracy/F1; emits teacher-schema predictions |
| `tests/test_sense_rewards.py` | Reward contracts (runs on CPU in a second) |
| `run_train.sh` / `run_infer.sh` | Launchers for the GRPO trainer and the vLLM rollout server |

```bash
uv run pytest        # reward unit tests
```
