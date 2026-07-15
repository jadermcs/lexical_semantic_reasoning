# lexical_semantic_reasoning

Word-in-context (WiC) sense discrimination via RLVR. A small policy (Qwen3) reasons
about what a target word means in each of two sentences, then decides whether the
two uses share the same sense. The verdict is a **gold label**, so the RL reward is
verifiable rather than an estimate.

The pipeline, in order:

1. **Distil** reasoning traces from a teacher (`deepseek-v4-flash`, via OpenRouter).
2. **SFT** warm-start the policy on the traces the teacher got right.
3. **GRPO** on the verifiable same/different label.
4. **Self-distillation** — feed the policy's own correct GRPO rollouts back into SFT
   *(groundwork in place, not yet wired end-to-end — see §5)*.

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

## 1. Data

| File | What it is |
|------|------------|
| `data/mcl-wic.{train,dev,test}.json` | The gold MCL-WiC benchmark. Two sentences + a same/different label, no glosses. |
| `data/mcl_semcor.json` | Teacher predictions from `call_api.py`: per pair, `k` sampled reasoning traces, their JSON answers, and a self-consistency vote. |

`sense_data.py` loads both into one record shape (`lemma`, `pos`, `label`,
`usage1`, `usage2`) and owns the prompt, so SFT, GRPO and eval all see exactly the
same format.

**Materialise the RL task** so the examples the policy rolls out against can be
read before spending a run on them:

```bash
uv run python src/wic_task.py --splits train dev          # -> data/wic_task.<split>.jsonl
uv run python src/wic_task.py --splits dev --show 3       # print a few rendered prompts
```

GRPO builds these on first use if the files are missing.

### (Optional) Re-distil teacher traces

```bash
# k-sample self-consistency over the WiC pairs (needs an OpenRouter key)
uv run python src/call_api.py --model deepseek/deepseek-v4-flash

# score/filter the traces: cheap CPU rules, then a local LLM judge (isolated vLLM env)
uv run python src/filter_reasoning.py
```

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

```bash
uv run python src/sft_sense.py --reasoning-select longest
```

Writes `./qwen-sense-sft-wic-<strategy>`, resumes from the latest checkpoint in
that dir, logs to wandb.

---

## 3. GRPO

Warm-start from the SFT checkpoint and optimise the verifiable label:

```bash
uv run python src/grpo_sense.py --model ./qwen-sense-sft-wic-longest

# RL from base instead of the warm-started model (ablation)
uv run python src/grpo_sense.py
```

Writes `./qwen-sense-grpo-wic`. vLLM rollout offload via
`--vllm-server-host/--vllm-server-port` (see `run_train.sh` / `run_infer.sh`).
`--distill-out PATH` additionally logs the policy's own correct rollouts for
self-distillation (§5) without affecting training.

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
uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic
uv run python src/eval_sense.py --model Qwen/Qwen3-0.6B      # zero-shot baseline
```

Greedy-decodes the test split and reports accuracy + same/different P/R/F1
(completions with no extractable verdict are counted as `empty` and excluded from
P/R/F1). Saves `predictions_sense_wic_<split>.json`.

---

## 5. Self-distillation (next)

The collection half exists: `grpo_sense.py --distill-out data/self_distill` appends
every rollout whose verdict was **correct** to `data/self_distill.rank<N>.jsonl`, one
file per process rank, as:

```json
{"score": 1.0, "completion": "<think>…</think>\n{…}", "prompt": "…", "lemma": "bank", "label": "different"}
```

**Not yet wired:** `sft_sense.py` reads the *teacher* schema
(`sentence1`/`sentence2`/`reasonings`/`answers`/`prediction`, via
`sense_data.load_teacher_traces`), which is not the schema above. Closing the loop
needs a small loader that turns saved rollouts into SFT records — splitting
`completion` into its `think` block and `sense1`/`sense2` JSON (both parsers already
exist: `sense_data.parse_wic_answer` and `sense_rewards._extract_think`) and
recovering `usage1`/`usage2` from the pair. Dedup across ranks, and consider keeping
only one rollout per pair so easy pairs don't dominate the SFT mix.

---

## File map

| File | Role |
|------|------|
| `src/sense_data.py` | Record loading (MCL-WiC + teacher traces), the shared prompt, answer parsing |
| `src/wic_task.py` | Materialise the RL task to `data/wic_task.<split>.jsonl` for inspection |
| `src/call_api.py` | Teacher self-consistency sampling over the WiC pairs |
| `src/filter_reasoning.py` | Quality-filter the distilled traces (rules + LLM judge) |
| `src/sft_sense.py` | SFT warm-start on the distilled traces |
| `src/grpo_sense.py` | GRPO on the verifiable label |
| `src/sense_rewards.py` | The reward functions |
| `src/eval_sense.py` | Test-split generation + accuracy/F1 |
| `tests/test_sense_rewards.py` | Reward contracts (runs on CPU in a second) |

```bash
uv run pytest        # reward unit tests
```
