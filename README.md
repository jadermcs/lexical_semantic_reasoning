# lexical_semantic_reasoning

Word-in-context (WiC) sense discrimination via RLVR. A small policy (Qwen3-1.7B)
reasons about what a target word means in each of two sentences, then decides
whether the two uses share the same sense. The verdict is a **gold label**, so the
RL reward is verifiable rather than an estimate.

The method runs in **two rounds**. Round 1 exists to produce a *trace generator*;
round 2 trains the model that ships.

| | Round 1 — bootstrap | Round 2 — the real run |
|---|---|---|
| **SFT on** | `deepseek-v4-flash` traces (OpenRouter) | traces sampled by the round-1 policy |
| **Starting weights** | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-1.7B` again — **from scratch** |
| **What GRPO buys** | prompt-format adherence, reasoning long enough to be worth distilling | gloss quality: no `or`-hedging, no two-word glosses |

The round-1 checkpoint is scaffolding. Nothing downstream inherits its weights —
only its traces — so round 1 is judged by the corpus it produces, not by its own
test score.

```
deepseek-v4-flash ──distil──▶ SFT ──▶ GRPO ──sample k traces──▶ explode + decontaminate
   (call_api.py)          (sft_sense) (grpo_lora)   (eval_sense -k)   (analysis.ipynb)
                                                                          │
                                                                          ▼
                        held-out MCL-WiC test ◀── GRPO ◀── SFT from base Qwen3-1.7B
                             (eval_sense)      (grpo_lora)    (sft_sense)
```

> **Running things.** Always use `uv` (`uv run python ...`). The heavy stack
> (`torch`, `trl`, vLLM) runs on the servers via `run_train.sh` / `run_infer.sh`;
> corpus prep and the reward unit tests run locally.

---

## The task

Given two sentences using the same target word (marked with `<t>` tags), the model
emits its reasoning and then a single JSON verdict:

```
<think>...reasoning about what the word means in each sentence...</think>
{"sense1": "a financial institution", "sense2": "sloping land beside a river", "same_sense": false}
```

`sense1`/`sense2` make the model commit to a gloss per usage before judging. Only
`same_sense` is scored for correctness; the *form* of the glosses is shaped by a
reward term in round 2.

---

## 0. Corpora

Corpus construction lives in the **sibling repo `../lexical_datasets`**
(`data_preprocess.py`), which builds `fews`, `masc`, `semcor`, `dwug` and
`chainnet` plus the `xl-lexeme.{train,dev}.json` merge. `analysis.ipynb` in this
repo concatenates the two xl-lexeme splits into the corpus everything else reads:

```python
data = pd.concat([pd.read_json(f"../lexical_datasets/xl-lexeme.{s}.json")
                  for s in ("train", "dev")]).reset_index(drop=True)
data.to_json("data/xl-lexeme.json", indent=2, orient="records", force_ascii=False)
```

| Corpus | Shape | What it is |
|---|---|---|
| `data/xl-lexeme.json` | marked, `source`/`lang1`/`lang2`/`split`/`origin` | The 29-language merge of `wic`, `mcl-wic`, `xl` and `am2ico` — 272 493 pairs. The rollout and self-distillation corpus. |
| `fews` / `masc` / `semcor` / `dwug` | **unmarked**, `word1`/`word2`, int label | Built by the sibling repo, grouped train/dev/test at `random_state=42`. `masc.train.json` is empty by construction — MASC is dev/test only. |
| `data/mcl-wic.{train,dev,test}.json` | marked | The gold MCL-WiC benchmark. **The test split is held out for evaluation only.** |

The two families are not interchangeable: `xl-lexeme` ships `<t> word </t>` inside
the stored sentence, placed from exact character offsets, while the FEWS/MASC/
SemCor/DWUG sets give `word1`/`word2` and leave the sentence untouched. They need
marking and the `source`/`lang` fields before they can join a rollout batch.

`am2ico` is the reason for the normalization the merge does: it takes whole
Wikipedia paragraphs and is PTB-tokenized in 99.9% of them, so contexts are
chopped to the sentence holding the target, detokenized, and windowed — budgets in
**tokens, never characters**, because chars/token runs 4.73 (en-en) to 1.22
(zh-zh) and a character cut costs CJK ~4× what it costs English. PTB spacing
(`the league 's teams`) is a *source tell* before it is a cost: mixed into one
rollout batch it says which corpus a pair came from before the policy has read a
word.

`src/prepare_lexeme.py` and `src/semcor_pairs.py` are this repo's own builders and
predate the move to `../lexical_datasets`; their inputs (`data/{train,dev}/*.data`,
`data/semcor_en.json.gz`) no longer live here, but they document the normalization
and the WordNet-3.0 gold-gloss extraction the pairs went through.

---

## 1. Teacher distillation

Each pair is sent to `deepseek-v4-flash` **k = 3 times** (self-consistency). Every
record stores the per-sample chains of thought and JSON answers plus a majority
vote:

```json
{"lemma": "...", "pos": "...", "sentence1": "...", "sentence2": "...", "label": 0,
 "prediction": false, "confidence": 1.0, "votes": [false, false, false],
 "answers": ["{\"sense1\": ..., \"sense2\": ..., \"same_sense\": ...}", "..."],
 "reasonings": ["<CoT sample 1>", "..."]}
```

This is the **teacher schema** — the interchange format of the whole repo. The
filters annotate it, `utils.build_sft_dataset` trains from it, and `eval_sense.py`
writes its own predictions back in it, which is what makes round 2 need no glue
code.

```bash
# needs OPENROUTER_API_KEY; streams one JSONL line per pair, resumable with -r
uv run python src/call_api.py -f data/pairs.json -m deepseek/deepseek-v4-flash
```

### Triage the questions before paying for answers (`-t quality`)

`call_api.py -t quality` rubric-grades each *pair* — booleans `well_formed1/2`,
`target_ok1/2`, ordinals `evidence1/2` (1–3) and `difficulty` (1–5) — at k=1, and
`filter_quality.py` prunes on them. The verdict is answered last, so it doubles as
a label check.

```bash
uv run python src/call_api.py -f data/semcor.full.json -t quality
uv run python src/filter_quality.py --preds predictions_semcor.full_*.jsonl \
    --data data/semcor.full.json --out data/semcor.clean.json
```

Read the **agreement gap** the metrics print, not the flag rate: teacher-vs-gold
agreement on the items an axis flags vs the ones it clears. An axis with no gap is
finding nothing however plausible it reads. On the first `semcor.full.json` build
`well_formed1` fired on 58% of items and its flagged items agreed with gold *more*
often (0.835 vs 0.734) — it was reporting the builder's own truncation, not
anything about the example.

### Filter the traces (`filter_reasoning.py`)

* **Stage 1 — rules (CPU, free).** Null, non-Latin script, stub, blowup,
  repetition loop; then a vote rule that drops every slot whose own vote
  disagrees with the gold label, and a gloss rule for glosses that contradict it.
  Roughly half of all slots die here and never reach the GPU.
* **Stage 2 — LLM judge (GPU).** Survivors are graded by a local Qwen3.5-9B
  (AWQ int4, vLLM, structured output) on `english`, `coherent`, `consistent`.

There is deliberately **no `faithful` axis, and the judge is not shown the gold
label.** Stage 1's vote check settles faithfulness exactly and for free, so every
trace reaching the judge is faithful by construction — over 20 804 judgements the
axis fired 346 times, all judge errors, and correlated `consistent` into agreeing
with it (345 shared slots) because the label sat in the prompt above both
questions.

The script **annotates rather than deletes**, so the accept threshold can be
retuned without re-judging. `--emit-filtered` writes the pruned corpus:

```bash
# runs in the isolated vLLM env (torch ABI conflict with the training env)
VIRTUAL_ENV=/path/to/vllm-env uv run --active python src/filter_reasoning.py \
    --data data/mcl_train_dev.json \
    --out data/mcl_train_dev_scored.json \
    --emit-filtered data/mcl_train_dev_filtered.json
```

---

## 2. SFT warm start

```bash
uv run python src/sft_sense.py --model Qwen/Qwen3-1.7B \
    --file_path data/mcl_train_dev_filtered.json
```

There is no separate prepare step: `utils.build_sft_dataset` runs the teacher-schema
file through `sense_data.load_teacher_traces` and renders `{prompt, completion}`
rows on the fly, so switching corpora is one flag. Writes `./qwen-<data-stem>`
(override with `--output-dir`), carves a fixed 200-row dev split for
best-checkpoint selection, and logs to MLflow (`wic-sft`).

---

## 3. GRPO

```bash
uv run python src/grpo_lora.py --model ./qwen-mcl_train_dev_filtered \
    --file-path data/xl-lexeme.json
```

LoRA (r=32, all-linear) over a merged policy — it refuses an adapter directory at
`--model`. Defaults: `dr_grpo` loss, `scale_rewards=none`, DAPO clip-higher
(0.2 / 0.28), `beta=0`, and rollout temperature **1.0 with top-k off**. Qwen3's
published 0.6 / top-k-20 recipe is an *inference* setting; used for rollouts it
caps diversity and manufactures unanimous groups with zero advantage.
`--vllm-server-host` offloads rollouts to `run_infer.sh`'s server; without it vLLM
runs colocated.

### Reward

Defined in `sense_rewards.py` (importable without torch/trl, unit-tested in
`tests/test_sense_rewards.py`):

| Term | Range | What it buys |
|------|-------|--------------|
| `reward_wic_accuracy` | ±1.0 | the verdict is right (exact — the gold label is known) |
| `reward_wic_json` | −0.2 … +0.3 | a parseable JSON object, exactly the three keys, a real boolean verdict |
| `reward_wic_format` | 0 … +0.2 | a `<think>` block and an extractable verdict |
| `reward_wic_consistency` | −0.3 … 0 | punishes glosses that contradict the verdict (identical glosses called *different*, unrelated glosses called *same*) |
| `reward_wic_gloss_form` | −0.34 … 0 | punishes `or`-hedged and stub glosses (§5) |
| `reward_think_length` | −0.3 … 0 | punishes a stubbed, missing or unclosed `<think>` |

`grpo_lora.py` adds TRL's repetition penalty and soft overlong punishment on top.
The shape terms span 1.44 against an accuracy gap of 1.5, so **being right always
beats being tidy** — a test pins that invariant, and there is 0.06 of headroom
before any new term has to shrink an existing one.

---

## 4. Self-distillation, then train from scratch

Because `eval_sense.py` writes the teacher schema, the loop closes with the two
existing scripts plus one notebook:

```bash
# 1. sample k traces per pair with the round-1 policy
uv run python src/eval_sense.py --model ./qwen-lora-grpo-wic --force-json \
    --path data/xl-lexeme.json -k 3 --output predictions_sense_wic_xl-lexeme.json
```

**2. Build the SFT set in `analysis.ipynb`** — four steps, in order:

```python
test_lemmas = set(pd.read_json("data/mcl-wic.test.json").lemma.str.lower().values)
data = pd.read_json("predictions_sense_wic_xl-lexeme.json")
data = data.explode(["reasonings", "answers", "votes"], ignore_index=True).sample(frac=1.0, random_state=42)
data = data[~data.lemma.str.lower().isin(test_lemmas)]
data = data[~data.pos.isna()]
data[data.label.astype(bool) == data.votes].to_json("data/lexeme.correct.json", orient="records", indent=2)
```

* **explode** turns one row per *pair* into one row per *sample* — 272 493 pairs
  become 792 696 rows (short of 3× because empty slots are already gone). This is
  the format `sense_data.load_teacher_traces` reads: scalars, not lists.
* **test-lemma decontamination** drops every row whose lowercased lemma appears in
  the held-out split. On *lemmas*, not pairs — the same lemma reappears across
  corpora with different sentences, and held-out accuracy would otherwise measure
  recall of a sense inventory that was trained on.
* **`pos.isna()`** drops the cross-lingual `am2ico` rows, which carry `lang2: en`
  and no POS; the prompt renders `pos` in its first line.
* **`label == votes`, not `label == prediction`.** After the explode, `prediction`
  is still the *pair's* majority vote, so filtering on it would keep the minority
  samples that argued against it. `votes` is the individual sample's own verdict.

```bash
# 3. SFT a *fresh* Qwen3-1.7B on them, then GRPO again
uv run python src/sft_sense.py --model Qwen/Qwen3-1.7B --file_path data/lexeme.correct.json
uv run python src/grpo_lora.py --model ./qwen-lexeme.correct --file-path data/xl-lexeme.json
```

**`-k` is what makes this worth running.** At k=1 (greedy) each pair yields one
trace and a lucky-but-wrong chain survives whenever its verdict happens to match
gold. At k>1 the pairs decode with thinking-mode sampling, `prediction` becomes a
majority vote (ties → `None`), and each sample is an independent candidate.
`answers`, `reasonings` and `votes` stay index-aligned and always length k — a
sample that lost only its JSON still holds its slot, or the next sample's verdict
would be paired with this one's reasoning. That alignment is exactly what lets the
three columns be exploded together.

Every field of the source record that isn't an output key rides through, so
`source`/`lang1`/`split` survive into the distilled file and the corpus can still
be stratified by where its pairs came from.

### Hard-example mining

The same notebook keeps the complement — the pairs the policy gets *wrong* — as a
rollout pool rather than an SFT set:

```python
data = pd.read_json("predictions_sense_wic_mcl_train_dev.json")
data[data.label != data.prediction].to_json("data/mcl-hard.json", orient="records", indent=2)
```

---

## 5. What round-2 GRPO fixes: the glosses

`same_sense` is the only thing verified, so the glosses are free to degenerate in
two specific ways, and `reward_wic_gloss_form` prices both **per gloss** (hedging
in both usages costs twice):

* **−0.12 for a bare `or`.** "a river bank or the edge of a road" commits to
  neither sense, keeps both readings alive, and still satisfies the consistency
  term.
* **−0.05 for fewer than 3 words.** "money", "the bank" is a label, not a
  definition, and carries no evidence the usage was read.

Length is measured on the **longest disjunct, not the whole string**. The first
version scored the raw string at −0.05/−0.05 and the two halves cancelled: 83% of
hedged glosses in a live rollout batch had *every* disjunct under 3 words ("fame
or acclaim", "credit or discredit"), so deleting the `or` merely swapped one
penalty for the other. Net zero, a flat optimum, and the policy kept hedging on
43% of completions for the whole run. The disjunction has to cost strictly more
than the stub it hides behind — but not much more: it enters the shape span twice,
and −0.15 closes the accuracy gap exactly and fails the invariant test.

The term is silent when the answer does not parse. That is
`reward_wic_json`/`reward_wic_format`'s charge, and double-billing it would deepen
the shape floor for nothing.

---

## 6. Evaluate

```bash
uv run python src/eval_sense.py --model Qwen/Qwen3-1.7B                      # zero-shot baseline
uv run python src/eval_sense.py --model ./qwen-lora-grpo-wic --force-json    # trained policy
```

Decodes the held-out MCL-WiC test split (vLLM, continuous batching) and prints
accuracy plus same/different P/R/F1. Completions with no extractable verdict are
counted as `empty` and left out of P/R/F1. With `--force-json` the answer region
is constrained to schema-valid JSON via xgrammar — a two-phase decode, free
reasoning up to `</think>` then a grammar-guided continuation reusing the phase-1
KV cache — so `empty` drops to 0, including pairs whose reasoning overran the
budget. `--lora` serves an unmerged adapter on top of its base model.

Predictions are written in the teacher schema, so any eval run is also a
distillation run.

---

## File map

| File | Role |
|------|------|
| `analysis.ipynb` | Corpus assembly and the round-2 SFT set: explode, decontaminate, keep the correct samples |
| `src/sense_data.py` | Record loading, the shared WiC prompt, message building, answer parsing |
| `src/utils.py` | PTB detokenizer + `<t>` marker contract; `build_sft_dataset` / `build_grpo_dataset` |
| `src/prepare_lexeme.py` | The in-repo 29-language WiC merge (superseded by `../lexical_datasets`) |
| `src/semcor_pairs.py` | Gold-sense WiC pairs from SemCor (glosses + sense inventory) |
| `src/call_api.py` | Teacher self-consistency sampling (`-t wic`) and corpus triage (`-t quality`) |
| `src/filter_quality.py` | Prune the *pairs* on the triage rubric |
| `src/filter_reasoning.py` | Quality-filter the *traces* (rules + LLM judge) |
| `src/sft_sense.py` | SFT on a teacher-schema corpus |
| `src/sense_rewards.py` | The reward functions and the shape-span invariant |
| `src/grpo_lora.py` | GRPO (LoRA) on the verifiable label |
| `src/eval_sense.py` | Test-split accuracy/F1; emits teacher-schema predictions for the next round |
| `src/dm_gloss_data.py` | Builds gloss-eval records from the Ishiwatari et al. definition-modelling corpora |
| `run_train.sh` / `run_infer.sh` | Launchers for the trainer and the vLLM rollout server |

```bash
uv run pytest        # reward + corpus unit tests, CPU, seconds
```

### Known gaps

* **FEWS, MASC and SemCor are built but not yet in the sampling pool** — round 2
  has so far been sampled over `data/xl-lexeme.json` only. Folding them in means
  marking their targets and adding `source`/`lang` first.
* `src/gloss_wordnet.py` was deleted; `eval_gloss.py`, `eval_gloss_t5.py` and
  `gloss_arena.py` still import it, so the gloss-evaluation/arena side of the repo
  does not currently import. It sits outside the pipeline above.
* `sense_data.load_teacher_traces` only accepts the **exploded** notebook output —
  pointed at a raw teacher file (length-k lists) it fails rather than falling back.
  It also trusts the row's own verdict as the training label, which is correct only
  because the notebook already filtered `votes == label`.
* `run_train.sh` still calls the deleted `src/prepare_data.py`, and
  `tests/test_uopsd_vote.py` imports the deleted `sdpo_lora` — which fails
  collection, so `uv run pytest` currently errors before running anything.
