# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Word-in-context (WiC) sense discrimination via RLVR: a small Qwen3 policy reasons about a target word's meaning in two sentences, then emits a JSON verdict on whether the senses match. The verdict is a gold label, so the RL reward is verifiable. See README.md for the full pipeline write-up.

**The method is two rounds.** Round 1 buys a *trace generator*; round 2 trains the model that ships.

1. Distil `deepseek-v4-flash` into Qwen3-1.7B (SFT warm start).
2. GRPO that warm start on the verifiable label — what this round actually buys is prompt-format adherence and reasoning traces that are long enough to be worth distilling.
3. Sample reasoning traces with *that* policy over the pair corpora — `data/xl-lexeme.json` today, with FEWS/MASC/SemCor built next door and not yet folded in — keeping the samples it gets right (`analysis.ipynb`).
4. SFT **from the base model again, not from the round-1 checkpoint** on that self-distilled corpus.
5. GRPO again, this time with the gloss-form term carrying weight: no bare-`or` disjunctions, no glosses under three committed words.

The round-1 checkpoint is scaffolding. Nothing downstream inherits its weights — only its traces — so a change to round 1 is evaluated by the corpus it produces, not by its own test score.

## Commands

Always run Python through uv (`uv run python ...`), never bare `python`/`python3`.

```bash
uv run pytest                                # reward unit tests (CPU, fast)
uv run pytest tests/test_sense_rewards.py -k <name>   # single test

# --- corpus construction: the sibling repo ../lexical_datasets, then analysis.ipynb ---
# ../lexical_datasets/data_preprocess.py builds fews / masc / semcor / dwug / chainnet
# and xl-lexeme.{train,dev}.json; analysis.ipynb concatenates the two xl-lexeme splits
# into data/xl-lexeme.json (272,493 pairs, already <t>-marked).
# In-repo builders that predate that split (their inputs no longer live here):
uv run python -c "import wn; wn.download('omw-en:1.4')"   # one-time, ~10MB into ~/.wn_data
uv run python src/semcor_pairs.py            # → data/semcor_wic.json (detokenized, gold glosses)
uv run python src/prepare_lexeme.py          # data/{train,dev}/*.data → data/xl-lexeme.json

# --- round 1: teacher distillation ---
uv run python src/call_api.py -f data/pairs.json -m deepseek/deepseek-v4-flash  # k=3, needs OPENROUTER_API_KEY; -r resumes
# corpus triage before distillation: rubric-grade each raw pair, then prune
uv run python src/call_api.py -f data/semcor.full.json -t quality              # k=1, one call per pair
uv run python src/filter_quality.py --preds predictions_semcor.full_*.jsonl \
    --data data/semcor.full.json --out data/semcor.clean.json
# trace filtering (stage 2 needs a GPU + vLLM)
uv run python src/filter_reasoning.py --data data/mcl_train_dev.json \
    --out data/mcl_train_dev_scored.json --emit-filtered data/mcl_train_dev_filtered.json
uv run python src/sft_sense.py --model Qwen/Qwen3-1.7B --file_path data/mcl_train_dev_filtered.json
uv run python src/grpo_lora.py --model ./qwen-mcl_train_dev_filtered --file-path data/xl-lexeme.json

# --- round 2: self-distillation, then train from scratch on it ---
uv run python src/eval_sense.py --model ./qwen-lora-grpo-wic --force-json \
    --path data/xl-lexeme.json -k 3 --output predictions_sense_wic_xl-lexeme.json
# then analysis.ipynb: explode → drop test lemmas → drop pos-less rows → keep vote==gold
#   → data/lexeme.correct.json
uv run python src/sft_sense.py --model Qwen/Qwen3-1.7B --file_path data/lexeme.correct.json
uv run python src/grpo_lora.py --model ./qwen-lexeme.correct --file-path data/xl-lexeme.json

uv run python src/eval_sense.py --model ./qwen-lora-grpo-wic --force-json  # held-out test split
```

The heavy stack (torch, trl, vLLM) runs on the servers via `run_train.sh` (trainer) and `run_infer.sh` (`trl vllm-serve` rollout server); data prep and tests run locally. Both scripts `uv sync` from the shared `uv.lock`, and torch/vLLM resolve from the cu130 wheel index pinned in `pyproject.toml` to keep CUDA ABIs compatible.

## Architecture

**The teacher schema is the interchange format of the whole repo.** One JSON record per WiC pair with `lemma/pos/sentence1/sentence2/label` plus per-sample `votes`, `answers`, `reasonings`, and a majority-vote `prediction`. `call_api.py` produces it, `filter_reasoning.py` annotates it in place, `sense_data.load_teacher_traces` reads it, and `eval_sense.py` writes its own predictions back in it — that last fact is what makes step 3 above need no glue code. Every field of an input record that isn't an output key rides through `eval_sense.py`, so `source`/`lang1`/`senses`/`split` survive into the self-distilled file and the corpus can still be stratified by where its pairs came from.

**`-k` is what makes the self-distillation round worth running.** At k=1 (greedy) each pair yields one trace, `prediction` is that trace, and a lucky-but-wrong chain is kept whenever its verdict happens to match gold. At k>1 the pairs decode with Qwen3 thinking-mode sampling, `prediction` becomes a majority vote (ties → `None`), and the k samples become k independent distillation candidates. `answers`, `reasonings` and `votes` are kept index-aligned and always length k — a sample that lost only its JSON still holds its slot, or the next sample's verdict would be paired with this one's reasoning. That alignment is what lets `analysis.ipynb` explode the three columns together.

**Round-2 corpus prep is `analysis.ipynb`, not a script**, and it is where the round-2 SFT set is actually defined. Four steps on the `eval_sense.py` output, in order:

1. `explode(["reasonings", "answers", "votes"])` — **one row per sample, not per pair**. This is why `sense_data.load_teacher_traces` reads `answers`/`reasonings` as scalars rather than lists: it is fed the exploded file, not a raw teacher file. A k=3 run over `xl-lexeme` explodes 272,493 pairs into 792,696 rows (short of 3× because empty slots are already gone), then shuffles at `random_state=42`.
2. **Test-lemma decontamination** — drop every row whose lowercased `lemma` appears in `data/mcl-wic.test.json`. Held-out accuracy otherwise measures recall of a sense inventory that was trained on. Do this on lemmas, not pairs: the same lemma reappears across corpora with different sentences.
3. `dropna` on `pos` — the cross-lingual `am2ico` rows carry `lang2: en` and no POS, and the prompt renders `pos` into its first line.
4. **Keep rows where the sample's own `votes` entry equals gold `label`**, not where `prediction` does. Post-explode, `prediction` is still the pair's majority vote, so filtering on it would keep the minority samples that voted against it. → `data/lexeme.correct.json`.

The same notebook mines the complement for hard cases: rows of `predictions_sense_wic_mcl_train_dev.json` where `label != prediction` → `data/mcl-hard.json` (pairs the policy gets wrong at the pair level, a rollout pool rather than an SFT set).

**`sentence1`/`sentence2` are stored already marked** — `<t> word </t>` is part of the record, and there is no separate `usage1`/`usage2`. The producers (`semcor_pairs.py`, `prepare_lexeme.py`) mark from exact character offsets, so nothing downstream re-derives it and no loader calls `sense_data.mark_target` on a WiC record — that function survives only for the gloss-eval corpora, whose targets are inline. Two consequences: the markers are part of `sense_data.pair_key`, which is what lets one sentence pair carry several items differing only in which occurrence is marked (a bare-sentence key reads 21,902 of them as contradictions), and prompts render `sentence1` directly.

**Dataset building lives in `src/utils.py`, not a prepare step.** `build_sft_dataset` runs a teacher-schema file through `load_teacher_traces` and renders `{prompt, completion}` rows; `build_grpo_dataset` renders the prompt alone and keeps `KEEP_COLS` (`lemma`/`pos`/`label`) beside it, because GRPO scores its own rollouts — the label is a reward input, not a target. Both split with a fixed seed. There is no `prepare_data.py` and no on-disk `DatasetDict` stage any more: `sft_sense.py --file_path` and `grpo_lora.py --file-path` take the corpus JSON directly, so a corpus change is one flag, not a rebuild.

**Corpus triage vs trace filtering.** `filter_reasoning.py` grades the *answer*; `call_api.py -t quality` → `filter_quality.py` grades the *question*, before any distillation is paid for. `QUALITY_AXES` holds the booleans (`well_formed1/2`, `target_ok1/2`) mapped to their *good* polarity; `QUALITY_SCALES` holds the ordinals (`evidence1/2` 1–3, `difficulty` 1–5), kept separate because they are thresholds the filter picks rather than verdicts the scorer reaches. The teacher answers `same_sense` last, so the verdict doubles as a label check. An axis that came back `None` means the teacher declined to grade it, never a rejection.

The metrics report each axis's **agreement gap** — teacher-vs-gold on the items it flags vs the ones it clears. An axis with no gap is finding nothing worth dropping however plausible it reads, and one with an *inverted* gap is actively wrong: on the first `semcor.full.json` build `well_formed1` fired on 58% of items and its flagged items agreed with gold *more* often (0.835 vs 0.734), because it was reporting the builder's character-window truncation rather than anything about the example. `evidence` is likewise a rewrite of a boolean `ambiguous` that fired 0/220 — a yes/no at "a careful reader could not tell which sense is meant" is a bar no real sentence clears, so it measured nothing. Runs at k=1 (`TASKS["quality"]["samples"]`): the axes are surface judgements, not the contested call `wic` votes on.

**Trace filtering** (`filter_reasoning.py`) is two stages. Stage 1 (CPU, free) is regex/statistical rules — null, non-Latin script, stub, blowup, repetition loop — plus a vote rule that drops every slot whose own `votes[j]` disagrees with gold, and a gloss rule for glosses that contradict the label. Roughly half of all slots die here and never reach the GPU. Stage 2 is a local Qwen3.5-9B (AWQ int4) judge on vLLM scoring `english`/`coherent`/`consistent`. There is deliberately **no `faithful` axis and the judge is not shown the gold label**: stage 1's vote check settles faithfulness exactly and for free, and over 20,804 judgements the axis fired 346 times — all judge errors — dragging `consistent` with it (345 shared slots) because the label sat in the prompt above both questions. The script annotates rather than deletes; `--emit-filtered` writes the pruned corpus.

**Reward invariant** (`src/sense_rewards.py`, importable without torch/trl): the accuracy term (±1.0) dominates the shaping terms (JSON validity, format, consistency, gloss form, think-length), so being right always beats being tidy. A test in `tests/test_sense_rewards.py` pins this — don't rescale terms without checking it. Shaping spans 1.44 (ceiling +0.50, floor −0.94) against an accuracy gap of 1.5, so there is 0.06 of headroom before an existing term has to shrink.

**`reward_wic_gloss_form` is round 2's point.** It shapes the *shape* of a gloss, not its content: −0.12 per gloss that hedges with a standalone `or` (a disjunction commits to neither sense yet still passes the consistency term) and −0.05 per gloss under `GLOSS_MIN_WORDS` (3) tokens. **Length is measured on the longest disjunct, not the whole string** (`_committed_len`), and the disjunction costs more than the stub — both because the first version scored the raw string at −0.05/−0.05 and the two halves cancelled: 83% of hedged glosses in a live rollout batch had every disjunct under 3 words ("fame or acclaim"), so deleting the `or` just swapped one penalty for the other. Zero gradient, and the policy hedged on 43% of completions through the whole run. The two constants are bounded above by the shape-span invariant, which the disjunction penalty enters twice (once per gloss): −0.15 closes the accuracy gap exactly and fails the test. It is silent when the answer does not parse — that is the json/format terms' charge, and double-billing it would deepen the shape floor for nothing.

`REWARDS` is accuracy, format, json, consistency, gloss-form, think-length. `grpo_lora.py` adds TRL's repetition penalty and soft overlong punishment on top. The old `reward_wic_gloss` term that scored gloss *content* against gold WordNet glosses is gone.

`src/sense_data.py` holds record loading, the shared WiC prompt, message building, and answer parsing used by everything downstream. Scripts import siblings as top-level modules (`import sense_data`), so `pythonpath = ["src"]` in pyproject makes tests resolve the same way.

**GRPO objective** (`src/grpo_lora.py`): LoRA (r=32, all-linear) over a merged policy — it refuses an adapter dir at `--model`. Defaults are `dr_grpo` loss, `scale_rewards=none`, DAPO clip-higher (0.2/0.28), `beta=0`, and rollout temperature **1.0 with top-k off**. Qwen3's published 0.6/top-k-20 recipe is an *inference* setting; used for rollouts it caps diversity and manufactures unanimous groups with zero advantage. `--vllm-server-host` offloads rollouts to `run_infer.sh`'s server; without it vLLM runs colocated.

**Gold-gloss data.** `src/semcor_pairs.py` turns `data/semcor_en.json.gz` into WiC pairs carrying a gold gloss per usage plus `senses`, the lemma's full WordNet 3.0 gloss inventory. Everything stays inside WordNet 3.0, read through the `wn` package's `omw-en:1.4` lexicon, **not** NLTK. SemCor stores Princeton's canonical synset names, which omw-en carries in `metadata()['identifier']`, so inverting that index (117,659 names, ~0.6s, `semcor_pairs._canonical`) resolves every name in the corpus. NLTK's per-lemma index does not: it misses 247 of them and silently returns a *different* synset for 386 more, all adjective satellites it numbers differently from `index.sense`. Two API traps when porting anything else off NLTK: `Synset.pos` is a property, and `len(a.shortest_path(b))` is NLTK's `shortest_path_distance` — `wn`'s path excludes the start node, so there is no `-1`. Multiword lemmas use spaces in `wn` and underscores in SemCor/NLTK.

**Detokenization.** PTB tokenization (`the league 's teams`) is a source tell — mixed into a rollout batch it says which corpus a pair came from before the policy has read a word. `src/utils.py` holds the one detokenizer and the `<t>`/`</t>` marker contract (`detokenize`, `target_of`); `semcor_pairs.mark_usage` runs it at build time so `data/semcor_wic.json` ships as prose, and `prepare_lexeme.py` runs it on `am2ico`. Keep shared helpers here rather than importing across corpus readers. It treats the markers as transparent so the span survives byte-exact, and re-pads them before splitting so it is idempotent on its own output — without that, its own `</t>.` reads back as one opaque token and the target is lost. Unbalanced quotes are the residual (0.3% of usages); `tests/test_utils.py` pins the rest.

**Corpora come from the sibling repo `../lexical_datasets`.** Its `data_preprocess.py` builds `fews`, `masc`, `semcor`, `dwug` and `chainnet` (train/dev/test + a `.full.json`, grouped splits at `random_state=42`) and the `xl-lexeme.{train,dev}.json` merge. `analysis.ipynb` concatenates the two xl-lexeme splits into `data/xl-lexeme.json`. **The two families are not the same shape**: `xl-lexeme` ships `<t>`-marked with `task`/`id`/`split`/`origin`/`source`/`lang1`/`lang2`, while `fews`/`masc`/`semcor`/`dwug` ship *unmarked* `lemma`/`pos`/`label`/`word1`/`word2`/`sentence1`/`sentence2` with an int label — they need marking (`word1`/`word2` are the surface forms) and carry no source/lang fields before they can join a rollout batch. `masc.train.json` is empty by construction; MASC is a dev/test-only corpus.

**The 29-language WiC merge.** `data/xl-lexeme.json` is 272,493 pairs (`wic`, `mcl-wic`, `xl`, `am2ico`) and is the default rollout corpus for `grpo_lora.py`. `src/prepare_lexeme.py` is the in-repo builder that predates the move to `../lexical_datasets`; its input `data/{train,dev}/*.data` is no longer in this repo, but the normalization it documents is what the pairs went through. One corpus is written unlike the other three: `am2ico` takes Wikipedia *paragraphs* (2.06 sentences, 85.5 tokens each vs 8.9–30.8) and is PTB-tokenized in 99.9% of them. Both are tells before they are costs, so every corpus is normalized to one shape — chop to the sentence holding the target (−38% tokens on `am2ico`, ~0 elsewhere; the span never straddled a boundary in 545,736 sentences), detokenize, then a token window for the 153 sentences still over 256. Budgets are in **tokens, never characters**: chars/token runs 4.73 (en-en) to 1.22 (zh-zh), so a character cut costs CJK/Georgian/Bengali ~4× what it costs English. Sentence splitting is multi-script and whitespace is language-aware — `zh`/`ja` contexts have no word spaces, so their only spaces are am2ico's padding around the target and get deleted. `pair_key` here keys on the *marked* usages. Output carries `split`/`source`/`lang1`/`lang2`/`origin`, because `data/train/` holds 66 `dev.*.data` files next to its 32 `train.*` ones and the directory — not the filename — is what says which split a pair is in.

Model output contract: `<think>...</think>` followed by `{"sense1": ..., "sense2": ..., "same_sense": bool}`. Only `same_sense` is scored; the glosses force per-usage commitment, and their *form* is shaped by `reward_wic_gloss_form`.

## Known gaps

These are real, in the tree right now, and not to be papered over in docs:

- **FEWS/MASC/SemCor are built but not yet folded into the sampling pool.** `../lexical_datasets` produces them; `data/xl-lexeme.json` (mcl-wic/wic/xl/am2ico) is what round 2 has actually been sampled over. Folding them in means marking the targets and adding the `source`/`lang` fields first — see the shape mismatch above.
- **`sense_data.mark_target` is the only marker left in this repo**, and it is fuzzy (`rapidfuzz`, threshold 70) with an append-on-failure fallback. The sibling repo marks from character offsets. Prefer offsets when adding a corpus; a fuzzy mark that lands on the wrong token silently changes the question.
- **`src/gloss_wordnet.py` was deleted** but is still imported by `src/eval_gloss.py`, `src/eval_gloss_t5.py`, `src/gloss_arena.py` and the WordNet-lexicon check in `run_train.sh`. The whole gloss-evaluation/arena side of the repo does not import as it stands; it is out of the pipeline above, and the reward that depended on it is gone.
- **`sense_data.load_teacher_traces` only accepts an exploded file** — it does `json.loads(r["answers"])` on what `call_api.py` / `filter_reasoning.py` / `eval_sense.py` write as length-k *lists*. That is by design for the `analysis.ipynb` output, but it means pointing `sft_sense.py --file_path` at a raw teacher file fails rather than falling back. It also takes the row's own `same_sense` as the training label (correct only because the notebook already filtered `votes == label`) and ignores its `strategy` argument, which is now the notebook's job.
- **`run_train.sh` still calls `src/prepare_data.py`**, which no longer exists, and `tests/test_uopsd_vote.py` imports the deleted `sdpo_lora` — which fails collection, so `uv run pytest` errors out before running anything.
- `call_api.py -t assign` imports the deleted `eval_assign`; the `assign` task and its data builder are gone from the pipeline.

## Conventions

- `sft_sense.py` names its output `./qwen-<data-stem>` (override with `--output-dir`) and loads the best checkpoint at end of training; `grpo_lora.py` writes `./<run-name>` (default `qwen-lora-grpo-wic`) and resumes with `--resume`. MLflow logging is on by default, each script defaulting to its own experiment (`wic-sft` / `wic-grpo`) — `run_train.sh` deliberately does not export `MLFLOW_EXPERIMENT_NAME`, which would win over all of them.
- Tracking is MLflow (`mlflow-skinny`, not full `mlflow` — the full package caps `pandas<3` and drags the resolver back to mlflow 1.27, which imports `google.protobuf.service` and dies on protobuf>=4). MLflow 3.15 put the `./mlruns` file store in maintenance mode, so `MLFLOW_TRACKING_URI` must be a database URI — `run_train.sh` exports `sqlite:///$PWD/mlflow.db`, which is why `sqlalchemy`/`alembic` are dependencies. View with `uvx mlflow ui --backend-store-uri sqlite:///mlflow.db`.
- For constrained/structured generation use xgrammar, not lm-format-enforcer.
- `data/mcl-wic.test.json` is held out for evaluation only — never train on it.
