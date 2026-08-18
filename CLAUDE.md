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
uv run python src/call_api.py -f data/pairs.json -m deepseek/deepseek-v4-flash  # WiC teacher traces (needs OPENROUTER_API_KEY; -r resumes)
# corpus triage before distillation: rubric-grade each raw pair, then prune
uv run python src/call_api.py -f data/semcor.full.json -t quality            # k=1, one call per pair
uv run python src/filter_quality.py --preds predictions_semcor.full_*.jsonl \
    --data data/semcor.full.json --out data/semcor.clean.json
uv run python src/filter_reasoning.py --data data/mcl_semcor.json               # quality filter (stage 2 needs a GPU + vLLM)
uv run python src/prepare_data.py --data data/mcl_semcor_filtered.json --reasoning-select longest --out data/sft_wic  # → inspectable DatasetDict (+ .preview.jsonl)
uv run python src/sft_sense.py --data data/sft_wic
uv run python src/grpo_sense.py --model ./qwen-sense-sft-sft_wic
uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic --force-json
# gloss quality vs gold definitions (lemma/word/usage/definition records)
uv run --with sacrebleu python src/eval_gloss.py --model ./qwen-sense-grpo-wic --data data/gloss_eval.json
# BERTScore (the metric the DM literature + the FLAN-T5 card report). bert-score is
# installed unmanaged (`uv pip install --no-deps bert-score`) because `--with` pulls its
# own torch and breaks against the project's torchvision; hence --no-sync.
uv run --no-sync python src/eval_gloss.py --score-only preds.json --bertscore
# WordNet 3.0 for semcor_pairs.py — one-time, ~10MB into ~/.wn_data
uv run python -c "import wn; wn.download('omw-en:1.4')"
uv run python src/semcor_pairs.py            # → data/semcor_wic.json (detokenized)
# the 29-language WiC merge: chop to the target's sentence, detokenize, cap the tail
uv run python src/prepare_lexeme.py          # data/{train,dev}/*.data → data/xl-lexeme.json
# build the gloss-eval records from the definition-modelling corpora in data/{wordnet,oxford,slang,wiki}
uv run python src/dm_gloss_data.py --corpus wordnet oxford slang wiki
# open-inventory sense assignment: pick a listed WordNet sense, or coin the missing one
uv run python src/sense_assign_data.py                       # → data/sense_assign_semcor_{known,novel,mixed}.json
uv run python src/call_api.py -f data/sense_assign_semcor_mixed.json -t assign   # teacher baseline
uv run python src/eval_assign.py --model ./qwen-sense-grpo-wic \
    --data data/sense_assign_semcor_mixed.json --force-json
uv run python src/eval_assign.py --paired preds_known.json preds_novel.json      # the paired flip rate
```

The heavy stack (torch, trl, vLLM) runs on the servers via `run_train.sh` (GRPO trainer) and `run_infer.sh` (`trl vllm-serve` rollout server); data prep and tests run locally. Both scripts `uv sync` from the shared `uv.lock`, and torch/vLLM resolve from the cu130 wheel index pinned in `pyproject.toml` to keep CUDA ABIs compatible.

## Architecture

**The teacher schema is the interchange format of the whole repo.** One JSON record per WiC pair with `lemma/pos/sentence1/sentence2/label` plus per-sample `votes`, `answers`, `reasonings`, and a majority-vote `prediction`.

**`sentence1`/`sentence2` are stored already marked** — `<t> word </t>` is part of the record, and there is no separate `usage1`/`usage2`. The producers (`semcor_pairs.py`, `prepare_lexeme.py`) mark from exact character offsets, so nothing downstream re-derives it and no loader calls `sense_data.mark_target` on a WiC record — that function survives only for the gloss-eval corpora, whose targets are inline. Two consequences: the markers are part of `sense_data.pair_key`, which is what lets one sentence pair carry several items differing only in which occurrence is marked (a bare-sentence key reads 21,902 of them as contradictions), and prompts render `sentence1` directly. `call_api.py` produces it, `filter_reasoning.py` annotates it in place, `prepare_data.py` builds the SFT set from it, and `eval_sense.py` writes its own predictions back in it — that last fact is what closes the self-distillation loop (eval the trained policy on the train split, feed the output straight back to `--data` of `prepare_data.py`).

**Task-dispatched SFT.** `prepare_data.py` renders each record through a shared `sense_data.build_messages` dispatch (keyed by a per-record `task` tag) into one uniform conversational `{prompt, completion}` set, then splits it. One task today: **wic** (the pair sense-discrimination above). The dispatch stays so a task = a loader + a builder in `sense_data.py` and one line in `prepare_data.py`; `sft_sense.py` never changes because it only loads the prepared `DatasetDict` from disk.

**Corpus triage vs trace filtering.** `filter_reasoning.py` grades the *answer*; `call_api.py -t quality` → `filter_quality.py` grades the *question*, before any distillation is paid for. `QUALITY_AXES` holds the booleans (`well_formed1/2`, `target_ok1/2`) mapped to their *good* polarity; `QUALITY_SCALES` holds the ordinals (`evidence1/2` 1–3, `difficulty` 1–5), kept separate because they are thresholds the filter picks rather than verdicts the scorer reaches. The teacher answers `same_sense` last, so the verdict doubles as a label check. An axis that came back `None` means the teacher declined to grade it, never a rejection.

The metrics report each axis's **agreement gap** — teacher-vs-gold on the items it flags vs the ones it clears. An axis with no gap is finding nothing worth dropping however plausible it reads, and one with an *inverted* gap is actively wrong: on the first `semcor.full.json` build `well_formed1` fired on 58% of items and its flagged items agreed with gold *more* often (0.835 vs 0.734), because it was reporting the builder's character-window truncation rather than anything about the example. `evidence` is likewise a rewrite of a boolean `ambiguous` that fired 0/220 — a yes/no at "a careful reader could not tell which sense is meant" is a bar no real sentence clears, so it measured nothing. Runs at k=1 (`TASKS["quality"]["samples"]`): the axes are surface judgements, not the contested call `wic` votes on.

Pipeline flow: `call_api.py` (teacher self-consistency, k=3) → `filter_reasoning.py` (stage 1: CPU regex/consistency rules; stage 2: local Qwen3.5-9B judge on vLLM scoring english/coherent/consistent — no `faithful` axis and the gold label is withheld, because stage 1's vote check already settles faithfulness and leaking the label correlated the two axes) → `prepare_data.py` (keeps only pairs where teacher vote == gold; `--reasoning-select first|longest` is the ablation axis for which trace to keep; `--balance-labels` down-samples the majority same/different class to 50/50; splits, saves an inspectable dataset + `.preview.jsonl`) → `sft_sense.py` (loads the prepared dataset, SFT warm-start) → `grpo_sense.py` (GRPO on the verifiable label) → `eval_sense.py` (Qwen3 thinking-mode sampling: temp 0.6 / top-p 0.95 / top-k 20 / min-p 0; `--force-json` constrains the answer region with xgrammar after free `<think>` reasoning).

`src/sense_data.py` holds record loading, the shared prompts, per-task message builders, and answer parsing used by everything downstream. Scripts import siblings as top-level modules (`import sense_data`), so `pythonpath = ["src"]` in pyproject makes tests resolve the same way.

**Reward invariant** (`src/sense_rewards.py`, importable without torch/trl): the accuracy term (±1.0) dominates the shaping terms (JSON validity, format, think-length, gloss grounding), so being right always beats being tidy. A test in `tests/test_sense_rewards.py` pins this — don't rescale terms without checking it. Shaping spans 1.10 (ceiling +0.50, floor −0.60) against an accuracy gap of 1.5, so a new term has 0.4 of headroom before an existing one has to shrink.

**Gold-gloss data.** `src/semcor_pairs.py` turns `data/semcor_en.json.gz` into WiC pairs that carry a gold gloss per usage plus `senses`, the lemma's full WordNet 3.0 gloss inventory (`uv run python src/semcor_pairs.py` → `data/semcor_wic.json`; mix in with `grpo_lora.py --semcor-pairs`). The `reward_wic_gloss` term these fed is **gone** — `REWARDS` is accuracy, format, json, consistency, think-length. The gold glosses still matter to `eval_gloss.py`, which scores by *margin* (token-F1 against gold minus the best rival sense) rather than absolute similarity to gold: WordNet's wording is terse enough that a level punishes valid paraphrase. Everything stays inside WordNet 3.0; never compare a synset ID against `gloss_wordnet`'s OEWN ones, the ID spaces are disjoint.

WordNet 3.0 is read through the `wn` package's `omw-en:1.4` lexicon, **not** NLTK. SemCor stores Princeton's canonical synset names, which omw-en carries in `metadata()['identifier']`, so inverting that index (117,659 names, ~0.6s, `semcor_pairs._canonical`) resolves every name in the corpus. NLTK's per-lemma index does not: it misses 247 of them and silently returns a *different* synset for 386 more, all adjective satellites it numbers differently from `index.sense`. Two API traps when porting anything else off NLTK: `Synset.pos` is a property, and `len(a.shortest_path(b))` is NLTK's `shortest_path_distance` — `wn`'s path excludes the start node, so there is no `-1`. Multiword lemmas use spaces in `wn` and underscores in SemCor/NLTK.

**Detokenization.** PTB tokenization (`the league 's teams`) is a source tell — mixed into a rollout batch it says which corpus a pair came from before the policy has read a word. `src/utils.py` holds the one detokenizer and the `<t>`/`</t>` marker contract (`detokenize`, `target_of`); `semcor_pairs.mark_usage` runs it at build time so `data/semcor_wic.json` ships as prose, and `prepare_lexeme.py` runs it on `am2ico`. Keep shared helpers here rather than importing across corpus readers. It treats the markers as transparent so the span survives byte-exact, and re-pads them before splitting so it is idempotent on its own output — without that, its own `</t>.` reads back as one opaque token and the target is lost. Unbalanced quotes are the residual (0.3% of usages); `tests/test_utils.py` pins the rest.

**The 29-language WiC merge.** `src/prepare_lexeme.py` folds `data/{train,dev}/*.data` — `wic`, `mcl-wic`, `xl`, `am2ico` — into `data/xl-lexeme.json` (272,440 pairs, 50/50). One corpus is written unlike the other three: `am2ico` takes Wikipedia *paragraphs* (2.06 sentences, 85.5 tokens each vs 8.9–30.8) and is PTB-tokenized in 99.9% of them. Both are tells before they are costs, so every corpus is normalized to one shape — chop to the sentence holding the target (−38% tokens on `am2ico`, ~0 elsewhere; the span never straddled a boundary in 545,736 sentences), detokenize, then a token window for the 153 sentences still over 256. Budgets are in **tokens, never characters**: chars/token runs 4.73 (en-en) to 1.22 (zh-zh), so a character cut costs CJK/Georgian/Bengali ~4× what it costs English. Sentence splitting is multi-script and whitespace is language-aware — `zh`/`ja` contexts have no word spaces, so their only spaces are am2ico's padding around the target and get deleted. `pair_key` here keys on the *marked* usages, not the bare sentences: one sentence pair can hold several items differing only in which occurrence is marked, and `sense_data.pair_key`'s bare-sentence key reads 21,902 of them as contradictions. Output carries `split`/`source`/`lang1`/`lang2`/`origin`, because `data/train/` holds 66 `dev.*.data` files next to its 32 `train.*` ones and the directory — not the filename — is what says which split a pair is in.

**Out-of-domain gloss evaluation.** `data/{wordnet,oxford,slang,wiki}` hold the Ishiwatari et al. (2019) definition-modelling release: line-aligned `<split>.txt` (key/pos/source/definition) and `<split>.eg` (context, target replaced by `<TRG>` — except `oxford`, which leaves it inline for `mark_target` to find). `src/dm_gloss_data.py` converts them to `eval_gloss.py` records, sampling whole lemmas so `sense_acc` has polysemy to score. Only `wordnet` ships a real POS; the others get a WordNet-3.0 lemma prior (never the gold definition — that would leak into the prompt), falling back to `--default-pos`.

**Paired-usage gloss eval + the arena.** `eval_gloss.py` puts a constant filler in sentence 2 by default; a record carrying `usage2` replaces it with a real second usage, which makes sentence 2 an ablation axis instead of dead weight (`src/semcor_gloss_data.py` emits `single`/`same`/`diff` sets over one identical item set). Records carrying `senses` also get `inv_acc`/`inv_margin`, ranking the gloss against the lemma's real WordNet inventory rather than whatever else shares a lemma in the file; paired records get `agree` (F1 between the two emitted glosses — a *leak* signal in filler mode, a *consistency* signal here) and `verdict_acc`. Because BLEU/chrF against one reference reward the reference dictionary's register — a definition of the *wrong sense of the same lemma* outscores real models on 3 of the 4 corpora — `src/gloss_arena.py` is the trustworthy measurement: pairwise LLM judging, reference-free, both orders, fit with Bradley-Terry plus LMArena-style style covariates. It enters `gold` and `rival` as anonymous competitors to audit the judge. See `analysis/gloss_dm/summary.md` and `analysis/gloss_pair/summary.md`.

Model output contract: `<think>...</think>` followed by `{"sense1": ..., "sense2": ..., "same_sense": bool}`. Only `same_sense` is scored; the glosses force per-usage commitment.

## Conventions

- `prepare_data.py` names its output `data/sft_<tasks>-<strategy>-<data-stem>` (override with `--out`); `sft_sense.py` defaults its output dir to `./qwen-sense-sft-<prepared-stem>` and resumes from the latest checkpoint there; wandb logging is on by default.
- For constrained/structured generation use xgrammar, not lm-format-enforcer.
- `data/mcl-wic.test.json` is held out for evaluation only — never train on it.
