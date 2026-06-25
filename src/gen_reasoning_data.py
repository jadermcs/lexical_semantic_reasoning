"""Build SemCor-grounded datasets for distilling ChatGPT reasoning traces.

Two modes, matching the two sense-modeling ablation configs:

* ``triplet`` — an in-lexeme triplet of usages of one lemma:
      anchor, positive  -> share the SAME WordNet sense   (definition_same)
      negative          -> a DIFFERENT WordNet sense       (definition_diff)
  ChatGPT argues why the anchor/positive share a sense and why the negative
  differs; the trace warm-starts the contrastive policy. SFT target:
  `<think>τ</think><answer>1./2. d_same  3. d_diff</answer>`.

* ``direct`` — a single sense-tagged usage. ChatGPT argues, from the contextual
  cues, why the target word carries the supplied dictionary sense. SFT target:
  `<think>τ</think><answer>definition</answer>`.

Sense labels come from SemCor's gold annotations; the dictionary definitions come
from `wn` (Open English WordNet 2024), resolved from SemCor's WN3.0 sense keys.
These traces warm-start the RLVR policy in `ch05_implementation_plan.md` (Phase 3).

Pure data prep: by default it only writes the assembled prompts. Pass `--annotate`
(needs OPENAI_API_KEY + the `openai` package) to call ChatGPT and emit finished
`<think>…</think><answer>…</answer>` SFT records.

Quality control (annotate-only): empty/refused completions are always dropped, and
`--min-bertscore` filters traces whose reasoning is semantically far from the gold
definition it is meant to justify (BERTScore, baseline-rescaled; needs the
`bert-score` package: `uv add bert-score`).
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import wn

import sense_data as sd

WSDEVAL_URL = "http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip"
_BASE = "WSD_Evaluation_Framework/Training_Corpora/SemCor"
SEMCOR_DATA_MEMBER = f"{_BASE}/semcor.data.xml"
SEMCOR_KEY_MEMBER = f"{_BASE}/semcor.gold.key.txt"

# first digit of a WN sense key (the ss_type) -> human-readable POS
SS_TYPE_TO_POS = {"1": "noun", "2": "verb", "3": "adjective", "4": "adverb", "5": "adjective"}


def sense_key_to_oewn_id(sense_key: str) -> str:
    """WN3.0 sense key `lemma%a:b:c:d:e` -> OEWN sense id `oewn-lemma__a.b.c.d.e`."""
    return "oewn-" + sense_key.replace("%", "__").replace(":", ".")


# --------------------------------------------------------------------------- #
# SemCor download + parsing
# --------------------------------------------------------------------------- #
def ensure_semcor(cache_dir: Path, keep_zip: bool = False) -> tuple[Path, Path]:
    """Return paths to (data.xml, gold.key.txt), downloading/extracting if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_p = cache_dir / "semcor.data.xml"
    key_p = cache_dir / "semcor.gold.key.txt"
    if data_p.exists() and key_p.exists():
        return data_p, key_p

    zip_p = cache_dir / "WSD_Evaluation_Framework.zip"
    if not zip_p.exists():
        print(f"Downloading WSD Evaluation Framework (~165 MB) from {WSDEVAL_URL} ...")
        req = urllib.request.Request(WSDEVAL_URL, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_p, "wb") as out:
            shutil.copyfileobj(resp, out)

    print(f"Extracting SemCor from {zip_p.name} ...")
    with zipfile.ZipFile(zip_p) as z:
        for member, dest in [(SEMCOR_DATA_MEMBER, data_p), (SEMCOR_KEY_MEMBER, key_p)]:
            with z.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    if not keep_zip:
        zip_p.unlink()
    return data_p, key_p


def load_gold(key_p: Path) -> dict[str, str]:
    """instance_id -> first gold sense key."""
    gold = {}
    with open(key_p) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                gold[parts[0]] = parts[1]
    return gold


def mark(tokens: list[str], idx: int) -> str:
    out = list(tokens)
    out[idx] = f"<t> {out[idx]} </t>"
    return " ".join(out)


def iter_semcor_usages(data_p: Path, gold: dict[str, str]):
    """Yield one record per sense-tagged single-word instance."""
    for _event, sent in ET.iterparse(data_p, events=("end",)):
        if sent.tag != "sentence":
            continue
        tokens, instances = [], []
        for child in sent:
            surface = (child.text or "").strip()
            if child.tag == "instance":
                instances.append((len(tokens), child.get("id"), child.get("lemma")))
            tokens.append(surface)

        for idx, iid, lemma in instances:
            sense_key = gold.get(iid)
            if sense_key is None or not lemma or "_" in lemma:
                continue  # skip untagged and multiword instances
            yield {
                "lemma": lemma.lower(),
                "pos": SS_TYPE_TO_POS.get(sense_key.split("%")[1][0], "?"),
                "sense_key": sense_key,
                "text": " ".join(tokens),
                "marked": mark(tokens, idx),
            }
        sent.clear()


# --------------------------------------------------------------------------- #
# Grouping + triplet construction
# --------------------------------------------------------------------------- #
def collect_groups(data_p: Path, gold: dict[str, str], en: wn.Wordnet):
    """Group usages by (lemma, pos), attaching the resolved WordNet definition."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    def_cache: dict[str, str | None] = {}

    for u in iter_semcor_usages(data_p, gold):
        sk = u["sense_key"]
        if sk not in def_cache:
            try:
                definition = en.sense(sense_key_to_oewn_id(sk)).synset().definition()
            except wn.Error:
                definition = None
            def_cache[sk] = definition
        definition = def_cache[sk]
        if not definition:
            continue  # sense key did not resolve in OEWN 2024
        u["definition"] = definition
        groups[(u["lemma"], u["pos"])].append(u)
    return groups


def build_triplets(groups, max_per_lemma: int, rng: random.Random):
    """For each lemma: (anchor, positive) from one sense, negative from another."""
    records = []
    for (lemma, pos), usages in groups.items():
        by_sense: dict[str, list[dict]] = defaultdict(list)
        for u in usages:
            by_sense[u["sense_key"]].append(u)
        # senses with >=2 usages can supply the (anchor, positive) pair
        multi = [sk for sk, us in by_sense.items() if len(us) >= 2]
        if not multi or len(by_sense) < 2:
            continue

        per_lemma = []
        for same_sk in multi:
            same_us = by_sense[same_sk]
            for diff_sk, diff_us in by_sense.items():
                if diff_sk == same_sk:
                    continue
                u1, u2 = rng.sample(same_us, 2)
                u3 = rng.choice(diff_us)
                per_lemma.append((u1, u2, u3))
        rng.shuffle(per_lemma)
        for u1, u2, u3 in per_lemma[:max_per_lemma]:
            records.append({
                "lemma": lemma,
                "pos": pos,
                "sense_same": u1["sense_key"],
                "sense_diff": u3["sense_key"],
                "definition_same": u1["definition"],
                "definition_diff": u3["definition"],
                "anchor": {"text": u1["text"], "marked": u1["marked"]},
                "positive": {"text": u2["text"], "marked": u2["marked"]},
                "negative": {"text": u3["text"], "marked": u3["marked"]},
            })
    return records


def build_direct(groups, max_per_lemma: int, rng: random.Random):
    """One record per sense-tagged usage: (usage, its gold definition)."""
    records = []
    for (lemma, pos), usages in groups.items():
        pool = list(usages)
        rng.shuffle(pool)
        for u in pool[:max_per_lemma]:
            records.append({
                "lemma": lemma,
                "pos": pos,
                "sense_key": u["sense_key"],
                "definition": u["definition"],
                "usage": {"text": u["text"], "marked": u["marked"]},
            })
    return records


# --------------------------------------------------------------------------- #
# Prompt assembly + optional ChatGPT annotation
# --------------------------------------------------------------------------- #
# --- triplet (unchanged: this prompt was getting good results) ---
TRIPLET_SYSTEM_PROMPT = (
    "You are an expert lexicographer. You are given three usages of one target word "
    "(marked with <t> tags) — an anchor, a positive, and a negative — plus two "
    "candidate senses. Write a brief, tight argument (one short paragraph, no bullet "
    "lists) that: (1) briefly states what the word means in the anchor, positive, and "
    "negative usages, read from their context; (2) names the genus the anchor and "
    "positive share; (3) gives the differentia that sets the anchor and positive apart "
    "from the negative. Argue from the contextual cues and close by stating which sense "
    "each usage takes. Keep it concise — no padding or restating — and never use the "
    "target word to define itself."
)


def build_triplet_prompt(rec: dict) -> list[dict]:
    user = (
        f"Target word: {rec['lemma']} ({rec['pos']})\n\n"
        f"Anchor usage: {rec['anchor']['marked']}\n"
        f"Positive usage: {rec['positive']['marked']}\n"
        f"Negative usage: {rec['negative']['marked']}\n\n"
        f"Sense A: {rec['definition_same']}\n"
        f"Sense B: {rec['definition_diff']}\n\n"
        "Briefly describe the sense in the anchor, positive, and negative usages; give "
        "the genus the anchor and positive share and the differentia that separates them "
        "from the negative; then state that the anchor and positive carry Sense A and the "
        "negative carries Sense B."
    )
    return [
        {"role": "system", "content": TRIPLET_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def assemble_triplet_answer(rec: dict) -> str:
    return (
        "<answer>\n"
        f"1. {rec['definition_same']}\n"
        f"2. {rec['definition_same']}\n"
        f"3. {rec['definition_diff']}\n"
        "</answer>"
    )


# --- direct (single-usage reasoning, same concise/forward-argument style) ---
DIRECT_SYSTEM_PROMPT = (
    "You are an expert lexicographer. You are given one usage of a target word "
    "(marked with <t> tags) and the dictionary sense it carries there. Write a brief, "
    "tight argument (one short paragraph, no bullet lists) that reads the contextual "
    "cues in the usage, names the genus of the sense and the features that pin it down "
    "here, and arrives at that sense. Argue forward from the evidence to the sense; keep "
    "it concise — no padding or restating — and never use the target word to define itself."
)


def build_direct_prompt(rec: dict) -> list[dict]:
    user = (
        f"Target word: {rec['lemma']} ({rec['pos']})\n\n"
        f"Usage: {rec['usage']['marked']}\n\n"
        f"Sense: {rec['definition']}\n\n"
        "Briefly argue from the contextual cues why the target word carries this sense, "
        "and close by stating the sense."
    )
    return [
        {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def assemble_direct_answer(rec: dict) -> str:
    return f"<answer>\n{rec['definition']}\n</answer>"


# builder, prompt fn, answer fn per mode
MODES = {
    "triplet": (build_triplets, build_triplet_prompt, assemble_triplet_answer),
    "direct": (build_direct, build_direct_prompt, assemble_direct_answer),
}


def annotate(records: list[dict], model: str) -> None:
    """Call ChatGPT to fill each record's reasoning trace (in place).

    Each record must already carry ``prompt`` and ``sft_answer``; this fills
    ``argument`` and the assembled ``sft_target``.
    """
    from openai import OpenAI  # lazy import

    client = OpenAI()
    for i, rec in enumerate(records, 1):
        kwargs = {"model": model, "messages": rec["prompt"], "temperature": 0.7}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            # reasoning models (e.g. gpt-5-*) only allow the default temperature
            if "temperature" not in str(e):
                raise
            kwargs.pop("temperature")
            resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message.content
        argument = (msg or "").strip()
        rec["argument"] = argument
        if not argument:
            continue  # refusal / empty completion; dropped after the loop
        rec["sft_target"] = f"<think>\n{argument}\n</think>\n{rec['sft_answer']}"
        if i % 25 == 0:
            print(f"  annotated {i}/{len(records)}")


# the gold definition each mode's reasoning trace is meant to justify
DEF_KEY = {"triplet": "definition_same", "direct": "definition"}


def filter_by_bertscore(records, def_key, min_score, metric="f1", model_type=None):
    """Drop traces whose argument is semantically far from the gold definition.

    Scores each record's reasoning ``argument`` against the sense definition it is
    supposed to arrive at (BERTScore). Off-topic, wrong-sense, or punted traces
    land far from the gold gloss and fall below ``min_score``. Scores are
    baseline-rescaled (~0 for unrelated text, up to ~1) so the threshold is
    interpretable across runs; a starting value around 0.15 is reasonable.
    """
    from bert_score import score  # lazy import; needs the `bert-score` package

    cands = [r["argument"] for r in records]
    refs = [r[def_key] for r in records]
    kwargs = {"lang": "en", "rescale_with_baseline": True, "verbose": True}
    if model_type:  # baseline files only ship for the default models -> skip rescale
        kwargs.update(model_type=model_type, rescale_with_baseline=False)
    P, R, F1 = score(cands, refs, **kwargs)

    for r, p, rc, f in zip(records, P.tolist(), R.tolist(), F1.tolist()):
        r["bertscore"] = {"precision": p, "recall": rc, "f1": f}
    threshold = {"precision": P, "recall": R, "f1": F1}[metric].tolist()
    kept = [r for r, s in zip(records, threshold) if s >= min_score]
    print(f"  BERTScore {metric} filter (>= {min_score}): "
          f"kept {len(kept)}, dropped {len(records) - len(kept)}.")
    return kept


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["direct", "triplet", "both"], default="both")
    parser.add_argument("--lemma-split",
                        choices=["train", "non-eval", "dev", "test", "all"], default="non-eval",
                        help="restrict to lemmas in this split of sense_data. 'train' = strictly "
                        "the ablation's train lemmas; 'non-eval' = every lemma except dev/test "
                        "(leakage-safe and keeps SemCor lemmas with no WordNet example); "
                        "'all' disables filtering.")
    parser.add_argument("--split-seed", type=int, default=42, help="must match sense_data's seed")
    parser.add_argument("--lexicon", default="oewn:2024")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/semcor"))
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip")
    parser.add_argument("--max-per-lemma", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (single mode only; default data/semcor_distill_<mode>.jsonl)")
    parser.add_argument("--annotate", action="store_true", help="call ChatGPT for traces")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model for --annotate")
    parser.add_argument("--min-bertscore", type=float, default=0.0,
                        help="with --annotate, drop traces whose BERTScore vs. the gold "
                        "definition is below this (baseline-rescaled; try ~0.15). 0 = off.")
    parser.add_argument("--bertscore-metric", choices=["f1", "recall", "precision"],
                        default="f1", help="which BERTScore component to threshold; "
                        "'recall' rewards covering the gloss and is lenient on the "
                        "argument's extra reasoning")
    parser.add_argument("--bertscore-model", default=None,
                        help="override the BERTScore model (default: roberta-large via "
                        "lang=en, with baseline rescaling)")
    args = parser.parse_args()

    if args.annotate and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("--annotate needs OPENAI_API_KEY in the environment.")

    rng = random.Random(args.seed)
    en = wn.Wordnet(args.lexicon)

    data_p, key_p = ensure_semcor(args.cache_dir, keep_zip=args.keep_zip)
    gold = load_gold(key_p)
    print(f"Loaded {len(gold)} gold sense annotations.")

    groups = collect_groups(data_p, gold, en)
    print(f"Grouped usages for {len(groups)} (lemma, pos) keys.")

    if args.lemma_split != "all":
        train_l, dev_l, test_l = sd.lemma_splits(args.lexicon, args.split_seed)
        before = len(groups)
        if args.lemma_split == "non-eval":
            eval_l = dev_l | test_l
            groups = {k: v for k, v in groups.items() if k[0] not in eval_l}
        else:
            allowed = {"train": train_l, "dev": dev_l, "test": test_l}[args.lemma_split]
            groups = {k: v for k, v in groups.items() if k[0] in allowed}
        dropped = before - len(groups)
        print(f"Restricted to '{args.lemma_split}' lemmas: kept {len(groups)}, "
              f"dropped {dropped} (lemma, pos) groups.")

    modes = ["direct", "triplet"] if args.mode == "both" else [args.mode]
    if args.out and len(modes) > 1:
        raise SystemExit("--out only applies to a single --mode (not 'both').")

    for mode in modes:
        builder, prompt_fn, answer_fn = MODES[mode]
        records = builder(groups, args.max_per_lemma, rng)
        rng.shuffle(records)
        if args.max_examples:
            records = records[: args.max_examples]
        for rec in records:
            rec["prompt"] = prompt_fn(rec)
            rec["sft_answer"] = answer_fn(rec)
        print(f"[{mode}] built {len(records)} records.")

        if args.annotate:
            print(f"[{mode}] annotating with {args.model} ...")
            annotate(records, args.model)
            before = len(records)
            records = [r for r in records if r.get("argument")]
            if len(records) < before:
                print(f"  dropped {before - len(records)} empty/refused traces.")
            if args.min_bertscore > 0 and records:
                records = filter_by_bertscore(
                    records, DEF_KEY[mode], args.min_bertscore,
                    metric=args.bertscore_metric, model_type=args.bertscore_model)

        out = args.out or Path(f"data/semcor_distill_{mode}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{mode}] wrote {len(records)} records to {out}")

        if records:
            s = records[0]
            print(f"  sample [{s['lemma']} ({s['pos']})]")
            if mode == "triplet":
                print(f"    anchor:   {s['anchor']['marked']}")
                print(f"    negative: {s['negative']['marked']}")
            else:
                print(f"    usage: {s['usage']['marked']}")
                print(f"    sense: {s['definition']}")


if __name__ == "__main__":
    main()
