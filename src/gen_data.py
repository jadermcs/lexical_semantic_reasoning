"""Build a SemCor-grounded contrastive dataset for distilling ChatGPT reasoning.

Each example is an in-lexeme triplet of usages of one lemma:

    usage1, usage2  -> share the SAME WordNet sense   (definition_same)
    usage3          -> a DIFFERENT WordNet sense       (definition_diff)

Sense labels come from SemCor's gold annotations; the dictionary definitions come
from `wn` (Open English WordNet 2024), resolved from SemCor's WN3.0 sense keys.

The task we hand to ChatGPT (the distillation target, an SFT `<think>` trace) is:
explain *why* the supplied definition fits usages 1 & 2, and *why* usage 3 takes a
different sense. The resulting traces warm-start the RLVR policy described in
`ch05_implementation_plan.md` (Phase 3): the model emits `(tau, d_a, d_p, d_n)`,
here with `d_a == d_p == definition_same` and `d_n == definition_diff`.

Pure data prep: by default it only writes the assembled prompts. Pass `--annotate`
(needs OPENAI_API_KEY + the `openai` package) to call ChatGPT and emit finished
`<think>…</think><answer>…</answer>` SFT records.
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
    """For each lemma: (usage1, usage2) from one sense, usage3 from another."""
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
                "usage1": {"text": u1["text"], "marked": u1["marked"]},
                "usage2": {"text": u2["text"], "marked": u2["marked"]},
                "usage3": {"text": u3["text"], "marked": u3["marked"]},
            })
    return records


# --------------------------------------------------------------------------- #
# Prompt assembly + optional ChatGPT annotation
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are an expert lexicographer building the argument for a dictionary. You are "
    "given three usages of one target word (marked with <t> tags) and two candidate "
    "senses. Write a single piece of connected prose that argues, from the concrete "
    "contextual cues in each usage, toward a conclusion about which sense each usage "
    "carries. Begin with what the surrounding words reveal about the meaning in each "
    "usage, weigh that evidence, and let it build to the conclusion that two usages "
    "share one sense while the third diverges; end by stating that conclusion. Move "
    "forward from evidence to conclusion as a flowing argument, not as a list of "
    "points defending an answer given in advance. Bring out the genus the senses share "
    "and the differentia that separates them, and never use the target word to define "
    "itself."
)


def build_prompt(rec: dict) -> list[dict]:
    user = (
        f"Target word: {rec['lemma']} ({rec['pos']})\n\n"
        f"Usage 1: {rec['usage1']['marked']}\n"
        f"Usage 2: {rec['usage2']['marked']}\n"
        f"Usage 3: {rec['usage3']['marked']}\n\n"
        f"Sense A: {rec['definition_same']}\n"
        f"Sense B: {rec['definition_diff']}\n\n"
        "Make the case that usages 1 and 2 carry Sense A while usage 3 carries Sense B. "
        "Argue forward from the contextual cues in each usage to that conclusion, and "
        "close by stating which sense each usage takes."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def assemble_answer(rec: dict) -> str:
    return (
        "<answer>\n"
        f"1. {rec['definition_same']}\n"
        f"2. {rec['definition_same']}\n"
        f"3. {rec['definition_diff']}\n"
        "</answer>"
    )


def annotate(records: list[dict], model: str) -> None:
    """Call ChatGPT to fill each record's `think` trace (in place)."""
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
        argument = resp.choices[0].message.content.strip()
        rec["argument"] = argument
        rec["sft_target"] = f"<think>\n{argument}\n</think>\n{assemble_answer(rec)}"
        if i % 25 == 0:
            print(f"  annotated {i}/{len(records)}")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexicon", default="oewn:2024")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/semcor"))
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip")
    parser.add_argument("--max-per-lemma", type=int, default=4)
    parser.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/semcor_distill.jsonl"))
    parser.add_argument("--annotate", action="store_true", help="call ChatGPT for traces")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model for --annotate")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    en = wn.Wordnet(args.lexicon)

    data_p, key_p = ensure_semcor(args.cache_dir, keep_zip=args.keep_zip)
    gold = load_gold(key_p)
    print(f"Loaded {len(gold)} gold sense annotations.")

    groups = collect_groups(data_p, gold, en)
    print(f"Grouped usages for {len(groups)} (lemma, pos) keys.")

    records = build_triplets(groups, args.max_per_lemma, rng)
    rng.shuffle(records)
    if args.max_examples:
        records = records[: args.max_examples]
    print(f"Built {len(records)} contrastive triplets.")

    for rec in records:
        rec["prompt"] = build_prompt(rec)

    if args.annotate:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("--annotate needs OPENAI_API_KEY in the environment.")
        print(f"Annotating with {args.model} ...")
        annotate(records, args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.out}")

    if records:
        print("\n--- sample ---")
        s = records[0]
        print(f"[{s['lemma']} ({s['pos']})]  same={s['sense_same']}  diff={s['sense_diff']}")
        print(f"  u1: {s['usage1']['marked']}")
        print(f"  u2: {s['usage2']['marked']}")
        print(f"  u3: {s['usage3']['marked']}")
        print(f"  def(1,2): {s['definition_same']}")
        print(f"  def(3)  : {s['definition_diff']}")


if __name__ == "__main__":
    main()
