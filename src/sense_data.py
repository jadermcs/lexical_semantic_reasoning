"""Shared data + prompt + metric utilities for the sense-modeling ablations.

Two task framings over the *same* WordNet usages, with a **lemma-disjoint**
train/dev/test split (no lemma crosses splits):

* ``direct``  — given one usage (target marked with <t> tags), generate the
  WordNet gloss for that sense. Single-usage definition modeling.
* ``triplet`` — given an anchor + positive (same synset) and a negative
  (different synset of the same lemma), generate the contrastive reasoning and a
  *single* gloss for the sense shared by the anchor and positive. The negative is
  used only to sharpen the reasoning (the differentia) — no gloss is written for
  it. Mirrors the (u_a, u_p, u_n) format in ``ch05_implementation_plan.md``.

Gold glosses are WordNet definitions; usages are synset example sentences.
Evaluation metric is BLEU of the generated gloss against the gold gloss
(for ``triplet`` this is the shared anchor/positive gloss).

This module imports neither torch nor trl, so the dataset can be built and
inspected on the laptop; the training scripts import the heavy stack.
"""

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import wn
from rapidfuzz import fuzz, process

POS_MAP = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}

DATA_DIR = Path("data")

# --------------------------------------------------------------------------- #
# Prompts (shared by SFT / GRPO / eval so train and inference match exactly)
# --------------------------------------------------------------------------- #
DIRECT_SYSTEM = (
    "You are an expert lexicographer. You are given a sentence with one target "
    "word marked by <t> tags. Inside <think> tags, write a brief, tight argument "
    "(one short paragraph, no bullet lists) that reads the contextual cues and works "
    "out what the target word means here. Reason forward from the context only. Then, "
    "after </think>, give a single concise dictionary definition of the target word as "
    "used here — and nothing else. Keep it concise and never use the target word to "
    "define itself. Format: <think>...</think>\ndefinition"
)

TRIPLET_SYSTEM = (
    "You are an expert lexicographer. You are given three usages of one target word "
    "(marked with <t> tags) — an anchor, a positive, and a negative. The anchor and "
    "positive share one sense; the negative is a different sense. Inside "
    "<think> tags, state what the word means in each usage, name the "
    "genus the anchor and positive share, and give the differentia that sets them "
    "apart from the negative. Then, inside <answer> tags, give one concise "
    "dictionary definition — for the single sense shared by the anchor and positive — "
    "without using the target word or the sentence to define itself. Do not define the "
    "negative sense. Format: <think>...</think><answer>\n...\n</answer>"
)


# --------------------------------------------------------------------------- #
# Target marking
# --------------------------------------------------------------------------- #
def mark_target(sentence: str, word: str, fuzzy_threshold: float = 70.0) -> str:
    """Wrap the best match for *word* in *sentence* with <t> tags (handles inflection)."""
    pattern = rf"\b({re.escape(word)}\w*)\b"
    if re.search(pattern, sentence, flags=re.IGNORECASE):
        return re.sub(pattern, r"<t> \1 </t>", sentence, count=1, flags=re.IGNORECASE)
    tokens = re.findall(r"\w+", sentence)
    if tokens:
        match = process.extractOne(
            word.lower(),
            [t.lower() for t in tokens],
            scorer=fuzz.QRatio,
            score_cutoff=fuzzy_threshold,
        )
        if match is not None:
            best = tokens[match[2]]
            return re.sub(
                rf"\b{re.escape(best)}\b", f"<t> {best} </t>", sentence, count=1
            )
    return sentence + f" <t> {word} </t>"


# --------------------------------------------------------------------------- #
# Dataset construction
# --------------------------------------------------------------------------- #
def _load_wordnet(lexicon: str) -> wn.Wordnet:
    try:
        return wn.Wordnet(lexicon)
    except wn.Error:
        wn.download(lexicon)
        return wn.Wordnet(lexicon)


def _lemma_pool(en: wn.Wordnet) -> dict[tuple[str, str], dict[str, list]]:
    """(lemma, pos) -> {synset_id: [(synset, example), ...]} using example sentences."""
    pool: dict[tuple[str, str], dict[str, list]] = {}
    for syn in en.synsets():
        examples = syn.examples()
        if not examples:
            continue
        pos = POS_MAP.get(syn.pos, syn.pos)
        for lemma in syn.lemmas():
            if " " in lemma or "_" in lemma:
                continue
            key = (lemma.lower(), pos)
            uses = [ex for ex in examples if lemma.lower() in ex.lower()]
            if uses:
                pool.setdefault(key, {}).setdefault(syn.id, []).extend(
                    (syn, ex) for ex in uses
                )
    return pool


def _split_lemmas(lemmas: list[str], rng: random.Random, dev=0.1, test=0.1):
    """Lemma-disjoint split: a lemma string lands in exactly one split."""
    uniq = sorted(set(lemmas))
    rng.shuffle(uniq)
    n = len(uniq)
    n_test = int(n * test)
    n_dev = int(n * dev)
    test_l = set(uniq[:n_test])
    dev_l = set(uniq[n_test : n_test + n_dev])
    train_l = set(uniq[n_test + n_dev :])
    return train_l, dev_l, test_l


def lemma_splits(lexicon="oewn:2024", seed=42, dev=0.1, test=0.1):
    """Reproduce the lemma-disjoint split as (train, dev, test) sets of lemma strings.

    Matches ``build_dataset``'s assignment for the same lexicon/seed, so other
    scripts (e.g. the SemCor distillation builder) can restrict themselves to the
    train lemmas and avoid leaking dev/test lemmas into training.
    """
    en = _load_wordnet(lexicon)
    rng = random.Random(seed)
    pool = _lemma_pool(en)
    return _split_lemmas([lem for lem, _ in pool], rng, dev, test)


def build_dataset(lexicon="oewn:2024", max_per_lemma=4, seed=42):
    """Return {split: {'direct': [...], 'triplet': [...]}} with disjoint lemmas."""
    en = _load_wordnet(lexicon)
    rng = random.Random(seed)
    pool = _lemma_pool(en)

    train_l, dev_l, test_l = _split_lemmas([lem for lem, _ in pool], rng)
    which = {"train": train_l, "dev": dev_l, "test": test_l}

    out = {s: {"direct": [], "triplet": []} for s in which}
    for (lemma, pos), by_syn in pool.items():
        split = next(s for s, ls in which.items() if lemma in ls)

        # --- direct: one record per usage ---
        usages = [(s, ex) for pairs in by_syn.values() for (s, ex) in pairs]
        rng.shuffle(usages)
        for syn, ex in usages[:max_per_lemma]:
            out[split]["direct"].append(
                {
                    "lemma": lemma,
                    "pos": pos,
                    "synset": syn.id,
                    "usage": mark_target(ex, lemma),
                    "gloss": syn.definition(),
                }
            )

        # --- triplet: needs a synset with >=2 usages + another synset ---
        if len(by_syn) < 2:
            continue
        multi = [sid for sid, p in by_syn.items() if len(p) >= 2]
        if not multi:
            continue
        triplets = []
        for same_sid in multi:
            same = by_syn[same_sid]
            for diff_sid, diff in by_syn.items():
                if diff_sid == same_sid:
                    continue
                (sa, ua), (sp, up) = rng.sample(same, 2)
                sn, un = rng.choice(diff)
                triplets.append(
                    {
                        "lemma": lemma,
                        "pos": pos,
                        "synset_same": same_sid,
                        "synset_diff": diff_sid,
                        "anchor": mark_target(ua, lemma),
                        "positive": mark_target(up, lemma),
                        "negative": mark_target(un, lemma),
                        "gloss_same": sa.definition(),
                        "gloss_diff": sn.definition(),
                    }
                )
        rng.shuffle(triplets)
        out[split]["triplet"].extend(triplets[:max_per_lemma])

    for s in out:
        rng.shuffle(out[s]["direct"])
        rng.shuffle(out[s]["triplet"])
    return out


def save_dataset(data, data_dir: Path = DATA_DIR):
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, modes in data.items():
        for mode, recs in modes.items():
            (data_dir / f"sense_{mode}.{split}.json").write_text(
                json.dumps(recs, ensure_ascii=False, indent=2)
            )


def load_split(mode: str, split: str, data_dir: Path = DATA_DIR) -> list[dict]:
    return json.loads((data_dir / f"sense_{mode}.{split}.json").read_text())


# --------------------------------------------------------------------------- #
# Prompt / target formatting (chat messages)
# --------------------------------------------------------------------------- #
def direct_think(rec) -> str:
    # Templated argument (Phase-3 warm-start: optimise for format + gloss copy).
    return f"<think>\nIn this usage, {rec['lemma']} means: {rec['gloss']}.\n</think>"


def direct_messages(rec, with_target=False):
    msgs = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {
            "role": "user",
            "content": f"Word: {rec['lemma']} ({rec['pos']})\nSentence: {rec['usage']}\nDefinition:",
        },
    ]
    if with_target:
        msgs.append({"role": "assistant", "content": f"{direct_think(rec)}\n{rec['gloss']}"})
    return msgs


def triplet_answer(rec) -> str:
    return f"<answer>\n{rec['gloss_same']}\n</answer>"


def triplet_think(rec) -> str:
    # Templated contrast (Phase-3 warm-start: optimise for format + gloss copy).
    # The negative gloss is named only to motivate the differentia; it is never
    # emitted in the answer.
    return (
        f"<think>\nThe anchor and positive usages both mean: {rec['gloss_same']}. "
        f"They share that genus and are used the same way in context. The negative "
        f"usage instead means: {rec['gloss_diff']}, which is the differentia that "
        f"sets it apart.\n</think>"
    )


def triplet_messages(rec, with_target=False):
    user = (
        f"Target word: {rec['lemma']}\n"
        f"POS: {rec['pos']}\n\n"
        f"Anchor usage: {rec['anchor']}\n"
        f"Positive usage: {rec['positive']}\n"
        f"Negative usage: {rec['negative']}\n\n"
        "Give the gloss for the sense shared by the anchor and positive."
    )
    msgs = [
        {"role": "system", "content": TRIPLET_SYSTEM},
        {"role": "user", "content": user},
    ]
    if with_target:
        msgs.append(
            {"role": "assistant", "content": triplet_think(rec) + triplet_answer(rec)}
        )
    return msgs


def extract_direct_gloss(text: str) -> str:
    text = text.split("</think>")[-1]
    # An unclosed <think> (reasoning ran past the length budget) leaves no gloss;
    # don't pass the dangling tag through as one or the format reward fires on it.
    if "<think>" in text:
        return ""
    return text.strip().splitlines()[0].strip() if text.strip() else ""


def extract_shared_gloss(text: str) -> str:
    """The single gloss (sense shared by anchor and positive) from <answer>.

    Requires a closed ``<answer>`` block, then takes its first non-empty line.
    Tolerant of the markdown small models like to add — leading bullets/quotes
    and a leftover ``Positive:``/``Definition:`` label — so a correct gloss isn't
    scored 0 (and dropped from distillation) just for being decorated.
    """
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if not m:
        return ""
    for line in m.group(1).splitlines():
        line = line.strip(" >-*_\t")
        if line:
            return re.sub(
                r"^(positive|definition)[\s*_]*:[\s*_]*", "", line, flags=re.IGNORECASE
            ).strip()
    return ""


# --------------------------------------------------------------------------- #
# BLEU (self-contained; uses sacrebleu if available)
# --------------------------------------------------------------------------- #
def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _ngram_counts(toks, n):
    return Counter(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))


def sentence_bleu(hyp: str, ref: str, max_n: int = 4) -> float:
    """Smoothed sentence BLEU in [0,1] (Chen & Cherry smoothing 1). Reward signal."""
    h, r = _tok(hyp), _tok(ref)
    if not h:
        return 0.0
    logs = []
    for n in range(1, max_n + 1):
        hn = _ngram_counts(h, n)
        rn = _ngram_counts(r, n)
        overlap = sum(min(c, rn.get(g, 0)) for g, c in hn.items())
        total = sum(hn.values())
        if total == 0:
            logs.append(math.log(1e-9))
            continue
        p = overlap / total if overlap > 0 else 1e-9 / total
        logs.append(math.log(p))
    bp = 1.0 if len(h) > len(r) else math.exp(1 - len(r) / len(h))
    return bp * math.exp(sum(logs) / max_n)


def token_f1(hyp: str, ref: str) -> float:
    """Unigram overlap F1 in [0,1] — smooth similarity for short glosses."""
    h, r = _tok(hyp), _tok(ref)
    if not h or not r:
        return 0.0
    overlap = sum((Counter(h) & Counter(r)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(h), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def gloss_similarity(hyp: str, ref: str) -> float:
    """GRPO reward signal in [0,1]: mean of token-F1 and BLEU-2 (both smooth on
    short glosses, unlike BLEU-4 which is near-zero without 4-gram matches)."""
    return 0.5 * token_f1(hyp, ref) + 0.5 * sentence_bleu(hyp, ref, max_n=2)


def corpus_bleu(hyps: list[str], refs: list[str], max_n: int = 4) -> float:
    """Corpus BLEU in [0,100]. Reported eval metric; prefers sacrebleu if installed."""
    try:
        import sacrebleu

        return sacrebleu.corpus_bleu(hyps, [refs]).score
    except ImportError:
        pass
    clipped = [0] * max_n
    totals = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        h, r = _tok(hyp), _tok(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hn = _ngram_counts(h, n)
            rn = _ngram_counts(r, n)
            clipped[n - 1] += sum(min(c, rn.get(g, 0)) for g, c in hn.items())
            totals[n - 1] += max(sum(hn.values()), 0)
    if hyp_len == 0:
        return 0.0
    precisions = [
        (clipped[i] + 1) / (totals[i] + 1) for i in range(max_n)
    ]  # +1 smoothing
    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)
    return 100.0 * bp * geo


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Build the WordNet sense-modeling splits.")
    ap.add_argument("--lexicon", default="oewn:2024")
    ap.add_argument("--max-per-lemma", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = build_dataset(args.lexicon, args.max_per_lemma, args.seed)
    save_dataset(data)

    # report stats + verify lemma-disjointness
    lemma_sets = {}
    for split, modes in data.items():
        n_d, n_t = len(modes["direct"]), len(modes["triplet"])
        lemmas = {r["lemma"] for r in modes["direct"]} | {
            r["lemma"] for r in modes["triplet"]
        }
        lemma_sets[split] = lemmas
        print(f"{split:5s}  direct={n_d:6d}  triplet={n_t:6d}  lemmas={len(lemmas)}")
    a, b, c = lemma_sets.values()
    print(
        "lemma overlap across splits:",
        len(a & b),
        len(a & c),
        len(b & c),
        "(must be 0,0,0)",
    )


if __name__ == "__main__":
    main()
