"""Build the SFT WiC dataset and dump it for manual inspection.

Standalone (no torch import) so it can run without a working CUDA torch install.
"""

import argparse
import json
import random
import re
from itertools import combinations
from pathlib import Path

import wn
from rapidfuzz import fuzz, process


POS_MAP = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def mark_target(sentence: str, word: str, fuzzy_threshold: float = 70.0) -> str:
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
            _, _, idx = match
            best = tokens[idx]
            return re.sub(rf"\b{re.escape(best)}\b", f"<t> {best} </t>", sentence, count=1)

    return sentence + f" <t> {word} </t>"


def _load_wordnet(lexicon: str) -> wn.Wordnet:
    try:
        return wn.Wordnet(lexicon)
    except wn.Error:
        wn.download(lexicon)
        return wn.Wordnet(lexicon)


def build_wordnet_dataset(lexicon, max_pairs_per_lemma, max_total, seed=42):
    en = _load_wordnet(lexicon)
    rng = random.Random(seed)

    by_lemma: dict[tuple[str, str], list] = {}
    for syn in en.synsets():
        if not syn.examples():
            continue
        pos = syn.pos
        for lemma in syn.lemmas():
            if " " in lemma or "_" in lemma:
                continue
            by_lemma.setdefault((lemma.lower(), pos), []).append(syn)

    records = []
    items = list(by_lemma.items())
    rng.shuffle(items)

    for (lemma, pos), synsets in items:
        sense_examples = [
            (s, ex) for s in synsets for ex in s.examples() if lemma in ex.lower()
        ]
        if len(sense_examples) < 2:
            continue

        by_syn: dict = {}
        for s, ex in sense_examples:
            by_syn.setdefault(s.id, []).append((s, ex))

        positives = []
        for pairs in by_syn.values():
            positives.extend(combinations(pairs, 2))

        negatives = []
        syn_names = list(by_syn.keys())
        for i in range(len(syn_names)):
            for j in range(i + 1, len(syn_names)):
                for a in by_syn[syn_names[i]]:
                    for b in by_syn[syn_names[j]]:
                        negatives.append((a, b))

        rng.shuffle(positives)
        rng.shuffle(negatives)
        k = max_pairs_per_lemma // 2
        chosen = [(p, 1) for p in positives[:k]] + [(n, 0) for n in negatives[:k]]

        for ((s1, ex1), (s2, ex2)), label in chosen:
            records.append({
                "pos": POS_MAP.get(pos, pos),
                "word1": lemma,
                "word2": lemma,
                "sentence1": ex1,
                "sentence2": ex2,
                "gloss1": s1.definition(),
                "gloss2": s2.definition(),
                "label": label,
            })
        if max_total and len(records) >= max_total:
            break

    rng.shuffle(records)
    if max_total:
        records = records[:max_total]
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lexicon", default="oewn:2024")
    parser.add_argument("--max_pairs_per_lemma", type=int, default=4)
    parser.add_argument("--max_total", type=int, default=2000)
    parser.add_argument("--out", default="data/sft_wic_preview.jsonl")
    parser.add_argument("--sample", type=int, default=30)
    args = parser.parse_args()

    records = build_wordnet_dataset(
        lexicon=args.lexicon,
        max_pairs_per_lemma=args.max_pairs_per_lemma,
        max_total=args.max_total,
    )
    print(f"Generated {len(records)} pairs")

    fuzzy_used = 0
    fallback_used = 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in records:
            ex = dict(ex)
            for k_word, k_sent in [("word1", "sentence1"), ("word2", "sentence2")]:
                w, s = ex[k_word], ex[k_sent]
                exact = re.search(rf"\b({re.escape(w)}\w*)\b", s, flags=re.IGNORECASE)
                marked = mark_target(s, w)
                if not exact:
                    if marked.endswith(f" <t> {w} </t>"):
                        fallback_used += 1
                    else:
                        fuzzy_used += 1
                ex[k_sent + "_marked"] = marked
            f.write(json.dumps(ex, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")
    print(f"fuzzy matches:    {fuzzy_used}")
    print(f"appended fallback: {fallback_used}")

    print("\n--- sample ---")
    for ex in records[: args.sample]:
        s1 = mark_target(ex["sentence1"], ex["word1"])
        s2 = mark_target(ex["sentence2"], ex["word2"])
        print(f"[{ex['label']}] {ex['word1']} ({ex['pos']})")
        print(f"  s1: {s1}")
        print(f"  s2: {s2}")
        print()


if __name__ == "__main__":
    main()
