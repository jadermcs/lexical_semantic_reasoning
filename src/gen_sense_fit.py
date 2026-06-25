"""Generate sense-fit examples from WordNet.

Each row mirrors data/sense_fit.train.json:

    {"word", "usage", "positive", "negative"}

plus metadata fields (strategy, positive_type, difficulty, similarity, ...)
that make it easy to rebalance the mix later.

POSITIVES (both "fit" the usage):
  - self:     the synset's own gloss.
  - hypernym: an immediate hypernym's gloss -- the supersense subsumes the
              sense, so it still describes the usage.

NEGATIVES (do NOT fit the usage):
  1. same_lemma : another sense of the *same* word (polysemy). Hardest, keeps
                  the word string identical so the model can't cheat.
  3. cohyponym  : a synset from the same taxonomic neighbourhood (siblings /
                  cousins) that is NOT an ancestor or descendant of the sense.
                  Graded by Wu-Palmer similarity into hard/medium/easy.
  5. antonym    : the gloss of an antonym sense (same domain, opposite meaning).

Run:  uv run python src/gen_sense_fit.py --max_total 500
"""

import argparse
import json
import random
from pathlib import Path

import wn
import wn.similarity as sim

POS_MAP = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def _load_wordnet(lexicon: str) -> wn.Wordnet:
    try:
        return wn.Wordnet(lexicon)
    except wn.Error:
        wn.download(lexicon)
        return wn.Wordnet(lexicon)


def _single_word(lemma: str) -> bool:
    return " " not in lemma and "_" not in lemma


def _lemma_in_example(lemmas, example):
    """Return the lemma that surfaces in the example, or None."""
    low = example.lower()
    for lemma in lemmas:
        if _single_word(lemma) and lemma.lower() in low:
            return lemma
    return None


def _ancestors(syn, depth):
    """Immediate hypernyms up to `depth` levels (depth=1 -> direct parents)."""
    frontier, seen, out = [syn], {syn.id}, []
    for _ in range(depth):
        nxt = []
        for s in frontier:
            for h in s.hypernyms():
                if h.id not in seen:
                    seen.add(h.id)
                    out.append(h)
                    nxt.append(h)
        frontier = nxt
    return out


def _same_pos(a, b):
    """Compare coarse POS so satellite adjectives ('s') match adjectives ('a')."""
    return POS_MAP.get(a.pos, a.pos) == POS_MAP.get(b.pos, b.pos)


def _wup(a, b):
    try:
        return sim.wup(a, b, simulate_root=True)
    except Exception:
        return None


def _difficulty(score, hard_min, medium_min):
    if score is None:
        return "unknown"
    if score >= hard_min:
        return "hard"
    if score >= medium_min:
        return "medium"
    return "easy"


def neighborhood(syn, up):
    """Cousin candidates: descend from each ancestor (up to `up` levels) and
    collect its hyponyms, excluding the sense's own ancestors/descendants."""
    block = {syn.id}
    for a in _ancestors(syn, depth=10):
        block.add(a.id)
    for d in syn.hyponyms():
        block.add(d.id)

    cands = {}
    for anc in _ancestors(syn, depth=up):
        for cand in anc.hyponyms():
            if cand.id in block or cand.id in cands:
                continue
            if not cand.definition():
                continue
            cands[cand.id] = cand
    return list(cands.values())


def negatives_for(en, syn, lemma, pos, args, rng):
    """Yield (strategy, negative_synset, similarity) tuples."""
    out = []
    ancestor_ids = {a.id for a in _ancestors(syn, depth=10)}
    descendant_ids = {d.id for d in syn.hyponyms()}

    # 1. same lemma, different sense
    same = []
    for other in en.synsets(lemma, pos=pos):
        if other.id == syn.id or other.id in ancestor_ids or other.id in descendant_ids:
            continue
        if other.definition():
            same.append(other)
    rng.shuffle(same)
    for o in same[: args.max_per_strategy]:
        out.append(("same_lemma", o, _wup(syn, o)))

    # 3. cohyponym / cousin, similarity-graded
    cous = [(c, _wup(syn, c)) for c in neighborhood(syn, up=args.cousin_up)]
    # prefer the harder (more similar) candidates first, but keep a tail for variety
    cous.sort(key=lambda x: (x[1] is not None, x[1] or 0.0), reverse=True)
    for c, score in cous[: args.max_per_strategy]:
        out.append(("cohyponym", c, score))

    # 5. antonyms
    seen_ant = set()
    for sense in syn.senses():
        for ant in sense.get_related("antonym"):
            asyn = ant.synset()
            if asyn.id in seen_ant or not asyn.definition():
                continue
            seen_ant.add(asyn.id)
            out.append(("antonym", asyn, _wup(syn, asyn)))

    # keep negatives in the same coarse part of speech as the source sense
    return [(strat, n, sc) for strat, n, sc in out if _same_pos(syn, n)]


def build(args):
    en = _load_wordnet(args.lexicon)
    rng = random.Random(args.seed)

    synsets = [s for s in en.synsets() if s.examples()]
    rng.shuffle(synsets)

    rows = []
    for syn in synsets:
        lemma = _lemma_in_example(syn.lemmas(), syn.examples()[0])
        # fall back to scanning every example for a surfacing lemma
        usage = None
        for ex in syn.examples():
            l = _lemma_in_example(syn.lemmas(), ex)
            if l:
                lemma, usage = l, ex
                break
        if not lemma:
            continue
        pos = syn.pos

        positives = [("self", syn.definition())]
        for h in _ancestors(syn, depth=args.hypernym_depth):
            if h.definition():
                positives.append(("hypernym", h.definition()))

        negs = negatives_for(en, syn, lemma, pos, args, rng)
        if not negs:
            continue
        rng.shuffle(negs)

        for strategy, nsyn, score in negs[: args.max_per_sense]:
            ptype, pgloss = rng.choice(positives)
            ngloss = nsyn.definition()
            if ngloss == pgloss or ngloss in {p for _, p in positives}:
                continue
            rows.append({
                "word": lemma,
                "usage": usage,
                "positive": pgloss,
                "negative": ngloss,
                "strategy": strategy,
                "positive_type": ptype,
                "difficulty": _difficulty(score, args.hard_min, args.medium_min),
                "similarity": round(score, 4) if score is not None else None,
                "pos": POS_MAP.get(pos, pos),
                "synset_id": syn.id,
                "negative_synset_id": nsyn.id,
            })
        if args.max_total and len(rows) >= args.max_total:
            break

    rng.shuffle(rows)
    if args.max_total:
        rows = rows[: args.max_total]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lexicon", default="oewn:2024")
    p.add_argument("--out", default="data/sense_fit.generated.json")
    p.add_argument("--max_total", type=int, default=500)
    p.add_argument("--max_per_sense", type=int, default=3,
                   help="rows emitted per source synset")
    p.add_argument("--max_per_strategy", type=int, default=3,
                   help="candidate negatives kept per strategy before sampling")
    p.add_argument("--hypernym_depth", type=int, default=1,
                   help="how many levels of hypernym glosses to allow as positives")
    p.add_argument("--cousin_up", type=int, default=2,
                   help="ancestor levels to climb when gathering cohyponym candidates")
    p.add_argument("--hard_min", type=float, default=0.6)
    p.add_argument("--medium_min", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = build(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")

    by_strategy, by_diff = {}, {}
    for r in rows:
        by_strategy[r["strategy"]] = by_strategy.get(r["strategy"], 0) + 1
        by_diff[r["difficulty"]] = by_diff.get(r["difficulty"], 0) + 1
    print(f"Wrote {len(rows)} rows -> {out}")
    print("strategy:", by_strategy)
    print("difficulty:", by_diff)
    print("\n--- sample ---")
    for r in rows[:6]:
        print(f"[{r['strategy']}/{r['difficulty']} pos={r['positive_type']}] {r['word']} ({r['pos']})")
        print(f"  usage: {r['usage']}")
        print(f"  +    : {r['positive']}")
        print(f"  -    : {r['negative']}")
        print()


if __name__ == "__main__":
    main()
