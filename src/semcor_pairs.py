"""WiC pairs built from SemCor's gold sense annotations — gold glosses for WiC usages.

MCL-WiC carries a label and nothing else, so the only privileged context a supervised
self-distillation run can supply on it is the gold verdict — one bit, and the
same bit the reward already encodes. SemCor is sense-annotated against WordNet, so
every usage here comes with the *gold synset*, and the hint can say what the word
actually means in each sentence. Measured against the API teacher's own glosses on
``data/semcor.json``, the gold gloss overlaps them at token-F1 0.16 (46.5% share no
content token at all), so this is genuinely new information rather than a restatement
of what the policy already produces.

It also pays off where the policy is weakest. On that same file the teacher's
accuracy on different-sense pairs slid 71.2% → 64.0% → 60.9% → 56.8% as the two gold
synsets got closer in WordNet, which is exactly the slice ``--min-confusability``
lets you oversample.

**Source file.** ``data/semcor_en.json.gz``, produced outside this repo — nothing here
builds it, this module only reads it. One record per annotated token with
``text``, ``sent_id``, character offsets ``start``/``end``, ``lemma``, ``pos`` and
``synsets``. The offsets are why this file supersedes the older ``data/semcor.json``
pair set — that one had already thrown the sense keys away, and re-deriving them by
matching sentences back to NLTK SemCor left 9.1% of multi-occurrence sentences
ambiguous about *which* occurrence was the target. Here the span is exact, so
``mark_usage`` never has to guess and ``sense_data.mark_target``'s fuzzy fallback is
bypassed entirely.

**POS spelling matters.** Pairs are emitted with ``noun``/``verb``/``adj``/``adv`` —
the vocabulary MCL-WiC uses and the one ``gloss_wordnet._POS`` keys on. The old
pair file spelled these ``adjective``/``adverb``, which ``_POS.get(pos, ()) or
(None,)`` silently degrades to a POS-blind synset lookup, so ``reward_wic_wordnet``
was scoring adjectives against noun and verb synsets too.

**Lexicon.** Gold glosses are read from NLTK's WordNet 3.0, because that is the
inventory SemCor's annotations *refer to*; 99.05% of the synset names in the file
resolve there (the 247 that don't are adjective satellites). This is deliberately a
different lexicon from ``gloss_wordnet``'s Open English WordNet 2024 — that module
snaps free-text glosses and wants a maintained inventory, whereas a gold label must
be read in the inventory it was written in. Mapping 3.0 → OEWN would inject
alignment drift into the one signal here that is supposed to be ground truth.

**What a record carries.** Beyond the WiC fields every loader shares, each pair keeps
``synset1``/``synset2``, their gold glosses ``gloss1``/``gloss2``, and ``senses`` — the
*whole* WordNet 3.0 gloss inventory of the ``(lemma, pos)``, gold entries included.
That last field is what turns a gold gloss into a scoreable target: ``gloss1`` alone
only supports "does the emitted gloss look like this string", which is a similarity
measure with no scale (the teacher's own correct glosses overlap gold at token-F1
0.16). Against the inventory, the question becomes *discriminative* — is the emitted
gloss closer to the gold sense than to the other senses of the same word — which is
the ~1.76 bits/gloss ``sense_rewards.reward_wic_gloss`` scores. Persisting the
inventory alongside the pair also keeps that reward pure-Python at rollout time and
keeps every comparison inside one lexicon, so no synset ID ever has to be mapped
across WordNet versions.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path("data")
DEFAULT_SOURCE = DATA_DIR / "semcor_en.json.gz"

# Universal POS in the source file → the repo's spelling (MCL-WiC's, and the keys
# gloss_wordnet._POS looks up). Anything else in the file is dropped.
POS_MAP = {"NOUN": "noun", "VERB": "verb", "ADJ": "adj", "ADV": "adv"}

# Confusability of two synsets of one lemma, hardest first. Ordering is what
# ``--min-confusability`` thresholds on, so it is defined once here.
CONFUSABILITY = ["siblings", "near", "far", "unrelated", "cross-pos"]


def _wn():
    """Import NLTK's WordNet lazily, with an actionable error if the corpus is absent.

    Mirrors ``gloss_wordnet._wordnet``: the package is a dependency, the corpus data
    is a separate one-time download, and importing this module must not pay for it.
    """
    from nltk.corpus import wordnet

    try:
        wordnet.synsets("dog")
    except LookupError as e:
        raise LookupError(
            "NLTK's WordNet corpus is not installed; run:\n"
            "  uv run python -c \"import nltk; nltk.download('wordnet')\""
        ) from e
    return wordnet


@lru_cache(maxsize=1)
def _synset_getter():
    wn = _wn()
    return wn.synset


@lru_cache(maxsize=65536)
def synset(name: str):
    """WordNet 3.0 synset for a SemCor synset name, or None if it does not resolve.

    Returns None rather than raising: the ~1% of names that miss are adjective
    satellites, and one unresolvable synset should cost that pair, not the run.
    """
    try:
        return _synset_getter()(name)
    except Exception:
        return None


def gloss(name: str) -> str | None:
    s = synset(name)
    return s.definition() if s is not None else None


# NLTK's part-of-speech codes for this repo's POS spelling. Adjectives union 'a' and
# 's' for the same reason ``gloss_wordnet._POS`` does: satellites are a separate,
# disjoint POS, and SemCor annotates plenty of them.
NLTK_POS = {"noun": ("n",), "verb": ("v",), "adj": ("a", "s"), "adv": ("r",)}


@lru_cache(maxsize=16384)
def sense_inventory(lemma: str, pos: str) -> tuple[str, ...]:
    """Every WordNet 3.0 gloss of ``(lemma, pos)`` — the candidate set a gloss ranks in.

    Deduplicated and order-stable. NLTK spells multiword lemmas with underscores (the
    opposite of the `wn` package), so a lemma arriving with spaces is normalised.
    """
    wn = _wn()
    key = str(lemma).replace(" ", "_")
    out, seen = [], set()
    for p in NLTK_POS.get(pos, ()) or (None,):
        for s in wn.synsets(key, pos=p):
            d = s.definition()
            if d and d not in seen:
                seen.add(d)
                out.append(d)
    return tuple(out)


def confusability(name1: str, name2: str) -> str | None:
    """How hard two synsets of one lemma are to tell apart, as a CONFUSABILITY level.

    ``siblings`` (share a direct hypernym) and ``near`` (path distance ≤ 4) are the
    strata worth trusting. ``unrelated`` means *no path was found*, which for nouns
    means genuinely distant but for verbs often just means the two hierarchies are
    disjoint — WordNet's verb taxonomy is shallow and fragmented. Do not read
    ``unrelated`` as "easy"; read it as "unmeasured".
    """
    a, b = synset(name1), synset(name2)
    if a is None or b is None:
        return None
    if a.pos() != b.pos():
        return "cross-pos"
    common = a.lowest_common_hypernyms(b)
    if common and common[0] in a.hypernyms() and common[0] in b.hypernyms():
        return "siblings"
    dist = a.shortest_path_distance(b)
    if dist is None:
        return "unrelated"
    return "near" if dist <= 4 else "far"


def held_out_lemmas(splits=("test",), data_dir=DATA_DIR) -> set[str]:
    """Lemmas appearing in an MCL-WiC split, lowercased.

    SemCor and MCL-WiC share no *pairs*, so pair-level exclusion is vacuous between
    them — but they share plenty of *lemmas*, and a lemma is the unit of what the
    policy learns here. Rolling out on SemCor's ``bank`` teaches the sense inventory
    of ``bank``, and the test score on MCL-WiC's ``bank`` pairs then measures recall
    of that inventory rather than generalisation to an unseen word. Excluding the
    test lemmas outright is what makes the held-out number a real one.

    Missing split files yield an empty set rather than an error: the guard should not
    be the thing that stops a run in a checkout without the benchmark.
    """
    out: set[str] = set()
    for split in splits:
        p = Path(data_dir) / f"mcl-wic.{split}.json"
        if not p.exists():
            continue
        out |= {str(r["lemma"]).lower() for r in json.loads(p.read_text())}
    return out


def load_annotations(path=DEFAULT_SOURCE) -> list[dict]:
    """Every single-synset annotation in the source file, POS-normalised.

    Multi-synset annotations (653 of 226 036) are dropped: the whole point of this
    source is an unambiguous gold sense, and a token annotated with two synsets has
    no single gloss to hand the teacher.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        raw = json.load(f)
    return [
        {**r, "pos": POS_MAP[r["pos"]], "synset": r["synsets"][0]}
        for r in raw
        if r.get("pos") in POS_MAP and len(r.get("synsets") or []) == 1
    ]


def mark_usage(ann: dict) -> str:
    """The sentence with the annotated span wrapped in <t> tags.

    Spacing matches ``sense_data.mark_target``'s ``"<t> word </t>"`` output exactly,
    so a SemCor prompt and an MCL-WiC prompt are token-identical in shape and the
    two can be mixed in one rollout batch without the policy seeing a tell.
    """
    text, i, j = ann["text"], ann["start"], ann["end"]
    return f"{text[:i]}<t> {text[i:j]} </t>{text[j:]}"


def index_usages(anns: list[dict]) -> dict:
    """``(lemma, pos)`` → ``synset`` → list of usages."""
    idx: dict = defaultdict(lambda: defaultdict(list))
    for a in anns:
        idx[(a["lemma"], a["pos"])][a["synset"]].append(a)
    return idx


def _record(lemma, pos, u1, s1, u2, s2) -> dict:
    """One WiC record in the schema every loader in this repo shares.

    ``synset1``/``synset2`` and the gloss fields ride along beyond ``sense_data``'s;
    they are what ``gloss_feedback`` and ``sense_rewards.reward_wic_gloss`` read, and
    what makes this source worth more than MCL-WiC.

    An unresolvable synset (an adjective satellite WordNet 3.0 will not name) yields
    an empty gloss rather than dropping the pair: both consumers already degrade to
    the label-only path when a gloss is missing.

    So does a *different*-sense pair whose two synsets happen to carry byte-identical
    definitions — WordNet has several (``precarious.s.01`` and ``unstable.s.02`` are
    both "affording no ease or reassurance"; 26 of 9 678 sampled pairs). There is no
    gloss that could tell those two apart, so the pair keeps its label and loses its
    gloss supervision rather than asking for a distinction the inventory does not make.
    """
    g1, g2 = gloss(s1) or "", gloss(s2) or ""
    if s1 != s2 and g1 and g1 == g2:
        g1 = g2 = ""
    senses = list(sense_inventory(lemma, pos))
    # The gold gloss must be *in* the candidate set, or ranking against it is
    # unwinnable. It normally is; a lemma spelled differently in SemCor than in
    # WordNet is the exception this covers.
    for g in (g1, g2):
        if g and g not in senses:
            senses.append(g)
    return {
        "lemma": lemma,
        "pos": pos,
        "label": s1 == s2,
        "sentence1": u1["text"],
        "sentence2": u2["text"],
        "usage1": mark_usage(u1),
        "usage2": mark_usage(u2),
        "synset1": s1,
        "synset2": s2,
        "gloss1": g1,
        "gloss2": g2,
        "senses": senses,
    }


def build_pairs(
    path=DEFAULT_SOURCE,
    max_per_lemma: int = 4,
    min_confusability: str | None = None,
    min_sentence_tokens: int = 5,
    balance: bool = True,
    exclude_lemma_splits=("test",),
    seed: int = 42,
) -> list[dict]:
    """Sample WiC pairs from the annotations.

    ``max_per_lemma`` caps *each* of the same/different sides per ``(lemma, pos)``.
    Without a cap the set is swallowed by the handful of lemmas SemCor annotates
    thousands of times (one synset alone has 10 082 usages), and the policy would
    spend RL on ``be`` and ``have``.

    ``min_confusability`` keeps only different-sense pairs at or above a
    CONFUSABILITY level — ``"near"`` restricts to the sibling/near strata where the
    teacher measurably struggles. It does not touch same-sense pairs, which have no
    confusability to speak of.

    ``balance`` down-samples the larger side to 50/50, for the same reason
    ``grpo_lora.balance_labels`` exists: on a skewed rollout set the cheapest reward
    gradient is to collapse to the majority class.

    ``exclude_lemma_splits`` drops every lemma occurring in those MCL-WiC splits, so
    the test score stays a generalisation measurement — see ``held_out_lemmas``. Pass
    ``()`` to disable, and know what you are giving up when you do.
    """
    rng = random.Random(seed)
    idx = index_usages(load_annotations(path))
    cutoff = CONFUSABILITY.index(min_confusability) if min_confusability else None

    blocked = held_out_lemmas(exclude_lemma_splits) if exclude_lemma_splits else set()
    if blocked:
        before = len(idx)
        idx = {k: v for k, v in idx.items() if k[0].lower() not in blocked}
        print(
            f"semcor: held out {before - len(idx)}/{before} (lemma, pos) groups whose "
            f"lemma occurs in mcl-wic {'/'.join(exclude_lemma_splits)}"
        )

    def usable(u):
        return len(u["text"].split()) >= min_sentence_tokens

    same_recs, diff_recs = [], []
    for (lemma, pos), by_syn in sorted(idx.items()):
        by_syn = {s: [u for u in us if usable(u)] for s, us in by_syn.items()}
        by_syn = {s: us for s, us in by_syn.items() if us}
        if not by_syn:
            continue

        # --- same sense: two distinct usages of one synset ---
        same_here = []
        for s, us in sorted(by_syn.items()):
            if len(us) < 2:
                continue
            picked = rng.sample(us, min(len(us), 2 * max_per_lemma))
            for a, b in zip(picked[0::2], picked[1::2]):
                if a["text"] != b["text"]:
                    same_here.append(_record(lemma, pos, a, s, b, s))
        rng.shuffle(same_here)
        same_recs.extend(same_here[:max_per_lemma])

        # --- different sense: usages of two distinct synsets ---
        names = sorted(by_syn)
        combos = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                level = confusability(names[i], names[j])
                if level is None:
                    continue
                rank = CONFUSABILITY.index(level)
                if cutoff is not None and rank > cutoff:
                    continue
                combos.append((rank, names[i], names[j]))
        # Hardest combos first, so the per-lemma cap spends its budget on the pairs
        # that carry signal rather than on whatever sorted() happened to surface.
        combos.sort(key=lambda c: (c[0], c[1], c[2]))
        for _, s1, s2 in combos[:max_per_lemma]:
            a, b = rng.choice(by_syn[s1]), rng.choice(by_syn[s2])
            if a["text"] != b["text"]:
                diff_recs.append(_record(lemma, pos, a, s1, b, s2))

    if balance:
        n = min(len(same_recs), len(diff_recs))
        rng.shuffle(same_recs)
        rng.shuffle(diff_recs)
        same_recs, diff_recs = same_recs[:n], diff_recs[:n]

    out = same_recs + diff_recs
    rng.shuffle(out)
    return out


DEFAULT_PAIRS = DATA_DIR / "semcor_wic.json"


def load_pairs(path=DEFAULT_PAIRS) -> list[dict]:
    """Read a pair set written by ``main`` back off disk.

    Training reads the *saved* file rather than rebuilding: ``build_pairs`` samples
    with an RNG and queries WordNet for every ``(lemma, pos)``, so materialising it
    once is what makes a run reproducible and keeps NLTK off the training host's
    critical path. Missing gloss fields are filled in so an older file still loads —
    the pairs are then simply unscoreable by ``reward_wic_gloss``, not broken.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"{path} not found; build it first with:\n"
            f"  uv run python src/semcor_pairs.py --out {path}"
        )
    recs = json.loads(path.read_text())
    for r in recs:
        r.setdefault("gloss1", "")
        r.setdefault("gloss2", "")
        r.setdefault("senses", [])
    return recs


# --------------------------------------------------------------------------- #
# The privileged context SDPO reprompts the teacher with
# --------------------------------------------------------------------------- #
def gloss_feedback(rec: dict) -> str:
    """Gold-gloss hint for one pair — the reason this module exists.

    Two branches, and the asymmetry is forced by ``reward_wic_consistency``:

    * **same sense** — that reward scores a ``same_sense=true`` verdict as
      ``WIC_INCONSISTENT * (1 - gloss_similarity)``, so glosses that *agree* are what
      the reward wants and what ``sense_data.wic_answer`` writes as the SFT target.
      The hint therefore states the one gold gloss and lets both fields use it. An
      earlier draft told the teacher to word each gloss differently; that would have
      driven the similarity down and steered distillation straight into the penalty.
    * **different senses** — both gold glosses, plus their shared hypernym when they
      have one, since naming the genus is what makes a fine-grained contrast
      articulable. This branch is also where the gloss earns its keep against the
      gloss-identity shortcut: ``reward_wic_wordnet`` charges -0.25 when a
      *different* verdict's two glosses snap to one synset, and handing over two
      genuinely distinct gold glosses is the direct antidote.

    Falls back to the bare verdict if either gloss is unresolvable, so a satellite
    adjective costs its hint rather than the pair.
    """
    same = bool(rec["label"])
    lemma = rec["lemma"]
    g1, g2 = gloss(rec.get("synset1")), gloss(rec.get("synset2"))
    tail = (
        f'The correct verdict is "same_sense": {"true" if same else "false"}. '
        "Do not mention this hint; reason to it from the sentences themselves."
    )
    if g1 is None or g2 is None:
        verdict = "the same sense" if same else "different senses"
        return f'Hint: the two sentences use "{lemma}" in {verdict}. {tail}'

    if same:
        return f'Hint: both sentences use "{lemma}" in the sense "{g1}". {tail}'

    a, b = synset(rec["synset1"]), synset(rec["synset2"])
    contrast = ""
    if a is not None and b is not None and a.pos() == b.pos():
        common = a.lowest_common_hypernyms(b)
        if common:
            genus = common[0].lemma_names()[0].replace("_", " ")
            contrast = f" Both are kinds of {genus}, so the contrast is fine-grained."
    return (
        f'Hint: in sentence 1, "{lemma}" means "{g1}"; in sentence 2 it means '
        f'"{g2}".{contrast} {tail}'
    )


def main():
    """Inspect the pair set without training anything."""
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--max-per-lemma", type=int, default=4)
    ap.add_argument("--min-confusability", default=None, choices=CONFUSABILITY)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument(
        "--keep-test-lemmas",
        action="store_true",
        help="Do NOT hold out lemmas occurring in mcl-wic.test. Leaks the test "
        "vocabulary into the rollout set; the held-out score stops meaning much.",
    )
    ap.add_argument(
        "--out",
        default=str(DEFAULT_PAIRS),
        help="Where to write the pairs as JSON. Pass '' to only inspect them.",
    )
    ap.add_argument("--show", type=int, default=3)
    args = ap.parse_args()

    recs = build_pairs(
        args.source,
        max_per_lemma=args.max_per_lemma,
        min_confusability=args.min_confusability,
        balance=not args.no_balance,
        exclude_lemma_splits=() if args.keep_test_lemmas else ("test",),
    )
    leaked = {r["lemma"].lower() for r in recs} & held_out_lemmas(("test",))
    print(f"lemmas shared with mcl-wic.test: {len(leaked)}")
    print(f"{len(recs)} pairs  |  pos: {dict(Counter(r['pos'] for r in recs))}")
    print(f"labels: {dict(Counter(r['label'] for r in recs))}")
    levels = Counter(
        confusability(r["synset1"], r["synset2"]) for r in recs if not r["label"]
    )
    print(f"different-sense confusability: {dict(levels)}")
    # A pair is only scoreable by reward_wic_gloss if both gold glosses resolved and
    # the lemma has a candidate set to rank them against, so report that coverage.
    scoreable = sum(
        1 for r in recs if r["gloss1"] and r["gloss2"] and len(r["senses"]) > 1
    )
    n_senses = [len(r["senses"]) for r in recs]
    print(
        f"gloss-scoreable: {scoreable}/{len(recs)} "
        f"({scoreable / max(1, len(recs)):.1%})  |  candidate senses per pair: "
        f"mean {sum(n_senses) / max(1, len(n_senses)):.1f}, max {max(n_senses, default=0)}"
    )
    for r in recs[: args.show]:
        print(f"\n--- {r['lemma']}/{r['pos']}  label={r['label']}")
        print(f"  1: {r['usage1'][:120]}")
        print(f"  2: {r['usage2'][:120]}")
        print(f"  gold: {r['gloss1']!r} | {r['gloss2']!r}  ({len(r['senses'])} senses)")
        print(f"  feedback: {gloss_feedback(r)}")
    if args.out:
        Path(args.out).write_text(json.dumps(recs, ensure_ascii=False, indent=2))
        print(f"\nwrote {len(recs)} pairs → {args.out}")


if __name__ == "__main__":
    main()
