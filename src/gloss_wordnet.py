"""WordNet-anchored gloss checker -- a verifier signal worth more than one bit.

`same_sense` is binary, so an exact-match verifier carries a single bit: a
model guessing at random satisfies it half the time, and rejection sampling at
k=100 satisfies it for essentially every pair (see analysis/rejection_sample.py).
That makes "the label matched gold" nearly useless as an acceptance test for
self-distillation, and it is blind to the failure mode this repo actually has:
a trace that writes *the same gloss twice* and then declares the senses the
same (analysis/error_analysis.ipynb §5).

This module adds an independent check on the *glosses*. Each gloss is snapped
to the WordNet synset of its `(lemma, pos)` that it best matches, and the
relation between the two snapped synsets is compared against the gold label.
WordNet has no stake in the pipeline, so this is genuine outside information
rather than the model grading itself.

Calibration matters, and the checker is deliberately **asymmetric**. Measured on
the SFT model's test predictions, synset identity reproduces the gold label only
~72% of the time (§14) -- MCL-WiC sense granularity is not WordNet's -- so
demanding full agreement would reject masses of good traces. Only one rule fires
by default:

  ``wn_merged_diff_label`` gold says *different* but both glosses snap to the
                           same synset -- the trace never distinguished the
                           senses. This is exactly the shortcut cell, caught
                           from the outside.

Two further rules exist and are **off by default because calibration says so**
(``analysis/calibrate_gloss_check.py``; the teacher's own correct traces are the
quality ceiling any threshold must respect):

  ``gloss_unanchored``     (``min_anchor > 0``) the better-matching gloss scores
                           below ``min_anchor`` against every synset of the
                           lemma. Sound in principle, useless in practice: ~29%
                           of glosses share *zero* content tokens with any
                           WordNet gloss of their lemma -- including 35% of the
                           teacher's -- because WordNet's wording is terse and
                           idiosyncratic. Enabling it drops the rule's lift on
                           bad traces from 3.45x to 1.84x, i.e. it rejects
                           almost indiscriminately.
  ``wn_split_same_label``  (``strict=True``) gold *same* but the glosses snap to
                           different synsets. Anti-correlated with quality: the
                           teacher trips it (23.0%) *more* than the SFT model
                           (19.5%), because two valid paraphrases of one sense
                           routinely land on neighbouring synsets. Same reason
                           `filter_reasoning.sense_check` keeps its converse rule
                           behind a flag.

Default-mode selectivity on label-correct traces, i.e. traces a binary verifier
would already have accepted: base 31.3% > SFT 17.1% > teacher 15.9%, and 59.1%
on the SFT model's *wrong* traces. The ordering tracks gloss quality and the
lift is 3.45x, which is what makes it worth more than the bit it supplements.

Backed by the `wn` package (goodmami) against **Open English WordNet**, which is
still maintained -- NLTK ships WordNet 3.0, frozen since 2006. Set ``WN_LEXICON``
to pick a release (default ``oewn:2024``); install one with::

    uv run python -c "import wn; wn.download('oewn:2024')"

`wn` is a project dependency, but the *lexicon* is a separate one-time download,
so opening it stays lazy: importing this module must not pay for it, and
`sense_rewards` imports it at module scope to score every rollout.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

import wn

# Same stoplist the notebook uses for gloss overlap, so the measures agree.
STOP = frozenset(
    "a an the of to or and in for that is with by on as be it something someone".split()
)

# Disabled by default: see the module docstring. Token-F1 against WordNet's
# terse wording is too brittle to serve as an "is this a definition" gate.
MIN_ANCHOR = 0.0

LEXICON = os.environ.get("WN_LEXICON", "oewn:2024")

# Open English WordNet keeps adjective satellites in their own part of speech,
# and `wn` treats 'a' and 's' as disjoint, so adjectives must query both.
_POS = {"noun": ("n",), "verb": ("v",), "adj": ("a", "s"), "adv": ("r",)}

_W = None


def _wordnet():
    """Open the lexicon on first use, with an actionable error if it is missing.

    The package is a hard dependency; the lexicon it reads is not, so this is the
    one failure the caller still has to be told how to fix.
    """
    global _W
    if _W is None:
        try:
            _W = wn.Wordnet(LEXICON)
        except Exception as e:  # wn raises several types for a missing lexicon
            raise LookupError(
                f"lexicon {LEXICON!r} is not installed; run:\n"
                f"  uv run python -c \"import wn; wn.download('{LEXICON}')\"\n"
                f"(or set WN_LEXICON to one of: {{}})".format(
                    ", ".join(str(x) for x in _installed_lexicons())
                )
            ) from e
    return _W


def _installed_lexicons():
    try:
        return [f"{lex.id}:{lex.version}" for lex in wn.lexicons()]
    except Exception:  # pragma: no cover - only used to enrich an error message
        return []


def _toks(s) -> set[str]:
    return set(re.findall(r"\w+", str(s).lower())) - STOP


def _f1(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.0 if not inter else 2 * inter / (len(a) + len(b))


@lru_cache(maxsize=8192)
def synsets_for(lemma: str, pos: str) -> tuple:
    """Every synset for a (lemma, pos) as ``(id, definition)`` pairs.

    Note `wn` spells multiword lemmas with spaces, the opposite of NLTK's
    underscores. Adjectives union 'a' and 's' because satellites are a separate,
    disjoint part of speech here.
    """
    w = _wordnet()
    key = str(lemma).replace("_", " ")
    out, seen = [], set()
    for p in _POS.get(pos, ()) or (None,):
        for s in w.synsets(key, pos=p):
            d = s.definition()
            if s.id not in seen and d:
                seen.add(s.id)
                out.append((s.id, d))
    return tuple(out)


def snap(gloss: str, lemma: str, pos: str):
    """Best-matching synset for ``gloss`` among the lemma's senses.

    Returns ``(synset_id, score)``; ``(None, 0.0)`` when the lemma is absent
    from the lexicon. Token-F1 rather than BLEU: symmetric, order-free and far
    more stable on 5-15 token definitions, which is what retrieval needs.
    """
    cands = synsets_for(lemma, pos)
    if not cands:
        return None, 0.0
    g = _toks(gloss)
    best, best_score = None, -1.0
    for sid, definition in cands:
        sc = _f1(g, _toks(definition))
        if sc > best_score:
            best, best_score = sid, sc
    return best, max(0.0, best_score)


def explain(lemma: str, pos: str, sense1, sense2, label) -> dict:
    """Full detail behind a verdict -- for calibration and error analysis."""
    n_syn = len(synsets_for(lemma, pos))
    s1, sc1 = snap(sense1, lemma, pos)
    s2, sc2 = snap(sense2, lemma, pos)
    return {
        "in_wordnet": n_syn > 0, "n_synsets": n_syn,
        "synset1": s1, "synset2": s2, "score1": sc1, "score2": sc2,
        "wn_same": (s1 is not None and s1 == s2),
        "anchored": min(sc1, sc2),
        "label": bool(label),
    }


def check(lemma, pos, sense1, sense2, label, *,
          min_anchor: float = MIN_ANCHOR, strict: bool = False) -> str | None:
    """Name of the rule that rejects this gloss pair, or None if it passes.

    Mirrors the return convention of ``filter_reasoning``'s stage-1 checks so it
    can be dropped into the same ``or``-chain. Pairs whose lemma is missing from
    WordNet always pass -- absence of evidence is not evidence of a bad gloss.
    """
    if not isinstance(sense1, str) or not isinstance(sense2, str):
        return None  # unparseable answers already die in vote_check
    if not sense1.strip() or not sense2.strip():
        return None
    d = explain(lemma, pos, sense1, sense2, label)
    if not d["in_wordnet"]:
        return None

    if min_anchor > 0 and d["anchored"] < min_anchor:
        return "gloss_unanchored"
    if not d["label"] and d["wn_same"]:
        return "wn_merged_diff_label"
    if strict and d["label"] and not d["wn_same"]:
        return "wn_split_same_label"
    return None
