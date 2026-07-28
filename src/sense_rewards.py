import re
from difflib import SequenceMatcher

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import sense_data as sd

# --------------------------------------------------------------------------- #
# WiC (word-in-context): verifiable same/different-sense classification
# --------------------------------------------------------------------------- #
WIC_CORRECT = 1.0
WIC_WRONG = -0.5
WIC_ABSENT = -1.0
WIC_JSON_PARSES = 0.1  # answer region holds a parseable JSON object
WIC_JSON_KEYS = 0.1  # exactly the three required keys, nothing extra or missing
WIC_JSON_BOOL = 0.1  # same_sense is a JSON boolean, not "true" / 1 / "same"
WIC_JSON_MALFORMED = -0.2  # said something after </think>, but no JSON object in it
WIC_INCONSISTENT = -0.3
THINK_MIN_WORDS = 6  # below this many content words, reasoning is treated as a stub
THINK_MIN_PENALTY = -0.3

# --- gold-gloss grounding (SemCor pairs only; see reward_wic_gloss) ---------- #
GLOSS_MAX = 0.15  # magnitude of the term, per side, at full margin
GLOSS_FULL_MARGIN = 0.2  # token-F1 lead over the best rival sense that earns GLOSS_MAX


def _content_word_count(text):
    return sum(1 for t in sd._tok(text) if t not in ENGLISH_STOP_WORDS)


def _answer_region(text):
    seg = text.split("</think>")[-1]
    if "<think>" in seg:
        return ""
    return seg.strip()


def _think_answer_format_reward(completions, extractor):
    out = []
    for c in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if extractor(c) is not None:
            r += 0.1
        out.append(r)
    return out


def _extract_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def reward_think_length(completions, **kwargs):
    out = []
    for c in completions:
        think = _extract_think(c)
        out.append(
            THINK_MIN_PENALTY if _content_word_count(think) < THINK_MIN_WORDS else 0.0
        )
    return out


def reward_wic_accuracy(completions, **kwargs):
    out = []
    for c, label in zip(completions, kwargs["label"]):
        pred = sd.extract_wic_label(c)
        if pred is None:
            out.append(WIC_ABSENT)
        else:
            out.append(WIC_CORRECT if pred == bool(label) else WIC_WRONG)
    return out


def reward_wic_format(completions, **kwargs):
    """Reward a present <think> block (0.1) and an extractable verdict (0.1)."""
    return _think_answer_format_reward(completions, sd.extract_wic_label)


def reward_wic_json(completions, **kwargs):
    out = []
    for c in completions:
        obj = sd.parse_wic_answer(c)
        if obj is None:
            out.append(WIC_JSON_MALFORMED if _answer_region(c) else 0.0)
            continue
        r = WIC_JSON_PARSES
        if set(obj) == sd.WIC_ANSWER_KEYS:
            r += WIC_JSON_KEYS
        if isinstance(obj.get("same_sense"), bool):
            r += WIC_JSON_BOOL
        out.append(r)
    return out


def _gloss_similarity(s1, s2):
    return SequenceMatcher(None, sd._tok(s1), sd._tok(s2)).ratio()


def reward_wic_consistency(completions, **kwargs):
    out = []
    for c in completions:
        r = 0.0
        obj = sd.parse_wic_answer(c)
        if obj is not None:
            s1, s2, verdict = (
                obj.get("sense1"),
                obj.get("sense2"),
                obj.get("same_sense"),
            )
            if (
                isinstance(s1, str)
                and isinstance(s2, str)
                and isinstance(verdict, bool)
                and s1.strip()
                and s2.strip()
            ):
                if verdict:
                    r = WIC_INCONSISTENT * (1.0 - _gloss_similarity(s1, s2))
                elif sd._tok(s1) == sd._tok(s2):
                    r = WIC_INCONSISTENT
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
# Gold-gloss grounding
# --------------------------------------------------------------------------- #
def _content(text) -> set:
    return {t for t in sd._tok(str(text)) if t not in ENGLISH_STOP_WORDS}


def _token_f1(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 0.0 if not inter else 2 * inter / (len(a) + len(b))


def _gloss_margin(emitted: str, gold: str, senses) -> float:
    """How much better ``emitted`` matches the gold sense than any rival sense.

    Both terms are token-F1 over content words, so the difference lives in [-1, 1]:
    positive means the emitted gloss is nearest the gold sense, negative means some
    other sense of the same lemma fits it better, and 0 means no evidence either way
    — which includes the common case of a gloss sharing no content word with any
    WordNet wording (~29% of them; see ``gloss_wordnet``'s ``gloss_unanchored``
    calibration). Absolute similarity is deliberately *not* the score: the teacher's
    own correct glosses only reach token-F1 0.16 against gold, so a level would
    punish good paraphrase, whereas a margin asks the question that matters — did the
    trace pick out the right sense from the ones this word actually has.
    """
    g = _content(emitted)
    if not g:
        return 0.0
    gold_toks = _content(gold)
    rivals = [
        _token_f1(g, _content(s)) for s in senses if str(s).strip() != str(gold).strip()
    ]
    return _token_f1(g, gold_toks) - max(rivals, default=0.0)


def reward_wic_gloss(completions, **kwargs):
    """Score each emitted gloss against SemCor's gold sense for that usage.

    The verdict is one bit, and a bit is a weak verifier: rejection sampling at k=100
    satisfies ``same_sense`` on essentially every pair, and the reward is blind to a
    trace that reached the right answer with two glosses that describe the wrong
    senses (or the same sense twice — the shortcut ``reward_wic_consistency`` only
    catches when the verdict contradicts it). SemCor pairs carry the gold synset of
    *each* usage, so here the glosses themselves can be checked, against roughly 1.76
    bits per gloss rather than one bit for the pair.

    Per side the score is ``GLOSS_MAX * clamp(margin / GLOSS_FULL_MARGIN, -1, 1)``
    with ``margin`` from ``_gloss_margin``, and the two sides are averaged, so the
    term is bounded by ±GLOSS_MAX no matter how many senses the lemma has. It is
    graded rather than a hard argmax because neighbouring WordNet senses are often a
    near-tie, and a step function there would turn lexicographic hair-splitting into
    a cliff in the advantage.

    Returns 0.0 whenever the record has no gold gloss — MCL-WiC rows carry none — and
    whenever the lemma has a single candidate sense, where there is no rival to beat
    and the margin would collapse into the absolute-similarity score this term
    deliberately is not. So a rollout set mixing the two sources is scored on exactly
    the pairs that can be.
    """
    n = len(completions)
    golds1 = kwargs.get("gloss1") or [""] * n
    golds2 = kwargs.get("gloss2") or [""] * n
    inventories = kwargs.get("senses") or [[]] * n
    out = []
    for c, g1, g2, senses in zip(completions, golds1, golds2, inventories):
        obj = sd.parse_wic_answer(c)
        if obj is None or len(senses) < 2:
            out.append(0.0)
            continue
        scores = []
        for key, gold in (("sense1", g1), ("sense2", g2)):
            emitted = obj.get(key)
            if not isinstance(emitted, str) or not emitted.strip() or not gold:
                continue
            margin = _gloss_margin(emitted, gold, senses)
            scores.append(max(-1.0, min(1.0, margin / GLOSS_FULL_MARGIN)))
        out.append(GLOSS_MAX * sum(scores) / len(scores) if scores else 0.0)
    return out


REWARDS = [
    reward_wic_accuracy,
    reward_wic_format,
    reward_wic_json,
    reward_wic_consistency,
    reward_think_length,
    reward_wic_gloss,
]

# Columns build_dataset keeps on the rollout set: everything a reward reads as a
# kwarg. The gloss fields exist only on SemCor rows, so both trainers' format_prompt
# writes empty defaults for them — Arrow needs one schema across concatenated
# sources, and reward_wic_gloss reads an empty gold gloss as "not scoreable here".
GLOSS_COLS = {"gloss1": "", "gloss2": "", "senses": []}
KEEP_COLS = ["lemma", "pos", "label", *GLOSS_COLS]
