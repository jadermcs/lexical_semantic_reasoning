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


def _tok(text):
    return re.findall(r"\w+", text.lower())


def _content_word_count(text):
    return sum(1 for t in _tok(text) if t not in ENGLISH_STOP_WORDS)


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
    return SequenceMatcher(None, _tok(s1), _tok(s2)).ratio()


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
                elif _tok(s1) == _tok(s2):
                    r = WIC_INCONSISTENT
        out.append(r)
    return out


REWARDS = [
    reward_wic_accuracy,
    reward_wic_format,
    reward_wic_json,
    reward_wic_consistency,
    reward_think_length,
]

KEEP_COLS = ["lemma", "pos", "label"]
