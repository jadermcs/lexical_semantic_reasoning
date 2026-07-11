"""Verifiable reward functions for the GRPO sense-modeling tasks.

Split out of ``grpo_sense`` so the rewards can be imported, unit-tested and
eyeballed without the training stack: like ``sense_data``, this module imports
neither torch nor trl at module level. The two models some rewards need
(BERTScore for gloss fidelity, a sentence encoder for the supersense soft tier)
are imported lazily inside their getters, so they cost nothing until a reward that
uses them is actually called.

Per task:

  direct/triplet -> BERTScore fidelity to the WordNet gold gloss, plus format,
                    length, self-reference and reasoning-quality terms.
  wic            -> correctness of the same/different verdict (exact: the gold
                    label is known) plus format and JSON-shape terms.
  supersense     -> correctness of the WordNet lexicographer-file label, with a
                    soft nearest-candidate tier for off-vocabulary answers.
"""

import json
import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import sense_data as sd


# --------------------------------------------------------------------------- #
# Gloss hygiene: don't define a word with itself, don't answer with nothing
# --------------------------------------------------------------------------- #
SELF_REF_PENALTY = -0.5

_VOWELS = "aeiou"


def _target_variants(lemma):
    """Lemma plus its common English inflections, lower-cased."""
    l = lemma.lower()
    v = {l, l + "s", l + "es", l + "ed", l + "ing", l + "d", l + "er", l + "est"}
    if l.endswith("e"):
        v |= {l[:-1] + "ing", l[:-1] + "ed", l + "r", l + "st"}
    if l.endswith("y") and len(l) > 1 and l[-2] not in _VOWELS:
        v |= {l[:-1] + "ies", l[:-1] + "ied", l[:-1] + "ier", l[:-1] + "iest"}
    # CVC words double the final consonant: run -> running, big -> bigger
    if (len(l) >= 3 and l[-1] not in _VOWELS + "wxy"
            and l[-2] in _VOWELS and l[-3] not in _VOWELS):
        d = l + l[-1]
        v |= {d + "ing", d + "ed", d + "er", d + "est"}
    return v


def _uses_target(gloss, lemma):
    """True if the gloss defines the target word using the word itself (or an inflection)."""
    return bool(set(sd._tok(gloss)) & _target_variants(lemma))


MIN_CONTENT_WORDS = 2    # glosses need at least this many non-stopword tokens
MIN_CONTENT_PENALTY = -0.5


def _content_word_count(gloss):
    return sum(1 for t in sd._tok(gloss) if t not in ENGLISH_STOP_WORDS)


def _min_content_penalty(gloss):
    """Punish glosses with fewer than MIN_CONTENT_WORDS non-stopword tokens."""
    return MIN_CONTENT_PENALTY if _content_word_count(gloss) < MIN_CONTENT_WORDS else 0.0


# --------------------------------------------------------------------------- #
# BERTScore fidelity (batched; the model is loaded once and reused)
# --------------------------------------------------------------------------- #
_BERTSCORER = None


def _get_bertscorer():
    global _BERTSCORER
    if _BERTSCORER is None:
        from bert_score import BERTScorer

        _BERTSCORER = BERTScorer(lang="en", rescale_with_baseline=True)
    return _BERTSCORER


def bertscore_similarity(hyps, refs):
    """Batched BERTScore F1 in [0,1]; empty hypotheses score 0 without calling the model."""
    out = [0.0] * len(hyps)
    idxs = [i for i, h in enumerate(hyps) if h.strip()]
    if not idxs:
        return out
    scorer = _get_bertscorer()
    _, _, f1 = scorer.score([hyps[i] for i in idxs], [refs[i] for i in idxs])
    for i, f in zip(idxs, f1.tolist()):
        out[i] = max(0.0, min(1.0, float(f)))
    return out


LENGTH_TOL = 1.2      # free allowance: up to 1.2x the gold gloss length
LENGTH_PENALTY = -1.5  # floor, reached only once the gloss is ~4x the budget
LENGTH_RAMP = 3.0      # budgets of excess the penalty ramps over before clamping


def _length_penalty(hyp, ref):
    """Penalise glosses longer than the gold definition; in [LENGTH_PENALTY, 0].

    The ramp is deliberately gentle and clamps far out (at ~4x the budget) so the
    reward stays monotone — strictly shorter is strictly better — across the whole
    range the model explores. A tight clamp instead flat-lines once a gloss is
    long, and under GRPO's group normalisation a length reward with no within-group
    variance yields no advantage, so runaway glosses stop being pushed back.
    """
    budget = LENGTH_TOL * max(len(sd._tok(ref)), 1)
    excess = max(0, len(sd._tok(hyp)) - budget) / budget
    return LENGTH_PENALTY * min(1.0, excess / LENGTH_RAMP)


def _answer_region(text):
    """The whole answer the model emits: all text after </think>, not just the
    first line that the gloss extractor scores. Shared by every task, which all use
    the same `<think>...</think>\\nanswer` format.

    The model reward-hacks by keeping a short first line and then dumping a long
    tail of repeated `definition: ...` lines; that tail is invisible to the
    first-line extractor, so length is measured over the entire region instead.
    """
    seg = text.split("</think>")[-1]
    if "<think>" in seg:  # unclosed <think>: there is no real answer region
        return ""
    return seg.strip()


def _gold_gloss(kwargs):
    """Gold gloss of the (shared) sense, whichever column the mode carries it in.

    Direct records store it as `gloss`; triplet records as `gloss_same` (the sense
    shared by the anchor and positive). One accessor lets the reward fns below stay
    mode-agnostic.
    """
    return kwargs["gloss_same"] if "gloss_same" in kwargs else kwargs["gloss"]


def reward_no_target(completions, **kwargs):
    """Punish glosses that define the target word using the word itself."""
    out = []
    for c, lemma in zip(completions, kwargs["lemma"]):
        out.append(SELF_REF_PENALTY if _uses_target(sd.extract_shared_gloss(c), lemma) else 0.0)
    return out


def reward_min_content(completions, **kwargs):
    """Punish glosses with fewer than MIN_CONTENT_WORDS non-stopword tokens."""
    return [_min_content_penalty(sd.extract_shared_gloss(c)) for c in completions]


def reward_length(completions, **kwargs):
    """Punish a definition region much longer than the gold definition.

    Measures the full post-</think> region (not just the extracted first line) so
    a short first line followed by a long repeated tail is still penalised.
    """
    return [
        _length_penalty(_answer_region(c), g)
        for c, g in zip(completions, _gold_gloss(kwargs))
    ]


def _think_answer_format_reward(completions, extractor):
    """Reward a present <think> block (0.1) and an extractable answer (0.1)."""
    out = []
    for c in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if extractor(c):
            r += 0.1
        out.append(r)
    return out


def reward_format(completions, **kwargs):
    return _think_answer_format_reward(completions, sd.extract_shared_gloss)


def reward_fidelity(completions, **kwargs):
    """Similarity of the emitted gloss to its gold definition."""
    hyps = [sd.extract_shared_gloss(c) for c in completions]
    return bertscore_similarity(hyps, _gold_gloss(kwargs))


def reward_triplet_contrast(completions, **kwargs):
    """Punish the gloss for being similar to the negative (differentia) sense.

    Complements reward_fidelity, which already rewards closeness to the
    positive sense, so together they pull the gloss toward the shared sense and
    away from the contrastive one.
    """
    hyps = [sd.extract_shared_gloss(c) for c in completions]
    sim_neg = bertscore_similarity(hyps, kwargs["gloss_diff"])
    return [-s for s in sim_neg]


def _extract_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


THINK_MIN_WORDS = 6      # below this many content words, reasoning is treated as a stub
THINK_MIN_PENALTY = -0.3


def _think_length_penalty(completions):
    """Punish degenerate reasoning: a missing, truncated, or near-empty <think> block.

    Catches the model collapsing the reasoning step — whether by dropping the
    block entirely (no penalty elsewhere, so this is the only force keeping it),
    truncating it past the length budget (no closing tag, so `_extract_think`
    returns ""), or stubbing it (e.g. "<think>ok</think>") to farm the format
    reward's presence bonus without doing any real reasoning.
    """
    out = []
    for c in completions:
        think = _extract_think(c)
        out.append(THINK_MIN_PENALTY if _content_word_count(think) < THINK_MIN_WORDS else 0.0)
    return out


def reward_think_length(completions, **kwargs):
    return _think_length_penalty(completions)


# --------------------------------------------------------------------------- #
# WiC (word-in-context): verifiable same/different-sense classification
# --------------------------------------------------------------------------- #
WIC_CORRECT = 1.0
WIC_WRONG = -1.0


def reward_wic_accuracy(completions, **kwargs):
    """+1 for the right same/different verdict, -1 for the wrong one, 0 if absent.

    This is the verifiable signal for wic: the gold label is known, so the reward
    is exact rather than a similarity estimate.
    """
    out = []
    for c, label in zip(completions, kwargs["label"]):
        pred = sd.extract_wic_label(c)
        if not pred:
            out.append(0.0)
        else:
            out.append(WIC_CORRECT if pred == label else WIC_WRONG)
    return out


def reward_wic_format(completions, **kwargs):
    """Reward a present <think> block (0.1) and an extractable verdict (0.1)."""
    return _think_answer_format_reward(completions, sd.extract_wic_label)


# Graded credit for the answer *shape* the prompt asks for: a single JSON object
# with exactly {"sense1", "sense2", "same_sense"} and a real boolean verdict.
# Graded rather than all-or-nothing so a group that is only partway there still has
# reward variance for GRPO to push on; the ceiling (0.3) stays well under the +/-1
# accuracy reward so being right always dominates being well-formatted.
WIC_JSON_PARSES = 0.1      # answer region holds a parseable JSON object
WIC_JSON_KEYS = 0.1        # exactly the three required keys, nothing extra or missing
WIC_JSON_BOOL = 0.1        # same_sense is a JSON boolean, not "true" / 1 / "same"
WIC_JSON_MALFORMED = -0.2  # said something after </think>, but no JSON object in it


def reward_wic_json(completions, **kwargs):
    """Score the answer region's JSON structure, in [WIC_JSON_MALFORMED, 0.3].

    ``reward_wic_format`` only pays for a verdict being *extractable*, and
    ``sd.extract_wic_label`` deliberately falls back to a bare same/different token —
    so on its own it lets the model collect format credit while never emitting JSON.
    This reward is the strict counterpart, and it is what makes the JSON contract
    (which the SFT warm-start teaches, see ``sd.wic_answer``) survive RL.

    A completion that answers in prose is penalised; a blank or unclosed <think>
    stays at 0, since ``reward_think_length`` already punishes that failure and
    double-charging it would drown out the accuracy signal.
    """
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


# --------------------------------------------------------------------------- #
# Supersense: verifiable WordNet lexicographer-file classification
# --------------------------------------------------------------------------- #
SUPERSENSE_CORRECT = 1.0
SUPERSENSE_WRONG = -1.0
# Soft credit for an off-vocabulary answer, mapped to its nearest candidate.
# Kept below |1| so an exact candidate is always the higher-reward move (when
# right) and the riskier one (when wrong) — no incentive to hedge with OOV text.
# Also kept below the default --distill-threshold (0.5) so soft (mis-formatted)
# hits never leak into the self-distillation set; only exact answers do.
SUPERSENSE_SOFT_CORRECT = 0.4
SUPERSENSE_SOFT_WRONG = -0.4


# Sentence-embedding nearest-candidate mapping for the soft tier. Lives here (not
# in sense_data, which stays free of even the lazy model deps) and loads on CPU to
# avoid competing with the policy/vLLM for VRAM — it only ever encodes a handful of
# short strings.
#
# The candidate *names* alone ("change", "creation") embed too abstractly to
# place an off-vocab paraphrase near the right one, so each is anchored by its
# standard WordNet lexicographer-file description (e.g. "biological growth" then
# lands on "change" rather than "creation"). The returned label is still the bare
# suffix; only the embedding text is enriched.
_SUPERSENSE_DESC = {
    "noun": {
        "act": "nouns denoting acts or actions",
        "animal": "nouns denoting animals",
        "artifact": "nouns denoting man-made objects",
        "attribute": "nouns denoting attributes of people and objects",
        "body": "nouns denoting body parts",
        "cognition": "nouns denoting cognitive processes and contents",
        "communication": "nouns denoting communicative processes and contents",
        "event": "nouns denoting natural events",
        "feeling": "nouns denoting feelings and emotions",
        "food": "nouns denoting foods and drinks",
        "group": "nouns denoting groupings of people or objects",
        "location": "nouns denoting spatial position",
        "motive": "nouns denoting goals",
        "object": "nouns denoting natural objects (not man-made)",
        "person": "nouns denoting people",
        "phenomenon": "nouns denoting natural phenomena",
        "plant": "nouns denoting plants",
        "possession": "nouns denoting possession and transfer of possession",
        "process": "nouns denoting natural processes",
        "quantity": "nouns denoting quantities and units of measure",
        "relation": "nouns denoting relations between people or things or ideas",
        "shape": "nouns denoting two and three dimensional shapes",
        "state": "nouns denoting stable states of affairs",
        "substance": "nouns denoting substances",
        "time": "nouns denoting time and temporal relations",
    },
    "verb": {
        "body": "verbs of grooming, dressing and bodily care",
        "change": "verbs of size, temperature change, intensifying",
        "cognition": "verbs of thinking, judging, analyzing, doubting",
        "communication": "verbs of telling, asking, ordering, singing",
        "competition": "verbs of fighting, athletic activities",
        "consumption": "verbs of eating and drinking",
        "contact": "verbs of touching, hitting, tying, digging",
        "creation": "verbs of sewing, baking, painting, performing",
        "emotion": "verbs of feeling",
        "motion": "verbs of walking, flying, swimming",
        "perception": "verbs of seeing, hearing, feeling",
        "possession": "verbs of buying, selling, owning",
        "social": "verbs of political and social activities and events",
        "stative": "verbs of being, having, spatial relations",
        "weather": "verbs of raining, snowing, thawing, thundering",
    },
}
_ENCODER = None
_CAND_EMB = {}


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _ENCODER


def _candidate_embeddings(pos):
    """Normalized embeddings of ``pos``'s candidates (via description), once."""
    if pos not in _CAND_EMB:
        cands = sd.SUPERSENSES[pos]
        texts = [_SUPERSENSE_DESC[pos][c] for c in cands]
        emb = _get_encoder().encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        _CAND_EMB[pos] = (cands, emb)
    return _CAND_EMB[pos]


def _nearest_candidates(regions, poss):
    """The semantically nearest candidate name for each (answer region, pos)."""
    q = _get_encoder().encode(regions, convert_to_tensor=True, normalize_embeddings=True)
    out = []
    for i, pos in enumerate(poss):
        cands, emb = _candidate_embeddings(pos)
        out.append(cands[int((emb @ q[i]).argmax())])  # cosine: both normalized
    return out


def reward_supersense_accuracy(completions, **kwargs):
    """Graded reward over the closed candidate set.

    An exact candidate scores +1/-1 as before (verifiable: the gold lexicographer
    file is known). An answer that names no candidate but still says something
    ("biological growth" for gold "change") is mapped to its nearest candidate by
    sentence-embedding similarity and given the smaller +/-0.4; a blank or
    unclosed <think> stays 0.

    The soft tier matters under GRPO: when no completion in a group emits an exact
    label, an all-zero group has no advantage and teaches nothing, so the policy
    is never pulled from its free-form paraphrase toward the exact label word.
    """
    labels, poss = kwargs["supersense"], kwargs["pos"]
    out = [0.0] * len(completions)
    soft_idx, soft_regions, soft_pos = [], [], []
    for i, (c, label, pos) in enumerate(zip(completions, labels, poss)):
        pred = sd.extract_supersense(c, pos)
        if pred:
            out[i] = SUPERSENSE_CORRECT if pred == label else SUPERSENSE_WRONG
        elif _answer_region(c):  # said something, but no exact candidate in it
            soft_idx.append(i)
            soft_regions.append(_answer_region(c))
            soft_pos.append(pos)
        # else: blank / unclosed -> stays 0.0
    if soft_idx:
        for i, near in zip(soft_idx, _nearest_candidates(soft_regions, soft_pos)):
            out[i] = SUPERSENSE_SOFT_CORRECT if near == labels[i] else SUPERSENSE_SOFT_WRONG
    return out


def reward_supersense_format(completions, **kwargs):
    """Reward a present <think> block (0.1) and an extractable category (0.1).

    Unlike the other format rewards, the extractor is POS-aware, so it cannot use
    the shared ``_think_answer_format_reward`` (which passes a single-arg extractor).
    """
    out = []
    for c, pos in zip(completions, kwargs["pos"]):
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if sd.extract_supersense(c, pos):
            r += 0.1
        out.append(r)
    return out


REWARDS = {
    "direct": [
        reward_fidelity, reward_no_target, reward_min_content,
        reward_length, reward_format, reward_think_length,
    ],
    "triplet": [
        reward_fidelity, reward_triplet_contrast, reward_no_target,
        reward_min_content, reward_length, reward_format, reward_think_length,
    ],
    "wic": [
        reward_wic_accuracy, reward_wic_format, reward_wic_json, reward_think_length,
    ],
    "supersense": [
        reward_supersense_accuracy, reward_supersense_format, reward_think_length,
    ],
}
KEEP_COLS = {
    "direct": ["lemma", "gloss"],
    "triplet": ["lemma", "gloss_same", "gloss_diff"],
    # MCL-WiC carries no glosses; only the verifiable same/different label is scored.
    "wic": ["lemma", "label"],
    # pos selects the candidate label space for extraction; supersense is the gold label.
    "supersense": ["lemma", "pos", "supersense"],
}
# The fidelity reward already scores a completion against its gold gloss(es); the
# trace saver reuses it to decide which generations are "successful". For wic and
# supersense, "successful" means the verifiable label is correct, so accuracy plays
# the fidelity role.
FIDELITY = {
    "direct": reward_fidelity,
    "triplet": reward_fidelity,
    "wic": reward_wic_accuracy,
    "supersense": reward_supersense_accuracy,
}


# --------------------------------------------------------------------------- #
# Multitask: run a task's reward fn only on that task's completions
# --------------------------------------------------------------------------- #
def mask_by_task(task, fn):
    """Wrap a reward fn so it scores only rows whose ``task`` column matches.

    Under GRPO each prompt's ``num_generations`` completions form one group and
    all share a task, so the other tasks' reward fns return all-zero for the group
    — a constant reward yields zero advantage after group normalisation, leaving
    training untouched. Non-matching rows carry padded ("") gold columns from the
    schema union, so they must be filtered out before the wrapped fn reads them.
    """
    def wrapped(completions, **kwargs):
        tasks = kwargs["task"]
        idxs = [i for i, t in enumerate(tasks) if t == task]
        out = [0.0] * len(completions)
        if not idxs:
            return out
        sub_completions = [completions[i] for i in idxs]
        sub_kwargs = {
            k: ([v[i] for i in idxs] if isinstance(v, list) and len(v) == len(completions) else v)
            for k, v in kwargs.items()
        }
        for i, s in zip(idxs, fn(sub_completions, **sub_kwargs)):
            out[i] = s
        return out

    # GRPOTrainer logs/derives metric names from the fn name; keep them unique.
    wrapped.__name__ = f"{task}_{getattr(fn, '__name__', 'reward')}"
    return wrapped


# --------------------------------------------------------------------------- #
# Self-distillation: log successful reasoning traces for later SFT
# --------------------------------------------------------------------------- #
def make_trace_saver(mode, path, threshold):
    """Pass-through reward fn that appends high-reward completions to a JSONL.

    Returns all-zero rewards, so it never perturbs training (a constant reward
    yields zero advantage after GRPO's group normalisation). It records the
    prompt + generated reasoning/gloss + fidelity score + gold columns; a
    downstream builder can turn the best generations into an SFT dataset.

    Writes one file per process rank to avoid interleaved appends under
    multi-GPU launches.
    """
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    out_path = Path(f"{path}.rank{rank}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = KEEP_COLS[mode]
    fidelity = FIDELITY[mode]

    def save_successful_traces(completions, **kwargs):
        scores = fidelity(completions, **kwargs)
        prompts = kwargs.get("prompts")
        with out_path.open("a", encoding="utf-8") as f:
            for i, (c, s) in enumerate(zip(completions, scores)):
                if s < threshold:
                    continue
                rec = {"mode": mode, "score": round(float(s), 4), "completion": c}
                if prompts is not None:
                    rec["prompt"] = prompts[i]
                for col in keep:
                    rec[col] = kwargs[col][i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return [0.0] * len(completions)

    # GRPOTrainer logs/derives metric names from the fn name.
    save_successful_traces.__name__ = "save_successful_traces"
    return save_successful_traces
