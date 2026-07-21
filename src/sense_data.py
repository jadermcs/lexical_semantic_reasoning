"""Shared data + prompt utilities for the lexical-semantic reasoning tasks.

Every SFT record carries a ``task`` tag and is rendered to chat messages by a
task-specific builder (see ``build_messages``); ``sft_sense.py`` loads the prepared
prompt/completion set. One task lives here today:

* **wic** — given two sentences using the same target word (marked with <t> tags),
  decide whether both uses carry the *same* WordNet sense and gloss each usage. The
  verdict is a verifiable label, so GRPO can score it exactly (see
  ``sense_rewards.reward_wic_accuracy``). Record shape: ``lemma``, ``pos``, ``label``,
  ``usage1``, ``usage2`` (+ distilled ``think``/``sense1``/``sense2`` for SFT).

Data sources:

* ``load_mclwic`` — the gold MCL-WiC benchmark (``data/mcl-wic.<split>.json``).
  Carries the same/different label but no glosses. This is what GRPO rolls out
  against and what ``eval_sense.py`` scores.
* ``load_teacher_traces`` — teacher WiC predictions from ``call_api.py``, kept only
  where the teacher's self-consistency vote matched the gold label. These add a
  distilled ``think`` trace and the teacher's two sense glosses.

This module imports neither torch nor trl, so the data can be built and inspected
on the laptop; the training scripts import the heavy stack.
"""

import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

DATA_DIR = Path("data")

# --------------------------------------------------------------------------- #
# Prompt (shared by SFT / GRPO / eval so train and inference match exactly)
# --------------------------------------------------------------------------- #
WIC_SYSTEM = (
    "You are an expert lexicographer. You are given two sentences, each using the same "
    "target word (marked with <t> tags). Inside <think> tags, work out what the target "
    "word means in each sentence then compare the two senses. Then, after </think>, "
    "answer with a single JSON object and nothing else, with exactly these keys: "
    '"sense1" (string, the gloss of the target in sentence 1), "sense2" (string, the '
    'gloss of the target in sentence 2), and "same_sense" (boolean, true if the two '
    "uses share the same sense). "
    'Format: <think>...</think>\n{"sense1": ..., "sense2": ..., "same_sense": ...}'
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
# Data loading
# --------------------------------------------------------------------------- #
def pair_key(rec: dict) -> tuple:
    """Stable identity of a WiC pair, shared across every loader.

    Keys on ``(lemma, pos, sentence1, sentence2)`` — the raw, un-marked fields both
    ``load_mclwic`` and ``load_teacher_traces`` now carry — so a pair distilled into
    the SFT set can be recognised in the GRPO rollout source and held out of it.
    """
    return (rec["lemma"], rec["pos"], rec["sentence1"], rec["sentence2"])


def load_mclwic(
    split: str, data_dir: Path = DATA_DIR, exclude_pairs=None
) -> list[dict]:
    """Load the MCL-WiC benchmark split as internal wic records.

    MCL-WiC is a gold word-in-context dataset: two sentences, the target word's
    surface form in each, and a same/different label (1 = same sense, 0 =
    different) kept here as a boolean, matching the ``same_sense`` answer key.
    It carries no glosses, so only
    the verifiable same/different verdict can be scored — there is no gold gloss to
    reward the reasoning against. The two sentences aren't pre-tagged, so the
    surface form (``word1``/``word2``) is wrapped with <t> tags here to match the
    ``wic_messages`` prompt format.

    ``exclude_pairs`` is an optional set of ``pair_key`` tuples to drop — used to
    hold the SFT-consumed pairs out of the GRPO rollout set (avoiding rolling out on
    pairs the policy already memorised during SFT warm-start).
    """
    raw = json.loads((data_dir / f"mcl-wic.{split}.json").read_text())
    recs = [
        {
            "lemma": r["lemma"],
            "pos": r["pos"],
            "label": bool(r["label"]),
            # raw sentences kept so eval predictions can round-trip through
            # load_teacher_traces (which re-marks the target from the lemma)
            "sentence1": r["sentence1"],
            "sentence2": r["sentence2"],
            "usage1": mark_target(r["sentence1"], r["word1"]),
            "usage2": mark_target(r["sentence2"], r["word2"]),
        }
        for r in raw
    ]
    if exclude_pairs:
        exclude_pairs = set(exclude_pairs)
        recs = [r for r in recs if pair_key(r) not in exclude_pairs]
    return recs


def _wic_candidates(rec: dict) -> list[dict]:
    """Teacher samples whose own vote matches the majority prediction.

    ``call_api.py`` records one reasoning per self-consistency sample alongside its
    JSON answer (``{"sense1", "sense2", "same_sense"}``). Keeping only samples that
    voted with the majority makes the distilled <think> block consistent with the
    verdict we train toward. Each candidate carries the trimmed reasoning trace plus
    the two sense glosses from that same sample's answer, so the JSON training target
    can be reconstructed. Samples with an unparseable answer or an empty trace are
    dropped; the returned list preserves sample order.
    """
    pred = bool(rec["prediction"])
    cands = []
    for ans, rea in zip(rec.get("answers", []), rec.get("reasonings", [])):
        try:
            obj = json.loads(ans)
            same = bool(obj["same_sense"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if same != pred or not (rea and rea.strip()):
            continue
        cands.append(
            {
                "think": rea.strip(),
                "sense1": str(obj.get("sense1", "")).strip(),
                "sense2": str(obj.get("sense2", "")).strip(),
            }
        )
    return cands


def _select_candidate(cands, rec, strategy="first", scorer=None):
    """Pick one teacher sample from ``cands`` under the chosen ablation strategy.

    Task-neutral: every candidate carries a ``think`` field, which is all the
    ``first``/``longest`` strategies need.

    ``first``   keep the original behaviour: the earliest surviving sample.
    ``longest`` the sample with the longest reasoning trace (most CoT).
    ``entropy`` the sample the model is most uncertain about: ``scorer(rec, cands)``
                returns a per-candidate score (the model's mean predictive entropy
                over the trace) and the argmax wins. The scorer is supplied by the
                caller (``sft_sense``) since it needs the loaded model/tokenizer.
    """
    if strategy == "first":
        return cands[0]
    if strategy == "longest":
        return max(cands, key=lambda c: len(c["think"]))
    if strategy == "entropy":
        if scorer is None:
            raise ValueError("strategy='entropy' requires a scorer (pass one from sft_sense)")
        scores = scorer(rec, cands)
        return cands[max(range(len(cands)), key=scores.__getitem__)]
    raise ValueError(f"unknown reasoning-select strategy: {strategy!r}")


def load_teacher_traces(path: str | Path, strategy: str = "first", scorer=None) -> list[dict]:
    """Load teacher WiC predictions (from ``call_api.py``) as wic SFT records.

    Distillation source: each record carries the teacher's chain-of-thought
    (``reasonings``) and a self-consistency vote (``prediction``). Only pairs the
    teacher got right (vote == gold ``label``) are kept, so no confidently-wrong
    reasoning is distilled; the verdict trained toward is the gold label. Among the
    samples whose own vote agrees with the majority (see ``_wic_candidates``) one is
    picked per ``strategy`` (``first``/``longest``/``entropy`` — see
    ``_select_candidate``; ``entropy`` needs ``scorer``). The picked trace
    becomes the ``think`` field and its two sense glosses become ``sense1``/``sense2``
    for the JSON target (see ``wic_answer``). The trace already carries the <t> tags
    the teacher was shown, and the prompt sentences are re-marked from the lemma (the
    raw file drops the surface forms). Records with an errored/absent prediction or no
    usable trace are skipped.
    """
    raw = json.loads(Path(path).read_text())
    out = []
    for r in raw:
        if r.get("prediction") is None or r.get("label") is None:
            continue
        if bool(r["prediction"]) != bool(r["label"]):  # teacher-correct only
            continue
        cands = _wic_candidates(r)
        if not cands:
            continue
        chosen = _select_candidate(cands, r, strategy=strategy, scorer=scorer)
        out.append(
            {
                "task": "wic",
                "lemma": r["lemma"],
                "pos": r["pos"],
                "label": bool(r["label"]),
                # raw sentences kept so the SFT-consumed pairs can be keyed and held
                # out of the GRPO rollout set (see pair_key / prepare_data manifest)
                "sentence1": r["sentence1"],
                "sentence2": r["sentence2"],
                "usage1": mark_target(r["sentence1"], r["lemma"]),
                "usage2": mark_target(r["sentence2"], r["lemma"]),
                "think": chosen["think"],
                "sense1": chosen["sense1"],
                "sense2": chosen["sense2"],
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Prompt / target formatting (chat messages)
# --------------------------------------------------------------------------- #
def think_block(rec) -> str:
    """The distilled teacher trace, wrapped in <think> tags (shared by all tasks)."""
    return f"<think>\n{rec['think']}\n</think>"


# Kept as an alias for backward compatibility with earlier imports.
wic_think = think_block


def wic_answer(rec) -> str:
    """JSON verdict mirroring the teacher: sense gloss per usage + same_sense."""
    return json.dumps(
        {
            "sense1": rec.get("sense1", ""),
            "sense2": rec.get("sense2", ""),
            "same_sense": bool(rec["label"]),
        }
    )


def wic_messages(rec, with_target=False):
    """Chat messages for one pair.

    ``with_target`` appends the assistant turn (the SFT target), which needs the
    distilled ``think``/``sense1``/``sense2`` fields — i.e. a ``load_teacher_traces``
    record. Prompt-only rendering (``with_target=False``) works for any wic record,
    including the gloss-free MCL-WiC ones GRPO and eval use.
    """
    user = (
        f"Target word: {rec['lemma']} ({rec['pos']})\n\n"
        f"Sentence 1: {rec['usage1']}\n"
        f"Sentence 2: {rec['usage2']}\n\n"
        "Do both sentences use the target word in the same sense?"
    )
    msgs = [
        {"role": "system", "content": WIC_SYSTEM},
        {"role": "user", "content": user},
    ]
    if with_target:
        msgs.append(
            {"role": "assistant", "content": f"{think_block(rec)}\n{wic_answer(rec)}"}
        )
    return msgs


# Dispatch a record to its task's message builder via the ``task`` tag, so the SFT
# pipeline (``sft_sense.py``) stays task-agnostic and new tasks only need an entry
# here plus a loader above.
MESSAGE_BUILDERS = {"wic": wic_messages}


def build_messages(rec, with_target=False):
    """Render any tagged SFT record to chat messages via its task builder."""
    try:
        builder = MESSAGE_BUILDERS[rec["task"]]
    except KeyError:
        raise ValueError(f"unknown or missing task tag: {rec.get('task')!r}")
    return builder(rec, with_target=with_target)


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #
WIC_ANSWER_KEYS = {"sense1", "sense2", "same_sense"}


def _tok(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def parse_wic_answer(text: str) -> dict | None:
    """The JSON object in the answer region, or None if there isn't a parseable one.

    The answer region is everything after ``</think>``; an unclosed <think> means
    the reasoning ran past the budget with no answer at all. Only the object's
    *syntax* is checked here — whether it carries the right keys, and whether
    ``same_sense`` is a real boolean, is judged by the callers (``extract_wic_label``
    is lenient about it; ``sense_rewards.reward_wic_json`` scores it).
    """
    seg = text.split("</think>")[-1]
    if "<think>" in seg:  # unclosed <think>: reasoning ran on, no answer
        return None
    m = re.search(r"\{.*\}", seg, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_wic_label(text: str) -> bool | None:
    """Return the same-sense verdict as a boolean, or None if unclear.

    Prefer the JSON verdict (``{"same_sense": ...}``, coerced with ``bool`` so a
    stringly-typed "true" still yields a verdict); fall back to a bare
    same/different token for backward compatibility. Deliberately lenient: the
    accuracy reward should score a *decision* wherever the model expressed one, and
    the format/JSON rewards are what pay for expressing it in the required shape.
    """
    obj = parse_wic_answer(text)
    if obj is not None:
        try:
            return bool(obj["same_sense"])
        except (KeyError, TypeError):
            pass
    seg = text.split("</think>")[-1]
    if "<think>" in seg:
        return None
    m = re.search(r"\b(same|different)\b", seg, flags=re.IGNORECASE)
    return m.group(1).lower() == "same" if m else None
