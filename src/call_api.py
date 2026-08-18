import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

import sense_data as sd

DEFAULT_MODEL_ID = "deepseek/deepseek-v4-flash-0731"
BASE_URL = "https://openrouter.ai/api/v1"
SAMPLES = 3  # self-consistency: k samples per pair, majority vote
MAX_WORKERS = 8  # concurrent pairs in flight
MAX_RETRIES = 4  # per-call retries on transient errors

SYSTEM_PROMPT = (
    "You are an expert lexicographer. You are given two sentences, each using the same "
    "target word (marked with <t> tags). Inside <think> tags, work out what the target "
    "word means in each sentence then compare the two senses. Then, after </think>, "
    "answer with a single JSON object and nothing else, with exactly these keys: "
    '"sense1" (string, the gloss of the target in sentence 1), "sense2" (string, the '
    'gloss of the target in sentence 2), and "same_sense" (boolean, true if the two '
    "uses share the same sense). "
    'Format: <think>...</think>\n{"sense1": ..., "sense2": ..., "same_sense": ...}'
)

USER_TEMPLATE = """\
Lemma: {lemma}
POS: {pos}

Sentence 1: "{sentence1}"
Sentence 2: "{sentence2}"

Are the two <t>...</t> uses the same sense?
"""


def _safe_mark(word: str, sentence: str) -> str:
    idx = sentence.find(word)
    if idx < 0:
        idx = sentence.lower().find(word.lower())
        if idx < 0:
            return sentence
        word = sentence[idx : idx + len(word)]
    return sentence[:idx] + "<t>" + word + "</t>" + sentence[idx + len(word) :]


def build_messages(
    lemma: str,
    pos: str,
    sentence1: str,
    sentence2: str,
) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                lemma=lemma,
                pos=pos,
                sentence1=sentence1,
                sentence2=sentence2,
            ),
        },
    ]


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _split_think(raw: str) -> tuple[str | None, str]:
    """Split a ``<think>…</think> {json}`` completion into (reasoning, json body).

    Returns the think-block text (or ``None`` when the model omits the tags) and the
    remaining content with the think block stripped out, so the body is left as the
    JSON answer for downstream ``json.loads``.
    """
    m = _THINK_RE.search(raw)
    if not m:
        return None, raw.strip()
    reasoning = m.group(1).strip()
    body = (raw[: m.start()] + raw[m.end() :]).strip()
    return reasoning, body


def _extract_json(body: str) -> str:
    """Best-effort pull of the JSON object out of a completion body.

    Strips ``` fences and returns the substring from the first ``{`` to the last
    ``}`` so any stray prose around the object doesn't break ``json.loads``.
    """
    body = (
        body.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    i, j = body.find("{"), body.rfind("}")
    return body[i : j + 1] if i != -1 and j > i else body


def _sample(
    client: OpenAI, model_id: str, messages: list[dict]
) -> tuple[str, str | None]:
    """One chat completion → (json_answer, reasoning), with retries on transient errors.

    The model is prompted to emit a ``<think>…</think>`` block before the JSON, so the
    reasoning is captured from either the provider's dedicated ``reasoning_content``
    field or, failing that, the inline think block. ``response_format`` is intentionally
    *not* constrained to ``json_object``: that mode suppresses the reasoning channel,
    which is the whole point of collecting these teacher traces.
    """
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
            )
            msg = resp.choices[0].message
            think, body = _split_think(msg.content or "")
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or think
            )
            return _extract_json(body), reasoning
        except Exception as e:  # network / rate-limit / server errors
            last_err = e
            time.sleep(2**attempt)
    raise last_err  # exhausted retries


_DECODER = json.JSONDecoder()


def _parse_object(content: str) -> dict | None:
    """The first complete JSON object in *content*, or None.

    `_extract_json` spans the first ``{`` to the last ``}``, which is right
    when prose surrounds one object and wrong when the model emits two: the
    span then covers both and `json.loads` raises "Extra data". Decoding just
    the first object recovers those -- three of the eight unparsed items in the
    300-pair quality pilot were a duplicated object, each copy valid on its
    own. Strictly more permissive than `json.loads`, never less.

    The other five are not recoverable here and should stay unparsed: three
    came back with an empty body (the provider put everything in the reasoning
    channel), one was truncated mid-object, and one leaked reasoning prose
    containing a stray ``{`` into the answer region.
    """
    i = content.find("{")
    if i < 0:
        return None
    try:
        obj, _ = _DECODER.raw_decode(content[i:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _vote(contents: list[str]) -> tuple[bool | None, float, list[bool | None]]:
    votes: list[bool | None] = []
    for content in contents:
        obj = _parse_object(content)
        try:
            v = bool(obj["same_sense"])
        except (KeyError, TypeError):
            v = None
        votes.append(v)
    valid = [v for v in votes if v is not None]
    if not valid:
        return None, 0.0, votes
    trues = sum(valid)
    pred = trues > len(valid) / 2
    if trues * 2 == len(valid):  # tie → no prediction
        return None, 0.5, votes
    confidence = max(trues, len(valid) - trues) / len(valid)
    return pred, confidence, votes


def _metrics(results: list[dict]) -> dict:
    scored = [
        r for r in results if r["prediction"] is not None and r["label"] is not None
    ]
    n = len(scored)
    correct = sum(int(r["prediction"]) == int(r["label"]) for r in scored)
    tp = sum(1 for r in scored if r["prediction"] and r["label"] == 1)
    fp = sum(1 for r in scored if r["prediction"] and r["label"] == 0)
    tn = sum(1 for r in scored if not r["prediction"] and r["label"] == 0)
    fn = sum(1 for r in scored if not r["prediction"] and r["label"] == 1)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    by_pos: dict[str, dict] = {}
    for r in scored:
        pos = r.get("pos", "?")
        d = by_pos.setdefault(pos, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += int(int(r["prediction"]) == int(r["label"]))
    for d in by_pos.values():
        d["accuracy"] = d["correct"] / d["n"]

    return {
        "n_scored": n,
        "n_skipped": len(results) - n,
        "accuracy": correct / n if n else 0.0,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "by_pos": by_pos,
    }


def _paths(name: str) -> dict:
    return {
        "results": Path(f"predictions_{name}.jsonl"),
        "metrics": Path(f"predictions_{name}_metrics.json"),
    }


def _make_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY for the OpenRouter endpoint.", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def _evaluate_pair(client: OpenAI, model_id: str, item: dict) -> dict:
    base = {
        "lemma": item["lemma"],
        "pos": item["pos"],
        "sentence1": item["sentence1"],
        "sentence2": item["sentence2"],
        "label": item["label"],
    }
    messages = build_messages(
        item["lemma"],
        item["pos"],
        item["sentence1"],
        item["sentence2"],
    )
    try:
        samples = [_sample(client, model_id, messages) for _ in range(SAMPLES)]
    except Exception as e:
        return {**base, "prediction": None, "error": str(e)}
    contents = [c for c, _ in samples]
    reasonings = [r for _, r in samples]
    prediction, confidence, votes = _vote(contents)
    return {
        **base,
        "prediction": prediction,
        "confidence": confidence,
        "votes": votes,
        "answers": contents,
        "reasonings": reasonings,
    }


def _pair_key(item: dict) -> tuple:
    return (item["lemma"], item["pos"], item["sentence1"], item["sentence2"])


# --------------------------------------------------------------------------- #
# assign: open-inventory sense assignment (see sense_assign_data.py)
# --------------------------------------------------------------------------- #
# Unlike the wic prompt above -- which is frozen inline so the existing teacher
# runs stay comparable -- this one is imported from `sense_data`, so the teacher
# and the local policy are asked byte-identical questions and any gap between
# them is the model rather than the phrasing.
def _vote_assign(contents: list[str]) -> tuple[int | None, float, list[int | None]]:
    """Majority vote over the sampled sense ids; ties and all-unparseable → None."""
    votes = [sd.extract_assign_id(c) for c in contents]
    valid = [v for v in votes if v is not None]
    if not valid:
        return None, 0.0, votes
    counts = Counter(valid)
    (top, n_top), *rest = counts.most_common()
    if rest and rest[0][1] == n_top:  # tie → no prediction
        return None, n_top / len(valid), votes
    return top, n_top / len(valid), votes


def _evaluate_assign(client: OpenAI, model_id: str, item: dict) -> dict:
    base = {
        k: item[k]
        for k in (
            "id", "lemma", "pos", "usage", "candidates",
            "gold_index", "gold_gloss", "held_out",
        )
        if k in item
    }
    if "nearest_rival_f1" in item:
        base["nearest_rival_f1"] = item["nearest_rival_f1"]
    messages = sd.assign_messages(item, with_target=False)
    try:
        samples = [_sample(client, model_id, messages) for _ in range(SAMPLES)]
    except Exception as e:
        return {**base, "prediction": None, "error": str(e)}
    contents = [c for c, _ in samples]
    prediction, confidence, votes = _vote_assign(contents)
    return {
        **base,
        "prediction": prediction,
        "confidence": confidence,
        "votes": votes,
        "answers": contents,
        "reasonings": [r for _, r in samples],
    }


def _assign_gloss(r: dict) -> str:
    """The gloss from a sample that agrees with the majority vote.

    Picking any sample's gloss would risk pairing the voted sense id with a gloss
    written for a different one; this mirrors `sense_data._wic_candidates`, which
    keeps only the samples consistent with the prediction.
    """
    for content in r.get("answers", []):
        if sd.extract_assign_id(content) == r.get("prediction"):
            return sd.extract_assign_gloss(content)
    return ""


def _assign_metrics(results: list[dict]) -> dict:
    """Score the teacher with `eval_assign`'s own scorer -- one implementation.

    The teacher and the trained policy are then reported on exactly the same
    quantities (known/novel accuracy, the novel-decision F1, the coined-gloss
    margin), which is the comparison this run exists to make.
    """
    import eval_assign as ea

    rows = [
        {
            "id": r.get("id", ""),
            "lemma": r.get("lemma", ""),
            "pos": r.get("pos", ""),
            "usage": r.get("usage", ""),
            "candidates": r.get("candidates", []),
            "gold_index": r.get("gold_index", 0),
            "gold_gloss": r.get("gold_gloss", ""),
            "held_out": r.get("held_out", False),
            "nearest_rival_f1": r.get("nearest_rival_f1"),
            "pred_index": r.get("prediction"),
            "gloss": _assign_gloss(r),
        }
        for r in results
        if "error" not in r
    ]
    # The paraphrase check needs sentence-transformers; it is the difference
    # between the strict and the lenient novel recall, and running without it
    # would report the floor as if it were the number. Failing soft keeps a long
    # API run from dying at the scoring step over a missing local model.
    try:
        sbert = ea._Sbert("sentence-transformers/all-MiniLM-L6-v2", "cpu")
    except Exception as e:
        print(f"[warn] no SBERT ({e}); novel recall is the strict floor", file=sys.stderr)
        sbert = None
    summary = ea.score_rows(rows, sbert=sbert)
    summary["n_errored"] = len(results) - len(rows)
    return summary


def _assign_key(item: dict) -> tuple:
    # the full rendered question: the same usage under the known and the novel
    # condition is two different items, and only `candidates` tells them apart
    return (item["lemma"], item["pos"], item["usage"], tuple(item["candidates"]))


# --------------------------------------------------------------------------- #
# quality: rubric-grade a raw pair, then answer it (see filter_quality.py)
# --------------------------------------------------------------------------- #
# `wic` asks the teacher to answer the question; this asks it to audit the
# question first. A corpus-derived pair carries noise the gold label cannot
# express -- OCR residue ("The * * f distance in * * f is 2.26 <t> A </t>"), a
# marker sitting on a token that isn't the lemma, a context too thin to fix the
# sense at all -- and such a pair teaches the policy a coin flip whatever its
# label says. So the axes are graded before the verdict, and the `same_sense`
# verdict at the end doubles as a label check: an item the teacher answers
# against gold is either mislabelled or genuinely contested.
#
# Each axis is one independent judgement about *one* sentence, never a summary
# score: "is this a good example" collapses unrelated failures into a number
# with no threshold, whereas `target_ok2 = false` names what to drop and why.
# The value below is the axis's *good* polarity, which is what lets the metrics
# and the filter read every axis the same way.
QUALITY_AXES = {
    "well_formed1": True,
    "well_formed2": True,
    "target_ok1": True,
    "target_ok2": True,
}

# Ordinals, kept out of QUALITY_AXES because they are thresholds rather than
# verdicts -- the filter picks the cut, the scorer only reports the level.
#
# `evidence` started as a boolean `ambiguous`, and it was dead on arrival: 0 of
# 220 items on sentence 1 and 3 of 220 on sentence 2. The bar ("a careful
# reader could not tell which sense is meant") is one almost no real sentence
# clears, so the axis measured nothing. Graded, the question becomes how much
# discriminating context there is, which every sentence has an answer to.
QUALITY_SCALES = {
    "evidence1": (1, 3),  # higher is better: how well the context pins the sense down
    "evidence2": (1, 3),
    "difficulty": (1, 5),  # lower is easier
}

QUALITY_SYSTEM = (
    "You are an expert lexicographer auditing a word-in-context dataset. Each item "
    "gives a lemma, its part of speech, and two sentences that each contain one "
    "occurrence of that lemma marked with <t> tags. Inside <think> tags, inspect the "
    "sentences one at a time -- is this usable English, is the right token marked, "
    "does the context actually pin the sense down -- and only then compare the two "
    "senses. After </think>, answer with a single JSON object and nothing else, with "
    "exactly these keys: "
    '"well_formed1", "well_formed2" (boolean, true if that sentence is fluent, '
    "complete English, free of corpus artifacts such as OCR residue, placeholder "
    "tokens, stray markup or a clause truncated mid-thought); "
    '"target_ok1", "target_ok2" (boolean, true if the <t> tags in that sentence mark '
    "a real occurrence of the lemma, used with the stated part of speech); "
    '"evidence1", "evidence2" (integer 1-3, how far that sentence alone pins down '
    "which sense of the lemma is meant: 1 the context is compatible with several "
    "senses and gives no cue between them, 2 it favours one sense but does not settle "
    "it, 3 the sense is unmistakable from this sentence); "
    '"difficulty" (integer 1-5: 1 the two uses are obviously the same or obviously '
    "different, 5 a judgement call trained lexicographers would split on); "
    '"sense1", "sense2" (string, the gloss of the marked word in each sentence); '
    '"same_sense" (boolean, true if the two uses share the same sense). '
    "Judge each sentence on its own -- the evidence in one sentence does not depend "
    "on how the other sentence uses the word. "
    'Format: <think>...</think>\n{"well_formed1": ..., ..., "same_sense": ...}'
)

QUALITY_TEMPLATE = """\
Lemma: {lemma}
POS: {pos}

Sentence 1: "{sentence1}"
Sentence 2: "{sentence2}"

Grade this item on every criterion, then decide whether the two <t>...</t> uses
are the same sense.
"""

# Written back by the evaluator, so re-scoring a results file doesn't feed a
# previous run's verdicts into the prompt as if they were corpus fields.
_QUALITY_OUTPUT_KEYS = (
    "quality", "prediction", "confidence", "votes", "answers", "reasonings", "error",
)


def _as_bool(value) -> bool | None:
    """Coerce one model-emitted flag to a bool, or None if it isn't one.

    Models occasionally quote the JSON booleans ("true"), which `json.loads`
    hands back as strings; everything else -- null, a number, a missing key --
    is a non-answer and must not be silently read as false.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _majority_bool(values: list) -> bool | None:
    """Majority over the samples for one axis; ties and all-invalid → None.

    The task runs at k=1, where this is just "the value, if it parsed as a
    bool"; it stays a vote so raising -k needs no other change.
    """
    valid = [b for b in (_as_bool(v) for v in values) if b is not None]
    if not valid:
        return None
    trues = sum(valid)
    if trues * 2 == len(valid):  # tie → no verdict, same convention as _vote
        return None
    return trues * 2 > len(valid)


def _median_level(values: list, lo: int, hi: int) -> int | None:
    """Median of the samples for one ordinal, clipped to its range.

    `bool` is an `int` in Python, so a flag leaking into an ordinal field would
    otherwise score as 0 or 1 -- a valid-looking level nobody asked for.
    """
    valid = sorted(
        int(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and lo <= v <= hi
    )
    return valid[len(valid) // 2] if valid else None  # median_low, k is small


def _evaluate_quality(client: OpenAI, model_id: str, item: dict) -> dict:
    base = {k: v for k, v in item.items() if k not in _QUALITY_OUTPUT_KEYS}
    messages = [
        {"role": "system", "content": QUALITY_SYSTEM},
        {
            "role": "user",
            "content": QUALITY_TEMPLATE.format(
                lemma=item["lemma"],
                pos=item["pos"],
                sentence1=item["sentence1"],
                sentence2=item["sentence2"],
            ),
        },
    ]
    try:
        samples = [_sample(client, model_id, messages) for _ in range(SAMPLES)]
    except Exception as e:
        return {**base, "prediction": None, "error": str(e)}
    contents = [c for c, _ in samples]
    objs = [_parse_object(c) for c in contents]

    quality = {
        axis: _majority_bool([o.get(axis) if o else None for o in objs])
        for axis in QUALITY_AXES
    }
    for scale, (lo, hi) in QUALITY_SCALES.items():
        quality[scale] = _median_level(
            [o.get(scale) if o else None for o in objs], lo, hi
        )
    quality["n_parsed"] = sum(o is not None for o in objs)
    # `same_sense` sits in the same object as the axes, so the wic vote counter
    # reads it unchanged -- one implementation of the majority rule.
    prediction, confidence, votes = _vote(contents)
    return {
        **base,
        "quality": quality,
        "prediction": prediction,
        "confidence": confidence,
        "votes": votes,
        "answers": contents,
        "reasonings": [r for _, r in samples],
    }


def _agreement(rows: list[dict]) -> dict:
    """Teacher-vs-gold agreement over rows that produced a usable prediction."""
    scored = [
        r for r in rows if r.get("prediction") is not None and r.get("label") is not None
    ]
    n = len(scored)
    correct = sum(int(r["prediction"]) == int(r["label"]) for r in scored)
    return {"n": n, "agreement": correct / n if n else 0.0}


def _quality_metrics(results: list[dict]) -> dict:
    """Flag rates, plus what each flag buys.

    The number that matters per axis is not how often it fires but the
    *agreement gap*: teacher-vs-gold agreement on the items it flags against
    the items it clears. An axis whose flagged items agree with gold as often
    as the clean ones is finding nothing worth dropping, however plausible it
    reads.
    """
    scored = [r for r in results if "error" not in r and "quality" in r]
    n = len(scored)
    overall = _agreement(scored)

    axes: dict[str, dict] = {}
    for axis, good in QUALITY_AXES.items():
        graded = [r for r in scored if r["quality"].get(axis) is not None]
        flagged = [r for r in graded if bool(r["quality"][axis]) != good]
        clean = [r for r in graded if bool(r["quality"][axis]) == good]
        axes[axis] = {
            "n_graded": len(graded),
            "n_flagged": len(flagged),
            "rate": len(flagged) / len(graded) if graded else 0.0,
            "flagged": _agreement(flagged),
            "clean": _agreement(clean),
        }

    def _is_flagged(r: dict) -> bool:
        q = r["quality"]
        return any(
            q.get(a) is not None and bool(q[a]) != good for a, good in QUALITY_AXES.items()
        )

    any_flagged = [r for r in scored if _is_flagged(r)]
    # Reported per level rather than as a correlation: the useful question is
    # where to put the cut, and a monotone drop across levels is what says a
    # cut exists at all.
    by_scale = {}
    for scale, (lo, hi) in QUALITY_SCALES.items():
        levels = {}
        for level in range(lo, hi + 1):
            rows = [r for r in scored if r["quality"].get(scale) == level]
            if rows:
                levels[level] = {"n": len(rows), **_agreement(rows)}
        by_scale[scale] = levels

    return {
        "n_scored": n,
        "n_errored": len(results) - n,
        "n_unparsed": sum(1 for r in scored if r["quality"].get("n_parsed") == 0),
        "overall": overall,
        "any_flag": {
            "n_flagged": len(any_flagged),
            "rate": len(any_flagged) / n if n else 0.0,
            "flagged": _agreement(any_flagged),
            "clean": _agreement([r for r in scored if not _is_flagged(r)]),
        },
        "axes": axes,
        "by_scale": by_scale,
    }


# A task bundles the three things run() varies on: how to query one item, how to
# key it for resume/caching, and how to score a batch of results. Adding a task
# elsewhere in the repo means adding an entry here.
TASKS = {
    "wic": {"evaluate": _evaluate_pair, "key": _pair_key, "metrics": _metrics},
    "assign": {
        "evaluate": _evaluate_assign,
        "key": _assign_key,
        "metrics": _assign_metrics,
    },
    "quality": {
        "evaluate": _evaluate_quality,
        "key": _pair_key,
        "metrics": _quality_metrics,
        # Self-consistency buys nothing here: the axes are near-deterministic
        # surface judgements, not the contested call `wic` votes on. One call
        # per item, k times cheaper over a corpus this size (override with -k).
        "samples": 1,
    },
}


def _load_resume(resume_path: str, key_fn) -> dict[tuple, dict]:
    """Map completed items (no error) from a previous results JSONL by their key."""
    done: dict[tuple, dict] = {}
    for line in Path(resume_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" not in r:
            done[key_fn(r)] = r
    print(
        f"Resuming: {len(done)} completed items loaded from {resume_path}",
        file=sys.stderr,
    )
    return done


def run(
    input_path: str,
    model_id: str = DEFAULT_MODEL_ID,
    resume_path: str | None = None,
    task: str = "wic",
    samples: int | None = None,
    workers: int = MAX_WORKERS,
) -> None:
    handlers = TASKS[task]
    global SAMPLES  # the evaluators read it directly; set once, before any call
    SAMPLES = samples or handlers.get("samples", SAMPLES)
    evaluate, key_fn, metrics_fn = (
        handlers["evaluate"],
        handlers["key"],
        handlers["metrics"],
    )
    path = Path(input_path)
    model_slug = model_id.replace("/", "_")
    p = _paths(f"{path.stem}_{model_slug}")
    data = json.loads(path.read_text())
    print(f"{len(data)} {task} items → {model_id} @ {BASE_URL}", file=sys.stderr)

    # Read any resume file before opening the output for writing, since the
    # two may be the same path and opening in "w" mode truncates it.
    resume = _load_resume(resume_path, key_fn) if resume_path else {}

    client = _make_client()
    results: list[dict] = []
    out = p["results"].open("w")

    def _write(r: dict) -> None:
        out.write(json.dumps(r) + "\n")
        out.flush()  # one line per pair, streamed — nothing buffered in memory
        results.append(r)

    interrupted = False
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {}
        for item in data:
            cached = resume.get(key_fn(item))
            if cached is not None:
                _write(cached)
            else:
                futures[pool.submit(evaluate, client, model_id, dict(item))] = item
        skipped = len(results)
        if skipped:
            print(
                f"  {skipped} items already done, {len(futures)} to go", file=sys.stderr
            )
        done = skipped
        for fut in as_completed(futures):
            _write(fut.result())
            done += 1
            if done % 25 == 0 or done == len(data):
                print(f"  {done}/{len(data)}", file=sys.stderr)
                p["metrics"].write_text(json.dumps(metrics_fn(results), indent=2))
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted — saving partial results …", file=sys.stderr)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        out.close()

    metrics = metrics_fn(results)
    p["metrics"].write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {len(results)} results to {p['results']}", file=sys.stderr)
    print(json.dumps(metrics, indent=2), file=sys.stderr)
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the input JSON file (WiC pairs, or assignment records "
        "from sense_assign_data.py with -t assign).",
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "-r",
        "--resume",
        help="Path to a previous (partial) results JSONL to continue from; "
        "already-completed items are skipped.",
    )
    parser.add_argument("-t", "--task", choices=list(TASKS), default="wic")
    parser.add_argument(
        "-k",
        "--samples",
        type=int,
        default=None,
        help="Self-consistency samples per item; default is per-task (3 for wic "
        "and assign, 1 for quality). The vote is a majority, so an even k ties "
        "out more often than an odd one.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help="Concurrent items in flight (default %(default)s). Transient "
        "rate-limit errors are retried with backoff, so overshooting costs "
        "latency rather than results.",
    )
    args = parser.parse_args()
    run(args.file, args.model, args.resume, args.task, args.samples, args.workers)
