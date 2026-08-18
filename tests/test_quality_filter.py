"""Unit tests for the rubric scorer's aggregation and the corpus filter.

The contract worth pinning is the difference between *not graded* and *graded
badly*. An axis the teacher declined to answer -- absent key, tie across
samples, unparseable JSON -- must never be read as a rejection, or the filter
silently deletes every item the teacher was merely quiet about.

Nothing here calls the API: `call_api` imports `openai` at module scope but
makes no client until `_make_client`.
"""

import json

import pytest

import call_api as C
import filter_quality as F


def _args(**over):
    """The filter's argparse defaults, overridable per test."""
    base = dict(
        skip_axis=[],
        max_difficulty=None,
        min_evidence=None,
        min_confidence=0.0,
        teacher_disagrees="drop",
    )
    base.update(over)
    return type("Args", (), base)


def _rec(**over):
    """A scored record that survives every rule, before *over* breaks one."""
    quality = {axis: good for axis, good in C.QUALITY_AXES.items()}
    quality.update(evidence1=3, evidence2=3, difficulty=2, n_parsed=1)
    quality.update(over.pop("quality", {}))
    rec = {
        "lemma": "bank",
        "pos": "noun",
        "sentence1": "he sat on the <t> bank </t> .",
        "sentence2": "she robbed the <t> bank </t> .",
        "label": 0,
        "quality": quality,
        "prediction": False,
        "confidence": 1.0,
    }
    rec.update(over)
    return rec


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def test_as_bool_accepts_quoted_booleans_only():
    assert C._as_bool(True) is True
    assert C._as_bool("false") is False
    assert C._as_bool(None) is None
    assert C._as_bool(1) is None  # a number is not a verdict
    assert C._as_bool("maybe") is None


def test_majority_bool_single_sample_is_the_sample():
    assert C._majority_bool([True]) is True
    assert C._majority_bool([None]) is None


def test_majority_bool_ties_abstain():
    assert C._majority_bool([True, False]) is None
    assert C._majority_bool([True, True, False]) is True


def test_median_level_ignores_booleans_and_out_of_range():
    # bools are ints in Python; a flag leaking into the field must not be a level
    assert C._median_level([True, False], 1, 5) is None
    assert C._median_level([1, 5, 3], 1, 5) == 3
    assert C._median_level([7, 0, 2], 1, 3) == 2  # only the in-range value survives
    assert C._median_level([], 1, 5) is None


# --------------------------------------------------------------------------- #
# rejection rules
# --------------------------------------------------------------------------- #
def test_clean_record_survives():
    assert F.reject(_rec(), _args()) is None


@pytest.mark.parametrize("axis,good", list(C.QUALITY_AXES.items()))
def test_each_axis_rejects_on_its_bad_polarity(axis, good):
    bad = _rec(quality={axis: not good})
    assert F.reject(bad, _args()) == axis
    assert F.reject(bad, _args(skip_axis=[axis])) is None


def test_ungraded_axis_is_not_a_rejection():
    assert F.reject(_rec(quality={"target_ok1": None}), _args()) is None


def test_unparsed_beats_every_other_rule():
    # nothing was graded, so no axis may be blamed for the drop
    rec = _rec(quality={"n_parsed": 0, "target_ok1": False}, prediction=None)
    assert F.reject(rec, _args()) == "unparsed"


def test_teacher_disagreement_is_optional():
    rec = _rec(prediction=True, label=0)
    assert F.reject(rec, _args()) == "teacher_disagrees"
    assert F.reject(rec, _args(teacher_disagrees="keep")) is None


def test_ordinal_and_confidence_thresholds_are_off_by_default():
    rec = _rec(quality={"difficulty": 5, "evidence2": 1}, confidence=0.34)
    assert F.reject(rec, _args()) is None
    assert F.reject(rec, _args(max_difficulty=4)) == "too_difficult"
    assert F.reject(rec, _args(min_evidence=2)) == "evidence2"
    assert F.reject(rec, _args(min_confidence=0.5)) == "low_confidence"


def test_ungraded_ordinal_is_not_a_rejection():
    rec = _rec(quality={"evidence1": None, "difficulty": None})
    assert F.reject(rec, _args(min_evidence=3, max_difficulty=1)) is None


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_metrics_report_the_agreement_gap_per_axis():
    # flagged items disagree with gold, clean ones agree: the gap is the point
    rows = [_rec() for _ in range(3)] + [
        _rec(quality={"target_ok2": False}, prediction=True, label=0) for _ in range(2)
    ]
    m = C._quality_metrics(rows)
    assert m["n_scored"] == 5
    axis = m["axes"]["target_ok2"]
    assert axis["n_flagged"] == 2
    assert axis["flagged"]["agreement"] == 0.0
    assert axis["clean"]["agreement"] == 1.0
    assert m["by_scale"]["difficulty"][2]["n"] == 5
    assert m["by_scale"]["evidence1"][3]["n"] == 5


def test_metrics_skip_errored_records():
    rows = [_rec(), {"lemma": "x", "prediction": None, "error": "boom"}]
    m = C._quality_metrics(rows)
    assert (m["n_scored"], m["n_errored"]) == (1, 1)


def test_evaluator_strips_a_previous_runs_verdicts():
    """Re-scoring a results file must not feed old verdicts back as corpus fields."""
    stale = _rec(answers=["{}"], reasonings=["old"], votes=[True])
    calls = []

    def fake_sample(client, model_id, messages):
        calls.append(messages)
        return json.dumps({**{a: g for a, g in C.QUALITY_AXES.items()},
                           "evidence1": 3, "evidence2": 3,
                           "difficulty": 1, "same_sense": False}), "trace"

    real, C._sample = C._sample, fake_sample
    try:
        out = C._evaluate_quality(None, "fake/model", stale)
    finally:
        C._sample = real

    assert len(calls) == C.SAMPLES
    assert out["quality"]["n_parsed"] == C.SAMPLES
    assert out["reasonings"] == ["trace"] * C.SAMPLES
    assert "old" not in json.dumps(out)


def test_quality_task_defaults_to_one_call_per_item():
    """The axes are surface judgements; a k=3 vote would triple the bill for nothing."""
    assert C.TASKS["quality"]["samples"] == 1
    assert "samples" not in C.TASKS["wic"]  # wic keeps the module default, k=3


def test_parse_object_takes_the_first_of_two_emitted_objects():
    """A duplicated answer must not read as unparseable.

    `_extract_json` hands over first-`{`-to-last-`}`, so when the model emits
    the object twice the span covers both and a plain `json.loads` raises
    "Extra data" -- 3 of 8 unparsed pilot items were exactly this.
    """
    doubled = '{"same_sense": true, "difficulty": 1}\n{"same_sense": true}'
    assert C._parse_object(doubled) == {"same_sense": True, "difficulty": 1}
    assert C._vote([doubled])[0] is True
    assert C._parse_object("no json here") is None
    assert C._parse_object("[1, 2]") is None  # an array is not an answer
    assert C._parse_object('{"a": 1} trailing prose') == {"a": 1}


def test_well_formed_is_off_by_default():
    """Its agreement gap is inverted at n=2025, so it must not delete by default."""
    args = F.build_parser().parse_args(["--preds", "x.jsonl"])
    assert args.skip_axis == ["well_formed1", "well_formed2"]
    assert F.reject(_rec(quality={"well_formed1": False}), args) is None
    # the flag with no values turns every axis back on
    on = F.build_parser().parse_args(["--preds", "x.jsonl", "--skip-axis"])
    assert on.skip_axis == []
    assert F.reject(_rec(quality={"well_formed1": False}), on) == "well_formed1"


def test_target_ok_still_filters_by_default():
    """The one boolean axis with a real gap (-0.19/-0.21) stays on."""
    args = F.build_parser().parse_args(["--preds", "x.jsonl"])
    assert F.reject(_rec(quality={"target_ok2": False}), args) == "target_ok2"
