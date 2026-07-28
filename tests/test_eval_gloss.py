"""CPU tests for the gloss-generation eval (no vLLM, no sacrebleu, no SBERT)."""

import json

import pytest

import eval_gloss as eg
import sense_data as sd


def rec(**kw):
    base = {
        "lemma": "bank",
        "word": "banks",
        "pos": "noun",
        "usage": "The river banks were flooded.",
        "definition": "sloping land beside a body of water",
    }
    return {**base, **kw}


def test_load_marks_the_surface_form_not_the_lemma(tmp_path):
    path = tmp_path / "gloss.json"
    path.write_text(json.dumps([rec()]))
    (loaded,) = eg.load_gloss_records(path)
    assert "<t> banks </t>" in loaded["marked"]
    assert loaded["usage"] == "The river banks were flooded."


def test_load_defaults_pos_and_keeps_pre_marked_usage(tmp_path):
    path = tmp_path / "gloss.json"
    marked = "He <t> banked </t> the plane."
    path.write_text(json.dumps([rec(usage=marked, word="banked", pos=None)]))
    (loaded,) = eg.load_gloss_records(path, default_pos="verb")
    assert loaded["pos"] == "verb"
    assert loaded["marked"] == marked  # already tagged: matcher must not re-tag


@pytest.mark.parametrize("missing", eg.REQUIRED_KEYS)
def test_load_rejects_missing_fields(tmp_path, missing):
    path = tmp_path / "gloss.json"
    path.write_text(json.dumps([rec(**{missing: ""})]))
    with pytest.raises(ValueError, match=missing):
        eg.load_gloss_records(path)


def test_prompt_puts_the_example_first_and_the_filler_second():
    wic = eg.build_wic_rec(rec(marked="The river <t> banks </t> were flooded."))
    user = sd.wic_messages(wic)[-1]["content"]
    assert "Sentence 1: The river <t> banks </t> were flooded." in user
    assert f"Sentence 2: {sd.mark_target(eg.FILLER_USAGE, eg.FILLER_WORD)}" in user
    # the target-word line names the record being scored, not the filler
    assert "Target word: bank (noun)" in user


def test_parse_glosses():
    decoded = (
        "<think>river-side land vs money place</think>\n"
        '{"sense1": "the land at the edge of a river", '
        '"sense2": "a financial institution", "same_sense": false}'
    )
    gloss, filler, think = eg.parse_glosses(decoded)
    assert gloss == "the land at the edge of a river"
    assert filler == "a financial institution"
    assert think == "river-side land vs money place"


def test_parse_glosses_unparseable():
    assert eg.parse_glosses("<think>ran out of budget")[0] == ""


def row(gloss, definition, filler_gloss="a financial institution", **kw):
    return {
        "lemma": "bank",
        "pos": "noun",
        "gloss": gloss,
        "definition": definition,
        "filler_gloss": filler_gloss,
        **kw,
    }


def test_score_rows_counts_empty_and_scores_the_rest():
    rows = [
        row("sloping land beside a body of water", "sloping land beside a body of water"),
        row("", "a financial institution"),
    ]
    summary = eg.score_rows(rows)
    assert summary["n"] == 2 and summary["n_scored"] == 1 and summary["empty"] == 1
    assert summary["f1"] == pytest.approx(1.0)


def test_filler_echo_is_visible_in_the_echo_rate():
    """A model that copies the constant sentence-2 gloss must not hide."""
    echo = [row("a financial institution", "sloping land beside a river")]
    honest = [row("sloping land beside a river", "sloping land beside a river")]
    assert eg.score_rows(echo)["filler_echo_rate"] == pytest.approx(1.0)
    assert eg.score_rows(honest)["filler_echo_rate"] == pytest.approx(0.0)


def test_records_whose_gold_sense_is_the_fillers_are_not_counted_as_echo():
    """Glossing a real 'financial institution' usage correctly is not a leak."""
    rows = [row("a financial institution", "a financial institution")]
    summary = eg.score_rows(rows)
    assert summary["filler_f1"] == pytest.approx(1.0)  # they do match ...
    import math

    assert math.isnan(summary["filler_echo_rate"])  # ... but the row is excluded


def test_sense_accuracy_only_scores_polysemous_lemmas():
    rows = [
        row("land at the edge of a river", "sloping land beside a river"),
        row("a financial institution that holds money", "a financial institution"),
        {**row("a domestic animal", "a domestic animal"), "lemma": "dog"},
    ]
    out = eg.sense_accuracy(rows, eg.token_f1)
    assert out["n_sense_scored"] == 2  # the single-sense 'dog' is excluded
    assert out["sense_acc"] == pytest.approx(1.0)


def test_sense_accuracy_catches_the_wrong_sense():
    rows = [
        # the river usage was glossed as the money sense: definition-shaped, wrong
        row("a financial institution for money", "sloping land beside a river"),
        row("an institution holding money", "a financial institution for money"),
    ]
    out = eg.sense_accuracy(rows, eg.token_f1)
    assert out["n_sense_scored"] == 2
    assert out["sense_acc"] == pytest.approx(0.5)
