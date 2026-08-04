"""Chopping, whitespace and the target-span invariant across scripts.

The invariant here is the same one ``test_utils`` pins for SemCor: every
stage of ``prepare_lexeme`` moves text *and* the target's character offsets, so a
stage that shifts one without the other silently changes what the policy is asked
about. Each test below therefore checks the surviving span, not just the sentence.
"""

import re

import prepare_lexeme as pl


# A whitespace tokenizer standing in for Qwen3: `token_window` only needs offsets,
# and a real tokenizer would put a model download in the unit-test path.
class WhitespaceTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def roundtrip(text, start, end, lang="en", chop=True, detok=True):
    """Run one side through the pipeline, returning (marked, target)."""
    gold = text[start:end]
    if chop:
        text, start, end = pl.chop_to_sentence(text, start, end)
    text, start, end = pl.normalize_whitespace(
        text, start, end, lang in pl.UNSPACED_LANGS
    )
    assert text[start:end] == gold, (text[start:end], gold)
    marked = pl.mark_target(text, start, end)
    if detok:
        marked = pl.detokenize(marked, lang)
    return marked, gold


# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #
def test_splits_on_latin_terminators():
    text = "He read the book. She left. Nobody spoke."
    assert [text[a:b] for a, b in pl.sentence_spans(text)] == [
        "He read the book.",
        "She left.",
        "Nobody spoke.",
    ]


def test_does_not_split_a_decimal_or_an_unspaced_period():
    text = "The rate rose 3.5 points in J.P. Morgan's report."
    assert len(pl.sentence_spans(text)) == 1


def test_splits_cjk_and_danda_without_trailing_space():
    text = "内阁外另设立南书房。使之成为决策中心。"
    assert [text[a:b] for a, b in pl.sentence_spans(text)] == [
        "内阁外另设立南书房。",
        "使之成为决策中心。",
    ]
    bn = "শহরটি চা বাগান দ্বারা পরিবেষ্ঠিত৷ বাগডোগরা শিলিগুড়ি শহরের একটি অংশ৷"
    assert len(pl.sentence_spans(bn)) == 2


# --------------------------------------------------------------------------- #
# Chopping
# --------------------------------------------------------------------------- #
def test_chop_keeps_the_targets_own_sentence():
    text = (
        "Maxwell wrote it down. The girders lie on a shelf and are not fixed. It held."
    )
    start = text.index("shelf")
    kept, a, b = pl.chop_to_sentence(text, start, start + len("shelf"))
    assert kept == "The girders lie on a shelf and are not fixed."
    assert kept[a:b] == "shelf"


def test_chop_is_a_noop_on_a_single_sentence():
    text = "A political system with no place for the less prominent groups."
    start = text.index("place")
    assert pl.chop_to_sentence(text, start, start + 5) == (text, start, start + 5)


def test_chop_keeps_the_whole_context_when_the_span_straddles_a_boundary():
    # a target split by a sentence-final period keeps every character it needs
    text = "Founded by J. Smith. Ltd. grew fast."
    start, end = text.index("J. Smith. Ltd"), text.index("J. Smith. Ltd") + 13
    assert pl.chop_to_sentence(text, start, end) == (text, start, end)


# --------------------------------------------------------------------------- #
# Whitespace, per script
# --------------------------------------------------------------------------- #
def test_unspaced_context_loses_the_padding_around_its_target():
    # am2ico marks the target by spacing it out; in Chinese that padding is the
    # only whitespace in the sentence, so leaving it would flag the target
    text = "在  内阁 外另设立南书房"
    start = text.index("内阁")
    marked, gold = roundtrip(text, start, start + 2, lang="zh")
    assert marked == "在<t> 内阁 </t>外另设立南书房"
    assert pl.target_of(marked) == gold == "内阁"


def test_spaced_context_keeps_its_word_spaces():
    text = "жиры являются  одним из основных источников"
    start = text.index("одним")
    marked, gold = roundtrip(text, start, start + len("одним"), lang="ru")
    assert marked == "жиры являются <t> одним </t> из основных источников"
    assert pl.target_of(marked) == gold


# --------------------------------------------------------------------------- #
# Detokenization
# --------------------------------------------------------------------------- #
def test_ptb_context_is_detokenized_and_the_span_survives():
    text = "They include AT & T , beIN , Bharti Airtel , and the league 's teams ."
    start = text.index("league")
    marked, gold = roundtrip(text, start, start + len("league"))
    assert marked == (
        "They include AT & T, beIN, Bharti Airtel, and the <t> league </t>'s teams."
    )
    assert pl.target_of(marked) == gold == "league"


def test_french_typography_is_not_normalized_into_english():
    text = "Le jeu est fini ; il a gagné 50 % des points !"
    start = text.index("jeu")
    marked, _ = roundtrip(text, start, start + 3, lang="fr")
    assert marked == "Le <t> jeu </t> est fini ; il a gagné 50 % des points !"


def test_arabic_and_urdu_punctuation_closes_up():
    text = "درس في مدرسة الرازي الابتدائية ، ثم في مدرسة جد حفص ."
    start = text.index("مدرسة")
    marked, gold = roundtrip(text, start, start + len("مدرسة"), lang="ar")
    assert marked == "درس في <t> مدرسة </t> الرازي الابتدائية، ثم في مدرسة جد حفص."
    assert pl.target_of(marked) == gold


# --------------------------------------------------------------------------- #
# The token cap
# --------------------------------------------------------------------------- #
def test_token_window_bounds_the_length_and_keeps_the_target_centred():
    words = ["w%d" % i for i in range(60)]
    words[30] = "target"
    text = " ".join(words)
    start = text.index("target")
    out, a, b = pl.token_window(text, start, start + 6, 11, WhitespaceTokenizer())
    assert out[a:b] == "target"
    assert len(out.split()) == 11
    assert out.split()[5] == "target"


def test_token_window_never_cuts_a_target_bigger_than_the_budget():
    text = "lead in " + " ".join("t%d" % i for i in range(10)) + " tail"
    start = text.index("t0")
    end = text.index("t9") + 2
    out, a, b = pl.token_window(text, start, end, 4, WhitespaceTokenizer())
    assert out[a:b] == text[start:end]


def test_token_window_spends_the_whole_budget_at_an_edge():
    # a target at the start cannot use its left half, so the window grows right
    text = " ".join(["target"] + ["w%d" % i for i in range(40)])
    out, a, b = pl.token_window(text, 0, 6, 9, WhitespaceTokenizer())
    assert out[a:b] == "target"
    assert len(out.split()) == 9


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_parse_name_reads_split_language_and_corpus():
    from pathlib import Path

    assert pl.parse_name(Path("data/train/dev.ka-en.am2ico.data")) == (
        "dev",
        "ka",
        "en",
        "am2ico",
    )
    assert pl.parse_name(Path("data/train/train.en-en.mcl-wic.data")) == (
        "train",
        "en",
        "en",
        "mcl-wic",
    )


def test_pos_is_normalized_across_corpora():
    assert pl.normalize_pos("N") == "NOUN"  # xl
    assert pl.normalize_pos("VERB") == "VERB"  # mcl-wic
    assert pl.normalize_pos("None") is None  # am2ico, wic
    assert pl.normalize_pos(None) is None


def test_build_record_carries_provenance_and_marks_both_sides():
    example = {
        "id": "train.de-en.am2ico.7",
        "lemma": "bank",
        "pos": "None",
        "sentence1": "Er ging zur Bank . Dann kam er zurück .",
        "sentence2": "She sat on the river bank .",
        "start1": 12,
        "end1": 16,
        "start2": 21,
        "end2": 25,
    }
    rec = pl.build_record(
        example, True, "train", "train", "de", "en", "am2ico", chop=True
    )
    rec = pl.finish_record(rec, detok=True)
    assert rec["split"] == "train" and rec["source"] == "am2ico"
    assert (rec["lang1"], rec["lang2"]) == ("de", "en")
    assert rec["task"] == "wic" and rec["pos"] is None and rec["label"] is True
    assert rec["sentence1"] == "Er ging zur <t> Bank </t>."
    assert pl.target_of(rec["sentence1"]) == "Bank"
    assert pl.target_of(rec["sentence2"]) == "bank"
    assert "usage1" not in rec and "usage2" not in rec


def test_build_record_rejects_a_side_whose_offsets_do_not_land_on_a_word():
    example = {
        "id": "x.0",
        "lemma": "bank",
        "pos": "N",
        "sentence1": "Er ging zur Bank .",
        "sentence2": "She sat on the river bank .",
        "start1": 2,
        "end1": 3,  # a space
        "start2": 21,
        "end2": 25,
    }
    assert (
        pl.build_record(example, True, "train", "train", "de", "en", "am2ico", True)
        is None
    )


def test_dedupe_drops_repeats_and_every_copy_of_a_contradiction():
    def rec(u1, label):
        return {
            "lemma": "bank",
            "pos": "NOUN",
            "sentence1": u1,
            "sentence2": "b",
            "label": label,
        }

    records = [rec("a", True), rec("a", True), rec("c", True), rec("c", False)]
    kept, dup, conflict = pl.dedupe(records)
    assert [r["sentence1"] for r in kept] == ["a"]
    assert (dup, conflict) == (1, 2)


def test_dedupe_keeps_two_occurrences_of_one_lemma_in_the_same_sentence():
    # `a bank by the bank` is two items, not one contradiction: the marked
    # occurrence differs, so the labels are allowed to
    def rec(u1, label):
        return {
            "lemma": "bank",
            "pos": "NOUN",
            "sentence1": u1,
            "sentence2": "she sat on the <t> bank </t>",
            "label": label,
        }

    records = [
        rec("a <t> bank </t> by the bank", True),
        rec("a bank by the <t> bank </t>", False),
    ]
    kept, dup, conflict = pl.dedupe(records)
    assert len(kept) == 2 and (dup, conflict) == (0, 0)
