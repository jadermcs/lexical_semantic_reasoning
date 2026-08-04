"""Detokenization and the target-span invariant.

Everything downstream (``sense_data.build_messages``, ``reward_wic_gloss``) reads the
``<t> ... </t>`` span, so a detokenizer that shifts a marker by one character
silently changes what the policy is asked about. These pin that it does not.
"""

import utils as u


def test_detokenize_ptb_spacing():
    raw = "Nine of the league 's teams play in baseball parks ."
    assert u.detokenize(raw) == "Nine of the league's teams play in baseball parks."


def test_detokenize_quotes_and_contractions():
    raw = "`` I do n't think so , '' he said ."
    assert u.detokenize(raw) == '"I don\'t think so," he said.'


def test_detokenize_brackets_and_escapes():
    raw = "The result -LRB- see Table 2 -RRB- was clear ."
    assert u.detokenize(raw) == "The result (see Table 2) was clear."


def test_detokenize_is_idempotent():
    raw = "But this is a public park and it 's a city ordinance ."
    once = u.detokenize(raw)
    assert u.detokenize(once) == once


def test_detokenize_is_idempotent_with_a_marker_against_punctuation():
    """The hard case: this function's own output glues the closing marker to what
    follows, and a plain whitespace split would read ``</t>".`` as one opaque token
    and lose the target entirely."""
    once = u.detokenize('`` Let \'s hear it , <t> anyway </t> . \'\'')
    assert u.target_of(once) == "anyway"
    assert u.detokenize(once) == once


def test_markers_survive_and_keep_their_spacing():
    raw = "But this is a public <t> park </t> and it 's an ordinance ."
    out = u.detokenize(raw)
    assert out == "But this is a public <t> park </t> and it's an ordinance."
    assert u.target_of(out) == "park"


def test_marker_adjacent_to_punctuation_does_not_absorb_it():
    raw = "Nine teams play in baseball <t> parks </t> ."
    out = u.detokenize(raw)
    assert u.target_of(out) == "parks"
    assert out.endswith("<t> parks </t>.")


def test_multiword_target_span_is_preserved():
    raw = "`` You <t> shut up </t> !"
    out = u.detokenize(raw)
    assert u.target_of(out) == "shut up"


def test_repeated_target_word_marks_the_annotated_occurrence():
    # `` bank `` occurs twice; the marker must stay on the second one.
    raw = "The bank of the river met the <t> bank </t> of the road ."
    out = u.detokenize(raw)
    assert out.count("<t>") == 1
    assert out.index("<t>") > out.index("bank of the river")
