"""Tests for the WordNet gloss checker.

`wn` is a project dependency, so these run under a plain ``uv run pytest`` like
the rest of the suite. The lexicon is a separate one-time download; when it is
missing these fail with `gloss_wordnet`'s LookupError, which names the command
that fixes it, rather than skipping themselves into green.

The checker no longer feeds a reward term — `reward_wic_wordnet` was removed in
favour of SemCor's gold sense annotations as SDPO privileged context (see
`sense_rewards`' module docstring). It remains the shared implementation behind the
offline calibration and error analysis under `analysis/`, so these tests pin the
behaviour those reports are read against.
"""

import gloss_wordnet as gw


FIN = "a financial institution that accepts deposits"
RIVER = "the sloping land beside a body of water"


def test_identical_glosses_under_different_label_are_rejected():
    """The shortcut cell: the trace never actually distinguished the senses."""
    assert gw.check("bank", "noun", FIN, FIN, False) == "wn_merged_diff_label"


def test_distinct_glosses_under_different_label_pass():
    assert gw.check("bank", "noun", FIN, RIVER, False) is None


def test_same_label_passes_by_default_even_when_glosses_diverge():
    """`wn_split_same_label` is off by default -- calibration found it
    anti-correlated with quality (the teacher trips it more than the student)."""
    assert gw.check("bank", "noun", FIN, RIVER, True) is None
    assert gw.check("bank", "noun", FIN, RIVER, True, strict=True) == "wn_split_same_label"


def test_unknown_lemma_always_passes():
    """Absence of evidence is not evidence of a bad gloss."""
    assert gw.check("zzqqxyz", "noun", "one thing", "another thing", False) is None


def test_unparseable_glosses_pass():
    """Those already die in `filter_reasoning.vote_check`; don't double-count."""
    assert gw.check("bank", "noun", None, FIN, False) is None
    assert gw.check("bank", "noun", "", FIN, False) is None


def test_anchoring_is_disabled_by_default():
    """MIN_ANCHOR default of 0 must not fire, however unlike WordNet the wording.

    A positive threshold rejects ~35% of the teacher's own good traces, so the
    default has to leave it off; this pins that.
    """
    assert gw.MIN_ANCHOR == 0.0
    weird = "zzz qqq vvv"  # shares no content token with any bank gloss
    assert gw.check("bank", "noun", weird, RIVER, True) is None
    assert gw.check("bank", "noun", weird, RIVER, True, min_anchor=0.1) == "gloss_unanchored"


def test_snap_returns_a_sense_of_the_right_lemma():
    sid, score = gw.snap(RIVER, "bank", "noun")
    assert sid is not None
    assert sid in {i for i, _ in gw.synsets_for("bank", "noun")}
    assert 0.0 < score <= 1.0


def test_adjectives_union_satellites():
    """OEWN keeps satellites in a disjoint pos 's'; querying only 'a' loses senses."""
    import wn

    w = wn.Wordnet(gw.LEXICON)
    plain = {s.id for s in w.synsets("direct", pos="a")}
    sats = {s.id for s in w.synsets("direct", pos="s")}
    assert sats and not (plain & sats), "expected 'a' and 's' to be disjoint"
    assert {i for i, _ in gw.synsets_for("direct", "adj")} == plain | sats


def test_multiword_lemmas_use_spaces_not_underscores():
    """`wn` spells multiword lemmas with spaces -- the opposite of NLTK."""
    assert gw.synsets_for("ice cream", "noun")
    assert gw.synsets_for("ice_cream", "noun")  # normalised internally


def test_check_matches_explain():
    d = gw.explain("bank", "noun", FIN, FIN, False)
    assert d["wn_same"] is True
    assert d["in_wordnet"] and d["n_synsets"] > 1
