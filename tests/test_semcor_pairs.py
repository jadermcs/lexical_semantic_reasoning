"""Pair construction and the feedback contract it has to honour.

The one test that matters most here is ``test_no_test_lemma_leakage``: the whole
point of a held-out MCL-WiC test score is that the policy has not been trained on
those lemmas' sense inventories, and SemCor shares vocabulary with MCL-WiC even
though it shares no pairs.
"""

import json

import pytest

import semcor_pairs as sp
import sense_data as sd

pytest.importorskip("nltk")

SOURCE = sp.DEFAULT_SOURCE
needs_source = pytest.mark.skipif(
    not SOURCE.exists(), reason=f"{SOURCE} not present in this checkout"
)


def _wordnet_available():
    try:
        sp._wn()
    except Exception:
        return False
    return True


needs_wordnet = pytest.mark.skipif(
    not _wordnet_available(), reason="NLTK WordNet corpus not downloaded"
)


# --------------------------------------------------------------------------- #
# Marking and schema
# --------------------------------------------------------------------------- #
def test_mark_usage_uses_exact_span_not_fuzzy_match():
    """The offsets are the reason this source beats re-deriving pairs by matching."""
    ann = {"text": "He set the table and set out.", "start": 3, "end": 6}
    assert sp.mark_usage(ann) == "He <t> set </t> the table and set out."


def test_mark_usage_spacing_matches_sense_data():
    """A SemCor prompt and an MCL-WiC prompt must be indistinguishable in shape."""
    ann = {"text": "The bank was closed.", "start": 4, "end": 8}
    assert sp.mark_usage(ann) == sd.mark_target("The bank was closed.", "bank")


def test_pos_uses_repo_spelling_not_wordnet_long_form():
    """``adj``/``adv``, the keys MCL-WiC and gloss_wordnet._POS both use.

    The older pair file spelled these ``adjective``/``adverb``, which
    ``_POS.get(pos, ()) or (None,)`` degrades to a POS-blind synset lookup.
    """
    gloss_wordnet = pytest.importorskip("gloss_wordnet")

    assert set(sp.POS_MAP.values()) == {"noun", "verb", "adj", "adv"}
    for pos in sp.POS_MAP.values():
        assert pos in gloss_wordnet._POS


# --------------------------------------------------------------------------- #
# Feedback contract
# --------------------------------------------------------------------------- #
@needs_wordnet
def test_same_sense_feedback_states_one_gloss():
    """Reward-driven: ``reward_wic_consistency`` scores a true verdict as
    ``WIC_INCONSISTENT * (1 - gloss_similarity)``, so the hint must not push the two
    glosses apart. It states the shared gloss once and never asks for variation.
    """
    rec = {"lemma": "dog", "pos": "noun", "label": True,
           "synset1": "dog.n.01", "synset2": "dog.n.01"}
    fb = sp.gloss_feedback(rec)
    assert '"same_sense": true' in fb
    assert fb.count(sp.gloss("dog.n.01")) == 1
    for banned in ("own words", "do not repeat", "differently"):
        assert banned not in fb.lower()


@needs_wordnet
def test_different_sense_feedback_gives_both_glosses():
    rec = {"lemma": "bank", "pos": "noun", "label": False,
           "synset1": "bank.n.01", "synset2": "bank.n.02"}
    fb = sp.gloss_feedback(rec)
    assert '"same_sense": false' in fb
    assert sp.gloss("bank.n.01") in fb and sp.gloss("bank.n.02") in fb


@needs_wordnet
def test_feedback_falls_back_when_synset_unresolvable():
    """An adjective satellite that WordNet 3.0 will not resolve costs its gloss, not
    the pair — the hint degrades to the one-bit verdict MCL-WiC rows already use."""
    rec = {"lemma": "regional", "pos": "adj", "label": False,
           "synset1": "not.a.real.synset", "synset2": "bank.n.02"}
    fb = sp.gloss_feedback(rec)
    assert '"same_sense": false' in fb
    assert "different senses" in fb


@needs_wordnet
def test_feedback_never_reveals_itself():
    rec = {"lemma": "dog", "pos": "noun", "label": True,
           "synset1": "dog.n.01", "synset2": "dog.n.01"}
    assert "Do not mention this hint" in sp.gloss_feedback(rec)


# --------------------------------------------------------------------------- #
# Pair construction over the real source
# --------------------------------------------------------------------------- #
@needs_source
@needs_wordnet
def test_no_test_lemma_leakage():
    """No SemCor pair may use a lemma occurring in mcl-wic.test.

    Without this the held-out score measures recall of a sense inventory the policy
    was trained on, not generalisation to an unseen word.
    """
    recs = sp.build_pairs(SOURCE, max_per_lemma=1)
    blocked = sp.held_out_lemmas(("test",))
    assert blocked, "expected mcl-wic.test to contribute lemmas"
    assert not {r["lemma"].lower() for r in recs} & blocked


@needs_source
@needs_wordnet
def test_keeping_test_lemmas_is_opt_in():
    """The guard is a default, not a hard wall — but disabling it must be explicit."""
    leaky = sp.build_pairs(SOURCE, max_per_lemma=1, exclude_lemma_splits=())
    assert {r["lemma"].lower() for r in leaky} & sp.held_out_lemmas(("test",))


@needs_source
@needs_wordnet
def test_pairs_are_well_formed_and_balanced():
    recs = sp.build_pairs(SOURCE, max_per_lemma=1)
    assert recs
    assert sum(r["label"] for r in recs) * 2 == len(recs)
    for r in recs:
        assert r["sentence1"] != r["sentence2"]
        assert "<t>" in r["usage1"] and "<t>" in r["usage2"]
        assert r["label"] == (r["synset1"] == r["synset2"])
        assert sd.pair_key(r)  # keyed like every other loader's records


@needs_wordnet
def test_sense_inventory_covers_the_lemmas_senses_and_dedups():
    senses = sp.sense_inventory("bank", "noun")
    assert len(senses) == len(set(senses))
    assert sp.gloss("bank.n.01") in senses
    assert sp.gloss("depository_financial_institution.n.01") in senses


@needs_wordnet
def test_sense_inventory_includes_adjective_satellites():
    """SemCor annotates satellites, so a POS-blind or 'a'-only lookup would leave the
    gold gloss out of its own candidate set — an unwinnable ranking."""
    assert sp.gloss("direct.s.01") in sp.sense_inventory("direct", "adj")


@needs_source
@needs_wordnet
def test_records_carry_gold_glosses_inside_their_candidate_set():
    """The contract ``sense_rewards.reward_wic_gloss`` scores against: a gold gloss per
    usage, and an inventory to rank it in that actually contains it."""
    recs = sp.build_pairs(SOURCE, max_per_lemma=1)
    scoreable, blanked = 0, 0
    for r in recs:
        for g in (r["gloss1"], r["gloss2"]):
            assert not g or g in r["senses"]
        if r["gloss1"]:
            # A different-sense pair whose two synsets share one definition string
            # would ask for a distinction WordNet does not make, so it is blanked
            # rather than kept; anything still carrying glosses is the real gloss.
            assert r["gloss1"] == sp.gloss(r["synset1"])
            assert r["gloss2"] == sp.gloss(r["synset2"])
            assert r["label"] == (r["gloss1"] == r["gloss2"])
        else:
            blanked += 1
        scoreable += bool(r["gloss1"] and r["gloss2"] and len(r["senses"]) > 1)
    assert blanked < 0.02 * len(recs)
    # The rest of the shortfall is monosemous lemmas, which reward_wic_gloss skips for
    # want of a rival sense (~19%). The bulk of the set still has to be scoreable or
    # the reward is decoration.
    assert scoreable > 0.75 * len(recs)


@needs_source
@needs_wordnet
def test_min_confusability_restricts_the_hard_negatives():
    recs = sp.build_pairs(SOURCE, max_per_lemma=2, min_confusability="near")
    levels = {
        sp.confusability(r["synset1"], r["synset2"]) for r in recs if not r["label"]
    }
    assert levels <= {"siblings", "near"}
