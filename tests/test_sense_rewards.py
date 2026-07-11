"""Unit tests for the GRPO reward functions.

These pin the *contract* each reward enforces, which is what stops a silent
reward-hack from creeping back in: the failure modes named in the reward
docstrings (a stubbed <think>, a long repeated tail, a prose answer instead of
JSON, defining a word with itself) each have a test here asserting they are
actually punished.

The two model-backed rewards (``reward_fidelity``/``reward_triplet_contrast`` via
BERTScore, and the supersense soft tier via a sentence encoder) are exercised with
their model calls monkeypatched, so the suite runs on CPU in a second and needs no
downloads.
"""

import json

import pytest

import sense_data as sd
import sense_rewards as R


def wrap(think, answer):
    """A completion in the format every task uses: <think>...</think>\\nanswer."""
    return f"<think>\n{think}\n</think>\n{answer}"


GOOD_THINK = "the first usage names a financial institution while the second names a river edge"
JSON_ANSWER = json.dumps({"sense1": "a financial institution", "sense2": "sloping land", "same_sense": False})


# --------------------------------------------------------------------------- #
# wic: JSON answer shape
# --------------------------------------------------------------------------- #
class TestWicJson:
    def test_well_formed_answer_gets_full_credit(self):
        c = wrap(GOOD_THINK, JSON_ANSWER)
        assert R.reward_wic_json([c]) == [
            pytest.approx(R.WIC_JSON_PARSES + R.WIC_JSON_KEYS + R.WIC_JSON_BOOL)
        ]

    def test_stringly_typed_verdict_loses_the_boolean_credit(self):
        # {"same_sense": "true"} parses and has the right keys, but the verdict is
        # not a JSON boolean, so it forfeits WIC_JSON_BOOL.
        c = wrap(GOOD_THINK, '{"sense1": "a", "sense2": "b", "same_sense": "true"}')
        assert R.reward_wic_json([c]) == [pytest.approx(R.WIC_JSON_PARSES + R.WIC_JSON_KEYS)]

    def test_integer_verdict_loses_the_boolean_credit(self):
        # 1 is truthy and bool(1) is True, so extract_wic_label still reads a
        # verdict off it -- the JSON reward is what makes it a real boolean.
        c = wrap(GOOD_THINK, '{"sense1": "a", "sense2": "b", "same_sense": 1}')
        assert R.reward_wic_json([c]) == [pytest.approx(R.WIC_JSON_PARSES + R.WIC_JSON_KEYS)]

    @pytest.mark.parametrize(
        "answer",
        [
            '{"same_sense": true}',                                              # missing sense keys
            '{"sense1": "a", "sense2": "b", "same_sense": true, "conf": 0.9}',   # extra key
        ],
    )
    def test_wrong_key_set_loses_the_keys_credit(self, answer):
        c = wrap(GOOD_THINK, answer)
        assert R.reward_wic_json([c]) == [pytest.approx(R.WIC_JSON_PARSES + R.WIC_JSON_BOOL)]

    @pytest.mark.parametrize(
        "answer",
        [
            "The two uses are different.",                 # prose, no JSON at all
            '{"sense1": "a", "same_sense": tru',           # truncated / unparseable
            "same",                                        # bare verdict token
        ],
    )
    def test_non_json_answer_is_penalised(self, answer):
        c = wrap(GOOD_THINK, answer)
        assert R.reward_wic_json([c]) == [pytest.approx(R.WIC_JSON_MALFORMED)]

    @pytest.mark.parametrize(
        "completion",
        [
            "<think>the reasoning ran past the budget and never closed",  # unclosed
            wrap(GOOD_THINK, ""),                                         # nothing after </think>
        ],
    )
    def test_no_answer_region_is_neutral_not_penalised(self, completion):
        # reward_think_length already charges for these; double-charging here would
        # swamp the accuracy signal.
        assert R.reward_wic_json([completion]) == [0.0]


# --------------------------------------------------------------------------- #
# wic: verdict accuracy + format
# --------------------------------------------------------------------------- #
class TestWicAccuracy:
    def test_correct_and_wrong_verdicts(self):
        right = wrap(GOOD_THINK, json.dumps({"sense1": "a", "sense2": "b", "same_sense": True}))
        wrong = wrap(GOOD_THINK, json.dumps({"sense1": "a", "sense2": "b", "same_sense": False}))
        assert R.reward_wic_accuracy([right, wrong], label=["same", "same"]) == [
            R.WIC_CORRECT,
            R.WIC_WRONG,
        ]

    def test_absent_verdict_scores_zero(self):
        c = "<think>never closed and never answered"
        assert R.reward_wic_accuracy([c], label=["same"]) == [0.0]

    def test_prose_verdict_still_scores_but_forfeits_json_credit(self):
        # The accuracy reward is deliberately lenient (it scores the decision
        # wherever it was expressed); reward_wic_json is what prices the shape.
        c = wrap(GOOD_THINK, "These are two different senses.")
        assert R.reward_wic_accuracy([c], label=["different"]) == [R.WIC_CORRECT]
        assert R.reward_wic_json([c]) == [pytest.approx(R.WIC_JSON_MALFORMED)]

    def test_format_reward_pays_for_think_and_extractable_verdict(self):
        full = wrap(GOOD_THINK, JSON_ANSWER)
        no_think = JSON_ANSWER
        no_answer = wrap(GOOD_THINK, "")
        assert R.reward_wic_format([full, no_think, no_answer]) == [
            pytest.approx(0.2),
            pytest.approx(0.1),
            pytest.approx(0.1),
        ]


# --------------------------------------------------------------------------- #
# Reasoning quality
# --------------------------------------------------------------------------- #
class TestThinkLength:
    def test_stub_think_is_penalised(self):
        assert R.reward_think_length([wrap("ok", JSON_ANSWER)]) == [R.THINK_MIN_PENALTY]

    def test_missing_think_is_penalised(self):
        assert R.reward_think_length([JSON_ANSWER]) == [R.THINK_MIN_PENALTY]

    def test_unclosed_think_is_penalised(self):
        # No closing tag -> _extract_think returns "" -> treated as a stub, which is
        # what stops the model from farming the format bonus with runaway reasoning.
        assert R.reward_think_length(["<think>" + "reasoning " * 50]) == [R.THINK_MIN_PENALTY]

    def test_real_reasoning_is_not_penalised(self):
        assert R.reward_think_length([wrap(GOOD_THINK, JSON_ANSWER)]) == [0.0]


# --------------------------------------------------------------------------- #
# Gloss hygiene (direct / triplet)
# --------------------------------------------------------------------------- #
class TestGlossHygiene:
    @pytest.mark.parametrize("gloss", ["a bank of a river", "banks of a river", "banking on it"])
    def test_self_referential_gloss_is_penalised(self, gloss):
        # Inflections count: the point is the gloss must not lean on the word itself.
        c = wrap(GOOD_THINK, gloss)
        assert R.reward_no_target([c], lemma=["bank"]) == [R.SELF_REF_PENALTY]

    def test_gloss_avoiding_the_target_is_clean(self):
        c = wrap(GOOD_THINK, "sloping land beside a body of water")
        assert R.reward_no_target([c], lemma=["bank"]) == [0.0]

    def test_contentless_gloss_is_penalised(self):
        assert R.reward_min_content([wrap(GOOD_THINK, "a thing")]) == [R.MIN_CONTENT_PENALTY]

    def test_contentful_gloss_is_clean(self):
        c = wrap(GOOD_THINK, "sloping land beside a river")
        assert R.reward_min_content([c]) == [0.0]

    def test_length_penalty_is_monotone_in_excess(self):
        ref = "sloping land beside a river"
        short = wrap(GOOD_THINK, "sloping land beside a river")
        longer = wrap(GOOD_THINK, " ".join(["sloping land beside a river"] * 4))
        longest = wrap(GOOD_THINK, " ".join(["sloping land beside a river"] * 12))
        s, l, ll = R.reward_length([short, longer, longest], gloss=[ref] * 3)
        assert s == 0.0
        assert R.LENGTH_PENALTY <= ll < l < s

    def test_length_measures_the_whole_answer_region_not_just_line_one(self):
        # The hack this defends against: a short first line (all the gloss extractor
        # sees) followed by a long repeated tail.
        ref = "sloping land beside a river"
        tail = wrap(GOOD_THINK, "sloping land beside a river\n" + "definition: sloping land beside a river\n" * 10)
        assert R.reward_length([tail], gloss=[ref])[0] < 0.0


# --------------------------------------------------------------------------- #
# Fidelity (BERTScore stubbed: the reward's plumbing is what's under test)
# --------------------------------------------------------------------------- #
class TestFidelity:
    def test_reads_gold_gloss_from_either_task_schema(self, monkeypatch):
        seen = {}

        def fake(hyps, refs):
            seen["hyps"], seen["refs"] = hyps, refs
            return [1.0] * len(hyps)

        monkeypatch.setattr(R, "bertscore_similarity", fake)
        c = wrap(GOOD_THINK, "sloping land beside a river")

        assert R.reward_fidelity([c], gloss=["direct gold"]) == [1.0]
        assert seen["refs"] == ["direct gold"]  # direct records carry `gloss`

        assert R.reward_fidelity([c], gloss_same=["triplet gold"]) == [1.0]
        assert seen["refs"] == ["triplet gold"]  # triplet records carry `gloss_same`
        assert seen["hyps"] == ["sloping land beside a river"]

    def test_triplet_contrast_negates_similarity_to_the_negative_sense(self, monkeypatch):
        monkeypatch.setattr(R, "bertscore_similarity", lambda hyps, refs: [0.8] * len(hyps))
        c = wrap(GOOD_THINK, "sloping land beside a river")
        assert R.reward_triplet_contrast([c], gloss_diff=["a financial institution"]) == [
            pytest.approx(-0.8)
        ]


# --------------------------------------------------------------------------- #
# Supersense
# --------------------------------------------------------------------------- #
class TestSupersense:
    def test_exact_candidate_scores_hard_reward(self):
        right = wrap(GOOD_THINK, "animal")
        wrong = wrap(GOOD_THINK, "artifact")
        out = R.reward_supersense_accuracy(
            [right, wrong], supersense=["animal", "animal"], pos=["noun", "noun"]
        )
        assert out == [R.SUPERSENSE_CORRECT, R.SUPERSENSE_WRONG]

    def test_pos_scopes_the_candidate_space(self):
        # "change" is a verb supersense only; for a noun record it is off-vocabulary,
        # so it falls through to the soft tier rather than matching exactly.
        assert sd.extract_supersense(wrap(GOOD_THINK, "change"), "verb") == "change"
        assert sd.extract_supersense(wrap(GOOD_THINK, "change"), "noun") == ""

    def test_off_vocabulary_answer_gets_the_soft_tier(self, monkeypatch):
        monkeypatch.setattr(R, "_nearest_candidates", lambda regions, poss: ["change"])
        c = wrap(GOOD_THINK, "biological growth")
        assert R.reward_supersense_accuracy(
            [c], supersense=["change"], pos=["verb"]
        ) == [R.SUPERSENSE_SOFT_CORRECT]
        assert R.reward_supersense_accuracy(
            [c], supersense=["motion"], pos=["verb"]
        ) == [R.SUPERSENSE_SOFT_WRONG]

    def test_soft_credit_stays_below_the_distillation_threshold(self):
        # Guards the invariant in the reward's comment: only exact answers may leak
        # into the self-distillation set (default --distill-threshold 0.5).
        assert R.SUPERSENSE_SOFT_CORRECT < 0.5 < R.SUPERSENSE_CORRECT

    def test_blank_answer_stays_zero_and_never_calls_the_encoder(self, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("the encoder must not be loaded for an empty answer region")

        monkeypatch.setattr(R, "_nearest_candidates", boom)
        assert R.reward_supersense_accuracy(
            ["<think>unclosed"], supersense=["animal"], pos=["noun"]
        ) == [0.0]


# --------------------------------------------------------------------------- #
# Multitask masking
# --------------------------------------------------------------------------- #
class TestMaskByTask:
    def test_scores_only_its_own_rows_and_zeroes_the_rest(self):
        wic = wrap(GOOD_THINK, json.dumps({"sense1": "a", "sense2": "b", "same_sense": True}))
        other = wrap(GOOD_THINK, "some gloss from the direct task")
        masked = R.mask_by_task("wic", R.reward_wic_accuracy)

        out = masked(
            [other, wic, other],
            task=["direct", "wic", "direct"],
            label=["", "same", ""],  # padded columns on the non-wic rows
        )
        assert out == [0.0, R.WIC_CORRECT, 0.0]

    def test_a_batch_with_none_of_its_task_is_all_zero(self):
        masked = R.mask_by_task("wic", R.reward_wic_accuracy)
        assert masked([wrap(GOOD_THINK, "a gloss")], task=["direct"], label=[""]) == [0.0]

    def test_metric_names_stay_unique_per_task(self):
        # GRPOTrainer derives the logged metric name from __name__; collisions would
        # silently merge two tasks' curves.
        names = {
            R.mask_by_task(t, fn).__name__
            for t, fns in R.REWARDS.items()
            for fn in fns
        }
        assert len(names) == sum(len(fns) for fns in R.REWARDS.values())


# --------------------------------------------------------------------------- #
# Registry wiring
# --------------------------------------------------------------------------- #
def test_every_task_has_rewards_keep_cols_and_a_fidelity_fn():
    for task in ("direct", "triplet", "wic", "supersense"):
        assert R.REWARDS[task] and R.KEEP_COLS[task] and R.FIDELITY[task]


def test_wic_json_reward_is_registered():
    assert R.reward_wic_json in R.REWARDS["wic"]
