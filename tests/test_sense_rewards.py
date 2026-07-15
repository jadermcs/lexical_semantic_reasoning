"""Unit tests for the GRPO WiC reward functions.

These pin the *contract* each reward enforces, which is what stops a silent
reward-hack from creeping back in: the failure modes named in the reward
docstrings (a stubbed <think>, an unclosed reasoning block, a prose answer instead
of JSON) each have a test here asserting they are actually punished.

Nothing here loads a model, so the suite runs on CPU in a second.
"""

import json

import pytest

import sense_rewards as R


def wrap(think, answer):
    """A completion in the format the task uses: <think>...</think>\\nanswer."""
    return f"<think>\n{think}\n</think>\n{answer}"


GOOD_THINK = "the first usage names a financial institution while the second names a river edge"
JSON_ANSWER = json.dumps({"sense1": "a financial institution", "sense2": "sloping land", "same_sense": False})


# --------------------------------------------------------------------------- #
# Answer shape
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
# Verdict accuracy + format
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
# Registry wiring
# --------------------------------------------------------------------------- #
def test_accuracy_dominates_the_shape_rewards():
    # Being right must always beat being merely well-formatted, or the policy can
    # farm the format terms while answering wrongly.
    shape_ceiling = 0.2 + R.WIC_JSON_PARSES + R.WIC_JSON_KEYS + R.WIC_JSON_BOOL
    assert shape_ceiling < R.WIC_CORRECT


def test_every_reward_is_registered():
    assert R.REWARDS == [
        R.reward_wic_accuracy,
        R.reward_wic_format,
        R.reward_wic_json,
        R.reward_think_length,
    ]
    assert R.KEEP_COLS == ["lemma", "label"]
