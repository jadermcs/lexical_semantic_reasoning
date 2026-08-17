"""U-OPSD's vote → mask decision, pinned without a GPU or a real trainer.

``VoteConsensusContextBuilder.build`` is where the whole method lives: it decides which
rollouts train (the disagreeing ones), what the teacher is shown (the longest agreeing
one), and which groups are skipped (indecisive vote, or nothing to correct). None of that
touches torch's autograd or a model, so it is testable against a stub trainer — and worth
pinning, because every one of those decisions is silent at runtime: get it wrong and the
run still trains, just on the wrong rollouts.
"""

import types

import pytest
import torch

import sdpo_lora as U

PAD = 0
G = 8


def answer(same, filler=""):
    """A rollout in the output contract, with `filler` to control its length."""
    verdict = "true" if same else "false"
    return (
        f"<think>because{filler}</think>"
        f'{{"sense1": "a", "sense2": "b", "same_sense": {verdict}}}'
    )


class _Accel:
    device = "cpu"
    process_index = 0
    num_processes = 1

    def gather(self, t):
        return t

    def pad_across_processes(self, t, dim=1, pad_index=0):
        return t


class _Trainer:
    """The slice of SDPOTrainer that build() reaches for.

    Completions are encoded as byte+1 so that 0 is free to act as the pad id, which lets
    the stub decode without a tokenizer while keeping build()'s `row != pad` length logic
    (the ranking key for longest/shortest) honest.
    """

    def __init__(self):
        self.accelerator = _Accel()
        self._tokenizer = types.SimpleNamespace(pad_token_id=PAD)
        self.processing_class = types.SimpleNamespace(decode=self._decode)
        self.model = types.SimpleNamespace(training=True)
        self.num_generations = G
        self.num_generations_eval = 1
        self.args = types.SimpleNamespace(
            solution_template=U.SOLUTION_TEMPLATE,
            reprompt_template=U.REPROMPT_TEMPLATE,
            max_reprompt_len=4096,
        )
        self.teacher_prompts = None

    @staticmethod
    def _decode(ids, skip_special_tokens=True):
        return bytes(int(i) - 1 for i in ids).decode()

    def _tokenize_prompts_untruncated(self, prompts):
        # Keep the teacher text instead of tokenizing it, so a test can assert on it.
        self.teacher_prompts = prompts
        return [[1] for _ in prompts]


def build(texts, **kw):
    trainer = _Trainer()
    builder = U.VoteConsensusContextBuilder(trainer, **kw)
    width = max(len(t.encode()) for t in texts) + 1
    ids = torch.tensor(
        [[b + 1 for b in t.encode()] + [PAD] * (width - len(t.encode())) for t in texts]
    )
    batch = {"completion_ids": ids, "completion_mask": (ids != PAD).long()}
    prompts = [[{"role": "user", "content": f"p{i // G}"}] for i in range(len(texts))]
    out = builder.build(
        batch,
        prompts,
        torch.zeros(len(texts)),
        feedbacks=["same"] * len(texts),  # gold label, metrics channel only
    )
    return builder, out, trainer


# 6 "same" / 2 "different": vote is same at confidence 6/8, one target, longest-first.
SPLIT_6_2 = [answer(True, "x" * i) for i in range(6)] + [
    answer(False, "yyyy"),
    answer(False),
]


def test_distills_the_longest_disagreeing_rollout_only():
    builder, out, _ = build(SPLIT_6_2)
    assert out["self_distillation_mask"].tolist() == [0, 0, 0, 0, 0, 0, 1, 0]
    assert builder.last_metrics["uopsd/valid_fraction"] == 1.0
    assert builder.last_metrics["uopsd/trusted_group_fraction"] == 1.0
    assert builder.last_metrics["uopsd/correctable_group_fraction"] == 1.0


def test_teacher_sees_the_longest_agreeing_rollout_and_only_targets_get_a_reprompt():
    _, _, trainer = build(SPLIT_6_2)
    reprompt = trainer.teacher_prompts[6][-1]["content"]
    assert "xxxxx" in reprompt  # longest agreeing rollout
    assert "yyyy" not in reprompt  # never its own trace, and never a disagreeing one
    # A masked-out row still gets forwarded, but with the untouched prompt.
    assert trainer.teacher_prompts[0] == [{"role": "user", "content": "p0"}]


def test_target_and_reference_selection_axes():
    _, out, _ = build(SPLIT_6_2, num_targets=0)
    assert out["self_distillation_mask"].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]
    _, out, trainer = build(SPLIT_6_2, target="shortest", reference="shortest")
    assert out["self_distillation_mask"].tolist() == [0, 0, 0, 0, 0, 0, 0, 1]
    assert "because<" in trainer.teacher_prompts[7][-1]["content"]


def test_unanimous_group_has_nothing_to_correct():
    builder, out, _ = build([answer(True, "x" * i) for i in range(G)])
    assert out["self_distillation_mask"].sum() == 0
    assert builder.last_metrics["uopsd/trusted_group_fraction"] == 1.0
    assert builder.last_metrics["uopsd/correctable_group_fraction"] == 0.0


@pytest.mark.parametrize("threshold,expected", [(0.5, 1), (0.6, 0)])
def test_confidence_is_normalised_by_all_rollouts_not_the_valid_ones(threshold, expected):
    """4 agree, 2 disagree, 2 truncate: 4/8 clears tau=0.5 but 4/6 would clear 0.6 too."""
    texts = [answer(True, "x" * i) for i in range(4)] + [answer(False)] * 2
    texts += ["<think>truncated mid thought", "<think>also truncated"]
    builder, out, _ = build(texts, threshold=threshold)
    assert out["self_distillation_mask"].sum() == expected
    assert builder.last_metrics["uopsd/valid_fraction"] == 0.75


def test_invalid_rollouts_are_never_distillation_targets():
    texts = [answer(True, "x" * i) for i in range(5)] + ["<think>no verdict here"] * 3
    _, out, _ = build(texts)
    assert out["self_distillation_mask"].sum() == 0  # 5/8 trusted, but nothing valid to fix


def test_vote_matches_gold_is_a_monitor_and_can_disagree_with_the_label():
    """The label never gates the mask: a wrong vote still trains, and is still logged."""
    builder, out, _ = build([answer(False, "x" * i) for i in range(6)] + [answer(True)] * 2)
    assert builder.last_metrics["uopsd/vote_matches_gold"] == 0.0  # feedbacks say "same"
    assert out["self_distillation_mask"].sum() == 1


def test_groups_are_scored_independently_within_a_batch():
    texts = [answer(True, "x" * i) for i in range(4)] + [answer(True)] * 4  # unanimous
    texts += [answer(True, "x" * i) for i in range(6)] + [answer(False)] * 2
    builder, out, _ = build(texts)
    mask = out["self_distillation_mask"].tolist()
    assert mask[:G] == [0] * G
    assert sum(mask[G:]) == 1
    assert builder.last_metrics["uopsd/correctable_group_fraction"] == 0.5
