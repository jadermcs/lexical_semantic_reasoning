"""U-OPSD: unsupervised on-policy self-distillation on the WiC task, LoRA.

Implements "On-Policy Self-Distillation without Any Supervision" (arXiv:2608.06296) on
top of TRL's experimental SDPO trainer. SDPO/OPSD trains the student on its own rollouts
while a teacher — the same weights — is conditioned on a *privileged context*: the gold
solution, environment feedback, or (as this script used to do) the gold verdict for the
pair. U-OPSD removes that context. Per prompt group it takes a majority vote over the
verdicts the rollouts themselves produced, conditions the teacher on an *agreeing*
rollout, and distills the teacher's next-token distribution into the rollouts that
disagreed. Nothing outside the model enters the objective.

Concretely, against the supervised SDPO this file used to run:

* the privileged context is a pseudo-solution mined from the group (``y+``), not
  ``gold_feedback``/``semcor_pairs.gloss_feedback`` — both hint paths are gone;
* the distillation targets are the rollouts that disagree with the vote, not the rollouts
  that scored below a reward threshold;
* groups where the vote is not decisive (confidence < ``--vote-threshold``) or where
  nothing disagrees contribute no gradient — that skip is the paper's implicit curriculum;
* the loss is pure distillation (``distillation_weight=1.0``, Eq. 5). There is no policy
  gradient, so no reward reaches the loss and ``--loss-type``/``--scale-rewards``/clipping
  are gone with it. The rewards in ``sense_rewards`` are still computed and logged, as the
  gold-label *diagnostics* that make the run readable against grpo_lora.py's curve;
* the divergence is forward KL, teacher → student (Table 5: reverse KL diverges outright,
  JSD lands at the untrained model), where supervised SDPO used reverse KL.

The GRPO baseline lives in grpo_lora.py, unchanged; that is the comparison.
"""

import argparse
import json
import os
import random
from collections import Counter
from functools import wraps
from pathlib import Path

import torch
from accelerate.utils import gather_object
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder

import semcor_pairs
import sense_data as sd
from grpo_lora import load_policy
from sense_rewards import GLOSS_COLS, KEEP_COLS, REWARDS

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# The teacher is conditioned exactly as OPSD conditions it on a gold solution, with the
# pseudo-solution substituted in (paper Eq. 4 → Eq. 5): the point is to elicit the
# next-token distribution of a model that *has* an answer in hand, so the demonstration is
# presented as an answer rather than as a guess. It is the group's own consensus trace, so
# no claim from outside the model is being made to it.
REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n"
    "Now answer the original question, reasoning in <think> tags and ending with the "
    "required JSON object.\n"
)
SOLUTION_TEMPLATE = (
    "\nHere is a correct answer to this pair from an earlier attempt:\n\n"
    "{successful_previous_attempt}\n\n"
)


def format_prompt(rec):
    """Render the prompt column, plus the label channel the vote is *scored* against.

    ``privileged_context`` keeps its name because that is the only per-example column
    ``SDPOTrainer._prepare_training_batch`` forwards to the context builder, but under
    U-OPSD it is emphatically not privileged context: ``VoteConsensusContextBuilder``
    never renders ``feedbacks`` into a teacher prompt, and ``include_environment_feedback``
    is hard-wired ``False`` below so no other code path can either. It carries the gold
    verdict solely so ``uopsd/vote_matches_gold`` can be logged — the paper's "86.7% of
    pseudo-labels match the gold answer" probe, and the number that says whether a run's
    consensus is worth anything. Two independent switches therefore have to be flipped
    before a label could reach the teacher; if you touch either, drop this column too.

    The gold-gloss columns ``reward_wic_gloss`` scores ride along, likewise for logging
    only: with ``distillation_weight=1.0`` no reward enters the gradient.
    """
    out = {
        "prompt": sd.wic_messages(rec, with_target=False),
        "privileged_context": "same" if bool(rec["label"]) else "different",
    }
    for col, default in GLOSS_COLS.items():
        out[col] = rec.get(col) or default
    return out


class VoteConsensusContextBuilder(SuccessfulRolloutTeacherContextBuilder):
    """The privileged context is the group's own majority vote (paper Algorithm 1).

    Per prompt group of G rollouts:

    1. parse a verdict from each rollout (``sd.extract_wic_label`` is the paper's
       ``Ans(·)``; ``None`` is its invalid/unparsable class),
    2. take the plurality verdict as the pseudo-answer, ties broken uniformly, and score
       vote confidence as agreeing / G — normalised by *all* G rollouts, so truncations
       lower confidence instead of being quietly dropped from the denominator,
    3. skip the group when confidence < ``threshold`` (vote not trusted) or when nothing
       disagrees (nothing to correct),
    4. otherwise condition the teacher on the longest agreeing rollout and distill only
       the disagreeing ones.

    Invalid rollouts are neither reference nor target — Algorithm 1 line 4 excludes the
    empty answer from ``Y-`` — so a truncated completion never becomes a target, which is
    also why this script does not need ``mask_truncated_completions``.

    Only ``build``'s mask and teacher text matter to the objective, and neither reads a
    label; ``feedbacks`` is consumed for metrics only (see ``format_prompt``).
    """

    def __init__(
        self, trainer, threshold=0.5, num_targets=1, reference="longest", target="longest"
    ):
        super().__init__(trainer)
        self.threshold = threshold
        self.num_targets = num_targets  # 0 = every disagreeing rollout
        self.reference = reference
        self.target = target
        self._rng = random.Random(0)

    def _rank(self, idxs, lengths, how):
        """Order rollouts along Figure 5's two axes: completion length."""
        if how == "longest":
            return sorted(idxs, key=lambda j: -lengths[j])
        if how == "shortest":
            return sorted(idxs, key=lambda j: lengths[j])
        order = list(idxs)
        self._rng.shuffle(order)
        return order

    def build(self, output, prompts, rewards, feedbacks=None):
        device = self.trainer.accelerator.device
        mode = "train" if self.trainer.model.training else "eval"
        num_generations = (
            self.trainer.num_generations
            if mode == "train"
            else self.trainer.num_generations_eval
        )
        completion_ids = output["completion_ids"]
        completion_mask = output["completion_mask"]
        pad = self.trainer._tokenizer.pad_token_id

        num_local = len(prompts)
        process_start = self.trainer.accelerator.process_index * num_local

        # Same gather as the parent builder, for the same reason: the vote has to see the
        # whole group, and a group can straddle ranks. Completions are padded to each
        # rank's local max, so equalise widths before gathering.
        padded = self.trainer.accelerator.pad_across_processes(
            completion_ids, dim=1, pad_index=pad
        )
        all_completion_ids = self.trainer.accelerator.gather(padded)
        all_prompts = gather_object(prompts)
        total = all_completion_ids.shape[0]
        all_labels = gather_object(feedbacks) if feedbacks is not None else [None] * total

        texts, lengths, answers = [], [], []
        for row in all_completion_ids:
            ids = row[row != pad]
            text = self.trainer.processing_class.decode(ids, skip_special_tokens=True)
            texts.append(text)
            lengths.append(int(ids.numel()))
            answers.append(sd.extract_wic_label(text))

        mask = torch.zeros(total, device=device)
        reference_of = [None] * total
        pseudo_of = [None] * total
        n_groups = n_trusted = n_correctable = n_vote_correct = 0
        for start in range(0, total, num_generations):
            group = range(start, start + num_generations)
            n_groups += 1
            valid = [j for j in group if answers[j] is not None]
            if not valid:
                continue
            counts = Counter(answers[j] for j in valid)
            top = max(counts.values())
            # Sorted so the tie-break is over a deterministic candidate order; the draw
            # itself is the paper's uniform one.
            pseudo = self._rng.choice(sorted(a for a, c in counts.items() if c == top))
            for j in group:
                pseudo_of[j] = pseudo
            if top / num_generations < self.threshold:
                continue
            n_trusted += 1
            agree = [j for j in valid if answers[j] == pseudo]
            disagree = [j for j in valid if answers[j] != pseudo]
            if not disagree:
                continue
            n_correctable += 1
            if all_labels[start] is not None:
                n_vote_correct += pseudo == (all_labels[start] == "same")
            ref = self._rank(agree, lengths, self.reference)[0]
            targets = self._rank(disagree, lengths, self.target)
            if self.num_targets:
                targets = targets[: self.num_targets]
            for j in targets:
                mask[j] = 1.0
                reference_of[j] = ref

        local_messages = []
        for global_idx in range(process_start, process_start + num_local):
            original_prompt = all_prompts[global_idx]
            ref = reference_of[global_idx]
            if ref is None:
                # Not a distillation target. The teacher is still forwarded on this row
                # (the batch is rectangular) but its loss is masked out, so the unmodified
                # prompt is the cheapest context to hand it.
                local_messages.append(original_prompt)
                continue
            solution = self.trainer.args.solution_template.format(
                successful_previous_attempt=texts[ref]
            )
            if isinstance(original_prompt, list):
                reprompt = self._build_reprompt_text(
                    original_prompt[-1]["content"], solution, ""
                )
                local_messages.append(
                    original_prompt[:-1] + [{"role": "user", "content": reprompt}]
                )
            else:
                local_messages.append(
                    self._build_reprompt_text(original_prompt, solution, "")
                )

        teacher_batch = self._tokenize_teacher_messages(local_messages)
        teacher_input_ids = torch.cat([teacher_batch["prompt_ids"], completion_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [teacher_batch["prompt_mask"], completion_mask], dim=1
        )

        local = slice(process_start, process_start + num_local)
        # Stashed for RolloutDumpCallback: the trainer fires
        # on_self_distillation_batch_prepared a few lines after this returns and passes
        # neither the rewards nor the vote, and this is the last place they are in scope.
        self.trainer._last_rollout_rewards = rewards.detach().clone()
        self.trainer._last_vote = {
            "answer": answers[local],
            "pseudo_answer": pseudo_of[local],
        }
        self.last_metrics = {
            "uopsd/valid_fraction": sum(a is not None for a in answers) / max(1, total),
            "uopsd/trusted_group_fraction": n_trusted / max(1, n_groups),
            # The fraction of groups that actually train: decisive vote AND something to
            # correct. If this collapses, raise --temperature or lower --vote-threshold.
            "uopsd/correctable_group_fraction": n_correctable / max(1, n_groups),
            "uopsd/vote_matches_gold": n_vote_correct / max(1, n_correctable),
            "self_distillation/reprompt_sample_fraction": mask.mean().item(),
        }
        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "self_distillation_mask": mask[local],
        }


class RenormalizedSDPOTrainer(SDPOTrainer):
    """Average the distillation loss over the distilled rollouts, not the micro-batch.

    ``_compute_self_distillation_loss`` sums the per-sequence losses and divides by the row
    count of the whole micro-batch, while masked-out rows contribute 0. That is harmless
    when nearly every row is distilled (supervised SDPO's regime) but not here: at the
    paper's default of one target per group of 8, it scales the gradient by ~1/8, and the
    factor moves with the vote statistics from step to step, so the effective learning rate
    drifts as the policy becomes more self-consistent. Eq. 5 normalises by |B_x^-|; this
    rescales back to that.
    """

    def _compute_self_distillation_loss(self, model, inputs, distillation_logits):
        loss = super()._compute_self_distillation_loss(model, inputs, distillation_logits)
        rows = distillation_logits.loss_mask.sum(-1)
        active = int((rows > 0).sum().item())
        if active == 0:
            return loss
        return loss * (rows.numel() / active)


class RolloutDumpCallback(TrainerCallback):
    """Write rollout text to disk, since SDPO has nowhere to log it.

    ``SDPOConfig`` subclasses ``_BaseConfig``, not ``GRPOConfig``, so it has no
    ``log_completions``/``num_completions_to_print`` — the knobs grpo_lora.py sets.
    ``SDPOTrainer`` instead exposes its own hooks via
    ``_dispatch_self_distillation_callback``, which calls any same-named method on a
    registered callback; ``on_self_distillation_batch_prepared`` is the one that fires
    once the rollouts, the teacher reprompts and the distillation mask all exist.

    Each record pairs the student's completion with the teacher reprompt built for it and
    with the vote that decided its fate, so a run reads as "what did the policy say, what
    did its group agree on, and was this rollout corrected".
    """

    def __init__(self, trainer, path, every=25, per_step=8):
        self.trainer = trainer
        self.path = Path(path)
        self.every = every
        self.per_step = per_step
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def on_self_distillation_batch_prepared(
        self,
        args=None,
        state=None,
        model=None,
        processing_class=None,
        prompt_ids=None,
        completion_ids=None,
        teacher_input_ids=None,
        self_distillation_mask=None,
        **kwargs,
    ):
        # Eval fires this too, with num_generations_eval=1 and no gradient; only the
        # training rollouts are worth the disk.
        if not model.training or state.global_step % self.every:
            return
        if not state.is_world_process_zero:
            # Every tensor here is this rank's slice, so on multi-GPU each rank would
            # dump a different, equally valid shard. Keep one file.
            return

        pad = processing_class.pad_token_id
        rewards = getattr(self.trainer, "_last_rollout_rewards", None)
        vote = getattr(self.trainer, "_last_vote", None)
        mask = self_distillation_mask

        def text(ids):
            return processing_class.decode(ids[ids != pad], skip_special_tokens=True)

        with self.path.open("a") as fh:
            for i in range(min(self.per_step, completion_ids.size(0))):
                rec = {
                    "step": state.global_step,
                    # Diagnostic only under U-OPSD: the reward is not in the loss.
                    "reward": None if rewards is None else round(rewards[i].item(), 4),
                    "distilled": None if mask is None else bool(mask[i].item()),
                    "answer": None if vote is None else vote["answer"][i],
                    "pseudo_answer": None if vote is None else vote["pseudo_answer"][i],
                    "prompt": text(prompt_ids[i]),
                    "completion": text(completion_ids[i]),
                    "teacher_reprompt": text(teacher_input_ids[i]),
                }
                fh.write(json.dumps(rec) + "\n")


def as_text_reward(fn):
    @wraps(fn)
    def wrapper(completions, **kwargs):
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        return fn(texts, **kwargs)

    return wrapper


def _prepare(recs, cap=None):
    """Records → a Dataset carrying exactly KEEP_COLS + prompt + privileged_context.

    Mapping each source to this fixed shape *before* concatenation is what lets the
    two be merged: the raw records differ (SemCor adds ``synset1``/``synset2``), the
    mapped ones do not.
    """
    ds = Dataset.from_list(recs)
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(format_prompt, remove_columns=drop)


def build_dataset(split, cap=None, semcor_path=None):
    """Rollout set for one split, optionally mixing in the saved SemCor pairs.

    ``semcor_path`` points at the file ``semcor_pairs.py`` writes, exactly as in
    grpo_lora.py — the pair set is materialised once (``run_train.sh`` does it) rather
    than resampled per run, which is what makes a run reproducible and keeps NLTK off the
    training host. Under U-OPSD those pairs contribute prompts and gloss *diagnostics*
    only: their gold senses can no longer condition the teacher, since a hint from the
    annotation is the supervision the method exists without.
    """
    parts = [_prepare(sd.load_mclwic(split), cap=cap)]
    if semcor_path:
        sc = semcor_pairs.load_pairs(semcor_path)
        print(f"[{split}] semcor: +{len(sc)} gloss-annotated pairs")
        parts.append(_prepare(sc, cap=cap))
    if len(parts) == 1:
        return parts[0]
    # Interleave the sources: each prompt forms its own rollout group, but leaving
    # them in blocks would make every optimizer step see one source only.
    return concatenate_datasets(parts).shuffle(seed=42)


def main():
    ap = argparse.ArgumentParser()
    # --- rollout/batching knobs: grpo_lora.py's, so the runs stay comparable ------ #
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Backward micro-batch, in completions. This is the knob that OOMs, and "
        "the ONLY thing it controls is peak activation memory — see --generation-batch "
        "for the one that sets how many prompts each gradient averages over. Note the "
        "teacher forward doubles the activation cost of a step relative to GRPO.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="Rollouts per prompt, i.e. the paper's G — here the resolution of the vote "
        "and the pool the disagreeing rollouts are drawn from, not a group to centre an "
        "advantage on. 8 is the paper's default; its Figure 4 (middle) finds G=4 and G=8 "
        "indistinguishable and G=12 worth ~4.7%% more, so raise it if generation is not "
        "the bottleneck. Below ~4 the vote stops being a signal at all.",
    )
    ap.add_argument(
        "--generation-batch",
        type=int,
        default=256,
        help="Completions per optimizer step; grad-accum is derived from it as "
        "(this // --batch-size). Prompt groups per step = this // --num-generations, "
        "and that is what the gradient averages over.",
    )
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=128)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="5e-6 with LoRA r=64/alpha=128 is the paper's recipe, and both differ from "
        "grpo_lora.py's 3e-5/r=32: a KL against a teacher that has seen a solution is a "
        "much larger per-token signal than a clipped advantage, so the GRPO step size is "
        "not transferable.",
    )
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.4)
    ap.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="The paper raises this to 4096 because a vote needs rollouts that reach a "
        "boxed answer; here the answer is a short JSON object, so 512 already leaves "
        "--vote-threshold measuring reasoning rather than truncation. Watch "
        "uopsd/valid_fraction: if it sags, this is the knob.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.1,
        help="Rollout temperature; 1.1 (with top-p 0.95 / top-k 20) is the paper's "
        "training value and this is NOT the same knob it was under supervised SDPO. There "
        "the gold hint guaranteed a teacher signal on every group; here the only signal is "
        "disagreement among the rollouts, so at the repo's eval-convention 0.6 the groups "
        "agree unanimously and uopsd/correctable_group_fraction collapses toward 0. "
        "Evaluation still runs at 0.6.",
    )
    ap.add_argument(
        "--semcor-pairs",
        nargs="?",
        const=str(semcor_pairs.DEFAULT_PAIRS),
        default=None,
        help="Mix the saved SemCor pair set into the training rollouts (bare flag uses "
        f"{semcor_pairs.DEFAULT_PAIRS}). Build the file with 'uv run python "
        "src/semcor_pairs.py', whose CLI carries the sampling knobs (--max-per-lemma, "
        "--min-confusability, --keep-test-lemmas). The dev set stays pure MCL-WiC so the "
        "eval curve remains comparable across runs.",
    )
    ap.add_argument(
        "--gloss-reward-weight",
        type=float,
        default=1.0,
        help="Weight on reward_wic_gloss in the logged reward. Affects logging only: with "
        "distillation_weight=1.0 no reward term enters the gradient.",
    )
    # --- U-OPSD: the vote that replaces the gold label ---------------------- #
    ap.add_argument(
        "--vote-threshold",
        type=float,
        default=0.5,
        help="Self-consistency threshold tau: the fraction of ALL --num-generations "
        "rollouts that must agree before the group's plurality verdict is trusted as a "
        "pseudo-label. Below it the prompt is treated as unlabeled and contributes no "
        "gradient. 0.5 (absolute majority) is the paper's default, but its Figure 4 "
        "(left) sweep is monotone in the *loose* direction — 0.3 beat 0.5 by 1.5 points "
        "and 0.9 by 14 — so 0.3 is worth trying, and a high value mostly buys silence.",
    )
    ap.add_argument(
        "--distill-targets",
        type=int,
        default=1,
        help="How many disagreeing rollouts to distill per group (the paper's |B_x^-|); "
        "0 means all of them. The paper's reported setting is 1, picked longest-first.",
    )
    ap.add_argument(
        "--teacher-reference",
        default="longest",
        choices=["longest", "random", "shortest"],
        help="Which agreeing rollout becomes the pseudo-solution the teacher is "
        "conditioned on. Figure 5(a) favours the longest, and shows the whole method "
        "resting on this: conditioning on the extracted answer alone instead of a full "
        "trace costs 10-16 points and lands below the untrained model.",
    )
    ap.add_argument(
        "--distill-target-select",
        default="longest",
        choices=["longest", "random", "shortest"],
        help="Which disagreeing rollouts to correct first when --distill-targets caps "
        "them. Figure 5(b): longest > random > shortest, a ~2-point axis.",
    )
    ap.add_argument(
        "--distillation-topk",
        type=int,
        default=100,
        help="Support size for the divergence, over the STUDENT's top-k with a tail "
        "bucket for the rest of the mass. 0 uses the full vocabulary. Table 4 ranks "
        "full-vocab and top-100 as a tie (59.0 vs 57.1 avg, each best on some benchmark) "
        "and both far above token-level (43.5), so top-k here is a cost optimisation, not "
        "a concession; note this is the opposite ranking to supervised SDPO's default "
        "sampled-token mode.",
    )
    ap.add_argument(
        "--divergence",
        default="forward_kl",
        choices=["forward_kl", "jsd", "reverse_kl"],
        help="Table 5, and not a free choice: forward KL(teacher || student) is the only "
        "one that trains. JSD lands within noise of the untrained model, and reverse KL "
        "diverges by losing termination — completions grew from 2.7k to 99k characters "
        "and parsable answers fell 99%% -> 33%%. 'reverse_kl' is exposed only because it "
        "was supervised SDPO's setting here.",
    )
    ap.add_argument(
        "--teacher-kind",
        default="base",
        choices=["base", "ema", "live"],
        help="'base' is the paper's default: the teacher frozen at the initial policy, "
        "which under LoRA costs nothing — it is the student with the adapter disabled. "
        "Figure 4 (right) shows EMA doing better (decay 0.995: +2.4 at the best "
        "checkpoint, +4.1 at step 150), reachable as '--teacher-kind ema "
        "--teacher-update-rate 0.005'. Never 'live': an unregularized teacher is the "
        "configuration that diverges.",
    )
    ap.add_argument(
        "--teacher-update-rate",
        type=float,
        default=0.005,
        help="EMA teacher rate, i.e. 1 - decay: 0.005 is Figure 4's best (decay 0.995). "
        "Ignored unless --teacher-kind ema.",
    )
    ap.add_argument(
        "--renormalize-distillation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalise the loss by the distilled rollouts rather than by the whole "
        "micro-batch; see RenormalizedSDPOTrainer. Off reproduces TRL's arithmetic, which "
        "at --distill-targets 1 shrinks the gradient ~8x.",
    )
    ap.add_argument(
        "--dump-rollouts",
        type=int,
        default=25,
        metavar="EVERY_N_STEPS",
        help="Append rollout text to <output-dir>/rollouts.jsonl every N steps (0 "
        "disables). SDPOConfig has no log_completions — it subclasses _BaseConfig, not "
        "GRPOConfig — so this is the only way to see what the policy actually wrote, and "
        "under U-OPSD it is also the only way to see what the vote agreed on.",
    )
    ap.add_argument("--resume", nargs="?", const=True, default=None)
    # --- eval: grpo_lora.py's settings, exposed because SDPO evals cost more --- #
    ap.add_argument(
        "--eval-prompts",
        type=int,
        default=200,
        help="Dev prompts per eval; 0 uses the whole 1000-pair dev split. 200 matches "
        "grpo_lora.py's hard-coded cap, so the two runs' eval curves are the same "
        "measurement. The subset is FIXED (shuffle(seed=42).select), so the wobble "
        "between evals is pure sampling noise, sd ~= sqrt(p(1-p)/(prompts * "
        "generations)) — at 200x1 that floor is ~0.037 in reward units. Spending a fixed "
        "generation budget on more prompts beats spending it on more generations: both "
        "cut variance by the same 1/(n*k), but more prompts also kills the bias of "
        "scoring a 200-pair slice of dev. The eval loss is uninformative here (a group of "
        "1 has no disagreement, so the distillation term is masked out everywhere); the "
        "reward metrics are the curve to read.",
    )
    ap.add_argument(
        "--num-generations-eval",
        type=int,
        default=1,
        help="Rollouts per dev prompt, as in grpo_lora.py. Must divide --batch-size, "
        "which trl uses as the eval batch (grpo_config.py:1082).",
    )
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Optimizer steps between evals; 50 matches grpo_lora.py.",
    )
    args = ap.parse_args()

    # Fail loudly at launch rather than silently training on 2 prompt groups.
    if args.generation_batch % args.batch_size:
        ap.error(
            f"--generation-batch ({args.generation_batch}) must be divisible by --batch-size ({args.batch_size})"
        )
    if args.generation_batch % args.num_generations:
        # SDPOConfig enforces this too (grpo_config.py:1090); catching it here reports
        # the prompt-group count that actually motivates the constraint.
        ap.error(
            f"--generation-batch ({args.generation_batch}) must be divisible by "
            f"--num-generations ({args.num_generations})"
        )
    if args.batch_size % args.num_generations_eval:
        # trl raises this only after the datasets are built (grpo_config.py:1082); an
        # eval config error should not cost a dataset build first.
        ap.error(
            f"--batch-size ({args.batch_size}) is the eval batch and must be divisible "
            f"by --num-generations-eval ({args.num_generations_eval})"
        )
    if args.batch_size % args.num_generations:
        # Not required by trl, but a group split across micro-batches is split across
        # optimizer sub-steps, so the renormalization above sees a fraction of the group.
        print(
            f"warning: --batch-size ({args.batch_size}) is not a multiple of "
            f"--num-generations ({args.num_generations}); rollout groups will straddle "
            "micro-batches"
        )
    grad_accum = args.generation_batch // args.batch_size
    prompt_groups = args.generation_batch // args.num_generations
    print(
        f"batching: micro={args.batch_size} x grad_accum={grad_accum} = "
        f"{args.generation_batch} completions/step / {args.num_generations} rollouts "
        f"= {prompt_groups} prompt groups averaged per optimizer step"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset("train", semcor_path=args.semcor_pairs)
    dev_ds = build_dataset("dev", cap=args.eval_prompts or None)
    print(train_ds[0])

    model, peft_config = load_policy(
        args.model,
        dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
    )

    # Wider than grpo_lora.py's 512: the teacher reprompt is the student prompt plus the
    # pseudo-solution, and max_reprompt_len below budgets from the same number.
    prompt_headroom = 768
    if args.vllm_server_host:
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=args.vllm_server_host,
            vllm_server_port=args.vllm_server_port,
        )
    else:
        vllm_kwargs = dict(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=args.vllm_gpu_mem,
            vllm_max_model_length=prompt_headroom + args.max_completion_length,
        )

    if args.distillation_topk:
        # The tail bucket is the "term capturing the tail probability" of the paper's
        # top-K approximation; without it the divergence is over the renormalised top-K
        # only, which is a different objective, not an approximation of the full one.
        distill_mode_kwargs = dict(
            distillation_mode="topk_logits",
            distillation_topk=args.distillation_topk,
            distillation_add_tail=True,
        )
    else:
        distill_mode_kwargs = dict(distillation_mode="full_logits")

    # trl's alpha is the generalized-JSD beta: 0 = forward KL(teacher || student), 1 =
    # reverse KL, in between = JSD. See loss_utils.compute_divergence.
    alpha = {"forward_kl": 0.0, "jsd": 0.5, "reverse_kl": 1.0}[args.divergence]

    # Rewards are kept purely as gold-label diagnostics: distillation_weight=1.0 means
    # compute_loss never calls _compute_policy_loss, so advantages — and therefore these
    # rewards — do not reach the gradient. DAPO's overlong shaping and the repetition
    # penalty are dropped with the policy term they used to shape.
    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_weights = [
        args.gloss_reward_weight if f.__name__ == "reward_wic_gloss" else 1.0
        for f in REWARDS
    ]
    print(
        f"distillation: forward={args.divergence} (alpha={alpha}) "
        f"mode={distill_mode_kwargs['distillation_mode']} "
        f"teacher={args.teacher_kind}(rate={args.teacher_update_rate}) "
        f"tau={args.vote_threshold} G={args.num_generations} "
        f"ref={args.teacher_reference} targets={args.distill_targets or 'all'}"
        f"({args.distill_target_select}) renorm={args.renormalize_distillation}"
    )

    run_name = "qwen-lora-uopsd-wic"
    output_dir = f"./{run_name}"
    training_args = SDPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        # An unparsable rollout is excluded from the vote and can never be a distillation
        # target, so truncation is already handled upstream of the loss; zeroing the
        # completion mask on top of that would only corrupt the teacher alignment.
        mask_truncated_completions=False,
        reward_weights=reward_weights,
        beta=0.0,
        disable_dropout=True,
        optim="paged_adamw_8bit",
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=2,
        warmup_steps=0.03,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        max_grad_norm=0.1,  # the paper's clip; loosen it and the KL spikes carry through
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        logging_steps=25,
        report_to="wandb",
        run_name=run_name,
        # --- the objective: Eq. 5, pure distillation ------------------------- #
        distillation_weight=1.0,
        distillation_alpha=alpha,
        **distill_mode_kwargs,
        teacher_model_kind=args.teacher_kind,
        teacher_update_rate=args.teacher_update_rate,
        # The reprompt is assembled by VoteConsensusContextBuilder, which ignores every
        # feedback path: no environment feedback, and the demonstration is chosen by vote
        # agreement rather than by a reward threshold.
        include_environment_feedback=False,
        use_successful_as_teacher=False,
        reprompt_template=REPROMPT_TEMPLATE,
        solution_template=SOLUTION_TEMPLATE,
        max_reprompt_len=prompt_headroom + 2 * args.max_completion_length,
        **vllm_kwargs,
    )

    trainer_cls = RenormalizedSDPOTrainer if args.renormalize_distillation else SDPOTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        peft_config=peft_config,
    )

    trainer.teacher_context_builder = VoteConsensusContextBuilder(
        trainer,
        threshold=args.vote_threshold,
        num_targets=args.distill_targets,
        reference=args.teacher_reference,
        target=args.distill_target_select,
    )

    if args.dump_rollouts:
        dump_path = Path(output_dir) / "rollouts.jsonl"
        trainer.add_callback(
            RolloutDumpCallback(trainer, dump_path, every=args.dump_rollouts)
        )
        print(f"dumping rollout text every {args.dump_rollouts} steps → {dump_path}")

    if args.resume:
        ckpt = (
            args.resume
            if isinstance(args.resume, str)
            else get_last_checkpoint(output_dir)
        )
        if ckpt is None:
            ap.error(f"--resume given but no checkpoint-* found in {output_dir}")
        state = json.loads((Path(ckpt) / "trainer_state.json").read_text())
        print(
            f"Resuming from {ckpt} at step {state['global_step']}/{state.get('max_steps')}"
        )
    else:
        ckpt = None

    trainer.train(resume_from_checkpoint=ckpt)
    trainer.save_model(output_dir)
    print(f"Saved final adapter → {output_dir}")


if __name__ == "__main__":
    main()
