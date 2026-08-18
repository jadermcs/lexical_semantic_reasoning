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

The GRPO baseline lives in grpo_lora.py, unchanged; that is the comparison, and every
knob that is not the objective is pinned to that run's value so the comparison is about
the dense feedback and nothing else. Identical: the dataset and its split
(``utils.build_grpo_dataset``, same file, same ``test_size=200, seed=42``), the rendered
prompts, LoRA r=32/alpha=32/dropout 0.05, lr 3e-5 cosine with 200 warmup steps,
``max_grad_norm`` 1.0, 2 epochs, micro-batch x grad-accum, G=16 rollouts per prompt,
sampling at T=1.0/top-p 0.95/top-k 0/min-p 0, 512 completion tokens, ``beta=0``,
``paged_adamw_8bit``, and the full reward stack including DAPO's overlong shaping and the
repetition penalty -- kept not because they shape anything here (they cannot; see below)
but so the logged ``reward`` curve is the *same statistic* as the baseline's.

What differs, and only this: GRPO turns the reward into a group-relative advantage and
takes a clipped policy-gradient step; U-OPSD ignores the reward in the loss entirely
(``distillation_weight=1.0``) and takes a forward-KL step toward a teacher conditioned on
the group's own consensus trace. Dense per-token target vs one scalar per rollout.
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
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment

import sense_data as sd
from grpo_lora import load_policy
from sense_rewards import REWARDS
from utils import build_grpo_dataset

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


def add_privileged_context(rec):
    """Attach the label channel the vote is *scored* against.

    ``privileged_context`` keeps its name because that is the only per-example column
    ``SDPOTrainer._prepare_training_batch`` forwards to the context builder, but under
    U-OPSD it is emphatically not privileged context: ``VoteConsensusContextBuilder``
    never renders ``feedbacks`` into a teacher prompt, and ``include_environment_feedback``
    is hard-wired ``False`` below so no other code path can either. It carries the gold
    verdict solely so ``uopsd/vote_matches_gold`` can be logged — the paper's "86.7% of
    pseudo-labels match the gold answer" probe, and the number that says whether a run's
    consensus is worth anything. Two independent switches therefore have to be flipped
    before a label could reach the teacher; if you touch either, drop this column too.
    """
    return {"privileged_context": "same" if bool(rec["label"]) else "different"}


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
        # The GLOBAL distilled fraction over the whole generation batch, which is what
        # RenormalizedSDPOTrainer needs and what no per-micro-batch view can recover:
        # by the time the loss runs, it sees 2 rows of the 256 this mask covers.
        self.trainer._last_active_fraction = mask.mean().item()
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
    """Average the distillation loss over the step's distilled rollouts, per Eq. 5.

    ``_compute_self_distillation_loss`` sums the per-sequence losses and divides by the row
    count of the whole micro-batch, while masked-out rows contribute 0. That is harmless
    when nearly every row is distilled (supervised SDPO's regime) but not here: only the
    disagreeing rollouts of a decisive group are targets, ~2-3% of rows in practice.

    The correction has to span gradient accumulation, which is the part that is easy to get
    wrong. Normalising *within* the micro-batch is not enough, because HF divides every
    micro-batch loss by ``gradient_accumulation_steps`` (trainer.py:1942) including the
    ~97% that hold no distilled row at all. Writing R for the micro-batch rows, G for the
    accumulation steps and N for the step's distilled rows, the accumulated loss is
    ``s/(G*R) * sum(L_i)``, so recovering Eq. 5's ``mean(L_i)`` needs ``s = G*R/N`` -- i.e.
    the reciprocal of the GLOBAL distilled fraction, ~34 at G=128/R=2, not the ~2 a
    per-micro-batch count yields. (The same identity holds under DDP, which averages across
    ranks: G*R*W/N_global is still 1/fraction.) Getting this wrong is silent -- the run
    trains, ``grad_norm`` just sits ~17x below where it belongs and the curve goes flat.
    """

    def _compute_self_distillation_loss(self, model, inputs, distillation_logits):
        loss = super()._compute_self_distillation_loss(model, inputs, distillation_logits)
        # Set by VoteConsensusContextBuilder.build once per generation batch, so it stays
        # constant across the micro-batches that accumulate into one optimizer step --
        # which is exactly the scope Eq. 5 normalises over. 0.0 means nothing was
        # distilled anywhere (every eval batch, where a group of 1 cannot disagree).
        frac = getattr(self, "_last_active_fraction", 0.0)
        if not frac:
            return loss
        return loss / frac


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


def build_datasets(file_path):
    """The baseline's train/dev split verbatim, plus the vote-vs-gold diagnostic column.

    ``utils.build_grpo_dataset`` is the function grpo_lora.py calls, so pointing both
    scripts at the same ``--file-path`` gives them the same records, the same rendered
    prompts, the same ``train_test_split(test_size=200, seed=42)`` and therefore the same
    step count per epoch. The 200-pair dev split is fixed by that seed, so the wobble
    between evals is pure sampling noise and the two eval curves are one measurement.

    The eval *loss* is uninformative here (``--num-generations-eval 1`` leaves a group of
    one, which has no disagreement, so the distillation term is masked out everywhere);
    the reward metrics are the curve to read against GRPO's.
    """
    ds = build_grpo_dataset(file_path).map(add_privileged_context)
    return ds["train"], ds["test"]


def main():
    ap = argparse.ArgumentParser()
    # --- everything here is grpo_lora.py's value, so the runs stay comparable ----- #
    # Changing any default in this block breaks the controlled comparison unless the
    # same change is made in grpo_lora.py and the baseline is retrained.
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--file-path",
        default="data/xl-lexeme.json",
        help="Pair file, read through utils.build_grpo_dataset exactly as grpo_lora.py "
        "reads it. Must be the SAME file the baseline was trained on — the finished "
        "qwen-lora-grpo-wic run used data/mcl_train_dev.json (9000 pairs → 8800 train + "
        "200 dev → 1100 steps at 2 epochs), not this default.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Backward micro-batch, in completions. This is the knob that OOMs, and "
        "the ONLY thing it controls is peak activation memory — see --generation-batch "
        "for the one that sets how many prompts each gradient averages over, which is "
        "what has to match GRPO. Note the teacher forward roughly doubles a step's "
        "activation cost relative to GRPO, so this may have to drop below the "
        "baseline's 2 even though the gradient is unaffected.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Rollouts per prompt: GRPO's group to centre an advantage on, here the "
        "resolution of the vote and the pool the disagreeing rollouts are drawn from. "
        "16 is the baseline's. The paper defaults to 8 and its Figure 4 (middle) finds "
        "G=4 and G=8 indistinguishable with G=12 worth ~4.7%% more, so 16 is on the "
        "generous side of that curve — no reason to diverge from the baseline for it.",
    )
    ap.add_argument(
        "--generation-batch",
        type=int,
        default=256,
        help="Completions per optimizer step; grad-accum is derived from it as "
        "(this // --batch-size). Prompt groups per step = this // --num-generations, "
        "and that is what the gradient averages over.",
    )
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="grpo_lora.py's, held fixed so the step size is not a confound. The U-OPSD "
        "paper's own recipe is 5e-6 with r=64/alpha=128 — a forward KL against a teacher "
        "holding a solution is a much larger per-token signal than a clipped advantage, "
        "so if the run diverges (watch self_distillation/distillation_loss and "
        "completions/mean_length) this is the first thing to lower, at the cost of the "
        "comparison being confounded.",
    )
    ap.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="grpo_lora.py's (the HF default). The paper clips at 0.1 and notes KL "
        "spikes carry through above that; same trade as --learning-rate.",
    )
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.4)
    ap.add_argument(
        "--max-completion-length",
        type=int,
        default=512,
        help="The paper raises this to 4096 because a vote needs rollouts that reach a "
        "boxed answer; here the answer is a short JSON object, and 512 is the baseline's. "
        "Watch uopsd/valid_fraction: if it sags, this is the knob.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Rollout temperature, grpo_lora.py's. The paper trains at 1.1/top-k 20, but "
        "1.0/top-k 0 is strictly the more diverse setting, and diversity is what the vote "
        "needs: the only signal here is disagreement among the rollouts, so if "
        "uopsd/correctable_group_fraction collapses toward 0 this is the knob — at the "
        "cost of no longer sampling like the baseline.",
    )
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0, help="0 disables top-k.")
    ap.add_argument(
        "--soft-punish-cache",
        type=int,
        default=128,
        help="DAPO overlong shaping, as in grpo_lora.py. Diagnostic only here — no "
        "reward reaches the gradient — but kept so the logged reward is the same "
        "statistic as the baseline's. 0 masks truncated completions instead.",
    )
    ap.add_argument("--overlong-penalty", type=float, default=0.2)
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
    # --- schedule/eval: grpo_lora.py's settings ----------------------------- #
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--warmup-steps", type=float, default=200)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--run-name", default="qwen-lora-uopsd-wic")
    ap.add_argument("--logging-steps", type=int, default=25)
    ap.add_argument(
        "--save-strategy", default="steps", choices=["steps", "epoch", "no"]
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

    train_ds, dev_ds = build_datasets(args.file_path)
    print(train_ds[0])

    model, peft_config = load_policy(
        args.model,
        dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
    )

    # grpo_lora.py's, and it budgets the same thing: the *student's* prompt, which is
    # byte-identical to the baseline's. The teacher reprompt is wider (student prompt +
    # a whole pseudo-solution) but never goes through vLLM, so it is budgeted separately
    # by max_reprompt_len below.
    prompt_headroom = 512
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
    # rewards — do not reach the gradient. That is exactly why grpo_lora.py's full stack
    # is reproduced here rather than trimmed to the terms that "matter": the logged
    # reward has to be the same statistic as the baseline's to be read against it, and a
    # shaping term that cannot shape anything costs nothing but the forward pass.
    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_weights = [1.0 for f in REWARDS]
    reward_funcs.append(get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.2))
    reward_weights.append(1.0)
    if args.soft_punish_cache > 0:
        # Not wrapped in as_text_reward: this one scores completion_ids, not text.
        reward_funcs.append(
            get_soft_overlong_punishment(
                max_completion_len=args.max_completion_length,
                soft_punish_cache=args.soft_punish_cache,
            )
        )
        reward_weights.append(args.overlong_penalty)
    # An unparsable rollout is excluded from the vote and can never be a distillation
    # target, so truncation is already handled upstream of the loss; this follows the
    # baseline only because zeroing a completion mask would also corrupt the teacher
    # alignment, which is the stronger reason to leave it off.
    mask_truncated = args.soft_punish_cache == 0
    print(
        f"distillation: forward={args.divergence} (alpha={alpha}) "
        f"mode={distill_mode_kwargs['distillation_mode']} "
        f"teacher={args.teacher_kind}(rate={args.teacher_update_rate}) "
        f"tau={args.vote_threshold} G={args.num_generations} "
        f"ref={args.teacher_reference} targets={args.distill_targets or 'all'}"
        f"({args.distill_target_select}) renorm={args.renormalize_distillation}"
    )
    # The parity line: every field here must read the same in grpo_lora.py's launch log
    # for the comparison to be about the objective. Rewards are logged, never gradient.
    print(
        f"parity vs grpo_lora: data={args.file_path} lora=r{args.lora_r}/a{args.lora_alpha} "
        f"lr={args.learning_rate} clip={args.max_grad_norm} warmup={args.warmup_steps} "
        f"sampling=T{args.temperature}/top_p{args.top_p}/top_k{args.top_k} "
        f"len={args.max_completion_length} rewards={len(reward_funcs)}"
        + (
            f"(overlong=shape(cache={args.soft_punish_cache}, w={args.overlong_penalty}))"
            if not mask_truncated
            else "(overlong=mask)"
        )
    )

    run_name = args.run_name
    output_dir = args.output_dir or f"./{run_name}"
    training_args = SDPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=mask_truncated,
        reward_weights=reward_weights,
        beta=0.0,
        disable_dropout=True,
        optim="paged_adamw_8bit",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=0.0,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=2,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        max_grad_norm=args.max_grad_norm,
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=100,
        save_total_limit=6,
        logging_steps=args.logging_steps,
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
