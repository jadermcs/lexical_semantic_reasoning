import argparse
import json
import os
from functools import partial, wraps
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer
from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder
from trl.rewards import get_repetition_penalty_reward, get_soft_overlong_punishment

import semcor_pairs
import sense_data as sd
from grpo_lora import load_policy
from sense_rewards import GLOSS_COLS, KEEP_COLS, REWARDS

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


REPROMPT_TEMPLATE = (
    "{prompt}{solution}{feedback}\n"
    "Now answer the original question, reasoning in <think> tags and ending with the "
    "required JSON object.\n"
)
SOLUTION_TEMPLATE = (
    "\nHere is a correct answer to this pair from an earlier attempt:\n\n"
    "{successful_previous_attempt}\n\n"
)
FEEDBACK_TEMPLATE = "\n{feedback_raw}\n"


def gold_feedback(rec):
    """The privileged context: the gold verdict, stated as a hint to the teacher.

    Only ever conditions the reprompted teacher — never the student's own prompt.
    Without it, a group where every rollout is wrong has no successful sibling to
    distill from and SDPO degenerates to GRPO on that group; with it, the hardest
    pairs are exactly the ones that still produce a learning signal.
    """
    same = bool(rec["label"])
    verdict = "the same sense" if same else "different senses"
    return (
        f'Hint: the two sentences use "{rec["lemma"]}" in {verdict}, so the correct '
        f'verdict is "same_sense": {"true" if same else "false"}. Do not mention this '
        "hint; reason to it from the sentences themselves."
    )


def format_prompt(rec, with_feedback=True):
    """Render the prompt column, plus the hint the teacher is reprompted with.

    ``privileged_context`` is always written, empty when there is no hint: Arrow needs
    one schema across the concatenated MCL and SemCor sets, and the trainer's
    ``has_feedback`` test (``isinstance(str) and strip() != ""``) reads ``""`` as "no
    feedback" exactly as it reads a missing key. The gold-gloss columns
    ``reward_wic_gloss`` scores ride along on the same rule.
    """
    out = {"prompt": sd.wic_messages(rec, with_target=False), "privileged_context": ""}
    for col, default in GLOSS_COLS.items():
        out[col] = rec.get(col) or default
    if rec.get("synset1") and rec.get("synset2"):
        # A SemCor record carries gold senses, so its hint can name them. This is the
        # feedback --semcor-pairs exists to supply, and --no-gold-feedback does not
        # turn it off; see that flag's help.
        out["privileged_context"] = semcor_pairs.gloss_feedback(rec)
    elif with_feedback:
        # MCL-WiC has only the label, so the best available hint is the one bit.
        out["privileged_context"] = gold_feedback(rec)
    return out


class FailedRolloutContextBuilder(SuccessfulRolloutTeacherContextBuilder):
    """Restrict self-distillation to rollouts that actually failed.

    ``dont_reprompt_on_self_success`` reads as if it already did this ("Skip
    reprompting when model generates correct response"), but its implementation
    (``sdpo_trainer.py``, the ``if dont_reprompt_self and j == i`` line) only bars a
    rollout from being its *own* demonstration: a rollout that scored above
    ``success_reward_threshold`` still gets a *sibling's* demonstration and is still
    distilled. At ~0.65 train accuracy over 8 rollouts almost every group has a
    success, which is why ``success_sample_fraction`` logged 0.95 and
    ``reprompt_sample_fraction`` logged 1.0 — every token in the batch was pulled
    toward the frozen SFT teacher, including the correct rollouts GRPO was busy
    reinforcing. The two objectives then fight, and the reward term loses (see
    --distillation-weight).

    Zeroing the mask on successful rollouts restores the intended semantics: the
    policy gradient owns the rollouts that worked, distillation only densifies the
    ones that failed.
    """

    def __init__(self, trainer, failures_only=True):
        super().__init__(trainer)
        self.failures_only = failures_only

    def build(self, output, prompts, rewards, feedbacks=None):
        ctx = super().build(output, prompts, rewards, feedbacks=feedbacks)
        # ``rewards`` and the returned mask are both already sliced to this process.
        # Stash them for RolloutDumpCallback: the trainer fires
        # on_self_distillation_batch_prepared a few lines after this returns, but does
        # not pass rewards along, and this is the last place they are in scope.
        self.trainer._last_rollout_rewards = rewards.detach().clone()
        if not self.failures_only:
            return ctx
        failed = (rewards < self.trainer.args.success_reward_threshold).float()
        mask = ctx["self_distillation_mask"] * failed
        ctx["self_distillation_mask"] = mask
        self.last_metrics["self_distillation/reprompt_sample_fraction"] = (
            self.trainer.accelerator.gather(mask).mean().item()
        )
        return ctx


class RolloutDumpCallback(TrainerCallback):
    """Write rollout text to disk, since SDPO has nowhere to log it.

    ``SDPOConfig`` subclasses ``_BaseConfig``, not ``GRPOConfig``, so it has no
    ``log_completions``/``num_completions_to_print`` — the knobs grpo_lora.py sets.
    Every SDPO run so far therefore produced reward curves with no rollout text to
    explain them. ``SDPOTrainer`` instead exposes its own hooks via
    ``_dispatch_self_distillation_callback``, which calls any same-named method on a
    registered callback; ``on_self_distillation_batch_prepared`` is the one that fires
    once the rollouts, the teacher reprompts and the distillation mask all exist.

    Each record pairs the student's completion with the teacher reprompt actually built
    for it, so a run can be read as "what did the policy say, what was the teacher shown
    instead, and was this rollout in the distillation set".
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
        mask = self_distillation_mask

        def text(ids):
            return processing_class.decode(ids[ids != pad], skip_special_tokens=True)

        with self.path.open("a") as fh:
            for i in range(min(self.per_step, completion_ids.size(0))):
                rec = {
                    "step": state.global_step,
                    "reward": None if rewards is None else round(rewards[i].item(), 4),
                    "distilled": None if mask is None else bool(mask[i].item()),
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


def _prepare(recs, cap=None, with_feedback=True):
    """Records → a Dataset carrying exactly KEEP_COLS + prompt + privileged_context.

    Mapping each source to this fixed shape *before* concatenation is what lets the
    two be merged: the raw records differ (SemCor adds ``synset1``/``synset2``), the
    mapped ones do not.
    """
    ds = Dataset.from_list(recs)
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(
        partial(format_prompt, with_feedback=with_feedback), remove_columns=drop
    )


def build_dataset(split, cap=None, with_feedback=True, semcor_path=None):
    """Rollout set for one split, optionally mixing in the saved SemCor pairs.

    ``semcor_path`` points at the file ``semcor_pairs.py`` writes, exactly as in
    grpo_lora.py — the pair set is materialised once (``run_train.sh`` does it) rather
    than resampled per run, which is what makes a run reproducible and keeps NLTK off
    the training host. Those pairs are the only ones carrying gold senses, so they are
    both the only ones ``reward_wic_gloss`` can score and the only ones whose SDPO hint
    can name what the word actually means; MCL-WiC pairs keep the one-bit verdict hint.
    """
    parts = [_prepare(sd.load_mclwic(split), cap=cap, with_feedback=with_feedback)]
    if semcor_path:
        sc = semcor_pairs.load_pairs(semcor_path)
        print(f"[{split}] semcor: +{len(sc)} gloss-annotated pairs")
        parts.append(_prepare(sc, cap=cap, with_feedback=with_feedback))
    if len(parts) == 1:
        return parts[0]
    # Interleave the sources: each prompt forms its own rollout group, but leaving
    # them in blocks would make every optimizer step see one source only.
    return concatenate_datasets(parts).shuffle(seed=42)


def main():
    ap = argparse.ArgumentParser()
    # --- everything down to --resume mirrors grpo_lora.py knob for knob ---------- #
    # SDPO is that objective plus an on-policy distillation term, so a comparison is
    # only readable if the GRPO half is configured identically. The SDPO-only flags
    # start at --distillation-weight.
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Backward micro-batch, in completions. This is the knob that OOMs, and "
        "the ONLY thing it controls is peak activation memory — see --generation-batch "
        "for the one that sets how many prompts each gradient averages over.",
    )
    ap.add_argument(
        "--num-generations",
        type=int,
        default=16,
        help="Rollouts per prompt (the GRPO group size the advantage is centred on).",
    )
    ap.add_argument(
        "--generation-batch",
        type=int,
        default=256,
        help="Completions per optimizer step; grad-accum is derived from it as "
        "(this // --batch-size). Prompt groups per step = this // --num-generations, "
        "and that is what the gradient averages over. Deriving grad-accum this way "
        "keeps the group count fixed when you trade micro-batch size for memory — the "
        "previous 'gradient_accumulation_steps=32//batch_size' pinned the generation "
        "batch at 32, so --batch-size bought OOM and left the group count at 2.",
    )
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument("--vllm-gpu-mem", type=float, default=0.4)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument(
        "--semcor-pairs",
        nargs="?",
        const=str(semcor_pairs.DEFAULT_PAIRS),
        default=None,
        help="Mix the saved SemCor pair set into the training rollouts (bare flag uses "
        f"{semcor_pairs.DEFAULT_PAIRS}). Build the file with 'uv run python "
        "src/semcor_pairs.py', whose CLI carries the sampling knobs (--max-per-lemma, "
        "--min-confusability, --keep-test-lemmas). Under SDPO these pairs do double "
        "duty: reward_wic_gloss can score them, and their hint names the gold sense "
        "instead of just the verdict bit. The dev set stays pure MCL-WiC so the eval "
        "curve remains comparable across runs.",
    )
    ap.add_argument(
        "--gloss-reward-weight",
        type=float,
        default=1.0,
        help="Weight on reward_wic_gloss. At 1.0 the term spans ±0.15, which is what "
        "the shape/accuracy invariant in tests/test_sense_rewards.py was checked "
        "against; raising it eats that headroom.",
    )
    # --- Dr. GRPO / DAPO -------------------------------------------------- #
    # The two papers disagree only on how token losses are aggregated, so that axis
    # is a flag; every other term below is shared and on by default.
    ap.add_argument(
        "--loss-type",
        default="dr_grpo",
        choices=["dr_grpo", "dapo", "grpo", "bnpo"],
        help="Token-loss aggregation. 'dr_grpo' normalises by the constant "
        "--max-completion-length (Dr. GRPO); 'dapo' normalises by the active-token "
        "count of the whole accumulated batch. Both remove the length bias that "
        "makes plain 'grpo' prefer short positive-advantage completions; they are "
        "mutually exclusive, hence the choice.",
    )
    ap.add_argument(
        "--scale-rewards",
        default="none",
        choices=["none", "group", "batch"],
        help="Dr. GRPO's second correction: dividing the advantage by the group std "
        "('group', TRL's default) up-weights prompts the policy already answers "
        "consistently, which is a question-level difficulty bias. 'none' keeps the "
        "mean-centred advantage only.",
    )
    ap.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Lower PPO clip bound.",
    )
    ap.add_argument(
        "--epsilon-high",
        type=float,
        default=0.28,
        help="Upper PPO clip bound (DAPO 'clip-higher'). Decoupling it from "
        "--epsilon gives low-probability tokens room to grow, which is what keeps "
        "the policy from collapsing to a single sampled mode; 0.28 is the paper's "
        "value. Pass the same value as --epsilon for symmetric clipping.",
    )
    ap.add_argument(
        "--soft-punish-cache",
        type=int,
        default=128,
        help="DAPO overlong reward shaping (their Eq. 13): completions in the last "
        "N tokens before --max-completion-length take a linearly ramped penalty "
        "instead of being truncated and silently discarded. Set 0 to fall back to "
        "overlong *filtering* (mask_truncated_completions) instead — the two are "
        "alternatives in the paper, not a stack.",
    )
    ap.add_argument(
        "--overlong-penalty",
        type=float,
        default=0.2,
        help="Weight on the overlong term. The raw reward bottoms out at -1.0, which "
        "would blow the shape/accuracy invariant in sense_rewards (0.4 of headroom "
        "against a 1.5 accuracy gap, already spent down to 0.2 by the repetition "
        "penalty), so it is scaled to match the repetition penalty's magnitude.",
    )
    ap.add_argument("--resume", nargs="?", const=True, default=None)
    # --- SDPO: on-policy distillation from a reprompted teacher ------------- #
    ap.add_argument(
        "--distillation-weight",
        type=float,
        default=0.1,
        help="Convex blend: loss = (1-w)*policy_grad + w*self_distillation. 1.0 is "
        "pure SDPO, 0.0 collapses to GRPO. 0.1 is the paper's value: SDPO section 4.5 "
        "writes the hybrid as lambda*A_GRPO + (1-lambda)*A_SDPO with lambda=0.9, i.e. "
        "w = 1 - lambda = 0.1, and Figure 11 measures it on Qwen3-0.6B specifically, "
        "where the hybrid beats pure SDPO because 'in a weaker model the SDPO "
        "advantages are less reliable'. w is NOT the effective share of the gradient: "
        "the policy term is bounded by |advantage| (~0.02/token as logged) while the "
        "distillation term is |student_logp - teacher_logp| against a teacher shown "
        "the answer (~0.14/token as logged), a ~7x gap. At the old default of 0.3 "
        "distillation therefore carried ~2x the policy gradient and dev accuracy fell "
        "monotonically (0.82 -> 0.75 over 300 steps) while the same run at w=0.0 held "
        "0.80-0.85 for 2000 steps. Compare self_distillation/policy_loss against "
        "self_distillation/distillation_loss in wandb when retuning.",
    )
    ap.add_argument(
        "--distill-failures-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the distillation term only to rollouts that scored below "
        "--success-reward-threshold. On by default because TRL's "
        "dont_reprompt_on_self_success does not do it (see FailedRolloutContextBuilder) "
        "and without it the teacher overrides the reward on rollouts that were already "
        "correct.",
    )
    ap.add_argument(
        "--no-successful-teacher",
        action="store_true",
        help="Do not reprompt the teacher with a successful sibling rollout, leaving "
        "the privileged context as the only teacher conditioning. This is the flag to "
        "use when the question is what the feedback is worth: with the demonstration "
        "path on, environment_feedback_only_without_solution routes ~95%% of samples "
        "to a sibling's answer and only ~5%% to the hint, so the run measures "
        "demonstration copying, not feedback.",
    )
    ap.add_argument(
        "--teacher-kind",
        default="ema",
        choices=["base", "live", "ema"],
        help="SDPO Table 4 ranks these on best/avg accuracy: trust-region 50.6/45.6 > "
        "ema 49.3/45.3 > frozen-at-init 48.8/44.4 >> unregularized live 36.1/29.8, "
        "which diverges. 'base' is the frozen-at-init row (the merged SFT weights, "
        "reached by disabling the adapter); it works, but caps the student at what the "
        "SFT model can recognise, and the paper's bootstrapping claim (Figure 10 right: "
        "the student surpasses the initial teacher) rests on the teacher improving too. "
        "Switching costs nothing at step 0: trl builds the EMA teacher as a second LoRA "
        "adapter initialised to zero, and a zero LoRA *is* the base model, so 'ema' "
        "starts identical to 'base' and only diverges as the EMA fills in. Never "
        "'live' — Table 4's unregularized teacher diverges.",
    )
    ap.add_argument(
        "--teacher-update-rate",
        type=float,
        default=0.01,
        help="EMA teacher rate. SDPO Table 12 uses 0.01 for the with-feedback setup "
        "(0.05 is their no-feedback value); Table 4's regularized teachers use 0.01.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Rollout temperature; 0.6 is the GRPO run's value and the repo's eval "
        "convention. SDPO trains at 1.0 and only validates at 0.6/0.95 (Table 12), and "
        "0.6 is also why success_group_fraction sits at ~0.95 — low-temperature "
        "rollouts rarely disagree, so few groups are hard enough for the distillation "
        "term to matter. Raise it only when the GRPO baseline is not the comparison.",
    )
    ap.add_argument(
        "--distillation-topk",
        type=int,
        default=20,
        help="Support size for logit-level distillation. SDPO Figure 10 finds "
        "logit-level > token-level > sequence-level credit assignment, and Table 12 "
        "uses K=20 for the with-feedback setup (K=100 without). Set to 0 to fall back "
        "to TRL's token-level 'sampled_token' mode.",
    )
    ap.add_argument(
        "--success-reward-threshold",
        type=float,
        default=1.0,
        help="Minimum total reward for a rollout to be reused as a demonstration. "
        "The shaping terms sit on top of the accuracy term (+1.0 correct / -0.5 wrong "
        "/ -1.0 no verdict), whose ceiling puts a perfectly-formed wrong answer at "
        "0.0, so 1.0 means 'correct verdict and reasonably well formed' and no "
        "incorrect rollout can qualify.",
    )
    ap.add_argument(
        "--no-gold-feedback",
        action="store_true",
        help="Do not supply the gold verdict as privileged context on MCL-WiC pairs; "
        "those groups then teach only from successful sibling rollouts, and groups "
        "where every rollout fails carry no distillation signal (the GRPO failure mode "
        "SDPO is here to fix). It does NOT suppress --semcor-pairs' gloss feedback, "
        "which is a different and strictly more informative signal: combining the two "
        "gives a rollout set where only the gold-sense pairs are hinted, which is the "
        "clean way to ask what the gloss feedback is worth.",
    )
    ap.add_argument(
        "--dump-rollouts",
        type=int,
        default=25,
        metavar="EVERY_N_STEPS",
        help="Append rollout text to <output-dir>/rollouts.jsonl every N steps (0 "
        "disables). SDPOConfig has no log_completions — it subclasses _BaseConfig, not "
        "GRPOConfig — so this is the only way to see what the policy actually wrote; "
        "each record carries the reward, whether the rollout was in the distillation "
        "set, the completion, and the teacher reprompt built for it. This is the SDPO "
        "stand-in for grpo_lora.py's log_completions=True.",
    )
    # --- eval: grpo_lora.py's settings, exposed because SDPO evals cost more --- #
    ap.add_argument(
        "--eval-prompts",
        type=int,
        default=200,
        help="Dev prompts per eval; 0 uses the whole 1000-pair dev split. 200 matches "
        "grpo_lora.py's hard-coded cap, so the two runs' eval curves are the same "
        "measurement. The subset is FIXED (shuffle(seed=42).select), so the wobble "
        "between evals is pure sampling noise, sd ~= sqrt(p(1-p)/(prompts * "
        "generations)) — at 200x1 that floor is ~0.037 in reward units against an "
        "observed 0.062. Spending a fixed generation budget on more prompts beats "
        "spending it on more generations: both cut variance by the same 1/(n*k), but "
        "more prompts also kills the bias of scoring a 200-pair slice of dev.",
    )
    ap.add_argument(
        "--num-generations-eval",
        type=int,
        default=1,
        help="Rollouts per dev prompt, as in grpo_lora.py. SDPO Table 12 validates "
        "with 4 (with-feedback setup) at temp 0.6/top-p 0.95. Raise --eval-prompts "
        "first; this only helps once you are already scoring all of dev. Must divide "
        "--batch-size, which trl uses as the eval batch (grpo_config.py:1082).",
    )
    ap.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Optimizer steps between evals; 50 matches grpo_lora.py. Raise it if you "
        "raise --eval-prompts to the full split — scoring all of dev costs ~5x the "
        "200-prompt eval, which is ~36%% of wall-clock here against ~9%% at 200.",
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
    grad_accum = args.generation_batch // args.batch_size
    prompt_groups = args.generation_batch // args.num_generations
    print(
        f"batching: micro={args.batch_size} x grad_accum={grad_accum} = "
        f"{args.generation_batch} completions/step / {args.num_generations} rollouts "
        f"= {prompt_groups} prompt groups averaged per optimizer step"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    with_feedback = not args.no_gold_feedback
    train_ds = build_dataset(
        "train", with_feedback=with_feedback, semcor_path=args.semcor_pairs
    )
    dev_ds = build_dataset(
        "dev", cap=args.eval_prompts or None, with_feedback=with_feedback
    )
    print(train_ds[0])

    model, peft_config = load_policy(
        args.model,
        dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout),
    )

    # Wider than grpo_lora.py's 512: the teacher reprompt is the student prompt plus a
    # successful sibling completion plus the hint, and max_reprompt_len below budgets
    # from the same number.
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
        # Logit-level SDPO. The tail bucket is the "term capturing the tail
        # probability" of the paper's top-K approximation (their Appendix A.3);
        # without it the divergence is over the renormalised top-K only.
        distill_mode_kwargs = dict(
            distillation_mode="topk_logits",
            distillation_topk=args.distillation_topk,
            distillation_add_tail=True,
        )
    else:
        distill_mode_kwargs = dict(distillation_mode="sampled_token")

    reward_funcs = [as_text_reward(f) for f in REWARDS]
    reward_weights = [
        args.gloss_reward_weight if f.__name__ == "reward_wic_gloss" else 1.0
        for f in REWARDS
    ]
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
    # Overlong shaping and overlong filtering are the two alternatives DAPO offers
    # for truncated rollouts; masking on top of the shaping would zero the loss on
    # exactly the completions the penalty is meant to teach from.
    mask_truncated = args.soft_punish_cache == 0
    print(
        f"objective: loss_type={args.loss_type} scale_rewards={args.scale_rewards} "
        f"clip=[{args.epsilon}, {args.epsilon_high}] beta={args.beta} "
        + (
            f"overlong=shape(cache={args.soft_punish_cache}, w={args.overlong_penalty})"
            if not mask_truncated
            else "overlong=mask"
        )
    )
    print(
        f"distillation: w={args.distillation_weight} teacher={args.teacher_kind}"
        f"(rate={args.teacher_update_rate}) "
        f"mode={distill_mode_kwargs['distillation_mode']} "
        f"failures_only={args.distill_failures_only} "
        f"successful_teacher={not args.no_successful_teacher} "
        f"gold_feedback={with_feedback}"
    )

    run_name = "qwen-lora-sdpo-wic"
    output_dir = f"./{run_name}"
    training_args = SDPOConfig(
        output_dir=output_dir,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations_eval,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=mask_truncated,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        reward_weights=reward_weights,
        beta=args.beta,
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
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=6,
        logging_steps=25,
        report_to="wandb",
        run_name=run_name,
        # --- SDPO-only: everything above this line matches grpo_lora.py ---------- #
        # Reverse-KL is SDPO Table 12's with-feedback divergence (Jensen-Shannon is
        # their no-feedback one). alpha=1.0 is reverse KL in trl's parameterisation.
        distillation_alpha=1.0,
        **distill_mode_kwargs,
        distillation_weight=args.distillation_weight,
        teacher_model_kind=args.teacher_kind,
        teacher_update_rate=args.teacher_update_rate,
        use_successful_as_teacher=not args.no_successful_teacher,
        success_reward_threshold=args.success_reward_threshold,
        dont_reprompt_on_self_success=True,
        include_environment_feedback=with_feedback,
        environment_feedback_only_without_solution=True,
        reprompt_template=REPROMPT_TEMPLATE,
        solution_template=SOLUTION_TEMPLATE,
        feedback_template=FEEDBACK_TEMPLATE,
        max_reprompt_len=prompt_headroom + 2 * args.max_completion_length,
        **vllm_kwargs,
    )

    trainer = SDPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        peft_config=peft_config,
    )

    # Installed unconditionally: even with --no-distill-failures-only it is what stashes
    # the rewards RolloutDumpCallback reads back.
    trainer.teacher_context_builder = FailedRolloutContextBuilder(
        trainer, failures_only=args.distill_failures_only
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
