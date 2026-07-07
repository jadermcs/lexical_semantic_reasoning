"""GRPO for the sense-modeling ablations, warm-started from an SFT checkpoint.

Train on any subset of tasks at once (multitask) via ``--tasks``:

  direct   -> config 3: RL on single-usage definition generation
  triplet  -> config 4: RL on anchor/positive/negative contrastive glosses
  wic      -> reason about the gloss of each of two usages, then classify the pair
              as the same sense or a different sense (verifiable label reward)

Verifiable reward for the gloss tasks = BERTScore semantic similarity of the
generated gloss to the WordNet gold definition plus a small format term; for wic
it is the correctness of the same/different verdict plus a reasoning term. When
more than one task is given, each task's reward functions only score that task's
completions (see ``_mask_by_task``).
"""

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, concatenate_datasets
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import sense_data as sd


TASKS = ("direct", "triplet", "wic", "supersense")

# Message builder per task; format_prompt renders the prompt column from it.
_MSG_FN = {
    "direct": sd.direct_messages,
    "triplet": sd.triplet_messages,
    "wic": sd.wic_messages,
    "supersense": sd.supersense_messages,
}


# --------------------------------------------------------------------------- #
# Prompt formatting (keeps gold columns so reward fns can read them via kwargs)
# --------------------------------------------------------------------------- #
def format_prompt(rec, tokenizer, task):
    msgs = _MSG_FN[task](rec, with_target=False)
    return {"prompt": tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)}


# --------------------------------------------------------------------------- #
# Reward functions
# --------------------------------------------------------------------------- #
SELF_REF_PENALTY = -0.5


_VOWELS = "aeiou"


def _target_variants(lemma):
    """Lemma plus its common English inflections, lower-cased."""
    l = lemma.lower()
    v = {l, l + "s", l + "es", l + "ed", l + "ing", l + "d", l + "er", l + "est"}
    if l.endswith("e"):
        v |= {l[:-1] + "ing", l[:-1] + "ed", l + "r", l + "st"}
    if l.endswith("y") and len(l) > 1 and l[-2] not in _VOWELS:
        v |= {l[:-1] + "ies", l[:-1] + "ied", l[:-1] + "ier", l[:-1] + "iest"}
    # CVC words double the final consonant: run -> running, big -> bigger
    if (len(l) >= 3 and l[-1] not in _VOWELS + "wxy"
            and l[-2] in _VOWELS and l[-3] not in _VOWELS):
        d = l + l[-1]
        v |= {d + "ing", d + "ed", d + "er", d + "est"}
    return v


def _uses_target(gloss, lemma):
    """True if the gloss defines the word using the word itself (or an inflection)."""
    return bool(set(sd._tok(gloss)) & _target_variants(lemma))


MIN_CONTENT_WORDS = 2    # glosses need at least this many non-stopword tokens
MIN_CONTENT_PENALTY = -0.5


def _content_word_count(gloss):
    return sum(1 for t in sd._tok(gloss) if t not in ENGLISH_STOP_WORDS)


def _min_content_penalty(gloss):
    """Punish glosses with fewer than MIN_CONTENT_WORDS non-stopword tokens."""
    return MIN_CONTENT_PENALTY if _content_word_count(gloss) < MIN_CONTENT_WORDS else 0.0


# --------------------------------------------------------------------------- #
# BERTScore fidelity (batched; the model is loaded once and reused)
# --------------------------------------------------------------------------- #
_BERTSCORER = None


def _get_bertscorer():
    global _BERTSCORER
    if _BERTSCORER is None:
        from bert_score import BERTScorer

        _BERTSCORER = BERTScorer(lang="en", rescale_with_baseline=True)
    return _BERTSCORER


def bertscore_similarity(hyps, refs):
    """Batched BERTScore F1 in [0,1]; empty hypotheses score 0 without calling the model."""
    out = [0.0] * len(hyps)
    idxs = [i for i, h in enumerate(hyps) if h.strip()]
    if not idxs:
        return out
    scorer = _get_bertscorer()
    _, _, f1 = scorer.score([hyps[i] for i in idxs], [refs[i] for i in idxs])
    for i, f in zip(idxs, f1.tolist()):
        out[i] = max(0.0, min(1.0, float(f)))
    return out


LENGTH_TOL = 1.2      # free allowance: up to 1.2x the gold gloss length
LENGTH_PENALTY = -1.5  # floor, reached only once the gloss is ~4x the budget
LENGTH_RAMP = 3.0      # budgets of excess the penalty ramps over before clamping


def _length_penalty(hyp, ref):
    """Penalise glosses longer than the gold definition; in [LENGTH_PENALTY, 0].

    The ramp is deliberately gentle and clamps far out (at ~4x the budget) so the
    reward stays monotone — strictly shorter is strictly better — across the whole
    range the model explores. A tight clamp instead flat-lines once a gloss is
    long, and under GRPO's group normalisation a length reward with no within-group
    variance yields no advantage, so runaway glosses stop being pushed back.
    """
    budget = LENGTH_TOL * max(len(sd._tok(ref)), 1)
    excess = max(0, len(sd._tok(hyp)) - budget) / budget
    return LENGTH_PENALTY * min(1.0, excess / LENGTH_RAMP)


def _answer_region(text):
    """The whole definition the model emits: all text after </think>, not just the
    first line that the gloss extractor scores. Shared by both modes, which use the
    same `<think>...</think>\\ndefinition` format.

    The model reward-hacks by keeping a short first line and then dumping a long
    tail of repeated `definition: ...` lines; that tail is invisible to the
    first-line extractor, so length is measured over the entire region instead.
    """
    seg = text.split("</think>")[-1]
    if "<think>" in seg:  # unclosed <think>: there is no real answer region
        return ""
    return seg.strip()


def _gold_gloss(kwargs):
    """Gold gloss of the (shared) sense, whichever column the mode carries it in.

    Direct records store it as `gloss`; triplet records as `gloss_same` (the sense
    shared by the anchor and positive). One accessor lets the reward fns below stay
    mode-agnostic.
    """
    return kwargs["gloss_same"] if "gloss_same" in kwargs else kwargs["gloss"]


def reward_no_target(completions, **kwargs):
    """Punish glosses that define the target word using the word itself."""
    out = []
    for c, lemma in zip(completions, kwargs["lemma"]):
        out.append(SELF_REF_PENALTY if _uses_target(sd.extract_shared_gloss(c), lemma) else 0.0)
    return out


def reward_min_content(completions, **kwargs):
    """Punish glosses with fewer than MIN_CONTENT_WORDS non-stopword tokens."""
    return [_min_content_penalty(sd.extract_shared_gloss(c)) for c in completions]


def reward_length(completions, **kwargs):
    """Punish a definition region much longer than the gold definition.

    Measures the full post-</think> region (not just the extracted first line) so
    a short first line followed by a long repeated tail is still penalised.
    """
    return [
        _length_penalty(_answer_region(c), g)
        for c, g in zip(completions, _gold_gloss(kwargs))
    ]


def _think_answer_format_reward(completions, extractor):
    """Reward a present <think> block (0.1) and an extractable gloss (0.1)."""
    out = []
    for c in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if extractor(c):
            r += 0.1
        out.append(r)
    return out


def reward_format(completions, **kwargs):
    return _think_answer_format_reward(completions, sd.extract_shared_gloss)


def reward_fidelity(completions, **kwargs):
    """Similarity of the emitted gloss to its gold definition."""
    hyps = [sd.extract_shared_gloss(c) for c in completions]
    return bertscore_similarity(hyps, _gold_gloss(kwargs))


def reward_triplet_contrast(completions, **kwargs):
    """Punish the gloss for being similar to the negative (differentia) sense.

    Complements reward_fidelity, which already rewards closeness to the
    positive sense, so together they pull the gloss toward the shared sense and
    away from the contrastive one.
    """
    hyps = [sd.extract_shared_gloss(c) for c in completions]
    sim_neg = bertscore_similarity(hyps, kwargs["gloss_diff"])
    return [-s for s in sim_neg]


def _extract_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


THINK_MIN_WORDS = 6      # below this many content words, reasoning is treated as a stub
THINK_MIN_PENALTY = -0.3
THINK_DIFFERENTIA_WEIGHT = 0.3


def _think_length_penalty(completions):
    """Punish degenerate reasoning: a missing, truncated, or near-empty <think> block.

    Catches the model collapsing the reasoning step — whether by dropping the
    block entirely (no penalty elsewhere, so this is the only force keeping it),
    truncating it past the length budget (no closing tag, so `_extract_think`
    returns ""), or stubbing it (e.g. "<think>ok</think>") to farm the format
    reward's presence bonus without doing any real reasoning.
    """
    out = []
    for c in completions:
        think = _extract_think(c)
        out.append(THINK_MIN_PENALTY if _content_word_count(think) < THINK_MIN_WORDS else 0.0)
    return out


def reward_think_length(completions, **kwargs):
    return _think_length_penalty(completions)


# --------------------------------------------------------------------------- #
# WiC (word-in-context): verifiable same/different-sense classification
# --------------------------------------------------------------------------- #
WIC_CORRECT = 1.0
WIC_WRONG = -1.0


def reward_wic_accuracy(completions, **kwargs):
    """+1 for the right same/different verdict, -1 for the wrong one, 0 if absent.

    This is the verifiable signal for wic: the gold label is known, so the reward
    is exact rather than a similarity estimate.
    """
    out = []
    for c, label in zip(completions, kwargs["label"]):
        pred = sd.extract_wic_label(c)
        if not pred:
            out.append(0.0)
        else:
            out.append(WIC_CORRECT if pred == label else WIC_WRONG)
    return out


def reward_wic_format(completions, **kwargs):
    """Reward a present <think> block (0.1) and an extractable verdict (0.1)."""
    return _think_answer_format_reward(completions, sd.extract_wic_label)


# --------------------------------------------------------------------------- #
# Supersense: verifiable WordNet lexicographer-file classification
# --------------------------------------------------------------------------- #
SUPERSENSE_CORRECT = 1.0
SUPERSENSE_WRONG = -1.0
# Soft credit for an off-vocabulary answer, mapped to its nearest candidate.
# Kept below |1| so an exact candidate is always the higher-reward move (when
# right) and the riskier one (when wrong) — no incentive to hedge with OOV text.
# Also kept below the default --distill-threshold (0.5) so soft (mis-formatted)
# hits never leak into the self-distillation set; only exact answers do.
SUPERSENSE_SOFT_CORRECT = 0.4
SUPERSENSE_SOFT_WRONG = -0.4


# Sentence-embedding nearest-candidate mapping for the soft tier. Lives here (not
# in sense_data, which stays torch-free) and loads on CPU to avoid competing with
# the policy/vLLM for VRAM — it only ever encodes a handful of short strings.
#
# The candidate *names* alone ("change", "creation") embed too abstractly to
# place an off-vocab paraphrase near the right one, so each is anchored by its
# standard WordNet lexicographer-file description (e.g. "biological growth" then
# lands on "change" rather than "creation"). The returned label is still the bare
# suffix; only the embedding text is enriched.
_SUPERSENSE_DESC = {
    "noun": {
        "act": "nouns denoting acts or actions",
        "animal": "nouns denoting animals",
        "artifact": "nouns denoting man-made objects",
        "attribute": "nouns denoting attributes of people and objects",
        "body": "nouns denoting body parts",
        "cognition": "nouns denoting cognitive processes and contents",
        "communication": "nouns denoting communicative processes and contents",
        "event": "nouns denoting natural events",
        "feeling": "nouns denoting feelings and emotions",
        "food": "nouns denoting foods and drinks",
        "group": "nouns denoting groupings of people or objects",
        "location": "nouns denoting spatial position",
        "motive": "nouns denoting goals",
        "object": "nouns denoting natural objects (not man-made)",
        "person": "nouns denoting people",
        "phenomenon": "nouns denoting natural phenomena",
        "plant": "nouns denoting plants",
        "possession": "nouns denoting possession and transfer of possession",
        "process": "nouns denoting natural processes",
        "quantity": "nouns denoting quantities and units of measure",
        "relation": "nouns denoting relations between people or things or ideas",
        "shape": "nouns denoting two and three dimensional shapes",
        "state": "nouns denoting stable states of affairs",
        "substance": "nouns denoting substances",
        "time": "nouns denoting time and temporal relations",
    },
    "verb": {
        "body": "verbs of grooming, dressing and bodily care",
        "change": "verbs of size, temperature change, intensifying",
        "cognition": "verbs of thinking, judging, analyzing, doubting",
        "communication": "verbs of telling, asking, ordering, singing",
        "competition": "verbs of fighting, athletic activities",
        "consumption": "verbs of eating and drinking",
        "contact": "verbs of touching, hitting, tying, digging",
        "creation": "verbs of sewing, baking, painting, performing",
        "emotion": "verbs of feeling",
        "motion": "verbs of walking, flying, swimming",
        "perception": "verbs of seeing, hearing, feeling",
        "possession": "verbs of buying, selling, owning",
        "social": "verbs of political and social activities and events",
        "stative": "verbs of being, having, spatial relations",
        "weather": "verbs of raining, snowing, thawing, thundering",
    },
}
_ENCODER = None
_CAND_EMB = {}


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    return _ENCODER


def _candidate_embeddings(pos):
    """Normalized embeddings of ``pos``'s candidates (via description), once."""
    if pos not in _CAND_EMB:
        cands = sd.SUPERSENSES[pos]
        texts = [_SUPERSENSE_DESC[pos][c] for c in cands]
        emb = _get_encoder().encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )
        _CAND_EMB[pos] = (cands, emb)
    return _CAND_EMB[pos]


def _nearest_candidates(regions, poss):
    """The semantically nearest candidate name for each (answer region, pos)."""
    q = _get_encoder().encode(regions, convert_to_tensor=True, normalize_embeddings=True)
    out = []
    for i, pos in enumerate(poss):
        cands, emb = _candidate_embeddings(pos)
        out.append(cands[int((emb @ q[i]).argmax())])  # cosine: both normalized
    return out


def reward_supersense_accuracy(completions, **kwargs):
    """Graded reward over the closed candidate set.

    An exact candidate scores +1/-1 as before (verifiable: the gold lexicographer
    file is known). An answer that names no candidate but still says something
    ("biological growth" for gold "change") is mapped to its nearest candidate by
    sentence-embedding similarity and given the smaller +/-0.4; a blank or
    unclosed <think> stays 0.

    The soft tier matters under GRPO: when no completion in a group emits an exact
    label, an all-zero group has no advantage and teaches nothing, so the policy
    is never pulled from its free-form paraphrase toward the exact label word.
    """
    labels, poss = kwargs["supersense"], kwargs["pos"]
    out = [0.0] * len(completions)
    soft_idx, soft_regions, soft_pos = [], [], []
    for i, (c, label, pos) in enumerate(zip(completions, labels, poss)):
        pred = sd.extract_supersense(c, pos)
        if pred:
            out[i] = SUPERSENSE_CORRECT if pred == label else SUPERSENSE_WRONG
        elif _answer_region(c):  # said something, but no exact candidate in it
            soft_idx.append(i)
            soft_regions.append(_answer_region(c))
            soft_pos.append(pos)
        # else: blank / unclosed -> stays 0.0
    if soft_idx:
        for i, near in zip(soft_idx, _nearest_candidates(soft_regions, soft_pos)):
            out[i] = SUPERSENSE_SOFT_CORRECT if near == labels[i] else SUPERSENSE_SOFT_WRONG
    return out


def reward_supersense_format(completions, **kwargs):
    """Reward a present <think> block (0.1) and an extractable category (0.1).

    Unlike the other format rewards, the extractor is POS-aware, so it cannot use
    the shared ``_think_answer_format_reward`` (which passes a single-arg extractor).
    """
    out = []
    for c, pos in zip(completions, kwargs["pos"]):
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if sd.extract_supersense(c, pos):
            r += 0.1
        out.append(r)
    return out


REWARDS = {
    "direct": [
        reward_fidelity, reward_no_target, reward_min_content,
        reward_length, reward_format, reward_think_length,
    ],
    "triplet": [
        reward_fidelity, reward_triplet_contrast, reward_no_target,
        reward_min_content, reward_length, reward_format, reward_think_length,
    ],
    "wic": [
        reward_wic_accuracy, reward_wic_format, reward_think_length,
    ],
    "supersense": [
        reward_supersense_accuracy, reward_supersense_format, reward_think_length,
    ],
}
KEEP_COLS = {
    "direct": ["lemma", "gloss"],
    "triplet": ["lemma", "gloss_same", "gloss_diff"],
    # MCL-WiC carries no glosses; only the verifiable same/different label is scored.
    "wic": ["lemma", "label"],
    # pos selects the candidate label space for extraction; supersense is the gold label.
    "supersense": ["lemma", "pos", "supersense"],
}
# The fidelity reward already scores a completion against its gold gloss(es); the
# trace saver reuses it to decide which generations are "successful". For wic and
# supersense, "successful" means the verifiable label is correct, so accuracy plays
# the fidelity role.
FIDELITY = {
    "direct": reward_fidelity,
    "triplet": reward_fidelity,
    "wic": reward_wic_accuracy,
    "supersense": reward_supersense_accuracy,
}


# --------------------------------------------------------------------------- #
# Multitask: run a task's reward fn only on that task's completions
# --------------------------------------------------------------------------- #
def _mask_by_task(task, fn):
    """Wrap a reward fn so it scores only rows whose ``task`` column matches.

    Under GRPO each prompt's ``num_generations`` completions form one group and
    all share a task, so the other tasks' reward fns return all-zero for the group
    — a constant reward yields zero advantage after group normalisation, leaving
    training untouched. Non-matching rows carry padded ("") gold columns from the
    schema union, so they must be filtered out before the wrapped fn reads them.
    """
    def wrapped(completions, **kwargs):
        tasks = kwargs["task"]
        idxs = [i for i, t in enumerate(tasks) if t == task]
        out = [0.0] * len(completions)
        if not idxs:
            return out
        sub_completions = [completions[i] for i in idxs]
        sub_kwargs = {
            k: ([v[i] for i in idxs] if isinstance(v, list) and len(v) == len(completions) else v)
            for k, v in kwargs.items()
        }
        for i, s in zip(idxs, fn(sub_completions, **sub_kwargs)):
            out[i] = s
        return out

    # GRPOTrainer logs/derives metric names from the fn name; keep them unique.
    wrapped.__name__ = f"{task}_{getattr(fn, '__name__', 'reward')}"
    return wrapped


# --------------------------------------------------------------------------- #
# Self-distillation: log successful reasoning traces for later SFT
# --------------------------------------------------------------------------- #
def make_trace_saver(mode, path, threshold):
    """Pass-through reward fn that appends high-reward completions to a JSONL.

    Returns all-zero rewards, so it never perturbs training (a constant reward
    yields zero advantage after GRPO's group normalisation). It records the
    prompt + generated reasoning/gloss + fidelity score + gold columns; a
    downstream builder can turn the best generations into an SFT dataset.

    Writes one file per process rank to avoid interleaved appends under
    multi-GPU launches.
    """
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    out_path = Path(f"{path}.rank{rank}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keep = KEEP_COLS[mode]
    fidelity = FIDELITY[mode]

    def save_successful_traces(completions, **kwargs):
        scores = fidelity(completions, **kwargs)
        prompts = kwargs.get("prompts")
        with out_path.open("a", encoding="utf-8") as f:
            for i, (c, s) in enumerate(zip(completions, scores)):
                if s < threshold:
                    continue
                rec = {"mode": mode, "score": round(float(s), 4), "completion": c}
                if prompts is not None:
                    rec["prompt"] = prompts[i]
                for col in keep:
                    rec[col] = kwargs[col][i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return [0.0] * len(completions)

    # GRPOTrainer logs/derives metric names from the fn name.
    save_successful_traces.__name__ = "save_successful_traces"
    return save_successful_traces


def _load_split(task, split):
    """Load one task/split, building all datasets on first use if missing.

    ``wic`` trains on the gold MCL-WiC benchmark (verifiable same/different label,
    no glosses); the WordNet-built ``direct``/``triplet`` splits are generated on
    first use if absent.
    """
    if task == "wic":
        return sd.load_mclwic(split)
    try:
        return sd.load_split(task, split)
    except FileNotFoundError:
        sd.save_dataset(sd.build_dataset())
        return sd.load_split(task, split)


def _task_dataset(task, split, tokenizer, dev_cap=None):
    """Rendered prompts + kept gold columns + a ``task`` tag for one task/split."""
    recs = _load_split(task, split)
    ds = Dataset.from_list(recs)
    if dev_cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(dev_cap, len(ds))))
    fmt = partial(format_prompt, tokenizer=tokenizer, task=task)
    drop = [c for c in ds.column_names if c not in KEEP_COLS[task]]
    ds = ds.map(fmt, remove_columns=drop)
    return ds.add_column("task", [task] * len(ds))


def _combine(tasks, split, tokenizer, dev_cap=None):
    """Concatenate per-task datasets, padding each to the shared column union.

    Different tasks keep different gold columns; ``concatenate_datasets`` needs a
    single schema, so missing columns are filled with "" (all kept columns are
    strings). ``_mask_by_task`` filters these padded rows back out per reward fn.
    """
    parts = [_task_dataset(t, split, tokenizer, dev_cap) for t in tasks]
    all_cols = sorted({c for p in parts for c in p.column_names})
    padded = []
    for p in parts:
        for c in all_cols:
            if c not in p.column_names:
                p = p.add_column(c, [""] * len(p))
        padded.append(p.select_columns(all_cols))
    return concatenate_datasets(padded).shuffle(seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        required=True,
        help="One or more tasks to train on jointly, e.g. --tasks direct triplet wic.",
    )
    ap.add_argument("--vllm-server-host", default=None)
    ap.add_argument("--vllm-server-port", type=int, default=8000)
    ap.add_argument(
        "--distill-out",
        default=None,
        help="If set, append completions with fidelity >= --distill-threshold to "
        "'<path>.rank<N>.jsonl' for self-distillation. Does not affect training.",
    )
    ap.add_argument("--distill-threshold", type=float, default=0.5)
    args = ap.parse_args()

    # De-duplicate while preserving order so the run name is stable.
    tasks = list(dict.fromkeys(args.tasks))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = _combine(tasks, "train", tokenizer)
    dev_ds = _combine(tasks, "dev", tokenizer, dev_cap=200)
    print(train_ds[0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", dtype=torch.bfloat16,
        trust_remote_code=True, attn_implementation="sdpa",
    )

    vllm_kwargs = {}
    if args.vllm_server_host:
        vllm_kwargs = dict(
            use_vllm=True, vllm_mode="server",
            vllm_server_host=args.vllm_server_host, vllm_server_port=args.vllm_server_port,
        )
    else:
        vllm_kwargs = dict(use_vllm=True, vllm_max_model_length=1024)

    run_name = f"qwen-sense-grpo-{'-'.join(tasks)}"
    output_dir = f"./{run_name}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=8,
        max_completion_length=1024,
        optim="paged_adamw_8bit",
        temperature=1.0,
        top_p=0.95,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=50,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        bf16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        log_completions=True,
        num_completions_to_print=8,
        report_to="wandb",
        run_name=run_name,
        use_liger_kernel=True,
        **vllm_kwargs,
    )

    # Each task's reward fns (and optional trace saver) are masked to their own
    # task so they never perturb the other tasks' groups.
    reward_funcs = []
    for task in tasks:
        funcs = list(REWARDS[task])
        if args.distill_out:
            funcs.append(make_trace_saver(task, args.distill_out, args.distill_threshold))
        reward_funcs.extend(_mask_by_task(task, fn) for fn in funcs)
    if args.distill_out:
        print(
            f"Self-distillation: saving completions with fidelity >= "
            f"{args.distill_threshold} to {args.distill_out}.rank*.jsonl"
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
    )

    last = None
    out = Path(output_dir)
    if out.exists():
        cks = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if cks:
            last = str(cks[-1])
            print(f"Resuming from checkpoint: {last}")
    trainer.train(resume_from_checkpoint=last)
    trainer.save_model(output_dir)
    print(f"Saved final adapter → {output_dir}")


if __name__ == "__main__":
    main()
