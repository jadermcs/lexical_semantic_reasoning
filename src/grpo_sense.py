"""GRPO for the sense-modeling ablations, warm-started from an SFT checkpoint.

  --mode direct   -> config 3: RL on single-usage definition generation
  --mode triplet  -> config 4: RL on anchor/positive/negative contrastive glosses

Verifiable reward = BERTScore semantic similarity of the generated gloss to the
WordNet gold definition plus a small format term.
"""

import argparse
import json
import os
import re
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import sense_data as sd


# --------------------------------------------------------------------------- #
# Prompt formatting (keeps gold columns so reward fns can read them via kwargs)
# --------------------------------------------------------------------------- #
def format_prompt(rec, tokenizer, mode):
    msgs = (sd.direct_messages if mode == "direct" else sd.triplet_messages)(
        rec, with_target=False
    )
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


LENGTH_TOL = 1.5      # free allowance: up to 1.5x the gold gloss length
LENGTH_PENALTY = -0.5  # max penalty, reached once the excess equals the allowance


def _length_penalty(hyp, ref):
    """Penalise glosses longer than the gold definition; in [LENGTH_PENALTY, 0]."""
    budget = LENGTH_TOL * max(len(sd._tok(ref)), 1)
    excess = max(0, len(sd._tok(hyp)) - budget) / budget
    return LENGTH_PENALTY * min(1.0, excess)


def reward_direct_fidelity(completions, **kwargs):
    golds = kwargs["gloss"]
    hyps = [sd.extract_direct_gloss(c) for c in completions]
    return bertscore_similarity(hyps, golds)


def reward_direct_no_target(completions, **kwargs):
    """Punish glosses that define the target word using the word itself."""
    out = []
    for c, lemma in zip(completions, kwargs["lemma"]):
        out.append(SELF_REF_PENALTY if _uses_target(sd.extract_direct_gloss(c), lemma) else 0.0)
    return out


def reward_direct_min_content(completions, **kwargs):
    """Punish glosses with fewer than MIN_CONTENT_WORDS non-stopword tokens."""
    return [_min_content_penalty(sd.extract_direct_gloss(c)) for c in completions]


def reward_direct_length(completions, **kwargs):
    """Punish glosses much longer than the gold definition."""
    return [
        _length_penalty(sd.extract_direct_gloss(c), g)
        for c, g in zip(completions, kwargs["gloss"])
    ]


def reward_direct_format(completions, **kwargs):
    """Reward a single concise line; penalise empty output."""
    return [0.1 if sd.extract_direct_gloss(c) else 0.0 for c in completions]


def reward_triplet_fidelity(completions, **kwargs):
    """Similarity of the shared (anchor/positive) gloss to its gold definition."""
    same = kwargs["gloss_same"]
    hyps = [sd.extract_shared_gloss(c) for c in completions]
    return bertscore_similarity(hyps, same)


def reward_triplet_contrast(completions, **kwargs):
    """Punish the gloss for being similar to the negative (differentia) sense.

    Complements reward_triplet_fidelity, which already rewards closeness to the
    positive sense, so together they pull the gloss toward the shared sense and
    away from the contrastive one.
    """
    hyps = [sd.extract_shared_gloss(c) for c in completions]
    sim_neg = bertscore_similarity(hyps, kwargs["gloss_diff"])
    return [-s for s in sim_neg]


def reward_triplet_no_target(completions, **kwargs):
    """Punish the gloss for defining the target word using the word itself."""
    out = []
    for c, lemma in zip(completions, kwargs["lemma"]):
        out.append(SELF_REF_PENALTY if _uses_target(sd.extract_shared_gloss(c), lemma) else 0.0)
    return out


def reward_triplet_min_content(completions, **kwargs):
    """Punish the gloss for having fewer than MIN_CONTENT_WORDS non-stopword tokens."""
    return [_min_content_penalty(sd.extract_shared_gloss(c)) for c in completions]


def reward_triplet_length(completions, **kwargs):
    """Punish the gloss for running much longer than its gold definition."""
    same = kwargs["gloss_same"]
    return [_length_penalty(sd.extract_shared_gloss(c), gs)
            for c, gs in zip(completions, same)]


def reward_triplet_format(completions, **kwargs):
    out = []
    for c in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if sd.extract_shared_gloss(c):
            r += 0.1
        out.append(r)
    return out


REWARDS = {
    "direct": [
        reward_direct_fidelity, reward_direct_no_target,
        reward_direct_min_content, reward_direct_length, reward_direct_format,
    ],
    "triplet": [
        reward_triplet_fidelity, reward_triplet_contrast, reward_triplet_no_target,
        reward_triplet_min_content, reward_triplet_length, reward_triplet_format,
    ],
}
KEEP_COLS = {
    "direct": ["lemma", "gloss"],
    "triplet": ["lemma", "gloss_same", "gloss_diff"],
}
# The fidelity reward already scores a completion against its gold gloss(es); the
# trace saver reuses it to decide which generations are "successful".
FIDELITY = {"direct": reward_direct_fidelity, "triplet": reward_triplet_fidelity}


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


def _load_or_build(mode):
    try:
        train, dev = sd.load_split(mode, "train"), sd.load_split(mode, "dev")
    except FileNotFoundError:
        sd.save_dataset(sd.build_dataset())
        train, dev = sd.load_split(mode, "train"), sd.load_split(mode, "dev")
    return DatasetDict({"train": Dataset.from_list(train), "dev": Dataset.from_list(dev)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--mode", choices=["direct", "triplet"], required=True)
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

    dataset = _load_or_build(args.mode)
    dataset["dev"] = dataset["dev"].shuffle(seed=42).select(range(min(200, len(dataset["dev"]))))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    fmt = partial(format_prompt, tokenizer=tokenizer, mode=args.mode)
    drop = [c for c in dataset["train"].column_names if c not in KEEP_COLS[args.mode]]
    dataset = dataset.map(fmt, remove_columns=drop)
    print(dataset["train"][0])

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

    output_dir = f"./qwen-sense-grpo-{args.mode}"
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=8,
        max_completion_length=512,
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
        logging_steps=10,
        report_to="wandb",
        run_name=f"qwen-sense-grpo-{args.mode}",
        use_liger_kernel=True,
        **vllm_kwargs,
    )
    reward_funcs = list(REWARDS[args.mode])
    if args.distill_out:
        reward_funcs.append(
            make_trace_saver(args.mode, args.distill_out, args.distill_threshold)
        )
        print(
            f"Self-distillation: saving completions with fidelity >= "
            f"{args.distill_threshold} to {args.distill_out}.rank*.jsonl"
        )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
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
