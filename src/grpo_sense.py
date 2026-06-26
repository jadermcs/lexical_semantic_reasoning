"""GRPO for the sense-modeling ablations, warm-started from an SFT checkpoint.

  --mode direct   -> config 3: RL on single-usage definition generation
  --mode triplet  -> config 4: RL on anchor/positive/negative contrastive glosses

Verifiable reward = similarity of the generated gloss to the WordNet gold
definition (token-F1 + BLEU-2, see ``sense_data.gloss_similarity``) plus a small
format term.
"""

import argparse
import re
from functools import partial
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
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


LENGTH_TOL = 1.5      # free allowance: up to 1.5x the gold gloss length
LENGTH_PENALTY = -0.5  # max penalty, reached once the excess equals the allowance


def _length_penalty(hyp, ref):
    """Penalise glosses longer than the gold definition; in [LENGTH_PENALTY, 0]."""
    budget = LENGTH_TOL * max(len(sd._tok(ref)), 1)
    excess = max(0, len(sd._tok(hyp)) - budget) / budget
    return LENGTH_PENALTY * min(1.0, excess)


def reward_direct_fidelity(completions, **kwargs):
    golds = kwargs["gloss"]
    return [sd.gloss_similarity(sd.extract_direct_gloss(c), g) for c, g in zip(completions, golds)]


def reward_direct_no_target(completions, **kwargs):
    """Punish glosses that define the target word using the word itself."""
    out = []
    for c, lemma in zip(completions, kwargs["lemma"]):
        out.append(SELF_REF_PENALTY if _uses_target(sd.extract_direct_gloss(c), lemma) else 0.0)
    return out


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
    """Mean similarity of the shared & negative glosses to their gold definitions."""
    same, diff = kwargs["gloss_same"], kwargs["gloss_diff"]
    out = []
    for c, gs, gd in zip(completions, same, diff):
        scores = [
            sd.gloss_similarity(sd.extract_shared_gloss(c), gs),
            sd.gloss_similarity(sd.extract_negative_gloss(c), gd),
        ]
        out.append(sum(scores) / 2)
    return out


def reward_triplet_no_target(completions, **kwargs):
    """Punish either gloss for defining the target word using the word itself."""
    out = []
    for c, lemma in zip(completions, kwargs["lemma"]):
        pen = 0.0
        if _uses_target(sd.extract_shared_gloss(c), lemma):
            pen += SELF_REF_PENALTY / 2
        if _uses_target(sd.extract_negative_gloss(c), lemma):
            pen += SELF_REF_PENALTY / 2
        out.append(pen)
    return out


def reward_triplet_length(completions, **kwargs):
    """Punish either gloss for running much longer than its gold definition."""
    same, diff = kwargs["gloss_same"], kwargs["gloss_diff"]
    out = []
    for c, gs, gd in zip(completions, same, diff):
        p = _length_penalty(sd.extract_shared_gloss(c), gs)
        p += _length_penalty(sd.extract_negative_gloss(c), gd)
        out.append(p / 2)
    return out


def reward_triplet_format(completions, **kwargs):
    out = []
    for c in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if sd.extract_shared_gloss(c) and sd.extract_negative_gloss(c):
            r += 0.1
        out.append(r)
    return out


REWARDS = {
    "direct": [
        reward_direct_fidelity, reward_direct_no_target,
        reward_direct_length, reward_direct_format,
    ],
    "triplet": [
        reward_triplet_fidelity, reward_triplet_no_target,
        reward_triplet_length, reward_triplet_format,
    ],
}
KEEP_COLS = {
    "direct": ["lemma", "gloss"],
    "triplet": ["lemma", "gloss_same", "gloss_diff"],
}


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
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARDS[args.mode],
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
