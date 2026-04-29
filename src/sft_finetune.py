"""
SFT pretraining for the GRPO WiC task.

Builds a synthetic Word-in-Context dataset from WordNet so the model can learn
the <think>…</think><answer>true|false</answer> response format expected by
``grpo_finetune.py`` *before* RL starts.

For each (lemma, pos) with multiple sense examples we emit:
  * positive pairs (label=1): two example sentences from the same synset
  * negative pairs (label=0): example sentences from two different synsets

The assistant target follows the GRPO system prompt exactly:
  <think>
    1. Gloss for use 1: <definition of synset 1>
    2. Gloss for use 2: <definition of synset 2>
    3. Do these glosses describe the same concept? yes/no
  </think><answer>true|false</answer>
"""

import argparse
import random
import re
from functools import partial
from itertools import combinations
from pathlib import Path

import torch
import wn
from datasets import Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "You are a linguistic expert specializing in word sense disambiguation. "
    "Given a target word and two sentences, determine whether the word is used "
    "in the same sense in both sentences. "
    "The target word is marked with <t> tags in each sentence.\n\n"
    "First, reason step by step inside <think> tags:\n"
    "  1. Gloss for use 1: short dictionary-style definition\n"
    "  2. Gloss for use 2: short dictionary-style definition\n"
    "  3. Do these glosses describe the same concept? yes/no\n"
    "Then provide your final answer inside <answer> tags as exactly 'true' or 'false'.\n"
    "Format: <think>your reasoning here</think><answer>true or false</answer>"
)

POS_MAP = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def mark_target(sentence: str, word: str) -> str:
    """Wrap the first occurrence of *word* (case-insensitive) with <t> tags."""
    pattern = rf"\b({re.escape(word)}\w*)\b"
    if re.search(pattern, sentence, flags=re.IGNORECASE):
        return re.sub(pattern, r"<t> \1 </t>", sentence, count=1, flags=re.IGNORECASE)
    # fall back to lemma form if surface form not literally present
    return sentence + f" <t> {word} </t>"


def _load_wordnet(lexicon: str) -> wn.Wordnet:
    """Load the requested wn lexicon, downloading it on first use."""
    try:
        return wn.Wordnet(lexicon)
    except wn.Error:
        wn.download(lexicon)
        return wn.Wordnet(lexicon)


def build_wordnet_dataset(
    lexicon: str = "oewn:2024",
    max_pairs_per_lemma: int = 4,
    max_total: int | None = 50000,
    seed: int = 42,
) -> Dataset:
    """Generate synthetic WiC examples from WordNet sense examples."""
    en = _load_wordnet(lexicon)
    rng = random.Random(seed)

    by_lemma: dict[tuple[str, str], list] = {}
    for syn in en.synsets():
        if not syn.examples():
            continue
        pos = syn.pos
        for lemma in syn.lemmas():
            if " " in lemma or "_" in lemma:
                continue
            by_lemma.setdefault((lemma.lower(), pos), []).append(syn)

    records: list[dict] = []
    items = list(by_lemma.items())
    rng.shuffle(items)

    for (lemma, pos), synsets in items:
        # gather (synset, example) pool
        sense_examples = [
            (s, ex) for s in synsets for ex in s.examples() if lemma in ex.lower()
        ]
        if len(sense_examples) < 2:
            continue

        # positives: same synset, different sentence
        positives = []
        by_syn: dict = {}
        for s, ex in sense_examples:
            by_syn.setdefault(s.id, []).append((s, ex))
        for pairs in by_syn.values():
            positives.extend(combinations(pairs, 2))

        # negatives: different synsets
        negatives = []
        syn_names = list(by_syn.keys())
        for i in range(len(syn_names)):
            for j in range(i + 1, len(syn_names)):
                for a in by_syn[syn_names[i]]:
                    for b in by_syn[syn_names[j]]:
                        negatives.append((a, b))

        rng.shuffle(positives)
        rng.shuffle(negatives)
        k = max_pairs_per_lemma // 2
        chosen = [(p, 1) for p in positives[:k]] + [(n, 0) for n in negatives[:k]]

        for ((s1, ex1), (s2, ex2)), label in chosen:
            records.append(
                {
                    "lemma": lemma,
                    "pos": POS_MAP.get(pos, pos),
                    "word1": lemma,
                    "word2": lemma,
                    "sentence1": ex1,
                    "sentence2": ex2,
                    "gloss1": s1.definition(),
                    "gloss2": s2.definition(),
                    "label": label,
                }
            )
        if max_total and len(records) >= max_total:
            break

    rng.shuffle(records)
    if max_total:
        records = records[:max_total]
    return Dataset.from_list(records)


def format_example(example, tokenizer):
    s1 = mark_target(example["sentence1"], example["word1"])
    s2 = mark_target(example["sentence2"], example["word2"])
    same = example["gloss1"].strip() == example["gloss2"].strip()
    yes_no = "yes" if same or example["label"] == 1 else "no"
    answer = "true" if example["label"] else "false"
    think = (
        f"1. Gloss for use 1: {example['gloss1']}\n"
        f"2. Gloss for use 2: {example['gloss2']}\n"
        f"3. Do these glosses describe the same concept? {yes_no}"
    )
    assistant = f"<think>{think}</think><answer>{answer}</answer>"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Word: {example['lemma']} ({example['pos']})\n"
                f"Sentence 1: {s1}\n"
                f"Sentence 2: {s2}\n"
            ),
        },
        {"role": "assistant", "content": assistant},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lexicon", type=str, default="oewn:2024")
    parser.add_argument("--max_pairs_per_lemma", type=int, default=4)
    parser.add_argument("--max_total", type=int, default=50000)
    parser.add_argument("--eval_size", type=int, default=500)
    args = parser.parse_args()

    full = build_wordnet_dataset(
        lexicon=args.lexicon,
        max_pairs_per_lemma=args.max_pairs_per_lemma,
        max_total=args.max_total,
    )
    print(f"Generated {len(full)} WordNet WiC pairs")
    split = full.train_test_split(test_size=args.eval_size, seed=42)
    dataset = DatasetDict({"train": split["train"], "dev": split["test"]})

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    partial_format = partial(format_example, tokenizer=tokenizer)
    cols = ["lemma", "pos", "word1", "word2", "sentence1", "sentence2", "gloss1", "gloss2", "label"]
    dataset = dataset.map(partial_format, remove_columns=cols)
    print(dataset["train"][0]["text"])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=64,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="./qwen-wic-sft",
        dataset_text_field="text",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        torch_compile=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="qwen-wic-sft-wordnet",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=training_args,
    )

    last_checkpoint = None
    output_path = Path(training_args.output_dir)
    if output_path.exists():
        checkpoints = sorted(output_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    main()
