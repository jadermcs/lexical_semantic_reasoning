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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from rapidfuzz import fuzz, process
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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



def mark_target(sentence: str, word: str, fuzzy_threshold: float = 70.0) -> str:
    """Wrap the best match for *word* in *sentence* with <t> tags.

    Tries an exact word-boundary match first; falls back to a rapidfuzz QRatio
    search over tokens to catch inflected forms (run → ran, mouse → mice).
    """
    pattern = rf"\b({re.escape(word)}\w*)\b"
    if re.search(pattern, sentence, flags=re.IGNORECASE):
        return re.sub(pattern, r"<t> \1 </t>", sentence, count=1, flags=re.IGNORECASE)

    tokens = re.findall(r"\w+", sentence)
    if tokens:
        match = process.extractOne(
            word.lower(),
            [t.lower() for t in tokens],
            scorer=fuzz.QRatio,
            score_cutoff=fuzzy_threshold,
        )
        if match is not None:
            _, _, idx = match
            best = tokens[idx]
            return re.sub(
                rf"\b{re.escape(best)}\b", f"<t> {best} </t>", sentence, count=1
            )

    return sentence + f" <t> {word} </t>"




def format_example(example, tokenizer):
    answer = "true" if example["label"] else "false"
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
        {"role": "assistant", "content": answer},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma3-12b-it")
    args = parser.parse_args()

    dataset_train = load_data("mcl-wic", split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    partial_format = partial(format_example, tokenizer=tokenizer)
    cols = [
        "lemma",
        "pos",
        "word1",
        "word2",
        "sentence1",
        "sentence2",
        "label",
    ]
    dataset_train = dataset_train.map(partial_format, remove_columns=cols)
    print(dataset["train"][0]["text"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model)

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
        num_train_epochs=1,
        warmup_steps=100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        torch_compile=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=500,
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
        checkpoints = sorted(
            output_path.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        if checkpoints:
            last_checkpoint = str(checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)


if __name__ == "__main__":
    main()
