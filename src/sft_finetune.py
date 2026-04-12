import argparse
import re
from pathlib import Path
from functools import partial

import numpy as np
import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils import load_data


def preprocess_logits_for_metrics(logits, labels):
    """Reduce logits to argmax token ids before accumulation to save memory."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred, tokenizer):
    predictions, labels = eval_pred
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    gold_token_lists = []
    pred_token_lists = []

    for pred_seq, label_seq in zip(predictions, labels):
        mask = label_seq != -100
        if not mask.any():
            gold_token_lists.append([])
            pred_token_lists.append([])
            continue

        positions = np.where(mask)[0]
        gold_token_lists.append(label_seq[positions].tolist())
        # Shift left by 1: logits[t-1] predicts token[t]
        pred_token_lists.append(pred_seq[positions - 1].tolist())

    golds = tokenizer.batch_decode(gold_token_lists, skip_special_tokens=True)
    preds = tokenizer.batch_decode(pred_token_lists, skip_special_tokens=True)

    correct, total = 0, 0
    for gold, pred in zip(golds, preds):
        gold = gold.strip().lower()
        pred = pred.strip().lower()

        if "true" in gold:
            gold_label = "true"
        elif "false" in gold:
            gold_label = "false"
        else:
            continue

        pred_label = "true" if "true" in pred else "false" if "false" in pred else None
        total += 1
        if pred_label == gold_label:
            correct += 1

    return {"accuracy": correct / total if total > 0 else 0.0}


def mark_target(sentence: str, word: str) -> str:
    """Wrap the first occurrence of *word* (case-insensitive) with <t> tags."""
    return re.sub(
        rf"\b({re.escape(word)})\b",
        r"<t>\1</t>",
        sentence,
        count=1,
    )


def format_prompt(example, tokenizer):
    answer = "true" if example["label"] else "false"
    s1 = mark_target(example["sentence1"], example["word1"])
    s2 = mark_target(example["sentence2"], example["word2"])
    messages = [
        {
            "role": "system",
            "content": (
                "You are a linguistic expert. Determine if the target word is used "
                "in the same sense in both sentences. The target word is marked with "
                "<t> tags in each sentence. Answer only 'true' or 'false'."
            ),
        },
        {
            "role": "user",
            "content": f"Word: {example['lemma']} ({example['pos']})\nSentence 1: {s1}\nSentence 2: {s2}",
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
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="mcl-wic")
    args = parser.parse_args()
    dataset = DatasetDict(
        {split: load_data(args.dataset, split=split) for split in ("train", "dev")}
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    partial_format = partial(format_prompt, tokenizer=tokenizer)
    partial_metrics = partial(compute_metrics, tokenizer=tokenizer)
    dataset = dataset.map(
        partial_format, remove_columns=["lemma", "word1", "word2", "pos", "sentence1", "sentence2"]
    )
    print(dataset["train"][0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        trust_remote_code=True,
        dtype=torch.bfloat16,
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
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=10,
        warmup_steps=100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        bf16=True,
        torch_compile=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        report_to="wandb",
        run_name="qwen-wic-sft",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # use processing_class, not tokenizer= (deprecated)
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=partial_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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
    trainer.train(resume_from_checkpoint=True)


if __name__ == "__main__":
    main()
