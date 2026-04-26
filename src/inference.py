import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sft_finetune
import grpo_finetune
from utils import load_data


def _format_sft_inference_prompt(example, tokenizer):
    """SFT-style prompt for inference: no assistant answer, add generation prompt."""
    s1 = sft_finetune.mark_target(example["sentence1"], example["word1"])
    s2 = sft_finetune.mark_target(example["sentence2"], example["word2"])
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
            "content": (
                f"Word: {example['lemma']} ({example['pos']})\n"
                f"Sentence 1: {s1}\n"
                f"Sentence 2: {s2}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _f1(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora", type=str, default="qwen-wic-grpo/checkpoint-1000")
    parser.add_argument("--dataset", type=str, default="mcl-wic")
    parser.add_argument("--sft", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output", type=str, default="predictions.json")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Left-pad so all sequences in a batch align on the right (generation side)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.lora)
    model.eval()

    dataset_test = load_data(args.dataset, split=args.split)

    records = []
    y_true: list[int] = []
    y_pred: list[int] = []

    total_batches = (len(dataset_test) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(dataset_test.iter(batch_size=args.batch_size), total=total_batches):
        batch_size = len(batch["label"])

        if args.sft:
            texts = [
                _format_sft_inference_prompt({k: batch[k][i] for k in batch}, tokenizer)
                for i in range(batch_size)
            ]
        else:
            texts = [
                grpo_finetune.format_prompt({k: batch[k][i] for k in batch}, tokenizer)["prompt"]
                for i in range(batch_size)
            ]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i in range(batch_size):
            decoded = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            pred = grpo_finetune._extract_answer(decoded)
            label_str = "true" if batch["label"][i] else "false"
            pred_str = (pred or "").lower()

            # Treat unparseable outputs as "false" for metric purposes
            pred_bin = 1 if pred_str == "true" else 0
            y_true.append(1 if batch["label"][i] else 0)
            y_pred.append(pred_bin)

            records.append({
                "sentence1": batch["sentence1"][i],
                "sentence2": batch["sentence2"][i],
                "lemma": batch["lemma"][i],
                "pos": batch["pos"][i],
                "label": label_str,
                "prediction": pred_str if pred_str in ("true", "false") else None,
                "raw_output": decoded,
            })

    # Save predictions
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(records)} predictions → {output_path}")

    # Metrics
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    accuracy = correct / len(y_true)
    f1 = _f1(y_true, y_pred)
    unparseable = sum(1 for r in records if r["prediction"] is None)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"F1:          {f1:.4f}")
    if unparseable:
        print(f"Unparseable: {unparseable}/{len(records)} outputs")


if __name__ == "__main__":
    main()
