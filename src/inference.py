import argparse

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sft_finetune
import grpo_finetune
from utils import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--lora", type=str, default="qwen-wic-grpo/checkpoint-1000")
    parser.add_argument("--dataset", type=str, default="mcl-wic")
    parser.add_argument("--sft", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Left-pad so all sequences in a batch align on the right (generation side)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.lora)
    model.eval()
    dataset_test = load_data(args.dataset, split="test")

    # Evaluate accuracy on test set
    # Dataset.iter(batch_size=N) yields dicts of lists without converting to a plain list
    correct = 0
    for batch in tqdm(dataset_test.iter(batch_size=args.batch_size), total=(len(dataset_test) + args.batch_size - 1) // args.batch_size):
        batch_size = len(batch["label"])

        if args.sft:
            texts = [sft_finetune.format_prompt({k: batch[k][i] for k in batch}, tokenizer)["text"] for i in range(batch_size)]
        else:
            texts = [grpo_finetune.format_prompt({k: batch[k][i] for k in batch}, tokenizer)["prompt"] for i in range(batch_size)]

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
                temperature=0.7,  # low temp for eval — greedy-ish
                top_k=20,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i in range(batch_size):
            decoded = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            pred = grpo_finetune._extract_answer(decoded)
            label = "true" if batch["label"][i] else "false"
            correct += (pred or "").lower() == label.lower()

    print(f"Accuracy: {correct / len(dataset_test):.4f}")


if __name__ == "__main__":
    main()
