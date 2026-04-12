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
    args = parser.parse_args()
    # Load the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model = PeftModel.from_pretrained(base_model, args.lora)
    model.eval()
    dataset_test = load_data(args.dataset, split="test")

    # Evaluate accuracy on test set
    correct = 0
    for example in tqdm(dataset_test):
        if args.sft:
            text = stf_finetune.format_prompt(example, tokenizer)["text"]
        else:
            text = grpo_finetune.format_prompt(example, tokenizer)["prompt"]
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,  # low temp for eval — greedy-ish
                top_k=20,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        pred = grpo_finetune._extract_answer(decoded)
        label = "true" if example["label"] else "false"
        correct += (pred or "").lower() == label.lower()
    print(f"Accuracy: {correct / len(dataset_test):.4f}")


if __name__ == "__main__":
    main()
