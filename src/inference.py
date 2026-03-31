import torch
from tqdm import tqdm
from pprint import pprint
from peft import PeftModel
from utils import load_data
import grpo_finetune
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

MODEL_NAME = "Qwen/Qwen3-1.7B"
# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "qwen-wic-grpo")
model.eval()
dataset_test = load_data("mcl-wic", split="test")

# Evaluate accuracy on test set
correct = 0
for example in tqdm(dataset_test):
    text = grpo_finetune.format_prompt(example)["prompt"]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,  # low temp for eval — greedy-ish
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    pred = grpo_finetune._extract_answer(decoded)
    label = "True" if example["label"] else "False"
    correct += (pred or "").lower() == label.lower()
print(f"Accuracy: {correct / len(dataset_test):.4f}")
