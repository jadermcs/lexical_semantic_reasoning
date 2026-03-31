import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils import load_data

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
# --- 1. Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token


def format_message(example):
    answer = "True" if example["label"] else "False"
    messages = [
        {
            "role": "system",
            "content": "You are a linguistic expert. Determine if the target word is used in the same sense in both sentences. Answer only 'True' or 'False'.\n",
        },
        {
            "role": "user",
            "content": f"Word: {example['lemma']}\nSentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nSame sense?",
        },
        {"role": "assistant", "content": answer},
    ]
    return {
        "text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    }


# --- 3. Load and format dataset ---
dataset = DatasetDict(
    {split: load_data("mcl-wic", split=split) for split in ("train", "dev")}
)
dataset_test = load_data("mcl-wic", split="test")
dataset = dataset.map(format_message)  # apply prompt template from above
print(dataset["train"][0])
print(dataset_test[0])


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="cuda",
    trust_remote_code=True,
)

# --- 2. LoRA config ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,  # rank
    lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# --- 5. Trainer ---
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # use processing_class, not tokenizer= (deprecated)
    train_dataset=dataset["train"],
    eval_dataset=dataset["dev"],
    args=SFTConfig(
        output_dir="./qwen-wic-lora",
        dataset_text_field="text",
        per_device_train_batch_size=16,
        num_train_epochs=5,
        warmup_steps=100,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name="qwen-wic-sft",
    ),
)
trainer.train()
trainer.save_model("./qwen-wic-lora-final")

model.eval()


def predict(word, sentence1, sentence2):
    prompt = (
        f"<|im_start|>system\nYou are a linguistic expert. Determine if the target word is used in the same sense in both sentences. Answer only 'True' or 'False'.\n<|im_end|>\n"
        f"<|im_start|>user\nWord: {word}\nSentence 1: {sentence1}\nSentence 2: {sentence2}\nSame sense?\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return decoded.strip()


# Evaluate accuracy on test set
correct = 0
for example in dataset_test:
    messages = [
        {
            "role": "system",
            "content": "You are a linguistic expert. Determine if the target word is used in the same sense in both sentences. Answer only 'True' or 'False'.\n",
        },
        {
            "role": "user",
            "content": f"Word: {example['lemma']}\nSentence 1: {example['sentence1']}\nSentence 2: {example['sentence2']}\nSame sense?",
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    pred = decoded.strip()
    label = "True" if example["label"] else "False"
    correct += pred.lower() == label.lower()
print(f"Accuracy: {correct / len(dataset_test):.4f}")
