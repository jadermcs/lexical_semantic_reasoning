import argparse
import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from utils import load_data



def format_message(example, tokenizer):
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
    dataset = dataset.map(format_prompt, remove_columns=["lemma", "sentence1", "sentence2"])
    print(dataset["train"][0])

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        trust_remote_code=True,
    )

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


    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # use processing_class, not tokenizer= (deprecated)
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        args=SFTConfig(
            output_dir="./qwen-wic-lora",
            dataset_text_field="text",
            per_device_train_batch_size=16,
            num_train_epochs=10,
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
