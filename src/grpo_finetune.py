"""
GRPO fine-tuning for Word-in-Context (WiC) with R1-style chain-of-thought reasoning.

The model is prompted to reason inside <think> tags before giving a final
<answer>true/false</answer>. GRPO uses relative rewards within a group of
completions per prompt, so the reward signal can be sparse without issue.
"""

import argparse
import json
import re
from functools import partial
from pathlib import Path
from threading import Lock

import torch
from datasets import DatasetDict
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from utils import load_data

SYSTEM_PROMPT = (
    "You are a linguistic expert specializing in word sense disambiguation. "
    "Given a target word and two sentences, determine whether the word is used "
    "in the same sense in both sentences. "
    "The target word is marked with <t> tags in each sentence.\n\n"
    "First, reason step by step inside <think> tags:\n"
    "  1. Gloss for use 1: <short dictionary-style definition>\n"
    "  2. Gloss for use 2: <short dictionary-style definition>\n"
    "  3. Do these glosses describe the same concept? <yes/no>\n"
    "  4. Final answer: <true/false>\n\n"
    "Then provide your final answer inside <answer> tags as exactly 'true' or 'false'.\n"
    "Format: <think>your reasoning here</think><answer>true or false</answer>"
)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def mark_target(sentence: str, word: str) -> str:
    """Wrap the first occurrence of *word* (case-insensitive) with <t> tags."""
    return re.sub(
        rf"\b({re.escape(word)})\b",
        r"<t>\1</t>",
        sentence,
        count=1,
        flags=re.IGNORECASE,
    )

def format_prompt(example, tokenizer):
    s1 = mark_target(example["sentence1"], example["word1"])
    s2 = mark_target(example["sentence2"], example["word2"])
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
    ]
    return {
        "prompt": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ),
        # keep label so reward functions can access it via kwargs
        "label": example["label"],
    }


SUCCESSFUL_DATA_PATH = Path("./grpo_successful_completions.jsonl")


class SuccessLogger:
    """
    Collects completions where the model answered correctly and flushes them
    to a JSONL file.  Each record is a full SFT-ready chat (prompt + reasoning
    + answer) so the file can be directly used for warm-start fine-tuning.

    Thread-safe: reward functions may be called from multiple workers.
    """

    def __init__(self, path: Path, flush_every: int = 50):
        self.path = path
        self.flush_every = flush_every
        self._buffer: list[dict] = []
        self._seen: set[str] = set()  # deduplicate by prompt
        self._lock = Lock()
        # Load already-seen prompts so we don't re-save across runs.
        if path.exists():
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    self._seen.add(rec["prompt"])

    def log(self, prompt: str, completion: str, label: int) -> None:
        with self._lock:
            if prompt in self._seen:
                return
            self._seen.add(prompt)
            self._buffer.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    # SFT-ready: full text = prompt + completion
                    "text": prompt + completion,
                    "label": label,
                }
            )
            if len(self._buffer) >= self.flush_every:
                self._flush()

    def _flush(self) -> None:
        """Append buffered records to disk (called with lock held)."""
        if not self._buffer:
            return
        with open(self.path, "a") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._buffer.clear()

    def flush(self) -> None:
        """Public flush — call at end of training to drain any remainder."""
        with self._lock:
            self._flush()


success_logger = SuccessLogger(SUCCESSFUL_DATA_PATH, flush_every=50)


def _extract_answer(text: str) -> str | None:
    """Return 'true' or 'false' from the completion, or None if not found."""
    # prefer explicit <answer> tags
    m = re.search(r"<answer>\s*(true|false)\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # fallback: last bare true/false token
    tokens = re.findall(r"\b(true|false)\b", text, re.IGNORECASE)
    return tokens[-1].capitalize() if tokens else None


def reward_correctness(completions: list[str], **kwargs) -> list[float]:
    """
    Primary reward signal.
      +1.0  correct answer  → also logged to JSONL for future SFT warm-start
      -1.0  wrong answer
      -0.5  no answer found (model didn't follow format at all)
    """
    if not any(k == "prompts" for k in kwargs):
        print("WARNING: 'prompts' not in reward kwargs — keys:", list(kwargs.keys()))
    labels = kwargs["label"]
    prompts = kwargs.get("prompts", [None] * len(completions))
    rewards = []
    for prompt, completion, lbl in zip(prompts, completions, labels):
        expected = "true" if lbl else "false"
        pred = _extract_answer(completion)
        if pred is None:
            rewards.append(-0.5)
        elif pred == expected:
            rewards.append(1.0)
            # Save correct, well-formatted completions for iterative SFT.
            if re.search(r"<think>.+?</think>", completion, re.DOTALL):
                success_logger.log(prompt or "", completion, int(lbl))
        else:
            rewards.append(-1.0)
    return rewards


def reward_format(completions: list[str], **kwargs) -> list[float]:
    """
    Structural reward: encourage <think>…</think><answer>…</answer> format.
    Kept small so it doesn't dominate over correctness.
      +0.2  proper think block present
      +0.1  proper answer tag present
    """
    rewards = []
    for completion in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", completion, re.DOTALL):
            r += 0.2
        if re.search(r"<answer>(true|false)</answer>", completion, re.IGNORECASE):
            r += 0.1
        rewards.append(r)
    return rewards


def reward_reasoning_quality(completions: list[str], **kwargs) -> list[float]:
    """
    Soft reward for reasoning that follows the gloss-based structure:
      1. Gloss for use 1: ...
      2. Gloss for use 2: ...
      3. Do these glosses describe the same concept? yes/no
      4. Final answer: true/false

    +0.1  for each gloss line present (up to +0.2)
    +0.1  for the same-concept question with a yes/no answer
    +0.1  for the final answer line
    -0.1  if <think> block is absent entirely
    -0.05 if <think> block is trivially short (< 10 words)
    """
    rewards = []
    for completion in completions:
        m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if not m:
            rewards.append(-0.1)
            continue
        think = m.group(1)
        if len(think.split()) < 10:
            rewards.append(-0.05)
            continue
        r = 0.0
        if re.search(r"gloss for use 1\s*:", think, re.IGNORECASE):
            r += 0.1
        if re.search(r"gloss for use 2\s*:", think, re.IGNORECASE):
            r += 0.1
        if re.search(r"same concept\b.{0,30}\b(yes|no)\b", think, re.IGNORECASE):
            r += 0.1
        if re.search(r"final answer\s*:\s*(true|false)", think, re.IGNORECASE):
            r += 0.1
        rewards.append(r)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="mcl-wic")
    args = parser.parse_args()
    dataset = DatasetDict(
        {split: load_data(args.dataset, split=split) for split in ("train", "dev")}
    )
    dataset["dev"] = dataset["dev"].shuffle(seed=42).select(range(200))

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    partial_format = partial(format_prompt, tokenizer=tokenizer)
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

    training_args = GRPOConfig(
        output_dir="./qwen-wic-grpo",
        # -- generation --
        num_generations=8,
        max_completion_length=512,
        temperature=0.9,
        weight_decay = 0.001,
        # -- training --
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        warmup_steps=50,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        bf16=True,
        torch_compile=True,
        # -- eval & save --
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # -- logging --
        logging_steps=10,
        report_to="wandb",
        run_name="qwen-wic-grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correctness,
            reward_format,
            reward_reasoning_quality,
        ],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        peft_config=lora_config,
    )

    # Resume from the latest checkpoint if one exists (e.g. after a SLURM time-limit requeue)

    trainer.train(resume_from_checkpoint=True)
    success_logger.flush()
    print(
        f"Saved {len(success_logger._seen)} successful completions → {SUCCESSFUL_DATA_PATH}"
    )


if __name__ == "__main__":
    main()
