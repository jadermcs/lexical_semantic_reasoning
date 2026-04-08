"""
GRPO fine-tuning for Word-in-Context (WiC) with R1-style chain-of-thought reasoning.

The model is prompted to reason inside <think> tags before giving a final
<answer>True/False</answer>. GRPO uses relative rewards within a group of
completions per prompt, so the reward signal can be sparse without issue.
"""

import json
import re
import argparse
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
    "in the same sense in both sentences.\n\n"
    "First, reason step by step inside <think> tags:\n"
    "  1. Identify the specific meaning/sense of the word in Sentence 1.\n"
    "  2. Identify the specific meaning/sense of the word in Sentence 2.\n"
    "  3. Compare the two senses.\n\n"
    "Then provide your final answer inside <answer> tags as exactly 'True' or 'False'.\n"
    "Format: <think>your reasoning here</think><answer>True or False</answer>"
)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------


def format_prompt(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Word: {example['lemma']}\n"
                f"Sentence 1: {example['sentence1']}\n"
                f"Sentence 2: {example['sentence2']}\n"
                "Same sense?"
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
    """Return 'True' or 'False' from the completion, or None if not found."""
    # prefer explicit <answer> tags
    m = re.search(r"<answer>\s*(True|False)\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    # fallback: last bare True/False token
    tokens = re.findall(r"\b(True|False)\b", text, re.IGNORECASE)
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
        expected = "True" if lbl else "False"
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
        if re.search(r"<answer>(True|False)</answer>", completion, re.IGNORECASE):
            r += 0.1
        rewards.append(r)
    return rewards


def reward_reasoning_quality(completions: list[str], **kwargs) -> list[float]:
    """
    Soft reward for reasoning that is substantive but concise.
    Rewards 30–200-word think blocks; penalises empty or runaway reasoning.
    Keeps the model from collapsing to zero-shot answers or rambling.
    """
    rewards = []
    for completion in completions:
        m = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        if not m:
            rewards.append(-0.1)
            continue
        n_words = len(m.group(1).split())
        if n_words < 10:
            rewards.append(-0.05)  # trivially short
        elif 30 <= n_words <= 200:
            rewards.append(0.1)  # good range
        else:
            rewards.append(0.0)  # too long / borderline
    return rewards

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", type=str, default="mcl-wic")
    args = parser.parse_args()
    dataset = DatasetDict(
        {split: load_data(args.dataset, split=split) for split in ("train", "dev")}
    )
    dataset = dataset.map(format_prompt, remove_columns=["lemma", "sentence1", "sentence2"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        trust_remote_code=True,
        dtype=torch.bfloat16,  # bfloat16 is preferred for RL stability
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
        num_generations=8,  # G: group size for relative reward computation
        max_completion_length=512,  # enough room for <think> + <answer>
        temperature=0.9,  # high temp encourages diverse completions in group
        # -- training --
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 4*4 = 16 prompts × 8 completions
        num_train_epochs=10,
        warmup_steps=50,
        learning_rate=5e-6,  # lower LR than SFT — RL is less stable
        lr_scheduler_type="cosine",
        bf16=True,
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
        use_liger_kernel=True,
    )


    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reward_correctness,  # main signal  (scale ≈ ±1)
            reward_format,  # structure    (scale  0–0.3)
            reward_reasoning_quality,  # length proxy (scale -0.1–0.1)
        ],
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        peft_config=lora_config,
    )

    trainer.train()
    success_logger.flush()
    print(
        f"Saved {len(success_logger._seen)} successful completions → {SUCCESSFUL_DATA_PATH}"
    )

if __name__ == "__main__":
    main()
