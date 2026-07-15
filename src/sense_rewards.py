"""Verifiable reward functions for the GRPO WiC task.

Split out of ``grpo_sense`` so the rewards can be imported, unit-tested and
eyeballed without the training stack: like ``sense_data``, this module imports
neither torch nor trl.

The reward has two halves:

* **Correctness** — ``reward_wic_accuracy``. The gold same/different label is
  known, so this is exact rather than a similarity estimate. It dominates
  (+/-1.0) everything else.
* **Shape** — ``reward_wic_format``, ``reward_wic_json`` and
  ``reward_think_length`` pay for the reasoning block and the JSON answer contract
  the prompt asks for, and punish the ways models degenerate out of it (a stubbed
  <think>, a prose answer, an unclosed reasoning block). Their combined ceiling
  stays well under the accuracy term, so being right always beats being tidy.
"""

import json
import os
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import sense_data as sd


def _content_word_count(text):
    return sum(1 for t in sd._tok(text) if t not in ENGLISH_STOP_WORDS)


def _answer_region(text):
    """The whole answer the model emits: all text after </think>.

    An unclosed <think> means the reasoning ran past the length budget and there is
    no answer region at all.
    """
    seg = text.split("</think>")[-1]
    if "<think>" in seg:
        return ""
    return seg.strip()


def _think_answer_format_reward(completions, extractor):
    """Reward a present <think> block (0.1) and an extractable answer (0.1)."""
    out = []
    for c in completions:
        r = 0.0
        if re.search(r"<think>.+?</think>", c, re.DOTALL):
            r += 0.1
        if extractor(c):
            r += 0.1
        out.append(r)
    return out


def _extract_think(text):
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


THINK_MIN_WORDS = 6      # below this many content words, reasoning is treated as a stub
THINK_MIN_PENALTY = -0.3


def reward_think_length(completions, **kwargs):
    """Punish degenerate reasoning: a missing, truncated, or near-empty <think> block.

    Catches the model collapsing the reasoning step — whether by dropping the
    block entirely (no penalty elsewhere, so this is the only force keeping it),
    truncating it past the length budget (no closing tag, so `_extract_think`
    returns ""), or stubbing it (e.g. "<think>ok</think>") to farm the format
    reward's presence bonus without doing any real reasoning.
    """
    out = []
    for c in completions:
        think = _extract_think(c)
        out.append(THINK_MIN_PENALTY if _content_word_count(think) < THINK_MIN_WORDS else 0.0)
    return out


# --------------------------------------------------------------------------- #
# WiC (word-in-context): verifiable same/different-sense classification
# --------------------------------------------------------------------------- #
WIC_CORRECT = 1.0
WIC_WRONG = -1.0


def reward_wic_accuracy(completions, **kwargs):
    """+1 for the right same/different verdict, -1 for the wrong one, 0 if absent.

    This is the verifiable signal: the gold label is known, so the reward is exact
    rather than a similarity estimate.
    """
    out = []
    for c, label in zip(completions, kwargs["label"]):
        pred = sd.extract_wic_label(c)
        if not pred:
            out.append(0.0)
        else:
            out.append(WIC_CORRECT if pred == label else WIC_WRONG)
    return out


def reward_wic_format(completions, **kwargs):
    """Reward a present <think> block (0.1) and an extractable verdict (0.1)."""
    return _think_answer_format_reward(completions, sd.extract_wic_label)


# Graded credit for the answer *shape* the prompt asks for: a single JSON object
# with exactly {"sense1", "sense2", "same_sense"} and a real boolean verdict.
# Graded rather than all-or-nothing so a group that is only partway there still has
# reward variance for GRPO to push on; the ceiling (0.3) stays well under the +/-1
# accuracy reward so being right always dominates being well-formatted.
WIC_JSON_PARSES = 0.1      # answer region holds a parseable JSON object
WIC_JSON_KEYS = 0.1        # exactly the three required keys, nothing extra or missing
WIC_JSON_BOOL = 0.1        # same_sense is a JSON boolean, not "true" / 1 / "same"
WIC_JSON_MALFORMED = -0.2  # said something after </think>, but no JSON object in it


def reward_wic_json(completions, **kwargs):
    """Score the answer region's JSON structure, in [WIC_JSON_MALFORMED, 0.3].

    ``reward_wic_format`` only pays for a verdict being *extractable*, and
    ``sd.extract_wic_label`` deliberately falls back to a bare same/different token —
    so on its own it lets the model collect format credit while never emitting JSON.
    This reward is the strict counterpart, and it is what makes the JSON contract
    (which the SFT warm-start teaches, see ``sd.wic_answer``) survive RL.

    A completion that answers in prose is penalised; a blank or unclosed <think>
    stays at 0, since ``reward_think_length`` already punishes that failure and
    double-charging it would drown out the accuracy signal.
    """
    out = []
    for c in completions:
        obj = sd.parse_wic_answer(c)
        if obj is None:
            out.append(WIC_JSON_MALFORMED if _answer_region(c) else 0.0)
            continue
        r = WIC_JSON_PARSES
        if set(obj) == sd.WIC_ANSWER_KEYS:
            r += WIC_JSON_KEYS
        if isinstance(obj.get("same_sense"), bool):
            r += WIC_JSON_BOOL
        out.append(r)
    return out


REWARDS = [reward_wic_accuracy, reward_wic_format, reward_wic_json, reward_think_length]

# Gold columns the reward fns (and the trace saver) read off the dataset.
KEEP_COLS = ["lemma", "label"]


# --------------------------------------------------------------------------- #
# Self-distillation: log successful reasoning traces for later SFT
# --------------------------------------------------------------------------- #
def make_trace_saver(path, threshold):
    """Pass-through reward fn that appends high-reward completions to a JSONL.

    Returns all-zero rewards, so it never perturbs training (a constant reward
    yields zero advantage after GRPO's group normalisation). "Successful" means the
    verifiable verdict was correct, so ``reward_wic_accuracy`` is what gates the
    write. It records the prompt + generated reasoning/verdict + score + gold
    columns; a downstream builder can turn the best generations into an SFT dataset.

    Writes one file per process rank to avoid interleaved appends under
    multi-GPU launches.
    """
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    out_path = Path(f"{path}.rank{rank}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save_successful_traces(completions, **kwargs):
        scores = reward_wic_accuracy(completions, **kwargs)
        prompts = kwargs.get("prompts")
        with out_path.open("a", encoding="utf-8") as f:
            for i, (c, s) in enumerate(zip(completions, scores)):
                if s < threshold:
                    continue
                rec = {"score": round(float(s), 4), "completion": c}
                if prompts is not None:
                    rec["prompt"] = prompts[i]
                for col in KEEP_COLS:
                    rec[col] = kwargs[col][i]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return [0.0] * len(completions)

    # GRPOTrainer logs/derives metric names from the fn name.
    save_successful_traces.__name__ = "save_successful_traces"
    return save_successful_traces
