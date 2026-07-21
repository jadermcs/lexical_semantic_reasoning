"""Evaluate a WiC checkpoint on a held-out MCL-WiC split.

Generates one verdict per test pair (vLLM, continuous batching; Qwen3
thinking-mode sampling — temperature 0.6, top-p 0.95, top-k 20, min-p 0) and
reports accuracy plus same/different precision/recall/F1. Completions with no
extractable verdict are counted as ``empty`` and left out of the P/R/F1 (they
can't be scored as either class).

With ``--force-json`` the answer region is constrained to schema-valid JSON
(two-phase decode: free reasoning up to ``</think>``, then an xgrammar-guided
continuation via vLLM structured outputs), so every completion parses and
``empty`` drops to 0 — including pairs whose reasoning overran the budget,
which get force-closed. Prefix caching means the phase-2 pass reuses the
phase-1 KV cache instead of re-prefilling prompt + reasoning.

Predictions are saved in the ``call_api.py`` teacher schema (one sample
per pair), so the output file can be fed straight to ``prepare_data.py --data``
(which builds the SFT dataset ``sft_sense.py`` then trains on).

Examples
--------
  # a trained checkpoint
  uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic
  # zero-shot base-model baseline
  uv run python src/eval_sense.py --model Qwen/Qwen3-0.6B
  # guarantee a parseable JSON verdict on every pair
  uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic --force-json
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import sense_data as sd


def build_prompt(rec, tokenizer):
    msgs = sd.wic_messages(rec, with_target=False)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# Shape of the answer object (mirrors sense_data.wic_answer / WIC_ANSWER_KEYS).
WIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sense1": {"type": "string"},
        "sense2": {"type": "string"},
        "same_sense": {"type": "boolean"},
    },
    "required": ["sense1", "sense2", "same_sense"],
}

# Qwen3 thinking-mode sampling settings (greedy decoding is discouraged for
# reasoning models — it degenerates into repetition).
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)


def generate_all(llm, texts, force_json=False):
    """Sampled completions for all prompts; vLLM schedules the batch internally.

    With ``force_json``, decoding runs in two phases: free reasoning stopped at
    ``</think>`` (force-closed if the token budget runs out first), then a
    continuation whose tokens are constrained to schema-valid JSON (xgrammar,
    via vLLM structured outputs), so every completion ends in a parseable
    verdict. Prefix caching makes phase 2 a near-pure decode of the answer.
    """
    if not force_json:
        sp = SamplingParams(max_tokens=1024, **SAMPLING)
        return [out.outputs[0].text for out in llm.generate(texts, sp)]

    # Phase 1: free-form reasoning, halted at the close of the think block.
    sp1 = SamplingParams(
        max_tokens=1024, **SAMPLING,
        stop=["</think>"], include_stop_str_in_output=True,
    )
    thinks = []
    for out in llm.generate(texts, sp1):
        think = out.outputs[0].text
        if "</think>" not in think:  # budget ran out mid-reasoning: force-close
            think += "\n</think>"
        thinks.append(think.rstrip() + "\n")

    # Phase 2: constrained continuation — only tokens forming schema-valid JSON.
    sp2 = SamplingParams(
        max_tokens=512, **SAMPLING,
        structured_outputs=StructuredOutputsParams(json=WIC_JSON_SCHEMA),
    )
    outs2 = llm.generate([p + t for p, t in zip(texts, thinks)], sp2)
    return [think + out.outputs[0].text for think, out in zip(thinks, outs2)]


def wic_metrics(preds, golds):
    """Accuracy + same/different P/R/F1 over parsed verdicts; None preds are unscored."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    scored = [(p, g) for p, g in zip(preds, golds) if p is not None]
    n = len(scored)
    y_pred = [p for p, _ in scored]
    y_true = [g for _, g in scored]
    if n:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=True, zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
    else:
        prec = rec = f1 = acc = 0.0
    return {
        "n": len(preds), "n_scored": n, "empty": len(preds) - n,
        "accuracy": float(acc),
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--force-json", action="store_true",
                    help="constrain the answer region to schema-valid JSON (xgrammar)")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,  # phase 2 of --force-json reuses phase-1 KV
    )
    tokenizer = llm.get_tokenizer()

    data = sd.load_mclwic(args.split)
    if args.max_samples:
        data = data[: args.max_samples]

    texts = [build_prompt(r, tokenizer) for r in data]
    decoded_all = generate_all(llm, texts, force_json=args.force_json)

    hyps, refs, records = [], [], []
    for rec, decoded in zip(data, decoded_all):
        hyp, gold = sd.extract_wic_label(decoded), rec["label"]
        hyps.append(hyp)
        refs.append(gold)
        # Teacher-predictions schema (call_api.py), single sample —
        # the output file feeds prepare_data.py --data via load_teacher_traces.
        think, closed, _ = decoded.partition("</think>")
        answer = sd.parse_wic_answer(decoded)
        records.append({
            "lemma": rec["lemma"], "pos": rec["pos"],
            "sentence1": rec["sentence1"], "sentence2": rec["sentence2"],
            "label": gold, "prediction": hyp,
            "answers": [json.dumps(answer, ensure_ascii=False)] if answer is not None else [],
            "reasonings": [think.replace("<think>", "").strip()] if closed else [],
        })

    metrics = wic_metrics(hyps, refs)
    print(f"\n[wic] n={metrics['n']}  acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  empty={metrics['empty']}")

    print("\nExamples (first 10):")
    print(f"{'lemma':<18}  {'gold':<10}  {'prediction':<10}")
    for rec in records[:10]:
        pred = "—" if rec["prediction"] is None else str(rec["prediction"])
        print(f"{rec['lemma']:<18}  {str(rec['label']):<10}  {pred:<10}")

    # Bare list in the call_api.py teacher schema — directly consumable by
    # prepare_data.py --data. Metrics are printed above, not saved.
    out_path = Path(args.output or f"predictions_sense_wic_{args.split}.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
