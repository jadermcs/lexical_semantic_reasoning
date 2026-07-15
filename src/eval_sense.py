"""Evaluate a WiC checkpoint on a held-out MCL-WiC split.

Greedily generates a verdict per test pair and reports accuracy plus same/different
precision/recall/F1. Completions with no extractable verdict are counted as
``empty`` and left out of the P/R/F1 (they can't be scored as either class).

Examples
--------
  # a trained checkpoint
  uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic
  # zero-shot base-model baseline
  uv run python src/eval_sense.py --model Qwen/Qwen3-0.6B
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sense_data as sd


def build_prompt(rec, tokenizer):
    msgs = sd.wic_messages(rec, with_target=False)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def wic_metrics(preds, golds):
    """Accuracy + same/different P/R/F1 over parsed verdicts; '' preds are unscored."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    scored = [(p, g) for p, g in zip(preds, golds) if p]
    n = len(scored)
    y_pred = [p for p, _ in scored]
    y_true = [g for _, g in scored]
    if n:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label="same", zero_division=0,
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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16)
    model.eval()

    data = sd.load_mclwic(args.split)
    if args.max_samples:
        data = data[: args.max_samples]

    hyps, refs, records = [], [], []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i : i + args.batch_size]
        texts = [build_prompt(r, tokenizer) for r in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for rec, out in zip(batch, outputs):
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            hyp, gold = sd.extract_wic_label(decoded), rec["label"]
            hyps.append(hyp)
            refs.append(gold)
            records.append({"lemma": rec["lemma"], "gold": gold,
                            "prediction": hyp, "raw_output": decoded})

    metrics = wic_metrics(hyps, refs)
    print(f"\n[wic] n={metrics['n']}  acc={metrics['accuracy']:.3f}  "
          f"f1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
          f"R={metrics['recall']:.3f}  empty={metrics['empty']}")

    print("\nExamples (first 10):")
    print(f"{'lemma':<18}  {'gold':<10}  {'prediction':<10}")
    for rec in records[:10]:
        print(f"{rec['lemma']:<18}  {rec['gold']:<10}  {(rec['prediction'] or '—'):<10}")

    out_path = Path(args.output or f"predictions_sense_wic_{args.split}.json")
    out_path.write_text(json.dumps(
        {"model": args.model, "split": args.split, **metrics, "predictions": records},
        ensure_ascii=False, indent=2))
    print(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
