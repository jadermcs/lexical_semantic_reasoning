"""Evaluate a sense-modeling checkpoint on the held-out WordNet test split.

Generates a gloss per test usage and reports corpus BLEU against the WordNet
gold definition. For ``--mode triplet`` the single shared (anchor/positive) gloss
is scored, giving a per-usage quantity comparable to ``--mode direct``.

Examples
--------
  # config 1 (SFT direct)
  uv run python src/eval_sense.py --mode direct  --model ./qwen-sense-sft-direct
  # config 4 (GRPO triplet)
  uv run python src/eval_sense.py --mode triplet --model ./qwen-sense-grpo-triplet
  # zero-shot base-model baseline
  uv run python src/eval_sense.py --mode direct
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sense_data as sd


_MESSAGES = {
    "direct": sd.direct_messages,
    "triplet": sd.triplet_messages,
    "wic": sd.wic_messages,
}


def build_prompt(rec, tokenizer, mode):
    msgs = _MESSAGES[mode](rec, with_target=False)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _wic_metrics(preds, golds):
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


def print_examples(records, n=10):
    """Print a table of the first ``n`` predictions next to their gold gloss."""
    def trunc(s, w):
        s = " ".join(s.split())
        return s if len(s) <= w else s[: w - 1] + "…"

    lemma_w, col_w = 18, 55
    header = f"{'lemma':<{lemma_w}}  {'gold':<{col_w}}  {'prediction':<{col_w}}"
    print(f"\nExamples (first {min(n, len(records))}):")
    print(header)
    print("-" * len(header))
    for rec in records[:n]:
        print(f"{trunc(rec['lemma'], lemma_w):<{lemma_w}}  "
              f"{trunc(rec['gold'], col_w):<{col_w}}  "
              f"{trunc(rec['prediction'], col_w):<{col_w}}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--mode", choices=["direct", "triplet", "wic"], required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--output", default=None)
    ap.add_argument("--bertscore", action="store_true",
                    help="also report BERTScore F1 (downloads roberta-large; server-only)")
    ap.add_argument("--bertscore-model", default=None,
                    help="override the BERTScore model (default: roberta-large via lang=en)")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16)
    model.eval()

    if args.mode == "wic":
        data = sd.load_mclwic(args.split)
    else:
        data = sd.load_split(args.mode, args.split)
    if args.max_samples:
        data = data[: args.max_samples]
    gold_key = "gloss" if args.mode == "direct" else "gloss_same"
    extract = sd.extract_direct_gloss if args.mode == "direct" else sd.extract_shared_gloss

    hyps, refs, records = [], [], []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i : i + args.batch_size]
        texts = [build_prompt(r, tokenizer, args.mode) for r in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for rec, out in zip(batch, outputs):
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            if args.mode == "wic":
                hyp = sd.extract_wic_label(decoded)
                gold = rec["label"]
            else:
                hyp = extract(decoded)
                gold = rec[gold_key]
            hyps.append(hyp)
            refs.append(gold)
            records.append({"lemma": rec["lemma"], "gold": gold,
                            "prediction": hyp, "raw_output": decoded})

    if args.mode == "wic":
        metrics = _wic_metrics(hyps, refs)
        print(f"\n[wic] n={metrics['n']}  acc={metrics['accuracy']:.3f}  "
              f"f1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
              f"R={metrics['recall']:.3f}  empty={metrics['empty']}")
        print("\nExamples (first 10):")
        print(f"{'lemma':<18}  {'gold':<10}  {'prediction':<10}")
        for rec in records[:10]:
            print(f"{rec['lemma']:<18}  {rec['gold']:<10}  {(rec['prediction'] or '—'):<10}")
        out_path = Path(args.output or f"predictions_sense_wic_{args.split}.json")
        out_path.write_text(json.dumps(
            {"mode": "wic", "model": args.model, "split": args.split,
             **metrics, "predictions": records}, ensure_ascii=False, indent=2))
        print(f"Saved predictions → {out_path}")
        return

    bleu = sd.corpus_bleu(hyps, refs)
    mean_sim = sum(sd.gloss_similarity(h, r) for h, r in zip(hyps, refs)) / len(hyps)
    empty = sum(1 for h in hyps if not h.strip())

    bertscore = None
    if args.bertscore:
        from bert_score import score  # lazy import; needs the `bert-score` package
        kwargs = {"lang": "en", "rescale_with_baseline": True}
        if args.bertscore_model:  # baseline files only ship for the default models
            kwargs.update(model_type=args.bertscore_model, rescale_with_baseline=False)
        _, _, F1 = score(hyps, refs, **kwargs)
        f1_list = F1.tolist()
        bertscore = sum(f1_list) / len(f1_list)
        for rec, f in zip(records, f1_list):
            rec["bertscore_f1"] = f

    msg = f"\n[{args.mode}] n={len(hyps)}  BLEU={bleu:.2f}  mean_sim={mean_sim:.3f}"
    if bertscore is not None:
        msg += f"  BERTScore_F1={bertscore:.3f}"
    print(f"{msg}  empty={empty}")

    print_examples(records, n=10)

    out_path = Path(args.output or f"predictions_sense_{args.mode}.json")
    out_path.write_text(json.dumps(
        {"mode": args.mode, "model": args.model, "bleu": bleu, "mean_sim": mean_sim,
         "bertscore_f1": bertscore, "n": len(hyps), "predictions": records},
        ensure_ascii=False, indent=2))
    print(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
