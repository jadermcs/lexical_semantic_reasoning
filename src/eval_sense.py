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


def build_prompt(rec, tokenizer, mode):
    msgs = (sd.direct_messages if mode == "direct" else sd.triplet_messages)(
        rec, with_target=False
    )
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--mode", choices=["direct", "triplet"], required=True)
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
                **inputs, max_new_tokens=512, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        for rec, out in zip(batch, outputs):
            decoded = tokenizer.decode(out[input_len:], skip_special_tokens=True)
            hyp = extract(decoded)
            hyps.append(hyp)
            refs.append(rec[gold_key])
            records.append({"lemma": rec["lemma"], "gold": rec[gold_key],
                            "prediction": hyp, "raw_output": decoded})

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

    out_path = Path(args.output or f"predictions_sense_{args.mode}.json")
    out_path.write_text(json.dumps(
        {"mode": args.mode, "model": args.model, "bleu": bleu, "mean_sim": mean_sim,
         "bertscore_f1": bertscore, "n": len(hyps), "predictions": records},
        ensure_ascii=False, indent=2))
    print(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
