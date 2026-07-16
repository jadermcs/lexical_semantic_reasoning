"""Evaluate a WiC checkpoint on a held-out MCL-WiC split.

Greedily generates a verdict per test pair and reports accuracy plus same/different
precision/recall/F1. Completions with no extractable verdict are counted as
``empty`` and left out of the P/R/F1 (they can't be scored as either class).

With ``--force-json`` the answer region is constrained to schema-valid JSON
(two-phase decode: free reasoning up to ``</think>``, then an xgrammar-guided
continuation), so every completion parses and ``empty`` drops to 0 —
including pairs whose reasoning overran the budget, which get force-closed.

Predictions are saved in the ``call_api.py`` teacher schema (one greedy sample
per pair), so the output file can be fed straight to ``sft_sense.py --data``.

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

import torch
import xgrammar as xgr
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

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


def compile_wic_grammar(model, tokenizer):
    """Compile WIC_JSON_SCHEMA against the tokenizer, once per run.

    ``vocab_size`` comes from the model config, not the tokenizer: Qwen pads the
    embedding matrix past the tokenizer vocab, and the grammar bitmask must match
    the logits width.
    """
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=model.config.vocab_size)
    return xgr.GrammarCompiler(info).compile_json_schema(json.dumps(WIC_JSON_SCHEMA))


def generate_batch(model, tokenizer, texts, grammar=None):
    """Greedy completions for a batch of prompts.

    With a compiled ``grammar``, decoding runs in two phases: free reasoning
    stopped at ``</think>`` (force-closed if the token budget runs out first),
    then a continuation whose tokens are constrained to the grammar, so every
    completion ends in a parseable verdict.
    """
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    gen_kwargs = dict(do_sample=False, pad_token_id=tokenizer.pad_token_id)
    input_len = inputs["input_ids"].shape[1]

    if grammar is None:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=1024, **gen_kwargs)
        return [tokenizer.decode(o[input_len:], skip_special_tokens=True) for o in outputs]

    # Phase 1: free-form reasoning, halted at the close of the think block.
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, stop_strings=["</think>"],
            tokenizer=tokenizer, **gen_kwargs,
        )
    thinks = []
    for out in outputs:
        think = tokenizer.decode(out[input_len:], skip_special_tokens=True)
        if "</think>" not in think:  # budget ran out mid-reasoning: force-close
            think += "\n</think>"
        thinks.append(think.rstrip() + "\n")

    # Phase 2: constrained continuation — only tokens forming schema-valid JSON.
    # The xgrammar processor is stateful (one matcher per row), so build a fresh
    # one for every generate call; the compiled grammar itself is reused.
    inputs2 = tokenizer(
        [p + t for p, t in zip(texts, thinks)],
        return_tensors="pt", padding=True, truncation=True,
    ).to(model.device)
    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2, max_new_tokens=512,
            logits_processor=LogitsProcessorList([xgr.contrib.hf.LogitsProcessor(grammar)]),
            **gen_kwargs,
        )
    input_len2 = inputs2["input_ids"].shape[1]
    return [
        think + tokenizer.decode(o[input_len2:], skip_special_tokens=True)
        for think, o in zip(thinks, outputs2)
    ]


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
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--force-json", action="store_true",
                    help="constrain the answer region to schema-valid JSON (xgrammar)")
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

    grammar = compile_wic_grammar(model, tokenizer) if args.force_json else None

    hyps, refs, records = [], [], []
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i : i + args.batch_size]
        texts = [build_prompt(r, tokenizer) for r in batch]
        decoded_batch = generate_batch(model, tokenizer, texts, grammar)
        for rec, decoded in zip(batch, decoded_batch):
            hyp, gold = sd.extract_wic_label(decoded), rec["label"]
            hyps.append(hyp)
            refs.append(gold)
            # Teacher-predictions schema (call_api.py), single greedy sample —
            # the output file feeds sft_sense.py --data via load_teacher_traces.
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
    # sft_sense.py --data. Metrics are printed above, not saved.
    out_path = Path(args.output or f"predictions_sense_wic_{args.split}.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved predictions → {out_path}")


if __name__ == "__main__":
    main()
