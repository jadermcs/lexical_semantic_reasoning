"""Evaluate a WiC checkpoint on a held-out MCL-WiC split.

Greedily generates a verdict per test pair (vLLM, continuous batching) and
reports accuracy plus same/different precision/recall/F1. Completions with no
extractable verdict are counted as ``empty`` and left out of the P/R/F1 (they
can't be scored as either class).

With ``--force-json`` the answer region is constrained to schema-valid JSON
(two-phase decode: free reasoning up to ``</think>``, then an xgrammar-guided
continuation via vLLM structured outputs), so every completion parses and
``empty`` drops to 0 — including pairs whose reasoning overran the budget,
which get force-closed. Prefix caching means the phase-2 pass reuses the
phase-1 KV cache instead of re-prefilling prompt + reasoning.

Predictions are saved in the ``call_api.py`` teacher schema, so the output file
feeds straight into ``sft_sense.py --file_path`` — ``utils.build_sft_dataset``
runs it through ``sense_data.load_teacher_traces``, which keeps the pairs the
policy got *right* and turns each into a ``{prompt, completion}`` row. That is
the self-distillation loop: eval a checkpoint on a train split, train the next
one on its own correct traces. Every field of the input record that isn't an
output key is carried through, so ``source``/``lang1``/``senses`` survive into
the SFT file and the distilled set can still be stratified.

``-k`` is what makes the loop worth running. At k=1 (greedy, the default) each
pair yields exactly one trace, so ``load_teacher_traces``'s ``longest``
selection has nothing to select from and a lucky-but-wrong trace is kept
whenever the verdict happens to match gold. At k>1 the pairs are decoded with
Qwen3 thinking-mode sampling, ``prediction`` becomes a majority vote (ties →
``None``, dropped downstream), and only the samples that agree with that vote
are distillation candidates.

Examples
--------
  # a trained checkpoint
  uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic
  # zero-shot base-model baseline
  uv run python src/eval_sense.py --model Qwen/Qwen3-0.6B
  # guarantee a parseable JSON verdict on every pair
  uv run python src/eval_sense.py --model ./qwen-sense-grpo-wic --force-json
  # an unmerged LoRA adapter served on top of its base model
  uv run python src/eval_sense.py --model ./qwen-sft_wic_filtered \\
      --lora ./qwen-lora-sdpo-wic/checkpoint-500
  # self-distillation: 5 sampled traces per train pair, then SFT on the output
  uv run python src/eval_sense.py --model ./qwen-lora-grpo-wic --force-json \\
      --path data/xl-lexeme.json -k 5 --output data/self_distill.json
  uv run python src/sft_sense.py --file_path data/self_distill.json
"""

import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

import sense_data as sd


def load_wic_records(path):
    """WiC records straight off disk, in the teacher schema.

    Not ``sense_data.load_teacher_traces`` (which keeps only teacher-correct pairs
    and needs a reasoning trace) and not ``utils.build_grpo_dataset`` (which drops
    everything outside ``KEEP_COLS``, including the two sentences): evaluation
    needs every labelled pair, unfiltered, with its record intact.
    """
    raw = json.loads(Path(path).read_text())
    return [
        r
        for r in raw
        if r.get("task", "wic") == "wic" and r.get("label") is not None
    ]


def build_prompt(rec, tokenizer):
    msgs = sd.wic_messages(rec, with_target=False)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


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


# Written by this script, so re-evaluating a predictions file doesn't feed a
# previous run's verdicts back out as if they were corpus fields.
OUTPUT_KEYS = frozenset(
    {"task", "prediction", "confidence", "votes", "answers", "reasonings", "error"}
)


def load_lora(path):
    if not path:
        return None, None
    from vllm.lora.request import LoRARequest

    adapter = Path(path).resolve()
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    return LoRARequest("adapter", 1, str(adapter)), int(cfg["r"])


def _sampling_params(n, **kw):
    """Greedy at n=1, Qwen3 thinking-mode sampling above it.

    k=1 stays deterministic so a plain eval run is reproducible; asking for more
    than one sample of a greedy decode would return n identical copies, so k>1
    switches to the sampling settings Qwen3 documents for thinking mode (and that
    the analysis scripts already match: temp 0.6 / top-p 0.95 / top-k 20 / min-p 0).
    """
    if n == 1:
        return SamplingParams(temperature=0.0, **kw)
    return SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, **kw)


def generate_all(llm, texts, force_json=False, lora_request=None, schema=None, n=1):
    """``n`` completions per prompt; ``force_json`` constrains the answer region.

    Returns one list of completions per prompt (length ``n``), so callers that
    only ever want a single verdict take ``[0]``.

    ``schema`` defaults to the WiC answer object — other tasks (``eval_assign``)
    pass their own, which is the only thing about the two-phase decode that is
    task-specific.
    """
    if not force_json:
        sp = _sampling_params(n, max_tokens=1024)
        return [
            [o.text for o in out.outputs]
            for out in llm.generate(texts, sp, lora_request=lora_request)
        ]

    # Phase 1: free-form reasoning, halted at the close of the think block.
    sp1 = _sampling_params(
        n,
        max_tokens=1024,
        stop=["</think>"],
        include_stop_str_in_output=True,
    )
    thinks = []  # one list of n think blocks per prompt
    for out in llm.generate(texts, sp1, lora_request=lora_request):
        per_prompt = []
        for o in out.outputs:
            think = o.text
            if "</think>" not in think:  # budget ran out mid-reasoning: force-close
                think += "\n</think>"
            per_prompt.append(think.rstrip() + "\n")
        thinks.append(per_prompt)

    # Phase 2: constrained continuation — only tokens forming schema-valid JSON.
    # Flattened to one request per (prompt, sample): the samples diverge in phase 1,
    # so each needs its own reasoning prefix. Greedy here even when n>1 — the
    # verdict is read off the reasoning, and sampling it would add noise the vote
    # would then have to average back out.
    sp2 = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        structured_outputs=StructuredOutputsParams(json=schema or WIC_JSON_SCHEMA),
    )
    flat = [p + t for p, per_prompt in zip(texts, thinks) for t in per_prompt]
    outs2 = llm.generate(flat, sp2, lora_request=lora_request)
    out_all, i = [], 0
    for per_prompt in thinks:  # re-group by each prompt's own sample count
        out_all.append(
            [t + out.outputs[0].text for t, out in zip(per_prompt, outs2[i:])]
        )
        i += len(per_prompt)
    return out_all


def majority_vote(votes):
    """Self-consistency vote over the sampled verdicts → (prediction, confidence).

    Mirrors ``call_api._vote``: unparseable samples abstain, and a tie is *no*
    prediction rather than a coin flip, so ``load_teacher_traces`` drops the pair
    instead of distilling whichever half happened to be listed first.
    """
    valid = [v for v in votes if v is not None]
    if not valid:
        return None, 0.0
    trues = sum(valid)
    if trues * 2 == len(valid):  # tie → no prediction
        return None, 0.5
    return trues * 2 > len(valid), max(trues, len(valid) - trues) / len(valid)


def wic_metrics(preds, golds):
    """Accuracy + same/different P/R/F1 over parsed verdicts; None preds are unscored."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    scored = [(p, g) for p, g in zip(preds, golds) if p is not None]
    n = len(scored)
    y_pred = [p for p, _ in scored]
    y_true = [g for _, g in scored]
    if n:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            pos_label=True,
            zero_division=0,
        )
        acc = accuracy_score(y_true, y_pred)
    else:
        prec = rec = f1 = acc = 0.0
    return {
        "n": len(preds),
        "n_scored": n,
        "empty": len(preds) - n,
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--path", default="data/mcl-wic.test.json")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = full split")
    ap.add_argument(
        "-k",
        "--samples",
        type=int,
        default=1,
        help="Completions per pair (default 1 = greedy). Above 1 the pairs are "
        "decoded with thinking-mode sampling and the verdict is a majority vote, "
        "which is what gives the SFT distillation candidates to choose between. "
        "An even k ties out more often than an odd one.",
    )
    ap.add_argument("--force-json", action="store_true")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    lora_req, lora_rank = load_lora(args.lora)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,  # phase 2 of --force-json reuses phase-1 KV
        **(dict(enable_lora=True, max_lora_rank=lora_rank) if lora_req else {}),
    )
    tokenizer = llm.get_tokenizer()

    data = load_wic_records(args.path)
    if args.max_samples:
        data = data[: args.max_samples]

    texts = [build_prompt(r, tokenizer) for r in data]
    decoded_all = generate_all(
        llm,
        texts,
        force_json=args.force_json,
        lora_request=lora_req,
        n=args.samples,
    )

    hyps, refs, records = [], [], []
    for rec, samples in zip(data, decoded_all):
        gold = rec["label"]
        votes = [sd.extract_wic_label(d) for d in samples]
        hyp, confidence = majority_vote(votes)
        hyps.append(hyp)
        refs.append(gold)
        # Teacher-predictions schema (call_api.py): the output file feeds
        # sft_sense.py --file_path via load_teacher_traces. `answers` and
        # `reasonings` are kept index-aligned and always length k — the
        # candidate builder zips them, so a sample that lost only its JSON must
        # still hold its slot or the next sample's verdict would be paired with
        # this one's reasoning. Empty strings are rejected there, so a padded
        # slot is dropped rather than distilled.
        answers, reasonings = [], []
        for decoded in samples:
            think, closed, _ = decoded.partition("</think>")
            answer = sd.parse_wic_answer(decoded)
            answers.append(
                json.dumps(answer, ensure_ascii=False) if answer is not None else ""
            )
            reasonings.append(think.replace("<think>", "").strip() if closed else "")
        # Everything the source record carried that isn't an output key rides
        # along (source/lang/senses/split), so the distilled SFT set can still be
        # stratified by where its pairs came from.
        base = {k: v for k, v in rec.items() if k not in OUTPUT_KEYS}
        records.append(
            {
                **base,
                "task": "wic",
                "prediction": hyp,
                "confidence": confidence,
                "votes": votes,
                "answers": answers,
                "reasonings": reasonings,
            }
        )

    metrics = wic_metrics(hyps, refs)
    print(
        f"\n[wic] n={metrics['n']}  acc={metrics['accuracy']:.3f}  "
        f"f1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  empty={metrics['empty']}"
    )

    print("\nExamples (first 10):")
    print(f"{'lemma':<18}  {'gold':<10}  {'prediction':<10}")
    for rec in records[:10]:
        pred = "—" if rec["prediction"] is None else str(rec["prediction"])
        label = "—" if rec["label"] is None else str(rec["label"])
        print(f"{rec['lemma']:<18}  {label:<10}  {pred:<10}")

    # Bare list in the call_api.py teacher schema — directly consumable by
    # sft_sense.py --file_path. Metrics are printed above, not saved.
    out_path = Path(args.output or f"predictions_sense_wic_{Path(args.path).stem}.json")
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"Saved predictions → {out_path}")

    # The SFT yield, measured through the loader that will actually read the file
    # rather than re-derived here: rows kept = pairs the policy got right that
    # also left a usable trace behind.
    kept = sd.load_teacher_traces(out_path, strategy="longest")
    print(
        f"SFT-usable: {len(kept)}/{len(records)} rows "
        f"(uv run python src/sft_sense.py --file_path {out_path})"
    )


if __name__ == "__main__":
    main()
