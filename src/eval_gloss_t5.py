"""Run a seq2seq definition-generation baseline over the `eval_gloss.py` records.

The reference point is ``ltg/flan-t5-definition-en-xl`` (Giulianelli et al. 2023),
which is fine-tuned *on the WordNet and Oxford training splits* -- i.e. on the
house style of two of the four corpora it is scored against here. That makes it
the natural probe for the style question: if overlap metrics mostly reward
register, its margin over an untuned policy should be large on wordnet/oxford
and collapse on slang/wiki, which are out of domain for both models.

Its prompt is a bare usage sentence followed by the question, with no target
markup::

    He ate a sweet apple. What is the definition of apple?

so the ``<t>`` tags `dm_gloss_data.py` inserts are stripped back out here.

Output is the same ``{"summary": ..., "records": [...]}`` shape `eval_gloss.py`
writes, so scoring goes through *its* code path, not a reimplementation::

    uv run python src/eval_gloss_t5.py --data data/gloss_eval_wordnet_test.json \\
        --output analysis/gloss_dm/wordnet_t5xl.json
    uv run --with sacrebleu python src/eval_gloss.py \\
        --score-only analysis/gloss_dm/wordnet_t5xl.json --output analysis/gloss_dm/wordnet_t5xl.json

``filler_gloss`` stays empty: the filler sentence is an artefact of casting a
WiC-pair policy at a single usage, and this model takes one usage natively.
The ``filler_*`` numbers in the summary are therefore meaningless for it.
"""

import argparse
import json
import re
from pathlib import Path

import eval_gloss as eg

TAGS = re.compile(r"</?t>")


def _clean(text):
    text = re.sub(r"\s+", " ", TAGS.sub("", text)).strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def build_prompt(rec, pair_mode="none"):
    """`<usage> What is the definition of <target>?` -- the card's format, untagged.

    ``pair_mode="concat"`` puts a second usage in front of the question too. The
    model takes one context natively, so this is an adaptation rather than its
    trained format -- but without it the paired condition would hand our policy a
    second usage and withhold it from the baseline, which is not a fair contrast.
    """
    usage = _clean(rec["usage"])
    if pair_mode == "concat" and rec.get("usage2"):
        usage = f"{usage} {_clean(rec['usage2'])}"
    return f"{usage} What is the definition of {rec['word']}?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ltg/flan-t5-definition-en-xl")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-beams", type=int, default=4, help="1 = greedy")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pair-mode", default="none", choices=("none", "concat"),
                    help="concat = prepend a second usage, when the record has one")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    data = eg.load_gloss_records(args.data)
    if args.max_samples:
        data = data[: args.max_samples]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.to(args.device).eval()

    prompts = [build_prompt(r, pair_mode=args.pair_mode) for r in data]
    glosses = []
    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=512).to(args.device)
        with torch.no_grad():
            out = model.generate(
                **enc, num_beams=args.num_beams, max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        glosses.extend(g.strip() for g in tok.batch_decode(out, skip_special_tokens=True))
        print(f"\r{min(i + args.batch_size, len(prompts))}/{len(prompts)}", end="", flush=True)
    print()

    rows = []
    for r, p, g in zip(data, prompts, glosses):
        row = {
            "lemma": r["lemma"], "word": r["word"], "pos": r["pos"],
            "usage": r["usage"], "definition": r["definition"],
            "gloss": g, "filler_gloss": "", "think": "",
            "prompt": p,
        }
        # carried through so the paired and inventory metrics in eval_gloss can
        # score this model on exactly the same terms as the policy
        for key in ("usage2", "definition2", "senses"):
            if key in r:
                row[key] = r[key]
        rows.append(row)
    Path(args.output).write_text(
        json.dumps({"summary": {}, "records": rows}, ensure_ascii=False, indent=2)
    )
    print(f"Saved → {args.output}  (score with: eval_gloss.py --score-only)")


if __name__ == "__main__":
    main()
