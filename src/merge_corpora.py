"""Union two filtered teacher-schema corpora into one training corpus.

The self-distillation loop re-annotates pairs the API teacher already covered,
so its output is not additive on its own: measured on the 9000 train+dev pairs,
the student corpus and the DeepSeek corpus each keep ~5.7k pairs but overlap on
only ~3.8k. Training on the student set *alone* is a downgrade -- it loses the
~1.9k pairs the student got wrong that the teacher got right. The union is what
gains, and the two are complementary by label: the teacher corpus skews
*different* (64%) because `--strict-gloss` rejects `diff_gloss_same_label`,
while the student's new pairs are 91% *same*, so the union lands near 50/50
without down-sampling.

Records are keyed by ``sense_data.pair_key``. Where both corpora cover a pair,
``--prefer`` decides: ``teacher`` (default) keeps the stronger model's trace,
``both`` concatenates them so ``prepare_data.py --reasoning-select longest`` can
choose. Where only one covers it, that one is taken.

    uv run python src/merge_corpora.py \
        --base data/mcl_train_dev_filtered.json \
        --add data/selfdistil_filtered.json \
        --out data/union_filtered.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sense_data as sd


def load(path):
    return json.loads(Path(path).read_text())


def has_trace(rec):
    return bool(rec.get("reasonings")) and bool(rec.get("answers"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="preferred corpus (e.g. the API teacher)")
    ap.add_argument("--add", required=True, help="corpus to union in (e.g. self-distilled)")
    ap.add_argument("--prefer", choices=["teacher", "both"], default="teacher",
                    help="how to resolve pairs present in both corpora")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = [r for r in load(args.base) if has_trace(r)]
    add = [r for r in load(args.add) if has_trace(r)]
    bykey = {sd.pair_key(r): r for r in base}

    stats = Counter()
    for r in add:
        k = sd.pair_key(r)
        if k not in bykey:
            bykey[k] = r
            stats["new"] += 1
        elif args.prefer == "both":
            tgt = bykey[k]
            tgt["reasonings"] = list(tgt.get("reasonings") or []) + list(r.get("reasonings") or [])
            tgt["answers"] = list(tgt.get("answers") or []) + list(r.get("answers") or [])
            tgt["votes"] = list(tgt.get("votes") or []) + list(r.get("votes") or [])
            stats["merged"] += 1
        else:
            stats["kept_base"] += 1

    out = list(bykey.values())
    labels = Counter(bool(r["label"]) for r in out)
    n = len(out)
    print(f"base {args.base}: {len(base)} pairs")
    print(f"add  {args.add}: {len(add)} pairs")
    print(f"  new from add : {stats['new']}")
    print(f"  overlapping  : {stats['kept_base'] + stats['merged']} "
          f"({'concatenated' if args.prefer == 'both' else 'base trace kept'})")
    print(f"union: {n} pairs  "
          f"same={labels[True]} ({labels[True] / n:.1%})  "
          f"different={labels[False]} ({labels[False] / n:.1%})")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
