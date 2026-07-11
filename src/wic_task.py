"""The WiC (word-in-context) RL task: materialise it to a file before training.

GRPO used to build its WiC records in memory at train time, so the examples the
policy actually saw were never written down. Here the task is built once and
dumped to ``data/wic_task.<split>.jsonl`` — one record per pair, carrying both the
fields GRPO needs (``lemma``, ``label``, the <t>-marked usages) and the fully
rendered ``system``/``user`` prompt, so the examples can be read and checked by
hand before spending a run on them.

The prompt is rendered by ``sense_data.wic_messages`` — the same builder
``sft_sense`` uses — so what is inspected here is exactly what SFT trained on and
what GRPO will roll out against.

Build/inspect (no torch needed):

    uv run python src/wic_task.py --splits train dev          # write the files
    uv run python src/wic_task.py --splits dev --show 3       # print a few examples
"""

import argparse
import json
from pathlib import Path

import sense_data as sd

DATA_DIR = Path("data")

# Columns GRPO's reward fns read (see grpo_sense.KEEP_COLS["wic"]) plus the two
# usages the prompt is rendered from. The rendered system/user prompt is stored
# alongside them for inspection only.
RECORD_COLS = ["lemma", "pos", "label", "usage1", "usage2"]


def task_path(split: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / f"wic_task.{split}.jsonl"


def build(split: str) -> list[dict]:
    """MCL-WiC pairs as wic records, each with its rendered prompt attached."""
    out = []
    for rec in sd.load_mclwic(split):
        msgs = sd.wic_messages(rec, with_target=False)
        out.append(
            {
                **{c: rec[c] for c in RECORD_COLS},
                "system": msgs[0]["content"],
                "user": msgs[1]["content"],
            }
        )
    return out


def save(split: str, recs: list[dict], data_dir: Path = DATA_DIR) -> Path:
    path = task_path(split, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def load(split: str, data_dir: Path = DATA_DIR) -> list[dict]:
    """Load the dumped task, building it on first use if the file is missing.

    Returns only the columns GRPO consumes; the rendered prompt is re-derived by
    the trainer from the same ``wic_messages`` builder, so carrying it into the
    dataset would just duplicate it.
    """
    path = task_path(split, data_dir)
    if not path.exists():
        recs = build(split)
        save(split, recs, data_dir)
        print(f"[wic] built {len(recs)} examples -> {path}")
    else:
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [{c: r[c] for c in RECORD_COLS} for r in recs]


def _show(recs, n):
    for r in recs[:n]:
        print("=" * 72)
        print(f"[{r['lemma']} ({r['pos']})] gold: {r['label']}")
        print("--- system ---")
        print(r["system"])
        print("--- user ---")
        print(r["user"])
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--show", type=int, default=0, help="Print this many rendered examples.")
    args = ap.parse_args()

    for split in args.splits:
        recs = build(split)
        path = save(split, recs)
        counts = {lab: sum(r["label"] == lab for r in recs) for lab in ("same", "different")}
        print(f"{split:5s} n={len(recs):6d}  same={counts['same']}  different={counts['different']}  -> {path}")
        if args.show:
            _show(recs, args.show)


if __name__ == "__main__":
    main()
