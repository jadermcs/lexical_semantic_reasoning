"""Build the mixed SFT dataset once, ahead of training, so it can be inspected.

Loads teacher traces for every task, distils/selects one trace per record, renders
each to the uniform conversational ``{prompt, completion}`` shape, mixes the tasks
into one deterministically shuffled set, carves off a dev split, and saves the
result as a HuggingFace ``DatasetDict`` that ``sft_sense.py`` loads straight from
disk. A ``<out>.preview.jsonl`` of readable examples is written alongside so the
data can be eyeballed before committing GPU time.

    uv run python src/prepare_data.py \
        --data mcl_semcor.json --def-data data/wordnet_def.json \
        --reasoning-select longest --out data/sft_wic-def

Adding a task later means adding a loader in ``sense_data.py`` and a line here — the
training script stays untouched because every record renders through the same
``build_messages`` dispatch.
"""

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

import sense_data as sd


def format_example(rec):
    # Conversational prompt/completion format: TRL renders the chat template and
    # masks the prompt so loss is computed on the assistant turn only. The record's
    # ``task`` tag picks the right prompt/target builder (wic, definition, ...).
    msgs = sd.build_messages(rec, with_target=True)
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def build_records(wic_data, def_data=None, strategy="first"):
    """Distilled, task-tagged SFT records pooled from every task, plus per-task counts."""
    recs = sd.load_teacher_traces(wic_data, strategy=strategy)
    counts = {"wic": len(recs)}
    if def_data is not None:
        def_recs = sd.load_definition_traces(def_data, strategy=strategy)
        counts["definition"] = len(def_recs)
        recs += def_recs
    return recs, counts


def split(recs, dev_frac=0.05, seed=42):
    """Deterministic shuffle (interleaving the tasks) then a front-carved dev split."""
    recs = list(recs)
    random.Random(seed).shuffle(recs)
    n_dev = max(1, int(len(recs) * dev_frac))
    return recs[n_dev:], recs[:n_dev]


def to_dataset_dict(train, dev):
    """Render both splits to uniform {prompt, completion} rows and wrap as a DatasetDict.

    Formatting happens here (not lazily in the trainer) because the raw records carry
    task-specific columns (wic vs definition); rendering first keeps the Arrow schema
    uniform across the mixed set.
    """
    return DatasetDict(
        {
            "train": Dataset.from_list([format_example(r) for r in train]),
            "dev": Dataset.from_list([format_example(r) for r in dev]),
        }
    )


def write_preview(recs, path, n=20):
    """Dump the first *n* records as readable JSONL (task + rendered turns) for review."""
    with Path(path).open("w") as f:
        for r in recs[:n]:
            msgs = sd.build_messages(r, with_target=True)
            f.write(
                json.dumps(
                    {
                        "task": r["task"],
                        "lemma": r.get("lemma"),
                        "prompt": msgs[-2]["content"],
                        "completion": msgs[-1]["content"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/mcl_semcor.json",
        help="WiC teacher predictions file written by call_api.py.",
    )
    ap.add_argument(
        "--def-data",
        default=None,
        help="Optional definition teacher-trace file (target word + usage → gloss "
        "over WordNet). When set, its examples are mixed into the WiC set.",
    )
    ap.add_argument(
        "--reasoning-select",
        choices=["first", "longest"],
        default="first",
        help="Which distilled teacher trace to keep per record: first surviving "
        "sample or the longest CoT.",
    )
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=None,
        help="Output dataset dir. Defaults to data/sft_<tasks>-<strategy>-<data-stem>.",
    )
    ap.add_argument("--preview", type=int, default=20, help="Examples to write to <out>.preview.jsonl.")
    args = ap.parse_args()

    recs, counts = build_records(
        args.data, def_data=args.def_data, strategy=args.reasoning_select
    )
    train, dev = split(recs, dev_frac=args.dev_frac, seed=args.seed)

    tasks = "wic-def" if args.def_data else "wic"
    out = args.out or f"data/sft_{tasks}-{args.reasoning_select}-{Path(args.data).stem}"

    ds = to_dataset_dict(train, dev)
    ds.save_to_disk(out)
    write_preview(train, f"{out}.preview.jsonl", n=args.preview)

    print(f"[prepare] tasks={counts} train={len(ds['train'])} dev={len(ds['dev'])}")
    print(f"[prepare] saved → {out}  (preview → {out}.preview.jsonl)")
    print(json.dumps(ds["train"][0], indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
