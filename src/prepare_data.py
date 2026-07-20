"""Build the SFT dataset once, ahead of training, so it can be inspected.

Loads teacher traces, distils/selects one trace per record, renders each to the
uniform conversational ``{prompt, completion}`` shape, deterministically shuffles,
carves off a dev split, and saves the result as a HuggingFace ``DatasetDict`` that
``sft_sense.py`` loads straight from disk. A ``<out>.preview.jsonl`` of readable
examples is written alongside so the data can be eyeballed before committing GPU time.

    uv run python src/prepare_data.py \
        --data mcl_semcor.json --reasoning-select longest --out data/sft_wic

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
    msgs = sd.build_messages(rec, with_target=True)
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def balance_wic_labels(recs, seed=42):
    """Down-sample the majority same/different class so the wic set is 50/50.

    The teacher-filtered WiC distillation set is skewed (``mcl_train_dev_filtered`` is
    ~64/36 toward ``same_sense=false``), while the dev/test eval splits are exactly
    50/50. An imbalanced warm-start instils a matching label prior, so GRPO starts from
    a low balanced-accuracy point. Balancing the SFT set removes that prior. Only the
    dropped pairs never enter the ``sft_pairs`` manifest, so they also become available
    to the GRPO rollout set (see ``--exclude-pairs`` in ``grpo_sense.py``).
    """
    same = [r for r in recs if r["label"]]
    diff = [r for r in recs if not r["label"]]
    n = min(len(same), len(diff))
    rng = random.Random(seed)
    rng.shuffle(same)
    rng.shuffle(diff)
    out = same[:n] + diff[:n]
    rng.shuffle(out)
    return out


def build_records(wic_data, strategy="first", balance=False, seed=42):
    """Distilled, task-tagged SFT records, plus per-task counts."""
    recs = sd.load_teacher_traces(wic_data, strategy=strategy)
    if balance:
        before = len(recs)
        recs = balance_wic_labels(recs, seed=seed)
        print(f"[prepare] balanced wic labels: {before} → {len(recs)} pairs (50/50)")
    counts = {"wic": len(recs)}
    return recs, counts


def split(recs, dev_frac=0.05, seed=42):
    """Deterministic shuffle then a front-carved dev split."""
    recs = list(recs)
    random.Random(seed).shuffle(recs)
    n_dev = max(1, int(len(recs) * dev_frac))
    return recs[n_dev:], recs[:n_dev]


def to_dataset_dict(train, dev):
    """Render both splits to uniform {prompt, completion} rows and wrap as a DatasetDict.

    Formatting happens here (not lazily in the trainer) because the raw records carry
    task-specific columns; rendering first keeps the Arrow schema uniform.
    """
    return DatasetDict(
        {
            "train": Dataset.from_list([format_example(r) for r in train]),
            "dev": Dataset.from_list([format_example(r) for r in dev]),
        }
    )


def write_sft_pairs(recs, path):
    """Write the identity of every WiC pair distilled into the SFT set.

    Emits a JSON list of ``pair_key`` tuples for all wic records (both the train and
    dev splits are seen by the model). ``grpo_sense.py --exclude-pairs`` reads this to
    hold those pairs out of the GRPO rollout set, so RL only ever rolls out on pairs
    the SFT warm-start never saw.
    """
    keys = [list(sd.pair_key(r)) for r in recs if r.get("task") == "wic"]
    Path(path).write_text(json.dumps(keys, ensure_ascii=False))
    return len(keys)


def write_preview(recs, path, n=20):
    """Dump the first *n* records as readable JSONL (task + rendered turns) for review."""
    with Path(path).open("w") as f:
        for r in recs[:n]:
            msgs = sd.build_messages(r, with_target=True)
            system = next(
                (m["content"] for m in msgs if m["role"] == "system"), None
            )
            f.write(
                json.dumps(
                    {
                        "task": r["task"],
                        "lemma": r.get("lemma"),
                        "system": system,
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
        "--reasoning-select",
        choices=["first", "longest"],
        default="first",
        help="Which distilled teacher trace to keep per record: first surviving "
        "sample or the longest CoT.",
    )
    ap.add_argument(
        "--balance-labels",
        action="store_true",
        help="Down-sample the majority same/different class among wic records so the "
        "distilled set is 50/50. The teacher-filtered set skews toward "
        "same_sense=false while dev/test are exactly balanced; an imbalanced "
        "warm-start biases the label prior and lowers GRPO's starting accuracy. "
        "Dropped pairs also fall out of the sft_pairs manifest, so they free up for "
        "the GRPO rollout set.",
    )
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=None,
        help="Output dataset dir. Defaults to data/sft_wic-<strategy>-<data-stem>.",
    )
    ap.add_argument("--preview", type=int, default=20, help="Examples to write to <out>.preview.jsonl.")
    args = ap.parse_args()

    recs, counts = build_records(
        args.data,
        strategy=args.reasoning_select,
        balance=args.balance_labels,
        seed=args.seed,
    )
    train, dev = split(recs, dev_frac=args.dev_frac, seed=args.seed)

    out = args.out or f"data/sft_wic-{args.reasoning_select}-{Path(args.data).stem}"

    ds = to_dataset_dict(train, dev)
    ds.save_to_disk(out)
    write_preview(train, f"{out}.preview.jsonl", n=args.preview)
    # Manifest of every WiC pair consumed by SFT (train ∪ dev, i.e. all of recs), so
    # GRPO can roll out only on the pairs the policy never saw during warm-start.
    n_pairs = write_sft_pairs(recs, f"{out}.sft_pairs.json")

    print(f"[prepare] tasks={counts} train={len(ds['train'])} dev={len(ds['dev'])}")
    print(f"[prepare] saved → {out}  (preview → {out}.preview.jsonl)")
    print(f"[prepare] SFT-consumed WiC pairs → {out}.sft_pairs.json  ({n_pairs} pairs to exclude from GRPO)")
    print(json.dumps(ds["train"][0], indent=2, ensure_ascii=False)[:1200])


if __name__ == "__main__":
    main()
