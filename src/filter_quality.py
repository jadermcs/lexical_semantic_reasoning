"""Prune a WiC corpus using the rubric verdicts from ``call_api.py -t quality``.

The scorer annotates, this deletes -- the same split ``filter_reasoning.py``
makes, and for the same reason: the judgements are the expensive part, so every
threshold here has to be retunable without paying for them again.

Each surviving record is written back in the *input* schema, so the output is a
drop-in replacement anywhere the original file was used (``call_api.py -f``,
``semcor_pairs``-style pair sets, ``prepare_data.py``). Pass ``--keep-scores``
to carry the ``quality`` block along for inspection instead.

Rules fire in a fixed order and a record is attributed to the first one that
rejects it, so the breakdown adds up to the number dropped rather than
double-counting a record that fails three ways::

    uv run python src/call_api.py -f data/semcor.full.json -t quality
    uv run python src/filter_quality.py \\
        --preds predictions_semcor.full_deepseek_deepseek-v4-flash-0731.jsonl \\
        --data data/semcor.full.json --out data/semcor.clean.json

``--teacher-disagrees keep`` is the one worth ablating: dropping every pair the
teacher answers against gold removes real annotation errors *and* every pair
hard enough to beat the teacher, which is not the same set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sense_data as sd
from call_api import QUALITY_AXES

# Fields the scorer adds; stripped on the way out unless --keep-scores.
SCORE_KEYS = ("quality", "prediction", "confidence", "votes", "answers", "reasonings")


def _load_preds(path: Path) -> dict[tuple, dict]:
    """Map scored records by pair key, last line winning on duplicates."""
    preds: dict[tuple, dict] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "error" in r:
            continue
        preds[sd.pair_key(r)] = r
    return preds


def reject(rec: dict, args) -> str | None:
    """Name of the first rule that rejects *rec*, or None if it survives.

    ``rec`` is a scored record: the corpus fields plus the ``quality`` block.
    An axis that came back None (the k samples tied, or none parsed) is *not*
    treated as a failure -- the model declined to grade it, which is a
    different thing from grading it badly, and `unparsed` already catches the
    case where it declined on everything.
    """
    q = rec.get("quality") or {}
    if not q.get("n_parsed"):
        return "unparsed"
    if rec.get("prediction") is None:
        return "no_prediction"
    for axis, good in QUALITY_AXES.items():
        if axis in args.skip_axis:
            continue
        v = q.get(axis)
        if v is not None and bool(v) != good:
            return axis
    difficulty = q.get("difficulty")
    if args.max_difficulty and difficulty is not None and difficulty > args.max_difficulty:
        return "too_difficult"
    if rec.get("confidence") is not None and rec["confidence"] < args.min_confidence:
        return "low_confidence"
    if args.teacher_disagrees == "drop" and rec.get("label") is not None:
        if int(rec["prediction"]) != int(rec["label"]):
            return "teacher_disagrees"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--preds", required=True, help="predictions_*.jsonl from call_api.py -t quality"
    )
    ap.add_argument(
        "--data",
        help="The corpus that was scored. Given, the output preserves its records "
        "verbatim and unscored items are reported; omitted, the records are "
        "rebuilt from --preds.",
    )
    ap.add_argument("--out", help="Output JSON (default: <data stem>.clean.json)")
    ap.add_argument(
        "--dropped-out", help="Optional JSONL of rejected records, each with its rule"
    )
    ap.add_argument(
        "--skip-axis",
        nargs="*",
        default=[],
        choices=list(QUALITY_AXES),
        help="Rubric axes to ignore when filtering.",
    )
    ap.add_argument(
        "--max-difficulty",
        type=int,
        default=None,
        help="Drop items graded harder than this (1-5). Off by default: hard is "
        "not the same as bad, and the hard items are where the reward signal is.",
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Drop items whose same_sense vote was less unanimous than this.",
    )
    ap.add_argument(
        "--teacher-disagrees",
        choices=("drop", "keep"),
        default="drop",
        help="What to do when the teacher's same_sense contradicts the gold label.",
    )
    ap.add_argument(
        "--keep-unscored",
        action="store_true",
        help="Keep --data records missing from --preds (default: drop them).",
    )
    ap.add_argument(
        "--keep-scores",
        action="store_true",
        help="Carry the quality/prediction fields into the output.",
    )
    args = ap.parse_args()

    preds = _load_preds(Path(args.preds))
    if args.data:
        corpus = json.loads(Path(args.data).read_text())
        out_path = Path(args.out or Path(args.data).with_suffix("").as_posix() + ".clean.json")
    else:
        corpus = list(preds.values())
        if not args.out:
            sys.exit("--out is required when --data is omitted")
        out_path = Path(args.out)

    kept: list[dict] = []
    dropped: list[dict] = []
    counts: dict[str, int] = {}
    for item in corpus:
        scored = preds.get(sd.pair_key(item))
        if scored is None:
            rule = None if args.keep_unscored else "unscored"
            rec = dict(item)
        else:
            # corpus fields win: --preds may hold an older copy of the record
            rec = {**scored, **item}
            rule = reject(rec, args)
        if rule is None:
            kept.append(
                rec if args.keep_scores else {k: v for k, v in rec.items() if k not in SCORE_KEYS}
            )
        else:
            counts[rule] = counts.get(rule, 0) + 1
            dropped.append({**rec, "drop_rule": rule})

    out_path.write_text(json.dumps(kept, indent=2, ensure_ascii=False))
    report = {
        "n_input": len(corpus),
        "n_scored": sum(1 for i in corpus if sd.pair_key(i) in preds),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped_by_rule": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "label_balance": _balance(kept),
        "label_balance_input": _balance(corpus),
    }
    Path(out_path.as_posix() + ".report.json").write_text(json.dumps(report, indent=2))
    if args.dropped_out:
        Path(args.dropped_out).write_text(
            "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dropped)
        )
    print(json.dumps(report, indent=2))
    print(f"Wrote {len(kept)} records to {out_path}", file=sys.stderr)


def _balance(records: list[dict]) -> dict:
    n = len(records)
    same = sum(1 for r in records if r.get("label"))
    return {"same": same, "different": n - same, "frac_same": same / n if n else 0.0}


if __name__ == "__main__":
    main()
