"""Pairwise LLM-judge arena over generated glosses, with arena-style style control.

Why this exists: BLEU/chrF against a single reference definition rewards writing in
the *reference dictionary's register* as much as denoting the right sense. The
distractor baselines make that concrete -- a real definition of a **different sense
of the same lemma** outscores a real model on three of the four corpora here. So
the overlap metrics are treated as the object under audit, not the measurement.

The measurement instead is a Bradley-Terry fit over pairwise judgments, exactly
the shape LMArena uses, plus its style-control extension:

    stage judge  every competitor pair on shared items, judged in **both orders**,
                 reference-free -- the judge sees the usage and two candidate
                 definitions, never the gold. With no reference on screen there is
                 no reference style left to match, which is the whole point.
    stage rate   fit  P(A beats B) = sigmoid(x^T β)  for model strengths β, then
                 refit as  sigmoid(x^T β + z^T γ)  with z the A-minus-B style
                 features. The shift from β to β|γ is the part of a model's
                 standing that was register, not content.

Two synthetic competitors anchor the scale and audit the judge itself:

    ``gold``   the reference definition, entered anonymously. It is the ceiling; a
               judge that does not put it near the top is not tracking content.
    ``rival``  the gold definition of a *different sense of the same lemma*:
               flawless dictionary style, deliberately wrong sense. It is the
               floor, and it is the exact failure mode chrF cannot see.

Examples
--------
  uv run python src/gloss_arena.py --stage judge --corpus wordnet \\
      --entry grpo=analysis/gloss_dm/wordnet_grpo.json \\
      --entry sft=analysis/gloss_dm/wordnet_sft.json \\
      --entry t5xl=analysis/gloss_dm/wordnet_t5xl.json \\
      --items 200 --out analysis/gloss_dm/battles_wordnet.jsonl
  uv run python src/gloss_arena.py --stage rate --battles analysis/gloss_dm/battles_wordnet.jsonl
"""

import argparse
import itertools
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import gloss_wordnet as gw

# Matches filter_reasoning.py's judge. Hybrid linear attention (8 of 32 layers
# carry a KV cache), int4, and the vision tower is skipped at load -- see the
# LLM(...) call below. Battles are stamped with the judge that produced them,
# because a Bradley-Terry fit over rows from two different judges is measuring
# neither.
JUDGE_MODEL = "cyankiwi/Qwen3.5-9B-AWQ-4bit"

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string", "maxLength": 200},
    },
    "required": ["winner", "reason"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "You are an expert lexicographer judging definitions. You are given a sentence "
    "with a target word marked by <t> tags, and two candidate definitions of that "
    "word AS IT IS USED IN THAT SENTENCE.\n\n"
    "Judge only whether the definition identifies the correct sense and describes it "
    "accurately. Explicitly ignore: length, wording style, register, capitalisation, "
    "punctuation, and whether it reads like a particular dictionary. A blunt, plainly "
    "worded definition of the right sense beats a polished definition of the wrong "
    "sense. Answer 'tie' only when both identify the same sense equally well.\n\n"
    'Reply with JSON: {"winner": "A" | "B" | "tie", "reason": "<10 words>"}'
)

TAGS = re.compile(r"</?t>")


def judge_user(item, gloss_a, gloss_b):
    return (
        f"Sentence: {item['usage']}\n"
        f"Target word: {item['lemma']} ({item['pos']})\n\n"
        f"Definition A: {gloss_a}\n"
        f"Definition B: {gloss_b}\n\n"
        "Which definition better captures the sense of the target word in this sentence?"
    )


# ---------------------------------------------------------------- competitors


def load_entries(entries, eval_data, seed=0):
    """``{name: {item_key: gloss}}`` for every competitor, synthetic ones included.

    Items are keyed on (lemma, usage, definition) -- the eval record's identity --
    because every candidate file is a row-aligned rescoring of the same eval set.
    """
    rng = random.Random(seed)
    items = {}
    for rec in eval_data:
        items[(rec["lemma"], rec["usage"], rec["definition"])] = rec

    out = {}
    for spec in entries:
        name, _, path = spec.partition("=")
        saved = json.loads(Path(path).read_text())
        rows = saved["records"] if isinstance(saved, dict) else saved
        out[name] = {
            (r["lemma"], r["usage"], r["definition"]): r.get("gloss", "")
            for r in rows
            if r.get("gloss")
        }

    # gold: the reference itself, entered anonymously
    out["gold"] = {k: rec["definition"] for k, rec in items.items()}

    # rival: gold definition of another sense of the same lemma -- right style,
    # wrong sense. Drawn from the record's own ``senses`` inventory when the
    # source ships one (the lemma's real competing WordNet senses), else from
    # whatever other senses of that lemma happen to be in the file.
    by_lemma = defaultdict(list)
    for rec in eval_data:
        by_lemma[rec["lemma"]].append(rec["definition"])
    rival = {}
    for key, rec in items.items():
        pool = rec.get("senses") or by_lemma[rec["lemma"]]
        others = [d for d in pool if d != rec["definition"]]
        if others:
            rival[key] = rng.choice(others)
    out["rival"] = rival
    return out, items


def build_battles(entries, items, n_items=200, seed=0):
    """Every competitor pair, on a sample of items where both have a gloss, both orders."""
    rng = random.Random(seed)
    keys = [k for k in items if sum(k in e for e in entries.values()) >= 2]
    rng.shuffle(keys)
    keys = keys[:n_items] if n_items else keys

    battles = []
    for key in keys:
        present = [n for n, e in entries.items() if key in e and e[key].strip()]
        for a, b in itertools.combinations(sorted(present), 2):
            for first, second in ((a, b), (b, a)):
                battles.append(
                    {
                        "key": list(key),
                        "model_a": first,
                        "model_b": second,
                        "gloss_a": entries[first][key],
                        "gloss_b": entries[second][key],
                    }
                )
    rng.shuffle(battles)
    return battles


# ---------------------------------------------------------------- style features


def style_features(gloss, corpus_shell):
    """Content-neutral surface features, the analogue of the arena's markdown counts.

    ``shell`` is the share of the gloss's tokens drawn from the corpus's own
    high-frequency definition vocabulary -- 'of or relating to', 'a person who' --
    i.e. how much the gloss *sounds like* this particular dictionary.
    """
    toks = gw._toks(gloss)
    words = gloss.split()
    return {
        "len": len(toks),
        "shell": (sum(t in corpus_shell for t in toks) / len(toks)) if toks else 0.0,
        "label": float(bool(re.match(r"^\s*\w+(?:/\w+)?\s*:", gloss))),
        "cap": float(bool(words) and words[0][:1].isupper()),
        "period": float(gloss.rstrip().endswith(".")),
    }


def corpus_shell_vocab(eval_data, top=100):
    """The top-N most frequent tokens across this dictionary's own definitions."""
    cnt = Counter(t for r in eval_data for t in gw._toks(r["definition"]))
    return {t for t, _ in cnt.most_common(top)}


def diff_features(fa, fb):
    """A-minus-B, with length normalised the way the arena normalises token counts."""
    total = fa["len"] + fb["len"]
    return {
        "d_len": ((fa["len"] - fb["len"]) / total) if total else 0.0,
        "d_shell": fa["shell"] - fb["shell"],
        "d_label": fa["label"] - fb["label"],
        "d_cap": fa["cap"] - fb["cap"],
        "d_period": fa["period"] - fb["period"],
    }


STYLE_COLS = ["d_len", "d_shell", "d_label", "d_cap", "d_period"]


# ---------------------------------------------------------------- Bradley-Terry


def fit_bt(battles, models, style=False, scale=400.0, init=1000.0, anchor="gold"):
    """Logistic BT fit; ties enter as two half-weight rows, one per direction.

    Returns Elo-scaled ratings. With ``style=True`` the design matrix gains the
    A-minus-B style columns, so the model coefficients are read *holding style
    fixed* -- the whole point of the control.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    idx = {m: i for i, m in enumerate(models)}
    rows, ys, ws = [], [], []
    for b in battles:
        x = [0.0] * len(models)
        x[idx[b["model_a"]]] = 1.0
        x[idx[b["model_b"]]] = -1.0
        z = [b["style"][c] for c in STYLE_COLS] if style else []
        if b["winner"] == "tie":
            rows += [x + z, x + z]
            ys += [1, 0]
            ws += [0.5, 0.5]
        else:
            rows.append(x + z)
            ys.append(1 if b["winner"] == "A" else 0)
            ws.append(1.0)

    X, y, w = np.array(rows), np.array(ys), np.array(ws)
    lr = LogisticRegression(fit_intercept=False, penalty="l2", C=1.0, max_iter=2000)
    lr.fit(X, y, sample_weight=w)
    coef = lr.coef_[0]
    ratings = {m: scale * coef[idx[m]] for m in models}
    # anchor so the two fits are on a comparable scale
    base = ratings.get(anchor, 0.0)
    ratings = {m: r - base + init for m, r in ratings.items()}
    gammas = dict(zip(STYLE_COLS, coef[len(models):])) if style else {}
    return ratings, gammas


def bootstrap_ci(battles, models, style=False, rounds=100, seed=0, **kw):
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = defaultdict(list)
    for _ in range(rounds):
        sample = [battles[i] for i in rng.integers(0, len(battles), len(battles))]
        try:
            r, _ = fit_bt(sample, models, style=style, **kw)
        except Exception:  # noqa: BLE001 - a degenerate resample just gets skipped
            continue
        for m, v in r.items():
            draws[m].append(v)
    return {
        m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for m, v in draws.items()
    }


# ---------------------------------------------------------------- stages


def stage_judge(args):
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    eval_data = json.loads(Path(args.data).read_text())
    entries, items = load_entries(args.entry, eval_data, seed=args.seed)
    battles = build_battles(entries, items, n_items=args.items, seed=args.seed)
    print(f"{len(entries)} competitors ({', '.join(entries)}), {len(battles)} judgments")

    llm = LLM(
        model=args.judge_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        # The judge never sees an image. Zeroing every modality makes vLLM skip
        # loading Qwen3.5's 451M-param vision tower outright (0.90GB, left in
        # bf16 by the quant), which becomes KV cache instead.
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    tok = llm.get_tokenizer()
    prompts = []
    for b in battles:
        item = items[tuple(b["key"])]
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": judge_user(item, b["gloss_a"], b["gloss_b"])},
        ]
        # Qwen3.5's template opens a <think> block unless told otherwise, and
        # max_tokens below is nowhere near enough to close it again.
        prompts.append(
            tok.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    sp = SamplingParams(
        temperature=0.0,
        max_tokens=96,
        structured_outputs=StructuredOutputsParams(json=JUDGE_SCHEMA),
    )
    outs = llm.generate(prompts, sp)

    shell = corpus_shell_vocab(eval_data)
    with open(args.out, "w") as fh:
        for b, out in zip(battles, outs):
            try:
                verdict = json.loads(out.outputs[0].text)
            except json.JSONDecodeError:
                continue
            b["winner"] = verdict.get("winner", "tie")
            b["reason"] = verdict.get("reason", "")
            b["style"] = diff_features(
                style_features(b["gloss_a"], shell), style_features(b["gloss_b"], shell)
            )
            b["corpus"] = args.corpus
            b["judge"] = args.judge_model
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    print(f"Saved → {args.out}")


def stage_rate(args):
    battles = [json.loads(l) for l in Path(args.battles).read_text().splitlines() if l.strip()]
    models = sorted({b["model_a"] for b in battles} | {b["model_b"] for b in battles})

    # A Bradley-Terry fit assumes one judge with one bias. Rows from two judges
    # produce a number that describes neither, and the failure is silent -- the
    # fit converges and the CIs look fine. Battles written before the judge was
    # stamped carry no key at all, hence the explicit unknown.
    judges = Counter(b.get("judge", "<unstamped>") for b in battles)
    if len(judges) > 1:
        listing = ", ".join(f"{j} ({n})" for j, n in judges.most_common())
        raise SystemExit(
            f"{args.battles} mixes judgments from {len(judges)} judges: {listing}.\n"
            "Re-run --stage judge over the whole set with one judge, or split the "
            "file and rate each judge separately."
        )

    plain, _ = fit_bt(battles, models, style=False)
    controlled, gammas = fit_bt(battles, models, style=True)
    ci = bootstrap_ci(battles, models, rounds=args.bootstrap) if args.bootstrap else {}

    # order the report by the uncontrolled fit, so movement is legible
    print(f"\n{len(battles)} judgments · {len(models)} competitors\n")
    print(f"{'model':<10} {'BT elo':>8} {'95% CI':>16} {'+style ctrl':>12} {'Δ':>7}")
    print("-" * 58)
    for m in sorted(models, key=lambda m: -plain[m]):
        lo, hi = ci.get(m, (float("nan"), float("nan")))
        print(f"{m:<10} {plain[m]:>8.0f} {f'[{lo:.0f}, {hi:.0f}]':>16} "
              f"{controlled[m]:>12.0f} {controlled[m] - plain[m]:>+7.0f}")

    print("\nstyle coefficients (log-odds of winning per unit of A-minus-B feature):")
    for k, v in gammas.items():
        print(f"  {k:<10} {v:>+7.3f}")

    # position bias: the same pair was judged in both orders, so disagreement is
    # measurable directly rather than assumed away
    seen, flips, both = {}, 0, 0
    for b in battles:
        key = (tuple(b["key"]), frozenset((b["model_a"], b["model_b"])))
        if key in seen:
            both += 1
            prev = seen[key]
            wa = prev["model_a"] if prev["winner"] == "A" else (
                prev["model_b"] if prev["winner"] == "B" else "tie")
            wb = b["model_a"] if b["winner"] == "A" else (
                b["model_b"] if b["winner"] == "B" else "tie")
            flips += wa != wb
        else:
            seen[key] = b
    if both:
        print(f"\nposition bias: {flips}/{both} pairs judged differently when swapped "
              f"({flips / both:.1%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("judge", "rate"), required=True)
    ap.add_argument("--corpus", default="")
    ap.add_argument("--data", help="the eval set the candidates were generated from")
    ap.add_argument("--entry", action="append", default=[], help="name=path.json")
    ap.add_argument("--items", type=int, default=200, help="0 = all")
    ap.add_argument("--out", default="battles.jsonl")
    ap.add_argument("--battles", default="battles.jsonl")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--bootstrap", type=int, default=100, help="0 = skip CIs")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    (stage_judge if args.stage == "judge" else stage_rate)(args)


if __name__ == "__main__":
    main()
