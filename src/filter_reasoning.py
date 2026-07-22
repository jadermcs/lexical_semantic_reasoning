"""Score the distilled reasoning traces in ``data/mcl_semcor.json`` for quality.

Each record holds up to three ``reasonings`` (one per vote). Many are unusable:
some are ``null``, some were emitted in Chinese, a few degenerate into repetition
loops or truncate to a stub. The rest need a judgement call that only a model can
make -- does the trace actually argue for the gold label, and does it support the
sense JSON it was paired with?

So filtering runs in two stages:

* **Stage 1 (cheap, CPU).** Regex/statistical rules catch the mechanical failures:
  null, non-Latin script, stub, blowup, repetition loop. A vote rule then drops
  every slot that reached the wrong conclusion -- ``votes[j]`` is the
  ``same_sense`` parsed out of ``answers[j]``, so comparing it to the gold
  ``label`` settles faithfulness exactly, for free, and there is nothing for a
  judge to add. A gloss rule then drops slots whose identical
  ``sense1``/``sense2`` glosses contradict a *different* gold label (with
  ``--strict-gloss``, also differing glosses under a *same* label -- but
  string comparison mistakes paraphrases for distinct senses, so that call
  defaults to the judge). Roughly half of all slots die here and never reach
  the GPU.
* **Stage 2 (LLM judge).** Everything that survives -- i.e. only traces that
  landed on the right answer -- is scored by a local Gemma 4 12B (QAT W4A16)
  served with vLLM, on four axes chosen to match the ways these traces actually
  go wrong: ``english``, ``coherent``, ``consistent`` (trace supports its paired
  ``answers`` JSON), and ``faithful``, which now only fires when the trace's
  prose argues against the very JSON verdict it was paired with.

The script *annotates* rather than deletes: it writes every reasoning back with
its verdict and a ``keep`` flag, so the accept threshold can be retuned without
paying for the judgements again. ``--emit-filtered`` writes the pruned corpus.

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"

# Any character from a writing system these traces should never contain. The
# corpus is English-only; a single CJK codepoint means the model switched
# language mid-trace, which in practice means the whole trace is non-English.
NON_LATIN = re.compile(
    r"[぀-ヿ"  # kana
    r"㐀-䶿一-鿿"  # CJK ideographs
    r"가-힯"  # hangul
    r"Ѐ-ӿ"  # cyrillic
    r"֐-׿؀-ۿ]"  # hebrew, arabic
)

# Stage-1 thresholds. Derived from the length distribution of the corpus: the
# median trace is ~590 chars and p99 is ~2.7k, so <100 is a stub and >5000 is a
# runaway (the worst offender in the corpus is 242k chars of looped text).
MIN_CHARS = 100
MAX_CHARS = 5000
# Below this unique/total word ratio a long trace is a repetition loop.
MIN_TTR = 0.25
TTR_MIN_WORDS = 50


@dataclass
class Verdict:
    """Stage-1/2 outcome for a single reasoning slot."""

    record: int
    slot: int
    stage: str  # "rule" | "judge"
    keep: bool
    reason: str  # rule name, or "judged" once stage 2 has spoken
    scores: dict[str, bool] = field(default_factory=dict)


def rule_check(text: str | None) -> str | None:
    """Return the name of the rule that rejects ``text``, or None if it passes."""
    if text is None:
        return "null"
    stripped = text.strip()
    if not stripped:
        return "empty"
    if NON_LATIN.search(stripped):
        return "non_english"
    if len(stripped) < MIN_CHARS:
        return "too_short"
    if len(stripped) > MAX_CHARS:
        return "too_long"
    words = stripped.split()
    if len(words) >= TTR_MIN_WORDS and len(set(words)) / len(words) < MIN_TTR:
        return "repetitive"
    return None


def vote_check(rec: dict, slot: int) -> str | None:
    """Reject slot ``slot`` if its conclusion disagrees with the gold label.

    ``votes[slot]`` is the ``same_sense`` field parsed out of ``answers[slot]``,
    so a vote that differs from ``label`` means the trace argued its way to the
    wrong answer -- no judge needed. A null vote means the paired answer JSON
    never parsed, which leaves the slot with no recoverable conclusion.
    """
    votes = rec.get("votes") or []
    if slot >= len(votes) or votes[slot] is None:
        return "no_vote"
    if bool(votes[slot]) != bool(rec["label"]):
        return "wrong_prediction"
    return None


def _normalize_gloss(gloss: str) -> str:
    """Canonicalize a sense gloss so trivially-identical wordings compare equal."""
    return " ".join(gloss.lower().split()).rstrip(".")


def sense_check(rec: dict, slot: int, strict_gloss: bool = False) -> str | None:
    """Reject slot ``slot`` if its sense glosses contradict the gold label.

    Identical glosses under a *different* gold label are always a
    contradiction: the trace never actually distinguished the senses.

    The converse -- differing glosses under a *same* gold label -- only fires
    with ``strict_gloss``. String comparison cannot tell a paraphrase of one
    sense ("to kill someone by firing a gun" vs "to hit or kill someone with a
    bullet from a gun") from two genuinely different senses, so by default the
    call is left to the judge's ``consistent`` axis; strict mode demands
    verbatim-identical glosses and skews the corpus towards *different* labels.
    """
    answers = rec.get("answers") or []
    if slot >= len(answers) or answers[slot] is None:
        return None
    try:
        answer = json.loads(answers[slot])
        sense1, sense2 = answer["sense1"], answer["sense2"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None  # unparseable answers already die in vote_check
    same_gloss = _normalize_gloss(str(sense1)) == _normalize_gloss(str(sense2))
    if same_gloss and not bool(rec["label"]):
        return "same_gloss_diff_label"
    if strict_gloss and not same_gloss and bool(rec["label"]):
        return "diff_gloss_same_label"
    return None


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "english": {"type": "boolean"},
        "coherent": {"type": "boolean"},
        "faithful": {"type": "boolean"},
        "consistent": {"type": "boolean"},
    },
    "required": ["english", "coherent", "faithful", "consistent"],
    "additionalProperties": False,
}

PROMPT = """\
You are grading a reasoning trace produced by a language model for a \
word-in-context task. The task: decide whether a target word carries the SAME \
sense in two sentences.

Target word: {lemma} ({pos})
Sentence 1: {sentence1}
Sentence 2: {sentence2}

GOLD ANSWER (ground truth): same_sense = {gold}

The sense JSON the trace was paired with:
{answer}

The reasoning trace to grade:
\"\"\"
{reasoning}
\"\"\"

Grade the trace on exactly four independent criteria:

- "english": the trace is written entirely in fluent English. false if any \
other language appears, even partially.
- "coherent": the trace is a well-formed, on-topic argument about the sense of \
the target word. false if it is truncated mid-thought, repeats itself, rambles, \
or never actually discusses the target word.
- "faithful": the conclusion the trace reaches agrees with the GOLD ANSWER above. \
false if the trace concludes the senses are the same when gold says different, \
or vice versa. Grade only the conclusion, not the elegance of the argument.
- "consistent": the trace's argument actually supports the sense JSON it was \
paired with. false if the trace's glosses contradict sense1/sense2, or its \
stated conclusion contradicts the JSON's same_sense field.

Reply with ONLY a JSON object with these four boolean keys."""


def build_prompt(rec: dict, slot: int) -> str:
    answers = rec.get("answers") or []
    answer = answers[slot] if slot < len(answers) else "(none)"
    return PROMPT.format(
        lemma=rec["lemma"],
        pos=rec["pos"],
        sentence1=rec["sentence1"],
        sentence2=rec["sentence2"],
        gold=bool(rec["label"]),
        answer=answer,
        reasoning=rec["reasonings"][slot],
    )


# Gemma 4's recommended sampling settings.
# SAMPLING = dict(temperature=1.0, top_p=0.95, top_k=64)
SAMPLING = dict()


def make_sampling_params(max_tokens: int):
    """Structured-output params, tolerating the pre/post-0.10 vLLM API rename."""
    from vllm import SamplingParams

    common = dict(max_tokens=max_tokens, **SAMPLING)
    try:  # vLLM >= 0.10
        from vllm.sampling_params import StructuredOutputsParams

        return SamplingParams(
            **common,
            structured_outputs=StructuredOutputsParams(json=JUDGE_SCHEMA),
        )
    except ImportError:  # older vLLM
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            **common, guided_decoding=GuidedDecodingParams(json=JUDGE_SCHEMA)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/mcl_semcor.json"))
    ap.add_argument("--out", type=Path, default=Path("data/mcl_semcor_scored.json"))
    ap.add_argument(
        "--emit-filtered",
        type=Path,
        default=None,
        help="also write a copy with rejected reasonings stripped out",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=4096)
    # 0.92 OOMs on a 16GB card: vLLM sizes the KV cache against this fraction,
    # but the structured-output logit buffers are allocated on top of it, and a
    # desktop session is already holding a few hundred MB.
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=32,
        help="cap on concurrently decoded sequences (guards against OOM)",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="traces per judge call; verdicts are checkpointed after each chunk",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/reasoning_verdicts.jsonl"),
        help="resumable verdict log; delete it to re-judge from scratch",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge only the first N surviving traces (smoke test)",
    )
    ap.add_argument(
        "--rules-only",
        action="store_true",
        help="skip the GPU judge; apply stage-1 rules only",
    )
    ap.add_argument(
        "--strict-gloss",
        type=bool,
        default=True,
        help="also reject same-label slots whose two glosses are not verbatim "
        "identical (skews towards different-label data; off by default, the "
        "judge's 'consistent' axis makes the call instead)",
    )
    ap.add_argument(
        "--require",
        nargs="+",
        default=["english", "coherent", "faithful", "consistent"],
        choices=["english", "coherent", "faithful", "consistent"],
        help="judge axes that must all be true for a trace to be kept",
    )
    args = ap.parse_args()

    data = json.loads(args.data.read_text())
    print(f"loaded {len(data)} records from {args.data}", file=sys.stderr)

    # ---- stage 1: rules -------------------------------------------------
    verdicts: list[Verdict] = []
    pending: list[tuple[int, int]] = []  # (record, slot) awaiting the judge
    for i, rec in enumerate(data):
        for j, text in enumerate(rec.get("reasonings") or []):
            rule = (
                rule_check(text)
                or vote_check(rec, j)
                or sense_check(rec, j, args.strict_gloss)
            )
            if rule is not None:
                verdicts.append(Verdict(i, j, "rule", False, rule))
            else:
                pending.append((i, j))

    rule_counts: dict[str, int] = {}
    for v in verdicts:
        rule_counts[v.reason] = rule_counts.get(v.reason, 0) + 1
    total = len(verdicts) + len(pending)
    print(f"\nstage 1 (rules): {total} slots", file=sys.stderr)
    for name, n in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
        print(f"  reject {name:<14} {n:>6}", file=sys.stderr)
    print(f"  -> {len(pending)} survive to the judge", file=sys.stderr)

    # ---- stage 2: LLM judge ---------------------------------------------
    if not args.rules_only and pending:
        todo = pending[: args.limit] if args.limit else pending

        # Resume: replay any verdicts a previous run already checkpointed.
        done: dict[tuple[int, int], dict] = {}
        if args.checkpoint.exists():
            for line in args.checkpoint.read_text().splitlines():
                if not line.strip():
                    continue
                v = json.loads(line)
                done[(v["record"], v["slot"])] = v
            print(f"resuming: {len(done)} verdicts already checkpointed", file=sys.stderr)

        # A checkpoint may predate the current rule set, so it can hold verdicts
        # for slots the rules now reject outright; replay only what is still due.
        todo_set = set(todo)
        for (i, j), v in done.items():
            if (i, j) in todo_set:
                verdicts.append(
                    Verdict(i, j, "judge", v["keep"], v["reason"], v.get("scores", {}))
                )
        todo = [ij for ij in todo if ij not in done]

        print(f"\nstage 2 (judge): scoring {len(todo)} traces", file=sys.stderr)

        if todo:
            from vllm import LLM

            llm = LLM(
                model=args.model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_seqs=args.max_num_seqs,
            )
            sp = make_sampling_params(args.max_tokens)

            with args.checkpoint.open("a") as ckpt:
                for start in range(0, len(todo), args.chunk_size):
                    chunk = todo[start : start + args.chunk_size]
                    # Gemma's chat template has no system role -- everything
                    # goes in the user turn.
                    convos = [
                        [{"role": "user", "content": build_prompt(data[i], j)}]
                        for i, j in chunk
                    ]
                    outs = llm.chat(convos, sp)

                    for (i, j), out in zip(chunk, outs):
                        raw = out.outputs[0].text.strip()
                        try:
                            scores = json.loads(raw)
                        except json.JSONDecodeError:
                            # Structured decoding should make this unreachable;
                            # if the JSON is still broken we cannot vouch for
                            # the trace, so drop it.
                            v = Verdict(i, j, "judge", False, "unparseable_verdict")
                        else:
                            keep = all(bool(scores.get(k)) for k in args.require)
                            v = Verdict(i, j, "judge", keep, "judged", scores)
                        verdicts.append(v)
                        ckpt.write(
                            json.dumps(
                                {
                                    "record": v.record,
                                    "slot": v.slot,
                                    "keep": v.keep,
                                    "reason": v.reason,
                                    "scores": v.scores,
                                }
                            )
                            + "\n"
                        )
                    ckpt.flush()
                    n = min(start + args.chunk_size, len(todo))
                    print(f"  judged {n}/{len(todo)}", file=sys.stderr)

        # Anything past --limit stays unjudged rather than silently kept.
        judged_ids = {(v.record, v.slot) for v in verdicts if v.stage == "judge"}
        for i, j in pending:
            if (i, j) not in judged_ids:
                verdicts.append(Verdict(i, j, "judge", False, "not_judged"))
    elif pending:
        for i, j in pending:
            verdicts.append(Verdict(i, j, "rule", True, "passed_rules"))

    # ---- report + write --------------------------------------------------
    judged = [v for v in verdicts if v.stage == "judge" and v.reason == "judged"]
    if judged:
        print("\njudge axis failures:", file=sys.stderr)
        for axis in ["english", "coherent", "faithful", "consistent"]:
            n = sum(1 for v in judged if not v.scores.get(axis))
            print(f"  {axis:<12} false: {n:>6} / {len(judged)}", file=sys.stderr)

    kept = sum(1 for v in verdicts if v.keep)
    print(f"\nkept {kept} / {total} reasoning slots ({kept / total:.1%})", file=sys.stderr)

    by_slot = {(v.record, v.slot): v for v in verdicts}
    for i, rec in enumerate(data):
        quality = []
        for j in range(len(rec.get("reasonings") or [])):
            v = by_slot[(i, j)]
            quality.append({"keep": v.keep, "reason": v.reason, "scores": v.scores})
        rec["reasoning_quality"] = quality
        rec["n_good_reasonings"] = sum(q["keep"] for q in quality)

    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)

    if args.emit_filtered:
        filtered = []
        for rec in data:
            good = [
                r
                for r, q in zip(rec["reasonings"], rec["reasoning_quality"])
                if q["keep"]
            ]
            if not good:
                continue
            rec = {k: v for k, v in rec.items() if k != "reasoning_quality"}
            rec["reasonings"] = good
            filtered.append(rec)
        args.emit_filtered.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
        print(
            f"wrote {args.emit_filtered}: {len(filtered)} / {len(data)} records retain "
            f"at least one good reasoning",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
