"""Detokenize SemCor usages and drop the ones whose sense is not readable in context.

``data/semcor_wic.json`` inherits SemCor's Penn-Treebank tokenization verbatim --
``the league 's teams``, `` `` Oh , yes . '' `` -- which no other corpus the policy
sees is written in. That spacing is a *tell*: mixed into a rollout batch with
MCL-WiC it tells the model which source a pair came from before it has read a word
of it. The second problem is editorial: SemCor annotates every content token of a
running text, so plenty of usages carry a gold synset that nothing in their own
sentence supports (``This selection rejection process takes place as the file is
read .`` -- gold ``rejection.n.01``, but the sentence would license the
computational reading just as well). A pair built from such a usage asks the policy
for a distinction the context does not contain, and the reward punishes it either
way.

Both are per-*usage* problems, so this script works on the deduplicated usage set
(33.7k unique strings behind 19.4k pairs) and only afterwards writes the result back
onto the pairs. Two stages, same shape as ``filter_reasoning.py``:

* **Stage 1 (cheap, CPU).** ``detokenize`` undoes the PTB spacing with token
  attachment rules, treating the ``<t>``/``</t>`` target markers as transparent so
  the span survives byte-exact. Rules then reject the mechanically hopeless: a
  usage with no context beyond the target, a headline/table row, a fragment with no
  final punctuation to speak of.
* **Stage 2 (LLM judge).** A local Gemma 4 12B (QAT W4A16) on vLLM sees the lemma,
  the rule-detokenized sentence and *the lemma's whole WordNet sense inventory in
  shuffled-free order*, and answers with a repaired sentence plus which sense the
  context supports. The gold synset is never shown -- that is the point. Agreement
  between the judge's pick and gold is what ``matches_gold`` scores, so a usage is
  kept only when an independent reader lands on the annotated sense unaided.

The judge is allowed to *repair* text, never to rewrite it: its sentence is accepted
only if it is alphanumerically identical to the input (``_signature``) and still
carries the exact target span. Anything else falls back to the rule detokenization,
which is why a bad rewrite costs nothing.

``matches_gold`` is the binding axis in practice -- WordNet splits senses finer than
one sentence can recover (*the fall of Balafrej*: gold "the act of surrendering",
judge "a sudden decline in importance"), so it rejects far more than fluency or
ambiguity do. That is a defensible cut for RL data and an over-cut for anything that
needs coverage, which is why it is one entry in ``--require`` and not hard-wired.

Like ``filter_reasoning.py`` this annotates rather than deletes -- ``--out`` keeps
every pair with a ``usage_quality`` verdict pair, ``--emit-filtered`` writes the
pruned corpus -- and vLLM is imported lazily so it can run in an isolated env::

    uv run python src/clean_usages.py --rules-only            # detokenize, no GPU
    uv run python src/clean_usages.py --emit-filtered data/semcor_wic_clean.json
    uv run python src/clean_usages.py --require fluent clear  # keep gold as annotated
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"

OPEN, CLOSE = "<t>", "</t>"

# --------------------------------------------------------------------------- #
# Stage 1: detokenization
# --------------------------------------------------------------------------- #
# Tokens that glue to the word on their left: PTB splits these off and we put them
# back. Clitics are listed explicitly because `` 'll `` and `` 'em `` are attached
# while a bare `` ' `` opening a quotation is not.
ATTACH_LEFT = set(".,;:!?%)]}") | {"''", "n't", "N'T", "..."}
CLITICS = {"'s", "'S", "'re", "'ve", "'ll", "'d", "'m", "'em", "'n'"}
# Tokens the *next* token glues to.
ATTACH_RIGHT = set("([{$#") | {"``"}
# PTB escapes for characters that cannot appear raw in a treebank.
PTB_ESCAPES = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
    "--": "—",
}

# Sentence-medial dashes and slashes take no surrounding space in normal prose.
_TIGHTEN = re.compile(r"\s*([—/])\s*")


def _detok_tokens(tokens: list[str]) -> str:
    """Join PTB tokens with English spacing, tracking quote nesting.

    ``"`` is ambiguous once ``````/``''`` have been folded to it, so the
    direction is tracked by parity: an odd-numbered quote opens (attaches right),
    an even-numbered one closes (attaches left).
    """
    out: list[str] = []
    glue_next = False  # previous token wants the next one flush against it
    open_quote = False
    for tok in tokens:
        # Unescape first: attachment is a property of the character the token
        # stands for, so ``-LRB-`` has to be a ``(`` before the rules see it.
        raw, tok = tok, PTB_ESCAPES.get(tok, tok)
        attach_left = tok in ATTACH_LEFT or tok in CLITICS
        attach_right = tok in ATTACH_RIGHT
        if tok == '"':
            # `` and '' disambiguate themselves; a bare " goes by parity.
            attach_right = raw == "``" or (raw == '"' and not open_quote)
            attach_left = not attach_right
            open_quote = attach_right
        if out and not glue_next and not attach_left:
            out.append(" ")
        out.append(tok)
        glue_next = attach_right
    return "".join(out)


def detokenize(text: str) -> str:
    """Undo PTB tokenization, leaving ``<t> ... </t>`` markers spaced as they were.

    The markers are held out of the attachment logic entirely: they are stripped,
    the surrounding tokens are joined as if the target were a plain word, and the
    markers are re-inserted around the identical span. So a target adjacent to
    punctuation (``<t> park </t> .``) detokenizes to ``<t> park </t>.`` and
    ``sense_data``'s ``"<t> word </t>"`` spacing convention is preserved exactly --
    a marked usage and its stripped sentence stay one edit apart.
    """
    tokens = text.split()
    span: list[str] = []
    before: list[str] = []
    after: list[str] = []
    where = before
    for tok in tokens:
        if tok == OPEN:
            where = span
            continue
        if tok == CLOSE:
            where = after
            continue
        where.append(tok)

    if not span:  # unmarked sentence: plain detokenization
        return _tighten(_detok_tokens(tokens))

    # Detokenize with the target in place, then split the result back apart on the
    # target's own detokenized form so the marker lands on the same characters.
    target = _tighten(_detok_tokens(span))
    joined = _tighten(_detok_tokens(before + span + after))
    head = _tighten(_detok_tokens(before))
    i = joined.find(target, max(0, len(head) - 1))
    if i < 0:  # attachment moved the span; fall back to naive reassembly
        return f"{head} {OPEN} {target} {CLOSE} {_tighten(_detok_tokens(after))}".strip()
    return f"{joined[:i]}{OPEN} {target} {CLOSE}{joined[i + len(target):]}"


def _tighten(text: str) -> str:
    return _TIGHTEN.sub(r"\1", text).strip()


def strip_markers(text: str) -> str:
    """The sentence without its target markers, spaced as prose."""
    return " ".join(text.replace(OPEN, "").replace(CLOSE, "").split()).strip()


def target_of(text: str) -> str | None:
    """The marked span, or None if the markers are missing or duplicated."""
    if text.count(OPEN) != 1 or text.count(CLOSE) != 1:
        return None
    i, j = text.index(OPEN) + len(OPEN), text.index(CLOSE)
    if j < i:
        return None
    return text[i:j].strip()


# --------------------------------------------------------------------------- #
# Stage 1: rejection rules
# --------------------------------------------------------------------------- #
# Context tokens (target excluded) below which a sentence cannot disambiguate
# anything -- `` `` Oh , <t> yes </t> . '' `` is 2, and no reader could name its
# sense either.
MIN_CONTEXT_WORDS = 5
# Above this share of non-alphabetic characters the "sentence" is a table row, a
# citation line or a run of figures.
MAX_NONALPHA = 0.4
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")


@dataclass
class Verdict:
    """Stage-1/2 outcome for one unique usage."""

    key: str  # the raw usage string this verdict is for
    stage: str  # "rule" | "judge"
    keep: bool
    reason: str  # rule name, or "judged" once stage 2 has spoken
    text: str = ""  # cleaned usage, markers included
    scores: dict = field(default_factory=dict)


def rule_check(usage: str) -> str | None:
    """Return the name of the rule rejecting this (detokenized) usage, or None."""
    target = target_of(usage)
    if not target:
        return "no_target"
    sentence = strip_markers(usage)
    words = _WORD.findall(sentence)
    if len(words) - len(_WORD.findall(target)) < MIN_CONTEXT_WORDS:
        return "no_context"
    alpha = sum(c.isalpha() or c.isspace() for c in sentence)
    if len(sentence) and 1 - alpha / len(sentence) > MAX_NONALPHA:
        return "not_prose"
    # An all-caps line is a heading or a table label, not a usage in running text.
    letters = [c for c in sentence if c.isalpha()]
    if len(letters) > 20 and all(c.isupper() for c in letters):
        return "all_caps"
    return None


# --------------------------------------------------------------------------- #
# Stage 2: the judge
# --------------------------------------------------------------------------- #
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "fluent": {"type": "boolean"},
        "clear": {"type": "boolean"},
        "sense_index": {"type": "integer"},
    },
    "required": ["text", "fluent", "clear", "sense_index"],
    "additionalProperties": False,
}

PROMPT = """\
You are preparing sentences for a word-sense dataset. The sentence below comes \
from a corpus that was tokenized for parsing, so its punctuation and spacing may \
be wrong. The target word is wrapped in <t> </t> tags.

Target word: {lemma} ({pos})
Sentence: {usage}

These are the dictionary senses of "{lemma}":
{senses}

Do two things.

1. Repair the sentence's formatting: fix spacing around punctuation, quotation \
marks, contractions and possessives so it reads as ordinary written English. \
Change NOTHING else -- do not add, remove, reorder or reword a single word, and \
keep the <t> </t> tags around the same word. If it already reads correctly, \
return it unchanged.

2. Judge the repaired sentence on:
- "fluent": it is a complete, well-formed sentence of running English text. false \
if it is a heading, a table row, a citation, a fragment, or garbled.
- "clear": a careful reader could tell which of the senses above the target word \
carries, using ONLY this sentence. false if the sentence would read just as well \
under two or more of the listed senses, or gives no clue at all.
- "sense_index": the 0-based index of the sense the sentence actually supports. \
Use -1 if none of them fits or you cannot tell.

Reply with ONLY a JSON object with keys "text", "fluent", "clear", "sense_index"."""


def build_prompt(usage: str, lemma: str, pos: str, senses: list[str]) -> str:
    listing = "\n".join(f"{i}. {s}" for i, s in enumerate(senses))
    return PROMPT.format(lemma=lemma, pos=pos, usage=usage, senses=listing)


_SIGNATURE = re.compile(r"[^a-z0-9]")


def _signature(text: str) -> str:
    """Lowercased alphanumerics only -- what a formatting fix must leave untouched."""
    return _SIGNATURE.sub("", text.lower())


def accept_rewrite(rewrite: str, baseline: str) -> str | None:
    """The judge's sentence if it only reformatted ``baseline``, else None.

    Two things have to hold: the alphanumeric content is unchanged (so no word was
    added, dropped or reworded), and the marked span still covers the same
    characters. Everything the judge is allowed to do -- spacing, punctuation,
    quote marks -- is invisible to both checks.

    Marker spacing is then re-imposed rather than trusted: the judge often tightens
    ``<t> early </t>`` to ``<t>early</t>``, and ``sense_data.mark_target`` renders
    the padded form, so an un-normalized rewrite would make SemCor prompts
    distinguishable from MCL-WiC ones by whitespace alone.
    """
    if not rewrite or not rewrite.strip():
        return None
    rewrite = " ".join(rewrite.split())
    old, new = target_of(baseline), target_of(rewrite)
    if new is None or old is None or _signature(new) != _signature(old):
        return None
    if _signature(rewrite) != _signature(baseline):
        return None
    i, j = rewrite.index(OPEN), rewrite.index(CLOSE) + len(CLOSE)
    return f"{rewrite[:i]}{OPEN} {new} {CLOSE}{rewrite[j:]}"


def make_sampling_params(max_tokens: int):
    """Structured-output params, tolerating the pre/post-0.10 vLLM API rename."""
    from vllm import SamplingParams

    common = dict(temperature=0.0, max_tokens=max_tokens)
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


# --------------------------------------------------------------------------- #
# Usage index
# --------------------------------------------------------------------------- #
def collect_usages(data: list[dict]) -> dict[str, dict]:
    """``raw usage`` → the context needed to judge it, deduplicated across pairs.

    A usage string recurs in many pairs (SemCor sentences are reused on both the
    same- and different-sense side), so judging it once and fanning the verdict back
    out is a ~1.7x saving on top of everything the rules already dropped.

    ``gold`` is the gold gloss for that side; it is used *only* to score the judge's
    independent pick afterwards and never enters the prompt.
    """
    index: dict[str, dict] = {}
    for rec in data:
        for side in (1, 2):
            raw = rec.get(f"usage{side}")
            if not raw:
                continue
            entry = index.setdefault(
                raw,
                {
                    "lemma": rec["lemma"],
                    "pos": rec["pos"],
                    "senses": list(rec.get("senses") or []),
                    "gold": rec.get(f"gloss{side}") or "",
                },
            )
            # A later pair may carry the gold gloss an earlier one lacked.
            if not entry["gold"]:
                entry["gold"] = rec.get(f"gloss{side}") or ""
            if len(rec.get("senses") or []) > len(entry["senses"]):
                entry["senses"] = list(rec["senses"])
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("data/semcor_wic.json"))
    ap.add_argument("--out", type=Path, default=Path("data/semcor_wic_scored.json"))
    ap.add_argument(
        "--emit-filtered",
        type=Path,
        default=None,
        help="also write a copy keeping only pairs whose two usages both passed",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    # Long enough to echo back the longest usage in the corpus (1.5k chars) plus the
    # three scalar fields.
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="usages per judge call; verdicts are checkpointed after each chunk",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/usage_verdicts.jsonl"),
        help="resumable verdict log; delete it to re-judge from scratch",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="judge only the first N surviving usages (smoke test)",
    )
    ap.add_argument(
        "--rules-only",
        action="store_true",
        help="skip the GPU judge; detokenize and apply stage-1 rules only",
    )
    ap.add_argument(
        "--require",
        nargs="+",
        default=["fluent", "clear", "matches_gold"],
        choices=["fluent", "clear", "matches_gold"],
        help="judge axes that must all hold for a usage to be kept. "
        "'matches_gold' is by far the strictest -- WordNet splits senses finer "
        "than a reader can recover from one sentence -- so drop it to keep only "
        "the formatting fix and the fluency/ambiguity filter",
    )
    ap.add_argument(
        "--show",
        type=int,
        default=5,
        help="print this many before/after examples",
    )
    args = ap.parse_args()

    data = json.loads(args.data.read_text())
    print(f"loaded {len(data)} pairs from {args.data}", file=sys.stderr)

    usages = collect_usages(data)
    print(f"{len(usages)} unique usages behind them", file=sys.stderr)

    # ---- stage 1: detokenize + rules -------------------------------------
    verdicts: dict[str, Verdict] = {}
    pending: list[str] = []
    for raw, ctx in usages.items():
        clean = detokenize(raw)
        ctx["clean"] = clean
        rule = rule_check(clean)
        if rule is not None:
            verdicts[raw] = Verdict(raw, "rule", False, rule, clean)
        else:
            pending.append(raw)

    rule_counts: dict[str, int] = {}
    for v in verdicts.values():
        rule_counts[v.reason] = rule_counts.get(v.reason, 0) + 1
    total = len(usages)
    print(f"\nstage 1 (rules): {total} usages", file=sys.stderr)
    for name, n in sorted(rule_counts.items(), key=lambda kv: -kv[1]):
        print(f"  reject {name:<14} {n:>6}", file=sys.stderr)
    print(f"  -> {len(pending)} survive to the judge", file=sys.stderr)

    if args.show:
        print("\ndetokenization samples:", file=sys.stderr)
        for raw in pending[: args.show]:
            print(f"  -  {raw}", file=sys.stderr)
            print(f"  +  {usages[raw]['clean']}", file=sys.stderr)

    # ---- stage 2: LLM judge ----------------------------------------------
    if not args.rules_only and pending:
        todo = pending[: args.limit] if args.limit else pending

        done: dict[str, dict] = {}
        if args.checkpoint.exists():
            for line in args.checkpoint.read_text().splitlines():
                if not line.strip():
                    continue
                v = json.loads(line)
                done[v["key"]] = v
            print(f"resuming: {len(done)} verdicts already checkpointed", file=sys.stderr)

        # A checkpoint may predate the current rules, so it can hold verdicts for
        # usages the rules now reject outright; replay only what is still due.
        todo_set = set(todo)
        for key, v in done.items():
            if key in todo_set:
                verdicts[key] = Verdict(
                    key, "judge", v["keep"], v["reason"], v["text"], v.get("scores", {})
                )
        todo = [k for k in todo if k not in done]

        print(f"\nstage 2 (judge): scoring {len(todo)} usages", file=sys.stderr)

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
                    # Gemma's chat template has no system role -- everything goes
                    # in the user turn.
                    convos = [
                        [
                            {
                                "role": "user",
                                "content": build_prompt(
                                    usages[k]["clean"],
                                    usages[k]["lemma"],
                                    usages[k]["pos"],
                                    usages[k]["senses"],
                                ),
                            }
                        ]
                        for k in chunk
                    ]
                    outs = llm.chat(convos, sp)

                    for key, out in zip(chunk, outs):
                        ctx = usages[key]
                        v = score_output(out.outputs[0].text, key, ctx, args.require)
                        verdicts[key] = v
                        ckpt.write(
                            json.dumps(
                                {
                                    "key": v.key,
                                    "keep": v.keep,
                                    "reason": v.reason,
                                    "text": v.text,
                                    "scores": v.scores,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    ckpt.flush()
                    n = min(start + args.chunk_size, len(todo))
                    print(f"  judged {n}/{len(todo)}", file=sys.stderr)

        # Anything past --limit stays unjudged rather than silently kept.
        for key in pending:
            if key not in verdicts:
                verdicts[key] = Verdict(
                    key, "judge", False, "not_judged", usages[key]["clean"]
                )
    else:
        for key in pending:
            verdicts[key] = Verdict(
                key, "rule", True, "passed_rules", usages[key]["clean"]
            )

    # ---- report -----------------------------------------------------------
    judged = [v for v in verdicts.values() if v.reason == "judged"]
    if judged:
        print("\njudge axis failures:", file=sys.stderr)
        for axis in ["fluent", "clear", "matches_gold"]:
            n = sum(1 for v in judged if not v.scores.get(axis))
            print(f"  {axis:<14} false: {n:>6} / {len(judged)}", file=sys.stderr)
        rejected = sum(1 for v in judged if not v.scores.get("rewrite_accepted"))
        print(
            f"  rewrite rejected (kept rule detok): {rejected} / {len(judged)}",
            file=sys.stderr,
        )

    kept = sum(1 for v in verdicts.values() if v.keep)
    print(f"\nkept {kept} / {total} usages ({kept / max(1, total):.1%})", file=sys.stderr)

    # ---- write ------------------------------------------------------------
    for rec in data:
        quality = []
        for side in (1, 2):
            v = verdicts[rec[f"usage{side}"]]
            rec[f"usage{side}"] = v.text
            rec[f"sentence{side}"] = strip_markers(v.text)
            quality.append({"keep": v.keep, "reason": v.reason, "scores": v.scores})
        rec["usage_quality"] = quality

    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}", file=sys.stderr)

    if args.emit_filtered:
        filtered = [
            {k: v for k, v in rec.items() if k != "usage_quality"}
            for rec in data
            if all(q["keep"] for q in rec["usage_quality"])
        ]
        args.emit_filtered.write_text(json.dumps(filtered, indent=2, ensure_ascii=False))
        n_same = sum(1 for r in filtered if r["label"])
        print(
            f"wrote {args.emit_filtered}: {len(filtered)} / {len(data)} pairs survive "
            f"({n_same} same / {len(filtered) - n_same} different)",
            file=sys.stderr,
        )
    return 0


def score_output(raw: str, key: str, ctx: dict, require: list[str]) -> Verdict:
    """Turn one judge completion into a verdict, falling back where it misbehaves."""
    baseline = ctx["clean"]
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        # Structured decoding should make this unreachable; without scores there is
        # nothing to vouch for the usage.
        return Verdict(key, "judge", False, "unparseable_verdict", baseline)

    rewrite = accept_rewrite(str(parsed.get("text", "")), baseline)
    text = rewrite or baseline

    senses, gold = ctx["senses"], ctx["gold"]
    try:
        idx = int(parsed.get("sense_index", -1))
    except (TypeError, ValueError):
        idx = -1
    picked = senses[idx] if 0 <= idx < len(senses) else ""
    # An unresolvable gold gloss (a WordNet adjective satellite) leaves nothing to
    # agree with, so the axis passes rather than deleting the usage.
    matches_gold = True if not gold else picked == gold

    scores = {
        "fluent": bool(parsed.get("fluent")),
        "clear": bool(parsed.get("clear")),
        "matches_gold": matches_gold,
        "sense_index": idx,
        "rewrite_accepted": rewrite is not None,
    }
    keep = all(bool(scores.get(k)) for k in require)
    return Verdict(key, "judge", keep, "judged", text, scores)


if __name__ == "__main__":
    raise SystemExit(main())
