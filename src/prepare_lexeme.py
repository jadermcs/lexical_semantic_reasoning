"""Build the multilingual WiC training set from ``data/{train,dev}/*.data``.

Four corpora ship in this layout -- ``wic``, ``mcl-wic``, ``xl`` and ``am2ico`` --
across 29 language pairs, and one of them is written differently from the other
three. ``am2ico`` (53% of the 272k pairs) takes its contexts straight out of
Wikipedia, so a context is a *paragraph*: 2.06 sentences and 85.5 Qwen3 tokens per
sentence on average, against 8.9-30.8 for the rest, with a p99 of 203 and a tail to
694. It is also PTB-tokenized (``AT & T , beIN ,``) in 99.9% of its sentences where
the other three are prose in 99.5% of theirs.

Both of those are *tells* before they are costs: mixed into one batch, context
length and punctuation spacing say which corpus a pair came from before the policy
has read a word of it -- the same problem ``utils.detokenize`` fixes for SemCor. So
this script normalizes every corpus to one shape, a single detokenized sentence:

* **Chop to the target's own sentence.** Only ``am2ico`` has anything to lose here
  (-38% tokens; -6.1% for ``xl``, ~0 for the other two), and it loses nothing that
  disambiguates: the sense annotation is a property of the sentence the target sits
  in. Measured over all 545,736 sentences the span never straddled a boundary and
  0.05% of results came back under 5 tokens.
* **Detokenize** through ``utils.detokenize``, which treats ``<t>``/``</t>``
  as transparent so the span survives byte-exact. Gated on ``_TOKENIZED``, so it is
  a no-op on corpora that were already prose rather than a normalization pass that
  quietly rewrites them.
* **Cap what is left.** 153 sentences of 545,722 stay over 256 tokens after the
  chop -- unpunctuated Georgian and Arabic runs, plus Bengali writing ``৷`` with no
  following space -- so a token window centred on the target bounds the tail. This
  is the one place a *character* budget would have been wrong: chars-per-token runs
  from 4.73 (en-en) to 1.22 (zh-zh), so a character cut costs Chinese, Korean,
  Georgian and Bengali roughly 4x what it costs English for the same model cost.

Sentence splitting is multi-script (``.``, ``。``, ``।``, ``۔``, ``৷`` ...) and
whitespace is language-aware: ``zh``/``ja`` contexts are not space-separated, and
what looks like tokenization there is only am2ico's own padding around the target.

The output carries the provenance the flat merge used to drop -- ``split``,
``source``, ``lang1``/``lang2``, ``id`` -- because ``data/train/`` holds 66
``dev.*.data`` files (25,127 pairs) next to its 32 ``train.*`` ones, and without a
split field nothing downstream can hold anything out or stratify by language. The
directory is authoritative for ``split`` (the two dev sets are disjoint by id);
``origin`` keeps the filename's own prefix.

272,440 pairs come out of 272,868, at 50/50 same/different: 128 exact duplicates,
247 labelled both ways, and 53 whose source offsets do not land on a word (52 of
them fa-fa spans that start or end on a space, one ar-ar).

    uv run python src/prepare_lexeme.py
    uv run python src/prepare_lexeme.py --max-tokens 0        # skip the token cap
    uv run python src/prepare_lexeme.py --no-chop --no-detok  # keep contexts verbatim
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import utils
from utils import CLOSE, OPEN, target_of  # noqa: F401  (re-exported)

DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
DEV_DIR = DATA_DIR / "dev"
OUT_PATH = DATA_DIR / "xl-lexeme.json"

DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_TOKENS = 192

# --------------------------------------------------------------------------- #
# Sentence splitting
# --------------------------------------------------------------------------- #
# Terminators actually observed across the 29 language pairs: Latin/Cyrillic/
# Georgian `.!?`, CJK fullwidth `。！？`, Devanagari/Bengali danda `।॥৷`, Urdu `۔`
# and Arabic `؟`. Runs are kept together so `...` and `?!` end one sentence.
TERMINATORS = ".!?。！？।॥৷۔؟"
_TERM_RUN = rf"[{re.escape(TERMINATORS)}]+"
# A boundary is a terminator run followed by whitespace -- except for the scripts
# that do not space their punctuation, where the run alone ends the sentence.
# `৷` in particular is written flush against the word before *and* after it.
_NO_SPACE_TERMS = "。！？।॥৷"
_BOUNDARY = re.compile(
    rf"({_TERM_RUN})(\s+)|([{re.escape(_NO_SPACE_TERMS)}]+)()"
)

# Scripts written without spaces between words. Korean is *not* one of them: it
# spaces its eojeol, so its am2ico contexts are ordinary tokenized text.
_UNSPACED = re.compile(
    "["
    "\u3000-\u303f"  # CJK punctuation
    "\u3040-\u30ff\u31f0-\u31ff"  # kana
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # han
    "\uff00-\uffef"  # fullwidth forms
    "]"
)
UNSPACED_LANGS = {"zh", "ja"}

# PTB spacing signals: a space before closing punctuation, after an opening
# bracket, or in front of a split English clitic. Detokenization only runs on
# sentences that show one, so prose corpora pass through untouched.
#
# Deliberately narrow on which punctuation counts: French writes ``Bonjour !`` and
# German writes ``50 %``, so a space before ``!;:?%`` is that language's typography
# rather than tokenizer output, and triggering on it would rewrite 96k fr-fr
# sentences into English spacing. Nothing spaces off a comma, period or closing
# bracket on purpose.
_TOKENIZED = re.compile(
    r"\s[,.)\]}]|[(\[{]\s|\s'(?:s|re|ve|ll|d|m|em)\b|\sn't\b"
    r"|\s``|''\s|\s--\s|-[LR][RSC]B-"
)
# Punctuation that attaches left in scripts `utils` never had to see.
_NON_ENGLISH_LEFT = re.compile(r"\s+([،؛؟۔।॥৷·])")

POS_MAP = {
    "N": "NOUN",
    "V": "VERB",
    "A": "ADJ",
    "R": "ADV",
    "NOUN": "NOUN",
    "VERB": "VERB",
    "ADJ": "ADJ",
    "ADV": "ADV",
}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Char spans of the sentence-ish units in *text*, terminators included."""
    spans, prev = [], 0
    for m in _BOUNDARY.finditer(text):
        end = m.end(1) if m.group(1) is not None else m.end(3)
        if end <= prev:  # a boundary inside one we already consumed
            continue
        if not _is_boundary(text, end, m.end()):
            continue
        spans.append((prev, end))
        prev = m.end()
    spans.append((prev, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def _is_boundary(text: str, end: int, resume: int) -> bool:
    """Whether the terminator run ending at *end* really closes a sentence.

    Two false positives matter here, and both only ever cost context -- an
    over-split trims the sentence the target needs. A period is not a boundary
    when the token it closes is an initialism (``J.P.``) or a one-letter initial
    (``J.``), and no terminator is one when the next sentence would open in
    lower case (``et al. the same year``). Caseless scripts are unaffected:
    ``str.islower`` is False for every Arabic, Georgian and CJK letter.
    """
    if text[end - 1] == "." and (end < 2 or text[end - 2] != "."):
        token = text[: end - 1].rsplit(maxsplit=1)
        head = token[-1] if token else ""
        if "." in head or len(head) == 1 and head.isalpha():
            return False
    return not (resume < len(text) and text[resume].islower())


def chop_to_sentence(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Keep only the sentence containing ``[start, end)``, re-basing the offsets.

    A span that straddles a boundary (0.00% of the corpus, but a split on an
    abbreviation would make one) keeps the whole context rather than losing half
    its target.
    """
    for a, b in sentence_spans(text):
        if a <= start and end <= b:
            kept = text[a:b].strip()
            lead = len(text[a:b]) - len(text[a:b].lstrip())
            return kept, start - a - lead, end - a - lead
    return text, start, end


# --------------------------------------------------------------------------- #
# Whitespace
# --------------------------------------------------------------------------- #
def normalize_whitespace(
    text: str, start: int, end: int, unspaced: bool
) -> tuple[str, int, int]:
    """Collapse whitespace runs, re-basing the target offsets onto the result.

    For ``zh``/``ja`` a run between two unspaced characters is deleted outright:
    those contexts have no word spaces of their own, so every space in them is
    am2ico's padding around the target (``在  内阁 外`` -> ``在内阁外``) and keeping
    it would leave the target visibly bracketed once the markers came off.
    """
    out: list[str] = []
    # where each original index lands in the result. Only `start` is read off it
    # directly: `end` is exclusive, so it is derived from the span's last
    # character instead -- reading `moved[end]` would swallow the space that
    # follows the target, which is where most targets sit.
    moved = [0] * (len(text) + 1)
    at = 0  # length of `out` so far: the offset the next character lands on
    i, n = 0, len(text)
    while i < n:
        if not text[i].isspace():
            moved[i] = at
            out.append(text[i])
            at += 1
            i += 1
            continue
        j = i
        while j < n and text[j].isspace():
            j += 1
        drop = unspaced and _flanked_unspaced(text, i, j)
        if out and j < n and not drop:
            out.append(" ")
            at += 1
        # a target never starts inside whitespace, but if the source says it does,
        # land it on the first character after the run
        for k in range(i, j):
            moved[k] = at
        i = j
    moved[n] = at
    if end <= start:
        return "".join(out), moved[start], moved[start]
    return "".join(out), moved[start], moved[end - 1] + 1


def _flanked_unspaced(text: str, i: int, j: int) -> bool:
    before = text[i - 1] if i else ""
    after = text[j] if j < len(text) else ""
    return bool(before and after and _UNSPACED.match(before) and _UNSPACED.match(after))


# --------------------------------------------------------------------------- #
# Marking and detokenization
# --------------------------------------------------------------------------- #
def mark_target(sentence: str, start: int, end: int) -> str:
    """Wrap ``[start, end)`` in ``<t> ... </t>``, spaced as ``sense_data`` writes it."""
    return f"{sentence[:start]}{OPEN} {sentence[start:end]} {CLOSE}{sentence[end:]}"


def detokenize(marked: str, lang: str) -> str:
    """Undo tokenizer spacing in a marked sentence, or return it unchanged.

    ``zh``/``ja`` are skipped outright -- ``utils.detokenize`` splits on whitespace, and
    an unspaced context is one token to it, so there is nothing for it to join and
    a space it invented would be a new artifact. Everything else goes through the
    shared detokenizer only if it shows a PTB signal, then a script-specific pass
    closes up the punctuation English never uses.
    """
    if lang in UNSPACED_LANGS:
        return _NON_ENGLISH_LEFT.sub(r"\1", marked)
    if not _TOKENIZED.search(marked):
        return _NON_ENGLISH_LEFT.sub(r"\1", marked)
    return _NON_ENGLISH_LEFT.sub(r"\1", utils.detokenize(marked))


# --------------------------------------------------------------------------- #
# The token cap
# --------------------------------------------------------------------------- #
def token_window(
    text: str, start: int, end: int, budget: int, tokenizer
) -> tuple[str, int, int]:
    """Trim *text* to ~*budget* tokens centred on the target, on token boundaries.

    The target span is never cut: a target that is itself over budget comes back
    whole, since a truncated target is not a WiC item at all.
    """
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    if len(offsets) <= budget:
        return text, start, end
    inside = [k for k, (a, b) in enumerate(offsets) if a < end and b > start]
    if not inside:
        return text, start, end
    first, last = inside[0], inside[-1]
    spare = budget - (last - first + 1)
    if spare <= 0:  # the target alone fills the budget; keep it whole
        lo, hi = first, last
    else:
        lo = max(0, first - spare // 2)
        hi = min(len(offsets) - 1, last + spare - (first - lo))
        lo = max(0, first - (spare - (hi - last)))  # spend what the right end could not
    a, b = offsets[lo][0], offsets[hi][1]
    return text[a:b], start - a, end - a


def _load_tokenizer(name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def parse_name(path: Path) -> tuple[str, str, str, str]:
    """``dev.ka-en.am2ico.data`` -> ``("dev", "ka", "en", "am2ico")``."""
    parts = path.name[: -len(".data")].split(".")
    origin, langs, source = parts[0], parts[1], ".".join(parts[2:])
    lang1, _, lang2 = langs.partition("-")
    return origin, lang1, lang2 or lang1, source


def normalize_pos(pos) -> str | None:
    if pos is None:
        return None
    key = str(pos).strip().upper()
    if key in {"", "NONE", "X"}:
        return None
    return POS_MAP.get(key, key)


def build_record(example, label, split, origin, lang1, lang2, source, chop):
    """One prepared pair, or None if either side lost its target span."""
    sides = {}
    for side, lang in (("1", lang1), ("2", lang2)):
        text = example["sentence" + side]
        start, end = int(example["start" + side]), int(example["end" + side])
        gold = text[start:end]
        if not gold.strip():
            return None
        unspaced = lang in UNSPACED_LANGS
        if chop:
            text, start, end = chop_to_sentence(text, start, end)
        text, start, end = normalize_whitespace(text, start, end, unspaced)
        if text[start:end].strip() != gold.strip():
            return None
        sides[side] = (text, start, end, lang)
    return {
        "task": "wic",
        "id": example["id"],
        "split": split,
        "origin": origin,
        "source": source,
        "lang1": lang1,
        "lang2": lang2,
        "lemma": example["lemma"],
        "pos": normalize_pos(example.get("pos")),
        "label": label,
        "_sides": sides,
    }


def finish_record(rec, detok: bool) -> dict:
    """Mark, detokenize and drop the working state, in place."""
    sides = rec.pop("_sides")
    for side, (text, start, end, lang) in sides.items():
        marked = mark_target(text, start, end)
        if detok:
            marked = detokenize(marked, lang)
        rec["sentence" + side] = marked
    return rec


def pair_key(rec) -> tuple:
    """Identity of a pair, keyed on the marked sentences.

    Marking is what makes this key correct, and it is why records store the marked
    sentence rather than a marked/unmarked pair: these corpora annotate a target
    *occurrence*, so one sentence pair can carry several items differing only in
    which occurrence is marked, and those legitimately disagree on the label. A key
    over bare sentences reads 21,902 of them as contradictions and deletes every copy.
    """
    return (rec["lemma"], rec["pos"], rec["sentence1"], rec["sentence2"])


def dedupe(records) -> tuple[list[dict], int, int]:
    """Drop repeats of the same pair; drop *every* copy of one labelled both ways."""
    labels = defaultdict(set)
    for rec in records:
        labels[pair_key(rec)].add(rec["label"])
    kept, seen, dropped_dup, dropped_conflict = [], set(), 0, 0
    for rec in records:
        key = pair_key(rec)
        if len(labels[key]) > 1:
            dropped_conflict += 1
            continue
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        kept.append(rec)
    return kept, dropped_dup, dropped_conflict


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    ap.add_argument("--dev-dir", type=Path, default=DEV_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="per-sentence token cap; 0 disables it (and the tokenizer load)",
    )
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--no-chop", action="store_true", help="keep whole contexts")
    ap.add_argument("--no-detok", action="store_true", help="keep tokenizer spacing")
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    files = [(args.train_dir, "train"), (args.dev_dir, "dev")]
    records, skipped = [], Counter()
    for directory, split in files:
        for path in sorted(directory.glob("*.data")):
            origin, lang1, lang2, source = parse_name(path)
            gold_path = path.with_suffix(".gold")
            labels = {
                entry["id"]: entry["tag"] == "T"
                for entry in json.loads(gold_path.read_text())
            }
            for example in json.loads(path.read_text()):
                if example["id"] not in labels:
                    skipped["no gold label"] += 1
                    continue
                rec = build_record(
                    example,
                    labels[example["id"]],
                    split,
                    origin,
                    lang1,
                    lang2,
                    source,
                    chop=not args.no_chop,
                )
                if rec is None:
                    skipped["lost target span"] += 1
                    continue
                records.append(rec)

    capped = 0
    if args.max_tokens > 0:
        tokenizer = _load_tokenizer(args.tokenizer)
        flat = [(rec, side) for rec in records for side in ("1", "2")]
        lengths = _batch_lengths(
            tokenizer, [rec["_sides"][side][0] for rec, side in flat]
        )
        for (rec, side), length in zip(flat, lengths):
            if length <= args.max_tokens:
                continue
            text, start, end, lang = rec["_sides"][side]
            gold = text[start:end]
            new = token_window(text, start, end, args.max_tokens, tokenizer)
            if new[0][new[1] : new[2]] != gold:  # never trade the span for the budget
                continue
            rec["_sides"][side] = (*new, lang)
            capped += 1

    records = [finish_record(rec, detok=not args.no_detok) for rec in records]

    dropped_dup = dropped_conflict = 0
    if not args.no_dedupe:
        records, dropped_dup, dropped_conflict = dedupe(records)

    args.out.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    _report(records, skipped, capped, dropped_dup, dropped_conflict, args.out)


def _batch_lengths(tokenizer, texts, batch=4096) -> list[int]:
    out: list[int] = []
    for i in range(0, len(texts), batch):
        enc = tokenizer(texts[i : i + batch], add_special_tokens=False)["input_ids"]
        out.extend(len(ids) for ids in enc)
    return out


def _report(records, skipped, capped, dropped_dup, dropped_conflict, out) -> None:
    by_split = Counter(r["split"] for r in records)
    by_source = Counter(r["source"] for r in records)
    by_lang = Counter(f"{r['lang1']}-{r['lang2']}" for r in records)
    print(f"wrote {len(records):,} pairs to {out}")
    print("  splits :", dict(sorted(by_split.items())))
    print("  sources:", dict(sorted(by_source.items())))
    print(f"  langs  : {len(by_lang)} pairs, top {by_lang.most_common(5)}")
    print(f"  labels : {Counter(r['label'] for r in records)}")
    if capped:
        print(f"  capped : {capped:,} sentences hit the token budget")
    if dropped_dup or dropped_conflict:
        print(
            f"  dropped: {dropped_dup:,} duplicate pairs, "
            f"{dropped_conflict:,} with conflicting labels"
        )
    for reason, count in skipped.items():
        print(f"  skipped: {count:,} ({reason})")


if __name__ == "__main__":
    main()
