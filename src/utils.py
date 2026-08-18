"""PTB detokenization and the ``<t> ... </t>`` target-marker contract.

Shared by every corpus reader that inherits parser output: ``semcor_pairs`` (SemCor
is PTB-tokenized throughout) and ``prepare_lexeme`` (``am2ico`` takes its contexts
from Wikipedia the same way). That spacing is a *tell* before it is a cost -- mixed
into one rollout batch, ``the league 's teams`` says which corpus a pair came from
before the policy has read a word of it.

The invariant everything downstream depends on is span survival: ``sense_data.
build_messages`` and every gloss scorer read the marked span, so a
detokenizer that shifts a marker by one character silently changes what the policy
is asked about. The markers are therefore held out of the attachment rules entirely
rather than treated as tokens.
"""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path

from datasets import Dataset

import sense_data as sd
from sense_rewards import KEEP_COLS

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

OPEN, CLOSE = "<t>", "</t>"

# Sentence-medial dashes and slashes take no surrounding space in normal prose.
_TIGHTEN = re.compile(r"\s*([—/])\s*")
# Two or more punctuation characters with no letters or digits: `".`, `"?`, `,"`.
# PTB never emits these -- it splits them -- so a token like this can only come from
# re-splitting text this function has already joined once.
_PUNCT_RUN = re.compile(r"[^\w\s]{2,}")


def _detok_tokens(tokens: list[str]) -> str:
    """Join PTB tokens with English spacing, tracking quote nesting.

    ``"`` is ambiguous once ``````/``''`` have been folded to it, so the
    direction is tracked by parity: an odd-numbered quote opens (attaches right),
    an even-numbered one closes (attaches left). Parity counts quote *characters*,
    not quote tokens — in text this function has already joined once, a quotation
    opens glued to its first word (``"Let's``), and reading only standalone tokens
    would leave the parity stuck and space the closing quote off.
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
        elif _PUNCT_RUN.fullmatch(tok):
            # A cluster this function itself produced. Its outer characters carry the
            # attachment: `".` closes a quotation and then ends the sentence, so it
            # glues left and nothing glues to it.
            attach_left = tok[0] in ATTACH_LEFT or (tok[0] == '"' and open_quote)
            attach_right = tok[-1] in ATTACH_RIGHT or (
                tok[-1] == '"' and not open_quote
            )
        if out and not glue_next and not attach_left:
            out.append(" ")
        out.append(tok)
        glue_next = attach_right
        if tok.count('"') % 2:
            open_quote = not open_quote
    return "".join(out)


def _tighten(text: str) -> str:
    return _TIGHTEN.sub(r"\1", text).strip()


def detokenize(text: str) -> str:
    """Undo PTB tokenization, leaving ``<t> ... </t>`` markers spaced as they were.

    The markers are held out of the attachment logic entirely: they are stripped,
    the surrounding tokens are joined as if the target were a plain word, and the
    markers are re-inserted around the identical span. So a target adjacent to
    punctuation (``<t> park </t> .``) detokenizes to ``<t> park </t>.`` and the
    ``"<t> word </t>"`` spacing convention is preserved exactly, so a SemCor sentence
    and an MCL-WiC one are token-identical in shape.

    The markers are re-padded before splitting so that this is idempotent on its own
    output, which glues the closing marker to following punctuation
    (``<t> park </t>.``); a plain whitespace split would read that as one opaque
    ``</t>.`` token and lose the target entirely. Text with unbalanced quotes is the
    residual (0.3% of SemCor usages) -- there the parity has no right answer to find.
    """
    text = text.replace(OPEN, f" {OPEN} ").replace(CLOSE, f" {CLOSE} ")
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
        return (
            f"{head} {OPEN} {target} {CLOSE} {_tighten(_detok_tokens(after))}".strip()
        )
    return f"{joined[:i]}{OPEN} {target} {CLOSE}{joined[i + len(target) :]}"


def target_of(text: str) -> str | None:
    """The marked span, or None if the markers are missing or duplicated."""
    if text.count(OPEN) != 1 or text.count(CLOSE) != 1:
        return None
    i, j = text.index(OPEN) + len(OPEN), text.index(CLOSE)
    if j < i:
        return None
    return text[i:j].strip()


def format_prompt(rec):
    out = {"prompt": sd.wic_messages(rec, with_target=False)}
    return out


def _prepare(recs, cap=None):
    ds = Dataset.from_list(recs)
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    drop = [c for c in ds.column_names if c not in KEEP_COLS]
    return ds.map(partial(format_prompt), remove_columns=drop)


def build_grpo_dataset(file_path, cap=None, with_split=False):
    file = Path(file_path)
    data = json.loads(file.read_text())
    return _prepare(data, cap=cap).train_test_split(test_size=200, seed=42)


def format_example(rec):
    """One distilled record as the SFT pair: every turn but the last, then the last."""
    msgs = sd.wic_messages(rec, with_target=True)
    return {"prompt": msgs[:-1], "completion": msgs[-1:]}


def build_sft_dataset(file_path, cap=None, strategy="longest"):
    """Teacher traces as ``{prompt, completion}`` rows, split train/test.

    ``build_grpo_dataset`` renders the prompt alone and keeps ``KEEP_COLS`` beside it,
    which is what GRPO needs: it scores its own rollouts, so the label is a reward
    input, not a target. SFT needs the teacher's assistant turn as the target
    instead, and that only exists on a record ``load_teacher_traces`` has distilled
    (teacher-correct pairs, one trace each) -- the raw file carries ``answers``/
    ``reasonings`` lists, not a single ``think``/``sense1``/``sense2``.
    """
    recs = sd.load_teacher_traces(file_path, strategy=strategy)
    ds = Dataset.from_list([format_example(r) for r in recs])
    if cap is not None:
        ds = ds.shuffle(seed=42).select(range(min(cap, len(ds))))
    return ds.train_test_split(test_size=200, seed=42)
