"""Convert the definition-modelling corpora in ``data/{wordnet,oxford,slang,wiki}``
into the record shape `eval_gloss.py` reads.

Those directories hold the Ishiwatari et al. (2019) release: two line-aligned
files per split, joined by position (not by key -- ``wiki`` repeats a key across
lines, once per context).

    <split>.txt   key \\t pos \\t source \\t definition \\t [] \\t []
    <split>.eg    key \\t example with the target replaced by <TRG>

``key`` is ``<lemma>%<source-sense-id>``, e.g. ``trump%w.trump.n.02`` or
``betty_crocker%slang.0``; it is the sense identity, so it is what senses of one
lemma are grouped by. The lemma's underscores become spaces.

**POS.** Only ``wordnet`` ships a real POS (``n/v/a/s/r``); the other three write
the literal string ``pos`` in that column. Since the WiC prompt names the POS,
``--pos-source wordnet`` (default) fills the gap by looking the lemma up in
WordNet 3.0 and taking its most-attested POS -- a lemma prior, never the gold
definition, so nothing leaks into the prompt. Lemmas absent from WordNet (most
of ``slang`` and ``wiki``) fall back to ``--default-pos``.

**Sampling** is by lemma, not by row: a sampled lemma contributes *all* of its
senses, because `eval_gloss`'s ``sense_acc`` can only score lemmas that carry two
or more definitions in the file. Within a sense only ``--examples-per-sense``
contexts are kept (default 1), so the budget buys lemma coverage rather than
repeated contexts.

Examples
--------
  # all four test splits, ~1000 records each
  uv run python src/dm_gloss_data.py
  # one corpus, everything in it
  uv run python src/dm_gloss_data.py --corpus wordnet --max-records 0
"""

import argparse
import json
import random
import re
from pathlib import Path

import sense_data as sd

CORPORA = ("wordnet", "oxford", "slang", "wiki")

# wordnet's tag set -> the POS names the WiC prompt was trained on ('s' is a
# WordNet satellite adjective)
WN_POS = {"n": "noun", "v": "verb", "a": "adj", "s": "adj", "r": "adverb"}

# slang definitions/examples mark cross-references as [phrase] and elisions as
# {word}; the brackets are markup, the text inside is part of the sentence
MARKUP = re.compile(r"[\[\]{}]")


def _norm(text, strip_markup):
    text = MARKUP.sub("", text) if strip_markup else text
    return re.sub(r"\s+", " ", text).strip()


def _pos_from_wordnet():
    """``lemma -> most-attested WordNet POS`` lookup, or None when NLTK has no data."""
    try:
        from nltk.corpus import wordnet as wn

        wn.synsets("bank")  # force the corpus open now, not mid-loop
    except Exception as exc:  # noqa: BLE001 - missing corpus is a fallback, not a crash
        print(f"[warn] WordNet POS lookup unavailable ({exc.__class__.__name__}); "
              "falling back to --default-pos")
        return None

    def lookup(lemma):
        counts = {}
        for syn in wn.synsets(lemma.replace(" ", "_")):
            tag = WN_POS.get(syn.pos())
            if tag:
                counts[tag] = counts.get(tag, 0) + 1
        return max(counts, key=counts.get) if counts else None

    return lookup


def read_corpus(
    corpus,
    split="test",
    data_dir=Path("data"),
    default_pos="noun",
    pos_source="wordnet",
    max_usage_words=80,
    strip_markup=None,
):
    """One record per (sense, context) line pair, before sampling.

    Rows are dropped when the definition is empty, the context has no ``<TRG>``
    marker (nothing to tag), or the context is longer than *max_usage_words* --
    the long tail of ``slang`` runs to hundreds of words and would crowd the
    generation window.
    """
    root = data_dir / corpus
    defs = (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
    egs = (root / f"{split}.eg").read_text(encoding="utf-8").splitlines()
    if len(defs) != len(egs):
        raise ValueError(f"{corpus}: {split}.txt has {len(defs)} lines, .eg has {len(egs)}")
    if strip_markup is None:
        strip_markup = corpus == "slang"
    wn_pos = _pos_from_wordnet() if pos_source == "wordnet" else None

    recs, dropped = [], {"definition": 0, "unmarkable": 0, "too_long": 0, "misaligned": 0}
    for line_def, line_eg in zip(defs, egs):
        fields = line_def.split("\t")
        if len(fields) < 4:
            dropped["definition"] += 1
            continue
        key, pos_field, _source, definition = fields[0], fields[1], fields[2], fields[3]
        eg_key, _, example = line_eg.partition("\t")
        if eg_key != key:
            dropped["misaligned"] += 1
            continue
        definition = _norm(definition, strip_markup)
        example = _norm(example, strip_markup)
        if not definition:
            dropped["definition"] += 1
            continue
        if len(example.split()) > max_usage_words:
            dropped["too_long"] += 1
            continue

        lemma = key.split("%")[0].replace("_", " ")
        pos = WN_POS.get(pos_field) or (wn_pos(lemma) if wn_pos else None) or default_pos
        if "<TRG>" in example:
            # <TRG> stands in for the surface form, which the release does not keep,
            # so the lemma goes back in its place; only the first mention is tagged.
            usage = example.replace("<TRG>", f"<t> {lemma} </t>", 1).replace("<TRG>", lemma)
        else:
            # oxford leaves the target inline: find it with the same fuzzy matcher
            # the WiC loaders use, and drop the row when it is simply not there
            # (mark_target's fallback is to append the word, which invents a usage).
            usage = sd.mark_target(example, lemma)
            if usage.endswith(f"<t> {lemma} </t>") and not example.endswith(lemma):
                dropped["unmarkable"] += 1
                continue
        recs.append(
            {
                "lemma": lemma,
                "word": lemma,
                "pos": pos,
                "usage": usage,
                "definition": definition,
                "sense_id": key,
                "corpus": corpus,
            }
        )
    return recs, dropped


def sample_by_lemma(recs, max_records=1000, examples_per_sense=1, seed=0):
    """Keep whole lemmas (all their senses) until *max_records* rows are collected."""
    by_lemma = {}
    for rec in recs:
        by_lemma.setdefault(rec["lemma"], {}).setdefault(rec["sense_id"], []).append(rec)

    rng = random.Random(seed)
    lemmas = sorted(by_lemma)
    rng.shuffle(lemmas)

    out = []
    for lemma in lemmas:
        if max_records and len(out) >= max_records:
            break
        for sense in sorted(by_lemma[lemma]):
            rows = by_lemma[lemma][sense]
            keep = rows if examples_per_sense <= 0 else rows[:examples_per_sense]
            out.extend(keep)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", default=list(CORPORA), choices=CORPORA)
    ap.add_argument("--split", default="test", help="test | valid | train")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--max-records", type=int, default=1000, help="per corpus; 0 = all")
    ap.add_argument("--examples-per-sense", type=int, default=1, help="0 = all")
    ap.add_argument("--max-usage-words", type=int, default=80)
    ap.add_argument("--pos-source", default="wordnet", choices=("wordnet", "file"))
    ap.add_argument("--default-pos", default="noun")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for corpus in args.corpus:
        recs, dropped = read_corpus(
            corpus,
            split=args.split,
            data_dir=args.data_dir,
            default_pos=args.default_pos,
            pos_source=args.pos_source,
            max_usage_words=args.max_usage_words,
        )
        sampled = sample_by_lemma(
            recs,
            max_records=args.max_records,
            examples_per_sense=args.examples_per_sense,
            seed=args.seed,
        )
        polysemous = sum(
            1
            for lemma in {r["lemma"] for r in sampled}
            if len({r["definition"] for r in sampled if r["lemma"] == lemma}) > 1
        )
        out = args.out_dir / f"gloss_eval_{corpus}_{args.split}.json"
        out.write_text(json.dumps(sampled, ensure_ascii=False, indent=1))
        print(
            f"{corpus}: {len(recs)} usable rows (dropped {dropped}) -> "
            f"{len(sampled)} sampled over {len({r['lemma'] for r in sampled})} lemmas "
            f"({polysemous} polysemous)  ->  {out}"
        )


if __name__ == "__main__":
    main()
