"""Evaluate *gloss generation* quality against gold definitions.

`eval_sense.py` scores the only verifiable thing the policy emits — the
same/different label. The glosses ride along unsupervised: nothing in the
pipeline ever checks that `sense1` is a *good* definition of the usage in front
of it. This script does, against a reference definition supplied per example.

Input is a JSON list of definition-modelling records::

    [{"lemma": "bank", "word": "banks", "pos": "noun",
      "usage": "The river banks were flooded.",
      "definition": "sloping land beside a body of water"}, ...]

``lemma``, ``word``, ``usage`` and ``definition`` are required; ``pos`` is
optional (falls back to ``--default-pos``). ``word`` is the surface form as it
appears in ``usage`` — that is what gets wrapped in ``<t>`` tags, exactly like
MCL-WiC's ``word1``/``word2``.

**The prompt adaptation.** The policy was never trained to define a word in
isolation: it is trained on *pairs*, and its answer schema has two gloss slots.
So each evaluation example is rendered as sentence 1 of an ordinary WiC prompt,
and sentence 2 is a **constant filler** — "bank" in its financial-institution
sense (``--filler-usage``). Only ``sense1`` is scored; ``sense2`` and
``same_sense`` are decoration, and the prompt stays exactly on-distribution.
Because the filler is constant, ``filler_echo_rate`` doubles as a leak detector:
a model that echoes the filler's gloss into ``sense1`` is not reading sentence 1
at all. (Records whose gold sense genuinely *is* the filler's are left out of
that rate -- there, agreement is the right answer, not a leak.)

Metrics, all generated-gloss vs. gold definition:

  ``f1``            content-token F1, via the same tokens/stoplist
                    `gloss_wordnet` snaps with, so this number is comparable to
                    the verifier's and to `analysis/gloss_quality.py`.
  ``bleu``/``chrf`` sentence-level means plus a corpus-level BLEU. Sentence BLEU
                    on ~10-token definitions is very noisy; treat the gaps
                    between checkpoints as the signal, never the absolute value.
                    Needs `sacrebleu` (see the invocation below); skipped, not
                    fatal, when it is missing.
  ``cos``           cosine similarity of sentence-transformer embeddings — the
                    only metric here that survives full paraphrase. Disable with
                    ``--no-sbert``.
  ``sense_acc``     *discriminativeness*: among all gold definitions sharing this
                    record's (lemma, pos), is the record's own definition the
                    nearest to the generated gloss? Restricted to lemmas with
                    2+ senses in the file, so it is only reported when the input
                    actually contains polysemy. BLEU cannot see this: a gloss
                    can be definition-shaped, score well, and still describe the
                    wrong sense.

Examples
--------
  uv run --with sacrebleu python src/eval_gloss.py \\
      --model ./qwen-sense-grpo-wic --data data/gloss_eval.json
  # unmerged LoRA on top of its base
  uv run --with sacrebleu python src/eval_gloss.py --model ./qwen-sft_wic_filtered \\
      --lora ./qwen-lora-sdpo-wic/checkpoint-500 --data data/gloss_eval.json
  # re-score a saved run on CPU (no GPU, no vLLM)
  uv run --with sacrebleu python src/eval_gloss.py --score-only gloss_eval_out.json
"""

import argparse
import json
import statistics
from pathlib import Path

import gloss_wordnet as gw
import sense_data as sd

# Sentence 2 of every prompt: a constant, unambiguous usage the model has no
# trouble glossing. It exists to put the prompt back in the two-sentence shape
# the policy was trained on -- its gloss is never scored.
FILLER_WORD = "bank"
FILLER_USAGE = "She deposited her paycheck at the bank on Friday morning."

REQUIRED_KEYS = ("lemma", "word", "usage", "definition")

# Two glosses this close are the same string in all but wording; used both for
# the filler-leak check and for reporting.
SAME_GLOSS_F1 = 0.8
# ... and a record whose gold definition is even this close to the filler's sense
# is dropped from the leak rate. Deliberately generous: an echo that happens to
# be correct is unprovable either way, and the rate is a diagnostic, not a score,
# so a missed leak costs less than a false accusation.
ECHO_EXCLUDE_F1 = 0.4


def load_gloss_records(path, default_pos="noun"):
    """Read the eval file, marking the target word in each usage."""
    raw = json.loads(Path(path).read_text())
    recs = []
    for i, r in enumerate(raw):
        missing = [k for k in REQUIRED_KEYS if not str(r.get(k, "")).strip()]
        if missing:
            raise ValueError(f"record {i}: missing/empty {', '.join(missing)}")
        usage = r["usage"]
        rec = {
            "lemma": r["lemma"],
            "word": r["word"],
            "pos": r.get("pos") or default_pos,
            "usage": usage,
            "definition": r["definition"],
            # already-tagged inputs are left alone, so a curated file can
            # override the fuzzy matcher's choice of span
            "marked": usage if "<t>" in usage else sd.mark_target(usage, r["word"]),
        }
        # Optional second usage of the *same* target word. When present it
        # replaces the constant filler in sentence 2, which is the shape the
        # policy was actually trained on: two real contexts to triangulate from.
        # `definition2` lets the second usage carry its own gold (a contrast
        # pair, where the two usages are different senses).
        if str(r.get("usage2", "")).strip():
            u2 = r["usage2"]
            rec["usage2"] = u2
            rec["marked2"] = u2 if "<t>" in u2 else sd.mark_target(u2, r.get("word2") or r["word"])
        if r.get("definition2"):
            rec["definition2"] = r["definition2"]
        # the lemma's full sense inventory, when the source knows it -- lets the
        # gloss be ranked against real rival senses instead of whatever other
        # senses happen to be in this file
        if r.get("senses"):
            rec["senses"] = list(r["senses"])
        recs.append(rec)
    return recs


def build_wic_rec(rec, filler_usage=FILLER_USAGE, filler_word=FILLER_WORD):
    """Cast one definition-modelling record into a WiC record (see module docstring).

    The target-word line names the record's own lemma, because sentence 1 is the
    only half being scored; the filler occupies sentence 2 unannounced, which is
    what keeps the surface form of the prompt identical to training.

    When the record carries its own ``marked2`` -- a second real usage of the same
    target word -- that takes sentence 2 instead. Nothing else about the prompt
    changes, so the two conditions are a clean ablation on the second sentence.
    """
    second = rec.get("marked2")
    if not second:
        second = (
            filler_usage
            if "<t>" in filler_usage
            else sd.mark_target(filler_usage, filler_word)
        )
    return {
        "task": "wic",
        "lemma": rec["lemma"],
        "pos": rec["pos"],
        "usage1": rec["marked"],
        "usage2": second,
    }


def build_prompt(rec, tokenizer, **filler):
    msgs = sd.wic_messages(build_wic_rec(rec, **filler), with_target=False)
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def token_f1(a, b):
    """Content-token F1 -- `gloss_wordnet`'s, so every report in the repo agrees."""
    return gw._f1(gw._toks(a), gw._toks(b))


def parse_glosses(decoded):
    """(gloss for sentence 1, gloss for the filler, think) from one completion."""
    obj = sd.parse_wic_answer(decoded) or {}
    think, closed, _ = decoded.partition("</think>")
    return (
        str(obj.get("sense1", "")).strip() if isinstance(obj, dict) else "",
        str(obj.get("sense2", "")).strip() if isinstance(obj, dict) else "",
        think.replace("<think>", "").strip() if closed else "",
    )


def parse_verdict(decoded):
    """The ``same_sense`` boolean, or None when the answer did not parse.

    Only meaningful in the paired conditions, where sentence 2 is a real usage:
    there the verdict is the policy's actual trained task, and it is verifiable
    against whether the pair was built same-sense or contrast.
    """
    obj = sd.parse_wic_answer(decoded) or {}
    val = obj.get("same_sense") if isinstance(obj, dict) else None
    return bool(val) if isinstance(val, bool) else None


def inventory_metrics(rows, sim):
    """Rank each gloss against the lemma's *real* sense inventory.

    `sense_accuracy` can only compare against whatever other senses happen to
    share a lemma inside the eval file. When the source ships ``senses`` -- the
    lemma's full WordNet inventory -- the rivals are the actual competing senses,
    which is a much stricter test. ``margin`` mirrors ``reward_wic_gloss``:
    similarity to gold *minus* the best rival, so a gloss that is vaguely
    definition-shaped scores near zero rather than positive.
    """
    scored = [
        r
        for r in rows
        if r.get("gloss") and r.get("senses") and r["definition"] in r["senses"]
    ]
    n = hit = 0
    margins = []
    for row in scored:
        rivals = [s for s in row["senses"] if s != row["definition"]]
        if not rivals:
            continue
        n += 1
        gold_sim = sim(row["gloss"], row["definition"])
        best_rival = max(sim(row["gloss"], s) for s in rivals)
        row["inv_correct"] = gold_sim > best_rival
        row["inv_margin"] = gold_sim - best_rival
        hit += row["inv_correct"]
        margins.append(row["inv_margin"])
    return {
        "inv_acc": (hit / n) if n else float("nan"),
        "inv_margin": _mean(margins),
        "n_inv_scored": n,
    }


def paired_metrics(rows):
    """Metrics that only exist when sentence 2 is a real usage.

    ``agree`` is the token-F1 between the policy's two emitted glosses. In the
    filler condition that same quantity is a *leak* detector; here it is the
    opposite -- two usages of one sense should be glossed alike -- so it is
    reported under its own name to keep the two readings from being confused.
    """
    paired = [r for r in rows if r.get("usage2")]
    if not paired:
        return {}
    scored = [r for r in paired if r.get("gloss")]
    for row in scored:
        g2 = row.get("filler_gloss", "")
        row["agree"] = token_f1(row["gloss"], g2) if g2 else float("nan")
        gold2 = row.get("definition2") or row["definition"]
        row["gloss2_f1"] = token_f1(g2, gold2) if g2 else float("nan")
    verdicts = [r for r in paired if r.get("same_pred") is not None and "same_gold" in r]
    return {
        "n_paired": len(paired),
        "agree": _mean([r["agree"] for r in scored if r.get("agree") == r.get("agree")]),
        "gloss2_f1": _mean(
            [r["gloss2_f1"] for r in scored if r.get("gloss2_f1") == r.get("gloss2_f1")]
        ),
        # the policy's own trained verdict, scored against how the pair was built
        "verdict_acc": _mean([float(r["same_pred"] == r["same_gold"]) for r in verdicts]),
        "n_verdict": len(verdicts),
    }


class _Sbert:
    """Lazy sentence-transformer, kept off the GPU vLLM is holding by default."""

    def __init__(self, model, device="cpu"):
        from sentence_transformers import SentenceTransformer

        self.m = SentenceTransformer(model, device=device)

    def encode(self, texts):
        return self.m.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )

    def cos(self, a, b):
        """Row-wise cosine of two equal-length text lists (embeddings are unit-norm)."""
        if not a:
            return []
        ea, eb = self.encode(a), self.encode(b)
        return [float((x * y).sum()) for x, y in zip(ea, eb)]


def _bertscore(hyps, refs, model=None, device="cuda", rescale=True):
    """BERTScore F1, the metric the definition-modelling literature reports.

    This is what makes our numbers comparable to a published card: FLAN-T5's
    reports BERT-F1 92.16 (wordnet) / 89.75 (oxford), and the unrescaled mean
    here reproduces both. ``rescale`` additionally returns the baseline-rescaled
    score, because raw BERTScore is badly compressed -- on these corpora the
    whole span from "definition of an unrelated word" to the best model is about
    0.84 to 0.93, which makes raw differences look far smaller than they are.

    Returns ``(raw_f1_list, rescaled_f1_list)``; ``(None, None)`` if unavailable.
    """
    try:
        from bert_score import BERTScorer
    except ImportError:
        print("[warn] bert-score not installed - skipping BERTScore "
              "(uv pip install --no-deps bert-score, then run with --no-sync)")
        return None, None
    kw = dict(lang="en", device=device, batch_size=64)
    raw = BERTScorer(rescale_with_baseline=False, **kw).score(hyps, refs, verbose=False)[2]
    if not rescale:
        return [float(x) for x in raw], None
    res = BERTScorer(rescale_with_baseline=True, **kw).score(hyps, refs, verbose=False)[2]
    return [float(x) for x in raw], [float(x) for x in res]


def _bleu_chrf():
    """sacrebleu scorers, or (None, None) when it is not installed."""
    try:
        from sacrebleu.metrics import BLEU, CHRF
    except ImportError:
        print("[warn] sacrebleu not installed - skipping BLEU/chrF "
              "(re-run with: uv run --with sacrebleu python src/eval_gloss.py ...)")
        return None, None
    # effective_order: short references must not zero out on a missing 4-gram
    return BLEU(effective_order=True), CHRF()


def sense_accuracy(rows, sim):
    """Is each gloss nearest to its *own* gold definition among same-lemma senses?

    Only lemmas carrying 2+ distinct definitions in the file are scored -- with a
    single sense there is nothing to discriminate. ``sim(gloss, definition)`` is
    SBERT cosine when available, content-token F1 otherwise.
    """
    groups = {}
    for row in rows:
        groups.setdefault((row["lemma"], row["pos"]), []).append(row)
    n = hit = 0
    for group in groups.values():
        defs = list(dict.fromkeys(r["definition"] for r in group))
        if len(defs) < 2:
            continue
        for row in group:
            if not row["gloss"]:
                continue
            n += 1
            scores = [sim(row["gloss"], d) for d in defs]
            best = defs[max(range(len(defs)), key=lambda i: scores[i])]
            hit += best == row["definition"]
            row["sense_correct"] = best == row["definition"]
    return {"sense_acc": (hit / n) if n else float("nan"), "n_sense_scored": n}


def _mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def score_rows(rows, sbert=None, bertscore=False, bertscore_device="cuda"):
    """Attach per-row scores and roll them up. Pure CPU; no vLLM in sight."""
    bleu, chrf = _bleu_chrf()
    for row in rows:  # tolerate hand-written / older files under --score-only
        row.setdefault("gloss", "")
        row.setdefault("filler_gloss", "")
    scored = [r for r in rows if r["gloss"]]
    hyps = [r["gloss"] for r in scored]
    refs = [r["definition"] for r in scored]

    for row in scored:
        row["f1"] = token_f1(row["gloss"], row["definition"])
        # the filler is constant, so this is a leak check, not a quality score
        row["filler_f1"] = token_f1(row["gloss"], row["filler_gloss"])
        if bleu is not None:
            row["bleu"] = bleu.sentence_score(row["gloss"], [row["definition"]]).score
            row["chrf"] = chrf.sentence_score(row["gloss"], [row["definition"]]).score
    if sbert is not None:
        for row, c in zip(scored, sbert.cos(hyps, refs)):
            row["cos"] = c
    if bertscore and scored:
        b_raw, b_res = _bertscore(hyps, refs, device=bertscore_device)
        if b_raw is not None:
            for i, row in enumerate(scored):
                row["bert_f1"] = b_raw[i]
                if b_res is not None:
                    row["bert_f1_rescaled"] = b_res[i]

    if sbert is None:
        sim = token_f1
    else:
        # one encode pass over every distinct string, then sim is a dot product.
        # The sense inventories have to be in here too: `inventory_metrics` calls
        # sim against rival senses, which are not otherwise any row's gloss.
        vocab = list(
            {
                t
                for r in scored
                for t in (r["gloss"], r["definition"], *r.get("senses", ()))
            }
        )
        vecs = dict(zip(vocab, sbert.encode(vocab)))
        sim = lambda g, d: float((vecs[g] * vecs[d]).sum())  # noqa: E731
    echoes = [
        float(r["filler_f1"] >= SAME_GLOSS_F1)
        for r in scored
        if token_f1(r["definition"], r["filler_gloss"]) < ECHO_EXCLUDE_F1
    ]
    summary = {
        "n": len(rows),
        "n_scored": len(scored),
        "empty": len(rows) - len(scored),
        "f1": _mean([r["f1"] for r in scored]),
        "cos": _mean([r["cos"] for r in scored if "cos" in r]),
        "bleu": _mean([r["bleu"] for r in scored if "bleu" in r]),
        "chrf": _mean([r["chrf"] for r in scored if "chrf" in r]),
        "bleu_corpus": bleu.corpus_score(hyps, [refs]).score
        if bleu is not None and hyps
        else float("nan"),
        "bert_f1": _mean([r["bert_f1"] for r in scored if "bert_f1" in r]),
        "bert_f1_rescaled": _mean(
            [r["bert_f1_rescaled"] for r in scored if "bert_f1_rescaled" in r]
        ),
        "len_gloss": _mean([len(gw._toks(r["gloss"])) for r in scored]),
        "len_gold": _mean([len(gw._toks(r["definition"])) for r in scored]),
        # sentence 2 is the same sentence every time, so a sense1 that matches
        # its gloss is the model echoing the filler instead of reading sentence
        # 1 -- except where the record's own sense really is the filler's, which
        # is why those rows are excluded rather than counted as failures
        "filler_f1": _mean([r["filler_f1"] for r in scored]),
        "filler_echo_rate": _mean(echoes),
        "n_echo_scored": len(echoes),
    }
    summary.update(sense_accuracy(scored, sim))
    summary.update(inventory_metrics(scored, sim))
    summary.update(paired_metrics(rows))
    return summary


def print_summary(summary, rows, k=10):
    # the filler line only means anything when sentence 2 *is* the constant
    # filler; in a paired run it would read as an echo rate against a real usage
    filler = (
        ""
        if summary.get("n_paired")
        else (
            f"  filler_f1={summary['filler_f1']:.3f}"
            f"  filler_echo={summary['filler_echo_rate']:.3f}"
            f" (n={summary['n_echo_scored']})"
        )
    )
    # BERTScore is the metric the definition-modelling papers report, so it leads
    # its own line when present; raw is the comparable-to-published number and
    # rescaled is the legible one
    bert = (
        f"\n        bertF1={summary['bert_f1']:.4f} "
        f"(rescaled {summary['bert_f1_rescaled']:+.4f})"
        if summary.get("bert_f1") == summary.get("bert_f1")
        else ""
    )
    print(
        f"\n[gloss] n={summary['n']}  scored={summary['n_scored']}  "
        f"empty={summary['empty']}\n"
        f"        f1={summary['f1']:.3f}  cos={summary['cos']:.3f}  "
        f"bleu={summary['bleu']:.2f} (corpus {summary['bleu_corpus']:.2f})  "
        f"chrf={summary['chrf']:.2f}{bert}\n"
        f"        sense_acc={summary['sense_acc']:.3f} "
        f"(n={summary['n_sense_scored']}){filler}\n"
        f"        len: gloss={summary['len_gloss']:.1f} gold={summary['len_gold']:.1f}"
    )
    if summary.get("n_inv_scored"):
        print(
            f"        inv_acc={summary['inv_acc']:.3f} "
            f"margin={summary['inv_margin']:+.3f} (n={summary['n_inv_scored']}, "
            "vs the lemma's real WordNet inventory)"
        )
    if summary.get("n_paired"):
        print(
            f"        paired: agree={summary['agree']:.3f}  "
            f"gloss2_f1={summary['gloss2_f1']:.3f}  "
            f"verdict_acc={summary['verdict_acc']:.3f} (n={summary['n_verdict']})"
        )
    print(f"\nExamples (first {k}):")
    for row in rows[:k]:
        print(f"  {row['lemma']} ({row['pos']})  f1={row.get('f1', float('nan')):.2f}")
        print(f"    gold: {row['definition']}")
        print(f"    pred: {row['gloss'] or '—'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--data", default=None, help="JSON: lemma/word/usage/definition")
    ap.add_argument(
        "--score-only",
        default=None,
        help="re-score a saved output file instead of generating (CPU only)",
    )
    ap.add_argument("--default-pos", default="noun", help="used when a record has no pos")
    ap.add_argument("--filler-usage", default=FILLER_USAGE)
    ap.add_argument("--filler-word", default=FILLER_WORD)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--no-force-json",
        action="store_true",
        help="free-form decode; unparseable answers then count as empty",
    )
    ap.add_argument("--no-sbert", action="store_true")
    ap.add_argument("--bertscore", action="store_true",
                    help="report BERTScore F1 (roberta-large), raw + baseline-rescaled")
    ap.add_argument("--bertscore-device", default="cuda")
    ap.add_argument("--sbert-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sbert-device", default="cpu")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.score_only:
        saved = json.loads(Path(args.score_only).read_text())
        rows = saved["records"] if isinstance(saved, dict) else saved
    else:
        if not (args.model and args.data):
            ap.error("--model and --data are required unless --score-only is given")
        # heavy imports stay here so the scoring half runs without a GPU
        from vllm import LLM

        import eval_sense as ev

        data = load_gloss_records(args.data, default_pos=args.default_pos)
        if args.max_samples:
            data = data[: args.max_samples]

        lora_req, lora_rank = ev.load_lora(args.lora)
        llm = LLM(
            model=args.model,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=True,
            **(dict(enable_lora=True, max_lora_rank=lora_rank) if lora_req else {}),
        )
        tokenizer = llm.get_tokenizer()
        filler = dict(filler_usage=args.filler_usage, filler_word=args.filler_word)
        texts = [build_prompt(r, tokenizer, **filler) for r in data]
        decoded_all = ev.generate_all(
            llm,
            texts,
            force_json=not args.no_force_json,
            lora_request=lora_req,
        )

        rows = []
        for rec, decoded in zip(data, decoded_all):
            gloss, filler_gloss, think = parse_glosses(decoded)
            row = {
                "lemma": rec["lemma"],
                "word": rec["word"],
                "pos": rec["pos"],
                "usage": rec["usage"],
                "definition": rec["definition"],
                "gloss": gloss,
                "filler_gloss": filler_gloss,
                "think": think,
            }
            for key in ("usage2", "definition2", "senses"):
                if key in rec:
                    row[key] = rec[key]
            if "usage2" in rec:
                row["same_pred"] = parse_verdict(decoded)
                # a contrast pair is exactly one that carries its own gold for
                # sentence 2, so how the pair was built is recoverable here
                row["same_gold"] = "definition2" not in rec
            rows.append(row)

    sbert = None if args.no_sbert else _Sbert(args.sbert_model, args.sbert_device)
    summary = score_rows(
        rows, sbert=sbert, bertscore=args.bertscore, bertscore_device=args.bertscore_device
    )
    print_summary(summary, rows)

    # --score-only never writes over the file it read unless asked to
    if args.output or not args.score_only:
        out_path = Path(args.output or "gloss_eval_predictions.json")
        out_path.write_text(
            json.dumps(
                {"summary": summary, "records": rows}, ensure_ascii=False, indent=2
            )
        )
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
