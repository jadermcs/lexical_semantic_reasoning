"""Build SemCor-grounded datasets for distilling ChatGPT reasoning traces.

Two modes, matching the two sense-modeling ablation configs:

* ``triplet`` — an in-lexeme triplet of usages of one lemma:
      anchor, positive  -> share the SAME WordNet sense
      negative          -> a DIFFERENT WordNet sense
  ChatGPT argues why the anchor/positive share a sense and why the negative
  differs, and *generates* a single gloss for the shared anchor/positive sense
  from the contexts (the gold WordNet definitions are never shown to it; the
  negative is used only to sharpen the reasoning, not glossed). SFT target:
  `<think>τ</think><answer>d_pos</answer>`.

* ``direct`` — a single sense-tagged usage. ChatGPT argues, from the contextual
  cues, what the target word means and *generates* the definition (again without
  seeing the gold). SFT target: `<think>τ</think>\ndefinition`.

One record is built per (lemma, pos): a single representative usage / triplet, so
the distilled set is one example per word. SemCor's gold sense keys, resolved to
`wn` (Open English WordNet 2024) definitions, are kept only as the gold *reference*
for quality control — never injected into the prompt or the answer.
These traces warm-start the RLVR policy in `ch05_implementation_plan.md` (Phase 3).

Pure data prep: by default it only writes the assembled prompts. Pass `--annotate`
(needs OPENAI_API_KEY + the `openai` package) to call ChatGPT and emit finished
`<think>…</think>…` SFT records with model-generated definitions.

Quality control (annotate-only): empty/refused completions and ones with no
extractable gloss are always dropped, and `--min-bertscore` drops records whose
*generated* definition (the shared anchor/positive gloss, for triplet) is semantically far from
the gold WordNet definition it should match (BERTScore, baseline-rescaled; needs
the `bert-score` package: `uv add bert-score`). The full, unfiltered set (before the
BERTScore cut) is always also written to `<out>.unfiltered.jsonl`, so nothing is
discarded.
"""

import argparse
import json
import os
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

import wn

import sense_data as sd

WSDEVAL_URL = "http://lcl.uniroma1.it/wsdeval/data/WSD_Evaluation_Framework.zip"
_BASE = "WSD_Evaluation_Framework/Training_Corpora/SemCor"
SEMCOR_DATA_MEMBER = f"{_BASE}/semcor.data.xml"
SEMCOR_KEY_MEMBER = f"{_BASE}/semcor.gold.key.txt"

# first digit of a WN sense key (the ss_type) -> human-readable POS
SS_TYPE_TO_POS = {"1": "noun", "2": "verb", "3": "adjective", "4": "adverb", "5": "adjective"}


def sense_key_to_oewn_id(sense_key: str) -> str:
    """WN3.0 sense key `lemma%a:b:c:d:e` -> OEWN sense id `oewn-lemma__a.b.c.d.e`."""
    return "oewn-" + sense_key.replace("%", "__").replace(":", ".")


# --------------------------------------------------------------------------- #
# SemCor download + parsing
# --------------------------------------------------------------------------- #
def ensure_semcor(cache_dir: Path, keep_zip: bool = False) -> tuple[Path, Path]:
    """Return paths to (data.xml, gold.key.txt), downloading/extracting if needed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_p = cache_dir / "semcor.data.xml"
    key_p = cache_dir / "semcor.gold.key.txt"
    if data_p.exists() and key_p.exists():
        return data_p, key_p

    zip_p = cache_dir / "WSD_Evaluation_Framework.zip"
    if not zip_p.exists():
        print(f"Downloading WSD Evaluation Framework (~165 MB) from {WSDEVAL_URL} ...")
        req = urllib.request.Request(WSDEVAL_URL, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(zip_p, "wb") as out:
            shutil.copyfileobj(resp, out)

    print(f"Extracting SemCor from {zip_p.name} ...")
    with zipfile.ZipFile(zip_p) as z:
        for member, dest in [(SEMCOR_DATA_MEMBER, data_p), (SEMCOR_KEY_MEMBER, key_p)]:
            with z.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    if not keep_zip:
        zip_p.unlink()
    return data_p, key_p


def load_gold(key_p: Path) -> dict[str, str]:
    """instance_id -> first gold sense key."""
    gold = {}
    with open(key_p) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                gold[parts[0]] = parts[1]
    return gold


def mark(tokens: list[str], idx: int) -> str:
    out = list(tokens)
    out[idx] = f"<t> {out[idx]} </t>"
    return " ".join(out)


def iter_semcor_usages(data_p: Path, gold: dict[str, str]):
    """Yield one record per sense-tagged single-word instance."""
    for _event, sent in ET.iterparse(data_p, events=("end",)):
        if sent.tag != "sentence":
            continue
        tokens, instances = [], []
        for child in sent:
            surface = (child.text or "").strip()
            if child.tag == "instance":
                instances.append((len(tokens), child.get("id"), child.get("lemma")))
            tokens.append(surface)

        for idx, iid, lemma in instances:
            sense_key = gold.get(iid)
            if sense_key is None or not lemma or "_" in lemma:
                continue  # skip untagged and multiword instances
            yield {
                "lemma": lemma.lower(),
                "pos": SS_TYPE_TO_POS.get(sense_key.split("%")[1][0], "?"),
                "sense_key": sense_key,
                "text": " ".join(tokens),
                "marked": mark(tokens, idx),
            }
        sent.clear()


# --------------------------------------------------------------------------- #
# Grouping + triplet construction
# --------------------------------------------------------------------------- #
def collect_groups(data_p: Path, gold: dict[str, str], en: wn.Wordnet):
    """Group usages by (lemma, pos), attaching the resolved WordNet definition."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    def_cache: dict[str, str | None] = {}

    for u in iter_semcor_usages(data_p, gold):
        sk = u["sense_key"]
        if sk not in def_cache:
            try:
                definition = en.sense(sense_key_to_oewn_id(sk)).synset().definition()
            except wn.Error:
                definition = None
            def_cache[sk] = definition
        definition = def_cache[sk]
        if not definition:
            continue  # sense key did not resolve in OEWN 2024
        u["definition"] = definition
        groups[(u["lemma"], u["pos"])].append(u)
    return groups


class GlossSimilarity:
    """Semantic similarity between two short glosses for hard-negative mining.

    A negative sense whose *definition* is close to the positive's makes the
    triplet hard (a fine-grained, easily-confused sense distinction). Similarity
    is BERTScore-style: layer-17 roberta-large contextual token embeddings with
    greedy cosine matching (F1) — captures paraphrase that word overlap misses
    (e.g. "link together" vs "form a pair"). torch/transformers are imported
    lazily so the default (random-negative) build needs neither.
    """

    LAYER = 17

    def __init__(self, model_name: str = "roberta-large", batch_size: int = 64,
                 device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size
        self._emb: dict = {}  # gloss -> (T, H) normalized token vectors, cpu/fp16

    def prepare(self, glosses) -> None:
        """Embed (once) every gloss we will compare, caching per-token vectors."""
        torch = self._torch
        todo = sorted({g for g in glosses if g and g not in self._emb})
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i: i + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=64).to(self.device)
            with torch.no_grad():
                hs = self.model(**enc, output_hidden_states=True).hidden_states[self.LAYER]
            hs = torch.nn.functional.normalize(hs, dim=-1)
            mask = enc["attention_mask"].bool()
            for j, g in enumerate(chunk):
                m = mask[j].clone()
                m[0] = False                   # drop <s>
                m[m.nonzero()[-1]] = False      # drop trailing </s>
                self._emb[g] = hs[j][m].half().cpu()

    def f1(self, a: str, b: str) -> float:
        ea, eb = self._emb[a].float(), self._emb[b].float()
        if ea.numel() == 0 or eb.numel() == 0:
            return 0.0
        sim = ea @ eb.T
        p = sim.max(dim=1).values.mean().item()  # each token of a -> best in b
        r = sim.max(dim=0).values.mean().item()  # each token of b -> best in a
        return 2 * p * r / (p + r) if p + r else 0.0


def build_triplets(groups, max_per_lemma: int, rng: random.Random,
                   sim: "GlossSimilarity | None" = None, min_sim: float = 0.0):
    """For each lemma: (anchor, positive) from one sense, negative from another.

    With ``sim`` (hard-negative mode): for every anchor/positive sense, the negative
    is the *most definition-similar* other sense — the hardest to tell apart — and the
    triplet is kept only if that similarity clears ``min_sim``. A lemma's hardest
    triplets are taken first. Without ``sim``: the negative is a random other sense
    (original behaviour).
    """
    eligible = []
    for (lemma, pos), usages in groups.items():
        by_sense: dict[str, list[dict]] = defaultdict(list)
        for u in usages:
            by_sense[u["sense_key"]].append(u)
        # senses with >=2 usages can supply the (anchor, positive) pair
        multi = [sk for sk, us in by_sense.items() if len(us) >= 2]
        if not multi or len(by_sense) < 2:
            continue
        eligible.append((lemma, pos, by_sense, multi))

    if sim is not None:
        defs = {us[0]["definition"] for _, _, bs, _ in eligible for us in bs.values()}
        print(f"  embedding {len(defs)} sense definitions for hard-negative mining ...")
        sim.prepare(defs)

    records = []
    for lemma, pos, by_sense, multi in eligible:
        def_of = {sk: us[0]["definition"] for sk, us in by_sense.items()}
        cands = []  # (score | None, same_sk, diff_sk)
        for same_sk in multi:
            others = [dk for dk in by_sense if dk != same_sk]
            if sim is not None:
                score, diff_sk = max(
                    (sim.f1(def_of[same_sk], def_of[dk]), dk) for dk in others)
                if score < min_sim:
                    continue
            else:
                score, diff_sk = None, rng.choice(others)
            cands.append((score, same_sk, diff_sk))

        if sim is not None:
            cands.sort(key=lambda c: c[0], reverse=True)  # hardest first
        else:
            rng.shuffle(cands)

        for score, same_sk, diff_sk in cands[:max_per_lemma]:
            u1, u2 = rng.sample(by_sense[same_sk], 2)
            u3 = rng.choice(by_sense[diff_sk])
            rec = {
                "lemma": lemma,
                "pos": pos,
                "sense_same": u1["sense_key"],
                "sense_diff": u3["sense_key"],
                "definition_same": u1["definition"],
                "definition_diff": u3["definition"],
                "anchor": {"text": u1["text"], "marked": u1["marked"]},
                "positive": {"text": u2["text"], "marked": u2["marked"]},
                "negative": {"text": u3["text"], "marked": u3["marked"]},
            }
            if score is not None:
                rec["sense_sim"] = round(score, 4)
            records.append(rec)
    return records


def build_direct(groups, max_per_lemma: int, rng: random.Random):
    """One record per sense-tagged usage: (usage, its gold definition)."""
    records = []
    for (lemma, pos), usages in groups.items():
        pool = list(usages)
        rng.shuffle(pool)
        for u in pool[:max_per_lemma]:
            records.append({
                "lemma": lemma,
                "pos": pos,
                "sense_key": u["sense_key"],
                "definition": u["definition"],
                "usage": {"text": u["text"], "marked": u["marked"]},
            })
    return records


TRIPLET_SYSTEM_PROMPT = (
    "You are a lexicographer. You are given one or more POSITIVE usages of a target word "
    "(marked with <t> ... </t>) that all share the same sense, plus one or more NEGATIVE "
    "usages where the same word carries a DIFFERENT sense.\n\n"
    "First think step by step inside <think> ... </think>:\n"
    "1. State the sense shared by all POSITIVE usages.\n"
    "2. State the sense of the NEGATIVE usages.\n"
    "3. Identify the distinguishing feature that separates them.\n"
    "4. Draft a definition and check it against each negative — it must reject every one. "
    "Revise if any negative slips through.\n\n"
    "Then output the final definition inside <answer> ... </answer>. The definition must:\n"
    "- fit every POSITIVE usage,\n"
    "- fit NONE of the NEGATIVE usages,\n"
    "- generalize to unseen usages of the same sense (don't overfit to these exact "
    "sentences).\n\n"
    "Do not mention the example sentences or the target word inside <answer>."
)


def _triplet_user(lemma: str, pos: str, anchor: str, positive: str, negative: str) -> str:
    return (
        f"Target word: {lemma} ({pos})\n"
        "POSITIVE usages (same sense):\n"
        f"1. {anchor}\n"
        f"2. {positive}\n"
        "NEGATIVE usages (different sense):\n"
        f"1. {negative}"
    )


def _triplet_shot(lemma, pos, anchor, positive, negative, think, pos_gloss):
    """One few-shot (user, assistant) demonstration pair for the triplet mode."""
    answer = f"<answer>\n{pos_gloss}\n</answer>"
    return [
        {"role": "user", "content": _triplet_user(lemma, pos, anchor, positive, negative)},
        {"role": "assistant", "content": f"<think>\n{think}\n</think>{answer}"},
    ]


TRIPLET_FEWSHOT = [
    _triplet_shot(
        "bank", "noun",
        "I deposited my paycheck at the <t> bank </t> this morning.",
        "The <t> bank </t> rejected her mortgage application.",
        "They sat on the grassy <t> bank </t> of the river.",
        "The positives both involve money handling — depositing a paycheck, ruling on a "
        "mortgage — so the shared sense is a financial institution. The negative is a "
        "physical landform, the sloped ground beside a river. The distinguishing feature "
        "is institution-that-handles-money vs. piece-of-terrain. A definition anchored on "
        "\"accepts deposits and lends money\" admits both positives and has no way to "
        "describe a riverside slope, so the negative is excluded.",
        "a financial institution that accepts deposits and lends money"),
    _triplet_shot(
        "crane", "noun",
        "A <t> crane </t> lifted the steel beams onto the roof.",
        "The construction site had three <t> crane </t>s running at once.",
        "A <t> crane </t> waded through the shallows hunting for fish.",
        "The positives are about lifting heavy loads on a construction site, so the shared "
        "sense is a lifting machine. The negative wades through the shallows hunting fish — "
        "a living bird. The distinguishing feature is inanimate machine vs. animate animal. "
        "Anchoring the definition on \"machine ... used to lift heavy objects\" covers both "
        "positives; a bird is neither a machine nor a lifting device, so the negative is "
        "rejected.",
        "a tall machine with a projecting arm, used to lift and move heavy objects"),
    _triplet_shot(
        "head", "noun",
        "the <t> head </t> of the company",
        "Apple's <t> head </t>",
        "the <t> head </t> of the tower",
        "Both positives pick out the person in charge of an organization — a company, "
        "Apple. The negative, \"the head of the tower,\" is the topmost physical part of a "
        "structure, not a person. The distinguishing feature is animate leader-of-a-group "
        "vs. inanimate uppermost-part-of-an-object. A generic gloss like \"the leading or "
        "topmost part of something\" would wrongly cover the tower, so I anchor on "
        "personhood and authority: \"the person who leads an organization\" admits company "
        "and Apple, and cannot describe a tower's top, so the negative is excluded.",
        "the person who leads or has authority over an organization or group"),
]


def build_triplet_prompt(rec: dict, shots=()) -> list[dict]:
    msgs = [{"role": "system", "content": TRIPLET_SYSTEM_PROMPT}]
    for shot in shots:
        msgs.extend(shot)
    msgs.append({"role": "user", "content": _triplet_user(
        rec["lemma"], rec["pos"], rec["anchor"]["marked"],
        rec["positive"]["marked"], rec["negative"]["marked"])})
    return msgs


# --- direct: argue the sense and generate the gloss from context ---
DIRECT_SYSTEM_PROMPT = (
    "You are an expert lexicographer. You are given one usage of a target word "
    "(marked with <t> tags). Inside <think> tags, write a brief, tight argument (one "
    "short paragraph, no bullet lists) that reads the contextual cues and works out what "
    "the target word means here. Reason forward from the context only — you are given a "
    "single usage and no other sense, so do not classify by genus or contrast against "
    "another sense. Then, after </think>, give a single concise dictionary definition of "
    "the target word as used here — and nothing else. Keep it concise and never use the "
    "target word to define itself. Format: <think>...</think>\ndefinition"
)


def _direct_user(lemma: str, pos: str, marked: str) -> str:
    return (
        f"Target word: {lemma} ({pos})\n\n"
        f"Usage: {marked}\n\n"
        "Argue from the contextual cues to the sense the target word carries here, then "
        "give the definition."
    )


def _direct_shot(lemma, pos, marked, think, definition):
    """One few-shot (user, assistant) demonstration pair for the direct mode."""
    return [
        {"role": "user", "content": _direct_user(lemma, pos, marked)},
        {"role": "assistant", "content": f"<think>\n{think}\n</think>\n{definition}"},
    ]


DIRECT_FEWSHOT = [
    _direct_shot(
        "bank", "noun",
        "They dragged the canoe up onto the <t> bank </t> and made camp for the night.",
        "The cues set a riverside scene: a canoe dragged up out of the water and a camp "
        "made beside it. The word picks out the ground along the side of the river — the "
        "land that slopes up from the water and that you climb onto when leaving it.",
        "the sloping ground bordering a river or other body of water"),
    _direct_shot(
        "decline", "verb",
        "Sales <t> declined </t> sharply in the months after the factory closed.",
        "'Sales' as the subject and 'sharply' as the manner make the word a movement of a "
        "measurable quantity over the months after the closure, and the context points it "
        "downward — there are fewer and fewer sales. So here it means to go down in amount.",
        "to go down or decrease in amount, quality, or value"),
    _direct_shot(
        "keen", "adjective",
        "She had a <t> keen </t> eye for the smallest flaw in the cut of a diamond.",
        "Attached to an 'eye' that catches the smallest flaw, the word describes how sharply "
        "the person perceives — the ability to notice fine detail that others would miss.",
        "quick to notice or understand; sharply perceptive"),
]


def build_direct_prompt(rec: dict, shots=()) -> list[dict]:
    msgs = [{"role": "system", "content": DIRECT_SYSTEM_PROMPT}]
    for shot in shots:
        msgs.extend(shot)
    msgs.append({"role": "user", "content": _direct_user(
        rec["lemma"], rec["pos"], rec["usage"]["marked"])})
    return msgs


# builder, prompt fn, gloss extractor, few-shot exemplars per mode. ChatGPT now
# generates the whole answer (no gold answer is assembled); the extractor pulls its
# generated definition back out (shared anchor/positive gloss for triplet) for the BERTScore filter.
MODES = {
    "triplet": (build_triplets, build_triplet_prompt, sd.extract_shared_gloss, TRIPLET_FEWSHOT),
    "direct": (build_direct, build_direct_prompt, sd.extract_direct_gloss, DIRECT_FEWSHOT),
}


def annotate(records: list[dict], model: str, extract_gloss) -> None:
    """Call ChatGPT to generate each record's full reasoning + answer (in place).

    Each record must already carry ``prompt``. This fills ``sft_target`` with the
    raw completion (`<think>…</think>` + the generated definition/glosses) and
    ``gen_gloss`` with the definition extracted back out of it (the shared
    anchor/positive gloss for triplet, the single definition for direct) — what the BERTScore filter then
    scores against the gold WordNet definition. Empty completions, and ones with no
    extractable gloss, leave ``gen_gloss`` empty and are dropped after the loop.
    """
    from openai import OpenAI  # lazy import

    client = OpenAI()
    for i, rec in enumerate(records, 1):
        kwargs = {"model": model, "messages": rec["prompt"], "temperature": 0.7}
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            # reasoning models (e.g. gpt-5-*) only allow the default temperature
            if "temperature" not in str(e):
                raise
            kwargs.pop("temperature")
            resp = client.chat.completions.create(**kwargs)
        completion = (resp.choices[0].message.content or "").strip()
        rec["sft_target"] = completion
        rec["gen_gloss"] = extract_gloss(completion) if completion else ""
        if i % 25 == 0:
            print(f"  annotated {i}/{len(records)}")


# the gold definition each mode's reasoning trace is meant to justify
DEF_KEY = {"triplet": "definition_same", "direct": "definition"}


def filter_by_bertscore(records, def_key, min_score, metric="f1", model_type=None):
    """Drop records whose generated definition is semantically far from the gold.

    Scores each record's ChatGPT-generated definition ``gen_gloss`` (the Positive
    anchor/positive gloss, for triplet) against the gold WordNet definition it should match
    (BERTScore). Off-topic or wrong-sense glosses land far from the gold and fall
    below ``min_score``. Scores are baseline-rescaled (~0 for unrelated text, up to
    ~1) so the threshold is interpretable across runs; ~0.15 is a reasonable start.
    """
    from bert_score import score  # lazy import; needs the `bert-score` package

    cands = [r["gen_gloss"] for r in records]
    refs = [r[def_key] for r in records]
    kwargs = {"lang": "en", "rescale_with_baseline": True, "verbose": True}
    if model_type:  # baseline files only ship for the default models -> skip rescale
        kwargs.update(model_type=model_type, rescale_with_baseline=False)
    P, R, F1 = score(cands, refs, **kwargs)

    for r, p, rc, f in zip(records, P.tolist(), R.tolist(), F1.tolist()):
        r["bertscore"] = {"precision": p, "recall": rc, "f1": f}
    threshold = {"precision": P, "recall": R, "f1": F1}[metric].tolist()
    kept = [r for r, s in zip(records, threshold) if s >= min_score]
    print(f"  BERTScore {metric} filter (>= {min_score}): "
          f"kept {len(kept)}, dropped {len(records) - len(kept)}.")
    return kept


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["direct", "triplet", "both"], default="both")
    parser.add_argument("--lemma-split",
                        choices=["train", "non-eval", "dev", "test", "all"], default="non-eval",
                        help="restrict to lemmas in this split of sense_data. 'train' = strictly "
                        "the ablation's train lemmas; 'non-eval' = every lemma except dev/test "
                        "(leakage-safe and keeps SemCor lemmas with no WordNet example); "
                        "'all' disables filtering.")
    parser.add_argument("--split-seed", type=int, default=42, help="must match sense_data's seed")
    parser.add_argument("--lexicon", default="oewn:2024")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/semcor"))
    parser.add_argument("--keep-zip", action="store_true", help="keep the downloaded zip")
    parser.add_argument("--max-per-lemma", type=int, default=1,
                        help="examples kept per (lemma, pos); default 1 = one example per word")
    parser.add_argument("--num-shots", type=int, default=3,
                        help="few-shot exemplars prepended to each annotation prompt "
                        "(0-3; clamped to the 3 hand-written ones per mode)")
    parser.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    parser.add_argument("--hard-negatives", action="store_true",
                        help="(triplet) mine hard negatives: pair each sense with its most "
                        "definition-similar other sense instead of a random one, for a harder, "
                        "fine-grained dataset. Needs torch + roberta-large (lazy-loaded).")
    parser.add_argument("--min-sense-sim", type=float, default=0.0,
                        help="(triplet, with --hard-negatives) drop triplets whose hardest-negative "
                        "gloss similarity is below this (roberta-large BERTScore-F1, ~0..1). 0 = keep all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (single mode only; default data/semcor_distill_<mode>.jsonl)")
    parser.add_argument("--annotate", action="store_true", help="call ChatGPT for traces")
    parser.add_argument("--model", default="gpt-5-nano", help="OpenAI model for --annotate")
    parser.add_argument("--min-bertscore", type=float, default=0.0,
                        help="with --annotate, drop records whose generated definition's "
                        "BERTScore vs. the gold definition is below this (baseline-rescaled; "
                        "try ~0.15). 0 = off.")
    parser.add_argument("--bertscore-metric", choices=["f1", "recall", "precision"],
                        default="f1", help="which BERTScore component to threshold; "
                        "'recall' rewards covering the gloss and is lenient on the "
                        "argument's extra reasoning")
    parser.add_argument("--bertscore-model", default=None,
                        help="override the BERTScore model (default: roberta-large via "
                        "lang=en, with baseline rescaling)")
    args = parser.parse_args()

    if args.annotate and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("--annotate needs OPENAI_API_KEY in the environment.")

    rng = random.Random(args.seed)
    en = wn.Wordnet(args.lexicon)

    data_p, key_p = ensure_semcor(args.cache_dir, keep_zip=args.keep_zip)
    gold = load_gold(key_p)
    print(f"Loaded {len(gold)} gold sense annotations.")

    groups = collect_groups(data_p, gold, en)
    print(f"Grouped usages for {len(groups)} (lemma, pos) keys.")

    if args.lemma_split != "all":
        train_l, dev_l, test_l = sd.lemma_splits(args.lexicon, args.split_seed)
        before = len(groups)
        if args.lemma_split == "non-eval":
            eval_l = dev_l | test_l
            groups = {k: v for k, v in groups.items() if k[0] not in eval_l}
        else:
            allowed = {"train": train_l, "dev": dev_l, "test": test_l}[args.lemma_split]
            groups = {k: v for k, v in groups.items() if k[0] in allowed}
        dropped = before - len(groups)
        print(f"Restricted to '{args.lemma_split}' lemmas: kept {len(groups)}, "
              f"dropped {dropped} (lemma, pos) groups.")

    modes = ["direct", "triplet"] if args.mode == "both" else [args.mode]
    if args.out and len(modes) > 1:
        raise SystemExit("--out only applies to a single --mode (not 'both').")

    sim = None
    if args.hard_negatives and "triplet" in modes:
        print("[triplet] loading roberta-large for hard-negative mining ...")
        sim = GlossSimilarity()

    for mode in modes:
        builder, prompt_fn, extract_gloss, fewshot = MODES[mode]
        shots = fewshot[: max(0, args.num_shots)]
        if mode == "triplet":
            records = build_triplets(groups, args.max_per_lemma, rng, sim, args.min_sense_sim)
        else:
            records = builder(groups, args.max_per_lemma, rng)
        rng.shuffle(records)
        if args.max_examples:
            records = records[: args.max_examples]
        for rec in records:
            rec["prompt"] = prompt_fn(rec, shots)
        msg = f"[{mode}] built {len(records)} records ({len(shots)}-shot prompts)."
        if mode == "triplet" and sim is not None and records:
            sims = sorted(r["sense_sim"] for r in records)
            mid = sims[len(sims) // 2]
            msg += f" hard-neg sense_sim: min={sims[0]:.3f} median={mid:.3f} max={sims[-1]:.3f}"
        print(msg)

        if args.annotate:
            print(f"[{mode}] annotating with {args.model} ...")
            annotate(records, args.model, extract_gloss)
            before = len(records)
            records = [r for r in records if r.get("gen_gloss")]
            if len(records) < before:
                print(f"  dropped {before - len(records)} empty/refused/unparseable traces.")

            out = args.out or Path(f"data/semcor_distill_{mode}.jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)
            unfiltered_out = out.with_suffix(f".unfiltered{out.suffix}")
            with unfiltered_out.open("w") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{mode}] wrote {len(records)} unfiltered records to {unfiltered_out}")

            if args.min_bertscore > 0 and records:
                records = filter_by_bertscore(
                    records, DEF_KEY[mode], args.min_bertscore,
                    metric=args.bertscore_metric, model_type=args.bertscore_model)
        else:
            out = args.out or Path(f"data/semcor_distill_{mode}.jsonl")
            out.parent.mkdir(parents=True, exist_ok=True)

        with out.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{mode}] wrote {len(records)} records to {out}")

        if records:
            s = records[0]
            print(f"  sample [{s['lemma']} ({s['pos']})]")
            if mode == "triplet":
                print(f"    anchor:   {s['anchor']['marked']}")
                print(f"    negative: {s['negative']['marked']}")
            else:
                print(f"    usage: {s['usage']['marked']}")
                print(f"    sense: {s['definition']}")


if __name__ == "__main__":
    main()
