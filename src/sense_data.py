import json
import re
from pathlib import Path

from rapidfuzz import fuzz, process

DATA_DIR = Path("data")

def wic_system(with_think: bool = True, with_gloss: bool = True) -> str:
    """The WiC system prompt, matching the answer shape the target is rendered in.

    The two flags are the SFT ablations: ``with_think=False`` trains on an empty
    ``<think></think>`` block (no reasoning trace), ``with_gloss=False`` drops the
    per-usage glosses from the JSON verdict. The instruction has to move with the
    target -- asking for keys the completion never contains teaches the model to
    disregard its own prompt.
    """
    head = (
        "You are an expert lexicographer. You are given two sentences, each using the "
        "same target word (marked with <t> tags). "
    )
    if with_think:
        reason = (
            "Inside <think> tags, work out what the target word means in each "
            "sentence then compare the two senses. Then, after </think>, answer "
        )
        think_fmt = "<think>...</think>"
    else:
        reason = (
            "Do not reason: emit an empty <think></think> block, then answer "
            "immediately "
        )
        think_fmt = "<think></think>"
    keys = (
        '"sense1" (string, the gloss of the target in sentence 1), "sense2" (string, '
        'the gloss of the target in sentence 2), and "same_sense" (boolean, true if '
        "the two uses share the same sense). "
        if with_gloss
        else '"same_sense" (boolean, true if the two uses share the same sense). '
    )
    shape = (
        '{"sense1": ..., "sense2": ..., "same_sense": ...}'
        if with_gloss
        else '{"same_sense": ...}'
    )
    return (
        f"{head}{reason}with a single JSON object and nothing else, with exactly "
        f"these keys: {keys}Format: {think_fmt}\n{shape}"
    )


WIC_SYSTEM = wic_system()


def mark_target(sentence: str, word: str, fuzzy_threshold: float = 70.0) -> str:
    """Wrap the best match for *word* in *sentence* with <t> tags (handles inflection)."""
    pattern = rf"\b({re.escape(word)}\w*)\b"
    if re.search(pattern, sentence, flags=re.IGNORECASE):
        return re.sub(pattern, r"<t> \1 </t>", sentence, count=1, flags=re.IGNORECASE)
    tokens = re.findall(r"\w+", sentence)
    if tokens:
        match = process.extractOne(
            word.lower(),
            [t.lower() for t in tokens],
            scorer=fuzz.QRatio,
            score_cutoff=fuzzy_threshold,
        )
        if match is not None:
            best = tokens[match[2]]
            return re.sub(
                rf"\b{re.escape(best)}\b", f"<t> {best} </t>", sentence, count=1
            )
    return sentence + f" <t> {word} </t>"


def load_teacher_traces(
    path: str | Path, strategy: str = "first", scorer=None
) -> list[dict]:
    raw = json.loads(Path(path).read_text())
    out = []
    for r in raw:
        ans = r.get("answers", None)
        if not ans:
            continue
        parsed = json.loads(r.get("answers", "null"))
        if not {"sense1", "sense2", "same_sense"} <= parsed.keys():
            continue
        out.append(
            {
                "task": "wic",
                "lemma": r["lemma"],
                "pos": r["pos"],
                "sentence1": r["sentence1"],
                "sentence2": r["sentence2"],
                "think": r["reasonings"],
                "sense1": parsed["sense1"],
                "sense2": parsed["sense2"],
                "label": parsed["same_sense"],
            }
        )
    return out


def pair_key(rec: dict) -> tuple:
    """Stable identity of a WiC pair, shared across every loader.

    Keys on ``(lemma, pos, sentence1, sentence2)`` — the *marked* sentences every
    loader carries — so a pair distilled into the SFT set can be recognised in the
    GRPO rollout source and held out of it. The markers belong in the key: these
    corpora annotate a target occurrence, and one sentence pair can hold several
    items differing only in which occurrence is marked.
    """
    return (rec["lemma"], rec["pos"], rec["sentence1"], rec["sentence2"])


def think_block(rec, with_think: bool = True) -> str:
    """The distilled teacher trace, wrapped in <think> tags (shared by all tasks).

    ``with_think=False`` keeps the tags and drops the trace, so the completion still
    has the shape the format reward and the answer parser expect.
    """
    return f"<think>\n{rec['think']}\n</think>" if with_think else "<think></think>"


wic_think = think_block


def wic_answer(rec, with_gloss: bool = True) -> str:
    """JSON verdict mirroring the teacher: sense gloss per usage + same_sense."""
    obj = {}
    if with_gloss:
        obj["sense1"] = rec.get("sense1", "")
        obj["sense2"] = rec.get("sense2", "")
    obj["same_sense"] = bool(rec["label"])
    return json.dumps(obj)


def wic_messages(rec, with_target=False, with_think=True, with_gloss=True):
    """Chat messages for one pair.

    ``with_target`` appends the assistant turn (the SFT target), which needs the
    distilled ``think``/``sense1``/``sense2`` fields — i.e. a ``load_teacher_traces``
    record. Prompt-only rendering (``with_target=False``) works for any wic record,
    including the gloss-free MCL-WiC ones GRPO and eval use.

    ``with_think``/``with_gloss`` are the SFT ablations, and they shape the *prompt*
    as well as the target: a run trained with empty think tags or a gloss-free verdict
    is asked for exactly what it is trained to emit.
    """
    keys = (
        'keys "sense1", "sense2" (the gloss of the target in each sentence) and '
        '"same_sense" (boolean)'
        if with_gloss
        else 'the key "same_sense" (boolean)'
    )
    user = (
        f"Target word: {rec['lemma']} ({rec['pos']})\n\n"
        f"Sentence 1: {rec['sentence1']}\n"
        f"Sentence 2: {rec['sentence2']}\n\n"
        "Do both sentences use the target word in the same sense? Respond with a "
        f"single JSON object with {keys}."
    )
    msgs = [
        {"role": "system", "content": wic_system(with_think, with_gloss)},
        {"role": "user", "content": user},
    ]
    if with_target:
        msgs.append(
            {
                "role": "assistant",
                "content": f"{think_block(rec, with_think)}\n"
                f"{wic_answer(rec, with_gloss)}",
            }
        )
    return msgs


WIC_ANSWER_KEYS = {"sense1", "sense2", "same_sense"}


def parse_json_answer(text: str) -> dict | None:
    """The JSON object in the answer region, or None. Shared by every task."""
    seg = text.split("</think>")[-1]
    if "<think>" in seg:  # unclosed <think>: reasoning ran on, no answer
        return None
    m = re.search(r"\{.*\}", seg, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_wic_answer(text: str) -> dict | None:
    return parse_json_answer(text)


def extract_wic_label(text: str) -> bool | None:
    obj = parse_wic_answer(text)
    if obj is not None:
        try:
            return bool(obj["same_sense"])
        except (KeyError, TypeError):
            pass
    seg = text.split("</think>")[-1]
    if "<think>" in seg:
        return None
    m = re.search(r"\b(same|different)\b", seg, flags=re.IGNORECASE)
    return m.group(1).lower() == "same" if m else None
