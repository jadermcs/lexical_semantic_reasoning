"""Single-example debug harness for the sense models.

Pick one prompt, point at a checkpoint, and see every stage:
    formatted prompt -> raw generation -> extracted gloss -> reward.

Choose the prompt one of three ways:
  --index N            take example N from a built split (default: direct/dev #0)
  --usage / --anchor   supply the fields inline (no dataset needed)
  --raw-prompt "..."   feed an arbitrary string, bypassing chat templating

Examples:
  uv run python src/debug_sense.py --model ./qwen-sense-grpo-direct --mode direct --index 0
  uv run python src/debug_sense.py --model ./qwen-sense-grpo-triplet \\
      --mode triplet --index 3 --split test
  uv run python src/debug_sense.py --model ./ckpt --mode direct \\
      --lemma bank --pos n --usage "He sat on the river <t>bank</t>."
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sense_data as sd


def build_record(args):
    """Return (record, has_gold). Inline flags override the dataset split."""
    if args.raw_prompt is not None:
        return None, False
    if args.usage or args.anchor:
        if args.mode == "direct":
            rec = {"lemma": args.lemma, "pos": args.pos, "usage": args.usage, "gloss": ""}
        else:
            rec = {"lemma": args.lemma, "pos": args.pos,
                   "anchor": args.anchor, "positive": args.positive,
                   "negative": args.negative, "gloss_same": "", "gloss_diff": ""}
        return rec, False
    data = sd.load_split(args.mode, args.split)
    return data[args.index], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base or merged checkpoint path")
    ap.add_argument("--mode", choices=["direct", "triplet"], default="direct")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--index", type=int, default=0)
    # inline prompt fields
    ap.add_argument("--lemma", default="")
    ap.add_argument("--pos", default="n")
    ap.add_argument("--usage", default="")
    ap.add_argument("--anchor", default="")
    ap.add_argument("--positive", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--raw-prompt", default=None)
    # decoding
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.eval()

    rec, has_gold = build_record(args)

    if args.raw_prompt is not None:
        prompt = args.raw_prompt
    else:
        msgs = (sd.direct_messages if args.mode == "direct" else sd.triplet_messages)(
            rec, with_target=False
        )
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print("=" * 70, "\nFORMATTED PROMPT\n", "=" * 70, sep="")
    print(prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if args.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=0.95)
    else:
        gen_kwargs.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    decoded = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print("\n", "=" * 70, "\nRAW COMPLETION\n", "=" * 70, sep="")
    print(decoded)

    extract = sd.extract_direct_gloss if args.mode == "direct" else sd.extract_shared_gloss
    print("\n", "=" * 70, "\nEXTRACTED GLOSS\n", "=" * 70, sep="")
    print(repr(extract(decoded)))

    if has_gold:
        gold = rec["gloss"] if args.mode == "direct" else rec["gloss_same"]
        print("\nGOLD:   ", repr(gold))
        print("REWARD: ", round(sd.gloss_similarity(extract(decoded), gold), 4))


if __name__ == "__main__":
    main()
