#!/usr/bin/env python
# coding: utf-8
import re

import pandas as pd
import wn
from rapidfuzz import fuzz, process
from tqdm import tqdm
from transformers import AutoTokenizer

OUTPUT_FILE = "data/wordnet_definition.train.json"

# --- Main processing loop ---
en = wn.Wordnet("oewn:2024")

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

def convert_pos_wordnet(pos):
    """Convert wordnet POS tags to standard format."""
    if pos == "n":
        return "noun"
    elif pos == "v":
        return "verb"
    elif pos == "a":
        return "adjective"
    elif pos == "r":
        return "adverb"
    elif pos == "s":
        return "adverb"
    else:
        raise ValueError(f"Unknown POS tag: {pos}")

st = []
for synset in tqdm(en.synsets()):
    examples = synset.examples()
    if not examples:
        continue

    lemmas = synset.lemmas()
    curr_max = 0
    best = None
    pos = convert_pos_wordnet(synset.pos)
    for lemma in lemmas:
        for example in examples:
            if not re.search(f"\\b{lemma}\\b", example):
                words = example.split()
                term_words = len(lemma.split("-"))  # account for hyphenated terms

                # Generate n-gram windows from the text
                candidates = []
                for n in range(1, len(words) + 1):
                    for i in range(len(words) - n + 1):
                        candidates.append(" ".join(words[i : i + n]))

                # Find best match
                best_match = process.extractOne(
                    lemma.replace("-", " "),  # normalize the term
                    candidates,
                    scorer=fuzz.ratio,
                )
                if best_match[1] > 80 and len(example) > curr_max and lemma.lower() in tokenizer.vocab:
                    word = best_match[0]
                    best = {
                        "lemma": lemma,
                        "pos": pos,
                        "word": word,
                        "sentence": example,
                        "definition": synset.definition()
                    }
    if best:
        st.append(best)

if st:
    # Append the new entries to the CSV file
    df_new = pd.DataFrame(st).sort_values(by="example", key=lambda col: -col.str.len())
    df_new = df_new.head(2000).sample(frac=1.0, random_state=42)
    df_new.to_json(OUTPUT_FILE, orient="records", indent=2)
    print(f"Processed {len(st)} entries and saved to {OUTPUT_FILE}")
