import json
import random

import numpy as np
import torch
from datasets import Dataset


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(datasets, split="train"):
    """Load and process the WiC dataset."""
    all_data = []
    for dataset in datasets.split(","):
        with open(f"data/{dataset}.{split}.json", "r") as f:
            data = json.load(f)
            if "label" in data[0]:
                assert isinstance(data[0]["label"], int), (
                    f"Expected label to be an int, got {type(data[0]['label'])}"
                )
            all_data.extend(data)
    return Dataset.from_list(all_data)


