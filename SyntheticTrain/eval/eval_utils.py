import json
import os

import numpy as np


def _load_split(data_dir: str, source: str, split: str, n: int = np.inf) -> list:
    """Loads the data from a specific dataset split.

    Args:
        data_dir: The directory where the data is stored.
        source: The source dataset.
        split: The dataset split.
        n: The maximum number of examples to read.

    Returns:
        A list of examples.
    """
    path = os.path.join(data_dir, f"{source}.{split}.jsonl")
    examples = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        examples.append(json.loads(line)["text"])
    return examples


def load_split(data_dir: str, source: str, split: str, n: int = np.inf) -> tuple:
    """Load webtext and source data.

    Args:
        data_dir: The directory where the data is stored.
        source: The source dataset.
        split: The dataset split.
        n: The maximum number of examples to read.

    Returns:
        A list of combined webtext and generated examples.
    """
    webtext = _load_split(data_dir, "webtext", split, n=n // 2)
    gen = _load_split(data_dir, source, split, n=n // 2)

    # Combine webtext and source data.
    texts = webtext + gen
    labels = [0] * len(webtext) + [1] * len(gen)

    return texts, labels
