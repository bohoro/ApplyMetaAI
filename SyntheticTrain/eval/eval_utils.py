import os
import json
import numpy as np


def _load_split(data_dir, source, split, n=np.inf):
    path = os.path.join(data_dir, f"{source}.{split}.jsonl")
    texts = []
    for i, line in enumerate(open(path)):
        if i >= n:
            break
        texts.append(json.loads(line)["text"])
    return texts


def load_split(data_dir, source, split, n=np.inf):
    webtext = _load_split(data_dir, "webtext", split, n=n // 2)
    gen = _load_split(data_dir, source, split, n=n // 2)
    texts = webtext + gen
    labels = [0] * len(webtext) + [1] * len(gen)
    return texts, labels
