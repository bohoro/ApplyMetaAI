import pdb
from typing import Optional

import eval_utils
import fire
import numpy as nn
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model to be used
dectector_model = "roberta-base-openai-detector"


def intialize_model(model_name: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, device_map="auto"
    )
    return tokenizer, model


def main(
    data_dir: str,
    log_dir: str,
    source: str = "xl-1542M-k40",
    model_name: str = dectector_model,
    n_train: int = 500000,
    n_valid: int = 325,  # was 10000
    n_jobs: Optional[int] = None,
    verbose: bool = False,
):
    valid_texts, valid_labels = eval_utils.load_split(
        data_dir, source, "valid", n=n_valid
    )

    tokenizer, model = intialize_model(model_name)

    input_ids = tokenizer(
        valid_texts, return_tensors="pt", truncation=True, padding=True
    ).input_ids.to(device)

    model_predictions = []
    with torch.no_grad():
        logits = model(input_ids).logits

        for i, logit in enumerate(logits):
            model_predictions.append(1 if logit[0] > logit[1] else 0)

    print(f"Accuracy is {accuracy_score(model_predictions, valid_labels)}")


if __name__ == "__main__":
    fire.Fire(main)
