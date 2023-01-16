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
# "roberta-base-openai-detector"  # Accuracy is 0.7345679012345679
# "distilbert-base-uncased"  # Accuracy is 0.8858024691358025
# "roberta-base-openai-detector" finetuned with 1 aditional epoch  # Accuracy is 0.7345679012345679
dectector_model = "distilbert-base-uncased"  # Accuracy is 0.8858024691358025

def intialize_model(model_dir: str, model_name: str, local_file=False) -> tuple:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name)
    if local_file:
        model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            model_dir
        )
        model.to(device)
    else:
        model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
            model_name, device_map="auto"
        )
    return tokenizer, model


def main(
    data_dir: str,
    log_dir: str,
    model_dir: str = "",
    source: str = "xl-1542M-k40",
    model_name: str = dectector_model,
    n_train: int = 500000,
    n_valid: int = 325,  # was 10000
    n_jobs: Optional[int] = None,
    verbose: bool = False,
):
    using_local_model_file = False

    valid_texts, valid_labels = eval_utils.load_split(
        data_dir, source, "valid", n=n_valid
    )

    if model_dir != "":
        using_local_model_file = True
        print(f"Loading model and tokenizer {model_name}.")
    tokenizer, model = intialize_model(model_dir, model_name, using_local_model_file)

    input_ids = tokenizer(
        valid_texts, return_tensors="pt", truncation=True, padding=True
    ).input_ids.to(device)

    model_predictions = []
    with torch.no_grad():
        logits = model(input_ids).logits

        # predictions are flipped in the roberta-base-openai-detector model
        if not using_local_model_file:
            for i, logit in enumerate(logits):
                model_predictions.append(1 if logit[0] > logit[1] else 0)
        else:
            for i, logit in enumerate(logits):
                model_predictions.append(torch.argmax(logit).item())

    print(f"Accuracy is {accuracy_score(model_predictions, valid_labels)}")


if __name__ == "__main__":
    fire.Fire(main)
