from typing import Optional

import fire
from load_datasets import get_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pdb

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model to be used
dectector_model = "roberta-base-openai-detector"


def initialize_model(model_name: str) -> tuple:
    """Initialize a tokenizer and model from a model name.

    Args:
        model_name: The model to use. For example, "distilbert-base-uncased".
    Returns:
        A tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, device_map="auto"
    )
    return tokenizer, model


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """Tokenizes the dataset.

    Args:
        dataset (HuggingFace Datasets object): The dataset to be tokenized.
        tokenizer (HuggingFace Tokenizer object): The tokenizer to be used.
        max_length (int, optional): The maximum length of the tokenized dataset. Defaults to 512.

    Returns:
        HuggingFace Datasets object: The tokenized dataset.
    """
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"], truncation=True, max_length=max_length
        ),
        batched=True,
    )
    return tokenized_dataset


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
    print(f"Loading model and tokenizer {model_name}.")
    tokenizer, model = initialize_model(model_name)

    print("Loading datasets.")
    train_dataset = get_dataset(data_dir, source, "train")
    validation_dataset = get_dataset(data_dir, source, "valid")

    print("Tokenizing datasets.")
    train_dataset_tokenized = tokenize_dataset(
        train_dataset, tokenizer, model.config.max_position_embeddings
    )
    validation_dataset_tokenized = tokenize_dataset(
        validation_dataset, tokenizer, model.config.max_position_embeddings
    )

    pdb.set_trace()


if __name__ == "__main__":
    fire.Fire(main)
