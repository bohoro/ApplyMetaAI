import pdb
from typing import Optional

import fire
import torch
from accelerate import Accelerator
from load_datasets import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_scheduler)

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define configuration
# model examples are "roberta-base-openai-detector", "distilbert-base-uncased", "xlnet-base-cased" (tokenizer failing)
dectector_model = "distilbert-base-uncased" 
BATCH_SIZE = 16
NUM_EPOCHS = 3


def initialize_model(model_name: str) -> tuple:
    """Initialize a tokenizer and model from a model name.

    Args:
        model_name: The model to use. For example, "distilbert-base-uncased".
    Returns:
        A tokenizer and model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,  # device_map="auto"
    ).to(device)
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
            examples["text"], truncation=True, padding=True, max_length=max_length
        ),
        batched=True,
    )
    tokenized_dataset.set_format("torch")
    return tokenized_dataset


def main(
    data_dir: str,
    log_dir: str,
    model_dir: str,
    source: str = "xl-1542M-k40",
    model_name: str = dectector_model,
    n_train: int = 500000,
    n_valid: int = 325,  # was 10000
    n_jobs: Optional[int] = None,
    verbose: bool = False,
):
    accelerator = Accelerator()

    print(f"Loading model and tokenizer {model_name}.")
    tokenizer, model = initialize_model(model_name)

    print("Loading datasets.")
    train_dataset = get_dataset(data_dir, source, "train")
    validation_dataset = get_dataset(data_dir, source, "valid")

    print("Tokenizing datasets.")
    train_dataset_tokenized = tokenize_dataset(
        train_dataset, tokenizer, model.config.max_position_embeddings - 2
    )
    validation_dataset_tokenized = tokenize_dataset(
        validation_dataset, tokenizer, model.config.max_position_embeddings - 2
    )

    print("Setting up relevant training objects")
    train_dataloader = DataLoader(train_dataset_tokenized, batch_size=BATCH_SIZE)
    validation_dataloader = DataLoader(
        validation_dataset_tokenized, batch_size=BATCH_SIZE
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    train_dataloader, validation_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, validation_dataloader, model, optimizer
    )
    num_epochs = NUM_EPOCHS
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    print("Training model.")
    for epoch in range(num_epochs):
        for i, batch in enumerate(train_dataloader):
            outputs = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                labels=batch["label"].to(device),
            )
            loss = outputs.loss
            if accelerator.is_main_process and i % 10 == 0:
                progress_bar.set_description(f"Epoch {epoch} Loss {loss.item()}")
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # save the model
    print(f"Saving model to {model_dir}")
    model.save_pretrained(model_dir)

    print("Complete")


if __name__ == "__main__":
    fire.Fire(main)
