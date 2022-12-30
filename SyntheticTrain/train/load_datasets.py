import os
import fire
from datasets import load_dataset, concatenate_datasets

import pdb
import psutil


def get_dataset(data_dir: str, source: str, split: str):
    """Loads and concatenates the real and fake datasets.

    Args:
        data_dir (str): The directory where the dataset is stored.
        source (str): The source of the fake dataset.
        split (str): The dataset split.

    Returns:
        HuggingFace Datasets object.
    """
    fake_path = os.path.join(data_dir, f"{source}.{split}.jsonl")
    fake_dataset = load_dataset("json", data_files=fake_path)
    label = [1] * len(fake_dataset["train"])
    fake_dataset = fake_dataset["train"].add_column("label", label)

    real_path = os.path.join(data_dir, f"webtext.{split}.jsonl")
    real_dataset = load_dataset("json", data_files=real_path)
    label = [0] * len(real_dataset["train"])
    real_dataset = real_dataset["train"].add_column("label", label)

    combined_dataset = concatenate_datasets([fake_dataset, real_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)
    return combined_dataset


def main(
    data_dir: str,
    log_dir: str,
    source: str = "xl-1542M-k40",
    n=500000,
):
    dataset = get_dataset(data_dir, source, "train")
    print(type(dataset))
    print("Number of samples: ", len(dataset))
    print("First sample: ", dataset["text"][0])
    print("First sample label: ", dataset["label"][0])

    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    fire.Fire(main)
