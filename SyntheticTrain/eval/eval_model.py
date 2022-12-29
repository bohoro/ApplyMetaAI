import torch
import fire
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import eval_utils
import pdb
import numpy as nn
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dectector_model = "roberta-base-openai-detector"


def intialize_model():
    tokenizer = AutoTokenizer.from_pretrained(dectector_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        dectector_model, device_map="auto"
    )
    return tokenizer, model


def main(
    data_dir,
    log_dir,
    source="xl-1542M-k40",
    model=dectector_model,
    n_train=500000,
    n_valid=325,  # was 10000
    n_jobs=None,
    verbose=False,
):
    print("Loading data...")
    # train_texts, train_labels = eval_utils.load_split(
    #    data_dir, source, "train", n=n_train
    # )
    valid_texts, valid_labels = eval_utils.load_split(
        data_dir, source, "valid", n=n_valid
    )
    # test_texts, test_labels = eval_utils.load_split(data_dir, source, "test")

    print("Initializing model...")
    tokenizer, model = intialize_model()

    print("Tokenizing data...")
    input_ids = tokenizer(
        valid_texts, return_tensors="pt", truncation=True, padding=True
    ).input_ids.to(device)
    print(
        f"should start with {tokenizer.bos_token_id} and end with {tokenizer.eos_token_id}"
    )

    model_predictions = []
    with torch.no_grad():
        # run the GPT model to get the logits
        logits = model(input_ids).logits

        for i, logit in enumerate(logits):
            model_predictions.append(1 if logit[0] > logit[1] else 0)

    print(f"Accuracy is {accuracy_score(model_predictions, valid_labels)}")


if __name__ == "__main__":
    fire.Fire(main)
