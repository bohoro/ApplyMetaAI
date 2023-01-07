import json

import eval_utils
import fire
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def main(
    data_dir,
    log_dir,
    source="xl-1542M-k40",
    n_train=500000,
    n_valid=10000,
    n_jobs=None,
    verbose=False,
):
    print("Loading data...")
    train_texts, train_labels = eval_utils.load_split(
        data_dir, source, "train", n=n_train
    )
    valid_texts, valid_labels = eval_utils.load_split(
        data_dir, source, "valid", n=n_valid
    )
    test_texts, test_labels = eval_utils.load_split(data_dir, source, "test")

    print("Training model...")
    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=2**21)
    train_features = vect.fit_transform(train_texts)
    valid_features = vect.transform(valid_texts)
    test_features = vect.transform(test_texts)

    print("Searching for best params...")
    model = LogisticRegression(solver="liblinear")
    params = {
        "C": [1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16, 32, 64]
    }
    split = PredefinedSplit([-1] * n_train + [0] * n_valid)
    search = GridSearchCV(
        model, params, cv=split, n_jobs=n_jobs, verbose=verbose, refit=False
    )
    search.fit(
        sparse.vstack([train_features, valid_features]), train_labels + valid_labels
    )
    model = model.set_params(**search.best_params_)

    print("fitting final model...")
    model.fit(train_features, train_labels)

    print("evaluating final model...")
    valid_accuracy = model.score(valid_features, valid_labels) * 100.0
    test_accuracy = model.score(test_features, test_labels) * 100.0
    data = {
        "source": source,
        "n_train": n_train,
        "valid_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
    }
    print(data)

    print("Saving results...")
    json.dump(data, open(os.path.join(log_dir, f"baseline_{source}.json"), "w"))


if __name__ == "__main__":
    fire.Fire(main)
