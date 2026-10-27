"""Reproduce our final AIcrowd submission (submission.csv).

Pipeline (details in the README and the report):
  1. load data/x_train.csv, data/y_train.csv, data/x_test.csv (cached in data/cache.npz);
  2. final model = experiments.FINAL_CONFIG, trained on the whole training set:
     - 5 one-hidden-layer neural networks (128 units, cosine learning-rate
       decay, seeds 0-4) on the one-hot design matrix of preprocessing.py,
       keeping the rare optional-module questions and adding count features;
     - gradient-boosted trees (gradient_boosting.py, 1000 trees, depth 6,
       learning rate 0.03) on the raw columns (preprocessing.tree_features);
     - average of the two models' log-odds (weights 0.5 / 0.5);
     - HAREHAB1 rule: people who answered HAREHAB1 (asked only after a heart
       attack) are predicted positive;
  3. threshold = F1-optimal threshold of the same pipeline on the 5-fold
     out-of-fold probabilities (python experiments.py, results/results.json);
  4. predictions in {-1, 1} written with helpers.create_csv_submission.

Usage:  python run.py        (about 10-15 minutes on a laptop; all seeds are fixed)
"""

import time

import numpy as np

from cross_validation import fit_predict_raw
from data_loading import load_data
from experiments import FINAL_CONFIG
from helpers import create_csv_submission

DATA_PATH = "data"
OUTPUT = "submission.csv"
# F1-optimal threshold of FINAL_CONFIG on the 5-fold out-of-fold probabilities
# (row "+ HAREHAB1 rule (final model)" of `python experiments.py`)
THRESHOLD = 0.205


def main():
    """Train the final model on all training data and write the submission"""
    t0 = time.time()
    x_train, x_test, y_train, _, test_ids, names = load_data(DATA_PATH)
    y01 = (y_train == 1).astype(np.float64)
    print(f"Train {x_train.shape}, test {x_test.shape}, positives {y01.mean():.1%}")

    prob = fit_predict_raw(x_train, y01, x_test, names, FINAL_CONFIG)
    y_pred = np.where(prob >= THRESHOLD, 1, -1)

    assert len(y_pred) == len(test_ids) == 109379, "unexpected number of test rows"
    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(
        f"Wrote {OUTPUT}: {len(y_pred):,} rows, {np.mean(y_pred == 1):.1%} "
        f"predicted MICHD (threshold {THRESHOLD}) in {(time.time() - t0) / 60:.1f} min"
    )


if __name__ == "__main__":
    main()
