"""Fast loading of the project CSV files, with a NumPy cache.

np.genfromtxt (used in the course helpers) needs several minutes and a lot of
memory on the 328,135 x 321 training matrix. We parse the files once line by
line into float32 arrays and save them in data/cache.npz; later runs load the
cache in a few seconds. Missing values (empty fields) become NaN.
"""

import os

import numpy as np


def read_csv_matrix(path, dtype=np.float32):
    """Read a numeric CSV file with a header line; empty fields become NaN

    Args:
        path: path of the CSV file
        dtype: dtype of the returned matrix

    Returns:
        header: list of column names
        matrix: array of shape (n_rows, n_columns)
    """
    with open(path) as f:
        header = f.readline().rstrip("\n").split(",")
        n_rows = sum(1 for _ in f)
    out = np.empty((n_rows, len(header)), dtype=dtype)
    with open(path) as f:
        f.readline()
        for i, line in enumerate(f):
            out[i] = [float(v) if v else np.nan for v in line.rstrip("\n").split(",")]
    return header, out


def load_data(data_path="data"):
    """Load features, labels, ids and feature names (cached after the first call)

    Args:
        data_path: folder containing x_train.csv, y_train.csv and x_test.csv

    Returns:
        x_train (N, D) float32, x_test (M, D) float32, y_train (N,) in {-1, 1},
        train_ids (N,), test_ids (M,), feature_names (list of D strings)
    """
    cache = os.path.join(data_path, "cache.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return (
            d["x_train"],
            d["x_test"],
            d["y_train"],
            d["train_ids"],
            d["test_ids"],
            [str(s) for s in d["feature_names"]],
        )
    header, xtr = read_csv_matrix(os.path.join(data_path, "x_train.csv"))
    _, xte = read_csv_matrix(os.path.join(data_path, "x_test.csv"))
    _, ytr = read_csv_matrix(os.path.join(data_path, "y_train.csv"), np.float64)
    data = dict(
        x_train=xtr[:, 1:],
        x_test=xte[:, 1:],
        y_train=ytr[:, 1].astype(int),
        train_ids=xtr[:, 0].astype(int),
        test_ids=xte[:, 0].astype(int),
        feature_names=np.array(header[1:]),
    )
    np.savez(cache, **data)
    return (
        data["x_train"],
        data["x_test"],
        data["y_train"],
        data["train_ids"],
        data["test_ids"],
        header[1:],
    )
