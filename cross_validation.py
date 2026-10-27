"""Metrics, model training and k-fold cross-validation.

Models (all return probabilities P(MICHD | x)):
- L2-regularized logistic regression trained by gradient descent. With
  `pos_weight == 1` it calls `reg_logistic_regression` from implementations.py;
  with `pos_weight > 1` each positive example counts `pos_weight` times in the
  loss (class weighting for the 9% minority class). With solver="newton" the
  same objective is solved exactly with Newton's method (IRLS);
- an ensemble of small neural networks (neural_net.MLP);
- gradient-boosted decision trees (gradient_boosting.GBDT), trained on the raw
  columns (preprocessing.tree_features) rather than the one-hot design matrix;
- a blend: the mean of the logistic and network probabilities;
- an "ensemble": weighted average, in logit space, of any of the above, each
  with its own preprocessing.
The "rehab rule" sets P = 1 for people who answered HAREHAB1 (a question only
asked to people who had a heart attack). The decision threshold is chosen to
maximise F1 on out-of-fold predictions.
"""

import numpy as np

from implementations import (
    reg_logistic_regression,
    sigmoid,
    compute_logistic_loss,
)
from gradient_boosting import GBDT
from neural_net import MLP
from preprocessing import fit_preprocessor, fit_tree_features, transform, tree_features

# Keys of a config dict that are passed to fit_preprocessor.
PREP_KEYS = (
    "min_count",
    "max_nan",
    "interactions",
    "drop_admin",
    "clean",
    "one_hot",
    "engineered",
    "exclude",
)


def f1_score(y_true, y_pred):
    """F1 score for 0/1 label vectors"""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def accuracy(y_true, y_pred):
    """Fraction of correct predictions"""
    return np.mean(y_true == y_pred)


def best_threshold(y_true, prob, grid=None):
    """Threshold on predicted probabilities that maximises F1

    Args:
        y_true: 0/1 labels
        prob: predicted probabilities
        grid: candidate thresholds (default 0.05 to 0.95 by 0.005)

    Returns:
        (threshold, F1 at that threshold)
    """
    if grid is None:
        grid = np.arange(0.05, 0.95, 0.005)
    f1s = [f1_score(y_true, (prob >= t).astype(int)) for t in grid]
    k = int(np.argmax(f1s))
    return grid[k], f1s[k]


def train_logistic(tx, y, lambda_, gamma, max_iters, pos_weight=1.0):
    """Train (class-weighted) L2-regularized logistic regression by gradient descent

    Args:
        tx: design matrix (N, D), float32 or float64
        y: 0/1 labels (N,), same dtype as tx (avoids up-casting tx)
        lambda_: L2 regularization strength (penalty lambda_ * ||w||^2)
        gamma: step size
        max_iters: number of gradient steps
        pos_weight: weight of positive examples in the loss (1 = unweighted)

    Returns:
        (w, loss): final weights and unweighted, unpenalized logistic loss
    """
    w0 = np.zeros(tx.shape[1], dtype=tx.dtype)
    if pos_weight == 1.0:
        return reg_logistic_regression(y, tx, lambda_, w0, max_iters, gamma)
    s = (1.0 + (pos_weight - 1.0) * y).astype(tx.dtype)
    s /= s.mean()  # keep the gradient scale comparable to the unweighted case
    w = w0
    for _ in range(max_iters):
        grad = tx.T @ (s * (sigmoid(tx @ w) - y)) / len(y) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)


def train_logistic_newton(tx, y, lambda_, max_iters=15, tol=1e-5, block=32768):
    """L2-regularized logistic regression solved with Newton's method (IRLS)

    Same objective as reg_logistic_regression (mean logistic loss +
    lambda_ * ||w||^2) but each step solves H d = grad with the exact Hessian
    H = tx^T diag(p (1 - p)) tx / N + 2 lambda_ I. It reaches the optimum in
    ~10 steps, whereas gradient descent on 1000+ correlated one-hot columns is
    not fully converged after 300 steps (in our ablation the two solutions give
    almost the same F1, so the GD solution was already good enough).

    Args:
        tx: design matrix (N, D) float32
        y: 0/1 labels (N,)
        lambda_: L2 regularization strength
        max_iters: maximum number of Newton steps
        tol: stop when the largest weight change is below tol
        block: rows per block when accumulating the Hessian (limits memory)

    Returns:
        (w, loss): weights (float64) and unpenalized logistic loss
    """
    n, d = tx.shape
    y = y.astype(np.float64)
    w = np.zeros(d)
    for _ in range(max_iters):
        p = sigmoid(tx @ w.astype(tx.dtype)).astype(np.float64)
        grad = tx.T @ (p - y).astype(tx.dtype) / n + 2 * lambda_ * w
        s = (p * (1 - p)).astype(tx.dtype)
        hess = 2 * lambda_ * np.eye(d)
        for i in range(0, n, block):
            b = tx[i : i + block]
            hess += (b * s[i : i + block, None]).T @ b / n
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < tol:
            break
    return w, compute_logistic_loss(y, tx, w.astype(tx.dtype))


def fit_predict(tx_tr, y_tr, tx_te, config):
    """Train the model described by `config` and return probabilities on tx_te

    config["model"] is one of
      - "logistic": regularized logistic regression (train_logistic, or
        train_logistic_newton if config["solver"] == "newton");
      - "mlp": average of `n_nets` one-hidden-layer networks with different seeds;
      - "blend": mean of the logistic and the "mlp" probabilities.
    tx_tr and tx_te are modified in place (clipped) when a network is used
    """
    model = config.get("model", "logistic")
    y_tr = y_tr.astype(tx_tr.dtype)
    probs = []
    if model in ("logistic", "blend"):
        if config.get("solver") == "newton":
            w, _ = train_logistic_newton(tx_tr, y_tr, config["lambda_"])
            w = w.astype(tx_te.dtype)
        else:
            w, _ = train_logistic(
                tx_tr,
                y_tr,
                config["lambda_"],
                config["gamma"],
                config["max_iters"],
                config.get("pos_weight", 1.0),
            )
        probs.append(sigmoid(tx_te @ w))
    if model in ("mlp", "blend"):
        # Rare one-hot levels give large standardized values: clip for stable training.
        np.clip(tx_tr, -10, 10, out=tx_tr)
        np.clip(tx_te, -10, 10, out=tx_te)
        nets = [
            MLP(seed=s, **config["mlp"]).fit(tx_tr, y_tr).predict_proba(tx_te)
            for s in range(config.get("n_nets", 5))
        ]
        probs.append(np.mean(nets, axis=0))
    return np.mean(probs, axis=0)


def logit(p, eps=1e-7):
    """Inverse of the sigmoid, with probabilities clipped to [eps, 1 - eps]"""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def combine(probs, weights):
    """Weighted average of several models' probabilities, taken in logit space
    (averaging log-odds keeps the confident predictions of each model)"""
    w = np.asarray(weights, dtype=np.float64)
    z = sum(wi * logit(p) for wi, p in zip(w / w.sum(), probs))
    return sigmoid(z)


def apply_rehab_rule(prob, x, names):
    """Set P(MICHD) = 1 for people who answered HAREHAB1.

    HAREHAB1 ("Following your heart attack, did you go to any kind of
    outpatient rehabilitation?") is only asked to respondents who reported a
    heart attack, and every one of the 773 training respondents who answered
    it has MICHD. The question is part of an optional module, so it is missing
    for 99.8% of people and was dropped by the old missing-value filter.
    """
    prob = prob.copy()
    prob[~np.isnan(x[:, names.index("HAREHAB1")])] = 1.0
    return prob


def fit_predict_raw(x_tr, y_tr, x_te, names, config):
    """Fit the model described by `config` on raw features; return P(y=1) on x_te

    Handles the preprocessing of each model family:
      - "gbdt": raw columns (tree_features), then the average of `n_models`
        GBDTs with seeds 0..n_models-1 and parameters config["gbdt"];
      - "ensemble": each entry of config["members"] is fitted with its own
        preprocessing and their probabilities are combined with
        config["weights"] (see `combine`);
      - otherwise: one-hot design matrix (fit_preprocessor with the PREP_KEYS
        of config), then `fit_predict`.
    If config["rehab_rule"] is True, `apply_rehab_rule` is applied at the end.
    """
    model = config.get("model", "logistic")
    if model == "ensemble":
        probs = [fit_predict_raw(x_tr, y_tr, x_te, names, m) for m in config["members"]]
        prob = combine(probs, config["weights"])
    elif model == "gbdt":
        prm = fit_tree_features(x_tr, names, exclude=config.get("exclude", ()))
        f_tr, f_te = tree_features(x_tr, prm), tree_features(x_te, prm)
        prob = np.mean(
            [
                GBDT(seed=s, **config["gbdt"]).fit(f_tr, y_tr).predict_proba(f_te)
                for s in range(config.get("n_models", 1))
            ],
            axis=0,
        )
    else:
        prep = {key: config[key] for key in PREP_KEYS if key in config}
        prm = fit_preprocessor(x_tr, names, **prep)
        prob = fit_predict(transform(x_tr, prm), y_tr, transform(x_te, prm), config)
    if config.get("rehab_rule"):
        prob = apply_rehab_rule(prob, x_te, names)
    return prob


def k_fold_indices(n, k, seed):
    """Shuffle 0..n-1 with a fixed seed and split into k folds"""
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def cross_validate(x, y01, names, config, k=5, seed=42, verbose=True):
    """k-fold cross-validation of the full pipeline (preprocessing + model)

    The preprocessing is re-fitted inside each fold on its training part only.
    Out-of-fold probabilities are collected for every training row, so one
    global F1-optimal threshold can be chosen and evaluated on all of them.

    Args:
        x: raw features (N, D)
        y01: labels in {0, 1}
        names: feature names
        config: model settings (see fit_predict_raw and fit_predict: model,
            lambda_, gamma, max_iters, pos_weight, solver, mlp, n_nets, gbdt,
            n_models, members, weights, rehab_rule) and optional
            preprocessing keys (PREP_KEYS)
        k: number of folds
        seed: shuffling seed
        verbose: print one line per fold

    Returns:
        dict with per-fold F1 (at each fold's best threshold), the global
        threshold, the out-of-fold F1 / accuracy at that threshold, and the
        out-of-fold probabilities
    """
    oof = np.zeros(len(y01))
    fold_f1, fold_t = [], []
    for f, va in enumerate(k_fold_indices(len(y01), k, seed)):
        tr = np.setdiff1d(np.arange(len(y01)), va)
        oof[va] = fit_predict_raw(x[tr], y01[tr], x[va], names, config)
        t, f1 = best_threshold(y01[va], oof[va])
        fold_f1.append(f1)
        fold_t.append(t)
        if verbose:
            print(f"  fold {f + 1}/{k}: F1 = {f1:.4f} at t = {t:.3f}")
    return summarize_oof(y01, oof, k, seed, fold_f1, fold_t)


def summarize_oof(y01, oof, k=5, seed=42, fold_f1=None, fold_t=None):
    """Scores of out-of-fold probabilities: per-fold best F1 and global threshold

    (also used to score an ensemble directly from its members' saved
    out-of-fold probabilities, which gives exactly the same numbers as
    re-running the ensemble in cross-validation since every member is
    deterministic given its seeds)
    """
    if fold_f1 is None:
        fold_f1, fold_t = [], []
        for va in k_fold_indices(len(y01), k, seed):
            t, f1 = best_threshold(y01[va], oof[va])
            fold_f1.append(f1)
            fold_t.append(t)
    t_glob, _ = best_threshold(y01, oof)
    pred = (oof >= t_glob).astype(int)
    return dict(
        fold_f1=np.array(fold_f1),
        fold_t=np.array(fold_t),
        threshold=t_glob,
        f1=f1_score(y01, pred),
        acc=accuracy(y01, pred),
        oof=oof,
    )
