"""All experiments reported in the paper (5-fold cross-validation).

Usage:
    python experiments.py                # ablation + weights + 6 required methods
    python experiments.py ablation       # only the ablation study
    python experiments.py ablation 8 9   # re-run only some ablation rows
    python experiments.py weights        # ensemble-weight search on saved OOF
    python experiments.py leakage        # final models without HAREHAB1
    python experiments.py methods        # only the comparison of the 6 required methods
    python experiments.py figures        # only redraw the figures

Results are written to results/results.json, out-of-fold probabilities of
every ablation row to results/oof/rowXX.npy, and figures to figures/.
run.py uses the configuration and threshold selected here (FINAL_CONFIG).

Every fold re-fits the whole preprocessing on its training part only; the
same 5 folds (seed 42) are used for every row, so rows are directly
comparable. Each row prints the mean +- std of the per-fold F1 and the F1 /
accuracy of all out-of-fold probabilities at one global F1-optimal threshold.
"""

import json
import os
import sys
import time

import numpy as np

from cross_validation import (
    apply_rehab_rule,
    best_threshold,
    combine,
    cross_validate,
    f1_score,
    k_fold_indices,
    summarize_oof,
)
from data_loading import load_data
from gradient_boosting import GBDT
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
    sigmoid,
)
from preprocessing import fit_preprocessor, fit_tree_features, transform, tree_features

SEED = 42
K = 5
LOGISTIC = dict(model="logistic", lambda_=1e-6, gamma=0.5, max_iters=300)
MLP_SETTINGS = dict(hidden=64, epochs=4, lr=5e-4, dropout=0.5, weight_decay=1e-4)
RAW = dict(drop_admin=False, clean=False, one_hot=False, interactions=False)

# --- Second round of improvements --------------------------------------------
# Keep every column, even the optional-module questions answered by < 1% of
# people (max_nan=1.0; they become one-hot "answered X" levels), and add the
# count features of preprocessing.engineered_features.
NEW_PREP = dict(max_nan=1.0, engineered=True)
# Logistic regression on the new features, solved exactly by Newton's method.
LOGISTIC_NEWTON = dict(model="logistic", solver="newton", lambda_=1e-4, **NEW_PREP)
# The 5-network ensemble on the new features, then with wider networks trained
# longer with a cosine learning-rate decay.
MLP_NEW = dict(model="mlp", mlp=MLP_SETTINGS, n_nets=5, **NEW_PREP)
MLP_WIDE_SETTINGS = dict(
    hidden=128, epochs=6, lr=1e-3, dropout=0.5, weight_decay=1e-4, cosine=True
)
MLP_WIDE = dict(model="mlp", mlp=MLP_WIDE_SETTINGS, n_nets=5, **NEW_PREP)
# Gradient-boosted trees (gradient_boosting.py) on the raw columns.
GBDT_SETTINGS = dict(
    n_trees=1000,
    learning_rate=0.03,
    max_depth=6,
    min_child_weight=10.0,
    reg_lambda=10.0,
    subsample=0.7,
    colsample=0.4,
    max_bins=64,
)
GBDT_CONFIG = dict(model="gbdt", gbdt=GBDT_SETTINGS, n_models=1)
# Ensemble: average of the networks' and the trees' log-odds. In the weight
# search on the out-of-fold probabilities (`python experiments.py weights`)
# every network weight between 0.3 and 0.65 gives the same F1 (+- 0.0005), so
# we use equal weights; adding the logistic regression as a third member gets a
# weight of ~0, so it is not part of the final ensemble.
ENSEMBLE = dict(
    model="ensemble",
    members=[MLP_WIDE, GBDT_CONFIG],
    weights=[0.5, 0.5],
)

# Each row changes one thing with respect to the previous one.
ABLATION = [
    ("Raw features (median imputation)", {**LOGISTIC, **RAW}),
    ("+ drop admin. / duplicate columns", {**LOGISTIC, **RAW, "drop_admin": True}),
    (
        "+ clean special & frequency codes, log",
        {**LOGISTIC, **RAW, "drop_admin": True, "clean": True},
    ),
    (
        "+ one-hot categorical features",
        {**LOGISTIC, "interactions": False},
    ),
    ("+ age / health interactions", {**LOGISTIC}),
    # Weighted loss: positives weigh ~2.5x after normalisation, so the step is
    # reduced to keep gradient descent stable (gamma = 0.5 diverges here).
    (
        "  class weights (positives x3)",
        {**LOGISTIC, "pos_weight": 3.0, "gamma": 0.15, "max_iters": 600},
    ),
    ("Ensemble of 5 neural networks", dict(model="mlp", mlp=MLP_SETTINGS, n_nets=5)),
    (
        "Blend: logistic + 5 networks",
        {**LOGISTIC, "model": "blend", "mlp": MLP_SETTINGS, "n_nets": 5},
    ),
    # ---- second round (rows 8-15) ----
    ("+ keep rare module questions (logistic)", {**LOGISTIC, "max_nan": 1.0}),
    ("+ count features (logistic)", {**LOGISTIC, **NEW_PREP}),
    ("  Newton solver, lambda=1e-4 (logistic)", LOGISTIC_NEWTON),
    ("5 neural networks, new features", MLP_NEW),
    ("  wider networks, cosine LR decay", MLP_WIDE),
    ("Gradient-boosted trees (NumPy)", GBDT_CONFIG),
    ("Ensemble: networks + trees", ENSEMBLE),
    ("+ HAREHAB1 rule (final model)", {**ENSEMBLE, "rehab_rule": True}),
]
FIRST_ROUND = 8  # rows before this index are the first version of the pipeline
# Final model: ensemble of networks and trees + the HAREHAB1 rule.
FINAL_CONFIG = ABLATION[-1][1]

# Leakage check (`python experiments.py leakage`): the final models with the
# HAREHAB1 column removed everywhere (also from the n_missing count), to
# measure how much of the score comes from this skip-logic question.
NO_HAREHAB = dict(exclude=("HAREHAB1",))
LEAKAGE_CHECK = [
    ("Wider networks without HAREHAB1", {**MLP_WIDE, **NO_HAREHAB}),
    ("GBDT without HAREHAB1", {**GBDT_CONFIG, **NO_HAREHAB}),
]
LEAKAGE_CHECK.append(
    (
        "Ensemble without HAREHAB1",
        dict(
            model="ensemble",
            members=[LEAKAGE_CHECK[0][1], LEAKAGE_CHECK[1][1]],
            weights=ENSEMBLE["weights"],
        ),
    )
)


def oof_path(i, prefix="row"):
    """File holding the out-of-fold probabilities of row i of a table
    (prefix "row": ABLATION, prefix "leak": LEAKAGE_CHECK)"""
    return os.path.join("results", "oof", f"{prefix}{i:02d}.npy")


def ensemble_oof(cfg, x, names, table=ABLATION, prefix="row"):
    """Out-of-fold probabilities of an ensemble row from its members' saved
    out-of-fold probabilities (None if a member has not been run yet).

    Identical to cross-validating the ensemble itself (same folds, and every
    member is deterministic given its seeds), but avoids training each
    member twice.
    """
    parts = []
    for member in cfg["members"]:
        i = next((k for k, (_, c) in enumerate(table) if c == member), None)
        if i is None or not os.path.exists(oof_path(i, prefix)):
            return None
        parts.append(np.load(oof_path(i, prefix)).astype(np.float64))
    oof = combine(parts, cfg["weights"])
    if cfg.get("rehab_rule"):
        oof = apply_rehab_rule(oof, x, names)
    return oof


def cross_validate_gbdt(x, y01, names, cfg, k=K, seed=SEED, every=50):
    """Cross-validation of a single GBDT that also records the out-of-fold
    probabilities after every `every` trees (to choose the number of trees)
    and the split counts of every feature (importance)."""
    staged, importance = {}, {}
    fold_f1, fold_t = [], []
    oof = np.zeros(len(y01))
    for f, va in enumerate(k_fold_indices(len(y01), k, seed)):
        tr = np.setdiff1d(np.arange(len(y01)), va)
        prm = fit_tree_features(x[tr], names)
        model = GBDT(seed=0, **cfg["gbdt"]).fit(
            tree_features(x[tr], prm),
            y01[tr],
            x_eval=tree_features(x[va], prm),
            eval_every=every,
        )
        for n_trees, p in model.staged_eval.items():
            staged.setdefault(n_trees, np.zeros(len(y01)))[va] = p
        oof[va] = model.staged_eval[cfg["gbdt"]["n_trees"]]
        for name, v in zip(prm["columns"], model.feature_importance(len(prm["columns"]))):
            importance[name] = importance.get(name, 0.0) + v / k
        t, f1 = best_threshold(y01[va], oof[va])
        fold_f1.append(f1)
        fold_t.append(t)
        print(f"  fold {f + 1}/{k}: F1 = {f1:.4f} at t = {t:.3f}")
    r = summarize_oof(y01, oof, k, seed, fold_f1, fold_t)
    r["staged"] = {
        int(n): float(best_threshold(y01, p)[1]) for n, p in sorted(staged.items())
    }
    r["staged_oof"] = staged
    r["importance"] = importance
    return r


def run_ablation(x, y01, names, results, only=None, key="ablation"):
    """Cross-validate the rows of ABLATION (key "ablation") or LEAKAGE_CHECK
    (key "leakage"): all of them, or the indices in `only`.

    Rows are stored in results[key] (list indexed like the table) and their
    out-of-fold probabilities in results/oof/.
    """
    table, prefix = (ABLATION, "row") if key == "ablation" else (LEAKAGE_CHECK, "leak")
    os.makedirs(os.path.join("results", "oof"), exist_ok=True)
    for i, (label, cfg) in enumerate(table):
        if only is not None and i not in only:
            continue
        t0 = time.time()
        print(f"\n[{i}] {label}")
        oof = None
        if cfg["model"] == "ensemble":
            oof = ensemble_oof(cfg, x, names, table, prefix)
        if oof is not None:
            r = summarize_oof(y01, oof, K, SEED)
            print("  (from the members' saved out-of-fold probabilities)")
        elif cfg["model"] == "gbdt" and key == "ablation" and cfg.get("n_models", 1) == 1:
            r = cross_validate_gbdt(x, y01, names, cfg)
            update_results(
                results, gbdt_trees=r["staged"], gbdt_importance=r["importance"]
            )
            np.savez_compressed(
                os.path.join("results", "oof", f"row{i:02d}_staged.npz"),
                **{str(n): p.astype(np.float32) for n, p in r["staged_oof"].items()},
            )
        else:
            r = cross_validate(x, y01, names, cfg, k=K, seed=SEED)
        np.save(oof_path(i, prefix), r["oof"].astype(np.float32))
        row = dict(
            label=label,
            config=cfg,
            f1_mean=float(r["fold_f1"].mean()),
            f1_std=float(r["fold_f1"].std()),
            oof_f1=float(r["f1"]),
            oof_acc=float(r["acc"]),
            threshold=float(r["threshold"]),
            minutes=(time.time() - t0) / 60,
        )
        print(
            f"  => F1 {row['f1_mean']:.4f} +- {row['f1_std']:.4f}, "
            f"out-of-fold F1 {r['f1']:.4f} / acc {r['acc']:.4f} at t={r['threshold']:.3f}"
        )
        update_results(results, row=(i, row), key=key)
    return results[key]


def fast_best_f1(y01, score):
    """Best F1 over all thresholds of a score, by sorting (exact and fast)"""
    order = np.argsort(-score)
    tp = np.cumsum(y01[order])
    f1 = 2 * tp / (np.arange(1, len(score) + 1) + y01.sum())
    return float(f1.max())


def simplex_grid(n, step):
    """All weight vectors of length n with entries in {0, step, 2 step, ...} summing to 1"""
    m = int(round(1 / step))

    def rec(k, left):
        if k == 1:
            yield (left,)
            return
        for a in range(left + 1):
            for rest in rec(k - 1, left - a):
                yield (a,) + rest

    return [[a / m for a in w] for w in rec(n, m)]


def run_weights(x, y01, names, members=None, step=0.05):
    """Grid search of the ensemble weights (simplex, step 0.05) on the saved
    out-of-fold probabilities of the members (default: ENSEMBLE["members"],
    plus the Newton logistic regression to check whether it helps); returns
    the table that justifies ENSEMBLE["weights"]."""
    if members is None:
        members = ENSEMBLE["members"] + [LOGISTIC_NEWTON]
    rows = [next(k for k, (_, c) in enumerate(ABLATION) if c == m) for m in members]
    parts = [np.load(oof_path(i)).astype(np.float64) for i in rows]
    labels = [ABLATION[i][0].strip() for i in rows]
    grid = []
    for w in simplex_grid(len(parts), step):
        s = apply_rehab_rule(combine(parts, w), x, names)
        grid.append(dict(weights=[round(v, 2) for v in w], f1=fast_best_f1(y01, s)))
    grid.sort(key=lambda g: -g["f1"])
    print("  members:", labels)
    for g in grid[:5]:
        print(f"  weights = {g['weights']}: OOF F1 = {g['f1']:.4f}")
    return dict(members=labels, grid=grid[:20])


def lipschitz(tx, n_iter=30, seed=SEED):
    """Largest eigenvalue of tx^T tx / N (Lipschitz constant of the MSE gradient)"""
    v = np.random.default_rng(seed).standard_normal(tx.shape[1]).astype(tx.dtype)
    for _ in range(n_iter):
        v = tx.T @ (tx @ v) / len(tx)
        lam = np.linalg.norm(v)
        v /= lam
    return float(lam)


def run_methods(x, y01, names):
    """5-fold CV of the 6 required methods on the final features.

    Regression methods are trained on y in {0, 1}; their score tx @ w is
    thresholded like a probability. Each fold uses its best F1 threshold.
    """

    def gd_mse(y, tx):
        # Step 1/L, with L the largest eigenvalue of tx^T tx / N (power iteration):
        # one-hot columns are correlated and L is about 57, so gamma = 0.1 diverges.
        return mean_squared_error_gd(
            y, tx, np.zeros(tx.shape[1], tx.dtype), 300, 1.0 / lipschitz(tx)
        )

    methods = {
        "GD (MSE)": gd_mse,
        "SGD (MSE)": lambda y, tx: mean_squared_error_sgd(
            y, tx, np.zeros(tx.shape[1], tx.dtype), 3 * len(y), 1e-4, seed=SEED
        ),
        "Least squares": lambda y, tx: least_squares(
            y.astype(np.float64), tx.astype(np.float64)
        ),
        "Ridge": lambda y, tx: ridge_regression(
            y.astype(np.float64), tx.astype(np.float64), 1e-5
        ),
        "Logistic": lambda y, tx: logistic_regression(
            y, tx, np.zeros(tx.shape[1], tx.dtype), 300, 0.5
        ),
        "Reg. logistic": lambda y, tx: reg_logistic_regression(
            y, tx, 1e-6, np.zeros(tx.shape[1], tx.dtype), 300, 0.5
        ),
    }
    scores = {m: [] for m in methods}
    for f, va in enumerate(k_fold_indices(len(y01), K, SEED)):
        tr = np.setdiff1d(np.arange(len(y01)), va)
        prm = fit_preprocessor(x[tr], names)
        tx_tr, tx_va = transform(x[tr], prm), transform(x[va], prm)
        y_tr = y01[tr].astype(np.float32)
        for name, fit in methods.items():
            t0 = time.time()
            try:
                w, _ = fit(y_tr, tx_tr)
            except np.linalg.LinAlgError:
                # One-hot levels + bias are collinear: X^T X can be singular.
                print(f"  fold {f + 1}: {name:14s} singular matrix")
                scores[name].append(np.nan)
                continue
            s = tx_va @ w.astype(tx_va.dtype)
            if not np.all(np.isfinite(s)):
                print(f"  fold {f + 1}: {name:14s} diverged")
                scores[name].append(np.nan)
                continue
            if "ogistic" in name:
                s = sigmoid(s)
            s = (s - s.min()) / (s.max() - s.min() + 1e-12)  # map scores to [0, 1]
            grid = np.quantile(s, np.linspace(0.5, 0.995, 200))
            f1 = max(f1_score(y01[va], (s >= t).astype(int)) for t in grid)
            scores[name].append(f1)
            print(f"  fold {f + 1}: {name:14s} F1 = {f1:.4f} ({time.time() - t0:.0f}s)")
    return {
        m: dict(f1_mean=float(np.nanmean(v)), f1_std=float(np.nanstd(v)))
        for m, v in scores.items()
    }


def make_figures(results, y01):
    """Figures for the report (PDF in figures/)"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    )
    os.makedirs("figures", exist_ok=True)
    rows = [r for r in results.get("ablation", []) if r is not None]
    if rows:
        idx = [i for i, r in enumerate(results["ablation"]) if r is not None]
        colors = [
            "#3d6fb6" if i < 6 else "#edae49" if i < FIRST_ROUND else "#2a9d8f"
            for i in idx
        ]
        fig, ax = plt.subplots(figsize=(3.6, 0.22 * len(rows) + 0.7))
        yy = np.arange(len(rows))[::-1]
        ax.barh(
            yy,
            [r["oof_f1"] for r in rows],
            xerr=[r["f1_std"] for r in rows],
            color=colors,
            capsize=2,
        )
        ax.set_yticks(yy)
        ax.set_yticklabels([r["label"].strip() for r in rows], fontsize=6.5)
        lo = min(r["oof_f1"] for r in rows)
        ax.set_xlim(lo - 0.01, max(r["oof_f1"] for r in rows) + 0.008)
        ax.set_xlabel("out-of-fold F1 (5-fold CV; bars: std over folds)")
        fig.tight_layout()
        fig.savefig("figures/ablation.pdf")
        plt.close(fig)
    final = len(ABLATION) - 1
    if os.path.exists(oof_path(final)):
        oof = np.load(oof_path(final)).astype(np.float64)
        ts = np.arange(0.05, 0.6, 0.005)
        f1s, prec, rec = [], [], []
        for t in ts:
            p = (oof >= t).astype(int)
            tp = np.sum((p == 1) & (y01 == 1))
            f1s.append(f1_score(y01, p))
            prec.append(tp / max(p.sum(), 1))
            rec.append(tp / y01.sum())
        fig, ax = plt.subplots(figsize=(3.4, 2.2))
        ax.plot(ts, prec, label="precision", color="#3d6fb6")
        ax.plot(ts, rec, label="recall", color="#d1495b")
        ax.plot(ts, f1s, label="F1", color="#edae49", lw=2)
        t_best = ts[int(np.argmax(f1s))]
        ax.axvline(t_best, ls="--", color="grey", lw=0.8)
        ax.set_xlabel("decision threshold on predicted probability")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig("figures/threshold_final.pdf")
        plt.close(fig)
    if "gbdt_trees" in results:
        st = results["gbdt_trees"]
        n = sorted(int(k) for k in st)
        fig, ax = plt.subplots(figsize=(3.4, 2.0))
        ax.plot(n, [st[str(k)] if str(k) in st else st[k] for k in n], marker="o", ms=3, color="#2a9d8f")
        ax.set_xlabel("number of trees")
        ax.set_ylabel("out-of-fold F1")
        fig.tight_layout()
        fig.savefig("figures/gbdt_trees.pdf")
        plt.close(fig)
    if "gbdt_importance" in results:
        imp = results["gbdt_importance"]
        top = sorted(imp, key=lambda c: -imp[c])[:20][::-1]
        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.barh(top, [imp[c] for c in top], color="#2a9d8f")
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("number of splits (mean over folds)")
        fig.tight_layout()
        fig.savefig("figures/gbdt_importance.pdf")
        plt.close(fig)


RESULTS = os.path.join("results", "results.json")


def load_results():
    """Current content of results/results.json ({} if it does not exist)"""
    if not os.path.exists(RESULTS):
        return {}
    with open(RESULTS) as f:
        return json.load(f)


def update_results(results, row=None, key="ablation", **entries):
    """Merge new entries (and/or one row (i, row_dict) of results[key]) into
    results/results.json, re-reading the file first so that several runs of
    this script (e.g. different ablation rows in parallel) do not overwrite
    each other. `results` (the in-memory copy) is updated too."""
    os.makedirs("results", exist_ok=True)
    disk = load_results()
    disk.update(entries)
    if row is not None:
        i, r = row
        rows = disk.get(key, [])
        n_rows = len(ABLATION) if key == "ablation" else len(LEAKAGE_CHECK)
        rows += [None] * (n_rows - len(rows))
        rows[i] = r
        disk[key] = rows
    with open(RESULTS, "w") as f:
        json.dump(disk, f, indent=1)
    results.clear()
    results.update(disk)


def main():
    """Run the requested experiments and save results and figures"""
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = load_results()
    x, _, y, _, _, names = load_data()
    y01 = (y == 1).astype(np.float64)
    if what in ("all", "ablation"):
        only = [int(i) for i in sys.argv[2:]] or None
        run_ablation(x, y01, names, results, only)
    if what in ("all", "leakage"):
        only = [int(i) for i in sys.argv[2:]] if what == "leakage" else []
        run_ablation(x, y01, names, results, only or None, key="leakage")
    if what in ("all", "weights"):
        update_results(results, weights=run_weights(x, y01, names))
    if what in ("all", "methods"):
        update_results(results, methods=run_methods(x, y01, names))
    make_figures(results, y01)


if __name__ == "__main__":
    main()
