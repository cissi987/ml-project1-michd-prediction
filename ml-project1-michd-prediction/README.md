# CS-433 Project 1 — Predicting coronary heart disease (MICHD) from the BRFSS survey

Binary classification of MICHD (coronary heart disease, heart attack or angina) for 109,379 survey
respondents, from 321 answers of the 2015 Behavioral Risk Factor Surveillance System. Only Python,
NumPy and matplotlib (figures only) are used, as the project rules require.

**Final model (AIcrowd submission):** an ensemble of 5 one-hidden-layer neural networks and
gradient-boosted decision trees, both implemented from scratch in NumPy, averaged in log-odds space,
plus a rule derived from the survey's skip logic.
5-fold cross-validated F1 = **0.4476** (accuracy 0.876), vs **0.4339** for our previous
submission (ensemble of 5 networks, AIcrowd F1 0.448).
On the AIcrowd test set the final model scores **F1 = 0.459, accuracy = 0.887** (1st place on 22 October 2026).

## Reproducing the submission

```
data/x_train.csv, data/y_train.csv, data/x_test.csv   # course dataset (not in the repository)
python run.py                                         # writes submission.csv
```

`run.py` trains the final model on the whole training set and writes `submission.csv` with
`helpers.create_csv_submission` (about 13 minutes and 3 GB of RAM on a 2-core machine). All random seeds
are fixed, so on a given machine it reproduces the submitted file exactly. On a different platform (CPU and
BLAS library), float32 rounding changes about 0.3% of the predictions (306 of 109,379 rows differed between our
Linux and macOS runs). The first run parses the CSV files and caches
them in `data/cache.npz` (a few seconds on later runs).

All cross-validation experiments of the report (ablation study, choice of the number of trees, ensemble
weights, comparison of the 6 required methods) are produced by

```
python experiments.py            # everything (a few hours on 2 CPU cores)
python experiments.py ablation 13  # a single ablation row (see the list in experiments.py)
python experiments.py weights    # ensemble-weight search from the saved out-of-fold predictions
python experiments.py leakage    # the final models without the HAREHAB1 column (leakage check)
python experiments.py methods    # the 6 required methods
```

Results go to `results/results.json` (scores and configurations) and `results/oof/` (out-of-fold
probabilities of each row); figures go to `figures/`.
`project_overview.ipynb` walks through the whole project with figures (exploratory analysis, the first
logistic-regression pipeline, then the second round of improvements and the ablation).

## Files

| File | Content |
| --- | --- |
| `implementations.py` | The 6 required methods (GD / SGD for MSE, least squares, ridge, logistic and regularized logistic regression). |
| `helpers.py` | Course helpers (CSV loading, `create_csv_submission`). |
| `data_loading.py` | Fast CSV parsing into float32 arrays, cached in `data/cache.npz`. |
| `preprocessing.py` | Feature engineering. One-hot design matrix for linear models / networks (`fit_preprocessor`, `transform`), raw-column matrix for the trees (`fit_tree_features`, `tree_features`), count features (`engineered_features`). |
| `neural_net.py` | One-hidden-layer ReLU network, dropout, L2, mini-batch Adam, optional cosine learning-rate decay. |
| `gradient_boosting.py` | Gradient-boosted decision trees: histogram binning, Newton (second-order) boosting, learned direction for missing values, histogram subtraction, row / column subsampling. |
| `cross_validation.py` | F1 / accuracy, threshold selection, logistic regression with class weights or Newton's method, model fitting for every model family (`fit_predict_raw`), ensembles (`combine`), the HAREHAB1 rule, k-fold cross-validation. |
| `experiments.py` | Every experiment of the report; `FINAL_CONFIG` is the configuration used by `run.py`. |
| `run.py` | Reproduces `submission.csv`. |
| `project_overview.ipynb` | Narrated walk-through of the project with figures. |
| `report/` | LaTeX source of the 2-page report (`report.tex`, `literature.bib`, IEEEtran files), its figure and tables, and the scripts that regenerate them from `results/results.json` (`python report/make_report_figures.py`, `python report/gen_tables.py`). |

## Pipeline

**1. Preprocessing** (all statistics learned on the training rows of each fold only)

* Drop administrative columns (dates, ids, sampling weights) and raw columns that have a cleaner computed
  version (e.g. `WEIGHT2` → `WTKG3`).
* Numeric columns: BRFSS special codes cleaned using the codebook (`88` = none → 0, `77` / `99` = don't
  know / refused → missing), frequency codes (`1xx` per week, `2xx` per month…) converted to a common unit,
  log of heavy-tailed counts, median imputation plus a "was missing" indicator.
* Categorical columns (≤ 60 values): one-hot encoding, with missing / don't know / refused as their own
  levels (levels seen fewer than 100 times are dropped).
* **Second round.** Keep the optional-module questions answered by < 1% of people (`max_nan=1.0`): the first
  version dropped 30 such columns, among them `HAREHAB1` (*"Following your heart attack, did you go to
  outpatient rehabilitation?"*), which is only asked to people who had a heart attack — all 773 training
  respondents who answered it have MICHD. Add three count features: number of chronic conditions, of
  disabilities and of unanswered questions.
* Age × sex / general-health interactions, standardization, bias column.
* The trees instead receive the raw columns (special codes of numeric columns cleaned, NaN kept): no
  imputation, encoding or scaling is needed.

**2. Models**

* Neural networks: 5 networks (seeds 0–4), 128 ReLU units, dropout 0.5, weight decay 1e-4, Adam with learning
  rate 1e-3 and cosine decay, 6 epochs, batch 1024; their probabilities are averaged.
* Gradient-boosted trees: 1000 trees of depth ≤ 6, learning rate 0.03, L2 penalty λ = 10 on leaf values, minimum hessian 10 per child, 70% of the rows and 40% of the features per tree, 64 bins per feature.
* Ensemble: weighted average of the two models' log-odds, weights 0.5 / 0.5 (networks, trees), chosen by
  a grid search on the out-of-fold predictions.
* HAREHAB1 rule: respondents who answered `HAREHAB1` are predicted positive.

**3. Decision threshold.** With 8.8% positives, the 0.5 threshold is far from F1-optimal. The threshold
0.205 maximizes F1 on the out-of-fold probabilities of the whole pipeline.

**4. Evaluation.** 5-fold cross-validation (seed 42), preprocessing re-fitted inside every fold; we report the
F1 of all out-of-fold predictions at one global threshold (as for the test set) and the mean ± std of the
per-fold F1.

## Ablation study (5-fold CV)

| # | Configuration | Model | Fold F1 (mean ± std) | OOF F1 | OOF acc. | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Raw features (median imputation) | logistic | 0.4161 ± 0.0014 | 0.4155 | 0.8690 | 0.205 |
| 1 | + drop admin. / duplicate columns | logistic | 0.4161 ± 0.0019 | 0.4154 | 0.8715 | 0.210 |
| 2 | + clean special & frequency codes, log | logistic | 0.4177 ± 0.0011 | 0.4173 | 0.8698 | 0.205 |
| 3 | + one-hot categorical features | logistic | 0.4326 ± 0.0014 | 0.4317 | 0.8647 | 0.185 |
| 4 | + age / health interactions | logistic | 0.4323 ± 0.0016 | 0.4313 | 0.8674 | 0.190 |
| 5 | class weights (positives x3) | logistic | 0.4322 ± 0.0016 | 0.4312 | 0.8671 | 0.415 |
| 6 | Ensemble of 5 neural networks | networks | 0.4343 ± 0.0021 | 0.4339 | 0.8795 | 0.215 |
| 7 | Blend: logistic + 5 networks | logistic + networks | 0.4335 ± 0.0016 | 0.4330 | 0.8745 | 0.205 |
| 8 | + keep rare module questions (logistic) | logistic | 0.4406 ± 0.0014 | 0.4395 | 0.8755 | 0.200 |
| 9 | + count features (logistic) | logistic | 0.4406 ± 0.0012 | 0.4395 | 0.8755 | 0.200 |
| 10 | Newton solver, lambda=1e-4 (logistic) | logistic (Newton) | 0.4399 ± 0.0009 | 0.4390 | 0.8770 | 0.205 |
| 11 | 5 neural networks, new features | networks | 0.4420 ± 0.0016 | 0.4411 | 0.8714 | 0.195 |
| 12 | wider networks, cosine LR decay | networks | 0.4457 ± 0.0021 | 0.4451 | 0.8808 | 0.215 |
| 13 | Gradient-boosted trees (NumPy) | GBDT | 0.4476 ± 0.0021 | 0.4469 | 0.8788 | 0.215 |
| 14 | Ensemble: networks + trees | networks + GBDT | 0.4489 ± 0.0014 | 0.4476 | 0.8760 | 0.205 |
| 15 | + HAREHAB1 rule (final model) | networks + GBDT | 0.4489 ± 0.0014 | 0.4476 | 0.8760 | 0.205 |

Rows 0–7 are the first version of the pipeline (row 6 was our previous AIcrowd submission); each later row
changes one thing. Ensemble rows are computed from their members' saved out-of-fold predictions, which is
identical to cross-validating the ensemble since every member is deterministic given its seeds.

## Leakage check

`python experiments.py leakage` repeats the two final models and their ensemble with the `HAREHAB1` column
removed everywhere (also from the `n_missing` count):

| # | Configuration | Fold F1 (mean ± std) | OOF F1 | OOF acc. |
| --- | --- | --- | --- | --- |
| L1 | Wider networks without HAREHAB1 | 0.4368 ± 0.0012 | 0.4366 | 0.8778 |
| L2 | GBDT without HAREHAB1 | 0.4384 ± 0.0024 | 0.4378 | 0.8817 |
| L3 | Ensemble without HAREHAB1 | 0.4399 ± 0.0018 | 0.4392 | 0.8796 |

Of the +0.0138 out-of-fold F1 gained over the previous submission (row 6, which did not see the
column either), +0.0053 comes from all other changes (GBDT, wider networks, ensemble, other module
questions, counts) and +0.0084 from the `HAREHAB1` column alone.

## Note on the HAREHAB1 rule

`HAREHAB1` is part of the survey's skip logic, not a medical risk factor: it is only asked after a reported
heart attack, which is part of the MICHD definition. It is present in the provided test features, so using it
is legitimate for the competition, but the report should present it as a property of the survey design (a form
of label leakage through question routing), not as a clinical finding.
