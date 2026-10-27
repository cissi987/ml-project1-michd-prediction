"""Feature preprocessing for the BRFSS MICHD dataset.

Every statistic (kept levels, medians, means, standard deviations) is learned
on the training rows only by `fit_preprocessor`, then applied to any matrix by
`transform`. The choices below come from the BRFSS 2015 codebook:

- administrative / survey-design columns (dates, ids, sampling weights) and
  raw columns that have a cleaner calculated version are dropped;
- columns with few distinct values are categorical: they are one-hot encoded,
  keeping "don't know" / "refused" codes and missing values as their own
  categories (in a survey, *not answering* is informative);
- numeric columns get their special codes cleaned (e.g. 88 = "none" -> 0,
  77 / 99 = "don't know" / "refused" -> missing), frequency codes such as
  "1xx = x times per week, 2xx = x times per month" are converted to a common
  unit, skewed counts are log-transformed, and missing values are imputed with
  the training median plus a "was missing" indicator column;
- all columns are standardized and a bias column is added.

Second round (options of `fit_preprocessor`, used by the final model):
- max_nan=1.0 keeps the optional-module questions answered by < 1% of people.
  Their answers become one-hot "answered X" columns; in particular HAREHAB1
  (rehabilitation after a heart attack) is only asked to people who had one;
- engineered=True adds counts of chronic conditions, of disabilities and of
  unanswered questions (`engineered_features`).
The gradient-boosted trees use `fit_tree_features` / `tree_features` instead:
raw columns with NaN kept, no one-hot encoding and no scaling.
"""

import numpy as np

# Columns that carry no health information (interview date, ids, survey
# weights, constant phone-screening answers) or duplicate a cleaner column
DROP = {
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO", "_PSU",
    "CTELENUM", "COLGHOUS", "STATERES", "CTELNUM1", "CELLFON2", "CCLGHOUS",
    "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_LLCPWT",
    "_DUALCOR", "QSTVER",
    "WEIGHT2", "HEIGHT3",  # use WTKG3, HTM4, _BMI5 instead
    "FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1",  # use *_DA1_
    "EXRACT11", "EXRACT21",  # activity type, 70+ codes
    "EXERHMM1", "EXERHMM2",  # use PADUR1_, PADUR2_
    "FLSHTMY2", "HIVTSTD3",  # month/year dates
}  # fmt: skip

# Numeric columns: value -> missing (don't know / refused) and value -> 0 (none)
_DAYS = dict(nan=(77, 99), zero=(88,))
NUMERIC = {
    "NUMADULT": {}, "NUMMEN": {}, "NUMWOMEN": {},
    "HHADULT": dict(nan=(77, 99)),
    "PHYSHLTH": _DAYS, "MENTHLTH": _DAYS, "POORHLTH": _DAYS,
    "DIABAGE2": dict(nan=(98, 99)),
    "CHILDREN": dict(nan=(99,), zero=(88,)),
    "AVEDRNK2": dict(nan=(77, 99)), "DRNK3GE5": _DAYS, "MAXDRNKS": dict(nan=(77, 99)),
    "DOCTDIAB": dict(nan=(77, 98, 99), zero=(88,)),
    "CHKHEMO3": dict(nan=(77, 98, 99), zero=(88,)),
    "FEETCHK": dict(nan=(77, 99), zero=(88,)),
    "JOINPAIN": dict(nan=(77, 99)),
    "SCNTWRK1": dict(nan=(97, 99), zero=(98,)), "SCNTLWK1": dict(nan=(97, 99), zero=(98,)),
    "ADPLEASR": _DAYS, "ADDOWN": _DAYS, "ADSLEEP": _DAYS, "ADENERGY": _DAYS,
    "ADEAT1": _DAYS, "ADFAIL": _DAYS, "ADTHINK": _DAYS, "ADMOVE": _DAYS,
    "_AGE80": {}, "HTIN4": {}, "HTM4": {}, "WTKG3": {}, "_BMI5": {},
    "DROCDY3_": dict(nan=(900,)), "_DRNKWEK": dict(nan=(99900,)),
    "FTJUDA1_": {}, "FRUTDA1_": {}, "BEANDAY_": {}, "GRENDAY_": {}, "ORNGDAY_": {},
    "VEGEDA1_": {}, "_FRUTSUM": {}, "_VEGESUM": {},
    "METVL11_": {}, "METVL21_": {},
    "MAXVO2_": dict(nan=(999,)), "FC60_": dict(nan=(999,)),
    "PADUR1_": {}, "PADUR2_": {}, "PAFREQ1_": {}, "PAFREQ2_": {},
    "_MINAC11": {}, "_MINAC21": {}, "STRFREQ_": {},
    "PAMIN11_": {}, "PAMIN21_": {}, "PA1MIN_": {},
    "PAVIG11_": {}, "PAVIG21_": {}, "PA1VIGM_": {},
}  # fmt: skip

# Frequency codes "Uxx": U = unit digit, xx = count. Converted to times per month
# Per feature: {unit digit: multiplier to per-month}, never-codes to 0
_WEEK_MONTH = {1: 30 / 7, 2: 1.0}
_DAY_TO_YEAR = {1: 30.0, 2: 30 / 7, 3: 1.0, 4: 1 / 12}
FREQUENCY = {
    "ALCDAY5": dict(units=_WEEK_MONTH, zero=(888,), nan=(777, 999)),
    "EXEROFT1": dict(units=_WEEK_MONTH, zero=(), nan=(777, 999)),
    "EXEROFT2": dict(units=_WEEK_MONTH, zero=(), nan=(777, 999)),
    "STRENGTH": dict(units=_WEEK_MONTH, zero=(888,), nan=(777, 999)),
    "BLDSUGAR": dict(units=_DAY_TO_YEAR, zero=(888,), nan=(777, 999)),
    "FEETCHK2": dict(units=_DAY_TO_YEAR, zero=(555, 888), nan=(777, 999)),
    "LONGWTCH": dict(units=_DAY_TO_YEAR, zero=(555, 888), nan=(777, 999)),
}

# Heavy-tailed numeric columns that are log1p-transformed
LOG = {
    "PHYSHLTH", "MENTHLTH", "POORHLTH", "AVEDRNK2", "DRNK3GE5", "MAXDRNKS",
    "DROCDY3_", "_DRNKWEK", "PADUR1_", "PADUR2_", "PAFREQ1_", "PAFREQ2_",
    "_MINAC11", "_MINAC21", "STRFREQ_", "PAMIN11_", "PAMIN21_", "PA1MIN_",
    "PAVIG11_", "PAVIG21_", "PA1VIGM_", "ALCDAY5", "EXEROFT1", "EXEROFT2",
    "STRENGTH", "BLDSUGAR", "FEETCHK2", "LONGWTCH", "FTJUDA1_", "FRUTDA1_",
    "BEANDAY_", "GRENDAY_", "ORNGDAY_", "VEGEDA1_", "_FRUTSUM", "_VEGESUM",
}  # fmt: skip

MAX_LEVELS = 60  # a column with more distinct values than this is not categorical
MISSING = -1.0  # level used for NaN in categorical columns (no BRFSS code is negative)

# Yes/no questions (code 1 = "yes") summarized into counts by `engineered_features`
CHRONIC = (
    "CVDSTRK3", "ASTHMA3", "CHCSCNCR", "CHCOCNCR", "CHCCOPD1", "HAVARTH3",
    "ADDEPEV2", "CHCKIDNY", "DIABETE3", "TOLDHI2", "BPHIGH4",
)  # fmt: skip
DISABILITY = ("QLACTLM2", "USEEQUIP", "BLIND", "DECIDE", "DIFFWALK", "DIFFDRES", "DIFFALON")
ENGINEERED = ["n_chronic", "n_disabilities", "n_missing"]


def engineered_features(x, names, exclude=()):
    """Hand-made summary features of a person's answers, shape (N, 3).

    - n_chronic: number of chronic conditions answered "yes" (stroke, asthma,
      cancers, COPD, arthritis, depression, kidney disease, diabetes, high
      cholesterol, high blood pressure). A linear model can only add the
      one-hot columns; trees in particular benefit from having the sum.
    - n_disabilities: number of activity limitations / difficulties ("yes").
    - n_missing: number of unanswered questions. BRFSS uses skip logic
      (follow-up questions are only asked after some answers) and optional
      state modules, so how much is missing says something about the person.
    Columns listed in `exclude` are ignored (used for the leakage check).
    """
    yes = lambda n: (x[:, names.index(n)] == 1).astype(np.float64)  # noqa: E731
    n_chronic = sum(yes(n) for n in CHRONIC)
    n_disab = sum(yes(n) for n in DISABILITY)
    keep = [j for j, n in enumerate(names) if n not in exclude]
    n_missing = np.isnan(x[:, keep]).sum(axis=1).astype(np.float64)
    return np.c_[n_chronic, n_disab, n_missing]


def clean_numeric(col, name):
    """Return a float64 copy of a numeric column with its special codes handled.

    Args:
        col: raw column values (may contain NaN).
        name: feature name, used to look up its codes.

    Returns:
        Cleaned column (special "none" codes -> 0, unknown/refused -> NaN,
        frequency codes converted to times per month, log1p for skewed columns).
    """
    col = col.astype(np.float64).copy()
    if name in FREQUENCY:
        spec = FREQUENCY[name]
        out = np.full_like(col, np.nan)
        out[np.isin(col, spec["zero"])] = 0.0
        unit = np.floor(col / 100)
        for u, mult in spec["units"].items():
            m = (unit == u) & ~np.isin(col, spec["zero"] + spec["nan"])
            out[m] = (col[m] - 100 * u) * mult
        col = out
    else:
        spec = NUMERIC.get(name, {})
        col[np.isin(col, spec.get("nan", ()))] = np.nan
        col[np.isin(col, spec.get("zero", ()))] = 0.0
    if name in LOG:
        col = np.log1p(np.maximum(col, 0))
    return col


def fit_preprocessor(
    x,
    names,
    min_count=100,
    max_nan=0.99,
    interactions=True,
    drop_admin=True,
    clean=True,
    one_hot=True,
    engineered=False,
    exclude=(),
):
    """Learn the preprocessing from training rows.

    Args:
        x: raw training features, shape (N, D)
        names: list of the D feature names
        min_count: a categorical level must appear this many times to get a column
        max_nan: columns missing in more than this fraction of rows are dropped
            (1.0 keeps the rare optional-module questions, e.g. HAREHAB1)
        interactions: add age x sex and age x general-health products
        drop_admin: drop the columns listed in DROP
        clean: handle special codes / frequency codes / log of numeric columns
            (False = use their raw values; only for the ablation study)
        one_hot: one-hot encode categorical columns (False = use their raw
            codes as numbers; only for the ablation study)
        engineered: add the count features of `engineered_features`
        exclude: names of columns to leave out entirely (leakage check)

    Returns:
        A dict describing the transformation, used by `transform`
    """
    cat, num = [], []
    nan_frac = np.mean(np.isnan(x), axis=0)
    for j, name in enumerate(names):
        if (drop_admin and name in DROP) or nan_frac[j] > max_nan or name in exclude:
            continue
        c = np.where(np.isnan(x[:, j]), MISSING, x[:, j])
        levels, counts = np.unique(c, return_counts=True)
        is_num = name in NUMERIC or name in FREQUENCY or len(levels) > MAX_LEVELS + 1
        if is_num or not one_hot:
            do_clean = clean and (name in NUMERIC or name in FREQUENCY)
            c = clean_numeric(x[:, j], name) if do_clean else x[:, j].astype(float)
            med = np.nanmedian(c) if np.any(~np.isnan(c)) else 0.0
            num.append((j, name, med, bool(np.any(np.isnan(c))), do_clean))
            continue
        keep = levels[counts >= min_count]
        if len(keep) >= 2 or (len(keep) == 1 and len(levels) > 1):
            cat.append((j, name, keep))
    prm = dict(
        cat=cat,
        num=num,
        interactions=interactions,
        engineered=engineered,
        exclude=tuple(exclude),
        names=list(names),
    )
    mean, sd = _column_stats(_build(x, prm))
    sd[sd == 0] = 1.0
    mean[0], sd[0] = 0.0, 1.0  # bias column
    prm["mean"], prm["sd"] = mean.astype(np.float32), sd.astype(np.float32)
    prm["columns"] = _column_names(prm)
    return prm


def _column_stats(tx, block=20000):
    """Column means and standard deviations in float64, by row blocks (low memory)"""
    s1 = np.zeros(tx.shape[1])
    s2 = np.zeros(tx.shape[1])
    for i in range(0, len(tx), block):
        b = tx[i : i + block].astype(np.float64)
        s1 += b.sum(axis=0)
        s2 += (b**2).sum(axis=0)
    mean = s1 / len(tx)
    return mean, np.sqrt(np.maximum(s2 / len(tx) - mean**2, 0))


def _build(x, prm):
    """Build the unstandardized float32 design matrix; column 0 is left for the bias"""
    n = x.shape[0]
    n_num = sum(1 + int(has_nan) for _, _, _, has_nan, _ in prm["num"])
    n_cat = sum(len(keep) for _, _, keep in prm["cat"])
    n_int = 6 if prm["interactions"] else 0
    n_eng = len(ENGINEERED) if prm.get("engineered") else 0
    out = np.zeros((n, 1 + n_num + n_cat + n_int + n_eng), dtype=np.float32)
    k = 1  # column 0 is reserved for the bias
    for j, name, med, has_nan, do_clean in prm["num"]:
        c = clean_numeric(x[:, j], name) if do_clean else x[:, j].astype(float)
        miss = np.isnan(c)
        out[:, k] = np.where(miss, med, c)
        k += 1
        if has_nan:
            out[:, k] = miss
            k += 1
    for j, _, keep in prm["cat"]:
        c = np.where(np.isnan(x[:, j]), MISSING, x[:, j])
        out[:, k : k + len(keep)] = c[:, None] == keep[None, :]
        k += len(keep)
    if prm["interactions"]:
        out[:, k : k + n_int] = _interactions(x, prm["names"])
        k += n_int
    if n_eng:
        out[:, k : k + n_eng] = engineered_features(x, prm["names"], prm.get("exclude", ()))
    return out


def _interactions(x, names):
    """Products of age with sex and general health, and squared terms"""
    age = x[:, names.index("_AGE80")].astype(np.float64)
    sex = (x[:, names.index("SEX")] == 1).astype(np.float64)  # 1 = male
    gh = x[:, names.index("GENHLTH")].astype(np.float64)
    gh = np.where(np.isin(gh, (7, 9)) | np.isnan(gh), 3.0, gh)  # unknown -> middle
    bmi = x[:, names.index("_BMI5")].astype(np.float64)
    bmi = np.where(np.isnan(bmi), np.nanmedian(bmi), bmi)
    a = (age - 50) / 18
    return np.c_[a * sex, a * gh, a**2, gh**2, a * a * sex, (bmi / 100) ** 2]


def _column_names(prm):
    """Readable names of the design-matrix columns, bias first"""
    cols = ["bias"]
    for _, name, _, has_nan, _ in prm["num"]:
        cols.append(name)
        if has_nan:
            cols.append(name + "=missing")
    for _, name, keep in prm["cat"]:
        cols += [f"{name}={'missing' if v == MISSING else f'{v:g}'}" for v in keep]
    if prm["interactions"]:
        cols += ["age*male", "age*GENHLTH", "age^2", "GENHLTH^2", "age^2*male", "BMI^2"]
    if prm.get("engineered"):
        cols += ENGINEERED
    return cols


def transform(x, prm):
    """Apply a fitted preprocessor; returns standardized float32 tx with a bias column"""
    tx = _build(x, prm)
    tx -= prm["mean"]
    tx /= prm["sd"]
    tx[:, 0] = 1.0
    return tx


# ---------------------------------------------------------------------------
# Features for the gradient-boosted trees
# ---------------------------------------------------------------------------
# Trees need neither one-hot encoding, imputation nor scaling: a split can
# isolate any code, missing values get their own bin, and splits are invariant
# to monotone transformations. They get the raw columns (special codes of the
# numeric columns cleaned so that their order is meaningful), NaN kept as NaN.


def fit_tree_features(x, names, drop_admin=True, engineered=True, exclude=()):
    """Choose the columns given to the trees (learned on training rows).

    Args:
        x: raw training features (N, D)
        names: feature names
        drop_admin: drop the administrative columns listed in DROP
        engineered: append the count features of `engineered_features`
        exclude: names of columns to leave out entirely (leakage check)

    Returns:
        dict used by `tree_features`
    """
    cols = []
    for j, name in enumerate(names):
        if (drop_admin and name in DROP) or name in exclude:
            continue
        c = x[:, j]
        has_nan = bool(np.isnan(c).any())
        n_values = len(np.unique(c[~np.isnan(c)]))
        if n_values >= 2 or (n_values == 1 and has_nan):  # skip constant columns
            cols.append(j)
    columns = [names[j] for j in cols] + (ENGINEERED if engineered else [])
    return dict(
        cols=cols,
        names=list(names),
        engineered=engineered,
        exclude=tuple(exclude),
        columns=columns,
    )


def tree_features(x, prm):
    """Float32 feature matrix for the trees: (N, len(prm['columns'])), NaN = missing"""
    names = prm["names"]
    n_eng = len(ENGINEERED) if prm["engineered"] else 0
    out = np.empty((len(x), len(prm["cols"]) + n_eng), dtype=np.float32)
    for k, j in enumerate(prm["cols"]):
        name = names[j]
        if name in NUMERIC or name in FREQUENCY:
            out[:, k] = clean_numeric(x[:, j], name)
        else:
            out[:, k] = x[:, j]
    if n_eng:
        out[:, len(prm["cols"]) :] = engineered_features(x, names, prm["exclude"])
    return out
