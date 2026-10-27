"""Gradient-boosted decision trees (GBDT) for binary classification, in NumPy.

This is a from-scratch implementation of second-order ("Newton") gradient
boosting with histogram-based split finding, in the spirit of XGBoost and
LightGBM, written only with NumPy:

- Binning: every feature is discretized once into at most `max_bins` bins
  (exact values for features with few distinct values, quantiles otherwise)
  Missing values (NaN) get their own bin 0, so no imputation is needed.
- Loss: logistic loss on the raw score F(x); each tree is fitted to the
  gradient g = p - y and hessian h = p (1 - p) of the loss at the current
  prediction p = sigmoid(F)
- Split finding: for a node, the per-bin sums of g and h are accumulated in a
  histogram (np.bincount); every threshold of every feature is then scored
  at once with the Newton gain
      G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda)
  Missing values are sent to the side (left or right) that gives the larger
  gain (learned default direction, as in XGBoost)
- Histogram subtraction: only the smaller child's histogram is computed, the
  larger one is parent - smaller
- Leaf values: -G / (H + lambda), scaled by the learning rate
- Regularization: depth limit, minimum hessian per child, L2 penalty lambda
  on leaf values, row subsampling and column subsampling per tree
"""

import numpy as np

from implementations import sigmoid


class Binner:
    """Discretize each column of a float matrix into uint8 bins (NaN -> bin 0)
    Args:
        max_bins: maximum number of bins for non-missing values (<= 255)
    """

    def __init__(self, max_bins=64):
        self.max_bins = max_bins

    def fit(self, x):
        """Learn the bin edges of every column of x (N, F) from training rows"""
        self.edges = []
        for j in range(x.shape[1]):
            c = x[:, j]
            v = c[~np.isnan(c)]
            if len(v) == 0:
                self.edges.append(np.array([], dtype=np.float64))
                continue
            u = np.unique(v)
            if len(u) <= self.max_bins:
                e = (u[:-1] + u[1:]) / 2.0  # one bin per distinct value
            else:
                q = np.quantile(v, np.linspace(0, 1, self.max_bins + 1)[1:-1])
                e = np.unique(q)
            self.edges.append(e.astype(np.float64))
        return self

    def transform(self, x):
        """Bin x (N, F) -> uint8 array of shape (F, N) (feature-major, so that
        one feature's bins are contiguous in memory)"""
        out = np.zeros((x.shape[1], x.shape[0]), dtype=np.uint8)
        for j, e in enumerate(self.edges):
            c = x[:, j]
            miss = np.isnan(c)
            b = np.searchsorted(e, c, side="right") + 1  # bins 1..len(e)+1
            b[miss] = 0
            out[j] = b
        return out


class Tree:
    """Arrays describing one fitted regression tree.
    Node i is a leaf when feature[i] == -1. Otherwise a row goes to left[i]
    when its bin b of feature[i] satisfies 1 <= b <= threshold[i], or when
    b == 0 (missing) and missing_left[i] is True; else it goes to right[i]
    """

    def __init__(self, feature, threshold, missing_left, left, right, value):
        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold, dtype=np.int64)
        self.missing_left = np.array(missing_left, dtype=bool)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.value = np.array(value, dtype=np.float64)

    def predict(self, xb, max_depth):
        """Leaf value of every column of the binned matrix xb (F, N)"""
        n = xb.shape[1]
        node = np.zeros(n, dtype=np.int64)
        cols = np.arange(n)
        for _ in range(max_depth):
            f = self.feature[node]
            inner = f >= 0
            if not inner.any():
                break
            b = xb[np.where(inner, f, 0), cols]
            go_left = np.where(
                b == 0, self.missing_left[node], b <= self.threshold[node]
            )
            nxt = np.where(go_left, self.left[node], self.right[node])
            node = np.where(inner, nxt, node)
        return self.value[node]


class GBDT:
    """Gradient-boosted trees for binary labels y in {0, 1}
    Args:
        n_trees: number of boosting rounds
        learning_rate: shrinkage applied to every tree
        max_depth: maximum depth of each tree
        min_child_weight: minimum sum of hessians in each child of a split
        reg_lambda: L2 penalty on leaf values
        min_split_gain: minimum Newton gain to accept a split
        subsample: fraction of rows used to grow each tree
        colsample: fraction of features considered by each tree
        max_bins: number of histogram bins per feature
        seed: random seed (row / column subsampling)
    """

    def __init__(
        self,
        n_trees=400,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5.0,
        reg_lambda=5.0,
        min_split_gain=0.0,
        subsample=0.7,
        colsample=0.5,
        max_bins=64,
        seed=0,
    ):
        self.n_trees, self.learning_rate = n_trees, learning_rate
        self.max_depth, self.min_child_weight = max_depth, min_child_weight
        self.reg_lambda, self.min_split_gain = reg_lambda, min_split_gain
        self.subsample, self.colsample = subsample, colsample
        self.max_bins, self.seed = max_bins, seed

    # ------------------------------------------------------------------
    # Histograms and split search
    # ------------------------------------------------------------------
    def _hist(self, xb, g, h, rows, feats):
        """Per-bin sums of g and h for the given rows and features: (F', B) each"""
        nb = self.max_bins + 2
        gr, hr = g[rows], h[rows]
        G = np.empty((len(feats), nb))
        H = np.empty((len(feats), nb))
        for i, f in enumerate(feats):
            b = xb[f][rows]
            G[i] = np.bincount(b, weights=gr, minlength=nb)
            H[i] = np.bincount(b, weights=hr, minlength=nb)
        return G, H

    def _best_split(self, G, H, g_tot, h_tot):
        """Best (feature index, threshold bin, missing_left, gain) or None"""
        lam, mcw = self.reg_lambda, self.min_child_weight
        g_miss, h_miss = G[:, :1], H[:, :1]
        gl = np.cumsum(G[:, 1:], axis=1)  # left = bins 1..k+1 (non-missing)
        hl = np.cumsum(H[:, 1:], axis=1)
        parent = g_tot**2 / (h_tot + lam)
        best = None
        for miss_left in (False, True):
            GL = gl + g_miss if miss_left else gl
            HL = hl + h_miss if miss_left else hl
            GR, HR = g_tot - GL, h_tot - HL
            gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - parent
            gain[(HL < mcw) | (HR < mcw)] = -np.inf
            k = int(np.argmax(gain))
            fi, ti = divmod(k, gain.shape[1])
            if best is None or gain[fi, ti] > best[3]:
                best = (fi, ti + 1, miss_left, gain[fi, ti])
        if not np.isfinite(best[3]) or best[3] <= self.min_split_gain:
            return None
        return best

    def _grow(self, xb, g, h, rows, feats):
        """Grow one tree on the given rows / features (depth-wise, with a stack)"""
        lam = self.reg_lambda
        feature, threshold, miss, left, right, value = [-1], [0], [False], [-1], [-1], [0.0]
        G, H = self._hist(xb, g, h, rows, feats)
        stack = [(0, rows, G, H, 0)]
        while stack:
            nid, r, G, H, depth = stack.pop()
            g_tot, h_tot = g[r].sum(), h[r].sum()
            value[nid] = -g_tot / (h_tot + lam) * self.learning_rate
            if depth >= self.max_depth or G is None:
                continue
            split = self._best_split(G, H, g_tot, h_tot)
            if split is None:
                continue
            fi, t, mleft, _ = split
            f = feats[fi]
            b = xb[f][r]
            go_left = np.where(b == 0, mleft, b <= t)
            rl, rr = r[go_left], r[~go_left]
            lid = len(feature)
            for _ in range(2):
                feature.append(-1), threshold.append(0), miss.append(False)
                left.append(-1), right.append(-1), value.append(0.0)
            feature[nid], threshold[nid], miss[nid] = f, t, mleft
            left[nid], right[nid] = lid, lid + 1
            if depth + 1 < self.max_depth:
                # Histogram subtraction: compute the smaller child only.
                if len(rl) <= len(rr):
                    Gl, Hl = self._hist(xb, g, h, rl, feats)
                    Gr, Hr = G - Gl, H - Hl
                else:
                    Gr, Hr = self._hist(xb, g, h, rr, feats)
                    Gl, Hl = G - Gr, H - Hr
            else:
                Gl = Hl = Gr = Hr = None  # children are leaves
            stack.append((lid, rl, Gl, Hl, depth + 1))
            stack.append((lid + 1, rr, Gr, Hr, depth + 1))
        return Tree(feature, threshold, miss, left, right, value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, x, y, x_eval=None, eval_every=50, verbose=False):
        """Fit on x (N, F) float with NaN for missing values and y in {0, 1}.
        If x_eval is given, the eval probabilities after every `eval_every`
        trees are stored in self.staged_eval (dict n_trees -> probabilities),
        which is used to choose the number of trees by cross-validation
        """
        rng = np.random.default_rng(self.seed)
        self.binner = Binner(self.max_bins).fit(x)
        xb = self.binner.transform(x)
        xb_eval = self.binner.transform(x_eval) if x_eval is not None else None
        y = y.astype(np.float64)
        n, d = len(y), xb.shape[0]
        p0 = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.base_score = float(np.log(p0 / (1 - p0)))
        raw = np.full(n, self.base_score)
        raw_eval = np.full(xb_eval.shape[1], self.base_score) if xb_eval is not None else None
        self.trees, self.staged_eval = [], {}
        n_feat = max(1, int(round(self.colsample * d)))
        for t in range(self.n_trees):
            p = sigmoid(raw)
            g, h = p - y, p * (1 - p)
            if self.subsample < 1:
                rows = np.flatnonzero(rng.random(n) < self.subsample)
            else:
                rows = np.arange(n)
            feats = np.sort(rng.choice(d, n_feat, replace=False))
            tree = self._grow(xb, g, h, rows, feats)
            self.trees.append(tree)
            raw += tree.predict(xb, self.max_depth)
            if xb_eval is not None:
                raw_eval += tree.predict(xb_eval, self.max_depth)
                if (t + 1) % eval_every == 0 or t + 1 == self.n_trees:
                    self.staged_eval[t + 1] = sigmoid(raw_eval)
            if verbose and (t + 1) % 50 == 0:
                loss = np.mean(np.logaddexp(0, raw) - y * raw)
                print(f"    tree {t + 1}: train loss {loss:.4f}", flush=True)
        return self

    def decision_function(self, x, n_trees=None):
        """Raw score F(x) using the first n_trees trees (default: all)"""
        xb = self.binner.transform(x)
        raw = np.full(xb.shape[1], self.base_score)
        for tree in self.trees[:n_trees]:
            raw += tree.predict(xb, self.max_depth)
        return raw

    def predict_proba(self, x, n_trees=None):
        """Predicted probabilities P(y = 1 | x)"""
        return sigmoid(self.decision_function(x, n_trees))

    def feature_importance(self, n_features):
        """Number of splits made on each feature (a simple importance measure)"""
        imp = np.zeros(n_features)
        for tree in self.trees:
            f = tree.feature[tree.feature >= 0]
            np.add.at(imp, f, 1)
        return imp
