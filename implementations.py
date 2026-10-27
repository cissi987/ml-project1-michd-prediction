"""The six required ML methods for Project 1 (CS-433).

Conventions (from the project description):
- every function returns (w, loss): the last weight vector and its loss;
- MSE uses the 1/2 factor: L(w) = 1/(2N) * ||y - Xw||^2;
- regularized methods return the loss WITHOUT the penalty term;
- all vectors are 1D numpy arrays of shape (D,).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Mean squared error with the 1/2 factor: 1/(2N) * ||y - tx @ w||^2"""
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss with respect to w: -1/N * tx^T (y - tx w)"""
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Numerically stable logistic function 1 / (1 + exp(-t)).
    Uses exp(-|t|), which never overflows, for both signs of t.
    """
    e = np.exp(-np.abs(t))
    return np.where(t >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def _init_weights(initial_w, tx):
    """Copy initial_w as a float vector matching tx's precision (float32 stays float32)"""
    return np.array(initial_w, dtype=np.result_type(tx.dtype, np.float32))


def compute_logistic_loss(y, tx, w):
    """Mean negative log-likelihood for labels y in {0, 1}.
    Uses log(1 + exp(z)) - y*z, computed stably with np.logaddexp.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the mean logistic loss: 1/N * tx^T (sigmoid(tx w) - y)"""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---------------------------------------------------------------------------
# Required functions
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent on the MSE.

    Args:
        y: labels, shape (N,).
        tx: features, shape (N, D).
        initial_w: initial weights, shape (D,).
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: final weights, shape (D,).
        loss: MSE of the final weights (scalar).
    """
    w = _init_weights(initial_w, tx)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, seed=None):
    """Linear regression using stochastic gradient descent (batch size 1).

    At each step one sample is drawn uniformly at random and the gradient of
    its MSE is used for the update.

    Args:
        y: labels, shape (N,).
        tx: features, shape (N, D).
        initial_w: initial weights, shape (D,).
        max_iters: number of SGD steps.
        gamma: step size.
        seed: optional random seed for reproducibility.

    Returns:
        w: final weights, shape (D,).
        loss: MSE of the final weights on the full data (scalar).
    """
    rng = np.random.default_rng(seed)
    w = _init_weights(initial_w, tx)
    n = len(y)
    for _ in range(max_iters):
        i = rng.integers(n)
        grad = compute_mse_gradient(y[i : i + 1], tx[i : i + 1], w)
        w = w - gamma * grad
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression via the normal equations tx^T tx w = tx^T y.

    Args:
        y: labels, shape (N,).
        tx: features, shape (N, D).

    Returns:
        w: optimal weights, shape (D,).
        loss: MSE of the optimal weights (scalar).
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression via the normal equations.

    Minimizes 1/(2N)||y - tx w||^2 + lambda_ ||w||^2, whose solution is
    (tx^T tx + 2 N lambda_ I) w = tx^T y.

    Args:
        y: labels, shape (N,).
        tx: features, shape (N, D).
        lambda_: regularization strength.

    Returns:
        w: optimal weights, shape (D,).
        loss: MSE WITHOUT the penalty term (scalar).
    """
    n, d = tx.shape
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: binary labels in {0, 1}, shape (N,).
        tx: features, shape (N, D).
        initial_w: initial weights, shape (D,).
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: final weights, shape (D,).
        loss: mean negative log-likelihood of the final weights (scalar).
    """
    w = _init_weights(initial_w, tx)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularized logistic regression using gradient descent.

    Minimizes the mean negative log-likelihood + lambda_ ||w||^2
    (gradient of the penalty is 2 lambda_ w).

    Args:
        y: binary labels in {0, 1}, shape (N,).
        tx: features, shape (N, D).
        lambda_: regularization strength.
        initial_w: initial weights, shape (D,).
        max_iters: number of gradient steps.
        gamma: step size.

    Returns:
        w: final weights, shape (D,).
        loss: logistic loss WITHOUT the penalty term (scalar).
    """
    w = _init_weights(initial_w, tx)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    return w, compute_logistic_loss(y, tx, w)
