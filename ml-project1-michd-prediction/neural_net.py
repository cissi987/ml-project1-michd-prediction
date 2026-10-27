"""A one-hidden-layer neural network (MLP) for binary classification, in NumPy.

Architecture: input -> Linear(D, H) -> ReLU -> dropout -> Linear(H, 1) -> sigmoid.
Loss: mean logistic (binary cross-entropy) loss + L2 weight decay.
Optimizer: mini-batch Adam. Everything is float32 to keep memory and time low.
"""

import numpy as np

from implementations import sigmoid


class MLP:
    """One-hidden-layer ReLU network trained with mini-batch Adam.

    Args:
        hidden: number of hidden units H
        lr: Adam learning rate
        epochs: number of passes over the training data
        batch_size: mini-batch size
        weight_decay: L2 penalty on the weight matrices (not on biases)
        dropout: probability of dropping a hidden unit during training
        seed: random seed (initialization, shuffling, dropout)
        cosine: if True, the learning rate decays from lr to 0 along a cosine
            over all training steps (default False = constant learning rate)
    """

    def __init__(
        self,
        hidden=64,
        lr=1e-3,
        epochs=8,
        batch_size=1024,
        weight_decay=1e-5,
        dropout=0.2,
        seed=42,
        cosine=False,
    ):
        self.hidden, self.lr, self.epochs = hidden, lr, epochs
        self.batch_size, self.weight_decay, self.dropout = (
            batch_size,
            weight_decay,
            dropout,
        )
        self.seed = seed
        self.cosine = cosine

    def _forward(self, x):
        """Hidden activations and output logits for a batch"""
        h = np.maximum(x @ self.W1 + self.b1, 0)
        return h, h @ self.w2 + self.b2

    def fit(self, tx, y, callback=None):
        """Train on tx (N, D) float32 and labels y in {0, 1}. Returns self.
        If given, callback(model, epoch) is called after each epoch.
        """
        rng = np.random.default_rng(self.seed)
        n, d = tx.shape
        f32 = np.float32
        self.W1 = (rng.standard_normal((d, self.hidden)) * np.sqrt(2 / d)).astype(f32)
        self.b1 = np.zeros(self.hidden, f32)
        self.w2 = (rng.standard_normal(self.hidden) * np.sqrt(1 / self.hidden)).astype(
            f32
        )
        self.b2 = f32(np.log(y.mean() / (1 - y.mean())))  # start at the base rate
        params = [self.W1, self.b1, self.w2, np.array([self.b2], f32)]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1_, b2_, eps, step = 0.9, 0.999, 1e-8, 0
        keep = 1.0 - self.dropout
        total = self.epochs * int(np.ceil(n / self.batch_size))
        for _ in range(self.epochs):
            order = rng.permutation(n)
            for i in range(0, n, self.batch_size):
                idx = order[i : i + self.batch_size]
                xb, yb = tx[idx], y[idx].astype(f32)
                h, z = self._forward(xb)
                mask = (rng.random(h.shape) < keep).astype(f32) / keep
                hd = h * mask
                z = hd @ self.w2 + params[3][0]
                dz = (sigmoid(z) - yb) / len(idx)  # d loss / d logit
                g_w2 = hd.T @ dz + self.weight_decay * self.w2
                g_b2 = np.array([dz.sum()], f32)
                dh = np.outer(dz, self.w2) * mask * (h > 0)
                g_W1 = xb.T @ dh + self.weight_decay * self.W1
                g_b1 = dh.sum(axis=0)
                step += 1
                lr = self.lr
                if self.cosine:  # cosine decay of the step size
                    lr = 0.5 * self.lr * (1 + np.cos(np.pi * (step - 1) / total))
                for p, g, mi, vi in zip(params, (g_W1, g_b1, g_w2, g_b2), m, v):
                    mi *= b1_
                    mi += (1 - b1_) * g
                    vi *= b2_
                    vi += (1 - b2_) * g * g
                    mhat = mi / (1 - b1_**step)
                    vhat = vi / (1 - b2_**step)
                    p -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(f32)
            self.b2 = params[3][0]
            if callback is not None:
                callback(self, _ + 1)
        return self

    def predict_proba(self, tx, block=50000):
        """Predicted probabilities P(y = 1 | x) for every row of tx"""
        out = np.empty(len(tx), dtype=np.float64)
        for i in range(0, len(tx), block):
            _, z = self._forward(tx[i : i + block])
            out[i : i + block] = sigmoid(z)
        return out
