"""Balanced batch generator for Stage 2 domain-adversarial training.

Ensures equal source and target samples per batch and assigns sample weights so that only
source samples contribute to regression loss while all contribute to domain loss.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class BalancedBatchGenerator(tf.keras.utils.Sequence):
    """Keras Sequence for balanced source/target batches used in Stage-2 domain training.

    Each batch contains half source and half target samples. The source labels are used for
    regression supervision, while target labels are dummy placeholders. Domain labels are 0
    for source and 1 for target. Regression loss is masked to zero for target samples via
    sample weights.

    Args:
        Xs: Source feature array of shape (Ns, F).
        ys: Source target array of shape (Ns, 1) or (Ns,).
        ds: Source domain labels (zeros) of shape (Ns, 1).
        Xt: Target feature array of shape (Nt, F).
        dt: Target domain labels (ones) of shape (Nt, 1).
        batch_size: Total batch size (half source, half target).
    """

    def __init__(self, Xs, ys, ds, Xt, dt, batch_size: int) -> None:
        self.Xs, self.ys, self.ds = Xs, ys, ds
        self.Xt, self.dt = Xt, dt
        self.batch_size = int(batch_size)
        self.half = self.batch_size // 2
        self.is_idx = np.arange(len(Xs))
        self.it_idx = np.arange(len(Xt))
        self.on_epoch_end()

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(max(len(self.Xs), len(self.Xt)) / self.half))

    def __getitem__(self, _: int):
        """Generate one balanced batch of data."""
        si = np.random.choice(self.is_idx, self.half, replace=True)
        ti = np.random.choice(self.it_idx, self.half, replace=True)

        Xb = np.vstack([self.Xs[si], self.Xt[ti]])
        y_src = self.ys[si]
        y_tgt = np.zeros((self.half, 1), dtype=y_src.dtype)
        yb = np.vstack([y_src, y_tgt])

        d_src = self.ds[si]
        d_tgt = self.dt[ti]
        db = np.vstack([d_src, d_tgt])

        # Regression weights: only source contributes
        w_reg = np.concatenate([np.ones(self.half), np.zeros(self.half)])
        # Domain weights: all samples contribute
        w_dom = np.ones(self.batch_size)

        perm = np.random.permutation(self.batch_size)
        Xb, yb, db = Xb[perm], yb[perm], db[perm]
        w_reg, w_dom = w_reg[perm], w_dom[perm]

        return (
            Xb,
            {"regressor_head": yb, "domain_output": db},
            {"regressor_head": w_reg, "domain_output": w_dom},
        )

    def on_epoch_end(self) -> None:
        """Shuffle indices for both source and target domains."""
        np.random.shuffle(self.is_idx)
        np.random.shuffle(self.it_idx)