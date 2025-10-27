"""Custom Keras callbacks for staged_bdann.

Includes: Early stopping factory, GRL lambda scheduler, domain AUC recorder, DANN alignment
early stop, and KL weight scheduler for Stage 3.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf


def _make_early_stopping(es_cfg: dict) -> Optional[tf.keras.callbacks.Callback]:
    """Create a standard EarlyStopping callback from a config dictionary.

    Args:
        es_cfg: Early stopping configuration. Recognized keys: enabled, monitor, patience,
            min_delta, mode, restore_best_weights.

    Returns:
        A configured EarlyStopping callback, or None if disabled.
    """
    if not es_cfg or not es_cfg.get("enabled", False):
        return None
    monitor = es_cfg.get("monitor", "val_loss")
    patience = int(es_cfg.get("patience", 50))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    mode = es_cfg.get("mode", "min")
    restore = bool(es_cfg.get("restore_best_weights", True))
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore,
    )


class LambdaScheduler(tf.keras.callbacks.Callback):
    """Schedule GRL lambda with optional warmup and logistic ramp.

    For epochs < warmup_epochs, lambda = 0. For remaining epochs, let p be the normalized epoch
    fraction in [0, 1]; compute base = 2/(1+exp(-k*p)) - 1, then lambda = lambda_max *
    max(lambda_min_frac, base).

    Args:
        lambda_var: A scalar tf.Variable assigned each epoch.
        lambda_max: Maximum value of lambda.
        total_epochs: Total number of training epochs for Stage 2.
        lambda_min_frac: Floor as a fraction of lambda_max.
        ramp_k: Logistic steepness parameter.
        warmup_epochs: Number of initial epochs with zero lambda.
    """

    def __init__(
        self,
        lambda_var: tf.Variable,
        lambda_max: float,
        total_epochs: int,
        lambda_min_frac: float = 0.0,
        ramp_k: float = 10.0,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.lambda_var = lambda_var
        self.lambda_max = float(lambda_max)
        self.total_epochs = int(total_epochs)
        self.lambda_min_frac = float(lambda_min_frac)
        self.ramp_k = float(ramp_k)
        self.warmup_epochs = int(warmup_epochs)

    def on_epoch_begin(self, epoch: int, logs=None) -> None:  # noqa: D401
        if epoch < self.warmup_epochs:
            self.lambda_var.assign(0.0)
            return
        denom = max(1, self.total_epochs - self.warmup_epochs)
        p = (epoch - self.warmup_epochs) / denom
        base = 2.0 / (1.0 + np.exp(-self.ramp_k * p)) - 1.0
        val = self.lambda_max * max(self.lambda_min_frac, float(base))
        self.lambda_var.assign(val)


class Stage2AUCCallback(tf.keras.callbacks.Callback):
    """Record per-epoch domain AUC on train and validation splits.

    Args:
        model_ref: The DANN model that outputs (regression, domain) tensors.
        Xtr: Stacked features for domain AUC on train, concatenated source then target.
        ytr: Domain labels for Xtr.
        Xv: Stacked features for domain AUC on validation, concatenated source then target.
        yv: Domain labels for Xv.
        out_path_curve: File to append a text record each epoch.
    """

    def __init__(
        self,
        model_ref: tf.keras.Model,
        Xtr,
        ytr,
        Xv,
        yv,
        out_path_curve,
    ) -> None:
        super().__init__()
        self.model_ref = model_ref
        self.Xtr, self.ytr = Xtr, ytr
        self.Xv, self.yv = Xv, yv
        self.out_path_curve = out_path_curve
        self.train_curve = []
        self.val_curve = []

    def on_epoch_end(self, epoch: int, logs=None) -> None:  # noqa: D401
        # Predict domain probabilities
        _, p_tr = self.model_ref(self.Xtr, training=False)
        _, p_v = self.model_ref(self.Xv, training=False)
        # Compute AUCs
        auc_tr = tf.keras.metrics.AUC()
        auc_v = tf.keras.metrics.AUC()
        auc_tr.update_state(self.ytr, p_tr)
        auc_v.update_state(self.yv, p_v)
        v_tr = float(auc_tr.result().numpy())
        v_v = float(auc_v.result().numpy())
        self.train_curve.append(v_tr)
        self.val_curve.append(v_v)
        # Append to file each epoch for easy tailing
        with open(self.out_path_curve, "a", encoding="utf-8") as f:
            f.write(f"epoch={epoch+1}, train_auc={v_tr:.6f}, val_auc={v_v:.6f}\n")


class DANNAlignmentEarlyStop(tf.keras.callbacks.Callback):
    """Early stop when the domain head appears sufficiently random on validation.

    Score minimized: |AUC_val - 0.5| + gamma * max(0, ln2 - BCE_val).
    Respects warmup and min_epochs; optionally restores best weights.

    Args:
        model_ref: The DANN model to snapshot and restore.
        Xv: Validation features for domain AUC and BCE.
        yv: Validation domain labels.
        warmup_epochs: Epochs to ignore the criterion.
        patience: Number of bad epochs before stopping.
        min_delta: Required improvement in the score to reset patience.
        min_epochs: Minimum epochs to run after warmup before considering stop.
        gamma: Weight on the BCE term relative to the AUC term.
        restore_best: If True, restore the model weights with the best score.
        log_path: Optional path to write a CSV-like log of scores.
    """

    def __init__(
        self,
        model_ref: tf.keras.Model,
        Xv,
        yv,
        warmup_epochs: int = 0,
        patience: int = 5,
        min_delta: float = 1e-3,
        min_epochs: int = 0,
        gamma: float = 0.5,
        restore_best: bool = True,
        log_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_ref = model_ref
        self.Xv = Xv
        self.yv = yv
        self.warmup = int(warmup_epochs)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_epochs = int(min_epochs)
        self.gamma = float(gamma)
        self.restore_best = bool(restore_best)
        self.log_path = log_path
        self.best = np.inf
        self.best_weights = None
        self.bad_epochs = 0
        self._bceln2 = float(np.log(2.0))
        self.vals = []
        self.scores = []
        if self.log_path:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("# epoch, auc_val, bce_val, score, best\n")

    def on_epoch_end(self, epoch: int, logs=None) -> None:  # noqa: D401
        # During warmup, do not influence training but still append NaN so plots are non-empty.
        if epoch < self.warmup:
            self.scores.append(float("nan"))
            if self.log_path:
                best_str = f"{self.best:.6f}" if np.isfinite(self.best) else "inf"
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(f"{epoch+1}, NaN, NaN, NaN, {best_str}\n")
            return

        # Evaluate domain head on validation
        _, p_v = self.model_ref(self.Xv, training=False)
        # AUC
        auc = tf.keras.metrics.AUC()
        auc.update_state(self.yv, p_v)
        auc_val = float(auc.result().numpy())
        # BCE
        bce_vec = tf.keras.losses.binary_crossentropy(self.yv, p_v)
        bce_val = float(tf.reduce_mean(bce_vec).numpy())
        # Alignment score
        score = abs(auc_val - 0.5) + self.gamma * max(0.0, self._bceln2 - bce_val)
        self.scores.append(score)

        # Logging
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1}, {auc_val:.6f}, {bce_val:.6f}, {score:.6f}, {self.best:.6f}\n")

        # Enforce a minimum number of epochs after warmup
        if (epoch + 1) < (self.warmup + self.min_epochs):
            if score + self.min_delta < self.best:
                self.best = score
                if self.restore_best:
                    self.best_weights = self.model_ref.get_weights()
            return

        # Usual early-stop logic
        improved = (self.best - score) > self.min_delta
        if improved:
            self.best = score
            self.bad_epochs = 0
            if self.restore_best:
                self.best_weights = self.model_ref.get_weights()
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                if self.restore_best and self.best_weights is not None:
                    self.model_ref.set_weights(self.best_weights)
                self.model.stop_training = True


class KLWeightScheduler(tf.keras.callbacks.Callback):
    """Anneal KL weight each epoch and scale per sample.

    The assigned value is ``kl_max * (2/(1+exp(-10*p)) - 1) / num_samples`` where
    ``p = epoch / total_epochs``.

    Args:
        kl_weight_var: Scalar tf.Variable to write into the KL closure used by Bayesian layers.
        kl_max: Maximum KL weight applied at the end of training.
        total_epochs: Total number of training epochs for Stage 3.
        num_samples: Number of training samples for per-sample scaling.
    """

    def __init__(
        self, kl_weight_var: tf.Variable, kl_max: float, total_epochs: int, num_samples: int
    ) -> None:
        super().__init__()
        self.kl_weight_var = kl_weight_var
        self.kl_max = float(kl_max)
        self.total_epochs = int(total_epochs)
        self.num_samples = max(1, int(num_samples))

    def on_epoch_begin(self, epoch: int, logs=None) -> None:  # noqa: D401
        p = epoch / max(1, self.total_epochs)
        annealed = self.kl_max * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)
        self.kl_weight_var.assign(annealed / float(self.num_samples))