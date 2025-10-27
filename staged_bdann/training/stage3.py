"""Stage 3 Bayesian fine-tuning on target data.

Builds a hybrid extractor with selective Bayesian layers and a deterministic or Bayesian head based on
configuration. Copies weights from Stage 2, enforces UQ placement rules, schedules KL weight, and trains
on the target split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow_probability import layers as tfpl

from ..models import build_stage3_hybrid_from_config
from ..utils import save_model_summary
from ..plotting import _save_loss_curve
from .callbacks import KLWeightScheduler, _make_early_stopping
from .common import normal_sp, stage3_safe_nll


def _resolve_unfreeze_start(total_layers: int, idx: int) -> int:
    """Convert possibly negative layer index to an absolute start index for unfreezing.

    Args:
        total_layers: Total number of layers in the extractor model.
        idx: Index from config, negative values count from the end (e.g., -7 means last 7 layers).

    Returns:
        Nonnegative index at which layers become trainable.
    """
    if idx < 0:
        start = total_layers + int(idx)
    else:
        start = int(idx)
    return max(0, min(total_layers, start))


def _map_dense_to_flipout(det_w: List[np.ndarray], flip_layer: tfpl.DenseFlipout) -> None:
    """Map deterministic Dense weights to a Flipout layer's posterior means when possible."""
    try:
        k_det, b_det = det_w
        # Assign to posterior loc parameters if available; fallback to trainable variables order.
        if hasattr(flip_layer, "kernel_posterior") and hasattr(flip_layer.kernel_posterior, "loc"):
            flip_layer.kernel_posterior.loc.assign(k_det)
        else:
            flip_layer.kernel_posterior.trainable_variables[0].assign(k_det)
        if hasattr(flip_layer, "bias_posterior") and hasattr(flip_layer.bias_posterior, "loc"):
            flip_layer.bias_posterior.loc.assign(b_det)
        else:
            flip_layer.bias_posterior.trainable_variables[0].assign(b_det)
    except Exception:
        # Best-effort mapping only; proceed without failing.
        pass


def stage3_finetune_hybrid(
    run_dir: Path,
    cfg: Dict[str, Any],
    input_shape: Iterable[int],
    feat_s2: Model,
    reg_head_s1: Model,
    Xt_tr: np.ndarray,
    yt_tr: np.ndarray,
    Xt_v: np.ndarray,
    yt_v: np.ndarray,
) -> Tuple[Model, bool, bool, List[str], Any]:
    """Build Stage 3 per configuration, map weights, enforce UQ rules, and train on target.

    Args:
        run_dir: Run directory for artifacts.
        cfg: Resolved configuration dictionary.
        input_shape: Model input feature shape, typically (F,).
        feat_s2: Stage 2 feature extractor (aligned).
        reg_head_s1: Stage 1 regression head (deterministic), used for initialization.
        Xt_tr: Target training features.
        yt_tr: Target training targets.
        Xt_v: Target validation features.
        yt_v: Target validation targets.

    Returns:
        Tuple of (model, is_head_bayesian, uq_allowed, warnings_list, history).
    """
    s3_arch = cfg["stage3"]["architecture"]
    policy = cfg["stage3"]["bayesian_policy"]

    # Fully deterministic override
    if bool(cfg["stage3"].get("fully_deterministic", False)):
        s3_arch = dict(s3_arch)
        s3_arch["reg_head"] = dict(s3_arch.get("reg_head", {}))
        s3_arch["reg_head"]["out_units"] = 1
        policy = dict(policy)
        policy["by_index"] = []

    # Resolve Bayesian indices with negative indexing support
    L = len(s3_arch["feature_extractor"]["layers"])
    by_index_raw = list(policy.get("by_index", []))
    by_index: List[int] = []
    for idx in by_index_raw:
        idx_adj = L + idx if idx < 0 else idx
        if 0 <= idx_adj < L:
            by_index.append(int(idx_adj))
    by_index = sorted(set(by_index))

    # Enforce UQ placement rules
    uq_valid = True
    warnings_list: List[str] = []
    if len(by_index) < 2:
        warnings_list.append(
            (
                "Not enough Bayesian layers to provide reliable uncertainty estimates. "
                f"Requested {len(by_index)}, require at least 2 for stability."
            )
        )
        uq_valid = False

    if policy.get("must_be_downstream", True) and len(by_index) > 0:
        K = len(by_index)
        expected = list(range(L - K, L))
        if by_index != expected:
            warnings_list.append(
                "Bayesian layers must be the last K layers in the extractor per policy. Uncertainty disabled."
            )
            uq_valid = False

    # Build extractor and head according to policy
    kl_weight_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    extractor_s3, reg_head_s3, is_head_bayesian = build_stage3_hybrid_from_config(
        input_shape, s3_arch, bayesian_indices=by_index, kl_weight=kl_weight_var
    )

    # Copy Stage-2 extractor weights into Stage-3
    for det_l, new_l in zip(feat_s2.layers, extractor_s3.layers):
        if isinstance(det_l, tf.keras.layers.Dense):
            det_w = det_l.get_weights()
            if isinstance(new_l, tfpl.DenseFlipout):
                _map_dense_to_flipout(det_w, new_l)
            elif isinstance(new_l, tf.keras.layers.Dense):
                try:
                    new_l.set_weights(det_w)
                except Exception:
                    pass

    # Map Stage-1 reg head to Stage-3 reg head
    last_s1 = reg_head_s1.layers[-1]
    last_s3 = reg_head_s3.layers[-1]
    if isinstance(last_s3, tfpl.DenseFlipout):
        if isinstance(last_s1, tf.keras.layers.Dense):
            det_w = last_s1.get_weights()  # (F,1), (1,)
            k_det, b_det = det_w[0], det_w[1]
            # Expand to (F,2) and (2,) for mean and pre-softplus scale
            k_exp = np.concatenate([k_det, np.zeros_like(k_det)], axis=1)
            b_exp = np.concatenate([b_det, np.zeros_like(b_det)], axis=0)
            try:
                if hasattr(last_s3.kernel_posterior, "loc"):
                    last_s3.kernel_posterior.loc.assign(k_exp)
                else:
                    last_s3.kernel_posterior.trainable_variables[0].assign(k_exp)
                if hasattr(last_s3.bias_posterior, "loc"):
                    last_s3.bias_posterior.loc.assign(b_exp)
                else:
                    last_s3.bias_posterior.trainable_variables[0].assign(b_exp)
            except Exception:
                pass
    else:
        if isinstance(last_s1, tf.keras.layers.Dense) and isinstance(last_s3, tf.keras.layers.Dense):
            try:
                last_s3.set_weights(last_s1.get_weights())
            except Exception:
                pass

    # Freeze policy
    unfreeze_idx_cfg = int(cfg["stage3"]["freeze"].get("unfreeze_from_layer_idx", -7))
    start_idx = _resolve_unfreeze_start(len(extractor_s3.layers), unfreeze_idx_cfg)
    for i, layer in enumerate(extractor_s3.layers):
        layer.trainable = i >= start_idx

    # Summaries
    save_model_summary(extractor_s3, run_dir / "arch" / "stage3_feature_extractor.txt")
    save_model_summary(reg_head_s3, run_dir / "arch" / "stage3_reg_head.txt")

    # Final model assembly
    inputs = extractor_s3.input
    feats = extractor_s3.output
    head_out = reg_head_s3(feats)

    steps_per_epoch = int(np.ceil(len(Xt_tr) / int(cfg["stage3"]["batch_size"])))
    decay_steps = int(max(1, int(cfg["stage3"]["optimizer"].get("decay_every_epochs", 1))) * steps_per_epoch)
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(cfg["stage3"]["optimizer"]["lr"]),
        decay_steps=decay_steps,
        decay_rate=float(cfg["stage3"]["optimizer"]["decay_rate"]),
        staircase=True,
    )

    callbacks = []
    if is_head_bayesian:
        y_dist = tfp.layers.DistributionLambda(normal_sp)(head_out)
        model = Model(inputs, y_dist, name="stage3_model_bayesian")
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_sched),
            loss=stage3_safe_nll,
        )
        kls = KLWeightScheduler(
            kl_weight_var,
            cfg["stage3"]["kl_weight_max"],
            total_epochs=int(cfg["stage3"]["epochs"]),
            num_samples=int(len(Xt_tr)),
        )
        callbacks.append(kls)
    else:
        model = Model(inputs, head_out, name="stage3_model_deterministic")
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_sched),
            loss="mse",
        )

    es_cb = _make_early_stopping(cfg["stage3"].get("early_stopping", {}))
    if es_cb is not None:
        callbacks.append(es_cb)

    # Warnings file for UQ policy
    if warnings_list:
        with open(run_dir / "logs" / "stage3_uq_warnings.txt", "w") as f:
            for w in warnings_list:
                f.write(w + "\n")

    history = model.fit(
        Xt_tr,
        yt_tr,
        validation_data=(Xt_v, yt_v),
        epochs=int(cfg["stage3"]["epochs"]),
        batch_size=int(cfg["stage3"]["batch_size"]),
        verbose=int(cfg["logging"]["verbose"]),
        callbacks=callbacks,
    )

    _save_loss_curve(
        history,
        run_dir / "plots" / "stage3_loss.png",
        "Stage-3 NLL" if is_head_bayesian else "Stage-3 MSE",
        yscale=None,
    )

    model.save_weights(run_dir / "weights" / "stage3_model.h5")

    return model, is_head_bayesian, bool(uq_valid and is_head_bayesian), warnings_list, history