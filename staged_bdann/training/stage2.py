"""Stage 2 Domain-Adversarial alignment (DANN).

This module trains the feature extractor adversarially against a domain classifier via a
Gradient Reversal Layer. The regression head is frozen, excluded from loss, and only used to
preserve the supervised signal during forward passes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

from ..models import GradientReversal, build_domain_head, build_feature_extractor_det
from .batching import BalancedBatchGenerator
from .callbacks import (
    LambdaScheduler,
    DANNAlignmentEarlyStop,
    Stage2AUCCallback,
    _make_early_stopping,
)
from ..plotting import _save_loss_curve


def _count_params(vars_like: Any) -> int:
    """Count scalar parameters in a list of variables."""
    n = 0
    for v in vars_like:
        try:
            n += int(np.prod(v.shape))
        except Exception:
            pass
    return n


def stage2_dann(
    run_dir: Path,
    cfg: Dict[str, Any],
    feature_extractor: Model,
    reg_head: Model,
    Xs_tr,
    ys_tr,
    Xs_v,
    Xt_tr,
    Xt_v,
    ds_tr,
    dt_tr,
) -> Tuple[Model, Any]:
    """Run Stage 2 DANN to align source and target feature distributions.

    The regression head is frozen and not optimized. Only the domain classifier and feature
    extractor receive gradients, with the GRL reversing gradients on the extractor by a scheduled
    factor lambda.

    Args:
        run_dir: Directory to write plots and logs.
        cfg: Resolved configuration dictionary.
        feature_extractor: Pretrained Stage 1 feature extractor model.
        reg_head: Pretrained Stage 1 regression head model, frozen in this stage.
        Xs_tr: Source training features.
        ys_tr: Source training targets (not used in the loss, but present for generator API).
        Xs_v: Source validation features.
        Xt_tr: Target training features.
        Xt_v: Target validation features.
        ds_tr: Source training domain labels, zeros with shape (N, 1).
        dt_tr: Target training domain labels, ones with shape (M, 1).

    Returns:
        A tuple of (aligned_feature_extractor_model, keras_history).
    """
    # Freeze regressor head
    reg_head.trainable = False
    for _layer in reg_head.layers:
        _layer.trainable = False

    # Ensure extractor is trainable
    feature_extractor.trainable = True
    for _layer in feature_extractor.layers:
        _layer.trainable = True

    # Diagnostics: ensure zero trainables in reg head
    checks_path = run_dir / "logs" / "stage2_checks.txt"
    n_trainable = len(reg_head.trainable_variables)
    n_params = _count_params(reg_head.trainable_variables)
    with open(checks_path, "w") as fchk:
        if n_trainable != 0 or n_params != 0:
            fchk.write(
                f"Stage-2 regressor head trainable vars: {n_trainable} params: {n_params}  [FAIL]\n"
            )
        else:
            fchk.write("Stage-2 regressor head trainable vars: 0 params: 0  [OK]\n")

    if n_trainable != 0 or n_params != 0:
        raise RuntimeError("Stage-2 regression head not frozen. Aborting.")

    # Per-layer trainability dump
    with open(checks_path, "a") as fchk:
        fe_tv = len(feature_extractor.trainable_variables)
        fe_tp = _count_params(feature_extractor.trainable_variables)
        fchk.write(
            f"Feature extractor trainable vars: {fe_tv} params: {fe_tp}  {'[OK]' if fe_tv > 0 else '[FAIL]'}\n"
        )
        fchk.write("  Extractor layer trainability:\n")
        for _i, _layer in enumerate(feature_extractor.layers):
            lv = len(_layer.trainable_variables)
            lp = _count_params(_layer.trainable_variables)
            fchk.write(
                f"    Layer {_i:02d} ({_layer.name}) trainable={_layer.trainable} vars={lv} params={lp}\n"
            )
        fchk.write("  Regressor head layer trainability:\n")
        for _i, _layer in enumerate(reg_head.layers):
            lv = len(_layer.trainable_variables)
            lp = _count_params(_layer.trainable_variables)
            fchk.write(
                f"    Layer {_i:02d} ({_layer.name}) trainable={_layer.trainable} vars={lv} params={lp}\n"
            )

    # Snapshot extractor weights before training for delta diagnostics
    w0 = (
        np.concatenate([w.flatten() for w in feature_extractor.get_weights()])
        if feature_extractor.get_weights()
        else np.array([0.0])
    )

    # Build DANN graph
    lambda_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    grl = GradientReversal(lambda_var)(feature_extractor.output)
    dom_out = build_domain_head(
        grl,
        widths=cfg["stage2"]["domain_head"]["widths"],
        dropout=cfg["stage2"]["domain_head"]["dropout"],
        l2=cfg["stage2"]["domain_head"]["l2"],
    )
    reg_out = reg_head(feature_extractor.output)

    dann = Model(inputs=feature_extractor.input, outputs=[reg_out, dom_out], name="stage2_dann")

    # Balanced generator
    gen = BalancedBatchGenerator(
        Xs_tr, ys_tr, ds_tr, Xt_tr, dt_tr, batch_size=cfg["stage2"]["batch_size"]
    )

    # Learning rate schedule
    steps_per_epoch = int(np.ceil(len(Xs_tr) / cfg["stage2"]["batch_size"]))
    decay_steps = int(max(1, cfg["stage2"]["optimizer"]["decay_every_epochs"]) * steps_per_epoch)
    lr_sched = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(cfg["stage2"]["optimizer"]["lr"]),
        decay_steps=decay_steps,
        decay_rate=float(cfg["stage2"]["optimizer"]["decay_rate"]),
        staircase=True,
    )

    # AUC monitoring over full splits (threshold-free separability)
    X_dom_tr = np.vstack([Xs_tr, Xt_tr]).astype(np.float32)
    y_dom_tr = np.vstack(
        [np.zeros((len(Xs_tr), 1), dtype=np.float32), np.ones((len(Xt_tr), 1), dtype=np.float32)]
    )
    X_dom_v = np.vstack([Xs_v, Xt_v]).astype(np.float32)
    y_dom_v = np.vstack(
        [np.zeros((len(Xs_v), 1), dtype=np.float32), np.ones((len(Xt_v), 1), dtype=np.float32)]
    )

    dann.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_sched),
        loss={"regressor_head": "mse", "domain_output": "binary_crossentropy"},
        loss_weights={"regressor_head": 0.0, "domain_output": 1.0},
        metrics={"domain_output": [tf.keras.metrics.AUC(name="auc")]},
    )

    # Callbacks
    cb_list = [
        LambdaScheduler(
            lambda_var,
            cfg["stage2"]["lambda_max"],
            cfg["stage2"]["epochs"],
            lambda_min_frac=float(cfg["stage2"].get("lambda_min_frac", 0.0)),
            ramp_k=float(cfg["stage2"].get("ramp_k", 10.0)),
            warmup_epochs=int(cfg["stage2"].get("warmup_epochs", 0)),
        )
    ]

    auc_curve_path = run_dir / "logs" / "stage2_auc_curve.txt"
    with open(auc_curve_path, "w") as _f:
        _f.write("# per-epoch AUC for domain classifier (higher means more separability)\n")
    auc_cb = Stage2AUCCallback(dann, X_dom_tr, y_dom_tr, X_dom_v, y_dom_v, auc_curve_path)
    cb_list.append(auc_cb)

    es_cfg = cfg["stage2"].get("early_stopping", {})
    if es_cfg.get("enabled", False) and es_cfg.get("monitor", "").lower() == "dann_alignment":
        es_log = run_dir / "logs" / "stage2_alignment_es.txt"
        es_cb = DANNAlignmentEarlyStop(
            dann,
            X_dom_v,
            y_dom_v,
            warmup_epochs=int(cfg["stage2"].get("warmup_epochs", 0)),
            patience=int(es_cfg.get("patience", 5)),
            min_delta=float(es_cfg.get("min_delta", 1e-3)),
            min_epochs=int(es_cfg.get("min_epochs", 0)),
            gamma=float(es_cfg.get("gamma", 0.5)),
            restore_best=bool(es_cfg.get("restore_best_weights", True)),
            log_path=str(es_log),
        )
        cb_list.append(es_cb)
    else:
        es_cb_std = _make_early_stopping(es_cfg)
        if es_cb_std is not None:
            cb_list.append(es_cb_std)

    history = dann.fit(
        gen,
        epochs=int(cfg["stage2"]["epochs"]),
        callbacks=cb_list,
        verbose=int(cfg["logging"]["verbose"]),
    )

    _save_loss_curve(
        history,
        run_dir / "plots" / "stage2_domain_loss.png",
        "Stage-2 Domain BCE",
        ideal_y=0.693,
        warmup_epochs=int(cfg["stage2"].get("warmup_epochs", 0)),
        yscale=None,
        keys=["domain_output_loss"],
    )

    # Plot AUC curves
    try:
        plt.figure()
        if len(auc_cb.train_curve) > 0:
            plt.plot(auc_cb.train_curve, label="train_auc")
        if len(auc_cb.val_curve) > 0:
            plt.plot(auc_cb.val_curve, label="val_auc")
        plt.axhline(y=0.5, label="ideal", color="black", linestyle="dashed", alpha=0.8, zorder=0)
        plt.axvline(x=int(cfg["stage2"].get("warmup_epochs", 0)), label="warmup_epochs", color="black", linestyle="dashed", alpha=0.8, zorder=0)  
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.title("Stage-2 Domain AUC")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "stage2_auc.png", dpi=300)
        plt.close()
    except Exception:
        pass

    # Compute extractor movement
    w1 = (
        np.concatenate([w.flatten() for w in feature_extractor.get_weights()])
        if feature_extractor.get_weights()
        else np.array([0.0])
    )
    l2_delta = float(np.linalg.norm(w1 - w0))
    base_norm = float(np.linalg.norm(w0))
    rel_delta = l2_delta / (base_norm + 1e-12)
    with open(checks_path, "a") as f:
        f.write(f"Extractor pre→post L2 delta: {l2_delta:.6e}\n")
        f.write(f"Extractor pre→post relative delta: {rel_delta:.6e}\n")

    # GRL gradient sanity on a small batch
    Xb, ydict, _ = gen[0]
    with tf.GradientTape() as tape:
        tape.watch(feature_extractor.trainable_variables)
        _, dlogit = dann(Xb, training=True)
        dom_loss = tf.keras.losses.binary_crossentropy(ydict["domain_output"], dlogit)
        dom_loss = tf.reduce_mean(dom_loss)
    grads = tape.gradient(dom_loss, feature_extractor.trainable_variables)
    grad_norm = 0.0
    for g in grads:
        if g is not None:
            grad_norm += float(tf.linalg.global_norm([g]).numpy())
    with open(checks_path, "a") as fchk:
        fchk.write(f"GRL gradient norm (non-zero expected if dom loss non-zero): {grad_norm:.6e}\n")

    # Save aligned extractor weights
    feature_extractor.save_weights(run_dir / "weights" / "stage2_extractor.h5")

    # L2 diff vs a fresh extractor with same architecture
    fresh = build_feature_extractor_det(
        feature_extractor.input_shape[1:],
        cfg["stage1"]["architecture"]["feature_extractor"]["layers"],
        cfg["stage1"]["architecture"]["feature_extractor"]["activations"],
    )
    w_aligned = (
        np.concatenate([w.flatten() for w in feature_extractor.get_weights()])
        if feature_extractor.get_weights()
        else np.array([0.0])
    )
    w_fresh = (
        np.concatenate([w.flatten() for w in fresh.get_weights()]) if fresh.get_weights() else np.array([0.0])
    )
    l2diff = float(np.linalg.norm(w_aligned - w_fresh))
    with open(checks_path, "a") as fchk:
        fe_weights = (
            np.concatenate([w.flatten() for w in feature_extractor.get_weights()])
            if feature_extractor.get_weights()
            else np.array([0.0])
        )
        fchk.write(f"Extractor weight L2 norm (post Stage-2): {np.linalg.norm(fe_weights):.6e}\n")
        if l2diff < 1e-8:
            fchk.write("WARNING: Stage-2 had near-zero impact on extractor weights.\n")

    return feature_extractor, history