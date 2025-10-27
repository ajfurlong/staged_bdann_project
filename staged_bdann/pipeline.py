"""Single-run pipeline orchestration for staged_bdann.

This module performs a full Stage 1 - Stage 2 - Stage 3 run for a single seed using precomputed
fixed splits from `data.load_data`. It handles diagnostics, predictions, metrics, plots, and
artifact persistence.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model

from .utils import save_metadata_h5, save_predictions_csv, save_versions, set_global_determinism
from .training import stage1_train_or_load, stage2_dann, stage3_finetune_hybrid
from .plotting import (
    _save_hist_kde,
    _save_loss_curve,
    _save_parity_plot,
    _save_rstd_components_plot,
    _save_std_components_plot,
    _save_uq_calibration_plot,
    compute_metrics,
)


def _inverse_transform_safe(arr: np.ndarray, scaler: Any) -> np.ndarray:
    """Inverse-transform with a StandardScaler if provided, otherwise return input.

    Args:
        arr: Array of shape (N, 1) or (N,) in standardized units or original units.
        scaler: A fitted sklearn-like scaler with ``inverse_transform`` or ``None``.

    Returns:
        Array in original units if scaler is provided, else the input array converted to 1-D.
    """
    if scaler is None:
        return np.asarray(arr).reshape(-1)
    return scaler.inverse_transform(np.asarray(arr)).reshape(-1)


def _y_scale_factor(scaler: Any) -> float:
    """Return the target scale factor for converting std-dev from standardized to original units.

    If scaler is None, the factor is 1.0.
    """
    if scaler is None:
        return 1.0
    return float(np.asarray(scaler.scale_).reshape(-1)[0])


def run_member_once(cfg: Dict[str, Any], member_run_dir: Path, seed: int, fixed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one full staged B-DANN run for a given seed.

    The function expects ``fixed_data`` to contain deterministic splits and scalers returned by
    :func:`staged_bdann.data.load_data`. No splitting occurs here; only training and evaluation.

    Args:
        cfg: Resolved configuration dictionary.
        member_run_dir: Output directory for this member's artifacts.
        seed: Random seed for this member's training.
        fixed_data: Dictionary from :func:`load_data` with standardized splits and scalers.

    Returns:
        A dictionary of final Stage 3 test metrics.
    """
    member_run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("plots", "arch", "splits", "weights", "logs"):
        (member_run_dir / sub).mkdir(exist_ok=True)

    with open(member_run_dir / "logs" / "config_resolved.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    save_versions(member_run_dir)

    # Unpack fixed split
    d = fixed_data
    Xs_tr, Xs_v, Xs_te = d["Xs_tr"], d["Xs_v"], d["Xs_te"]
    ys_tr, ys_v, ys_te = d["ys_tr"], d["ys_v"], d["ys_te"]
    ds_tr = d["ds_tr"]

    Xt_tr, Xt_v, Xt_te = d["Xt_tr"], d["Xt_v"], d["Xt_te"]
    yt_tr, yt_v, yt_te = d["yt_tr"], d["yt_v"], d["yt_te"]
    dt_tr = d["dt_tr"]

    X_scaler = d["X_scaler"]
    y_scaler_src = d["y_scaler_src"]
    y_scaler_tgt = d["y_scaler_tgt"]

    sizes_source = d["sizes_source"]
    sizes_target = d["sizes_target"]

    # Determinism and memory growth
    set_global_determinism(int(seed), intra_threads=2, inter_threads=2)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass

    input_shape = (Xs_tr.shape[1],)

    # Stage 1
    t0 = time.time()
    fe_s1, rh_s1, hist_s1 = stage1_train_or_load(
        member_run_dir,
        cfg,
        input_shape,
        Xs_tr,
        ys_tr,
        Xs_v,
        ys_v,
        Xs_te,
        ys_te,
        sizes_source,
        sizes_target,
        y_scaler_src,
    )
    stage1_time = time.time() - t0

    # Quick S1 diagnostic on target test
    diag_s1_tgt = Model(inputs=fe_s1.input, outputs=rh_s1(fe_s1.output))
    pred_s1_t_scaled = diag_s1_tgt.predict(Xt_te, verbose=0)
    pred_s1_t = _inverse_transform_safe(pred_s1_t_scaled, y_scaler_tgt)
    y_true_t = _inverse_transform_safe(yt_te, y_scaler_tgt)

    m_s1_t = compute_metrics(
        y_true_t,
        pred_s1_t,
        None,
        sizes_source,
        sizes_target,
        stage1_time=stage1_time,
        stage2_time=0.0,
        stage3_time=0.0,
        inference_time=0.0,
        ape_epsilon=float(cfg["metrics"]["ape_epsilon"]),
    )
    pd.DataFrame(m_s1_t, index=["Values"]).T.to_csv(member_run_dir / "logs" / "stage1_on_target_metrics.csv")
    if cfg["logging"]["save_plots"]:
        _save_parity_plot(
            y_true_t,
            pred_s1_t,
            member_run_dir / "plots" / "stage1_on_target_parity.png",
            "Stage-1 on Target Test",
        )

    # Stage 2
    if not cfg["stage2"].get("enabled", True):
        fe_s2, hist_s2, stage2_time = fe_s1, None, 0.0
    else:
        t2 = time.time()
        fe_s2, hist_s2 = stage2_dann(
            member_run_dir,
            cfg,
            fe_s1,
            rh_s1,
            Xs_tr,
            ys_tr,
            Xs_v,
            Xt_tr,
            Xt_v,
            ds_tr,
            dt_tr,
        )
        stage2_time = time.time() - t2

    # Stage 3
    t3 = time.time()
    model_s3, is_head_bayes, uq_allowed, _uq_warnings, hist_s3 = stage3_finetune_hybrid(
        member_run_dir,
        cfg,
        input_shape,
        fe_s2,
        rh_s1,
        Xt_tr,
        yt_tr,
        Xt_v,
        yt_v,
    )
    stage3_time = time.time() - t3

    # Prediction on target test
    mc_S = int(cfg["metrics"]["mc_samples_test"])
    Xt_te_np = np.asarray(Xt_te, dtype=np.float32)

    y_pred_mean: np.ndarray
    y_pred_sigma_aleatoric: np.ndarray | None = None
    y_pred_sigma_epistemic: np.ndarray | None = None
    y_pred_sigma_total: np.ndarray | None = None

    if is_head_bayes and uq_allowed:
        mu_samples, ale_var_samples = [], []
        y_scale = _y_scale_factor(y_scaler_tgt)
        for _ in range(mc_S):
            dist = model_s3(Xt_te_np, training=True)
            mu_s_scaled = dist.mean().numpy()
            sig_s_scaled = dist.stddev().numpy()
            mu = _inverse_transform_safe(mu_s_scaled, y_scaler_tgt)
            sig = np.asarray(sig_s_scaled).reshape(-1) * y_scale
            mu_samples.append(mu)
            ale_var_samples.append(sig ** 2)
        mu_samples = np.asarray(mu_samples)
        ale_var_samples = np.asarray(ale_var_samples)
        y_pred_mean = mu_samples.mean(axis=0)
        y_pred_sigma_epistemic = mu_samples.std(axis=0)
        aleatoric_var = ale_var_samples.mean(axis=0)
        y_pred_sigma_total = np.sqrt(np.square(y_pred_sigma_epistemic) + aleatoric_var)
        y_pred_sigma_aleatoric = np.sqrt(aleatoric_var)
    else:
        preds_scaled = model_s3(Xt_te_np, training=False)
        if hasattr(preds_scaled, "mean"):
            y_pred_mean = _inverse_transform_safe(preds_scaled.mean().numpy(), y_scaler_tgt)
        else:
            y_pred_mean = _inverse_transform_safe(preds_scaled.numpy(), y_scaler_tgt)

    y_true = _inverse_transform_safe(yt_te, y_scaler_tgt)

    # Metrics
    if is_head_bayes and uq_allowed and (y_pred_sigma_total is not None):
        metrics = compute_metrics(
            y_true,
            y_pred_mean,
            y_pred_sigma_total,
            sizes_source,
            sizes_target,
            stage1_time=stage1_time,
            stage2_time=stage2_time,
            stage3_time=stage3_time,
            inference_time=0.0,
            y_pred_sigma_aleatoric=y_pred_sigma_aleatoric,
            y_pred_sigma_epistemic=y_pred_sigma_epistemic,
            ape_epsilon=float(cfg["metrics"]["ape_epsilon"]),
        )
    else:
        metrics = compute_metrics(
            y_true,
            y_pred_mean,
            None,
            sizes_source,
            sizes_target,
            stage1_time=stage1_time,
            stage2_time=stage2_time,
            stage3_time=stage3_time,
            inference_time=0.0,
            ape_epsilon=float(cfg["metrics"]["ape_epsilon"]),
        )

    # Persist metrics, predictions, metadata
    dfm = pd.DataFrame(metrics, index=["Values"]).T
    dfm.to_csv(member_run_dir / "logs" / "stage3_final_metrics.csv")

    if cfg["logging"]["save_plots"]:
        _save_parity_plot(
            y_true,
            y_pred_mean,
            member_run_dir / "plots" / "stage3_test_parity.png",
            "Parity: Target Test",
        )
        err = y_pred_mean - y_true
        _save_hist_kde(err, member_run_dir / "plots" / "err_hist_kde.png", "Error", "Error")
        eps = float(cfg["metrics"]["ape_epsilon"])
        mask = np.abs(y_true) >= eps
        if np.any(mask):
            ape_vals = np.abs((y_pred_mean[mask] - y_true[mask]) / y_true[mask]) * 100.0
            _save_hist_kde(
                ape_vals,
                member_run_dir / "plots" / "ape_hist_kde.png",
                "Absolute Percentage Error",
                "APE (%)",
            )
        if is_head_bayes and uq_allowed and (y_pred_sigma_total is not None):
            _save_rstd_components_plot(
                y_pred_mean,
                y_pred_sigma_epistemic,
                y_pred_sigma_aleatoric,
                y_pred_sigma_total,
                member_run_dir / "plots" / "rStd_components.png",
            )
            _save_std_components_plot(
                y_pred_mean,
                y_pred_sigma_epistemic,
                y_pred_sigma_aleatoric,
                y_pred_sigma_total,
                member_run_dir / "plots" / "Std_components.png",
            )
            _save_uq_calibration_plot(
                y_true,
                y_pred_mean,
                y_pred_sigma_epistemic,
                y_pred_sigma_aleatoric,
                y_pred_sigma_total,
                member_run_dir / "plots" / "uq_calibration.png",
                path_txt=member_run_dir / "logs" / "uq_calibration.txt",
            )

    if cfg["outfiles"]["save_metadata_h5"]:
        save_metadata_h5(
            member_run_dir,
            X_scaler,
            y_scaler_src,
            y_scaler_tgt,
            arch_json=cfg,
        )

    save_predictions_csv(
        member_run_dir / "logs" / "test_predictions.csv",
        y_true,
        y_pred_mean,
        y_pred_sigma_aleatoric,
        y_pred_sigma_epistemic,
        y_pred_sigma_total,
    )

    return y_true, y_pred_mean, y_pred_sigma_aleatoric, y_pred_sigma_epistemic, y_pred_sigma_total, metrics