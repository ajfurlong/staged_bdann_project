"""Metrics, reporting, and plotting utilities for staged_bdann.

Plots rely on matplotlib only and optionally SciPy for KDE. 
If SciPy is missing, KDE overlays and calibration computations are
skipped without raising errors.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import gaussian_kde, norm
    from scipy.integrate import simps

    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ---------------------------
# Metrics and reporting
# ---------------------------

def compute_metrics(
    y_true,
    y_pred_mean,
    y_pred_sigma_total,
    sizes_source,
    sizes_target,
    stage1_time: float = 0.0,
    stage2_time: float = 0.0,
    stage3_time: float = 0.0,
    inference_time: float = 0.0,
    y_pred_sigma_aleatoric=None,
    y_pred_sigma_epistemic=None,
    ape_epsilon: float = 1e-8,
):
    """Compute regression and optional UQ metrics in original units.

    APE and rRMSE use masking that removes small |y_true| < ape_epsilon from percentage metrics.

    Args:
        y_true: Array-like of ground truth values, shape (N,).
        y_pred_mean: Array-like of predicted means, shape (N,).
        y_pred_sigma_total: Optional array-like of total predictive std dev per sample, shape (N,).
        sizes_source: Iterable of three ints, train, valid, test sizes for the source domain.
        sizes_target: Iterable of three ints, train, valid, test sizes for the target domain.
        stage1_time: Stage 1 runtime in seconds.
        stage2_time: Stage 2 runtime in seconds.
        stage3_time: Stage 3 runtime in seconds.
        inference_time: Prediction time in seconds.
        y_pred_sigma_aleatoric: Optional array-like aleatoric std dev per sample, shape (N,).
        y_pred_sigma_epistemic: Optional array-like epistemic std dev per sample, shape (N,).
        ape_epsilon: Masking threshold for percentage metrics.

    Returns:
        Dictionary of scalar metrics, with [UQ] keys included only if inputs are provided.
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred_mean, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Shapes of y_true {yt.shape} and y_pred_mean {yp.shape} must match.")

    # Optional arrays and shape checks
    has_tot = y_pred_sigma_total is not None
    has_ale = y_pred_sigma_aleatoric is not None
    has_epi = y_pred_sigma_epistemic is not None

    if has_tot:
        sigma_tot = np.asarray(y_pred_sigma_total, dtype=np.float64).reshape(-1)
        if sigma_tot.shape != yt.shape:
            raise ValueError(f"y_pred_sigma_total shape {sigma_tot.shape} must match y_true {yt.shape}.")
    if has_ale:
        sigma_ale = np.asarray(y_pred_sigma_aleatoric, dtype=np.float64).reshape(-1)
        if sigma_ale.shape != yt.shape:
            raise ValueError(f"y_pred_sigma_aleatoric shape {sigma_ale.shape} must match y_true {yt.shape}.")
    if has_epi:
        sigma_epi = np.asarray(y_pred_sigma_epistemic, dtype=np.float64).reshape(-1)
        if sigma_epi.shape != yt.shape:
            raise ValueError(f"y_pred_sigma_epistemic shape {sigma_epi.shape} must match y_true {yt.shape}.")

    # Absolute error stats
    ae = np.abs(yp - yt)
    mae = float(np.mean(ae))
    medae = float(np.median(ae))
    maxae = float(np.max(ae))
    minae = float(np.min(ae))
    stda = float(np.std(ae))

    # R^2
    ss_res = float(np.sum((yp - yt) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = np.nan if ss_tot <= np.finfo(float).eps else 1.0 - (ss_res / ss_tot)

    # APE-based metrics with masking
    mask = np.abs(yt) >= float(ape_epsilon)
    masked_count = int((~mask).sum())
    yt_m, yp_m = yt[mask], yp[mask]

    if yt_m.size > 0:
        ape = np.abs((yt_m - yp_m) / yt_m) * 100.0
        mape = float(np.mean(ape))
        maxape = float(np.max(ape))
        minape = float(np.min(ape))
        stdape = float(np.std(ape))
        ferr10 = 100.0 * float(np.sum(ape > 10.0)) / len(ape)
        ferr25 = 100.0 * float(np.sum(ape > 25.0)) / len(ape)
        rrmse = float(np.sqrt(np.mean(((yp_m - yt_m) / yt_m) ** 2)) * 100.0)
    else:
        mape = maxape = minape = stdape = ferr10 = ferr25 = rrmse = np.nan

    # Base metrics, no UQ keys here
    metrics = {
        "R2": r2,
        "[APE] MAPE": mape,
        "[APE] Max APE": maxape,
        "[APE] Min APE": minape,
        "[APE] STD APE": stdape,
        "[APE] rRMSE (%)": rrmse,
        "[APE] Ferr > 10% (%)": ferr10,
        "[APE] Ferr > 25% (%)": ferr25,
        "[AE] MAE": mae,
        "[AE] MedAE": medae,
        "[AE] Max AE": maxae,
        "[AE] Min AE": minae,
        "[AE] STD AE": stda,
    }

    # Conditionally add UQ totals
    if has_tot:
        rel_std = 100.0 * (sigma_tot / np.maximum(np.abs(yp), np.finfo(float).eps))
        metrics.update({
            "[UQ] Mean rStd (total)": float(np.mean(rel_std)),
            "[UQ] Max rStd (total)": float(np.max(rel_std)),
            "[UQ] Mean Std (total)": float(np.mean(sigma_tot)),
            "[UQ] Max Std (total)": float(np.max(sigma_tot)),
        })

    # Conditionally add aleatoric block
    if has_ale:
        rel_std_ale = 100.0 * (sigma_ale / np.maximum(np.abs(yp), np.finfo(float).eps))
        metrics.update({
            "[UQ] Mean rStd (aleatoric)": float(np.mean(rel_std_ale)),
            "[UQ] Max rStd (aleatoric)": float(np.max(rel_std_ale)),
            "[UQ] Mean Std (aleatoric)": float(np.mean(sigma_ale)),
            "[UQ] Max Std (aleatoric)": float(np.max(sigma_ale)),
        })

    # Conditionally add epistemic block
    if has_epi:
        rel_std_epi = 100.0 * (sigma_epi / np.maximum(np.abs(yp), np.finfo(float).eps))
        metrics.update({
            "[UQ] Mean rStd (epistemic)": float(np.mean(rel_std_epi)),
            "[UQ] Max rStd (epistemic)": float(np.max(rel_std_epi)),
            "[UQ] Mean Std (epistemic)": float(np.mean(sigma_epi)),
            "[UQ] Max Std (epistemic)": float(np.max(sigma_epi)),
        })

    metrics.update({
        "[INFO] Stage 1 Time (s)": float(stage1_time),
        "[INFO] Stage 2 Time (s)": float(stage2_time),
        "[INFO] Stage 3 Time (s)": float(stage3_time),
        "[INFO] Total Train Time (s)": float(stage1_time + stage2_time + stage3_time),
        "[INFO] Inference Time (s)": float(inference_time),
        "[INFO] APE masked count": masked_count,
        "[DATA] Source Train (#)": int(sizes_source[0]),
        "[DATA] Source Valid (#)": int(sizes_source[1]),
        "[DATA] Source Test (#)": int(sizes_source[2]),
        "[DATA] Target Train (#)": int(sizes_target[0]),
        "[DATA] Target Valid (#)": int(sizes_target[1]),
        "[DATA] Target Test (#)": int(sizes_target[2]),
    })

    return metrics


# ---------------------------
# Plotting helpers
# ---------------------------

def _save_parity_plot(y_true, y_pred, path, title):
    """Save a parity plot with ±10% and ±25% relative error bands."""
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    plt.figure()
    plt.plot(yt, yt, alpha=0.6, color="black")
    n = max(10, min(500, len(yt)))
    xs = np.linspace(np.min(yt), np.max(yt), n)
    plt.plot(xs, xs * 1.25, alpha=0.6, linestyle="--", color="black", label=r"$\pm$25% Rel. Error")
    plt.plot(xs, xs * 0.75, alpha=0.6, linestyle="--", color="black")
    plt.plot(xs, xs * 1.1, alpha=0.6, linestyle=":", color="black", label=r"$\pm$10% Rel. Error")
    plt.plot(xs, xs * 0.9, alpha=0.6, linestyle=":", color="black")
    plt.scatter(yt, yp, s=12, alpha=0.6)
    plt.xlabel("Target Actual")
    plt.ylabel("Target Predicted")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def _save_hist_kde(values, path_png, title, xlabel):
    """Save a histogram with optional KDE overlay if SciPy is available."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    plt.figure()
    plt.hist(v, bins=40, density=True, alpha=0.6, edgecolor="black")
    if _HAVE_SCIPY and v.size > 2:
        try:
            kde = gaussian_kde(v)
            xs = np.linspace(np.min(v), np.max(v), 256)
            plt.plot(xs, kde(xs))
        except Exception:
            pass
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()


def _save_uq_calibration_plot(
    y_true,
    y_pred_mean,
    y_pred_sigma_epi,
    y_pred_sigma_alea,
    y_pred_sigma_tot,
    path_png,
    path_txt: Optional[str] = None,
):
    """UQ calibration via CDF comparison of normalized residuals.

    Builds normalized residuals z = (y_true - y_pred_mean) / sigma, estimates empirical CDFs via KDE,
    and compares them with Normal(0, 1) CDF. Writes a miscalibration area to text if requested.
    No-ops if SciPy is unavailable or if any sigma input is None.
    """
    if (not _HAVE_SCIPY) or (y_pred_sigma_tot is None):
        return

    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred_mean, dtype=np.float64).reshape(-1)
    z = yp - yt

    sig_epi = np.maximum(np.asarray(y_pred_sigma_epi, dtype=np.float64).reshape(-1), np.finfo(float).eps)
    sig_alea = np.maximum(np.asarray(y_pred_sigma_alea, dtype=np.float64).reshape(-1), np.finfo(float).eps)
    sig_tot = np.maximum(np.asarray(y_pred_sigma_tot, dtype=np.float64).reshape(-1), np.finfo(float).eps)

    zn_epi = z / sig_epi
    zn_alea = z / sig_alea
    zn_tot = z / sig_tot

    try:
        z_vals = np.linspace(-4.0, 4.0, 1000)

        pdf_epi = gaussian_kde(zn_epi, bw_method="scott")(z_vals)
        emp_cdf_epi = np.cumsum(pdf_epi)
        emp_cdf_epi = emp_cdf_epi / emp_cdf_epi[-1]

        pdf_alea = gaussian_kde(zn_alea, bw_method="scott")(z_vals)
        emp_cdf_alea = np.cumsum(pdf_alea)
        emp_cdf_alea = emp_cdf_alea / emp_cdf_alea[-1]

        pdf_tot = gaussian_kde(zn_tot, bw_method="scott")(z_vals)
        emp_cdf_tot = np.cumsum(pdf_tot)
        emp_cdf_tot = emp_cdf_tot / emp_cdf_tot[-1]

        th_cdf = norm.cdf(z_vals)

        plt.figure(figsize=(6.5, 5))
        plt.plot(th_cdf, emp_cdf_alea, label="Aleatoric", linewidth=2)
        plt.plot(th_cdf, emp_cdf_epi, label="Epistemic", linewidth=2)
        plt.plot(th_cdf, emp_cdf_tot, label="Total", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration", linewidth=2)
        plt.fill_between(th_cdf, emp_cdf_alea, th_cdf, alpha=0.2)
        plt.fill_between(th_cdf, emp_cdf_epi, th_cdf, alpha=0.2)
        plt.fill_between(th_cdf, emp_cdf_tot, th_cdf, alpha=0.2)
        plt.xlabel("Expected Cumulative Distribution")
        plt.ylabel("Observed Cumulative Distribution")
        plt.legend(loc="upper left")
        plt.grid(zorder=0)
        plt.tight_layout()
        plt.savefig(path_png, dpi=300)
        plt.close()

        m_area_alea = float(simps(np.abs(emp_cdf_alea - th_cdf), x=th_cdf))
        m_area_epi = float(simps(np.abs(emp_cdf_epi - th_cdf), x=th_cdf))
        m_area_tot = float(simps(np.abs(emp_cdf_tot - th_cdf), x=th_cdf))
        if path_txt is not None:
            with open(path_txt, "w") as f:
                f.write(f"Miscalibration area (alea): {m_area_alea:.6e}\n")
                f.write(f"Miscalibration area (epi): {m_area_epi:.6e}\n")
                f.write(f"Miscalibration area (total): {m_area_tot:.6e}\n")
    except Exception:
        pass


def _save_rstd_components_plot(y_pred_mean, sigma_epi, sigma_ale, sigma_tot, path_png):
    """Plot histograms and optional KDE overlays for rStd (%) components."""
    mu = np.asarray(y_pred_mean, dtype=np.float64).reshape(-1)
    eps = np.finfo(float).eps

    def _to_rstd(sig):
        if sig is None:
            return None
        s = np.asarray(sig, dtype=np.float64).reshape(-1)
        return 100.0 * (s / np.maximum(np.abs(mu), eps))

    r_epi = _to_rstd(sigma_epi)
    r_ale = _to_rstd(sigma_ale)
    r_tot = _to_rstd(sigma_tot)

    sets = [
        ("Aleatoric", r_ale),
        ("Epistemic", r_epi),
        ("Total", r_tot),
    ]

    plt.figure(figsize=(12, 4))
    for i, (title, arr) in enumerate(sets, start=1):
        plt.subplot(1, 3, i)
        if arr is None or len(arr) == 0:
            plt.title(title + " (n/a)")
            plt.axis("off")
            continue
        plt.hist(arr, bins=40, density=True, alpha=0.6, edgecolor="black")
        if _HAVE_SCIPY and len(arr) > 2:
            try:
                kde = gaussian_kde(arr)
                xs = np.linspace(np.min(arr), np.max(arr), 256)
                plt.plot(xs, kde(xs))
            except Exception:
                pass
        plt.title(title)
        plt.xlabel("rStd (%)")
        plt.ylabel("Density")
        plt.grid(alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()


def _save_std_components_plot(y_pred_mean, sigma_epi, sigma_ale, sigma_tot, path_png):
    """Plot histograms and optional KDE overlays for absolute std components."""
    sets = [("Aleatoric", sigma_ale), ("Epistemic", sigma_epi), ("Total", sigma_tot)]

    plt.figure(figsize=(12, 4))
    for i, (title, arr) in enumerate(sets, start=1):
        plt.subplot(1, 3, i)
        if arr is None or len(arr) == 0:
            plt.title(title + " (n/a)")
            plt.axis("off")
            continue
        a = np.asarray(arr, dtype=np.float64).reshape(-1)
        plt.hist(a, bins=50, density=True, alpha=0.6, edgecolor="black")
        if _HAVE_SCIPY and len(a) > 2:
            try:
                kde = gaussian_kde(a)
                xs = np.linspace(np.min(a), np.max(a), 256)
                plt.plot(xs, kde(xs))
            except Exception:
                pass
        plt.title(title)
        plt.xlabel("Std (-)")
        plt.ylabel("Density")
        plt.grid(alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()


def _save_loss_curve(history, path_png, title, ideal_y: Optional[float] = None,
                     warmup_epochs: Optional[int] = None, yscale: Optional[str] = None, keys: Optional[Iterable[str]] = None):
    """Save training loss curves from a Keras History object.

    Args:
        history: Keras History with a ``history`` dict.
        path_png: Destination PNG path.
        title: Plot title.
        yscale: Optional y-scale, for example "log".
        keys: Optional iterable of keys to plot. Defaults to all keys in ``history.history``.
    """
    keys = list(keys) if keys is not None else list(history.history.keys())
    plt.figure()
    for k in keys:
        plt.plot(history.history[k], label=k)
    if ideal_y is not None:
        plt.axhline(y=ideal_y, label="ideal", color="black", linestyle="dashed", alpha=0.8, zorder=0)
    if warmup_epochs is not None:
        plt.axvline(x=warmup_epochs, label="warmup_epochs", color="black", linestyle="dashed", alpha=0.8, zorder=0)    
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if yscale:
        plt.yscale(yscale)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path_png, dpi=300)
    plt.close()