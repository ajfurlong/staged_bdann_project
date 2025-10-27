"""Reproducibility, run directory management, and artifact utilities.

This module centralizes utilities that are shared across the staged B-DANN pipeline:
  1) Deterministic seeding and TensorFlow thread pinning.
  2) Run directory creation with a consistent layout.
  3) Model summary export, metadata persistence, predictions export, and version stamping.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp


def set_global_determinism(seed: int, intra_threads: int = 2, inter_threads: int = 2) -> None:
    """Set deterministic behavior across Python, NumPy, and TensorFlow.

    Args:
        seed: Global random seed used for Python, NumPy, and TensorFlow.
        intra_threads: Number of intra-op threads for TensorFlow.
        inter_threads: Number of inter-op threads for TensorFlow.

    Returns:
        None.
    """
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra_threads))
        tf.config.threading.set_inter_op_parallelism_threads(int(inter_threads))
        # Not all TF builds expose this, so guard it.
        tf.config.experimental.enable_op_determinism()  # type: ignore[attr-defined]
    except Exception:
        # Keep silent but deterministic where possible.
        pass


def ensure_run_dir(run_name: str) -> Path:
    """Create a timestamped run directory with standard subdirectories.

    The directory structure is:
        runs/<YYYYMMDD_HHMMSS>_<run_name>/
            plots/   arch/   splits/   weights/   logs/

    Args:
        run_name: Short name used to identify this run.

    Returns:
        Path to the created run directory.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Subdirectories for organization
    for sub in ("plots", "arch", "splits", "weights", "logs"):
        (run_dir / sub).mkdir(exist_ok=True)

    return run_dir


def save_model_summary(model: tf.keras.Model, path_txt: Path) -> None:
    """Write the Keras model summary to a text file.

    Args:
        model: Keras model to summarize.
        path_txt: Destination text filepath.

    Returns:
        None.
    """
    with open(path_txt, "w") as f:
        model.summary(print_fn=lambda s: f.write(s + "\n"))


def save_metadata_h5(
    run_dir: Path,
    X_scaler: Optional[Any],
    y_scaler_src: Optional[Any],
    y_scaler_tgt: Optional[Any],
    arch_json: dict,
) -> None:
    """Persist scalers and architecture information to metadata.h5.

    This function does not store fixed test indices. That mechanism has been removed.

    Args:
        run_dir: Run directory where metadata.h5 will be written.
        X_scaler: Feature StandardScaler or compatible object with mean_ and scale_.
        y_scaler_src: Source-domain target scaler, or None if not used.
        y_scaler_tgt: Target-domain target scaler, or None if not used.
        arch_json: Dictionary describing the model architecture.

    Returns:
        None.
    """
    path = run_dir / "metadata.h5"
    with h5py.File(path, "w") as h5:
        g = h5.create_group("scaler")
        if X_scaler is not None:
            g.create_dataset("x_mean", data=X_scaler.mean_.astype(np.float64))
            g.create_dataset("x_std", data=X_scaler.scale_.astype(np.float64))
        if y_scaler_src is not None:
            g.create_dataset("y_mean_source", data=y_scaler_src.mean_.astype(np.float64))
            g.create_dataset("y_std_source", data=y_scaler_src.scale_.astype(np.float64))
        if y_scaler_tgt is not None:
            g.create_dataset("y_mean_target", data=y_scaler_tgt.mean_.astype(np.float64))
            g.create_dataset("y_std_target", data=y_scaler_tgt.scale_.astype(np.float64))

        ga = h5.create_group("arch")
        ga.create_dataset("json", data=np.string_(json.dumps(arch_json, indent=2)))


def save_predictions_csv(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_sigma_aleatoric: Optional[np.ndarray],
    y_pred_sigma_epistemic: Optional[np.ndarray],
    y_pred_sigma_total: Optional[np.ndarray],
) -> None:
    """Save predictions and optional UQ sigmas to CSV.

    Args:
        path: Destination CSV filepath.
        y_true: Array of true targets with shape (N, 1) or (N,).
        y_pred: Array of predictions with shape (N, 1) or (N,).
        y_pred_sigma_aleatoric: Optional, aleatoric std dev per sample.
        y_pred_sigma_epistemic: Optional, epistemic std dev per sample.
        y_pred_sigma_total: Optional, total predictive std dev per sample.

    Returns:
        None.
    """
    if y_pred_sigma_aleatoric is not None and y_pred_sigma_epistemic is not None and y_pred_sigma_total is not None:
        df = pd.DataFrame(
            {
                "y_true": np.asarray(y_true).ravel(),
                "y_pred": np.asarray(y_pred).ravel(),
                "y_pred_sigma_alea": np.asarray(y_pred_sigma_aleatoric).ravel(),
                "y_pred_sigma_epi": np.asarray(y_pred_sigma_epistemic).ravel(),
                "y_pred_sigma_tot": np.asarray(y_pred_sigma_total).ravel(),
            }
        )
    else:
        df = pd.DataFrame(
            {
                "y_true": np.asarray(y_true).ravel(),
                "y_pred": np.asarray(y_pred).ravel(),
            }
        )
    df.to_csv(path, index=False)


def save_versions(run_dir: Path) -> None:
    """Write a versions.json file with environment versions for reproducibility.

    Args:
        run_dir: Run directory where logs/versions.json is written.

    Returns:
        None.
    """
    info = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "h5py": h5py.__version__,
    }
    with open(run_dir / "logs" / "versions.json", "w") as f:
        json.dump(info, f, indent=2)