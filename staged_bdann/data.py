"""Data I/O, splitting, preprocessing, and scaling utilities for staged_bdann.

This module loads source and target datasets under flexible I/O modes, performs deterministic
splits, applies optional noise injection, and standardizes features and targets. It returns
standardized splits, scalers, domain labels, and sizes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Low-level helpers (private)
# -----------------------------

def _read_df(csv_path: str) -> pd.DataFrame:
    """Read a CSV into a pandas DataFrame with stripped column names.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        DataFrame with trimmed column names.
    """
    df = pd.read_csv(csv_path, header=0)
    df.columns = df.columns.str.strip()
    return df


def _get_Xy_from_df(df: pd.DataFrame, features: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix X and target vector y from a DataFrame.

    Args:
        df: Input DataFrame containing feature and target columns.
        features: Ordered list of feature column names.
        target_col: Name of the target column.

    Returns:
        Tuple of (X, y) where X is shape (N, F) and y is shape (N, 1), both float64 arrays.
    """
    X = df.loc[:, features].to_numpy(dtype=np.float64)
    y = df.loc[:, [target_col]].to_numpy(dtype=np.float64)
    return X, y


def _standardize_joint(
    X_src_tr: np.ndarray,
    X_tgt_tr: np.ndarray,
    X_src_v: np.ndarray,
    X_src_te: np.ndarray,
    X_tgt_v: np.ndarray,
    X_tgt_te: np.ndarray,
    y_src_tr: np.ndarray,
    y_src_v: np.ndarray,
    y_src_te: np.ndarray,
    y_tgt_tr: np.ndarray,
    y_tgt_v: np.ndarray,
    y_tgt_te: np.ndarray,
    do_X: bool = True,
    do_y: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[Any],
    Optional[Any],
    Optional[Any],
]:
    """Standardize features jointly across source+target train, and targets per-domain.

    Feature scaler is fit on the vertical stack of source and target train features. Targets are
    standardized with separate scalers per domain to preserve domain-wise calibration.

    Args:
        X_* and y_*: Source and target arrays for train, valid, and test.
        do_X: Whether to standardize features.
        do_y: Whether to standardize targets.

    Returns:
        Standardized arrays in the same order as inputs, followed by (X_scaler, y_scaler_src, y_scaler_tgt).
    """
    from sklearn.preprocessing import StandardScaler

    if do_X:
        X_scaler = StandardScaler()
        X_scaler.fit(np.vstack([X_src_tr, X_tgt_tr]))
        X_src_tr_s = X_scaler.transform(X_src_tr)
        X_src_v_s = X_scaler.transform(X_src_v)
        X_src_te_s = X_scaler.transform(X_src_te)
        X_tgt_tr_s = X_scaler.transform(X_tgt_tr)
        X_tgt_v_s = X_scaler.transform(X_tgt_v)
        X_tgt_te_s = X_scaler.transform(X_tgt_te)
    else:
        X_scaler = None
        X_src_tr_s, X_src_v_s, X_src_te_s = X_src_tr, X_src_v, X_src_te
        X_tgt_tr_s, X_tgt_v_s, X_tgt_te_s = X_tgt_tr, X_tgt_v, X_tgt_te

    if do_y:
        y_scaler_src = StandardScaler()
        y_scaler_src.fit(y_src_tr)
        y_src_tr_s = y_scaler_src.transform(y_src_tr)
        y_src_v_s = y_scaler_src.transform(y_src_v)
        y_src_te_s = y_scaler_src.transform(y_src_te)

        y_scaler_tgt = StandardScaler()
        y_scaler_tgt.fit(y_tgt_tr)
        y_tgt_tr_s = y_scaler_tgt.transform(y_tgt_tr)
        y_tgt_v_s = y_scaler_tgt.transform(y_tgt_v)
        y_tgt_te_s = y_scaler_tgt.transform(y_tgt_te)
    else:
        y_scaler_src = None
        y_scaler_tgt = None
        y_src_tr_s, y_src_v_s, y_src_te_s = y_src_tr, y_src_v, y_src_te
        y_tgt_tr_s, y_tgt_v_s, y_tgt_te_s = y_tgt_tr, y_tgt_v, y_tgt_te

    return (
        X_src_tr_s,
        X_src_v_s,
        X_src_te_s,
        X_tgt_tr_s,
        X_tgt_v_s,
        X_tgt_te_s,
        y_src_tr_s,
        y_src_v_s,
        y_src_te_s,
        y_tgt_tr_s,
        y_tgt_v_s,
        y_tgt_te_s,
        X_scaler,
        y_scaler_src,
        y_scaler_tgt,
    )


# ---------------------------------
# Public data loading entry point
# ---------------------------------

def load_data(cfg_data: Dict[str, Any], run_dir: Path, seed: int) -> Dict[str, Any]:
    """Load source and target datasets, split, standardize, and optionally inject noise.

    This function supports two I/O modes via cfg_data["mode"]:
      - "files": explicit train/valid/test CSVs per domain, or a single_csv.
      - "split": a single_csv per domain with split ratios in cfg_data["splits"].

    Target column resolution:
      - Per-domain: if data.source["outputs"] or data.target["outputs"] is provided (string or single-item list), that is used.
      - Otherwise, a global data.target_column may be used for backward compatibility.

    Args:
        cfg_data: The "data" configuration dictionary.
        run_dir: Output run directory used for any persistent files (not used for splits).
        seed: Global seed for reproducible splitting and noise.

    Returns:
        Dictionary containing standardized arrays, scalers, sizes, and domain labels.

    Raises:
        ValueError: If required paths, columns, or split settings are invalid.
    """
    rng = np.random.RandomState(seed)

    mode = cfg_data["mode"]
    src = cfg_data["source"]
    tgt = cfg_data["target"]
    splits = cfg_data["splits"]
    noise = cfg_data["noise_injection"]
    scaling = cfg_data["scaling"]

    # Target column resolution from per-domain outputs (preferred), with backward compatibility
    def _resolve_output_name(block: Dict[str, Any], fallback: Optional[str]) -> str:
        val = block.get("outputs", None)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], str):
            return val[0]
        if fallback is not None:
            return fallback
        raise ValueError(
            "Each of data.source and data.target must define 'outputs' as a string (or single-item list)."
        )

    # Backward compatibility: allow a global data.target_column if present
    global_target_legacy = cfg_data.get("target_column")
    src_target = _resolve_output_name(src, global_target_legacy)
    tgt_target = _resolve_output_name(tgt, global_target_legacy)

    # Validate splits quickly here to avoid silent errors downstream
    if abs(sum(splits["source"]) - 1.0) > 1e-8 or abs(sum(splits["target"]) - 1.0) > 1e-8:
        raise ValueError("data.splits must sum to 1.0 for both source and target.")

    # Read dataframes according to mode
    if mode == "files":
        # Source
        if src.get("single_csv"):
            dfS = _read_df(src["single_csv"])
            from sklearn.model_selection import train_test_split

            Xs, ys = _get_Xy_from_df(dfS, src["features"], src_target)
            Xs_tr, Xs_tv, ys_tr, ys_tv = train_test_split(
                Xs, ys, test_size=(1 - splits["source"][0]), random_state=seed, shuffle=True
            )
            Xs_v, Xs_te, ys_v, ys_te = train_test_split(
                Xs_tv,
                ys_tv,
                test_size=splits["source"][2] / (1 - splits["source"][0]),
                random_state=seed,
                shuffle=True,
            )
        else:
            Xs_tr, ys_tr = _get_Xy_from_df(_read_df(src["train_csv"]), src["features"], src_target)
            Xs_v, ys_v = _get_Xy_from_df(_read_df(src["valid_csv"]), src["features"], src_target)
            Xs_te, ys_te = _get_Xy_from_df(_read_df(src["test_csv"]), src["features"], src_target)

        # Target
        if tgt.get("single_csv"):
            dfT = _read_df(tgt["single_csv"])
            Xt, yt = _get_Xy_from_df(dfT, tgt["features"], tgt_target)
        else:
            Xt_tr, yt_tr = _get_Xy_from_df(_read_df(tgt["train_csv"]), tgt["features"], tgt_target)
            Xt_v, yt_v = _get_Xy_from_df(_read_df(tgt["valid_csv"]), tgt["features"], tgt_target)
            Xt_te, yt_te = _get_Xy_from_df(_read_df(tgt["test_csv"]), tgt["features"], tgt_target)
            dfT = None  # not used when triad provided

        # If target uses single_csv, always perform deterministic split using train_test_split and splits["target"]
        if tgt.get("single_csv"):
            from sklearn.model_selection import train_test_split
            Xt_tr, Xt_tv, yt_tr, yt_tv = train_test_split(
                Xt, yt, test_size=(1 - splits["target"][0]), random_state=seed, shuffle=True
            )
            Xt_v, Xt_te, yt_v, yt_te = train_test_split(
                Xt_tv,
                yt_tv,
                test_size=splits["target"][2] / (1 - splits["target"][0]),
                random_state=seed,
                shuffle=True,
            )

    else:
        # mode == "split"
        from sklearn.model_selection import train_test_split

        # Source
        dfS = _read_df(src["single_csv"])
        Xs, ys = _get_Xy_from_df(dfS, src["features"], src_target)
        Xs_tr, Xs_tv, ys_tr, ys_tv = train_test_split(
            Xs, ys, test_size=(1 - splits["source"][0]), random_state=seed, shuffle=True
        )
        Xs_v, Xs_te, ys_v, ys_te = train_test_split(
            Xs_tv,
            ys_tv,
            test_size=splits["source"][2] / (1 - splits["source"][0]),
            random_state=seed,
            shuffle=True,
        )

        # Target
        dfT = _read_df(tgt["single_csv"])
        Xt, yt = _get_Xy_from_df(dfT, tgt["features"], tgt_target)
        # Always perform deterministic split using train_test_split and splits["target"]
        Xt_tr, Xt_tv, yt_tr, yt_tv = train_test_split(
            Xt, yt, test_size=(1 - splits["target"][0]), random_state=seed, shuffle=True
        )
        Xt_v, Xt_te, yt_v, yt_te = train_test_split(
            Xt_tv,
            yt_tv,
            test_size=splits["target"][2] / (1 - splits["target"][0]),
            random_state=seed,
            shuffle=True,
        )

    # Domain labels (0 for source, 1 for target)
    ds_tr = np.zeros((len(Xs_tr), 1), dtype=np.float32)
    ds_v = np.zeros((len(Xs_v), 1), dtype=np.float32)
    ds_te = np.zeros((len(Xs_te), 1), dtype=np.float32)

    dt_tr = np.ones((len(Xt_tr), 1), dtype=np.float32)
    dt_v = np.ones((len(Xt_v), 1), dtype=np.float32)
    dt_te = np.ones((len(Xt_te), 1), dtype=np.float32)

    # Shared noise schedule if enabled. Use rng for determinism.
    if noise.get("enabled", True):
        nf = float(noise.get("target_factor", noise.get("source_factor", 0.0)))
        if nf > 0.0:
            s_std = Xs_tr.std(axis=0)
            t_std = Xt_tr.std(axis=0)
            Xs_tr = Xs_tr + rng.normal(0.0, nf * s_std, size=Xs_tr.shape)
            Xt_tr = Xt_tr + rng.normal(0.0, nf * t_std, size=Xt_tr.shape)

    # Joint standardization for X; separate for y
    (
        Xs_tr_s,
        Xs_v_s,
        Xs_te_s,
        Xt_tr_s,
        Xt_v_s,
        Xt_te_s,
        ys_tr_s,
        ys_v_s,
        ys_te_s,
        yt_tr_s,
        yt_v_s,
        yt_te_s,
        X_scaler,
        y_scaler_src,
        y_scaler_tgt,
    ) = _standardize_joint(
        Xs_tr,
        Xt_tr,
        Xs_v,
        Xs_te,
        Xt_v,
        Xt_te,
        ys_tr,
        ys_v,
        ys_te,
        yt_tr,
        yt_v,
        yt_te,
        do_X=bool(scaling.get("standardize_X", True)),
        do_y=bool(scaling.get("standardize_y", True)),
    )

    # Sanity checks to prevent zero-sized splits that can crash training
    for arr, name in [
        (Xs_tr, "Xs_tr"),
        (ys_tr, "ys_tr"),
        (Xt_tr, "Xt_tr"),
        (yt_tr, "yt_tr"),
        (Xs_v, "Xs_v"),
        (ys_v, "ys_v"),
        (Xt_v, "Xt_v"),
        (yt_v, "yt_v"),
    ]:
        if arr is None or len(arr) == 0:
            raise ValueError(f"Empty array detected for {name}. Check data.splits or fixed_test_size in the JSON.")

    # Targets must be (N, 1)
    for y_arr, y_name in [
        (ys_tr, "ys_tr"),
        (ys_v, "ys_v"),
        (ys_te, "ys_te"),
        (yt_tr, "yt_tr"),
        (yt_v, "yt_v"),
        (yt_te, "yt_te"),
    ]:
        if y_arr.ndim != 2 or y_arr.shape[1] != 1:
            raise ValueError(
                f"Target array {y_name} must have shape (N, 1); got {y_arr.shape}. Verify your target column selection."
            )

    sizes_source = np.array([len(Xs_tr_s), len(Xs_v_s), len(Xs_te_s)], dtype=np.int64)
    sizes_target = np.array([len(Xt_tr_s), len(Xt_v_s), len(Xt_te_s)], dtype=np.int64)

    return {
        "Xs_tr": Xs_tr_s,
        "Xs_v": Xs_v_s,
        "Xs_te": Xs_te_s,
        "ys_tr": ys_tr_s,
        "ys_v": ys_v_s,
        "ys_te": ys_te_s,
        "ds_tr": ds_tr,
        "ds_v": ds_v,
        "ds_te": ds_te,
        "Xt_tr": Xt_tr_s,
        "Xt_v": Xt_v_s,
        "Xt_te": Xt_te_s,
        "yt_tr": yt_tr_s,
        "yt_v": yt_v_s,
        "yt_te": yt_te_s,
        "dt_tr": dt_tr,
        "dt_v": dt_v,
        "dt_te": dt_te,
        "X_scaler": X_scaler,
        "y_scaler_src": y_scaler_src,
        "y_scaler_tgt": y_scaler_tgt,
        "sizes_source": sizes_source,
        "sizes_target": sizes_target,
    }