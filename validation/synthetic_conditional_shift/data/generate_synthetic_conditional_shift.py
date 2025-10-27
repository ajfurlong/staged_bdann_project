#!/usr/bin/env python3
"""Synthetic data generator with a fixed 500-sample target training subset.

Outputs in --outdir:
    source_train.csv, source_val.csv, source_test.csv
    target_all.csv, target_val.csv, target_test.csv, target_train_pool.csv
    target_train_500.csv
    meta.json
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


# --------------------------
# Helpers
# --------------------------

def soft_open_interval(s: np.ndarray, eps: float = 0.02, k: float = 2.0) -> np.ndarray:
    """Map values in [0, 1] into an open interval (eps, 1-eps) smoothly."""
    s = np.clip(s, 0.0, 1.0)
    s_sig = 1.0 / (1.0 + np.exp(-k * (s - 0.5)))  # (0, 1)
    return eps + (1.0 - 2.0 * eps) * s_sig


# --------------------------
# Deterministic latent functions (noiseless)
# --------------------------

def _synthetic_core(X: np.ndarray) -> np.ndarray:
    y = (
        2.0
        + 5.0 * np.sin(2.0 * X[:, 0] + 1.2 * X[:, 1])
        + 0.8 * np.cos(1.8 * X[:, 0] * X[:, 1])
        + np.log1p(X[:, 1] ** 2)
        - 0.5 * X[:, 2]
        + 0.4 * X[:, 3] ** 2
        - 0.2 * X[:, 4]
        + 0.6 * np.sin(1.5 * X[:, 0] * X[:, 2])
        + 0.5 * np.cos(2.2 * X[:, 2] * X[:, 3])
        + 0.3 * np.sin(1.7 * (X[:, 0] + X[:, 4]) * X[:, 2])
    )
    return 2.0 * y


def _shifted_core(X: np.ndarray) -> np.ndarray:
    Xm = X.copy()
    Xm[:, 0] = 1.2 * np.sin(1.3 * X[:, 0]) + 1.5
    Xm[:, 1] = X[:, 1] + 0.4 * np.cos(1.5 * X[:, 2])
    Xm[:, 2] = X[:, 2] + 0.3 * np.sin(0.8 * X[:, 0] * X[:, 1])
    y = (
        2.0
        + 5.0 * np.sin(2.2 * Xm[:, 0] + 1.0 * Xm[:, 1])
        + 0.8 * np.cos(2.0 * Xm[:, 0] * Xm[:, 1])
        + np.log1p(Xm[:, 1] ** 2)
        - 0.4 * Xm[:, 2]
        + 0.35 * Xm[:, 3] ** 2
        - 0.3 * Xm[:, 4]
        + 0.6 * np.sin(1.8 * Xm[:, 0] * Xm[:, 2])
        + 0.5 * np.cos(2.0 * Xm[:, 2] * Xm[:, 3])
        + 0.3 * np.sin(1.9 * (Xm[:, 0] + Xm[:, 4]) * Xm[:, 2])
    )
    return 2.0 * y


# --------------------------
# Data generation (noisy observed targets)
# --------------------------

def generate_source(n: int, rng: np.random.Generator, noise_std: float = 0.05) -> pd.DataFrame:
    """Generate source-domain samples."""
    X = rng.uniform(1.0, 3.0, size=(n, 5))
    y0 = _synthetic_core(X)
    eps = rng.normal(0.0, float(noise_std), size=n)
    y = y0 + eps
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    df["domain"] = 0
    return df


def generate_target(n: int, rng: np.random.Generator, noise_std: float = 0.05) -> pd.DataFrame:
    """Generate target-domain samples with a shifted latent mapping."""
    X = rng.uniform(1.0, 3.0, size=(n, 5))
    y0 = _shifted_core(X)
    eps = rng.normal(0.0, float(noise_std), size=n)
    y = y0 + eps
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(5)])
    df["target"] = y
    df["domain"] = 1
    return df


# --------------------------
# Main
# --------------------------

def main() -> None:
    """Generate synthetic CSVs and metadata with fixed settings and a 500-point target subset."""
    ap = argparse.ArgumentParser(description="Generate synthetic CSVs for a staged B-DANN benchmark.")
    ap.add_argument("--outdir", type=Path, default=Path("./synthetic_data_v2"), help="Output directory")
    ap.add_argument("--seed", type=int, default=1234, help="Global RNG seed")
    args = ap.parse_args()

    # Hardcoded settings
    n_source_train = 5000
    n_source_val = 1000
    n_source_test = 1000
    n_target_total = 1000  # must be >= n_target_val + n_test + 500
    n_test = 250
    n_target_val = 250
    ablation_size = 500

    noise_std_source = 0.05
    noise_std_target = 0.05

    scale_mode = "global_quantile"  # choices: none, global_quantile, per_domain_quantile
    range_min, range_max = 1.0, 5.0
    q_low, q_high = 0.05, 0.95
    soft_k, soft_eps = 2.0, 0.02

    rng = np.random.default_rng(args.seed)

    # Generate source
    n_source_total = n_source_train + n_source_val + n_source_test
    df_source_all = generate_source(n_source_total, rng, noise_std=noise_std_source)

    # Generate target
    if n_target_val + n_test + ablation_size > n_target_total:
        raise ValueError("n_target_total must be >= n_target_val + n_test + 500")
    df_target_all = generate_target(n_target_total, rng, noise_std=noise_std_target)

    # Scaling on 'target' only
    def _scale_targets_soft(y: np.ndarray, y_lo: float, y_hi: float, rmin: float, rmax: float, eps: float, k: float) -> np.ndarray:
        denom = max(1e-12, (y_hi - y_lo))
        t = (y - y_lo) / denom
        t_clip = np.clip(t, 0.0, 1.0)
        s_core = soft_open_interval(t_clip, eps=eps, k=k)
        tail = (t - t_clip) * (1.0 - 2.0 * eps)
        s = s_core + tail
        return rmin + (rmax - rmin) * s

    if scale_mode != "none":
        rmin, rmax = float(range_min), float(range_max)
        eps, k = float(soft_eps), float(soft_k)
        if scale_mode == "global_quantile":
            y_concat = np.concatenate([df_source_all["target"].to_numpy(), df_target_all["target"].to_numpy()])
            y_lo = float(np.quantile(y_concat, q_low))
            y_hi = float(np.quantile(y_concat, q_high))
            df_source_all["target"] = _scale_targets_soft(df_source_all["target"].to_numpy(), y_lo, y_hi, rmin, rmax, eps, k)
            df_target_all["target"] = _scale_targets_soft(df_target_all["target"].to_numpy(), y_lo, y_hi, rmin, rmax, eps, k)
        else:
            y_lo_s = float(np.quantile(df_source_all["target"].to_numpy(), q_low))
            y_hi_s = float(np.quantile(df_source_all["target"].to_numpy(), q_high))
            df_source_all["target"] = _scale_targets_soft(df_source_all["target"].to_numpy(), y_lo_s, y_hi_s, rmin, rmax, eps, k)
            y_lo_t = float(np.quantile(df_target_all["target"].to_numpy(), q_low))
            y_hi_t = float(np.quantile(df_target_all["target"].to_numpy(), q_high))
            df_target_all["target"] = _scale_targets_soft(df_target_all["target"].to_numpy(), y_lo_t, y_hi_t, rmin, rmax, eps, k)
        # Ensure SOURCE targets are strictly positive in benchmark units
        df_source_all["target"] = np.maximum(df_source_all["target"].to_numpy(), max(1e-8, rmin))

    # Split SOURCE after scaling
    idx_src = np.arange(n_source_total)
    src_perm = rng.permutation(idx_src)
    i0 = n_source_train
    i1 = i0 + n_source_val
    src_train_idx = src_perm[:i0]
    src_val_idx = src_perm[i0:i1]
    src_test_idx = src_perm[i1 : i1 + n_source_test]
    df_source_train = df_source_all.iloc[src_train_idx].reset_index(drop=True)
    df_source_val = df_source_all.iloc[src_val_idx].reset_index(drop=True)
    df_source_test = df_source_all.iloc[src_test_idx].reset_index(drop=True)

    # Target splits: test, validation, then training pool
    idx_all = np.arange(n_target_total)
    test_idx = rng.choice(idx_all, size=n_test, replace=False)
    remaining = np.setdiff1d(idx_all, test_idx, assume_unique=False)
    val_idx = rng.choice(remaining, size=n_target_val, replace=False)
    mask_test = np.zeros(n_target_total, dtype=bool)
    mask_val = np.zeros(n_target_total, dtype=bool)
    mask_test[test_idx] = True
    mask_val[val_idx] = True
    df_target_test = df_target_all.loc[mask_test].reset_index(drop=True)
    df_target_val = df_target_all.loc[mask_val].reset_index(drop=True)
    df_target_pool = df_target_all.loc[~(mask_test | mask_val)].reset_index(drop=True)

    if ablation_size > len(df_target_pool):
        raise ValueError("500-sample ablation exceeds available training pool size.")
    order = rng.permutation(len(df_target_pool))
    df_t500 = df_target_pool.iloc[order[:ablation_size]].reset_index(drop=True)

    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)

    # Save CSVs
    def _save(df: pd.DataFrame, name: str) -> None:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / name, index=False)

    _save(df_source_train, "source_train.csv")
    _save(df_source_val, "source_val.csv")
    _save(df_source_test, "source_test.csv")

    _save(df_target_all, "target_all.csv")
    _save(df_target_val, "target_val.csv")
    _save(df_target_test, "target_test.csv")
    _save(df_target_pool, "target_train_pool.csv")

    _save(df_t500, "target_train_500.csv")

    # Minimal metadata
    meta = {
        "seed": int(args.seed),
        "source_sizes": {
            "train": int(n_source_train),
            "val": int(n_source_val),
            "test": int(n_source_test),
            "total": int(n_source_total),
        },
        "target_sizes": {
            "total": int(n_target_total),
            "val": int(n_target_val),
            "test": int(n_test),
            "pool_after_val_test": int(len(df_target_pool)),
            "train_subset": 500,
        },
        "scaling": {
            "mode": scale_mode,
            "range": [float(range_min), float(range_max)],
            "q_low": float(q_low),
            "q_high": float(q_high),
            "soft_k": float(soft_k),
        },
        "columns": {
            "features": [f"feat_{i}" for i in range(5)],
            "target": "target",
            "domain": "domain",
        },
        "notes": [
            "Five features in (1.0, 3.0); deterministic latent functions with small Gaussian noise on targets.",
            "Target_train_500 is sampled from target_train_pool with a fixed RNG seed for reproducibility.",
            "Domain: 0=source, 1=target.",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Done. Files written to: {out.resolve()}")
    print(
        "Wrote: source_train.csv, source_val.csv, source_test.csv, target_all.csv, target_val.csv, "
        "target_test.csv, target_train_pool.csv, target_train_500.csv"
    )


if __name__ == "__main__":
    main()