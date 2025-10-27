"""Seed-ensemble utilities for staged_bdann.

This module coordinates multiple Stage 1-2-3 runs over different seeds while keeping data splits
fixed by reusing the preloaded `fixed_data`. It aggregates metrics across members and writes
both member-level and aggregate outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .pipeline import run_member_once


def run_seed_ensemble(cfg: Dict[str, Any], run_dir: Path, fixed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run an ensemble of seeds using fixed data splits and aggregate Stage 3 metrics.

    The function expects `fixed_data` to already contain deterministic splits generated with a
    specific seed. Each ensemble member varies only the training randomness via `seed_i`. This keeps
    comparisons fair across members.

    Args:
        cfg: Full resolved configuration dictionary.
        run_dir: Base run directory where the ensemble artifacts will be saved.
        fixed_data: Precomputed data splits and scalers to reuse for every member.

    Returns:
        Dictionary of aggregate statistics (means and standard deviations) across numeric metrics,
        plus the `n_members` count. If the ensemble is not enabled, returns an empty dictionary.
    """
    ens_cfg = cfg.get("seed_ensemble", {"enabled": False})
    if not ens_cfg.get("enabled", False):
        return {}

    # Users can still choose data.mode == "split"; since we reuse `fixed_data`, splits remain fixed here.
    ens_dir = run_dir / "seed_ensemble"
    ens_dir.mkdir(parents=True, exist_ok=True)

    n_members = max(1, int(ens_cfg.get("n", 10)))
    base_seed = int(ens_cfg.get("base_seed", cfg.get("seed", 1234)))
    step = int(ens_cfg.get("step", 1))

    # Containers for predictions export
    y_true_list: List[np.ndarray] = []
    pred_rows_long: List[Dict[str, Any]] = []

    rows: List[Dict[str, Any]] = []
    for i in range(n_members):
        seed_i = base_seed + i * step
        member_dir = ens_dir / f"member_{i:03d}_seed_{seed_i}"
        y_true_i, y_pred_mean_i, y_pred_sigma_aleatoric_i, y_pred_sigma_epistemic_i, y_pred_sigma_total_i, metrics_i = run_member_once(cfg, member_dir, seed_i, fixed_data=fixed_data)
        metrics_i = dict(metrics_i)  # ensure mutable copy
        metrics_i["seed"] = int(seed_i)
        rows.append(metrics_i)

        # Save per-member predictions (true and predicted) in the member directory
        y_true_arr = np.asarray(y_true_i).reshape(-1)
        y_mean_arr = np.asarray(y_pred_mean_i).reshape(-1)
        y_ale_arr = None if y_pred_sigma_aleatoric_i is None else np.asarray(y_pred_sigma_aleatoric_i).reshape(-1)
        y_epi_arr = None if y_pred_sigma_epistemic_i is None else np.asarray(y_pred_sigma_epistemic_i).reshape(-1)
        y_tot_arr = None if y_pred_sigma_total_i is None else np.asarray(y_pred_sigma_total_i).reshape(-1)

        member_pred_df = pd.DataFrame({
            "y_true": y_true_arr,
            "y_pred_mean": y_mean_arr,
            **({"sigma_ale": y_ale_arr} if y_ale_arr is not None else {}),
            **({"sigma_epi": y_epi_arr} if y_epi_arr is not None else {}),
            **({"sigma_total": y_tot_arr} if y_tot_arr is not None else {}),
        })
        (member_dir / "predictions").mkdir(parents=True, exist_ok=True)
        member_pred_df.to_csv(member_dir / "predictions" / "predictions.csv", index=False)

        # Accumulate for long-format ensemble export
        if i == 0:
            y_true_list.append(y_true_arr)
        for idx in range(y_mean_arr.shape[0]):
            rec = {
                "member_index": int(i),
                "seed": int(seed_i),
                "index": int(idx),
                "y_true": float(y_true_arr[idx]),
                "y_pred_mean": float(y_mean_arr[idx]),
            }
            if y_ale_arr is not None:
                rec["sigma_ale"] = float(y_ale_arr[idx])
            if y_epi_arr is not None:
                rec["sigma_epi"] = float(y_epi_arr[idx])
            if y_tot_arr is not None:
                rec["sigma_total"] = float(y_tot_arr[idx])
            pred_rows_long.append(rec)

    ens_df = pd.DataFrame(rows)
    ens_df.to_csv(ens_dir / "seed_ensemble_stage3_metrics.csv", index=False)

    # Aggregate numeric columns, excluding the seed column
    numeric_cols = [
        c for c in ens_df.columns if c != "seed" and pd.api.types.is_numeric_dtype(ens_df[c])
    ]
    agg: Dict[str, Any] = {f"{k}_mean": float(ens_df[k].mean()) for k in numeric_cols}
    agg.update({f"{k}_std": float(ens_df[k].std(ddof=1)) for k in numeric_cols})
    agg["n_members"] = int(len(ens_df))

    # Write long-format predictions across all members (no aggregation bookkeeping)
    pred_long_df = pd.DataFrame(pred_rows_long)
    pred_long_path = ens_dir / "seed_ensemble_predictions_long.csv"
    pred_long_df.to_csv(pred_long_path, index=False)

    # Identify best-performing member by rRMSE
    metric_key = "[APE] rRMSE (%)"
    if metric_key in ens_df.columns:
        best_idx = int(ens_df[metric_key].idxmin())
        best_row = ens_df.loc[best_idx].to_dict()
        best_member = {
            "member_index": int(best_row.get("seed", 0) - base_seed) // step if step != 0 else 0,
            "seed": int(best_row.get("seed", base_seed)),
            "rRMSE_percent": float(best_row[metric_key]),
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                         for k, v in best_row.items() if k != "seed"},
        }
        # Write JSON and a small text summary
        with open(ens_dir / "best_member.json", "w") as f:
            json.dump(best_member, f, indent=2)
        with open(ens_dir / "best_member.txt", "w") as f:
            f.write(
                (
                    f"Best member by rRMSE: seed={best_member['seed']}, index={best_member['member_index']}\n"
                    f"rRMSE (%): {best_member['rRMSE_percent']:.6f}\n"
                )
            )
    else:
        # If rRMSE is missing, skip best-member reporting
        best_member = None

    # Persist aggregate JSON exactly as before (metrics only)
    with open(ens_dir / "seed_ensemble_stage3_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Return includes references to new artifacts without changing the aggregate file format
    out = dict(agg)
    out["predictions_long_csv"] = str(pred_long_path)
    if "best_member" in locals() and best_member is not None:
        out["best_member_json"] = str(ens_dir / "best_member.json")
        out["best_member_txt"] = str(ens_dir / "best_member.txt")
    return out