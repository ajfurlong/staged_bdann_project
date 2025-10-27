"""Command-line interface for staged_bdann.

This entry point orchestrates configuration loading, environment setup, and execution of single or
ensemble runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tensorflow as tf

from .config import load_and_validate_config
from .utils import ensure_run_dir, set_global_determinism, save_versions
from .data import load_data
from .pipeline import run_member_once
from .multi_run import run_seed_ensemble


def main() -> int:
    """Parse arguments, load config, initialize environment, and execute the pipeline."""
    parser = argparse.ArgumentParser(description="Run staged Bayesian DANN training or ensemble.")
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to JSON configuration file."
    )
    args = parser.parse_args()

    cfg = load_and_validate_config(args.config)

    # Create run directory and log resolved configuration
    run_dir = ensure_run_dir(cfg["run_name"])
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Deterministic environment and GPU configuration
    set_global_determinism(cfg["seed"], intra_threads=2, inter_threads=2)
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass

    save_versions(run_dir)

    # Load and preprocess data
    fixed_data = load_data(cfg["data"], run_dir, seed=cfg["seed"])

    # Determine whether to run ensemble or single seed
    ens_cfg = cfg.get("seed_ensemble", {"enabled": False})
    if ens_cfg.get("enabled", False):
        run_seed_ensemble(cfg, run_dir, fixed_data)
    else:
        run_member_once(cfg, run_dir, cfg["seed"], fixed_data)

    print("Run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())