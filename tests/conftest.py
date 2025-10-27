# Ensure repo root is on sys.path when running from anywhere
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import numpy as np
import pandas as pd
import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "123"


@pytest.fixture(scope="session", autouse=True)
def _seed_all():
    import random, tensorflow as tf
    np.random.seed(123)
    random.seed(123)
    tf.keras.utils.set_random_seed(123)


@pytest.fixture
def tmp_run_dir(tmp_path):
    d = tmp_path / "runs" / "test_run"
    for sub in ["plots", "arch", "splits", "weights", "logs"]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def toy_csvs(tmp_path):
    Ns, Nt = 64, 64
    rng = np.random.RandomState(0)
    src = pd.DataFrame({
        "f1": rng.randn(Ns),
        "f2": rng.randn(Ns),
        "f3": rng.randn(Ns),
        "y":  2.0 * rng.randn(Ns) + 1.0,
    })
    tgt = pd.DataFrame({
        "f2": rng.randn(Nt),
        "f1_tgt": rng.randn(Nt),
        "f3": rng.randn(Nt),
        "y":  2.0 * rng.randn(Nt) - 1.0,
    })
    src_path = tmp_path / "src.csv"
    tgt_path = tmp_path / "tgt.csv"
    src.to_csv(src_path, index=False)
    tgt.to_csv(tgt_path, index=False)
    return {"src": str(src_path), "tgt": str(tgt_path)}


@pytest.fixture
def toy_cfg(toy_csvs):
    return {
        "run_name": "pytest",
        "seed": 123,
        "logging": {"verbose": 0, "save_plots": False},
        "metrics": {"ape_epsilon": 1e-8, "mc_samples_test": 4},
        "data": {
            "mode": "split",
            "splits": {"source": [0.7, 0.15, 0.15], "target": [0.7, 0.15, 0.15]},
            "noise_injection": {"enabled": False, "source_factor": 0.0, "target_factor": 0.0},
            "scaling": {"standardize_X": True, "standardize_y": True},
            "source": {
                "single_csv": toy_csvs["src"],
                "features": ["f1", "f2", "f3"],
                "outputs": ["y"],
            },
            "target": {
                "single_csv": toy_csvs["tgt"],
                "features": ["f2", "f1_tgt", "f3"],
                "outputs": ["y"],
            },
        },
        "stage1": {
            "mode": "train",
            "epochs": 2,
            "batch_size": 16,
            "optimizer": {"lr": 1e-3, "decay_rate": 0.5, "decay_every_epochs": 1},
            "early_stopping": {"enabled": False},
            "architecture": {
                "feature_extractor": {"layers": [8, 4], "activations": ["relu", "relu"]},
                "reg_head": {"out_units": 1, "activation": None},
            },
        },
        "stage2": {
            "enabled": True,
            "epochs": 2,
            "batch_size": 16,
            "optimizer": {"lr": 1e-3, "decay_rate": 0.5, "decay_every_epochs": 1},
            "lambda_max": 1.0,
            "lambda_min_frac": 0.0,
            "ramp_k": 5.0,
            "warmup_epochs": 0,
            "early_stopping": {"enabled": False},
            "domain_head": {"widths": [8], "dropout": 0.0, "l2": 0.0},
        },
        "stage3": {
            "epochs": 2,
            "batch_size": 16,
            "optimizer": {"lr": 1e-3, "decay_rate": 0.5, "decay_every_epochs": 1},
            "fully_deterministic": True,
            "kl_weight_max": 1.0,
            "early_stopping": {"enabled": False},
            "bayesian_policy": {"by_index": [], "must_be_downstream": True},
            "architecture": {
                "feature_extractor": {"layers": [8, 4], "activations": ["relu", "relu"]},
                "reg_head": {"out_units": 1, "activation": None},
            },
            "freeze": {"unfreeze_from_layer_idx": -2},
        },
        "outfiles": {
            "save_hdf5_model": True,
            "save_metadata_h5": True
        }
    }