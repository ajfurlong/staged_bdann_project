"""Configuration loading and validation for staged_bdann.

This module provides a single public function, ``load_and_validate_config``, which loads a JSON file
and returns a fully resolved configuration dictionary with defaults applied and rigorous validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

# Allowable Keras activation names for simple string-based configuration.
_allowed_keras_activations = {
    "relu",
    "tanh",
    "sigmoid",
    "linear",
    "softplus",
    "softsign",
    "selu",
    "elu",
    "exponential",
    "hard_sigmoid",
    "swish",
    "silu",
    "gelu",
    "softmax",
}


def _err(msgs: List[str], msg: str) -> None:
    """Append a formatted error message to the collector."""
    msgs.append(msg)

def _warn(msgs: List[str], msg: str) -> None:
    """Append a formatted warning message to the collector."""
    msgs.append(msg)


def _require_keys(d: Dict[str, Any], keys: List[str], prefix: str, errs: List[str]) -> None:
    """Require that all ``keys`` exist in dict ``d``; otherwise, record an error with ``prefix``."""
    for k in keys:
        if k not in d:
            _err(errs, f"Missing {prefix}{k}")


def _validate_bool(d: Dict[str, Any], key: str, prefix: str, errs: List[str]) -> None:
    if key in d and not isinstance(d[key], bool):
        _err(errs, f"{prefix}{key} must be a boolean")


def _validate_nonneg_number(d: Dict[str, Any], key: str, prefix: str, errs: List[str]) -> None:
    if key in d and not isinstance(d[key], (int, float)):
        _err(errs, f"{prefix}{key} must be a number")
    elif key in d and d[key] < 0:
        _err(errs, f"{prefix}{key} must be non-negative")


def _validate_prob(d: Dict[str, Any], key: str, prefix: str, errs: List[str]) -> None:
    if key in d:
        v = d[key]
        if not isinstance(v, (int, float)):
            _err(errs, f"{prefix}{key} must be a number in [0, 1]")
        elif not (0 <= float(v) <= 1):
            _err(errs, f"{prefix}{key} must be in [0, 1]")


def _validate_positive_int(d: Dict[str, Any], key: str, prefix: str, errs: List[str]) -> None:
    if key in d and (not isinstance(d[key], int) or d[key] <= 0):
        _err(errs, f"{prefix}{key} must be a positive integer")


def _validate_optimizer(block: Dict[str, Any], prefix: str, errs: List[str]) -> None:
    lr = block.get("lr", None)
    if lr is None:
        _err(errs, f"{prefix}lr is required")
    elif not isinstance(lr, (int, float)) or lr <= 0:
        _err(errs, f"{prefix}lr must be a positive number")

    de = block.get("decay_every_epochs", None)
    dr = block.get("decay_rate", None)
    if de is not None and (not isinstance(de, int) or de <= 0):
        _err(errs, f"{prefix}decay_every_epochs must be a positive integer")
    if dr is not None and (not isinstance(dr, (int, float)) or not (0 < float(dr) <= 1)):
        _err(errs, f"{prefix}decay_rate must be in (0, 1]")


def _validate_activations_list(acts: Any, n_layers: int, prefix: str, errs: List[str]) -> None:
    """Validate that `acts` is a list of length `n_layers` with allowed activation names or None.

    Args:
        acts: User-provided activations value.
        n_layers: Number of feature-extractor layers.
        prefix: Error prefix, e.g., 'stage1.architecture.feature_extractor.'.
        errs: Error collector.
    """
    if not isinstance(acts, list):
        _err(errs, f"{prefix}activations must be a list with length equal to feature_extractor.layers")
        return
    if len(acts) != n_layers:
        _err(errs, f"{prefix}activations must have length {n_layers}; got {len(acts)}")
        return
    for i, a in enumerate(acts):
        if a is None:
            continue
        if not isinstance(a, str):
            _err(errs, f"{prefix}activations[{i}] must be a string activation name or null")
            continue
        if a not in _allowed_keras_activations:
            _err(errs, f"{prefix}activations[{i}]='{a}' is not a supported Keras activation")


def _validate_architecture(arch: Dict[str, Any], prefix: str, errs: List[str]) -> None:
    fe = arch.get("feature_extractor", {})
    fe_layers = fe.get("layers", None)
    if not isinstance(fe_layers, list) or len(fe_layers) < 1 or not all(isinstance(x, int) and x > 0 for x in fe_layers):
        _err(errs, f"{prefix}feature_extractor.layers must be a non-empty list of positive integers")
    else:
        # Enforce per-layer activations list
        if "activation" in fe and fe.get("activation") is not None:
            _err(errs, f"{prefix}feature_extractor.activation is no longer supported; use 'activations' list")
        if "activations" not in fe:
            _err(errs, f"{prefix}feature_extractor.activations is required and must match the number of layers")
        else:
            _validate_activations_list(fe["activations"], len(fe_layers), f"{prefix}feature_extractor.", errs)

    rh = arch.get("reg_head", {})
    out_units = rh.get("out_units", None)
    if out_units is None or not isinstance(out_units, int) or out_units <= 0:
        _err(errs, f"{prefix}reg_head.out_units must be a positive integer")
    rh_activation = rh.get("activation", None)
    if rh_activation is not None:
        if not isinstance(rh_activation, str):
            _err(errs, f"{prefix}reg_head.activation must be a string or null")
        elif rh_activation not in _allowed_keras_activations:
            _err(errs, f"{prefix}reg_head.activation='{rh_activation}' is not a supported Keras activation")


def _validate_domain_head(dh: Dict[str, Any], prefix: str, errs: List[str]) -> None:
    widths = dh.get("widths", None)
    if not isinstance(widths, list) or len(widths) < 1 or not all(isinstance(x, int) and x > 0 for x in widths):
        _err(errs, f"{prefix}widths must be a non-empty list of positive integers")
    _validate_prob(dh, "dropout", prefix, errs)
    _validate_nonneg_number(dh, "l2", prefix, errs)


def _validate_bayesian_policy(bp: Dict[str, Any], prefix: str, errs: List[str]) -> None:
    by_idx = bp.get("by_index", None)
    if not isinstance(by_idx, list) or not all(isinstance(i, int) for i in by_idx):
        _err(errs, f"{prefix}by_index must be a list of integers")
    _validate_bool(bp, "must_be_downstream", prefix, errs)


def _validate_splits(vec: List[float], prefix: str, errs: List[str]) -> None:
    if not isinstance(vec, list) or len(vec) != 3 or not all(isinstance(x, (int, float)) for x in vec):
        _err(errs, f"{prefix} must be a list of three numbers summing to 1.0")
        return
    s = float(vec[0]) + float(vec[1]) + float(vec[2])
    if abs(s - 1.0) > 1e-6:
        _err(errs, f"{prefix} must sum to 1.0; got {s}")
    if any(x < 0 for x in vec):
        _err(errs, f"{prefix} entries must be non-negative")


def _resolve_pathlike(path: Any) -> Any:
    """Convert simple pathlike strings to strings, leaving None and other types untouched."""
    if isinstance(path, (str, Path)):
        return str(path)
    return path


def load_and_validate_config(path: str) -> dict:
    """Load a JSON configuration file and validate its contents.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        Validated configuration dictionary with defaults applied.

    Raises:
        ValueError: If required keys are missing or validation fails.
    """
    with open(path, "r") as f:
        cfg: Dict[str, Any] = json.load(f)

    errs: List[str] = []
    warns: List[str] = []

    # Top-level basics
    if "run_name" not in cfg:
        _err(errs, "Missing top-level key: run_name")
    if "seed" not in cfg:
        cfg["seed"] = 1234
    elif not isinstance(cfg["seed"], int):
        _err(errs, "seed must be an integer")

    # Logging
    cfg.setdefault("logging", {})
    log = cfg["logging"]
    log.setdefault("save_summaries", True)
    log.setdefault("save_plots", True)
    log.setdefault("verbose", 2)
    if not isinstance(log.get("verbose"), int) or log["verbose"] < 0:
        _err(errs, "logging.verbose must be a non-negative integer")
    _validate_bool(log, "save_summaries", "logging.", errs)
    _validate_bool(log, "save_plots", "logging.", errs)

    # Data block
    if "data" not in cfg:
        _err(errs, "Missing top-level key: data")
        data = {}
    else:
        data = cfg["data"]
    data.setdefault("mode", "split")  # "files" or "split"
    if data["mode"] not in ("files", "split"):
        _err(errs, "data.mode must be 'files' or 'split'")

    # Source and target blocks
    for dom in ("source", "target"):
        if dom not in data:
            _err(errs, f"Missing data.{dom} block")
            data[dom] = {}
        bd = data[dom]
        # Path-like entries
        for k in ("train_csv", "valid_csv", "test_csv", "single_csv"):
            if k in bd:
                bd[k] = _resolve_pathlike(bd[k])
            else:
                bd.setdefault(k, None)
        # Features
        if "features" not in bd or not isinstance(bd["features"], list) or len(bd["features"]) < 1:
            _err(errs, f"data.{dom}.features must be a non-empty list of feature names")
        else:
            if not all(isinstance(f, str) for f in bd["features"]):
                _err(errs, f"All items in data.{dom}.features must be strings")
        # Outputs: accept a single string or a list of strings; normalize to list[str]
        if "outputs" not in bd:
            _err(errs, f"data.{dom}.outputs is required")
        else:
            if isinstance(bd["outputs"], str):
                bd["outputs"] = [bd["outputs"]]
            elif isinstance(bd["outputs"], list):
                pass
            else:
                _err(errs, f"data.{dom}.outputs must be a string or a list of strings")

            if isinstance(bd.get("outputs"), list):
                if len(bd["outputs"]) < 1:
                    _err(errs, f"data.{dom}.outputs must be non-empty")
                elif not all(isinstance(f, str) for f in bd["outputs"]):
                    _err(errs, f"All items in data.{dom}.outputs must be strings")

    # Feature/output set consistency: allow mismatch but warn, since downstream may remap/order features.
    s_feats = data.get("source", {}).get("features", [])
    t_feats = data.get("target", {}).get("features", [])

    s_outputs = data.get("source", {}).get("outputs", [])
    t_outputs = data.get("target", {}).get("outputs", [])
    if isinstance(s_feats, list) and isinstance(t_feats, list) and s_feats and t_feats:
        if s_feats != t_feats:
            _warn(
                warns,
                (
                    "data.source.features and data.target.features differ. Proceeding, but you must ensure consistent "
                    "feature mapping and ordering downstream."
                ),
            )
    if isinstance(s_outputs, list) and isinstance(t_outputs, list) and s_outputs and t_outputs:
        if s_outputs != t_outputs:
            _warn(
                warns,
                (
                    "data.source.outputs and data.target.outputs differ. Proceeding, but you must ensure consistent "
                    "output mapping and ordering downstream."
                ),
            )

    # Splits
    data.setdefault("splits", {})
    data["splits"].setdefault("source", [0.8, 0.1, 0.1])
    data["splits"].setdefault("target", [0.25, 0.50, 0.25])
    _validate_splits(data["splits"].get("source", []), "data.splits.source", errs)
    _validate_splits(data["splits"].get("target", []), "data.splits.target", errs)

    # Scaling
    data.setdefault("scaling", {"standardize_X": True, "standardize_y": True})
    _validate_bool(data["scaling"], "standardize_X", "data.scaling.", errs)
    _validate_bool(data["scaling"], "standardize_y", "data.scaling.", errs)
    if "standardize_X" not in data["scaling"]:
        data["scaling"]["standardize_X"] = True
    if "standardize_y" not in data["scaling"]:
        data["scaling"]["standardize_y"] = True

    # Noise injection
    data.setdefault("noise_injection", {"enabled": False, "source_factor": 1e-4, "target_factor": 5e-3})
    ni = data["noise_injection"]
    _validate_bool(ni, "enabled", "data.noise_injection.", errs)
    _validate_nonneg_number(ni, "source_factor", "data.noise_injection.", errs)
    _validate_nonneg_number(ni, "target_factor", "data.noise_injection.", errs)

    # Mode-specific file requirements
    if data["mode"] == "files":
        # Either explicit train/valid/test, or a single_csv plus fixed_test_size (for target) is allowed.
        for dom in ("source", "target"):
            bd = data[dom]
            has_triad = all(bd.get(k) for k in ("train_csv", "valid_csv", "test_csv"))
            has_single = bool(bd.get("single_csv"))
            if not has_triad and not has_single:
                _err(errs, f"data.{dom} requires either train/valid/test CSVs or single_csv")

    # Stage 1
    if "stage1" not in cfg:
        _err(errs, "Missing top-level key: stage1")
        cfg["stage1"] = {}
    s1 = cfg["stage1"]
    s1.setdefault("mode", "train")
    if s1["mode"] not in ("train", "load"):
        _err(errs, "stage1.mode must be 'train' or 'load'")
    s1.setdefault("extractor_weights", None)
    s1.setdefault("reg_head_weights", None)
    s1.setdefault("optimizer", {"lr": 1e-3, "decay_every_epochs": 10, "decay_rate": 0.96})
    s1.setdefault("epochs", 150)
    s1.setdefault("batch_size", 16)
    _s1_default_layers = [16, 16, 16, 16, 17]
    s1.setdefault(
        "architecture",
        {
            "feature_extractor": {
                "layers": _s1_default_layers,
                "activations": ["relu"] * len(_s1_default_layers),
            },
            "reg_head": {"out_units": 1, "activation": None},
        },
    )
    s1.setdefault(
        "early_stopping",
        {"enabled": True, "patience": 50, "min_delta": 0.0, "monitor": "val_loss", "mode": "min", "restore_best_weights": True},
    )
    _validate_positive_int(s1, "epochs", "stage1.", errs)
    _validate_positive_int(s1, "batch_size", "stage1.", errs)
    _validate_optimizer(s1.get("optimizer", {}), "stage1.optimizer.", errs)
    _validate_architecture(s1.get("architecture", {}), "stage1.architecture.", errs)
    if s1.get("mode") == "load":
        if not s1.get("extractor_weights") or not s1.get("reg_head_weights"):
            _err(errs, "stage1.mode='load' requires extractor_weights and reg_head_weights paths")

    # Stage 2
    if "stage2" not in cfg:
        _err(errs, "Missing top-level key: stage2")
        cfg["stage2"] = {}
    s2 = cfg["stage2"]
    s2.setdefault("enabled", True)
    s2.setdefault("optimizer", {"lr": 3e-7, "decay_every_epochs": 5, "decay_rate": 0.96})
    s2.setdefault("epochs", 50)
    s2.setdefault("batch_size", 16)
    s2.setdefault("lambda_max", 0.5)
    s2.setdefault("lambda_min_frac", 0.0)
    s2.setdefault("ramp_k", 10.0)
    s2.setdefault("warmup_epochs", 3)
    s2.setdefault("domain_head", {"dropout": 0.23, "widths": [120, 11, 30, 65], "l2": 1.84e-4})
    s2.setdefault(
        "early_stopping",
        {"enabled": False, "patience": 10, "min_delta": 0.0, "monitor": "dann_alignment", "mode": "min", "restore_best_weights": True},
    )
    _validate_bool(s2, "enabled", "stage2.", errs)
    _validate_positive_int(s2, "epochs", "stage2.", errs)
    _validate_positive_int(s2, "batch_size", "stage2.", errs)
    _validate_optimizer(s2.get("optimizer", {}), "stage2.optimizer.", errs)
    _validate_nonneg_number(s2, "lambda_max", "stage2.", errs)
    _validate_prob(s2, "lambda_min_frac", "stage2.", errs)
    _validate_nonneg_number(s2, "ramp_k", "stage2.", errs)
    if not isinstance(s2.get("warmup_epochs"), int) or s2["warmup_epochs"] < 0:
        _err(errs, "stage2.warmup_epochs must be a non-negative integer")
    _validate_domain_head(s2.get("domain_head", {}), "stage2.domain_head.", errs)

    # Stage 3
    if "stage3" not in cfg:
        _err(errs, "Missing top-level key: stage3")
        cfg["stage3"] = {}
    s3 = cfg["stage3"]
    s3.setdefault("optimizer", {"lr": 2.3e-3, "decay_every_epochs": 10, "decay_rate": 0.96})
    s3.setdefault("epochs", 400)
    s3.setdefault("batch_size", 16)
    s3.setdefault("kl_weight_max", 0.11)
    s3.setdefault("freeze", {"unfreeze_from_layer_idx": -7})
    s3.setdefault("bayesian_policy", {"by_index": [-2, -1], "must_be_downstream": True})
    s3.setdefault("architecture", cfg.get("stage1", {}).get("architecture", {}))
    s3.setdefault(
        "early_stopping",
        {"enabled": False, "patience": 200, "min_delta": 0.0, "monitor": "val_loss", "mode": "min", "restore_best_weights": True},
    )
    s3.setdefault("fully_deterministic", False)
    _validate_positive_int(s3, "epochs", "stage3.", errs)
    _validate_positive_int(s3, "batch_size", "stage3.", errs)
    _validate_nonneg_number(s3, "kl_weight_max", "stage3.", errs)
    _validate_bool(s3, "fully_deterministic", "stage3.", errs)
    _validate_optimizer(s3.get("optimizer", {}), "stage3.optimizer.", errs)
    _validate_architecture(s3.get("architecture", {}), "stage3.architecture.", errs)
    # Enforce per-layer activations for Stage 3 as well
    if "feature_extractor" in s3.get("architecture", {}):
        fe3 = s3["architecture"]["feature_extractor"]
        if "activation" in fe3 and fe3.get("activation") is not None:
            _err(errs, "stage3.architecture.feature_extractor.activation is no longer supported; use 'activations' list")
    if "freeze" in s3 and not isinstance(s3["freeze"].get("unfreeze_from_layer_idx", -1), int):
        _err(errs, "stage3.freeze.unfreeze_from_layer_idx must be an integer")
    _validate_bayesian_policy(s3.get("bayesian_policy", {}), "stage3.bayesian_policy.", errs)

    # Metrics and outfiles
    cfg.setdefault("metrics", {"ape_epsilon": 1e-8, "mc_samples_test": 200})
    met = cfg["metrics"]
    if not isinstance(met.get("ape_epsilon"), (float, int)) or met["ape_epsilon"] <= 0:
        _err(errs, "metrics.ape_epsilon must be a positive number")
    if not isinstance(met.get("mc_samples_test"), int) or met["mc_samples_test"] <= 0:
        _err(errs, "metrics.mc_samples_test must be a positive integer")

    cfg.setdefault("outfiles", {"save_hdf5_model": True, "save_metadata_h5": True})
    out = cfg["outfiles"]
    _validate_bool(out, "save_hdf5_model", "outfiles.", errs)
    _validate_bool(out, "save_metadata_h5", "outfiles.", errs)
    if "save_hdf5_model" not in out:
        out["save_hdf5_model"] = True
    if "save_metadata_h5" not in out:
        out["save_metadata_h5"] = True

    if warns:
        print("Config warning(s):")
        for w in warns:
            print(f"  - {w}")

    if errs:
        msg = "Invalid configuration:\n  - " + "\n  - ".join(errs)
        raise ValueError(msg)

    return cfg