import pytest
from pathlib import Path
from staged_bdann.data import load_data
from staged_bdann.pipeline import run_member_once


def test_stage3_bayesian_smoke(tmp_run_dir, toy_cfg):
    toy_cfg["stage3"]["fully_deterministic"] = False
    toy_cfg["stage3"]["architecture"]["reg_head"]["out_units"] = 2
    toy_cfg["stage3"]["bayesian_policy"]["by_index"] = [-1,-2]
    fixed = load_data(toy_cfg["data"], Path(tmp_run_dir), seed=toy_cfg["seed"])
    _, _, _, _, _, metrics = run_member_once(toy_cfg, tmp_run_dir, seed=toy_cfg["seed"], fixed_data=fixed)
    assert any("UQ" in k for k in metrics)