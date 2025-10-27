from pathlib import Path
from staged_bdann.pipeline import run_member_once
from staged_bdann.data import load_data


def test_stage1_stage2_stage3_smoke(tmp_run_dir, toy_cfg):
    fixed = load_data(toy_cfg["data"], Path(tmp_run_dir), seed=toy_cfg["seed"])
    _, _, _, _, _, metrics = run_member_once(toy_cfg, tmp_run_dir, seed=toy_cfg["seed"], fixed_data=fixed)
    assert any("MAE" in k or "RMSE" in k for k in metrics)