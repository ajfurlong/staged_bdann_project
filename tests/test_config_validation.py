import json
import pytest
from staged_bdann.config import load_and_validate_config


def test_valid_minimal_config(tmp_path, toy_cfg):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(toy_cfg), encoding="utf-8")
    cfg = load_and_validate_config(str(p))
    assert cfg["data"]["source"]["outputs"] == ["y"]


def test_bad_splits_raises(tmp_path, toy_cfg):
    toy_cfg["data"]["splits"]["source"] = [0.5, 0.5, 0.5]
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(toy_cfg), encoding="utf-8")
    with pytest.raises(ValueError):
        load_and_validate_config(str(p))