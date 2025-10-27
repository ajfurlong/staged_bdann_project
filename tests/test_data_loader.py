import numpy as np
from pathlib import Path
from staged_bdann.data import load_data


def test_load_data_shapes(tmp_path, toy_cfg):
    d = load_data(toy_cfg["data"], Path(tmp_path), seed=toy_cfg["seed"])
    for k in ["Xs_tr", "Xt_tr", "ys_tr", "yt_tr"]:
        assert isinstance(d[k], np.ndarray)
    assert set(np.unique(d["ds_tr"])) == {0.0}
    assert set(np.unique(d["dt_tr"])) == {1.0}