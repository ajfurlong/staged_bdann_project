import numpy as np
from staged_bdann.training.batching import BalancedBatchGenerator
from staged_bdann.data import load_data


def test_balanced_batch_generator(toy_cfg, tmp_path):
    d = load_data(toy_cfg["data"], tmp_path, seed=toy_cfg["seed"])
    gen = BalancedBatchGenerator(d["Xs_tr"], d["ys_tr"], d["ds_tr"], d["Xt_tr"], d["dt_tr"], batch_size=16)
    Xb, yb, wb = gen[0]
    assert Xb.shape[0] == 16
    assert np.isclose(wb["regressor_head"].sum(), 8.0)
    assert wb["domain_output"].sum() == 16.0