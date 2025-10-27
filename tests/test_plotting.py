import numpy as np
from staged_bdann.plotting import _save_parity_plot, _save_hist_kde


def test_parity_and_histogram(tmp_path):
    y = np.linspace(-1, 1, 50)
    yp = y + 0.1 * np.sin(5 * y)
    _save_parity_plot(y, yp, tmp_path / "p.png", "Parity")
    _save_hist_kde(yp - y, tmp_path / "h.png", "Err", "Err")
    assert (tmp_path / "p.png").exists()
    assert (tmp_path / "h.png").exists()