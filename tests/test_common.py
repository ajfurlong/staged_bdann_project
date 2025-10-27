import numpy as np
import tensorflow as tf
from staged_bdann.training.common import normal_sp, stage3_safe_nll


def test_normal_sp_positive_scale():
    params = tf.constant([[0.0, -5.0], [1.0, 0.0]], dtype=tf.float32)
    d = normal_sp(params)
    s = d.stddev().numpy().reshape(-1)
    assert (s > 0).all()


def test_safe_nll_scalar():
    params = tf.constant([[0.0, 0.0]], dtype=tf.float32)
    d = normal_sp(params)
    y = tf.constant([0.0], dtype=tf.float32)
    nll = stage3_safe_nll(y, d)
    assert nll.shape == ()
    assert np.isfinite(nll.numpy())