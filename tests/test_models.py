import numpy as np
import tensorflow as tf
from staged_bdann.models import build_feature_extractor_det, build_reg_head_det, GradientReversal, build_domain_head


def test_shapes():
    fe = build_feature_extractor_det((3,), [4, 2], ["relu", "relu"])
    rh = build_reg_head_det(2, 1, None)
    x = tf.random.uniform((5, 3))
    f = fe(x)
    y = rh(f)
    assert f.shape[-1] == 2 and y.shape[-1] == 1

def test_grl_gradient_basic():
    u = tf.random.uniform((5, 2))
    lam = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    grl = GradientReversal(lam)

    with tf.GradientTape() as tape:
        tape.watch(u)
        y = grl(u)
        loss = tf.reduce_sum(y)  # dy/dy = 1 everywhere
    g = tape.gradient(loss, u)
    assert g is not None
    np.testing.assert_allclose(g.numpy(), -np.ones_like(u.numpy()), rtol=0, atol=0)

def test_grl_lambda_zero_gives_zero_grad():
    u = tf.random.uniform((3, 2))
    lam = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    grl = GradientReversal(lam)

    with tf.GradientTape() as tape:
        tape.watch(u)
        y = grl(u)
        loss = tf.reduce_sum(y)
    g = tape.gradient(loss, u)
    assert g is not None
    np.testing.assert_allclose(g.numpy(), 0.0, rtol=0, atol=0)

def test_grl_no_grad_wrt_lambda():
    u = tf.random.uniform((2, 2))
    lam = tf.Variable(0.7, trainable=False, dtype=tf.float32)
    grl = GradientReversal(lam)

    with tf.GradientTape() as tape:
        tape.watch(u)
        y = grl(u)
        loss = tf.reduce_sum(y)
    # We do not want gradients w.r.t. lambda_var
    g_lam = tape.gradient(loss, lam)
    assert g_lam is None