import tensorflow as tf
from staged_bdann.training.callbacks import LambdaScheduler, KLWeightScheduler


def test_lambda_scheduler_values():
    lam = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    cb = LambdaScheduler(lam, lambda_max=1.0, total_epochs=4, lambda_min_frac=0.0, ramp_k=5.0, warmup_epochs=1)
    for e in range(4):
        cb.on_epoch_begin(e)
    assert float(lam.numpy()) > 0.0


def test_kl_scheduler_scales_by_n():
    w = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    cb = KLWeightScheduler(w, kl_max=1.0, total_epochs=10, num_samples=100)
    cb.on_epoch_begin(9)
    val = float(w.numpy())
    assert 0.0 < val < 0.02