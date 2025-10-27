"""Common loss and distribution utilities for staged_bdann.

Provides safe negative log-likelihood and Normal distribution construction for Bayesian heads.
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp


tfd = tfp.distributions


def normal_sp(params: tf.Tensor) -> tfd.Distribution:
    """Construct a Normal distribution with softplus-transformed scale.

    Args:
        params: Tensor of shape (N, 2) where [:, 0] is the mean and [:, 1] is the raw scale.

    Returns:
        A tfp.distributions.Normal object with positive scale via softplus transformation.
    """
    loc = params[:, 0:1]
    scale = 1e-6 + tf.nn.softplus(params[:, 1:2])
    return tfd.Normal(loc=loc, scale=scale)


def stage3_safe_nll(y_true: tf.Tensor, y_pred: tfd.Distribution) -> tf.Tensor:
    """Compute a shape/dtype-safe negative log-likelihood for DistributionLambda outputs.

    Ensures ``y_true`` is float32 and shaped (-1, 1) to match the predicted distribution batch shape.
    This prevents 'Input to reshape has 0 values' graph errors inside ``log_prob``.

    Args:
        y_true: True target tensor.
        y_pred: A TFP distribution output from a DistributionLambda layer.

    Returns:
        Scalar tensor for the mean negative log-likelihood.
    """
    y_true = tf.cast(y_true, y_pred.dtype)
    y_true = tf.reshape(y_true, (-1, 1))
    return -tf.reduce_mean(y_pred.log_prob(y_true))