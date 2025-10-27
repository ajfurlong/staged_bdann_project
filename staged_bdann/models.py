"""Neural network model components for staged B-DANN.

This module defines deterministic feature extractors and heads for Stage 1 and Stage 2, a gradient
reversal layer for DANN, and hybrid Bayesian layers for Stage 3.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import tensorflow as tf
from tensorflow.keras import Model, Input, layers
import tensorflow_probability as tfp
from tensorflow_probability import layers as tfpl


__all__ = [
    "build_feature_extractor_det",
    "build_reg_head_det",
    "GradientReversal",
    "build_domain_head",
    "bayesian_dense",
    "build_stage3_hybrid_from_config",
]


# ---------------------------
# Deterministic building blocks
# ---------------------------

def build_feature_extractor_det(
    input_shape: Sequence[int],
    layers_list: Sequence[int],
    activations: Sequence[Optional[str]],
) -> Model:
    """Construct a deterministic feedforward feature extractor.

    The final element of ``layers_list`` is treated as the feature dimension. Hidden widths are in order.

    Args:
        input_shape: Input feature shape excluding batch, typically (F,).
        layers_list: List of positive integers for hidden widths; last value is the feature_dim.
        activations: List of Keras activation names, one per layer in layers_list. Use None for linear.

    Returns:
        Keras ``Model`` mapping input features to learned feature representation of size ``layers_list[-1]``.
    """
    inputs = Input(shape=tuple(input_shape))
    x = inputs
    if len(layers_list) < 1:
        raise ValueError("layers_list must be non-empty")
    acts = list(activations)
    for i, width in enumerate(layers_list[:-1]):
        x = layers.Dense(int(width), activation=acts[i], name=f"fe_dense_{i}")(x)
    features = layers.Dense(int(layers_list[-1]), activation=acts[-1], name="feature_layer")(x)
    return Model(inputs, features, name="feature_extractor")


def build_reg_head_det(
    feature_dim: int,
    out_units: int = 1,
    activation: Optional[str] = None,
) -> Model:
    """Construct a deterministic regression head that follows a feature extractor.

    Args:
        feature_dim: Dimensionality of the feature extractor output.
        out_units: Number of output units. For scalar regression use 1.
        activation: Optional activation for the output layer. None yields linear outputs.

    Returns:
        Keras ``Model`` mapping a feature vector of shape (feature_dim,) to predictions of shape (out_units,).
    """
    fin = Input(shape=(int(feature_dim),))
    out = layers.Dense(int(out_units), activation=activation, name="regressor_head")(fin)
    return Model(fin, out, name="regressor_head")


# ---------------------------
# Gradient Reversal Layer (for DANN)
# ---------------------------
class GradientReversal(layers.Layer):
    """Gradient Reversal Layer with externally provided lambda.

    The forward pass is identity. The backward pass multiplies the incoming gradient by ``-lambda_var``.

    Args:
        lambda_var: Either a Python float, a scalar ``tf.Tensor``, or a scalar ``tf.Variable``.
    """

    def __init__(self, lambda_var: tf.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.lambda_var = lambda_var

    def call(self, x: tf.Tensor) -> tf.Tensor:
        @tf.custom_gradient
        def _grl(x_inner: tf.Tensor):
            def grad(dy: tf.Tensor) -> tf.Tensor:
                return -tf.cast(self.lambda_var, dy.dtype) * dy

            return x_inner, grad

        return _grl(x)


# ---------------------------
# Domain classifier head
# ---------------------------

def build_domain_head(
    grl_out: tf.Tensor,
    widths: Sequence[int],
    dropout: float,
    l2: float,
) -> tf.Tensor:
    """Build the domain classifier branch used in Stage 2.

    Args:
        grl_out: Input tensor from the Gradient Reversal Layer.
        widths: Hidden layer widths for the domain classifier, non-empty list of positive integers.
        dropout: Dropout rate in [0, 1]. Applied once at the head input.
        l2: L2 regularization factor for Dense layers.

    Returns:
        A tensor representing the sigmoid domain probability output with shape (None, 1).
    """
    x = layers.Dropout(float(dropout), name="dom_dropout")(grl_out)
    reg = tf.keras.regularizers.l2(float(l2))
    if len(widths) < 1:
        raise ValueError("widths must be a non-empty sequence of positive integers")
    for i, w in enumerate(widths):
        x = layers.Dense(int(w), activation="relu", kernel_regularizer=reg, name=f"dom_dense_{i}")(x)
    dom_out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg, name="domain_output")(x)
    return dom_out


# ---------------------------
# Bayesian helpers for Stage 3
# ---------------------------

def bayesian_dense(units: int, activation: Optional[str], kl_weight: tf.Tensor, name: Optional[str] = None) -> tf.keras.layers.Layer:
    """Return a DenseFlipout layer with stable initializers and a scalar KL term.

    The KL function reduces elementwise KL to a scalar, then multiplies by ``kl_weight`` to allow annealing.

    Args:
        units: Output width of the layer.
        activation: Activation name, or None for linear.
        kl_weight: Scalar tensor or Python float controlling the KL strength. Can be annealed externally.
        name: Optional name for the layer.

    Returns:
        A configured ``tfpl.DenseFlipout`` layer.
    """

    def _safe_scale_initializer(shape, dtype=None):  # noqa: ANN001
        # Initialize so softplus(stddev_init) is small and stable around ~0.05.
        return tf.random.normal(shape, mean=-3.0, stddev=0.1, dtype=dtype)

    def _kl_fn(q, p, _):  # noqa: ANN001
        kl_elem = tfp.distributions.kl_divergence(q, p)
        kl_sum = tf.reduce_sum(kl_elem)
        return kl_sum * tf.cast(kl_weight, kl_sum.dtype)

    return tfpl.DenseFlipout(
        int(units),
        activation=activation,
        kernel_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        kernel_posterior_fn=tfpl.util.default_mean_field_normal_fn(
            loc_initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            untransformed_scale_initializer=_safe_scale_initializer,
        ),
        bias_posterior_fn=tfpl.util.default_mean_field_normal_fn(
            loc_initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            untransformed_scale_initializer=_safe_scale_initializer,
        ),
        kernel_divergence_fn=_kl_fn,
        bias_divergence_fn=_kl_fn,
        name=name,
    )


def build_stage3_hybrid_from_config(
    input_shape: Sequence[int],
    s3_arch: dict,
    bayesian_indices: Iterable[int],
    kl_weight: tf.Tensor,
) -> Tuple[Model, Model, bool]:
    """Construct a Stage 3 extractor and head with selective Bayesian layers.

    A single sequential stack is built. For each index ``i`` in the feature extractor stack, if
    ``i`` is in ``bayesian_indices``, a DenseFlipout layer is used, otherwise a deterministic Dense.

    The regression head is deterministic by default. If ``s3_arch['reg_head']['out_units'] == 2``, a
    Bayesian head is built with two units representing (loc, pre-softplus scale) and the function
    returns ``is_head_bayesian=True``.

    Args:
        input_shape: Feature shape excluding batch.
        s3_arch: Architecture dictionary with keys ``feature_extractor`` and ``reg_head``.
        bayesian_indices: Iterable of zero-based layer indices that should be Bayesian.
        kl_weight: Scalar tensor or Python float for KL weighting, typically annealed during training.

    Returns:
        Tuple of (feature_extractor_model, regression_head_model, is_head_bayesian).
    """
    fe_cfg = s3_arch["feature_extractor"]
    layers_list: List[int] = list(map(int, fe_cfg["layers"]))
    if "activations" not in fe_cfg:
        raise ValueError("Stage 3 feature_extractor.activations must be provided as a list matching 'layers'.")
    acts_per_layer = list(fe_cfg["activations"])  # list, one per layer
    if len(acts_per_layer) != len(layers_list):
        raise ValueError(
            "Stage 3 feature_extractor.activations length must equal feature_extractor.layers length"
        )

    if len(layers_list) < 1:
        raise ValueError("Stage 3 feature_extractor.layers must be non-empty")

    inputs = Input(shape=tuple(input_shape))
    x = inputs
    bset = set(int(i) for i in bayesian_indices)
    for i, width in enumerate(layers_list):
        name = f"fe_bayes_{i}" if i in bset else f"fe_dense_{i}"
        act_i = acts_per_layer[i]
        if i in bset:
            x = bayesian_dense(int(width), act_i, kl_weight, name=name)(x)
        else:
            x = layers.Dense(int(width), activation=act_i, name=name)(x)

    features = x
    feature_dim = int(layers_list[-1])

    head_out_units = int(s3_arch["reg_head"]["out_units"])
    head_activation = s3_arch["reg_head"].get("activation")

    fin = Input(shape=(feature_dim,))
    if head_out_units == 2:
        head = bayesian_dense(2, head_activation, kl_weight, name="reg_bnn_out")(fin)
        reg_head = Model(fin, head, name="regressor_head_bayesian")
        is_head_bayesian = True
    else:
        head = layers.Dense(head_out_units, activation=head_activation, name="regressor_head")(fin)
        reg_head = Model(fin, head, name="regressor_head")
        is_head_bayesian = False

    extractor = Model(inputs, features, name="feature_extractor_stage3")
    return extractor, reg_head, is_head_bayesian