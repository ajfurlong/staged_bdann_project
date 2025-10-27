"""Stage 1 training or loading of deterministic feature extractor and regression head.

Implements deterministic source-domain training used as initialization for Stage 2 (domain
alignment) and Stage 3 (Bayesian fine-tuning).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import Model

from ..models import build_feature_extractor_det, build_reg_head_det
from ..plotting import _save_loss_curve, compute_metrics
from ..utils import save_model_summary
from .callbacks import _make_early_stopping


def stage1_train_or_load(
    run_dir: Path,
    cfg,
    input_shape,
    Xs_tr,
    ys_tr,
    Xs_v,
    ys_v,
    Xs_te,
    ys_te,
    sizes_source,
    sizes_target,
    y_scaler_src,
):
    """Train or load Stage-1 deterministic models.

    Args:
        run_dir: Output directory for this run.
        cfg: Configuration dictionary.
        input_shape: Tuple of input feature dimension.
        Xs_tr, ys_tr, Xs_v, ys_v, Xs_te, ys_te: Source splits.
        sizes_source, sizes_target: Tuple of dataset sizes.
        y_scaler_src: Fitted StandardScaler for source target.

    Returns:
        feature_extractor: Trained Keras Model for feature extraction.
        reg_head: Trained Keras Model for regression head.
        history: Keras History object or None if in load mode.
    """
    arch = cfg["stage1"]["architecture"]
    feat_layers = arch["feature_extractor"]["layers"]
    feat_act = arch["feature_extractor"]["activations"]
    feature_dim = int(feat_layers[-1])

    # Sanity check: regression head output size must match target dimension
    y_shape = np.asarray(ys_tr).shape
    target_dim = int(y_shape[1]) if len(y_shape) > 1 else 1

    reg_out_units = int(arch["reg_head"]["out_units"])
    reg_act = arch["reg_head"]["activation"]

    if int(reg_out_units) != target_dim:
        raise ValueError(
            (
                f"Stage-1 config/reg_head.out_units={reg_out_units} does not match target dimension "
                f"derived from ys_tr ({target_dim})."
            )
        )

    feature_extractor = build_feature_extractor_det(input_shape, feat_layers, feat_act)
    reg_head = build_reg_head_det(feature_dim, reg_out_units, reg_act)

    save_model_summary(feature_extractor, run_dir / "arch" / "stage1_feature_extractor.txt")
    save_model_summary(reg_head, run_dir / "arch" / "stage1_reg_head.txt")

    mode = cfg["stage1"].get("mode", "train").lower()

    if mode == "load":
        fx_path = Path(cfg["stage1"].get("extractor_weights", ""))
        rh_path = Path(cfg["stage1"].get("reg_head_weights", ""))
        if (not fx_path.is_file()) or (not rh_path.is_file()):
            raise FileNotFoundError("Stage-1 load mode requires valid extractor and reg head weight paths.")

        # Quick structural compatibility check before loading weights
        try:
            feature_extractor.load_weights(fx_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load Stage-1 extractor weights from '{fx_path}'. Check user-entered architecture/loaded file: {e}"
            ) from e

        try:
            reg_head.load_weights(rh_path)
        except Exception as e:
            raise ValueError(
                f"Failed to load Stage-1 regression head weights from '{rh_path}'. Check user-entered architecture/loaded file: {e}"
            ) from e

        return feature_extractor, reg_head, None

    # Training mode
    opt_cfg = cfg["stage1"]["optimizer"]
    steps_per_epoch = int(np.ceil(len(Xs_tr) / int(cfg["stage1"]["batch_size"])))
    decay_steps = max(1, int(opt_cfg.get("decay_every_epochs", 1))) * steps_per_epoch

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=float(opt_cfg["lr"]),
        decay_steps=decay_steps,
        decay_rate=float(opt_cfg["decay_rate"]),
        staircase=True,
    )

    model = Model(inputs=feature_extractor.input, outputs=reg_head(feature_extractor.output), name="stage1_model")
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule),
        loss="mse",
    )

    es_cb = _make_early_stopping(cfg["stage1"].get("early_stopping", {}))
    callbacks = [es_cb] if es_cb is not None else []

    history = model.fit(
        Xs_tr,
        ys_tr,
        validation_data=(Xs_v, ys_v),
        epochs=int(cfg["stage1"]["epochs"]),
        batch_size=int(cfg["stage1"]["batch_size"]),
        verbose=int(cfg["logging"]["verbose"]),
        callbacks=callbacks,
    )

    _save_loss_curve(
        history,
        run_dir / "plots" / "stage1_loss.png",
        "Stage-1 MSE",
        yscale="log",
        keys=["loss", "val_loss"],
    )

    # Evaluate on source test set
    y_pred_s_scaled = model.predict(Xs_te, verbose=0)
    y_pred_s = y_scaler_src.inverse_transform(y_pred_s_scaled)
    y_true_s = y_scaler_src.inverse_transform(ys_te)

    metrics = compute_metrics(
        y_true_s.flatten(),
        y_pred_s.flatten(),
        None,
        sizes_source,
        sizes_target,
        stage1_time=len(history.history["loss"]),
        inference_time=0.0,
        ape_epsilon=float(cfg["metrics"].get("ape_epsilon", 1e-8)),
    )

    dfm = pd.DataFrame(metrics, index=["Values"]).T
    with open(run_dir / "logs" / "stage1_test_results.txt", "w", encoding="utf-8") as f:
        f.write(dfm.to_string(float_format="%.6f"))

    # Save weights
    model.save_weights(run_dir / "weights" / "stage1_full.h5")
    feature_extractor.save_weights(run_dir / "weights" / "stage1_extractor.h5")
    reg_head.save_weights(run_dir / "weights" / "stage1_reg_head.h5")

    return feature_extractor, reg_head, history