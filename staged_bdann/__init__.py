"""staged_bdann
===============

Bayesian Domain-Adversarial Neural Network (B-DANN) training framework.

Provides a modular three-stage pipeline:
    Stage 1 - Deterministic feature extractor pretraining on source data.
    Stage 2 - Domain-adversarial alignment with a Gradient Reversal Layer.
    Stage 3 - Bayesian fine-tuning on target data with uncertainty quantification.

Modules:
    cli          Command-line interface and execution entry point.
    config       Configuration loading and validation.
    data         Data ingestion, preprocessing, and splitting.
    models       Model construction utilities.
    training     Stage-specific training logic and callbacks.
    utils        Reproducibility and environment helpers.
    plotting     Plotting and diagnostic tools.
"""

from importlib.metadata import version

__all__ = ["__version__"]

try:
    __version__ = version("staged_bdann")
except Exception:
    __version__ = "0.0.0"