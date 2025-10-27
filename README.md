Repository Structure
====================

This repository implements a staged Bayesian Domain-Adversarial Neural Network (B-DANN) training framework.
Currently supports n-input, 1-output regression. Multi-output capabilities will be in a future update.

staged_bdann/
    __init__.py
        Package initializer with module-level documentation.

    cli.py
        Command-line entry point. Parses arguments, loads configuration,
        sets up the environment, and calls either a single pipeline run
        or a multi-run experiment.

    config.py
        Loads and validates JSON configuration files.

    utils.py
        Utility functions for determinism, run directory creation,
        and version stamping.

    data.py
        Handles data loading, splitting, and preprocessing.

    models.py
        Defines neural network architectures, including the feature extractor,
        regression head, domain head, and Bayesian layers.

    training/
        stage1.py      Stage 1: supervised source-domain training.
        stage2.py      Stage 2: domain-adversarial alignment.
        stage3.py      Stage 3: Bayesian fine-tuning on the target domain.
        callbacks.py   Custom Keras callbacks such as schedulers and early stopping.
        batching.py    Balanced batch generators and sampling utilities.
        common.py      Shared loss functions and probability distributions.

    pipeline.py
        Runs a full three-stage training sequence for one configuration and seed.

    multi_run.py
        Handles multiple runs on fixed splits, including seed ensembles,
        and will later include cross-validation and hyperparameter sweeps.

    plotting.py
        Generates diagnostic plots such as parity plots, histograms,
        uncertainty calibration, and loss curves.

scripts/
    staged_bdann_main.py
        Executable wrapper that runs staged_bdann.cli.main.  
        Example usage:
            python -m staged_bdann.cli --config configs/example_config.json

Notes
====================

- The dependency flow is one-directional:
  cli.py -> multi_run.py -> pipeline.py -> training -> models -> utils

- Run directories under runs/ store configurations, weights, metrics, and plots.

- TensorFlow determinism and reproducibility are enforced via utils.set_global_determinism.
