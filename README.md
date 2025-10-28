Staged Bayesian Domain Adversarial Neural Network (Staged B-DANN)
====================

The Staged B-DANN[^fn1] is a supervised transfer learning framework designed to add control on _what_ information is transferred from the source domain to the target domain/task, and to add uncertainty quantification (UQ) capabilities. This method emphasizes finding _domain invariant_ features (shared latent representations) between domains, focusing only on transfering information that applies to both. While initially inspired by the DANN-R[^fn2], a single-stage joint-optimization semi-supervised technique designed to predict both source and target tasks, the Staged B-DANN capitalizes on having labeled data to improve target-specific performance and training stability via serial single-optimization (the original DANN was _very_ sensitive to hyperparameters and training instabilities). It is arranged in three primary stages:

1. **Pre-training** (source data, deterministic): Feature extractor (FE) and regression head (RegHead) trained to predict in the source domain, provides initial problem context and improves stability for Stage 2.
2. **Domain Alignment** (source + target data, deterministic): FE then connected to a domain classifier (DC) via a gradient reversal layer (GRL). The DC attempts to classify the domain label of a specific training instance, backpropagating a BCE loss to improve DC parameters. This gradient is then negated at the GRL to modify the FE weights against the gradient, shifting the FE to make classification more difficult (i.e., incentivizing domain invariant learning). The optimization goal of Stage 2 is to maximize confusion of the DC, ending with an accuracy of 0.5 and a BCE of ~0.693. The Stage 1 RegHead is also attached to the FE, but is frozen and only useful for diagnostic information.
3. **Fine-Tuning** (target data, Bayesian): The Stage 2 FE and Stage 1 RegHead weights are then used to initialize a Bayesian neural network (BNN) using variational inference (VI), the deterministic point-value weights becoming the means of the BNN's now-Gaussian weight distributions. This stage now fine-tunes the resulting BNN using only target data, specializing the model towards the target task. Since the goal of the FE is to contain a maximal amount of shared information, it may be advantageous to freeze earlier layers in the BNN with only later layers trainable, or even keeping earlier layers deterministic and just transitioning later layers to be Bayesian. These are all considerations when optimizing Stage 3 architecture and hyperparameters.

Since the Stage 3 model is a full or partial BNN, it is capable of quantifying both _aleatoric_ (due to data) and _epistemic_ (due to model optimization) uncertainties, which are important when deploying in sensitive applications. For stable and reliable uncertainty estimates, it is recommended to have at least two Bayesian layers at the end of the Stage 3 model's architecture[^fn3]. To assist in ascertaining the quality of uncertainty estimates, calibration and uncertainty distribution metrics/plots are generated.

![alt text](https://github.com/ajfurlong/staged_bdann_project/blob/main/docs/b-dann_3stage_workflow.png "Staged B-DANN Workflow")

This script is designed to be a dynamic toolbox for applying the Staged B-DANN framework to fit the needs of the study. Here are some of the currently active features to choose from:

* Strict enforcement of determinism and user-selected seeding
* Single-file data entry (code splits into train/val/test) or manual spec (individual files for the partitions)
* Train Stage 1 FE/RegHead as part of the workflow, or load in a base model's FE and RegHead manually
* Bypass Stage 2 option, effectively only performing Pre-Training and Fine-Tuning
* Fully-deterministic option for Stage 3 (Staged D-DANN), if you're a die-hard frequentist
* User-spec Stage 3 number of Bayesian layers and how many layers unfrozen
* Random seed ensemble option to train a model config several times for statistics
* Stage 2 GRL lambda scheduling & multi-param early stopping (stability enhancements)
* A lot of plots, quantitative reporting, and diagnostics information
* Standard ML features: user-spec architecture depth/width/activation, data standardization, data noising, dropout, LR decay, early stopping

Future additions:

* Automated hyperparameter tuning
* Support for non-csv data file types such as HDF5

[^fn1]: Furlong, A., Salko, R., & Zhao, X., & Wu, X. (2025). A Three-Stage Bayesian Transfer Learning Framework to Improve Predictions in Data-Scarce Domains.
[^fn2]: Farahani, H. S., Fatehi, A., Nadali, A., & Shoorehdeli, M. A. (2021). Domain Adversarial Neural Network Regression to design transferable soft sensor in a power plant. Computers in Industry, 132, 103489.
[^fn3]: Zeng, J., Lesnikowski, A., & Alvarez, J. M. (2018). The relevance of Bayesian layer positioning to model uncertainty in deep Bayesian active learning. arXiv preprint arXiv:1811.12535.

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
