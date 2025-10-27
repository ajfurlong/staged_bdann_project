"""Training subpackage for staged_bdann.

Contains the three sequential training stages, callbacks, and utilities shared between them:
    stage1  - Deterministic source-domain pretraining.
    stage2  - Modification based on the idea of a Domain-adversarial alignment (DANN).
    stage3  - Bayesian fine-tuning on target data.
    callbacks, batching, common - Supporting modules.
"""

from .stage1 import stage1_train_or_load
from .stage2 import stage2_dann
from .stage3 import stage3_finetune_hybrid

__all__ = ["stage1_train_or_load", "stage2_dann", "stage3_finetune_hybrid"]