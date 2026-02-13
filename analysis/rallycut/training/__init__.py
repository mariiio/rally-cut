"""Training module for fine-tuning VideoMAE on beach volleyball."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.training.config import TrainingConfig
    from rallycut.training.dataset import BeachVolleyballDataset
    from rallycut.training.sampler import TrainingSample, generate_training_samples

__all__ = [
    "BeachVolleyballDataset",
    "TrainingConfig",
    "TrainingSample",
    "generate_training_samples",
]


def __getattr__(name: str) -> object:
    """Lazy imports to avoid pulling in heavy dependencies (transformers, etc.)."""
    if name == "TrainingConfig":
        from rallycut.training.config import TrainingConfig

        return TrainingConfig
    if name == "BeachVolleyballDataset":
        from rallycut.training.dataset import BeachVolleyballDataset

        return BeachVolleyballDataset
    if name == "TrainingSample":
        from rallycut.training.sampler import TrainingSample

        return TrainingSample
    if name == "generate_training_samples":
        from rallycut.training.sampler import generate_training_samples

        return generate_training_samples
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
