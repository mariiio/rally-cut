"""TemporalMaxer temporal action segmentation model.

Vendored and adapted from https://github.com/TuanTNG/TemporalMaxer.
Replaces per-window classification + handcrafted decoder with a learned
temporal action segmentation model that processes the full sequence.
"""

from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference
from rallycut.temporal.temporal_maxer.model import TemporalMaxer, TemporalMaxerConfig
from rallycut.temporal.temporal_maxer.training import (
    TemporalMaxerTrainer,
    TemporalMaxerTrainingConfig,
)

__all__ = [
    "TemporalMaxer",
    "TemporalMaxerConfig",
    "TemporalMaxerInference",
    "TemporalMaxerTrainer",
    "TemporalMaxerTrainingConfig",
]
