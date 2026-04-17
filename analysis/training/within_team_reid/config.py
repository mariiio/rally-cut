"""Training configuration for within-team ReID head.

Frozen dataclass — variants override fields via constructor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repository-relative roots (resolved against analysis/ working directory)
TRAINING_DATA_ROOT = Path("training_data/within_team_reid")
EVAL_CACHE_PATH = Path("reports/within_team_reid/eval_cache/eval_cache.npz")
EVAL_CACHE_META = Path("reports/within_team_reid/eval_cache/metadata.json")
WEIGHTS_ROOT = Path("weights/within_team_reid")
REPORTS_ROOT = Path("reports/within_team_reid")
AUDIT_DIR = Path("reports/tracking_audit")
REID_DEBUG_DIR = Path("reports/tracking_audit/reid_debug")
PROBE_CROSS_RALLY_JSON = Path("reports/tracking_audit/reid_debug/sota_probe_cross_rally.json")

# Held-out split: events 35-58 of the 58 sorted swap events.
RANKING_SIZE = 34            # events 1-34 are the ranking/training set (NEVER held out)
HELD_OUT_START = 34          # 0-indexed slice [34:58]
HELD_OUT_END = 58            # exclusive

# Cross-rally guard
CROSS_RALLY_BASELINE_DINOV2_S = 0.703   # DINOv2-S zero-shot rank-1 from probe
CROSS_RALLY_GUARD = 0.683               # 2pp tolerance below baseline

# Held-out within-team primary metric
HELDOUT_BASELINE_DINOV2_S = 0.10        # DINOv2-S zero-shot pos% from probe
HELDOUT_TARGET = 0.20                   # +10pp goal

# Backbone (matches reid_embeddings.py)
BACKBONE_DIM = 384


@dataclass(frozen=True)
class TrainConfig:
    """Per-variant training hyperparameters."""

    variant_id: str = "v1"

    # Architecture
    head_hidden: int = 192
    head_out: int = 128

    # Loss
    tau: float = 0.07
    teammate_margin_m: float = 0.30
    lam_tm: float = 0.50          # weight on teammate-margin loss
    label_smoothing_mid: float = 0.05  # downweights mid-tier hard-neg samples in SupCon

    # Optimization
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    early_stop_patience: int = 5

    # Batch composition (literal reading: 64+32+32 = 128 PAIRS per step → 256 crops)
    pos_pairs_per_batch: int = 64
    easy_neg_pairs_per_batch: int = 32
    hard_neg_pairs_per_batch: int = 32
    crops_per_track_per_pair: int = 1   # → 2 crops per pair (one per track)

    # Reproducibility
    seed: int = 42

    # Runtime
    device: str = "auto"           # "auto" | "cuda" | "mps" | "cpu"
    num_workers: int = 4
    mixed_precision: bool = True

    # Output (relative to analysis/)
    weights_dir: Path = field(default_factory=lambda: WEIGHTS_ROOT)
    reports_dir: Path = field(default_factory=lambda: REPORTS_ROOT)

    @property
    def batch_pairs(self) -> int:
        return self.pos_pairs_per_batch + self.easy_neg_pairs_per_batch + self.hard_neg_pairs_per_batch

    @property
    def batch_crops(self) -> int:
        return self.batch_pairs * 2 * self.crops_per_track_per_pair


VARIANT_CONFIGS: dict[str, TrainConfig] = {
    "v1": TrainConfig(variant_id="v1", lam_tm=0.50, label_smoothing_mid=0.05),
    "v2": TrainConfig(variant_id="v2", lam_tm=1.00, label_smoothing_mid=0.02),
    "v3": TrainConfig(variant_id="v3", lam_tm=0.25, label_smoothing_mid=0.00),
    "v4": TrainConfig(variant_id="v4", lam_tm=0.50, label_smoothing_mid=0.05),
}
