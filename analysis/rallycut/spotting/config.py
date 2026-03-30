"""Configuration dataclasses for E2E-Spot action spotting."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BackboneConfig:
    model_name: str = "regnety_002"  # RegNet-Y 200MF in timm
    pretrained: bool = True
    feature_dim: int = 368  # regnety_002 output channels
    gsm_stages: list[int] = field(default_factory=lambda: [1, 2, 3])


@dataclass
class TemporalConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3


@dataclass
class HeadConfig:
    num_classes: int = 7  # background + serve/receive/set/attack/dig/block


@dataclass
class PostprocessConfig:
    confidence_threshold: float = 0.3
    nms_window: int = 3  # frames for Soft-NMS Gaussian window
    nms_sigma: float = 1.0


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 150
    batch_size: int = 8
    clip_length: int = 96
    warmup_epochs: int = 3
    patience: int = 20
    focal_gamma: float = 2.0
    offset_weight: float = 0.1
    grad_clip: float = 1.0
    seed: int = 42
    num_workers: int = 2
    # Finetuning
    freeze_backbone_epochs: int = 0  # 0 = no freezing
    backbone_lr_scale: float = 0.1  # LR multiplier for backbone after unfreeze


@dataclass
class E2ESpotConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Action classes shared across the spotting module
ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]
ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(ACTION_TYPES)}  # 0 = background
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
NUM_CLASSES = len(ACTION_TYPES) + 1  # 7
