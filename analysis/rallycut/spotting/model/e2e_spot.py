"""Full E2E-Spot model: backbone + temporal + heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from rallycut.spotting.config import E2ESpotConfig
from rallycut.spotting.model.backbone import RegNetYGSM
from rallycut.spotting.model.heads import ClassificationHead, OffsetHead
from rallycut.spotting.model.temporal import BiGRUTemporal


class E2ESpot(nn.Module):
    """End-to-end action spotting model.

    Architecture:
        Raw frames → RegNet-Y + GSM (per-frame features)
        → BiGRU (temporal context)
        → Classification head (per-frame action logits)
        → Offset head (frame refinement)
    """

    def __init__(self, config: E2ESpotConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = RegNetYGSM(config.backbone)
        self.temporal = BiGRUTemporal(
            input_dim=config.backbone.feature_dim,
            config=config.temporal,
        )
        self.cls_head = ClassificationHead(
            input_dim=self.temporal.output_dim,
            num_classes=config.head.num_classes,
        )
        self.offset_head = OffsetHead(
            input_dim=self.temporal.output_dim,
        )

    def forward(self, clips: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass from raw frames to predictions.

        Args:
            clips: (B, T, 3, H, W) tensor of video clips.

        Returns:
            Dict with:
                logits: (B, T, num_classes) per-frame class logits
                offsets: (B, T, 1) per-frame temporal offsets
        """
        features = self.backbone(clips)       # (B, T, D)
        temporal = self.temporal(features)     # (B, T, 2*hidden)
        logits = self.cls_head(temporal)       # (B, T, num_classes)
        offsets = self.offset_head(temporal)   # (B, T, 1)
        return {"logits": logits, "offsets": offsets}

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters (for finetuning warmup)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_param_groups(self, backbone_lr_scale: float = 0.1) -> list[dict]:
        """Get parameter groups with optional lower LR for backbone.

        Args:
            backbone_lr_scale: Multiplier for backbone learning rate.

        Returns:
            List of param group dicts for optimizer.
        """
        backbone_params = list(self.backbone.parameters())
        other_params = [
            p
            for name, p in self.named_parameters()
            if not name.startswith("backbone.")
        ]
        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": other_params, "lr_scale": 1.0},
        ]
