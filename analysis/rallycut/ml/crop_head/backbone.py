"""Frozen ImageNet-pretrained ResNet-18 feature extractor for crop inputs."""
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class FrozenResNet18(nn.Module):
    """Per-frame feature extractor. Output: 512-dim feature per input image.

    Applied to (B*T, 3, H, W) → returns (B*T, 512). Caller reshapes to
    (B, T, 512). All weights frozen — this module contributes zero
    gradient to the training loop.
    """

    def __init__(self) -> None:
        super().__init__()
        m = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        x = (x - mean) / std
        feat: torch.Tensor = self.backbone(x)
        return feat.flatten(1)
