"""MLP classifier head over mean-pooled per-frame features."""
from __future__ import annotations

import torch
import torch.nn as nn


class CropHeadMLP(nn.Module):
    """Takes (B, T, D_player + D_ball) → (B,) binary logit.

    Mean-pools across T, then MLP. Minimal architecture for the Phase 1
    frozen-backbone sanity probe. If this works, Phase 2 ablates pooling
    choice (mean vs attention) and temporal window.
    """

    def __init__(self, d_in: int = 1024, d_hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        out: torch.Tensor = self.net(pooled)
        return out.squeeze(-1)
