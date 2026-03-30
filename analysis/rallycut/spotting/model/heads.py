"""Prediction heads for E2E-Spot: classification and offset regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """Per-frame action classification head.

    Produces logits for K+1 classes (background + K action types).
    """

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, D). Returns: (B, T, num_classes) logits."""
        return self.fc(x)  # type: ignore[no-any-return]


class OffsetHead(nn.Module):
    """Per-frame temporal offset regression head.

    Predicts a small frame offset (Δt) to refine the exact event frame.
    Only supervised on frames near ground truth events.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x (B, T, D). Returns: (B, T, 1) offsets."""
        return self.fc(x)  # type: ignore[no-any-return]
