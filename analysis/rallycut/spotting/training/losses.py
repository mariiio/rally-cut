"""Loss functions for E2E-Spot training."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch.nn import functional as nnf


class FocalLoss(nn.Module):
    """Focal loss for per-frame classification with class imbalance.

    Reduces loss contribution from well-classified examples, focusing
    training on hard negatives. Supports per-class weights.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: (N, C) or (B, T, C) class logits.
            targets: (N,) or (B, T) integer class labels.

        Returns:
            Scalar loss or per-element loss if reduction="none".
        """
        orig_shape = logits.shape
        if logits.dim() == 3:
            b, t, c = logits.shape
            logits = logits.reshape(b * t, c)
            targets = targets.reshape(b * t)

        weight = cast(torch.Tensor | None, self.weight)
        ce_loss = nnf.cross_entropy(
            logits, targets, weight=weight, reduction="none"
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        elif self.reduction == "none" and len(orig_shape) == 3:
            return focal_loss.reshape(orig_shape[0], orig_shape[1])
        return focal_loss


class OffsetLoss(nn.Module):
    """L1 loss for temporal offset regression, masked to event frames only."""

    def forward(
        self,
        pred_offsets: torch.Tensor,
        target_offsets: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked L1 offset loss.

        Args:
            pred_offsets: (B, T, 1) predicted offsets.
            target_offsets: (B, T) ground truth offsets.
            event_mask: (B, T) binary mask, 1 for event frames.

        Returns:
            Scalar loss (mean over event frames).
        """
        pred = pred_offsets.squeeze(-1)  # (B, T)
        loss = torch.abs(pred - target_offsets)
        masked = loss * event_mask
        num_events = event_mask.sum().clamp(min=1.0)
        return masked.sum() / num_events


def compute_class_weights(
    labels: list[torch.Tensor],
    num_classes: int,
    bg_weight_cap: float = 1.0,
    fg_weight_cap: float = 50.0,
) -> torch.Tensor:
    """Compute inverse-frequency class weights from training labels.

    Args:
        labels: List of (T,) label tensors from training clips.
        num_classes: Total number of classes including background.
        bg_weight_cap: Maximum weight for background class.
        fg_weight_cap: Maximum weight for foreground classes (prevents
            extremely rare classes like block from dominating gradients).

    Returns:
        (num_classes,) weight tensor.
    """
    import numpy as np

    all_labels = torch.cat(labels).numpy()
    counts = np.bincount(all_labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    max_count = counts.max()
    weights = max_count / counts

    # Cap weights to prevent gradient domination
    weights[0] = min(weights[0], bg_weight_cap)
    for i in range(1, num_classes):
        weights[i] = min(weights[i], fg_weight_cap)

    return torch.tensor(weights, dtype=torch.float32)
