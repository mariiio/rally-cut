"""Unified crop-head model.

Task 0 (kill-test) uses the minimal Phase 1-compatible variant:
``FrozenResNet18`` + ``CropHeadMLP`` with concat input. Task 2 will widen
the constructor signature with ``input_kind`` / ``pool_kind`` /
``backbone_kind`` parameters — kept here as keyword-only args with
Phase 1 defaults so callers that supply them won't break when Task 2
lands.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from rallycut.ml.crop_head.backbone import FrozenResNet18
from rallycut.ml.crop_head.head import CropHeadMLP


class CropHeadModel(nn.Module):
    """Phase 1 crop-head: frozen ResNet-18 backbone + mean-pool MLP head.

    Signature accepts Phase 2A ablation keywords (``input_kind`` etc.) but
    Task 0 only validates Phase 1 defaults — non-default values raise
    ``NotImplementedError`` so a Task 0 caller that accidentally loads a
    future checkpoint dict fails fast rather than silently mis-constructing.
    """

    def __init__(
        self,
        input_kind: str = "concat",
        pool_kind: str = "mean",
        backbone_kind: str = "frozen",
    ) -> None:
        super().__init__()
        if input_kind != "concat":
            raise NotImplementedError(
                f"input_kind={input_kind!r} not supported at Task 0 "
                "(Phase 2A adds player_only / ball_only)."
            )
        if pool_kind != "mean":
            raise NotImplementedError(
                f"pool_kind={pool_kind!r} not supported at Task 0 "
                "(Phase 2A adds attention / max)."
            )
        if backbone_kind != "frozen":
            raise NotImplementedError(
                f"backbone_kind={backbone_kind!r} not supported at Task 0 "
                "(Phase 2A adds layer4_unfrozen)."
            )
        self.input_kind = input_kind
        self.pool_kind = pool_kind
        self.backbone_kind = backbone_kind
        self.backbone = FrozenResNet18()
        self.head = CropHeadMLP(d_in=1024, d_hidden=256)

    def forward(
        self, player_crops: torch.Tensor, ball_patches: torch.Tensor
    ) -> torch.Tensor:
        b, t, c, h, w = player_crops.shape
        player_flat = player_crops.reshape(b * t, c, h, w)
        ball_flat = ball_patches.reshape(
            b * t, c, ball_patches.shape[-2], ball_patches.shape[-1]
        )
        player_feat = self.backbone(player_flat).reshape(b, t, 512)
        ball_feat = self.backbone(ball_flat).reshape(b, t, 512)
        combined = torch.cat([player_feat, ball_feat], dim=-1)  # (B, T, 1024)
        out: torch.Tensor = self.head(combined)
        return out
