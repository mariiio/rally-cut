"""MS-TCN++ model for temporal action segmentation.

Minimal 2-stage refinement architecture with dual dilated layers.
Designed for small datasets (51 videos) — uses 32 hidden channels
and 2 stages to keep params at ~100K and avoid overfitting.

Reference: Li et al., "MS-TCN++: Multi-Stage Temporal Convolutional
Network for Action Segmentation" (TPAMI 2020).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as nnf


@dataclass
class MSTCNConfig:
    """Configuration for MS-TCN++ model."""

    feature_dim: int = 768  # Visual feature dim (768 for VideoMAE)
    num_stages: int = 2  # Number of refinement stages
    num_layers: int = 10  # Layers per stage (receptive field = 2^10 = 1024)
    hidden_dim: int = 32  # Small to prevent overfitting on 51 videos
    num_classes: int = 2  # rally / no_rally
    dropout: float = 0.3

    # Ball feature MLP projection (same pattern as TemporalMaxer)
    ball_feature_dim: int = 0  # Raw ball feature dims (5 for WASB)
    ball_projection_dim: int = 64  # Projected ball feature dims


class DualDilatedLayer(nn.Module):
    """Dual dilated convolution layer from MS-TCN++.

    Two parallel dilated convolutions with different dilations are
    applied and combined, giving the network access to multiple
    temporal scales at each layer.
    """

    def __init__(self, hidden_dim: int, dilation: int, dropout: float = 0.3) -> None:
        super().__init__()
        # Primary dilated conv
        self.conv_dilated = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=dilation, dilation=dilation,
        )
        # Secondary conv at half dilation (minimum 1)
        half_dilation = max(1, dilation // 2)
        self.conv_dilated_2 = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=half_dilation, dilation=half_dilation,
        )
        self.conv_merge = nn.Conv1d(2 * hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, hidden_dim, T)
            mask: (batch, 1, T)

        Returns:
            (batch, hidden_dim, T)
        """
        out1 = nnf.relu(self.conv_dilated(x))[:, :, :x.shape[2]]
        out2 = nnf.relu(self.conv_dilated_2(x))[:, :, :x.shape[2]]
        merged = self.conv_merge(torch.cat([out1, out2], dim=1))
        merged = self.dropout(merged)

        # LayerNorm expects (batch, T, dim) — transpose, norm, transpose back
        merged_t = merged.transpose(1, 2)  # (batch, T, dim)
        merged_t = self.norm(merged_t)
        merged = merged_t.transpose(1, 2)  # (batch, dim, T)

        result: torch.Tensor = (x + merged) * mask[:, :, :x.shape[2]]
        return result


class SingleStageTCN(nn.Module):
    """One stage of MS-TCN++.

    Stack of dual dilated layers with exponentially increasing dilation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.layers = nn.ModuleList([
            DualDilatedLayer(hidden_dim, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.output_proj = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, input_dim, T)
            mask: (batch, 1, T)

        Returns:
            (batch, output_dim, T) logits
        """
        out = self.input_proj(x) * mask
        for layer in self.layers:
            out = layer(out, mask)
        logits: torch.Tensor = self.output_proj(out) * mask
        return logits


class MSTCN(nn.Module):
    """MS-TCN++ for temporal action segmentation.

    Multi-stage architecture where each stage refines the previous
    stage's predictions. Stage 1 takes raw features; stages 2+
    take the softmax output of the previous stage.

    Architecture:
        Stage 1: features → SingleStageTCN → logits
        Stage 2: softmax(logits) → SingleStageTCN → refined logits
        ...

    Input: (batch, feature_dim, T) — feature sequence
    Output: list of (batch, num_classes, T) — per-stage logits
    """

    def __init__(self, config: MSTCNConfig | None = None) -> None:
        super().__init__()
        self.config = config or MSTCNConfig()
        c = self.config

        # Ball feature MLP projection (same as TemporalMaxer)
        input_dim = c.feature_dim
        if c.ball_feature_dim > 0:
            self.ball_proj = nn.Sequential(
                nn.Linear(c.ball_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, c.ball_projection_dim),
            )
            input_dim += c.ball_projection_dim

        # Stage 1: raw features → logits
        self.stages = nn.ModuleList()
        self.stages.append(SingleStageTCN(
            input_dim=input_dim,
            hidden_dim=c.hidden_dim,
            output_dim=c.num_classes,
            num_layers=c.num_layers,
            dropout=c.dropout,
        ))

        # Stages 2+: previous softmax → refined logits
        for _ in range(1, c.num_stages):
            self.stages.append(SingleStageTCN(
                input_dim=c.num_classes,
                hidden_dim=c.hidden_dim,
                output_dim=c.num_classes,
                num_layers=c.num_layers,
                dropout=c.dropout,
            ))

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        ball_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (batch, feature_dim, T) visual features.
            mask: (batch, 1, T) binary mask for padding.
            ball_features: (batch, ball_feature_dim, T) raw ball features.

        Returns:
            (batch, num_classes, T) logits from the final stage.
            For multi-stage training loss, use forward_all_stages().
        """
        stages_out = self.forward_all_stages(features, mask, ball_features)
        return stages_out[-1]

    def forward_all_stages(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
        ball_features: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Forward pass returning all stage outputs.

        Returns:
            List of (batch, num_classes, T) logits per stage.
        """
        batch_size, _, seq_len = features.shape

        if mask is None:
            mask = torch.ones(batch_size, 1, seq_len, device=features.device)

        # Fuse ball features via learned MLP projection
        if self.config.ball_feature_dim > 0:
            if ball_features is not None:
                bf = ball_features.transpose(1, 2)  # (batch, T, ball_dim)
                bf_proj = self.ball_proj(bf)  # (batch, T, proj_dim)
                bf_proj = bf_proj.transpose(1, 2)  # (batch, proj_dim, T)
            else:
                bf_proj = torch.zeros(
                    batch_size, self.config.ball_projection_dim, seq_len,
                    device=features.device,
                )
            features = torch.cat([features, bf_proj], dim=1)

        stage_outputs: list[torch.Tensor] = []

        # Stage 1: raw features
        logits = self.stages[0](features, mask)
        stage_outputs.append(logits)

        # Stages 2+: refine from previous softmax
        for stage in self.stages[1:]:
            probs = nnf.softmax(logits.detach(), dim=1) * mask
            logits = stage(probs, mask)
            stage_outputs.append(logits)

        return stage_outputs
