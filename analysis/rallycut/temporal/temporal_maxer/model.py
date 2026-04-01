"""TemporalMaxer model for temporal action segmentation.

Adapted from https://github.com/TuanTNG/TemporalMaxer.
Uses temporal feature pyramid with max pooling between layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as nnf


@dataclass
class TemporalMaxerConfig:
    """Configuration for TemporalMaxer model."""

    feature_dim: int = 768  # Visual feature dim (768 for VideoMAE)
    hidden_dim: int = 768  # Hidden dimension (same as feature dim)
    num_classes: int = 2  # rally / no_rally
    num_layers: int = 4  # Temporal pyramid depth
    dropout: float = 0.3

    # Ball feature MLP projection (0 = disabled, legacy concat mode)
    ball_feature_dim: int = 0  # Raw ball feature dims (5 for WASB)
    ball_projection_dim: int = 64  # Projected ball feature dims


class DilatedResidualBlock(nn.Module):
    """1D dilated convolution with residual connection."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv_1x1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = nnf.relu(self.conv_dilated(x))
        # Truncate to match input length (dilated conv can change length slightly)
        out = out[:, :, :x.shape[2]]
        out = self.conv_1x1(out)
        out = self.dropout(out)
        # BatchNorm needs at least 2 time steps
        if out.shape[2] > 1:
            out = self.norm(out)
        result: torch.Tensor = (x + out) * mask[:, :, :x.shape[2]]
        return result


class TemporalMaxerLayer(nn.Module):
    """One layer of the TemporalMaxer pyramid.

    Each layer processes features at a particular temporal scale:
    1. Apply dilated residual blocks
    2. Max pool to reduce temporal resolution (downsample)
    3. Process at reduced resolution
    4. Upsample back
    """

    def __init__(
        self,
        hidden_dim: int,
        num_sublayers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            DilatedResidualBlock(hidden_dim, dilation=2 ** i, dropout=dropout)
            for i in range(num_sublayers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, mask)
        return x


class TemporalMaxer(nn.Module):
    """TemporalMaxer for temporal action segmentation.

    Architecture:
        Input projection → Temporal pyramid (multiple scales via max pooling)
        → Per-window class logits

    Input: (batch, feature_dim, T) — feature sequence
    Output: (batch, num_classes, T) — per-window class logits
    """

    def __init__(self, config: TemporalMaxerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalMaxerConfig()
        c = self.config

        # Ball feature MLP projection
        input_dim = c.feature_dim
        if c.ball_feature_dim > 0:
            self.ball_proj = nn.Sequential(
                nn.Linear(c.ball_feature_dim, 32),
                nn.ReLU(),
                nn.Linear(32, c.ball_projection_dim),
            )
            input_dim += c.ball_projection_dim

        # Input projection
        self.input_proj = nn.Conv1d(input_dim, c.hidden_dim, kernel_size=1)
        self.input_norm = nn.BatchNorm1d(c.hidden_dim)

        # Multi-scale temporal layers
        self.layers = nn.ModuleList([
            TemporalMaxerLayer(c.hidden_dim, num_sublayers=2, dropout=c.dropout)
            for _ in range(c.num_layers)
        ])

        # Output classification head
        self.output_proj = nn.Conv1d(c.hidden_dim, c.num_classes, kernel_size=1)

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
                Required when config.ball_feature_dim > 0. If None when
                ball features are configured, zeros are used (inference
                without ball data).

        Returns:
            (batch, num_classes, T) per-window class logits.
        """
        batch_size, _, seq_len = features.shape

        if mask is None:
            mask = torch.ones(batch_size, 1, seq_len, device=features.device)

        # Fuse ball features via learned MLP projection
        if self.config.ball_feature_dim > 0:
            if ball_features is not None:
                # ball_features: (batch, ball_dim, T) → project per-timestep
                # Transpose to (batch, T, ball_dim), project, transpose back
                bf = ball_features.transpose(1, 2)  # (batch, T, ball_dim)
                bf_proj = self.ball_proj(bf)  # (batch, T, proj_dim)
                bf_proj = bf_proj.transpose(1, 2)  # (batch, proj_dim, T)
            else:
                bf_proj = torch.zeros(
                    batch_size, self.config.ball_projection_dim, seq_len,
                    device=features.device,
                )
            features = torch.cat([features, bf_proj], dim=1)

        # Store original mask before downsampling modifies it
        original_mask = mask

        # Input projection
        x = self.input_proj(features)
        x = self.input_norm(x)
        x = nnf.relu(x) * mask

        # Multi-scale processing with max pooling
        # Process at multiple temporal resolutions
        multi_scale_features: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            x = layer(x, mask)
            multi_scale_features.append(x)

            if i < len(self.layers) - 1:
                # Downsample with max pooling (factor 2)
                if x.shape[2] > 1:
                    # Pad to even length before pooling to avoid size mismatches
                    if x.shape[2] % 2 != 0:
                        x = nnf.pad(x, (0, 1))
                        mask = nnf.pad(mask, (0, 1))
                    x = nnf.max_pool1d(x, kernel_size=2, stride=2)
                    mask = nnf.max_pool1d(mask, kernel_size=2, stride=2)

        # Upsample and combine multi-scale features
        # Start from the coarsest scale and upsample back
        out = multi_scale_features[-1]
        for i in range(len(multi_scale_features) - 2, -1, -1):
            target_len = multi_scale_features[i].shape[2]
            if out.shape[2] != target_len:
                out = nnf.interpolate(out, size=target_len, mode="linear", align_corners=False)
            out = out + multi_scale_features[i]

        # Output projection (use original mask, not the downsampled/padded one)
        logits: torch.Tensor = self.output_proj(out * original_mask)

        return logits
