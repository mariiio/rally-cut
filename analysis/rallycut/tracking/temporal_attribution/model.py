"""Temporal contact attribution model.

Small 1D-CNN with dilated convolutions over a ~21-frame window of ball + player
trajectories. Predicts which of 4 canonical player slots touched the ball.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class TemporalAttributionConfig:
    """Configuration for the temporal attribution model."""

    input_dim: int = 14  # 4 distances + 4 delta_dist + ball_speed + dir_change + 4 heights
    hidden_dim: int = 32
    num_players: int = 4  # output classes (canonical slots)
    window_size: int = 21  # frames in window (±10 around contact)
    num_conv_layers: int = 3
    dropout: float = 0.2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> TemporalAttributionConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class TemporalAttributionModel(nn.Module):
    """1D-CNN for temporal contact attribution.

    Architecture:
        3 dilated Conv1d layers (dilation=1,2,4) with BN + ReLU + Dropout
        → Global average pool → Linear → 4-class output

    Input: (batch, input_dim, window_size) — channels-first
    Output: (batch, num_players) — logits
    """

    def __init__(self, config: TemporalAttributionConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[nn.Module] = []
        in_channels = config.input_dim

        for i in range(config.num_conv_layers):
            dilation = 2**i
            padding = dilation  # Same padding for kernel_size=3
            layers.extend([
                nn.Conv1d(
                    in_channels,
                    config.hidden_dim,
                    kernel_size=3,
                    dilation=dilation,
                    padding=padding,
                ),
                nn.BatchNorm1d(config.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(config.dropout),
            ])
            in_channels = config.hidden_dim

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(config.hidden_dim, config.num_players)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, input_dim, window_size) trajectory features.

        Returns:
            (batch, num_players) class logits.
        """
        h = self.conv_layers(x)  # (batch, hidden_dim, window_size)
        h = self.pool(h).squeeze(-1)  # (batch, hidden_dim)
        out: torch.Tensor = self.classifier(h)  # (batch, num_players)
        return out
