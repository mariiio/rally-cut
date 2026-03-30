"""BiGRU temporal module for sequence modeling.

Takes per-frame features from the backbone and adds long-range temporal
context via bidirectional GRU layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rallycut.spotting.config import TemporalConfig


class BiGRUTemporal(nn.Module):
    """Bidirectional GRU for temporal context aggregation.

    Input:  (B, T, D) per-frame features from backbone
    Output: (B, T, 2*hidden_dim) temporally-contextualized features
    """

    def __init__(self, input_dim: int, config: TemporalConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout_in = nn.Dropout(config.dropout)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_dim = config.hidden_dim * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) per-frame features.

        Returns:
            (B, T, 2*hidden_dim) contextualized features.
        """
        x = self.norm(x)
        x = self.dropout_in(x)
        x, _ = self.gru(x)
        return x
