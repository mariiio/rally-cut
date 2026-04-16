"""MLP head: 384 → 192 → 128, BN + GELU, L2-normalized output."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class MLPHead(nn.Module):
    """2-layer projection head as locked in the plan.

    Input: (N, 384) L2-normalized DINOv2 features.
    Output: (N, 128) L2-normalized embeddings.
    """

    def __init__(self, in_dim: int = 384, hidden_dim: int = 192, out_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # Init: Xavier on linears, zero on biases — standard, prevents large initial logits
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        # BatchNorm needs > 1 sample to compute stats. Accept N=1 by skipping BN at eval time.
        if x.shape[0] > 1 or not self.training:
            x = self.bn1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return F.normalize(x, dim=1)
