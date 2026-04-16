"""Teammate-margin loss with hard mining.

For each anchor i, finds:
- hardest_teammate(i) = argmax over j with team_j == team_i AND y_j != y_i of cos(z_i, z_j)
- weakest_self(i)    = argmin over k with y_k == y_i AND k != i of cos(z_i, z_k)

Loss_i = relu(cos_team_max - cos_self_min + m).

Anchors lacking either are skipped. Returns mean over valid anchors plus the
running mean signal `cos_self - cos_team` for monitoring (must trend POSITIVE
during training — ≤ 0 after epoch 10 is the spec's escalation trigger).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TeammateMarginOutput:
    loss: torch.Tensor
    n_valid: int
    teammate_margin_mean: float   # mean(cos_self - cos_team) on valid anchors


class TeammateMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.30) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        feats: torch.Tensor,
        identities: torch.Tensor,
        teams: torch.Tensor,
    ) -> TeammateMarginOutput:
        """
        Args:
            feats:      (N, D) L2-normalized.
            identities: (N,) long — same value = same physical player (rally, canonical).
            teams:      (N,) long — same value = same team (rally, team).

        Returns:
            TeammateMarginOutput. If zero valid anchors, loss = 0 and n_valid = 0.
        """
        n = feats.shape[0]
        device = feats.device
        sim = feats @ feats.t()                              # (N, N)

        eye = torch.eye(n, dtype=torch.bool, device=device)
        same_id = identities.unsqueeze(0) == identities.unsqueeze(1)
        same_team = teams.unsqueeze(0) == teams.unsqueeze(1)

        pos_mask = same_id & ~eye                            # candidates for "self"
        team_mask = same_team & ~same_id & ~eye              # candidates for "teammate"

        # Hardest teammate: max sim where team_mask is True; else -inf.
        sim_team = sim.masked_fill(~team_mask, float("-inf"))
        cos_team_max = sim_team.max(dim=1).values            # (N,)

        # Weakest self: min sim where pos_mask is True; else +inf.
        sim_pos = sim.masked_fill(~pos_mask, float("inf"))
        cos_self_min = sim_pos.min(dim=1).values             # (N,)

        valid = team_mask.any(dim=1) & pos_mask.any(dim=1)   # (N,)
        n_valid = int(valid.sum().item())

        if n_valid == 0:
            return TeammateMarginOutput(
                loss=torch.zeros((), device=device, dtype=feats.dtype),
                n_valid=0,
                teammate_margin_mean=0.0,
            )

        # Loss is the hinge: penalize when team is closer than self.
        per_anchor = torch.relu(cos_team_max - cos_self_min + self.margin)
        loss = per_anchor[valid].mean()

        # Monitoring metric: signed margin (cos_self - cos_team), positive = healthy.
        with torch.no_grad():
            margin_signed = (cos_self_min - cos_team_max)[valid].mean().item()

        return TeammateMarginOutput(loss=loss, n_valid=n_valid, teammate_margin_mean=margin_signed)
