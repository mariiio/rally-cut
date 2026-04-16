"""Supervised contrastive loss (Khosla et al. 2020) with per-sample weighting.

Per-sample weight enables label smoothing of mid-tier hard-neg pairs by
downweighting their anchor contribution. weight=1.0 → full influence;
weight=(1-smoothing) → reduced influence (the smoothing budget acts as
"these labels might be wrong; reduce gradient").

L2-normed features, identity labels (long), per-sample weights (float).
Anchors with no in-batch positive contribute 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-anchor SupCon loss, weighted-averaged over batch.

        Args:
            feats: (N, D) L2-normalized embeddings.
            labels: (N,) long. Same value = same identity.
            weights: (N,) float in [0, 1] — per-anchor weight. None → all 1.0.

        Returns:
            Scalar loss.
        """
        n = feats.shape[0]
        device = feats.device
        if weights is None:
            weights = torch.ones(n, device=device)

        sim = feats @ feats.t() / self.tau                  # (N, N)
        eye = torch.eye(n, dtype=torch.bool, device=device)
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye

        # Numerical stability: subtract row max excluding self.
        sim_for_max = sim.masked_fill(eye, float("-inf"))
        sim_max = sim_for_max.max(dim=1, keepdim=True).values
        sim_stable = sim - sim_max
        exp_sim = torch.exp(sim_stable)
        exp_sim = exp_sim.masked_fill(eye, 0.0)

        # log denominator over all j != i
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12) + sim_max

        log_prob = sim - log_denom                          # (N, N)
        # Mean log prob over positives per anchor.
        n_pos = pos_mask.float().sum(dim=1)                 # (N,)
        log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1)
        loss_per_anchor = -log_prob_pos / n_pos.clamp(min=1)

        has_pos = (n_pos > 0).float()
        weighted = loss_per_anchor * weights * has_pos
        denom = (weights * has_pos).sum().clamp(min=1)
        return weighted.sum() / denom
