"""Held-out within-team rank-1 on the 24 reserved swap events.

Metric: for each event, signal = cos(query, anchor_correct) - cos(query, anchor_wrong)
where the three groups are mean-pooled L2-normalized head embeddings of the
cached DINOv2 features. rank-1 = (signal > 0).mean() over **SCORED** events
(matches probe `proportion_positive` denominator, not n_total).

DINOv2-S zero-shot baseline = 0.10 (1 / 10 scored). Target = 0.20 (+10 pp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from .cache import EvalCache, HeldOutEventMeta

logger = logging.getLogger("within_team_reid.eval.held_out")


@dataclass
class HeldOutResult:
    rank1: float                      # primary metric: pos / n_total_events
    n_total: int
    n_scored: int
    n_abstained: int
    n_positive: int                   # signal > 0
    teammate_margin_mean: float       # mean signal across SCORED events
    per_event_signals: list[float | None]  # None for abstained


def _embed_through_head(
    feats: NDArray[np.float32],
    head: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """Pass per-crop DINOv2 features through head, then mean-pool, then L2-norm.

    Returns a single 128-d torch tensor on `device`.
    """
    x = torch.from_numpy(feats).to(device)
    head.eval()
    with torch.no_grad():
        h = head(x)                   # (N, 128) — head L2-norms internally
    mean = h.mean(dim=0, keepdim=True)
    return F.normalize(mean, dim=1).squeeze(0)


def evaluate(
    cache: EvalCache,
    head: torch.nn.Module,
    device: torch.device,
) -> HeldOutResult:
    """Compute held-out within-team rank-1 by passing cached DINOv2 features
    through the (already-trained) head.
    """
    per_event_signals: list[float | None] = []
    scored_signals: list[float] = []
    n_positive = 0

    for ev in cache.held_out_events:
        if not ev.is_scorable():
            per_event_signals.append(None)
            continue
        c_feats = cache.held_out_features.get(f"e{ev.event_idx}_correct")
        w_feats = cache.held_out_features.get(f"e{ev.event_idx}_wrong")
        q_feats = cache.held_out_features.get(f"e{ev.event_idx}_query")
        if c_feats is None or w_feats is None or q_feats is None:
            per_event_signals.append(None)
            continue

        c = _embed_through_head(c_feats, head, device)
        w = _embed_through_head(w_feats, head, device)
        q = _embed_through_head(q_feats, head, device)

        sig_correct = float(torch.dot(q, c).item())
        sig_wrong = float(torch.dot(q, w).item())
        signal = sig_correct - sig_wrong
        per_event_signals.append(signal)
        scored_signals.append(signal)
        if signal > 0:
            n_positive += 1

    n_total = len(cache.held_out_events)
    n_scored = len(scored_signals)
    n_abstained = n_total - n_scored
    # Match probe convention: positives / SCORED (not / total). Mirrors
    # `aggregate_stats.proportion_positive` in probe_reid_models_on_swaps.py.
    rank1 = n_positive / n_scored if n_scored else 0.0
    teammate_margin_mean = (
        float(np.mean(scored_signals)) if scored_signals else 0.0
    )

    return HeldOutResult(
        rank1=rank1,
        n_total=n_total,
        n_scored=n_scored,
        n_abstained=n_abstained,
        n_positive=n_positive,
        teammate_margin_mean=teammate_margin_mean,
        per_event_signals=per_event_signals,
    )


class IdentityHead(torch.nn.Module):
    """No-op 'head' used for the smoke test — passes DINOv2 features through unchanged.

    L2-normalize at the end so semantics match a real head's output.
    """

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1)


def evaluate_identity_baseline(cache: EvalCache, device: torch.device) -> HeldOutResult:
    """Smoke test: with identity head, must reproduce DINOv2-S zero-shot baseline (~10%).

    Used to validate the eval cache + held-out rank-1 implementation.
    """
    return evaluate(cache, IdentityHead(), device)
