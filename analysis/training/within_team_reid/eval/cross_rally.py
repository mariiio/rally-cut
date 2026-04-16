"""Cross-rally rank-1 (catastrophic-forgetting guard).

Mirrors `probe_reid_models_on_swaps.compute_rank1`: leave-one-out per
(video, rally), gallery = other rallies in same video, find argmax cosine,
score 1 if best.canonical_id == query.canonical_id.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

from .cache import CrossRallyEntryMeta, EvalCache


@dataclass
class CrossRallyResult:
    rank1: float
    n_queries: int


def _embed_through_head(
    feats: NDArray[np.float32],
    head: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    x = torch.from_numpy(feats).to(device)
    head.eval()
    with torch.no_grad():
        h = head(x)
    mean = h.mean(dim=0, keepdim=True)
    return F.normalize(mean, dim=1).squeeze(0)


def evaluate(
    cache: EvalCache,
    head: torch.nn.Module,
    device: torch.device,
) -> CrossRallyResult:
    """Apply head to cached DINOv2 features, mean-pool, L2-norm, then leave-one-out
    rank-1 per (video, rally).
    """
    embeddings: dict[int, torch.Tensor] = {}
    for entry in cache.cross_rally_entries:
        feats = cache.cross_rally_features.get(f"g{entry.entry_idx}")
        if feats is None:
            continue
        embeddings[entry.entry_idx] = _embed_through_head(feats, head, device)

    by_video: dict[str, list[CrossRallyEntryMeta]] = defaultdict(list)
    for entry in cache.cross_rally_entries:
        if entry.entry_idx in embeddings:
            by_video[entry.video_id].append(entry)

    correct = 0
    total = 0
    for _vid, video_entries in by_video.items():
        rallies = {e.rally_id for e in video_entries}
        if len(rallies) < 2:
            continue
        for query in video_entries:
            gallery = [g for g in video_entries if g.rally_id != query.rally_id]
            if not gallery:
                continue
            q_emb = embeddings[query.entry_idx]
            best: CrossRallyEntryMeta | None = None
            best_sim = -2.0
            for g in gallery:
                sim = float(torch.dot(q_emb, embeddings[g.entry_idx]).item())
                if sim > best_sim:
                    best_sim = sim
                    best = g
            total += 1
            if best is not None and best.canonical_id == query.canonical_id:
                correct += 1

    rank1 = correct / total if total else 0.0
    return CrossRallyResult(rank1=rank1, n_queries=total)
