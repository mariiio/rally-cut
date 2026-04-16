"""Factored learned-ReID merge veto — used by every merge/rename pass.

Session 6 (tracklet_link.py) introduced the veto at one site. Session 8
extends it to every pass that plausibly creates same-team swaps. This
module owns the shared helpers so all adapter sites agree on signal
semantics, renormalization, and abstain behavior.

Public API:
    learned_cosine_veto(store, id_a, frames_a, id_b, frames_b, threshold) -> bool
        Returns True when the merge should be BLOCKED.

    _segment_median_embedding(store, track_id, frames) -> np.ndarray | None
        L2-renormalized median embedding for a track over `frames`.
        Returns None when fewer than 5 valid embeddings exist.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.color_repair import LearnedEmbeddingStore

LEARNED_MERGE_VETO_COS = float(
    os.environ.get("LEARNED_MERGE_VETO_COS", "0.0")
)

LEARNED_MERGE_VETO_MIN_FRAMES = 5
LEARNED_MERGE_VETO_MAX_SAMPLES = 30


def _segment_median_embedding(
    learned_store: LearnedEmbeddingStore | None,
    track_id: int,
    frames: set[int] | list[int],
    min_frames: int = LEARNED_MERGE_VETO_MIN_FRAMES,
    max_samples: int = LEARNED_MERGE_VETO_MAX_SAMPLES,
) -> np.ndarray | None:
    """L2-renormalized median learned-ReID embedding for a track over `frames`.

    Returns None when the store is absent, carries no data, or fewer than
    `min_frames` valid embeddings exist in the frame set. Samples up to
    `max_samples` uniformly-spaced frames when the fragment is longer.
    """
    if learned_store is None or not learned_store.has_data():
        return None
    sorted_frames = sorted(frames)
    if not sorted_frames:
        return None
    if len(sorted_frames) > max_samples:
        step = len(sorted_frames) / max_samples
        sorted_frames = [sorted_frames[int(i * step)] for i in range(max_samples)]
    vectors: list[np.ndarray] = []
    for fn in sorted_frames:
        emb = learned_store.get(track_id, fn)
        if emb is not None:
            vectors.append(emb)
    if len(vectors) < min_frames:
        return None
    stack = np.stack(vectors).astype(np.float32)
    med = np.median(stack, axis=0)
    norm = float(np.linalg.norm(med))
    if norm < 1e-8:
        return None
    normed: np.ndarray = (med / norm).astype(np.float32)
    return normed


def learned_cosine_veto(
    learned_store: LearnedEmbeddingStore | None,
    track_id_a: int,
    frames_a: set[int] | list[int],
    track_id_b: int,
    frames_b: set[int] | list[int],
    threshold: float = LEARNED_MERGE_VETO_COS,
) -> bool:
    """Return True when the learned head says these two fragments are
    different players — the proposed merge/rename should be BLOCKED.

    Abstain (return False = don't block) when the feature is disabled
    (threshold ≤ 0), either side has fewer than 5 valid embeddings, or
    the store carries no data.
    """
    if threshold <= 0.0:
        return False
    emb_a = _segment_median_embedding(learned_store, track_id_a, frames_a)
    emb_b = _segment_median_embedding(learned_store, track_id_b, frames_b)
    if emb_a is None or emb_b is None:
        return False
    cos_sim = float(np.dot(emb_a, emb_b))
    return cos_sim < threshold
