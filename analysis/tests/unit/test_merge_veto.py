"""Tests for the factored learned-ReID merge veto helper.

The helper ingests two track fragments (each as a set of frames) plus
a LearnedEmbeddingStore. It returns True when the median learned
embeddings are dissimilar enough that the merge should be blocked, and
abstains (returns False, allowing the merge) when either side lacks
enough evidence.
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.color_repair import LearnedEmbeddingStore
from rallycut.tracking.merge_veto import (
    _segment_median_embedding,
    learned_cosine_veto,
)


def _store_with(track_embeddings: dict[int, dict[int, np.ndarray]]) -> LearnedEmbeddingStore:
    store = LearnedEmbeddingStore()
    for tid, frames in track_embeddings.items():
        for fn, emb in frames.items():
            store.add(tid, fn, emb.astype(np.float32))
    return store


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = vec.astype(np.float32)
    return arr / np.linalg.norm(arr)


def test_veto_disabled_when_threshold_zero() -> None:
    """Threshold 0.0 → veto never fires (byte-identical default behavior)."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    b = _unit(np.array([-1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(10)},
        2: {fn: b for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(10)), 2, set(range(10, 20)), threshold=0.0
    ) is False


def test_veto_blocks_on_dissimilar_embeddings() -> None:
    """cos ≈ -1.0 with threshold 0.5 → block."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    b = _unit(np.array([-1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(10)},
        2: {fn: b for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(10)), 2, set(range(10, 20)), threshold=0.5
    ) is True


def test_veto_allows_on_similar_embeddings() -> None:
    """cos ≈ 1.0 with threshold 0.5 → allow."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(10)},
        2: {fn: a for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(10)), 2, set(range(10, 20)), threshold=0.5
    ) is False


def test_veto_abstains_when_embeddings_missing() -> None:
    """Fewer than 5 embeddings on either side → abstain (return False)."""
    a = _unit(np.array([1.0, 0.0] + [0.0] * 126))
    b = _unit(np.array([-1.0, 0.0] + [0.0] * 126))
    store = _store_with({
        1: {fn: a for fn in range(3)},   # only 3 frames
        2: {fn: b for fn in range(10, 20)},
    })
    assert learned_cosine_veto(
        store, 1, set(range(3)), 2, set(range(10, 20)), threshold=0.5
    ) is False


def test_veto_abstains_when_store_empty() -> None:
    """Empty store → abstain."""
    store = LearnedEmbeddingStore()
    assert learned_cosine_veto(
        store, 1, {0, 1, 2, 3, 4, 5}, 2, {10, 11, 12, 13, 14, 15}, threshold=0.5
    ) is False


def test_segment_median_renormalized() -> None:
    """Output of _segment_median_embedding is L2-normalized."""
    a = _unit(np.array([0.3, 0.4, 0.5] + [0.0] * 125))
    store = _store_with({1: {fn: a for fn in range(10)}})
    med = _segment_median_embedding(store, 1, set(range(10)))
    assert med is not None
    assert abs(float(np.linalg.norm(med)) - 1.0) < 1e-5
