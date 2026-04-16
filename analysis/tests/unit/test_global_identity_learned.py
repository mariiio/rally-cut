"""Session 4 — within-team learned-ReID integration tests.

Three guardrails:
1. Zero-weight parity: with ``WEIGHT_LEARNED_REID=0`` the learned cost block
   is completely inert — cost is byte-identical to pre-integration output.
2. Directionality: at ``WEIGHT_LEARNED_REID > 0`` the learned cost correctly
   prefers same-player over teammate-mismatch pairs.
3. Graceful degradation: ``extract_learned_embeddings`` returns an empty
   array (no crash) when the checkpoint is unavailable.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from rallycut.tracking.color_repair import LearnedEmbeddingStore
from rallycut.tracking.player_tracker import PlayerPosition


def _build_segment(track_id: int = 1, n_frames: int = 10):
    from rallycut.tracking.global_identity import TrackSegment

    positions = [
        PlayerPosition(
            track_id=track_id,
            frame_number=i,
            x=0.5,
            y=0.5,
            width=0.1,
            height=0.3,
            confidence=0.9,
        )
        for i in range(n_frames)
    ]
    return TrackSegment(
        track_id=track_id,
        start_frame=0,
        end_frame=n_frames - 1,
        team=0,
        positions=positions,
    )


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-8)


def test_zero_weight_byte_equivalent(monkeypatch):
    """At W=0, adding the learned params must not change the returned cost."""
    monkeypatch.setenv("WEIGHT_LEARNED_REID", "0.0")
    import rallycut.tracking.global_identity as gi

    gi = importlib.reload(gi)

    seg = _build_segment()
    profile = gi.PlayerProfile(player_id=1, team=0, histogram=None)

    cost_without_kw = gi._compute_assignment_cost(
        seg, profile, None, None, None, None, court_split_y=None,
    )

    rng = np.random.default_rng(0)
    e_a = _l2(rng.standard_normal(128).astype(np.float32))
    e_b = _l2(rng.standard_normal(128).astype(np.float32))

    cost_with_kw = gi._compute_assignment_cost(
        seg, profile, None, None, None, None,
        court_split_y=None,
        seg_learned_emb=e_a,
        profile_learned_emb=e_b,
    )

    assert cost_without_kw == cost_with_kw, (
        f"W=0 must ignore learned embeddings; got {cost_without_kw} vs {cost_with_kw}"
    )


def test_learned_cost_prefers_same_player(monkeypatch):
    """At W>0, same-player pair must have lower cost than teammate pair."""
    monkeypatch.setenv("WEIGHT_LEARNED_REID", "0.20")
    import rallycut.tracking.global_identity as gi

    gi = importlib.reload(gi)

    seg = _build_segment()
    profile = gi.PlayerProfile(player_id=1, team=0, histogram=None)

    rng = np.random.default_rng(7)
    e_self = _l2(rng.standard_normal(128).astype(np.float32))
    # Slightly perturbed — same-player post-swap
    e_self_noisy = _l2(e_self + 0.1 * rng.standard_normal(128).astype(np.float32))
    # Orthogonal-ish — teammate (anti-correlated foundation prior territory)
    e_team = _l2(rng.standard_normal(128).astype(np.float32))

    cost_self = gi._compute_assignment_cost(
        seg, profile, None, None, None, None,
        court_split_y=None,
        seg_learned_emb=e_self_noisy,
        profile_learned_emb=e_self,
    )
    cost_team = gi._compute_assignment_cost(
        seg, profile, None, None, None, None,
        court_split_y=None,
        seg_learned_emb=e_team,
        profile_learned_emb=e_self,
    )
    assert cost_self < cost_team, (
        f"same-player cost {cost_self} should be < teammate cost {cost_team}"
    )


def test_segment_embedding_below_min_frames_returns_none(monkeypatch):
    """< LEARNED_REID_MIN_FRAMES embeddings → segment embedding is None."""
    monkeypatch.setenv("WEIGHT_LEARNED_REID", "0.10")
    import rallycut.tracking.global_identity as gi

    gi = importlib.reload(gi)

    seg = _build_segment(n_frames=10)
    store = LearnedEmbeddingStore()
    rng = np.random.default_rng(1)
    # Only 2 embeddings — below LEARNED_REID_MIN_FRAMES (=5)
    for fn in (0, 1):
        store.add(seg.track_id, fn, _l2(rng.standard_normal(128).astype(np.float32)))

    emb = gi._get_segment_mean_embedding(seg, store)
    assert emb is None, "embedding below MIN_FRAMES must return None"


def test_extract_learned_embeddings_graceful_on_missing_checkpoint(monkeypatch):
    """Missing checkpoint → empty (0, 128) array, no crash."""
    import rallycut.tracking.reid_embeddings as rie

    # Point to a nonexistent path, clear the module-level cache.
    monkeypatch.setattr(rie, "HEAD_CHECKPOINT_PATH", rie.Path("/nonexistent/best.pt"))
    monkeypatch.setattr(rie, "_head_cache", {})
    monkeypatch.setattr(rie, "_head_load_warned", False)

    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (64, 48, 3), dtype=np.uint8) for _ in range(3)]
    out = rie.extract_learned_embeddings(crops, device="cpu")
    assert out.shape == (0, 128)
    assert out.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
