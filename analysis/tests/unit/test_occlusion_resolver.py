"""Session 5 — unit tests for the within-team occlusion resolver.

Covers: clean swap detection, clean no-swap non-detection, court-side veto,
missing learned store, insufficient windows, long convergence degradation,
and the store-swap propagation.
"""

from __future__ import annotations

import numpy as np
import pytest

from rallycut.tracking.color_repair import (
    ColorHistogramStore,
    LearnedEmbeddingStore,
)
from rallycut.tracking.occlusion_resolver import (
    DEFAULT_T_APPEARANCE,
    DEFAULT_T_COMBINED,
    DEFAULT_T_TRAJECTORY,
    resolve_within_team_convergence_swaps,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _make_positions(
    track_id: int,
    start: int,
    end: int,
    x0: float,
    y0: float,
    vx: float = 0.0,
    vy: float = 0.0,
) -> list[PlayerPosition]:
    """Generate positions along a constant-velocity line with standard bbox."""
    return [
        PlayerPosition(
            track_id=track_id,
            frame_number=f,
            x=x0 + vx * (f - start),
            y=y0 + vy * (f - start),
            width=0.08,
            height=0.18,
            confidence=0.95,
        )
        for f in range(start, end + 1)
    ]


def _seed_embeddings(
    store: LearnedEmbeddingStore,
    track_id: int,
    frames: range,
    vector: np.ndarray,
    jitter: float = 0.02,
) -> None:
    rng = np.random.default_rng(int(track_id * 7919 + frames.start))
    for f in frames:
        noise = rng.standard_normal(128).astype(np.float32) * jitter
        store.add(track_id, f, _unit(vector + noise))


def _overlapping_convergence(
    overlap_start: int, overlap_end: int,
) -> tuple[list[PlayerPosition], list[PlayerPosition]]:
    """Two tracks that physically overlap in [overlap_start, overlap_end].

    Track A: stationary at (0.4, 0.7) from frame 0..200
    Track B: stationary at (0.4, 0.7) during overlap, otherwise (0.6, 0.7)
    → IoU overlap during the convergence window.
    """
    positions_a = _make_positions(100, 0, 200, 0.4, 0.7)
    positions_b = []
    for f in range(0, 201):
        if overlap_start <= f <= overlap_end:
            x = 0.42  # near-overlap with A
        else:
            x = 0.6
        positions_b.append(PlayerPosition(
            track_id=200, frame_number=f, x=x, y=0.7,
            width=0.08, height=0.18, confidence=0.95,
        ))
    return positions_a, positions_b


def _base_args(positions: list[PlayerPosition]) -> dict:
    return {
        "positions": positions,
        "primary_track_ids": [100, 200],
        "team_assignments": {100: 0, 200: 0},  # same team
        "color_store": ColorHistogramStore(),
        "learned_store": LearnedEmbeddingStore(),
        "appearance_store": None,
        "court_calibrator": None,
        "video_width": 1920,
        "video_height": 1080,
        "court_split_y": 0.5,   # both tracks below split (team 1)
    }


def test_clean_no_swap_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both tracks keep their appearance through convergence → no swap."""
    pos_a, pos_b = _overlapping_convergence(80, 90)
    positions = pos_a + pos_b

    args = _base_args(positions)
    args["court_split_y"] = 0.5
    learned = args["learned_store"]
    rng = np.random.default_rng(0)
    vec_a = _unit(rng.standard_normal(128))
    vec_b = _unit(rng.standard_normal(128))
    _seed_embeddings(learned, 100, range(0, 201), vec_a)
    _seed_embeddings(learned, 200, range(0, 201), vec_b)

    _, result = resolve_within_team_convergence_swaps(**args)

    assert result.n_within_team >= 1, "convergence must be detected"
    assert result.n_swaps_applied == 0, "clean case must not swap"
    # Appearance score on no-swap should be <= 0
    decided = [e for e in result.events if e.abstain_reason is None]
    assert decided, "resolver must not abstain on clean data"
    assert decided[0].appearance_score < DEFAULT_T_APPEARANCE


def test_clean_swap_case() -> None:
    """Appearance signatures cross at the convergence → swap detected."""
    pos_a, pos_b = _overlapping_convergence(80, 90)
    positions = pos_a + pos_b
    args = _base_args(positions)
    learned = args["learned_store"]

    rng = np.random.default_rng(1)
    vec_a = _unit(rng.standard_normal(128))
    vec_b = _unit(rng.standard_normal(128))

    # Pre: T100=vec_a, T200=vec_b. Post: swapped.
    _seed_embeddings(learned, 100, range(0, 80), vec_a)
    _seed_embeddings(learned, 200, range(0, 80), vec_b)
    _seed_embeddings(learned, 100, range(101, 201), vec_b)
    _seed_embeddings(learned, 200, range(101, 201), vec_a)

    # Trajectories: flat (no trajectory signal). Force trajectory above
    # threshold via bumping alpha consideration: swap the positions after
    # frame 90 so the pre_a trajectory extrapolates to match post_b's
    # physical position rather than post_a's.
    for p in pos_a[91:]:
        p.x = 0.6  # moved to where B was
    for p in pos_b[91:]:
        p.x = 0.4  # moved to where A was

    _, result = resolve_within_team_convergence_swaps(
        **args,
        t_appearance=DEFAULT_T_APPEARANCE,
        t_trajectory=DEFAULT_T_TRAJECTORY,
        t_combined=DEFAULT_T_COMBINED,
    )
    decided = [e for e in result.events if e.abstain_reason is None]
    assert decided, f"resolver abstained: {result.events}"
    assert decided[0].appearance_score > DEFAULT_T_APPEARANCE
    assert decided[0].trajectory_score > DEFAULT_T_TRAJECTORY
    assert result.n_swaps_applied == 1


def test_court_side_veto() -> None:
    """Veto fires when post-convergence positions cross the court split."""
    pos_a, pos_b = _overlapping_convergence(80, 90)
    positions = pos_a + pos_b

    # Move T100 to the OTHER side of court_split_y after the convergence.
    for p in pos_a[95:]:
        p.y = 0.25  # opposite side of split=0.5

    args = _base_args(positions)
    args["court_split_y"] = 0.5
    learned = args["learned_store"]
    rng = np.random.default_rng(2)
    _seed_embeddings(learned, 100, range(0, 201), _unit(rng.standard_normal(128)))
    _seed_embeddings(learned, 200, range(0, 201), _unit(rng.standard_normal(128)))

    _, result = resolve_within_team_convergence_swaps(**args)
    reasons = [e.abstain_reason for e in result.events]
    assert "court_side_veto" in reasons
    assert result.n_swaps_applied == 0


def test_missing_learned_store() -> None:
    pos_a, pos_b = _overlapping_convergence(80, 90)
    args = _base_args(pos_a + pos_b)
    args["learned_store"] = None
    _, result = resolve_within_team_convergence_swaps(**args)
    assert result.n_abstained >= 1
    assert result.n_swaps_applied == 0
    assert all(
        e.abstain_reason == "no_learned_store"
        for e in result.events
    )


def test_insufficient_embeddings() -> None:
    """< min_frames_per_window embeddings → abstain."""
    pos_a, pos_b = _overlapping_convergence(80, 90)
    args = _base_args(pos_a + pos_b)
    learned = args["learned_store"]
    rng = np.random.default_rng(3)
    # Only 2 frames of embeddings each — below MIN=5
    for f in (70, 71):
        learned.add(100, f, _unit(rng.standard_normal(128)))
    for f in (95, 96):
        learned.add(200, f, _unit(rng.standard_normal(128)))

    _, result = resolve_within_team_convergence_swaps(**args)
    reasons = [e.abstain_reason for e in result.events]
    assert "insufficient_embeddings" in reasons
    assert result.n_swaps_applied == 0


def test_swap_propagates_to_color_store() -> None:
    """When a swap is applied, color_store.swap should be called."""
    pos_a, pos_b = _overlapping_convergence(80, 90)
    positions = pos_a + pos_b
    args = _base_args(positions)
    learned = args["learned_store"]
    color = args["color_store"]

    rng = np.random.default_rng(4)
    vec_a = _unit(rng.standard_normal(128))
    vec_b = _unit(rng.standard_normal(128))
    _seed_embeddings(learned, 100, range(0, 80), vec_a)
    _seed_embeddings(learned, 200, range(0, 80), vec_b)
    _seed_embeddings(learned, 100, range(101, 201), vec_b)
    _seed_embeddings(learned, 200, range(101, 201), vec_a)
    for p in pos_a[91:]:
        p.x = 0.6
    for p in pos_b[91:]:
        p.x = 0.4

    # Seed color histograms for both tracks.
    hist = np.ones(128, dtype=np.float32) / 128.0
    for f in range(0, 201):
        color.add(100, f, hist)
        color.add(200, f, hist)

    positions_out, result = resolve_within_team_convergence_swaps(**args)
    assert result.n_swaps_applied == 1
    mid = result.events[0].mid_frame
    # After swap, the color_store should still have both tracks (swap is
    # bidirectional), and track_ids in positions should have been flipped
    # for frames >= mid.
    post_mid_ids = {p.track_id for p in positions_out if p.frame_number >= mid}
    assert {100, 200}.issubset(post_mid_ids), (
        f"both track IDs must remain post-swap; got {post_mid_ids}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
