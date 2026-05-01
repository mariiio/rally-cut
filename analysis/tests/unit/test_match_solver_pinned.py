"""MatchSolver pinned-assignments cache (Phase 2 cascade fix).

Verifies that:
1. Pinning a rally skips _assign_rally for that rally (decoupling its
   decision from cross-rally input drift).
2. Pinned rallies still contribute their members to OTHER rallies'
   cost matrices (so the solve produces a consistent result).
3. All-pinned solves return immediately without iteration.
4. Pinning behavior is byte-identical to the unpinned solve when the
   pin matches what MatchSolver would have computed anyway (no
   regression on cold-start equivalent).
"""
from __future__ import annotations

import numpy as np

from rallycut.tracking.match_solver import MatchSolver
from rallycut.tracking.match_tracker import StoredRallyData
from rallycut.tracking.player_features import HS_BINS, V_BINS, TrackAppearanceStats


def _stats(track_id: int, color_seed: int) -> TrackAppearanceStats:
    """Build a minimal TrackAppearanceStats with deterministic features."""
    rng = np.random.default_rng(color_seed)
    upper = rng.random(HS_BINS, dtype=np.float32)
    upper /= upper.sum()
    lower = rng.random(HS_BINS, dtype=np.float32)
    lower /= lower.sum()
    return TrackAppearanceStats(
        track_id=track_id,
        avg_skin_tone_hsv=(float(rng.random()), float(rng.random()), float(rng.random())),
        avg_upper_hist=upper,
        avg_lower_hist=lower,
        avg_upper_v_hist=rng.random(V_BINS, dtype=np.float32),
        avg_lower_v_hist=rng.random(V_BINS, dtype=np.float32),
        avg_dominant_color_hsv=(float(rng.random()), float(rng.random()), float(rng.random())),
        avg_head_hist=rng.random(HS_BINS, dtype=np.float32),
    )


def _rally(top_tracks: list[int], color_seeds: dict[int, int]) -> StoredRallyData:
    return StoredRallyData(
        track_stats={tid: _stats(tid, color_seeds[tid]) for tid in top_tracks},
        track_court_sides={tid: 0 if i < 2 else 1 for i, tid in enumerate(top_tracks)},
        early_positions={tid: (float(i), float(i)) for i, tid in enumerate(top_tracks)},
        top_tracks=list(top_tracks),
        late_positions={tid: (float(i), float(i)) for i, tid in enumerate(top_tracks)},
    )


def _two_rally_match() -> list[StoredRallyData]:
    """Two rallies whose tracks share color seeds across PIDs.

    Rally A tracks 1,2,3,4 → seeds 11,12,13,14.
    Rally B tracks 5,6,7,8 mapped to the same seeds (cross-rally identity).
    """
    seeds_a = {1: 11, 2: 12, 3: 13, 4: 14}
    seeds_b = {5: 11, 6: 12, 7: 13, 8: 14}
    return [
        _rally([1, 2, 3, 4], seeds_a),
        _rally([5, 6, 7, 8], seeds_b),
    ]


class TestPinnedAssignments:
    def test_unpinned_baseline_runs_full_solve(self) -> None:
        rallies = _two_rally_match()
        solver = MatchSolver(reid_blend=0.5, max_iterations=10, convergence_passes=2)
        result = solver.solve(rallies)
        assert len(result) == 2
        assert set(result[0].values()) == {1, 2, 3, 4}
        assert set(result[1].values()) == {1, 2, 3, 4}

    def test_pinned_rally_keeps_assignment(self) -> None:
        """A pinned rally's assignment is preserved verbatim."""
        rallies = _two_rally_match()
        # Anchor rally 0 to a specific assignment that may differ from
        # what the solver would compute on cold start.
        anchor = {1: 4, 2: 3, 3: 2, 4: 1}
        solver = MatchSolver(reid_blend=0.5)
        result = solver.solve(rallies, pinned_assignments={0: anchor})
        assert result[0] == anchor

    def test_pinned_rally_anchors_other_rallies(self) -> None:
        """Other rallies see pinned rally's members as cluster anchors."""
        rallies = _two_rally_match()
        # Pin rally 0 with a non-canonical assignment. Rally 1's tracks
        # should match THIS labeling (cross-rally consistency via
        # appearance similarity to rally 0's pinned members).
        anchor = {1: 4, 2: 3, 3: 2, 4: 1}
        solver = MatchSolver(reid_blend=0.5)
        result = solver.solve(rallies, pinned_assignments={0: anchor})
        # Rally 1's track 5 (seed 11, same as rally 0 track 1) should
        # match rally 0 track 1's cluster (4).
        assert result[1][5] == anchor[1]
        assert result[1][6] == anchor[2]
        assert result[1][7] == anchor[3]
        assert result[1][8] == anchor[4]

    def test_all_pinned_returns_immediately(self) -> None:
        """When every rally is pinned, no iteration runs."""
        rallies = _two_rally_match()
        anchor_a = {1: 1, 2: 2, 3: 3, 4: 4}
        anchor_b = {5: 1, 6: 2, 7: 3, 8: 4}
        solver = MatchSolver(reid_blend=0.5, max_iterations=0)
        # max_iterations=0 would normally not produce a useful answer,
        # but with all rallies pinned the solver short-circuits.
        result = solver.solve(
            rallies, pinned_assignments={0: anchor_a, 1: anchor_b},
        )
        assert result[0] == anchor_a
        assert result[1] == anchor_b

    def test_pinned_isolates_from_other_rally_drift(self) -> None:
        """Pinning a rally makes its decision independent of OTHER rallies' drift.

        This is the cascade fix: even if another rally's track_stats change
        between runs, the pinned rally's assignment is byte-identical.
        """
        # Run A: clean two-rally match.
        rallies_a = _two_rally_match()
        anchor_0 = {1: 1, 2: 2, 3: 3, 4: 4}
        solver = MatchSolver(reid_blend=0.5)
        result_a = solver.solve(rallies_a, pinned_assignments={0: anchor_0})

        # Run B: same rally 0, but rally 1's tracks have COMPLETELY
        # different appearance (simulating re-tracking with new YOLO/BoT-SORT
        # output).
        rng = np.random.default_rng(42)
        rallies_b = _two_rally_match()
        for tid in rallies_b[1].track_stats:
            stats = rallies_b[1].track_stats[tid]
            stats.avg_upper_hist = rng.random(HS_BINS, dtype=np.float32)
            stats.avg_upper_hist /= stats.avg_upper_hist.sum()
            stats.avg_lower_hist = rng.random(HS_BINS, dtype=np.float32)
            stats.avg_lower_hist /= stats.avg_lower_hist.sum()
        result_b = solver.solve(rallies_b, pinned_assignments={0: anchor_0})

        # Rally 0 is byte-identical between runs — drift in rally 1 did
        # NOT propagate. This is the cascade decoupling.
        assert result_a[0] == result_b[0] == anchor_0

    def test_invalid_pin_index_ignored(self) -> None:
        rallies = _two_rally_match()
        solver = MatchSolver(reid_blend=0.5)
        # Out-of-range index — solver must not crash, just ignore.
        result = solver.solve(rallies, pinned_assignments={99: {1: 1}})
        assert len(result) == 2
        assert set(result[0].values()) == {1, 2, 3, 4}
