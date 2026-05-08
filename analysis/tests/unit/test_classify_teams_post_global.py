"""Test classify_teams precomputed-passthrough under simulated global-identity rewrite.

Regression test for PID invariant I-6: after optimize_global_identity rewrites
track_ids on positions, classify_teams called with the OLD team_assignments as
precomputed_assignments must produce a dict whose keys match the new track set,
preserving labels for surviving tracks and classifying new tracks via median-Y.
"""

from __future__ import annotations

from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition


def _make_position(frame: int, track_id: int, y: float) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=0.5,
        y=y,
        width=0.05,
        height=0.10,
        confidence=0.9,
    )


class TestClassifyTeamsPostGlobalRewrite:
    def test_rewrite_track_id_carries_label_for_survivors_and_classifies_new(self) -> None:
        # Pre-global track set: {1, 2, 3, 6}. team_assignments built BEFORE rewrite.
        pre_global_team_assignments = {1: 0, 2: 1, 3: 0, 6: 1}

        # Simulate optimize_global_identity rewriting all track_id=6 entries to track_id=4
        # in the position list. Tracks {1, 2, 3} survive untouched.
        # split_y = 0.5 (court middle); near team has higher y, far team has lower y.
        positions: list[PlayerPosition] = []
        for frame in range(60):
            positions.append(_make_position(frame, track_id=1, y=0.7))   # near (team 0)
            positions.append(_make_position(frame, track_id=2, y=0.3))   # far  (team 1)
            positions.append(_make_position(frame, track_id=3, y=0.8))   # near (team 0)
            positions.append(_make_position(frame, track_id=4, y=0.2))   # far  (team 1) — was 6 pre-rewrite

        result = classify_teams(
            positions,
            court_split_y=0.5,
            precomputed_assignments=pre_global_team_assignments,
        )

        # Keys match the post-global track set (no stale 6, includes new 4).
        assert set(result.keys()) == {1, 2, 3, 4}

        # Surviving tracks keep their pre-global labels via the precomputed branch.
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == 0

        # New track (4) is classified via median-Y fallback (not in precomputed dict).
        # y=0.2 with split_y=0.5 → median_y < split_y → far team (1).
        assert result[4] == 1

    def test_no_rewrite_is_idempotent(self) -> None:
        # If global_identity rewrote nothing, every track_id is in the old dict
        # → precomputed branch covers all → result equals input dict.
        team_assignments = {1: 0, 2: 1, 3: 0, 4: 1}
        positions: list[PlayerPosition] = []
        for frame in range(60):
            positions.append(_make_position(frame, track_id=1, y=0.7))
            positions.append(_make_position(frame, track_id=2, y=0.3))
            positions.append(_make_position(frame, track_id=3, y=0.8))
            positions.append(_make_position(frame, track_id=4, y=0.2))

        result = classify_teams(
            positions,
            court_split_y=0.5,
            precomputed_assignments=team_assignments,
        )

        assert result == team_assignments

    def test_stale_keys_in_precomputed_are_dropped(self) -> None:
        # Old dict has phantom key (5) that doesn't exist in post-global positions.
        # classify_teams iterates over track_positions (built from positions),
        # so the phantom is silently dropped.
        pre_global_team_assignments = {1: 0, 2: 1, 5: 0}  # 5 is phantom
        positions: list[PlayerPosition] = []
        for frame in range(60):
            positions.append(_make_position(frame, track_id=1, y=0.7))
            positions.append(_make_position(frame, track_id=2, y=0.3))

        result = classify_teams(
            positions,
            court_split_y=0.5,
            precomputed_assignments=pre_global_team_assignments,
        )

        assert set(result.keys()) == {1, 2}
        assert 5 not in result
