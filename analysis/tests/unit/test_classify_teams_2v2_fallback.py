"""Test classify_teams forces 2v2 partition for any 4-track input.

Regression test for the producer fix that closes I-8 violations on new
tracking runs. The existing fallback at player_filter.py:721 forced 2v2
ONLY when all tracks landed on the same side (0v4). The fix extends it
to fire on any non-2v2 partition (1v3, 3v1, 0v4, 4v0).
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


def _build_positions(track_ys: dict[int, float], n_frames: int = 60) -> list[PlayerPosition]:
    """Build n_frames of positions per track at the given y."""
    out: list[PlayerPosition] = []
    for tid, y in track_ys.items():
        for f in range(n_frames):
            out.append(_make_position(f, tid, y))
    return out


def _team_counts(team_assignments: dict[int, int]) -> tuple[int, int]:
    """Return (near_count, far_count) where near=0, far=1."""
    near = sum(1 for v in team_assignments.values() if v == 0)
    far = sum(1 for v in team_assignments.values() if v == 1)
    return near, far


class TestClassifyTeams2v2Fallback:
    def test_clean_2v2_unchanged(self) -> None:
        # Two tracks above split (near, team 0), two below (far, team 1).
        # Y-classification per track produces 2v2 directly; fallback shouldn't trigger.
        positions = _build_positions({1: 0.8, 2: 0.7, 3: 0.3, 4: 0.2})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # high y → near
        assert result[2] == 0
        assert result[3] == 1  # low y → far
        assert result[4] == 1

    def test_1a3b_input_forced_to_2v2(self) -> None:
        # Three tracks below the split, one above. Per-track Y classification
        # produces 1 near + 3 far. Fallback must reshuffle to 2v2.
        positions = _build_positions({1: 0.8, 2: 0.4, 3: 0.3, 4: 0.2})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        # Median-index split sorts by Y. Top-2 (highest y) = near (1, 2). Bottom-2 = far (3, 4).
        assert result[1] == 0  # highest y
        assert result[2] == 0  # second highest
        assert result[3] == 1
        assert result[4] == 1

    def test_3a1b_input_forced_to_2v2(self) -> None:
        # Three tracks above the split, one below. Per-track Y classification
        # produces 3 near + 1 far. Fallback must reshuffle to 2v2.
        positions = _build_positions({1: 0.9, 2: 0.8, 3: 0.7, 4: 0.2})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # highest y → near
        assert result[2] == 0  # second highest → near
        assert result[3] == 1  # third highest → far (median-index split)
        assert result[4] == 1  # lowest → far

    def test_0a4b_input_forced_to_2v2_existing_behavior(self) -> None:
        # All four tracks below split — existing 0v4 fallback case still works.
        positions = _build_positions({1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # highest among the 4 → near
        assert result[2] == 0
        assert result[3] == 1
        assert result[4] == 1

    def test_4a0b_input_forced_to_2v2_existing_behavior(self) -> None:
        # All four tracks above split — existing 4v0 fallback case still works.
        positions = _build_positions({1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # highest → near
        assert result[2] == 0
        assert result[3] == 1
        assert result[4] == 1

    def test_three_tracks_no_fallback(self) -> None:
        # 3 tracks: fallback requires len == 4. With 3 tracks, per-track Y
        # classification stands. 1 above, 2 below → 1 near + 2 far (NOT 2v2,
        # but acceptable because the rally is structurally I-1 territory).
        positions = _build_positions({1: 0.8, 2: 0.4, 3: 0.3})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        # Whatever per-track Y produces — fallback does NOT fire.
        assert near + far == 3
        assert result[1] == 0  # high y → near
        assert result[2] == 1  # low y → far
        assert result[3] == 1

    def test_precomputed_assignments_takes_precedence(self) -> None:
        # When precomputed_assignments is provided, those labels are used
        # directly. If precomputed produces 2v2, fallback doesn't fire.
        positions = _build_positions({1: 0.8, 2: 0.4, 3: 0.3, 4: 0.2})
        precomputed = {1: 0, 2: 0, 3: 1, 4: 1}  # forces 2v2 partition not matching Y
        result = classify_teams(
            positions, court_split_y=0.5, precomputed_assignments=precomputed,
        )
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        # Precomputed labels survive — fallback only fires for non-2v2.
        assert result == precomputed
