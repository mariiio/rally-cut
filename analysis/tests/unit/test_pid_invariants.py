"""Tests for pid_invariants module."""

from __future__ import annotations

from rallycut.tracking.pid_invariants import (
    check_i1_primary_set_size,
    check_i2_positions_in_primary,
    check_i3_action_attribution,
)


class TestCheckI1PrimarySetSize:
    def test_clean_size_4_passes(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[3, 7, 12, 15])
        assert violations == []

    def test_clean_size_0_passes(self) -> None:
        # filter disabled is allowed
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[])
        assert violations == []

    def test_size_3_fails(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[3, 7, 12])
        assert len(violations) == 1
        assert violations[0].invariant == "I-1"
        assert violations[0].rally_id == "r1"
        assert "size 3" in violations[0].detail

    def test_size_5_fails(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r2", primary_track_ids=[3, 7, 12, 15, 22])
        assert len(violations) == 1
        assert violations[0].invariant == "I-1"


class TestCheckI2PositionsInPrimary:
    def test_clean_passes(self) -> None:
        positions = [
            {"trackId": 3, "frameNumber": 0},
            {"trackId": 7, "frameNumber": 0},
            {"trackId": 12, "frameNumber": 1},
        ]
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=positions,
        )
        assert violations == []

    def test_empty_passes(self) -> None:
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=[],
        )
        assert violations == []

    def test_none_passes(self) -> None:
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=None,
        )
        assert violations == []

    def test_non_primary_track_fails(self) -> None:
        positions = [
            {"trackId": 3, "frameNumber": 0},
            {"trackId": 99, "frameNumber": 0},  # non-primary
            {"trackId": 99, "frameNumber": 1},  # same offender, second sighting
        ]
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=positions,
        )
        assert len(violations) == 1  # one violation per offending trackId
        assert violations[0].invariant == "I-2"
        assert "99" in violations[0].detail


class TestCheckI3ActionAttribution:
    def test_clean_passes(self) -> None:
        actions = [
            {"playerTrackId": 3, "action": "spike", "frame": 10},
            {"playerTrackId": -1, "action": "serve", "frame": 0, "isSynthetic": True},
        ]
        violations = check_i3_action_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], actions_json=actions,
        )
        assert violations == []

    def test_none_passes(self) -> None:
        violations = check_i3_action_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], actions_json=None,
        )
        assert violations == []

    def test_non_primary_attribution_fails(self) -> None:
        actions = [
            {"playerTrackId": 99, "action": "set", "frame": 5},
            {"playerTrackId": 101, "action": "dig", "frame": 8},
        ]
        violations = check_i3_action_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], actions_json=actions,
        )
        assert len(violations) == 2
        assert all(v.invariant == "I-3" for v in violations)
        assert any("99" in v.detail for v in violations)
        assert any("101" in v.detail for v in violations)
