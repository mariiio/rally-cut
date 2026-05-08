"""Tests for pid_invariants module."""

from __future__ import annotations

from rallycut.tracking.pid_invariants import (
    check_i1_primary_set_size,
    check_i2_positions_in_primary,
    check_i3_action_attribution,
    check_i4_contact_attribution,
    check_i5_track_to_player_total,
    check_i6_team_assignments_total,
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


class TestCheckI4ContactAttribution:
    def test_clean_passes(self) -> None:
        contacts = [
            {"playerTrackId": 7, "frame": 12},
            {"playerTrackId": -1, "frame": 20},
        ]
        violations = check_i4_contact_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], contacts_json=contacts,
        )
        assert violations == []

    def test_none_passes(self) -> None:
        violations = check_i4_contact_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], contacts_json=None,
        )
        assert violations == []

    def test_non_primary_contact_fails(self) -> None:
        contacts = [{"playerTrackId": 88, "frame": 30}]
        violations = check_i4_contact_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], contacts_json=contacts,
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-4"
        assert "88" in violations[0].detail


class TestCheckI5TrackToPlayerTotal:
    def test_clean_total_passes(self) -> None:
        # Note: trackToPlayer keys are str in JSON
        violations = check_i5_track_to_player_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            track_to_player={"3": 1, "7": 2, "12": 3, "15": 4},
        )
        assert violations == []

    def test_missing_primary_fails(self) -> None:
        violations = check_i5_track_to_player_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            track_to_player={"3": 1, "7": 2, "12": 3},  # 15 missing
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-5"
        assert "15" in violations[0].detail

    def test_pid_out_of_range_fails(self) -> None:
        violations = check_i5_track_to_player_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            track_to_player={"3": 1, "7": 2, "12": 3, "15": 7},  # 7 not in {1..4}
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-5"
        assert "pid=7" in violations[0].detail or "7" in violations[0].detail

    def test_empty_mapping_with_empty_primary_passes(self) -> None:
        violations = check_i5_track_to_player_total(
            rally_id="r1", primary_track_ids=[], track_to_player={},
        )
        assert violations == []


class TestCheckI6TeamAssignmentsTotal:
    def test_clean_total_passes(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            team_assignments={"3": "A", "7": "A", "12": "B", "15": "B"},
        )
        assert violations == []

    def test_missing_primary_fails(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            team_assignments={"3": "A", "7": "A", "12": "B"},
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-6"
        assert "15" in violations[0].detail

    def test_invalid_team_label_fails(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            team_assignments={"3": "A", "7": "A", "12": "B", "15": "X"},
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-6"
        assert "X" in violations[0].detail

    def test_empty_mapping_with_empty_primary_passes(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1", primary_track_ids=[], team_assignments={},
        )
        assert violations == []
