"""Tests for pid_invariants module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rallycut.tracking.pid_invariants import (
    check_i1_primary_set_size,
    check_i2_positions_in_primary,
    check_i3_action_attribution,
    check_i4_contact_attribution,
    check_i5_track_to_player_total,
    check_i6_team_assignments_total,
    check_i7_stats_canonical_pid,
    check_i8_team_partition_is_2v2,
    run_all,
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


class TestCheckI7StatsCanonicalPid:
    def test_clean_passes(self) -> None:
        violations = check_i7_stats_canonical_pid(
            rally_id="r1", mapped_track_ids=[1, 2, 3, 4, -1, 1, 2],
        )
        assert violations == []

    def test_unmapped_fails(self) -> None:
        # An unmapped raw track_id (e.g., 12) leaks through
        violations = check_i7_stats_canonical_pid(
            rally_id="r1", mapped_track_ids=[1, 2, 12, 4],
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-7"
        assert "12" in violations[0].detail

    def test_collision_shifted_fails(self) -> None:
        # 101 = collision-shifted unmapped ID
        violations = check_i7_stats_canonical_pid(
            rally_id="r1", mapped_track_ids=[1, 101, 3, 4],
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-7"
        assert "101" in violations[0].detail

    def test_empty_passes(self) -> None:
        violations = check_i7_stats_canonical_pid(rally_id="r1", mapped_track_ids=[])
        assert violations == []


class TestCheckI8TeamPartitionIs2v2:
    def test_clean_2v2_passes(self) -> None:
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "A", "2": "A", "3": "B", "4": "B"},
        )
        assert violations == []

    def test_side_switched_2v2_passes(self) -> None:
        # After a side-switch, A↔B labels flip but it's still a valid 2v2.
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "B", "2": "B", "3": "A", "4": "A"},
        )
        assert violations == []

    def test_cross_team_partition_passes(self) -> None:
        # Unusual partition (P1+P3 vs P2+P4) is still valid 2v2 — I-8 only
        # asserts the SHAPE, not which players are partnered.
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "A", "2": "B", "3": "A", "4": "B"},
        )
        assert violations == []

    def test_1v3_partition_fails(self) -> None:
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "A", "2": "B", "3": "B", "4": "B"},
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-8"
        assert violations[0].rally_id == "r1"
        assert "1A+3B" in violations[0].detail

    def test_3v1_partition_fails(self) -> None:
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "A", "2": "A", "3": "A", "4": "B"},
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-8"
        assert "3A+1B" in violations[0].detail

    def test_skips_when_fewer_than_4_primary(self) -> None:
        # Rally with size != 4 is caught by I-1; I-8 must skip to avoid
        # double-firing on an already-flagged condition.
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3],
            team_assignments={"1": "A", "2": "B", "3": "B"},
        )
        assert violations == []

    def test_skips_when_team_assignments_missing(self) -> None:
        # I-6 catches missing team_assignments. I-8 skips to avoid noise.
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments=None,
        )
        assert violations == []

    def test_skips_when_a_primary_lacks_label(self) -> None:
        # If even one primary track has no label, I-6 already fires.
        # I-8 skips because the partition is undefined for a missing label.
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "A", "2": "A", "3": "B"},  # 4 missing
        )
        assert violations == []

    def test_invalid_label_value_skips(self) -> None:
        # Non-A/B label is caught by I-6. I-8 doesn't double-flag.
        violations = check_i8_team_partition_is_2v2(
            rally_id="r1",
            primary_track_ids=[1, 2, 3, 4],
            team_assignments={"1": "A", "2": "A", "3": "B", "4": "X"},
        )
        assert violations == []


class TestRunAll:
    def _mock_conn(
        self,
        *,
        rallies: list[tuple],
        match_analysis: dict | None,
    ) -> MagicMock:
        """Build a mock connection that yields the given rally rows + video row."""
        cur = MagicMock()
        # Two execute calls happen: one for rallies, one for video.
        # fetchall returns rallies; fetchone returns (match_analysis_json,) tuple.
        cur.fetchall.return_value = rallies
        cur.fetchone.return_value = (match_analysis,)
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None

        conn = MagicMock()
        conn.cursor.return_value = cur
        conn.__enter__ = lambda self: self
        conn.__exit__ = lambda self, *a: None
        return conn

    def test_clean_video_returns_no_violations(self) -> None:
        rallies = [
            (
                "r1",
                [3, 7, 12, 15],  # primary_track_ids
                [{"trackId": 3, "frameNumber": 0}],  # positions_json
                {
                    "actions": [{"playerTrackId": 3, "action": "spike", "frame": 5}],
                    "teamAssignments": {"3": "A", "7": "A", "12": "B", "15": "B"},
                },  # actions_json
                [{"playerTrackId": 3, "frame": 5}],  # contacts_json
            ),
        ]
        match_analysis = {
            "rallies": [
                {
                    "rallyId": "r1",
                    "trackToPlayer": {"3": 1, "7": 2, "12": 3, "15": 4},
                }
            ]
        }
        conn = self._mock_conn(rallies=rallies, match_analysis=match_analysis)

        with patch("rallycut.tracking.pid_invariants.get_connection", return_value=conn):
            violations = run_all(video_id="v1")

        assert violations == []

    def test_dirty_video_aggregates_violations(self) -> None:
        rallies = [
            (
                "r1",
                [3, 7, 12],  # I-1: only 3 primary tracks
                [{"trackId": 99, "frameNumber": 0}],  # I-2: 99 not in primary
                {
                    "actions": [{"playerTrackId": 99, "action": "spike", "frame": 5}],  # I-3
                    "teamAssignments": {"3": "A", "7": "A"},  # I-6: 12 missing
                },
                [{"playerTrackId": 88, "frame": 5}],  # I-4: 88 not in primary
            ),
        ]
        match_analysis = {
            "rallies": [
                {
                    "rallyId": "r1",
                    "trackToPlayer": {"3": 1, "7": 2},  # I-5: 12 missing
                }
            ]
        }
        conn = self._mock_conn(rallies=rallies, match_analysis=match_analysis)

        with patch("rallycut.tracking.pid_invariants.get_connection", return_value=conn):
            violations = run_all(video_id="v1")

        invariants_seen = {v.invariant for v in violations}
        # Expect I-1, I-2, I-3, I-4, I-5, I-6 to all fire
        assert {"I-1", "I-2", "I-3", "I-4", "I-5", "I-6"}.issubset(invariants_seen)

    def test_snake_case_match_analysis_keys_are_accepted(self) -> None:
        """Some videos persist match_analysis with snake_case rally entry keys.

        Regression for Sub-1.1 follow-up: 073cb11b's match_analysis stores
        rally entries with `rally_id`/`track_to_player` instead of
        `rallyId`/`trackToPlayer`. The orchestrator must accept both forms,
        otherwise I-5 fires spuriously for every primary track on every rally.
        """
        rallies = [
            (
                "r1",
                [3, 7, 12, 15],
                [{"trackId": 3, "frameNumber": 0}],
                {
                    "actions": [{"playerTrackId": 3, "action": "spike", "frame": 5}],
                    "teamAssignments": {"3": "A", "7": "A", "12": "B", "15": "B"},
                },
                [{"playerTrackId": 3, "frame": 5}],
            ),
        ]
        match_analysis = {
            "rallies": [
                {
                    # snake_case form, no camelCase counterpart
                    "rally_id": "r1",
                    "track_to_player": {"3": 1, "7": 2, "12": 3, "15": 4},
                }
            ]
        }
        conn = self._mock_conn(rallies=rallies, match_analysis=match_analysis)

        with patch("rallycut.tracking.pid_invariants.get_connection", return_value=conn):
            violations = run_all(video_id="v1")

        # I-5 must NOT fire because the orchestrator accepts snake_case.
        assert all(v.invariant != "I-5" for v in violations)
        # And the video should be clean overall.
        assert violations == []
