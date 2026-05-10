"""Tests for coherence_invariants module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rallycut.tracking.coherence_invariants import (
    check_c1_three_contact_rule,
    check_c2_alternating_possessions,
    check_c3_first_action_is_serve,
    run_all,
)
from rallycut.tracking.pid_invariants import Violation as PidViolation


def _action(frame: int, action: str, player_track_id: int) -> dict:
    return {"frame": frame, "action": action, "playerTrackId": player_track_id}


class TestCheckC1ThreeContactRule:
    def test_clean_sequence_passes(self) -> None:
        # Standard 3-contact possession: receive, set, attack — then opposing team digs.
        actions = [
            _action(100, "serve", 3),    # team A: serve
            _action(140, "receive", 1),  # team B: receive
            _action(170, "set", 2),      # team B: set
            _action(200, "attack", 1),   # team B: attack (3 contacts, ball crosses)
            _action(230, "dig", 4),      # team A: dig
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_4_consecutive_contacts_fails(self) -> None:
        # Team B has 4 consecutive contacts — illegal.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 2),      # B
            _action(200, "set", 1),      # B (3rd)
            _action(230, "attack", 2),   # B (4th — illegal)
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 1
        assert result[0].invariant == "C-1"
        assert result[0].rally_id == "r1"
        assert "team B" in result[0].detail
        assert "4 consecutive" in result[0].detail

    def test_zero_actions_skips(self) -> None:
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=[], team_assignments={},
        )
        assert result == []

    def test_one_action_skips(self) -> None:
        actions = [_action(100, "serve", 1)]
        team_assignments = {"1": "A"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_unresolvable_team_skips(self) -> None:
        # Action references player not in team_assignments — orchestrator should
        # have skipped this rally upstream, but defensive check still applies.
        actions = [
            _action(100, "serve", 99),   # 99 not in team_assignments
            _action(140, "receive", 1),
        ]
        team_assignments = {"1": "B"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        # Skip — undeterminable team
        assert result == []


class TestCheckC2AlternatingPossessions:
    def test_clean_alternating_passes(self) -> None:
        # Standard exchange: A serves, B receives/sets/attacks, A digs/sets/attacks.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 2),      # B
            _action(200, "attack", 1),   # B (ends possession)
            _action(230, "dig", 4),      # A
            _action(260, "set", 3),      # A
            _action(290, "attack", 4),   # A (ends possession)
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_same_team_after_attack_fails(self) -> None:
        # Team B attacks, then team B digs — illegal (possession should transfer).
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 2),      # B
            _action(200, "attack", 1),   # B (ends possession — ball crosses)
            _action(230, "dig", 2),      # B — should be A!
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) >= 1
        assert any(v.invariant == "C-2" for v in result)
        assert any("expected" in v.detail and "team A" in v.detail for v in result)

    def test_zero_actions_skips(self) -> None:
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=[], team_assignments={},
        )
        assert result == []

    def test_one_action_skips(self) -> None:
        result = check_c2_alternating_possessions(
            rally_id="r1",
            actions=[_action(100, "serve", 1)],
            team_assignments={"1": "A"},
        )
        assert result == []

    def test_serve_after_serve_no_violation(self) -> None:
        # Two serves on different teams — actually possession alternates here, no violation.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "serve", 1),    # B (different team, no violation)
        ]
        team_assignments = {"1": "B", "3": "A"}
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []


class TestCheckC3FirstActionIsServe:
    def test_clean_first_serve_passes(self) -> None:
        actions = [
            _action(100, "serve", 1),
            _action(140, "receive", 2),
        ]
        result = check_c3_first_action_is_serve(rally_id="r1", actions=actions)
        assert result == []

    def test_first_action_attack_fails(self) -> None:
        actions = [
            _action(100, "attack", 1),
            _action(140, "dig", 2),
        ]
        result = check_c3_first_action_is_serve(rally_id="r1", actions=actions)
        assert len(result) == 1
        assert result[0].invariant == "C-3"
        assert "attack" in result[0].detail

    def test_first_action_receive_fails(self) -> None:
        actions = [
            _action(100, "receive", 1),
        ]
        result = check_c3_first_action_is_serve(rally_id="r1", actions=actions)
        assert len(result) == 1
        assert result[0].invariant == "C-3"
        assert "receive" in result[0].detail

    def test_zero_actions_skips(self) -> None:
        result = check_c3_first_action_is_serve(rally_id="r1", actions=[])
        assert result == []


class TestRunAll:
    def _mock_db_conn(
        self,
        *,
        rally_rows: list[tuple],
    ) -> MagicMock:
        cur = MagicMock()
        cur.fetchall.return_value = rally_rows
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        conn = MagicMock()
        conn.cursor.return_value = cur
        conn.__enter__ = lambda self: self
        conn.__exit__ = lambda self, *a: None
        return conn

    def test_clean_video_returns_no_violations(self) -> None:
        # One rally with a clean alternating sequence.
        actions_json = {
            "actions": [
                _action(100, "serve", 3),
                _action(140, "receive", 1),
                _action(170, "set", 2),
                _action(200, "attack", 1),
                _action(230, "dig", 4),
            ],
            "teamAssignments": {"1": "B", "2": "B", "3": "A", "4": "A"},
        }
        rally_rows = [("r1", actions_json)]
        conn = self._mock_db_conn(rally_rows=rally_rows)
        with patch(
            "rallycut.tracking.coherence_invariants.get_connection", return_value=conn
        ), patch(
            "rallycut.tracking.coherence_invariants.pid_run_all", return_value=[]
        ):
            violations = run_all(video_id="v1")
        assert violations == []

    def test_dirty_video_aggregates_violations(self) -> None:
        # First action is `attack` (C-3 fires), and 4 consecutive same-team contacts (C-1 fires).
        actions_json = {
            "actions": [
                _action(100, "attack", 1),  # C-3: should be serve
                _action(140, "set", 2),     # B (1)
                _action(170, "set", 1),     # B (2)
                _action(200, "set", 2),     # B (3)
                _action(230, "attack", 1),  # B (4 — C-1 fires)
            ],
            "teamAssignments": {"1": "B", "2": "B"},
        }
        rally_rows = [("r1", actions_json)]
        conn = self._mock_db_conn(rally_rows=rally_rows)
        with patch(
            "rallycut.tracking.coherence_invariants.get_connection", return_value=conn
        ), patch(
            "rallycut.tracking.coherence_invariants.pid_run_all", return_value=[]
        ):
            violations = run_all(video_id="v1")
        invariants_seen = {v.invariant for v in violations}
        assert "C-1" in invariants_seen
        assert "C-3" in invariants_seen

    def test_skips_rally_with_upstream_i6_violation(self) -> None:
        # Rally has illegal sequence (would fire C-1) BUT also has I-6 violation
        # — orchestrator should skip it entirely.
        actions_json = {
            "actions": [
                _action(100, "serve", 1),
                _action(140, "set", 1),    # 4 consecutive A
                _action(170, "set", 1),
                _action(200, "set", 1),
                _action(230, "attack", 1),
            ],
            "teamAssignments": {"1": "A"},
        }
        rally_rows = [("r1", actions_json)]
        conn = self._mock_db_conn(rally_rows=rally_rows)
        upstream = [
            PidViolation(
                invariant="I-6", rally_id="r1",
                detail="primary track 2 missing from team_assignments",
            )
        ]
        with patch(
            "rallycut.tracking.coherence_invariants.get_connection", return_value=conn
        ), patch(
            "rallycut.tracking.coherence_invariants.pid_run_all", return_value=upstream
        ):
            violations = run_all(video_id="v1")
        assert violations == []  # Skipped due to upstream I-6
