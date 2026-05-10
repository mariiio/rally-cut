"""Tests for coherence_invariants module."""

from __future__ import annotations

from rallycut.tracking.coherence_invariants import (
    check_c1_three_contact_rule,
    check_c2_alternating_possessions,
)


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
