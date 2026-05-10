"""Unit tests for contact_recovery: gap derivation from coherence violations."""

from __future__ import annotations

from rallycut.tracking.contact_recovery import (
    derive_gaps_from_actions,
)


def _action(frame: int, action: str, player_track_id: int) -> dict:
    return {
        "frame": frame,
        "action": action,
        "playerTrackId": player_track_id,
    }


def test_c2_gap_serve_then_same_team_set() -> None:
    """C-2: serve(B) then set(B) => gap (276, 306) expected team A."""
    actions = [
        _action(276, "serve", 1),
        _action(306, "set", 2),
        _action(398, "attack", 4),
    ]
    team_assignments = {"1": "B", "2": "B", "4": "A"}
    gaps = derive_gaps_from_actions(
        actions=actions,
        team_assignments=team_assignments,
        rally_start_frame=0,
    )
    assert any(
        g.rule == "C-2" and g.lo == 276 and g.hi == 306 and g.expected_team == "A"
        for g in gaps
    ), gaps


def test_c1_gap_four_consecutive_same_team() -> None:
    """C-1: 4 contacts by team B in a row => gap (3rd_frame, 4th_frame) expected team A."""
    actions = [
        _action(100, "receive", 1),
        _action(150, "set", 2),
        _action(220, "attack", 1),
        _action(280, "dig", 2),  # 4th consecutive team-B contact
    ]
    team_assignments = {"1": "B", "2": "B"}
    gaps = derive_gaps_from_actions(
        actions=actions,
        team_assignments=team_assignments,
        rally_start_frame=0,
    )
    assert any(
        g.rule == "C-1" and g.lo == 220 and g.hi == 280 and g.expected_team == "A"
        for g in gaps
    ), gaps


def test_c3_gap_first_action_not_serve() -> None:
    """C-3: first action is receive => gap (rally_start, first_frame) expected serve."""
    actions = [
        _action(80, "receive", 1),
        _action(140, "set", 2),
    ]
    team_assignments = {"1": "B", "2": "B"}
    gaps = derive_gaps_from_actions(
        actions=actions,
        team_assignments=team_assignments,
        rally_start_frame=0,
    )
    assert any(
        g.rule == "C-3" and g.lo == 0 and g.hi == 80 and g.expected_action == "serve"
        for g in gaps
    ), gaps


def test_no_gap_when_clean() -> None:
    """A clean rally produces zero gaps."""
    actions = [
        _action(50, "serve", 1),
        _action(110, "receive", 3),
        _action(160, "set", 4),
        _action(220, "attack", 3),
    ]
    team_assignments = {"1": "B", "3": "A", "4": "A"}
    gaps = derive_gaps_from_actions(
        actions=actions,
        team_assignments=team_assignments,
        rally_start_frame=0,
    )
    assert gaps == [], gaps
