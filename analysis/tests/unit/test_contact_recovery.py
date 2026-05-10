"""Unit tests for contact_recovery: gap derivation from coherence violations."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from rallycut.tracking.contact_detector import Contact
from rallycut.tracking.contact_recovery import (
    GATE_GBM_MIN,
    GATE_SEQ_TAU,
    Gap,
    RallyInputs,
    action_type_for_candidate,
    audit_violation_count_for_actions,
    derive_gaps_from_actions,
    filter_candidates_in_gap,
    load_rally_inputs,
    pick_best_candidate,
)


def _action(frame: int, action: str, player_track_id: int) -> dict[str, int | str]:
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


# fb7f9c23 in 073cb11b is the working example; integration data must exist.
FB7F9C23 = "fb7f9c23-3544-48bd-910d-10a8f12fd594"


@pytest.mark.integration
def test_load_rally_inputs_for_fb7f9c23() -> None:
    inputs = load_rally_inputs(FB7F9C23)
    assert isinstance(inputs, RallyInputs)
    assert inputs.rally_id == FB7F9C23
    # Ball positions span the rally; expect ≥ 200 entries.
    assert len(inputs.ball_positions) >= 200, len(inputs.ball_positions)
    # Players tracked through the rally — expect ≥ 4 unique track IDs.
    track_ids = {p.track_id for p in inputs.player_positions}
    assert len(track_ids) >= 4, track_ids
    # teamAssignments has the four primary tracks.
    assert set(["1", "2", "4"]).issubset(set(inputs.team_assignments_str.keys()))
    # Frame count and fps positive.
    assert inputs.frame_count > 0
    assert inputs.fps > 0


def _candidate(
    *,
    frame: int,
    conf: float,
    track_id: int,
    court_side: str = "near",
) -> Contact:
    """Helper: create a Contact for testing."""
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.4,
        velocity=0.05,
        direction_change_deg=30.0,
        player_track_id=track_id,
        player_distance=0.05,
        player_candidates=[(track_id, 0.05)],
        court_side=court_side,
        is_at_net=False,
        is_validated=True,
        confidence=conf,
        arc_fit_residual=0.01,
    )


def test_filter_drops_outside_gap() -> None:
    """filter_candidates_in_gap drops contacts outside the gap window."""
    contacts = [
        _candidate(frame=200, conf=0.30, track_id=1),
        _candidate(frame=290, conf=0.30, track_id=3),  # in gap
        _candidate(frame=350, conf=0.30, track_id=1),
    ]
    seq_probs = np.zeros((7, 500))
    seq_probs[2, 290] = 0.95  # non-bg peak in gap
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    out = filter_candidates_in_gap(
        contacts=contacts,
        gap=gap,
        sequence_probs=seq_probs,
        team_assignments_str={"1": "B", "3": "A"},
        ball_positions={290: 1.0, 200: 1.0, 350: 1.0},
        existing_action_frames=[],
    )
    assert [c.frame for c in out] == [290], out


def test_filter_drops_wrong_team() -> None:
    """filter_candidates_in_gap drops contacts from wrong team."""
    contacts = [_candidate(frame=290, conf=0.30, track_id=1)]
    seq_probs = np.zeros((7, 500))
    seq_probs[2, 290] = 0.95
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    out = filter_candidates_in_gap(
        contacts=contacts,
        gap=gap,
        sequence_probs=seq_probs,
        team_assignments_str={"1": "B"},  # track 1 is B; expected A
        ball_positions={290: 1.0},
        existing_action_frames=[],
    )
    assert out == []


def test_filter_drops_low_seq() -> None:
    """filter_candidates_in_gap drops contacts below seq tau."""
    contacts = [_candidate(frame=290, conf=0.30, track_id=3)]
    seq_probs = np.zeros((7, 500))
    seq_probs[2, 290] = GATE_SEQ_TAU - 0.05  # below tau
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    out = filter_candidates_in_gap(
        contacts=contacts,
        gap=gap,
        sequence_probs=seq_probs,
        team_assignments_str={"3": "A"},
        ball_positions={290: 1.0},
        existing_action_frames=[],
    )
    assert out == []


def test_filter_drops_low_gbm() -> None:
    """filter_candidates_in_gap drops contacts below GBM minimum."""
    contacts = [_candidate(frame=290, conf=GATE_GBM_MIN - 0.01, track_id=3)]
    seq_probs = np.zeros((7, 500))
    seq_probs[2, 290] = 0.95
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    out = filter_candidates_in_gap(
        contacts=contacts,
        gap=gap,
        sequence_probs=seq_probs,
        team_assignments_str={"3": "A"},
        ball_positions={290: 1.0},
        existing_action_frames=[],
    )
    assert out == []


def test_filter_drops_duplicate_of_existing_action() -> None:
    """filter_candidates_in_gap drops candidates near existing actions."""
    contacts = [_candidate(frame=290, conf=0.30, track_id=3)]
    seq_probs = np.zeros((7, 500))
    seq_probs[2, 290] = 0.95
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    out = filter_candidates_in_gap(
        contacts=contacts,
        gap=gap,
        sequence_probs=seq_probs,
        team_assignments_str={"3": "A"},
        ball_positions={290: 1.0},
        existing_action_frames=[295],  # within ±15
    )
    assert out == []


def test_pick_best_prefers_highest_gbm() -> None:
    """pick_best_candidate returns the contact with highest GBM confidence."""
    contacts = [
        _candidate(frame=285, conf=0.30, track_id=3),
        _candidate(frame=295, conf=0.42, track_id=3),
    ]
    best = pick_best_candidate(contacts)
    assert best is not None and best.frame == 295


def test_action_type_uses_seq_argmax_when_not_c3() -> None:
    """action_type_for_candidate uses seq argmax when not C-3."""
    seq_probs = np.zeros((7, 400))
    # ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]
    # Index in seq_probs (with bg at 0): receive == 2.
    seq_probs[2, 290] = 0.9
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    assert action_type_for_candidate(frame=290, gap=gap, sequence_probs=seq_probs) == "receive"


def test_action_type_forces_serve_for_c3() -> None:
    """action_type_for_candidate forces serve for C-3."""
    seq_probs = np.zeros((7, 400))
    seq_probs[3, 90] = 0.8  # set is the argmax
    gap = Gap(rule="C-3", lo=0, hi=80, expected_team=None, expected_action="serve")
    assert action_type_for_candidate(frame=90, gap=gap, sequence_probs=seq_probs) == "serve"


def test_action_type_rejects_seq_argmax_serve_in_non_c3_gap() -> None:
    """action_type_for_candidate rejects serve argmax outside C-3."""
    seq_probs = np.zeros((7, 400))
    seq_probs[1, 290] = 0.9  # serve argmax
    gap = Gap(rule="C-2", lo=276, hi=306, expected_team="A", expected_action=None)
    assert action_type_for_candidate(frame=290, gap=gap, sequence_probs=seq_probs) is None


def test_audit_count_decreases_after_inserting_recovered_receive() -> None:
    """audit_violation_count_for_actions reflects recovery impact."""
    # Reproduce the fb7f9c23 shape: serve(B) -> set(B) -> attack(A).
    actions = [
        {"frame": 276, "action": "serve", "playerTrackId": 1},
        {"frame": 306, "action": "set", "playerTrackId": 2},
        {"frame": 398, "action": "attack", "playerTrackId": 4},
    ]
    team_assignments = {"1": "B", "2": "B", "4": "A"}
    before = audit_violation_count_for_actions(
        actions=actions, team_assignments=team_assignments, rally_start_frame=0,
    )
    # Inject the missing team-A receive.
    after_actions = sorted(
        actions + [{"frame": 290, "action": "receive", "playerTrackId": 4}],
        key=lambda a: cast(int, a["frame"]),
    )
    after = audit_violation_count_for_actions(
        actions=after_actions, team_assignments=team_assignments, rally_start_frame=0,
    )
    assert after < before, (before, after)
