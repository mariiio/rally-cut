"""Unit tests for the joint rule-aware attribution v2 solver.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md
"""

from __future__ import annotations

import math

import pytest

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction
from rallycut.tracking.contact_detector import Contact
from rallycut.tracking.joint_attribution import (
    RallyState,
    _derive_state_after,
    _is_valid_candidate,
    _score_candidate,
)


def _a(action_type: ActionType, frame: int = 0) -> ClassifiedAction:
    """Minimal ClassifiedAction; player_track_id is unused by the state machine."""
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        player_track_id=0,
        court_side="near",
        confidence=0.9,
    )


class TestRallyStateMachine:
    """Truth table for _derive_state_after under R1-R5 semantics."""

    def test_initial_state_seeded_by_serving_team_0(self) -> None:
        """A SERVE by team 0 sets serving_team=0, count=1, expected_team=1 for next."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.SERVE), team_at_action=0, prior=prior)
        assert state.expected_team == 1   # net crossed; next contact is opposing team
        assert state.count_consecutive_same_team == 1  # opposing team's first contact starts at 1 (reset semantics)
        assert state.last_was_block is False
        assert state.serving_team == 0

    def test_initial_state_serving_team_seeds_when_prior_serving_team_none(self) -> None:
        """If serving_team is None pre-rally, first SERVE seeds it from team_at_action."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=None,
        )
        state = _derive_state_after(_a(ActionType.SERVE), team_at_action=1, prior=prior)
        assert state.serving_team == 1

    def test_receive_preserves_possession_and_increments_count(self) -> None:
        """RECEIVE keeps same team and increments count."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.RECEIVE), team_at_action=1, prior=prior)
        assert state.expected_team == 1
        assert state.count_consecutive_same_team == 2

    def test_attack_flips_possession_and_resets_count(self) -> None:
        """ATTACK is net-crossing; expected_team flips and count resets to 1."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=3,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.ATTACK), team_at_action=1, prior=prior)
        assert state.expected_team == 0  # ball crossed
        assert state.count_consecutive_same_team == 1
        assert state.last_was_block is False

    def test_block_does_not_flip_possession_and_resets_count(self) -> None:
        """BLOCK is a free pass: expected_team stays at the blocker's team; count resets to 0."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # The block is by the team that was the receiver of the prior attack.
        # Per R5, expected_team is now the blocker's team (cover follows).
        state = _derive_state_after(_a(ActionType.BLOCK), team_at_action=0, prior=prior)
        assert state.expected_team == 0  # cover by the blocking team
        assert state.count_consecutive_same_team == 0  # block itself doesn't count
        assert state.last_was_block is True

    def test_action_after_block_starts_counting_at_1(self) -> None:
        """First contact after a block (the cover) starts count at 1, not 2."""
        prior = RallyState(
            expected_team=0, count_consecutive_same_team=0,
            last_was_block=True, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.DIG), team_at_action=0, prior=prior)
        assert state.count_consecutive_same_team == 1
        assert state.last_was_block is False

    def test_unknown_action_passes_through(self) -> None:
        """UNKNOWN action: state carries over unchanged."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=2,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.UNKNOWN), team_at_action=0, prior=prior)
        assert state == prior  # no change


class TestIsValidCandidate:
    """Truth table for _is_valid_candidate: per-rule pass/fail cases."""

    def test_first_action_must_be_serve_by_serving_team(self) -> None:
        """R1: with serving_team known, the first action's team must match."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=0,
        )
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=0, prior=prior) is True
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=1, prior=prior) is False

    def test_first_action_serve_seeds_when_serving_team_none(self) -> None:
        """If serving_team is unset, any team is valid for the first SERVE."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=None,
        )
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=0, prior=prior) is True
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=1, prior=prior) is True

    def test_first_action_must_be_serve_not_other(self) -> None:
        """First action is SERVE — non-SERVE first actions are rejected."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=None,
        )
        assert _is_valid_candidate(ActionType.RECEIVE, candidate_team=1, prior=prior) is False
        assert _is_valid_candidate(ActionType.ATTACK, candidate_team=0, prior=prior) is False

    def test_r2_net_crossing_flips_required(self) -> None:
        """After SERVE/ATTACK, next non-BLOCK action must be on opposite team."""
        # State after a serve by team 0 (expected_team=1 set by state machine)
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # Valid: receive on opposite team
        assert _is_valid_candidate(ActionType.RECEIVE, candidate_team=1, prior=prior) is True
        # Invalid: receive on same team (R2 violation — serving team can't receive own serve)
        assert _is_valid_candidate(ActionType.RECEIVE, candidate_team=0, prior=prior) is False

    def test_r3_same_side_preserves(self) -> None:
        """After RECEIVE/SET/DIG, next non-BLOCK action must be on same team."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # Valid: set on same team
        assert _is_valid_candidate(ActionType.SET, candidate_team=1, prior=prior) is True
        # Invalid: set jumps team
        assert _is_valid_candidate(ActionType.SET, candidate_team=0, prior=prior) is False

    def test_r5_block_passes_through_team_constraint(self) -> None:
        """A BLOCK action can be on either team — the rule constraint is on
        the NEXT action's team (cover by blocker's team).

        However, in beach VB the block is always by the team OPPOSITE the
        attacker. The prior state's expected_team after an ATTACK is the
        receiving team; a BLOCK at this point should be by the receiving
        team. So the constraint actually does apply: block team == expected_team.
        """
        # State after ATTACK by team 0 — expected_team=1 (receiving)
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # Valid: block by receiving team 1
        assert _is_valid_candidate(ActionType.BLOCK, candidate_team=1, prior=prior) is True
        # Invalid: block by attacking team 0 (the attacker can't block their own attack)
        assert _is_valid_candidate(ActionType.BLOCK, candidate_team=0, prior=prior) is False

    def test_r4_max_3_same_side_blocks_4th_same_team(self) -> None:
        """After 3 same-team contacts, the next non-attack same-team action is rejected.

        In practice the 3rd same-team contact is usually an ATTACK which is net-crossing
        (R2 forces next to flip). R4 catches the abnormal case where the chain has
        3 same-side actions and a 4th would push the count to 4.
        """
        # After 3 consecutive same-team contacts
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=3,
            last_was_block=False, serving_team=0,
        )
        # The 4th same-team action is rejected by R4
        assert _is_valid_candidate(ActionType.SET, candidate_team=1, prior=prior) is False
        # But an attack by same team is still allowed (it crosses; R2 fires next)
        assert _is_valid_candidate(ActionType.ATTACK, candidate_team=1, prior=prior) is True

    def test_r5_after_block_cover_allowed_same_team(self) -> None:
        """After a BLOCK, the cover (same team as blocker) is allowed and
        the count resets — so the cover plus 3 more same-team contacts are legal."""
        prior = RallyState(
            expected_team=0, count_consecutive_same_team=0,
            last_was_block=True, serving_team=0,
        )
        # Cover by same team as blocker (team 0)
        assert _is_valid_candidate(ActionType.DIG, candidate_team=0, prior=prior) is True
        # Cover by opposing team (team 1) — invalid; ball is still on blocker's side
        assert _is_valid_candidate(ActionType.DIG, candidate_team=1, prior=prior) is False

    def test_unknown_always_valid(self) -> None:
        """UNKNOWN actions are passthrough — any team valid (state-machine skip)."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        assert _is_valid_candidate(ActionType.UNKNOWN, candidate_team=0, prior=prior) is True
        assert _is_valid_candidate(ActionType.UNKNOWN, candidate_team=1, prior=prior) is True


def _contact(
    frame: int = 0,
    candidates: list[tuple[int, float]] | None = None,
    court_side: str = "near",
) -> Contact:
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        direction_change_deg=60.0,
        player_track_id=(candidates[0][0] if candidates else -1),
        player_distance=(candidates[0][1] if candidates else float("inf")),
        player_candidates=candidates or [],
        court_side=court_side,
        is_validated=True,
    )


class TestScoreCandidate:
    """Truth table for the soft proximity scorer."""

    def test_returns_neg_log_distance_when_pid_in_candidates(self) -> None:
        """Score = -log(dist + ε) for a pid present in player_candidates."""
        contact = _contact(candidates=[(1, 0.05), (2, 0.10)])
        score_1 = _score_candidate(contact, candidate_pid=1)
        score_2 = _score_candidate(contact, candidate_pid=2)
        # Nearer pid scores higher (smaller distance → larger -log)
        assert score_1 > score_2
        # Sanity-check exact values
        assert score_1 == pytest.approx(-math.log(0.05 + 1e-3))
        assert score_2 == pytest.approx(-math.log(0.10 + 1e-3))

    def test_returns_neg_inf_when_pid_not_in_candidates(self) -> None:
        """A pid not in player_candidates → -inf score (effectively rejected)."""
        contact = _contact(candidates=[(1, 0.05), (2, 0.10)])
        assert _score_candidate(contact, candidate_pid=3) == float("-inf")

    def test_handles_zero_distance(self) -> None:
        """Zero distance is well-defined via the epsilon guard."""
        contact = _contact(candidates=[(1, 0.0)])
        score = _score_candidate(contact, candidate_pid=1)
        assert math.isfinite(score)
        assert score == pytest.approx(-math.log(1e-3))

    def test_empty_candidates_returns_neg_inf(self) -> None:
        """A contact with no candidates rejects everything."""
        contact = _contact(candidates=[])
        assert _score_candidate(contact, candidate_pid=1) == float("-inf")
