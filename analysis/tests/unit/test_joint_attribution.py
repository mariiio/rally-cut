"""Unit tests for the joint rule-aware attribution v2 solver.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md
"""

from __future__ import annotations

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction
from rallycut.tracking.joint_attribution import (
    RallyState,
    _derive_state_after,
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
