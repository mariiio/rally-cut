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
    _beam_search,
    _derive_state_after,
    _is_valid_candidate,
    _score_candidate,
    joint_attribute,
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


class TestBeamSearch:
    """End-to-end tests of the beam search returning a per-action pid list."""

    def test_canonical_bug_pattern_returns_rule_valid_assignment(self) -> None:
        """The user-quoted bug: serve by team B, receive currently attributed to team B.
        Beam search should pick the team-A receiver (which is in the candidate list)
        because that's the only R2-compliant assignment."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            # The wrong-team nearest candidate (4) is first; the correct candidate (1) is second.
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}  # teams 0=near=A, 1=far=B

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1,  # team B serves
            beam_width=50,
        )
        # Serve by track 3 (team B); receive by track 1 (team A, the correct team).
        assert assignment == [3, 1]

    def test_returns_none_when_no_valid_assignment_exists(self) -> None:
        """If every candidate path violates a rule, return None for the fallback."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            # Only team-B candidates exist for the receive — violates R2.
            _contact(frame=90, candidates=[(3, 0.05), (4, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        assert assignment is None  # fallback signal

    def test_picks_proximity_best_within_rule_valid_space(self) -> None:
        """Among rule-valid assignments, the highest soft-score (closest) wins."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04), (4, 0.20)]),  # both team B; 3 is closer
            _contact(frame=90, candidates=[(1, 0.05), (2, 0.10)]),  # both team A; 1 is closer
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        # Both serve candidates are team B (rule-valid); 3 is closer.
        # Both receive candidates are team A (rule-valid); 1 is closer.
        assert assignment == [3, 1]

    def test_unknown_action_passes_through_any_pid(self) -> None:
        """UNKNOWN actions accept any candidate; beam search picks the proximity-best."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.UNKNOWN, frame=70),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=70, candidates=[(2, 0.05), (3, 0.08)]),  # UNKNOWN: either valid
            _contact(frame=90, candidates=[(1, 0.05), (4, 0.07)]),  # receive: must be team A
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        # UNKNOWN: 2 is closer → picked.
        # Receive: 1 (team A) is valid; 4 (team B) violates R2.
        assert assignment == [3, 2, 1]

    def test_synthetic_first_action_passes_through(self) -> None:
        """A synthetic first SERVE: keeps existing pid, no beam expansion.
        Subsequent non-synthetic actions still get rule-checked from the
        synthetic action's seeded state."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE, frame=50, ball_x=0.5, ball_y=0.5,
                velocity=0.02, player_track_id=3, court_side="far", confidence=0.95,
                is_synthetic=True,
            ),
            _a(ActionType.RECEIVE, frame=90),
        ]
        # NO contact for the synthetic serve frame 50 — only frame 90 has a contact.
        contacts = [
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=None, beam_width=50,
        )
        # Synthetic serve track 3 (team B) passed through; receive must be team A (rule).
        assert assignment == [3, 1]

    def test_synthetic_first_action_with_no_team_skips_state(self) -> None:
        """A synthetic SERVE with pl_pid=-1 has no team — state passes unchanged
        and the next action becomes effectively the rally seed."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE, frame=50, ball_x=0.5, ball_y=0.5,
                velocity=0.02, player_track_id=-1, court_side="far", confidence=0.95,
                is_synthetic=True,
            ),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=90, candidates=[(1, 0.05), (4, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=None, beam_width=50,
        )
        # Synthetic placeholder; receive has no prior team constraint.
        # Either candidate is rule-valid as the "first non-synthetic action".
        # Beam picks the proximity-best (track 1).
        assert assignment == [-1, 1]

    def test_block_and_cover_legal_sequence(self) -> None:
        """A block followed by a same-team cover is legal under R5.

        Sequence: SERVE(B) → RECEIVE(A) → SET(A) → ATTACK(A) → BLOCK(B) → DIG(B) → SET(B) → ATTACK(B)
        """
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
            _a(ActionType.SET, frame=130),
            _a(ActionType.ATTACK, frame=170),
            _a(ActionType.BLOCK, frame=180),
            _a(ActionType.DIG, frame=200),
            _a(ActionType.SET, frame=240),
            _a(ActionType.ATTACK, frame=280),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=90, candidates=[(1, 0.05)]),
            _contact(frame=130, candidates=[(2, 0.06)]),
            _contact(frame=170, candidates=[(1, 0.07)]),
            _contact(frame=180, candidates=[(3, 0.05)]),  # block by team B
            _contact(frame=200, candidates=[(4, 0.06)]),  # cover by team B (same as blocker)
            _contact(frame=240, candidates=[(3, 0.05)]),  # set by team B
            _contact(frame=280, candidates=[(4, 0.07)]),  # attack by team B
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        assert assignment == [3, 1, 2, 1, 3, 4, 3, 4]


class TestJointAttribute:
    """End-to-end public entry: rewrites action.player_track_id from the beam result.
    Falls back to the input unchanged if no valid assignment exists."""

    def test_overrides_cross_team_receive_on_canonical_bug(self) -> None:
        """The user-quoted bug: receive currently attributed to team B; v2
        rewrites it to the team-A candidate that satisfies R2."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=50, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=3, court_side="far", confidence=0.95,
            ),
            ClassifiedAction(
                action_type=ActionType.RECEIVE,
                frame=90, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=4, court_side="near", confidence=0.9,
            ),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        result = joint_attribute(actions, contacts, team_assignments, serving_team=1)
        assert result[0].player_track_id == 3
        assert result[1].player_track_id == 1  # was 4, now 1

    def test_fallback_preserves_input_when_no_valid_assignment(self) -> None:
        """When the beam empties, return the input unchanged."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=50, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=3, court_side="far", confidence=0.95,
            ),
            ClassifiedAction(
                action_type=ActionType.RECEIVE,
                frame=90, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=4, court_side="near", confidence=0.9,
            ),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            # Both candidates on serving team — no rule-valid receive exists.
            _contact(frame=90, candidates=[(3, 0.05), (4, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        result = joint_attribute(actions, contacts, team_assignments, serving_team=1)
        # Input preserved
        assert result[0].player_track_id == 3
        assert result[1].player_track_id == 4

    def test_empty_actions_returns_empty(self) -> None:
        """Empty input returns empty list (no crash)."""
        result = joint_attribute([], [], {}, serving_team=None)
        assert result == []

    def test_serving_team_none_seeds_from_first_serve(self) -> None:
        """If serving_team is None, the first SERVE seeds it; valid assignment found."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=50, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=3, court_side="far", confidence=0.95,
            ),
            ClassifiedAction(
                action_type=ActionType.RECEIVE,
                frame=90, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=4, court_side="near", confidence=0.9,
            ),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        result = joint_attribute(actions, contacts, team_assignments, serving_team=None)
        assert result[0].player_track_id == 3
        assert result[1].player_track_id == 1  # team A; R2 satisfied even without pre-set serving_team
