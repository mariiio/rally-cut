"""Unit tests for the team-chain override predicate and its helpers.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md
"""

from __future__ import annotations

import math
from unittest.mock import patch

from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    _chain_integrity,
)
from rallycut.tracking.contact_detector import Contact


def _action(
    action_type: ActionType,
    frame: int,
    player_track_id: int = 1,
    confidence: float = 0.9,
    is_synthetic: bool = False,
    court_side: str = "near",
) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        player_track_id=player_track_id,
        court_side=court_side,
        confidence=confidence,
        is_synthetic=is_synthetic,
    )


class TestChainIntegrity:
    """Truth table for _chain_integrity."""

    def test_clean_chain_after_seed_serve(self) -> None:
        """Serve seeds the chain; subsequent known non-synthetic actions stay intact."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.SET, 50, player_track_id=3),
            _action(ActionType.ATTACK, 70, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True, True, True]

    def test_unknown_breaks_chain_from_that_point(self) -> None:
        """An UNKNOWN action sets all subsequent positions to False."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.UNKNOWN, 50, player_track_id=3),
            _action(ActionType.ATTACK, 70, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True, False, False]

    def test_synthetic_non_seed_breaks_chain(self) -> None:
        """A synthetic dig/set in the middle of a chain breaks downstream."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.DIG, 50, player_track_id=3, is_synthetic=True),
            _action(ActionType.ATTACK, 70, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True, False, False]

    def test_no_seed_serve_means_all_false(self) -> None:
        """Actions before the first valid SERVE all read False."""
        actions = [
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.SET, 50, player_track_id=3),
        ]
        result = _chain_integrity(actions)
        assert result == [False, False]

    def test_serve_with_unattributed_player_does_not_seed(self) -> None:
        """A SERVE with player_track_id=-1 does NOT seed the chain."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=-1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [False, False]

    def test_synthetic_serve_seeds_chain(self) -> None:
        """A synthetic SERVE with player_track_id >= 0 still seeds the chain.

        Rationale: synthetic serves are seeded with a server identity by
        _make_synthetic_serve. The downstream chain is observable from there.
        """
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1, is_synthetic=True),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True]

    def test_empty_actions(self) -> None:
        """Empty input returns empty output."""
        assert _chain_integrity([]) == []


def _contact_with_candidates(
    frame: int,
    nearest_dist: float,
    candidates: list[tuple[int, float]],
    court_side: str = "near",
) -> Contact:
    """Build a Contact with ranked player_candidates and player_distance set."""
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        direction_change_deg=60.0,
        player_track_id=candidates[0][0] if candidates else -1,
        player_distance=nearest_dist,
        player_candidates=candidates,
        court_side=court_side,
        is_validated=True,
    )


class TestTeamChainOverrideAllowed:
    """Truth table for the 4-gate predicate.

    Convention used in these tests: team_assignments[tid] = 0 means near
    (team A), = 1 means far (team B). The current (wrong) attribution is
    track 4 on team 1; the correct attribution should be track 1 on team 0.
    expected_team = 0 (near). The Contact reports court_side="near".
    """

    def _setup(
        self,
        *,
        confidence: float = 0.9,
        chain_ok: bool = True,
        candidate_dist: float = 0.06,
        current_dist: float = 0.05,
        court_side: str = "near",
        env_off: bool = False,
    ):
        from rallycut.tracking.action_classifier import (
            _team_chain_override_allowed,
        )
        action = _action(
            ActionType.RECEIVE, 100, player_track_id=4,
            confidence=confidence,
        )
        contact = _contact_with_candidates(
            frame=100,
            nearest_dist=current_dist,
            candidates=[(4, current_dist), (1, candidate_dist)],
            court_side=court_side,
        )
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}
        expected_team = 0  # The CORRECT team for the receive
        return (
            _team_chain_override_allowed,
            action, contact, expected_team, chain_ok, team_assignments,
            env_off,
        )

    def test_all_gates_pass_allows_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta, _ = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is True

    def test_env_flag_off_denies_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta, _ = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "0"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_low_action_confidence_denies_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            confidence=0.5,
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_broken_chain_denies_override(self) -> None:
        fn, action, contact, expected, _, ta, _ = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, False, ta) is False

    def test_no_candidate_within_distance_cap_denies_override(self) -> None:
        # candidate is 4x further than current → > 1.5x cap
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            current_dist=0.05, candidate_dist=0.25,
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_court_side_disagrees_denies_override(self) -> None:
        # expected_team=0 → expected_side="near"; contact reports "far"
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            court_side="far",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_court_side_unknown_is_soft_pass(self) -> None:
        # Allows override when court_side cannot corroborate (no calibration)
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            court_side="unknown",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is True

    def test_current_player_distance_infinite_denies_override(self) -> None:
        # No current distance → cannot enforce distance cap → deny
        fn, action, _contact, expected, chain_ok, ta, _ = self._setup()
        contact_inf = _contact_with_candidates(
            frame=100,
            nearest_dist=math.inf,
            candidates=[(4, math.inf), (1, 0.06)],
            court_side="near",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact_inf, expected, chain_ok, ta) is False
