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
        )

    def test_all_gates_pass_allows_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is True

    def test_env_flag_off_denies_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "0"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_low_action_confidence_denies_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta = self._setup(
            confidence=0.5,
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_broken_chain_denies_override(self) -> None:
        fn, action, contact, expected, _, ta = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, False, ta) is False

    def test_no_candidate_within_distance_cap_denies_override(self) -> None:
        # candidate is 4x further than current → > 1.5x cap
        fn, action, contact, expected, chain_ok, ta = self._setup(
            current_dist=0.05, candidate_dist=0.25,
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_court_side_disagrees_denies_override(self) -> None:
        # expected_team=0 → expected_side="near"; contact reports "far"
        fn, action, contact, expected, chain_ok, ta = self._setup(
            court_side="far",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_court_side_unknown_is_soft_pass(self) -> None:
        # Allows override when court_side cannot corroborate (no calibration)
        fn, action, contact, expected, chain_ok, ta = self._setup(
            court_side="unknown",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is True

    def test_current_player_distance_infinite_denies_override(self) -> None:
        # No current distance → cannot enforce distance cap → deny
        fn, action, _contact, expected, chain_ok, ta = self._setup()
        contact_inf = _contact_with_candidates(
            frame=100,
            nearest_dist=math.inf,
            candidates=[(4, math.inf), (1, 0.06)],
            court_side="near",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact_inf, expected, chain_ok, ta) is False

    def test_current_player_distance_nan_denies_override(self) -> None:
        """NaN player_distance fails the math.isfinite check in G3."""
        fn, action, _, expected, chain_ok, ta = self._setup()
        contact_nan = _contact_with_candidates(
            frame=100, nearest_dist=math.nan,
            candidates=[(4, math.nan), (1, 0.06)],
            court_side="near",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact_nan, expected, chain_ok, ta) is False


class TestReattributePlayersIntegration:
    """End-to-end test that reattribute_players Pass 2 fires when the new
    predicate passes — even when the wrong-team current attribution is the
    spatially nearest candidate (the canonical bug pattern)."""

    def test_cross_team_receive_overridden_when_chain_trustworthy(self) -> None:
        """The user-quoted bug: P2 receives P1's serve (same team) — should
        be overridden to a candidate on the receiving team when all gates
        pass.

        Setup (PIDs already in canonical 1-4 space):
          team 0 (near, A) = {1, 2}, team 1 (far, B) = {3, 4}
          actions:
            - serve by track 3 (team B, far)
            - receive currently attributed to track 4 (team B, NEAREST in
              candidates) — wrong, expected_team is 0 (team A)
            - receive's contact has candidates: [(4, 0.05), (1, 0.07)]
              => track 1 (team A) is within 1.5x distance of track 4.
            - Contact.court_side = "near" — corroborates expected_team=0
        """
        from rallycut.tracking.action_classifier import reattribute_players

        actions = [
            _action(
                ActionType.SERVE, 50, player_track_id=3, confidence=0.95,
                court_side="far",
            ),
            _action(
                ActionType.RECEIVE, 90, player_track_id=4, confidence=0.9,
                court_side="near",
            ),
        ]
        contacts = [
            _contact_with_candidates(
                frame=50, nearest_dist=0.04,
                candidates=[(3, 0.04)],
                court_side="far",
            ),
            _contact_with_candidates(
                frame=90, nearest_dist=0.05,
                candidates=[(4, 0.05), (1, 0.07)],
                court_side="near",
            ),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        # The receive should now be attributed to track 1 (team A), not 4.
        assert actions[1].action_type == ActionType.RECEIVE
        assert actions[1].player_track_id == 1

    def test_env_flag_off_preserves_old_behavior(self) -> None:
        """With env flag off, the old unconditional nearest-guard blocks
        the override and the wrong-team attribution is preserved."""
        from rallycut.tracking.action_classifier import reattribute_players

        actions = [
            _action(
                ActionType.SERVE, 50, player_track_id=3, confidence=0.95,
                court_side="far",
            ),
            _action(
                ActionType.RECEIVE, 90, player_track_id=4, confidence=0.9,
                court_side="near",
            ),
        ]
        contacts = [
            _contact_with_candidates(
                frame=50, nearest_dist=0.04,
                candidates=[(3, 0.04)],
                court_side="far",
            ),
            _contact_with_candidates(
                frame=90, nearest_dist=0.05,
                candidates=[(4, 0.05), (1, 0.07)],
                court_side="near",
            ),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "0"}):
            reattribute_players(actions, contacts, team_assignments)

        # Old behavior: wrong-team nearest attribution is preserved.
        assert actions[1].player_track_id == 4

    def test_broken_chain_blocks_override(self) -> None:
        """An UNKNOWN action between the serve and the receive breaks the
        chain — the override is blocked even with env flag on."""
        from rallycut.tracking.action_classifier import reattribute_players

        actions = [
            _action(
                ActionType.SERVE, 50, player_track_id=3, confidence=0.95,
                court_side="far",
            ),
            _action(
                ActionType.UNKNOWN, 70, player_track_id=2, confidence=0.4,
            ),
            _action(
                ActionType.RECEIVE, 90, player_track_id=4, confidence=0.9,
                court_side="near",
            ),
        ]
        contacts = [
            _contact_with_candidates(
                frame=50, nearest_dist=0.04,
                candidates=[(3, 0.04)],
                court_side="far",
            ),
            _contact_with_candidates(
                frame=70, nearest_dist=0.05,
                candidates=[(2, 0.05)],
                court_side="near",
            ),
            _contact_with_candidates(
                frame=90, nearest_dist=0.05,
                candidates=[(4, 0.05), (1, 0.07)],
                court_side="near",
            ),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        # Chain broken by UNKNOWN at frame 70 → no override fires.
        assert actions[2].player_track_id == 4
