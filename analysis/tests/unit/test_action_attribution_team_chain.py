"""Unit tests for the team-chain override predicate and its helpers.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

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
