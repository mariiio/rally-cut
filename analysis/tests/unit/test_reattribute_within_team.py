"""Unit tests for v3.1 within-team proximity swap (Pass 2c) in reattribute_players.

Spec: docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md
"""

from __future__ import annotations

from unittest.mock import patch

from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    reattribute_players,
)
from rallycut.tracking.contact_detector import Contact


def _action(
    action_type: ActionType,
    frame: int,
    player_track_id: int,
    confidence: float = 0.9,
) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5, ball_y=0.5, velocity=0.02,
        player_track_id=player_track_id,
        court_side="near", confidence=confidence,
    )


def _contact(
    frame: int,
    candidates: list[tuple[int, float]],
) -> Contact:
    return Contact(
        frame=frame,
        ball_x=0.5, ball_y=0.5, velocity=0.02, direction_change_deg=60.0,
        player_track_id=(candidates[0][0] if candidates else -1),
        player_distance=(candidates[0][1] if candidates else float("inf")),
        player_candidates=candidates,
        court_side="near", is_validated=True,
    )


class TestPass2cWithinTeamSwap:
    """Truth table for Pass 2c."""

    def test_swaps_when_rank1_differs_same_team(self) -> None:
        """Same-team mismatch + rank-1 differs from current → swap to rank-1."""
        # Serve currently attributed to track 2 (team A). Rank-1 candidate is
        # track 1 (also team A). Pass 2 skipped (same team). Pass 2c swaps.
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=2),
        ]
        contacts = [
            _contact(frame=50, candidates=[(1, 0.04), (2, 0.06)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}  # 1, 2 = team A; 3, 4 = team B

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 1  # swapped from 2 to 1

    def test_no_swap_when_rank1_matches_current(self) -> None:
        """If current is already rank-1, no swap."""
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=1),
        ]
        contacts = [
            _contact(frame=50, candidates=[(1, 0.04), (2, 0.06)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 1  # unchanged

    def test_no_swap_when_rank1_is_cross_team(self) -> None:
        """Cross-team rank-1 → Pass 2c declines (Pass 2's domain)."""
        # Current attribution is on the EXPECTED team (correct), but rank-1 is
        # on the OTHER team. Pass 2c should not swap.
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=2),  # team A
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04), (2, 0.06)]),  # rank-1=team B
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 2  # unchanged

    def test_env_flag_off_disables_swap(self) -> None:
        """WITHIN_TEAM_PROXIMITY_SWAP=0 restores pre-v3.1 behavior."""
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=2),
        ]
        contacts = [
            _contact(frame=50, candidates=[(1, 0.04), (2, 0.06)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "0"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 2  # unchanged (env flag off)
