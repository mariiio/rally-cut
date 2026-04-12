"""Point-winner detection from terminal action analysis.

NOTE: This module is retained for future use but is NOT active in the
production pipeline. Evaluation (2026-04-12) showed 43.8% accuracy —
below chance and harmful to the Viterbi decoder. The issue is that
terminal action court_side is unreliable without accurate court
calibration. Re-enable when court calibration improves.

Determines which physical SIDE (near/far) won each rally by analyzing
the last classified action. Outputs physical side rather than team label
(A/B) to avoid the team-mapping errors identified in Phase 0.

Volleyball terminal action logic:
  - NET-CROSSING terminal actions (serve, attack, block) → the side
    that made the action SENT the ball over. No return came → that
    side WON the point.
  - SAME-SIDE terminal actions (receive, set, dig) → the ball stayed
    on that side. Rally ended with ball on their court → that side
    LOST the point (opponent won).

This is used by the cross-rally Viterbi to enforce the hard transition
rule: serving_side[i+1] = winning_side[i].
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rallycut.tracking.action_classifier import ActionType, RallyActions


class WinMethod(str, Enum):
    """How the point was determined."""

    ATTACK_KILL = "attack_kill"  # Terminal attack → ball landed on opponent's side
    ACE = "ace"  # Serve with no return
    BLOCK_KILL = "block_kill"  # Block sent ball back, no re-play
    OPPONENT_ERROR = "opponent_error"  # Terminal receive/set/dig → couldn't complete
    UNKNOWN = "unknown"


# Actions that send the ball to the opponent's side.
_NET_CROSSING = frozenset({ActionType.SERVE, ActionType.ATTACK, ActionType.BLOCK})

# Actions that keep the ball on the same side.
_SAME_SIDE = frozenset({ActionType.RECEIVE, ActionType.SET, ActionType.DIG})


@dataclass
class PointWinnerResult:
    """Point winner prediction for one rally.

    Attributes:
        winning_side: "near" or "far" (physical court side that won),
            or None if undetermined.
        confidence: 0-1 confidence in the prediction.
        method: How the winner was determined.
        terminal_action_type: The type of the terminal action used.
        terminal_court_side: The court_side of the terminal action.
    """

    winning_side: str | None  # "near", "far", or None
    confidence: float
    method: WinMethod
    terminal_action_type: ActionType | None = None
    terminal_court_side: str | None = None


def detect_point_winner(rally_actions: RallyActions) -> PointWinnerResult:
    """Detect which physical side won the rally.

    Uses terminal action analysis: the last non-UNKNOWN action determines
    the point outcome based on volleyball rules.

    Args:
        rally_actions: Classified actions for one rally.

    Returns:
        PointWinnerResult with winning physical side and confidence.
    """
    # Find the terminal (last non-UNKNOWN, non-synthetic) action.
    terminal = None
    for action in reversed(rally_actions.actions):
        if action.action_type != ActionType.UNKNOWN and not action.is_synthetic:
            terminal = action
            break

    if terminal is None:
        return PointWinnerResult(
            winning_side=None,
            confidence=0.0,
            method=WinMethod.UNKNOWN,
        )

    side = terminal.court_side  # "near" or "far"
    action_type = terminal.action_type

    if side not in ("near", "far"):
        return PointWinnerResult(
            winning_side=None,
            confidence=0.0,
            method=WinMethod.UNKNOWN,
            terminal_action_type=action_type,
            terminal_court_side=side,
        )

    opposite = "far" if side == "near" else "near"

    if action_type in _NET_CROSSING:
        # The ball was sent to the opponent's court. No return came.
        # The side that made this action won.
        if action_type == ActionType.SERVE:
            method = WinMethod.ACE
        elif action_type == ActionType.BLOCK:
            method = WinMethod.BLOCK_KILL
        else:
            method = WinMethod.ATTACK_KILL
        return PointWinnerResult(
            winning_side=side,
            confidence=terminal.confidence,
            method=method,
            terminal_action_type=action_type,
            terminal_court_side=side,
        )

    if action_type in _SAME_SIDE:
        # The ball stayed on this side. Rally ended without the ball
        # crossing the net → this side lost (opponent won).
        return PointWinnerResult(
            winning_side=opposite,
            confidence=terminal.confidence * 0.7,  # lower confidence for inferred
            method=WinMethod.OPPONENT_ERROR,
            terminal_action_type=action_type,
            terminal_court_side=side,
        )

    # UNKNOWN or unexpected action type.
    return PointWinnerResult(
        winning_side=None,
        confidence=0.0,
        method=WinMethod.UNKNOWN,
        terminal_action_type=action_type,
        terminal_court_side=side,
    )
