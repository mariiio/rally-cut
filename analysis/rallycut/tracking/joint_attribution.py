"""Joint rule-aware action attribution (v2.0).

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Replaces per-contact local attribution with a beam-search joint solver
that enforces beach volleyball game rules (R1-R5) as HARD constraints
over the entire rally. The soft scoring reuses existing proximity
ranking from Contact.player_candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction

logger = logging.getLogger(__name__)

# Action category sets — these are the canonical R2/R3 partitions.
_NET_CROSSING_ACTIONS = {ActionType.SERVE, ActionType.ATTACK}
_SAME_SIDE_ACTIONS = {ActionType.RECEIVE, ActionType.SET, ActionType.DIG}

# Block is its own category (R5). UNKNOWN is passthrough.

# Soft-scoring epsilon for -log(dist + ε) stability.
_SCORE_EPSILON = 1e-3

# R4 hard limit: max consecutive same-team non-block contacts.
_MAX_SAME_TEAM = 3


@dataclass(frozen=True)
class RallyState:
    """State after processing action[i]; constrains action[i+1].

    Attributes
    ----------
    expected_team : int | None
        The team (0=near=A, 1=far=B) that should perform the NEXT action,
        per R2/R3/R5. None pre-rally (before the first SERVE seeds it).
    count_consecutive_same_team : int
        Number of consecutive non-block same-team contacts ending at this
        action. Used for R4. Resets to 1 after a net-crossing (R2 flip)
        and to 0 after a BLOCK (R5).
    last_was_block : bool
        True iff the most recent processed action was a BLOCK. Used by
        R5 to give the next same-team action a free-pass (count starts
        fresh at 1).
    serving_team : int | None
        The team that opened the rally with the serve. Seeded by the
        first SERVE if not pre-set. Constant for the rest of the rally.
    """

    expected_team: int | None
    count_consecutive_same_team: int
    last_was_block: bool
    serving_team: int | None


def _derive_state_after(
    action: ClassifiedAction,
    team_at_action: int,
    prior: RallyState,
) -> RallyState:
    """Compute the rally state AFTER processing this action, given its team.

    Implements the R2/R3/R4/R5 semantics described in the spec. UNKNOWN
    actions pass through (state unchanged). The first SERVE seeds
    serving_team if prior.serving_team is None.

    Parameters
    ----------
    action : ClassifiedAction
        The action being processed. Only `action_type` is read.
    team_at_action : int
        Team assignment of the player performing this action (0 or 1).
    prior : RallyState
        State BEFORE this action.

    Returns
    -------
    RallyState
        State AFTER this action — constrains the next action.
    """
    if action.action_type == ActionType.UNKNOWN:
        return prior

    serving_team = (
        prior.serving_team
        if prior.serving_team is not None
        else (team_at_action if action.action_type == ActionType.SERVE else None)
    )

    if action.action_type == ActionType.BLOCK:
        # R5: block does not flip possession (cover by same team is legal)
        # and does not count toward the 3-contact limit (count resets to 0).
        return RallyState(
            expected_team=team_at_action,
            count_consecutive_same_team=0,
            last_was_block=True,
            serving_team=serving_team,
        )

    if action.action_type in _NET_CROSSING_ACTIONS:
        # R2: net crossing flips possession; opposing team's first contact starts count at 1.
        return RallyState(
            expected_team=1 - team_at_action,
            count_consecutive_same_team=1,
            last_was_block=False,
            serving_team=serving_team,
        )

    # _SAME_SIDE_ACTIONS (RECEIVE, SET, DIG): R3 + R5 cover-reset semantics.
    new_count = 1 if prior.last_was_block else prior.count_consecutive_same_team + 1
    return RallyState(
        expected_team=team_at_action,
        count_consecutive_same_team=new_count,
        last_was_block=False,
        serving_team=serving_team,
    )
