"""Joint rule-aware action attribution (v2.0).

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Replaces per-contact local attribution with a beam-search joint solver
that enforces beach volleyball game rules (R1-R5) as HARD constraints
over the entire rally. The soft scoring reuses existing proximity
ranking from Contact.player_candidates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction
from rallycut.tracking.contact_detector import Contact

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


def _is_valid_candidate(
    action_type: ActionType,
    candidate_team: int,
    prior: RallyState,
) -> bool:
    """Return True iff assigning ``candidate_team`` to this action obeys R1-R5.

    Used by the beam search to prune rule-violating partial assignments.
    """
    # UNKNOWN passes through — no team constraint.
    if action_type == ActionType.UNKNOWN:
        return True

    # R1: first action of a rally must be SERVE.
    if prior.expected_team is None and prior.serving_team is None:
        if action_type != ActionType.SERVE:
            return False
        # Serving team unset → any team valid for the seed.
        return True

    # First action with known serving_team: must be SERVE and team must match.
    if prior.expected_team is None and prior.serving_team is not None:
        if action_type != ActionType.SERVE:
            return False
        return candidate_team == prior.serving_team

    # Non-first action: prior.expected_team is set.
    # R4: max 3 same-team contacts unless the next action is ATTACK (which crosses).
    if (
        prior.count_consecutive_same_team >= _MAX_SAME_TEAM
        and candidate_team == prior.expected_team
        and action_type not in _NET_CROSSING_ACTIONS
        and action_type != ActionType.BLOCK
    ):
        return False

    # R2/R3/R5: candidate_team must match prior.expected_team.
    # (For BLOCK after an ATTACK: prior.expected_team is the receiving team,
    # which is exactly the team that does the block.)
    return candidate_team == prior.expected_team


def _score_candidate(
    contact: Contact,
    candidate_pid: int,
) -> float:
    """Soft proximity score for assigning ``candidate_pid`` to this contact.

    Returns ``-log(rank_distance + ε)`` for a pid present in
    ``contact.player_candidates`` (the depth-corrected proximity ranking
    populated by ``detect_contacts``). Returns ``-inf`` for pids not in
    the candidates list — effectively rejecting them from the beam.

    Higher scores are better (smaller distance → larger -log).
    """
    for tid, dist in contact.player_candidates:
        if tid == candidate_pid:
            return -math.log(dist + _SCORE_EPSILON)
    return float("-inf")


def _beam_search(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int],
    serving_team: int | None,
    beam_width: int = 50,
) -> list[int] | None:
    """Beam search over per-action player assignments under R1-R5.

    Walks the action sequence left-to-right. At each step, expands each
    surviving partial assignment by every pid in the corresponding
    contact's player_candidates, filters by ``_is_valid_candidate``, scores
    by ``_score_candidate``, and retains the top ``beam_width`` partials.

    Returns the highest-scoring full assignment (list of pids parallel to
    actions) or ``None`` if no rule-valid full assignment exists.

    Notes
    -----
    - The contact for ``actions[i]`` is ``contacts[i]``. Lengths must match.
    - Rallies with fewer contacts than actions get -inf score and fall back.
    - UNKNOWN actions accept any pid in their contact's candidates (no
      team constraint), but still receive a proximity score.
    """
    if len(actions) != len(contacts):
        logger.warning(
            "joint_attribute: action/contact length mismatch (%d vs %d), "
            "falling back",
            len(actions), len(contacts),
        )
        return None
    if not actions:
        return []

    # Initial beam: one entry per valid candidate for actions[0].
    initial_state = RallyState(
        expected_team=None,
        count_consecutive_same_team=0,
        last_was_block=False,
        serving_team=serving_team,
    )

    # Each beam entry is (cumulative_score, assignment_so_far, state_after).
    beam: list[tuple[float, list[int], RallyState]] = []
    for tid, _dist in contacts[0].player_candidates:
        tid_team = team_assignments.get(tid)
        if tid_team is None:
            continue
        if not _is_valid_candidate(actions[0].action_type, tid_team, initial_state):
            continue
        score = _score_candidate(contacts[0], tid)
        if not math.isfinite(score):
            continue
        new_state = _derive_state_after(actions[0], tid_team, initial_state)
        beam.append((score, [tid], new_state))

    if not beam:
        return None

    # Expand for actions[1:].
    for i in range(1, len(actions)):
        next_beam: list[tuple[float, list[int], RallyState]] = []
        action_i = actions[i]
        contact_i = contacts[i]
        for cum_score, partial, state in beam:
            for tid, _dist in contact_i.player_candidates:
                tid_team = team_assignments.get(tid)
                if tid_team is None:
                    continue
                if not _is_valid_candidate(action_i.action_type, tid_team, state):
                    continue
                inc_score = _score_candidate(contact_i, tid)
                if not math.isfinite(inc_score):
                    continue
                new_score = cum_score + inc_score
                new_partial = partial + [tid]
                new_state = _derive_state_after(action_i, tid_team, state)
                next_beam.append((new_score, new_partial, new_state))
        if not next_beam:
            return None
        # Prune to beam_width by descending score.
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    # Return the highest-scoring full assignment.
    beam.sort(key=lambda x: x[0], reverse=True)
    return beam[0][1]
