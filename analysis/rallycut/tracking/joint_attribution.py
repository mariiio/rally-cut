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

    Special sentinel: ``count_consecutive_same_team == -1`` indicates that
    the prior action was a synthetic passthrough with an unresolved player
    (pid=-1). In this mode, all constraints are relaxed — the next action
    is treated as the effective rally seed with no team or type restriction.
    This arises when a synthetic SERVE carries no player attribution.
    """
    # UNKNOWN passes through — no team constraint.
    if action_type == ActionType.UNKNOWN:
        return True

    # Sentinel: synthetic-unresolved passthrough → unconstrained next action.
    if prior.count_consecutive_same_team == -1:
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
    - Contacts are indexed by frame: ``contact_by_frame[action.frame]``.
    - Synthetic actions (``action.is_synthetic=True``) are treated as
      PASSTHROUGH: the existing ``action.player_track_id`` is kept and
      state is advanced from that assignment (no beam expansion). If the
      synthetic action's pid has no team mapping (e.g. pl_pid=-1), state
      is left unchanged.
    - Non-synthetic actions with no matching contact frame are also treated
      as PASSTHROUGH (same logic as synthetic).
    - UNKNOWN actions accept any pid in their contact's candidates (no
      team constraint), but still receive a proximity score.
    """
    if not actions:
        return []

    # Index contacts by frame so non-synthetic actions can look up their
    # contact without requiring parallel list alignment.
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    # Initial beam: one entry per valid candidate for the first non-passthrough
    # action, OR a single passthrough entry if the first action is synthetic/
    # has no contact.
    initial_state = RallyState(
        expected_team=None,
        count_consecutive_same_team=0,
        last_was_block=False,
        serving_team=serving_team,
    )

    # Each beam entry is (cumulative_score, assignment_so_far, state_after).
    beam: list[tuple[float, list[int], RallyState]] = []

    # --- Seed beam from actions[0] ---
    action_0 = actions[0]
    contact_0 = contact_by_frame.get(action_0.frame)
    if action_0.is_synthetic or contact_0 is None:
        # Passthrough: keep existing pid, advance state if team is known.
        existing_pid = action_0.player_track_id
        existing_team = team_assignments.get(existing_pid)
        if existing_team is not None:
            new_state = _derive_state_after(action_0, existing_team, initial_state)
        else:
            # pid=-1 or unmapped: state cannot be seeded. Use sentinel
            # count=-1 so the next action is treated as unconstrained
            # (see _is_valid_candidate docstring).
            new_state = RallyState(
                expected_team=None,
                count_consecutive_same_team=-1,
                last_was_block=False,
                serving_team=None,
            )
        beam = [(0.0, [existing_pid], new_state)]
    else:
        for tid, _dist in contact_0.player_candidates:
            tid_team = team_assignments.get(tid)
            if tid_team is None:
                continue
            if not _is_valid_candidate(action_0.action_type, tid_team, initial_state):
                continue
            score = _score_candidate(contact_0, tid)
            if not math.isfinite(score):
                continue
            new_state = _derive_state_after(action_0, tid_team, initial_state)
            beam.append((score, [tid], new_state))

    if not beam:
        return None

    # Expand for actions[1:].
    for i in range(1, len(actions)):
        next_beam: list[tuple[float, list[int], RallyState]] = []
        action_i = actions[i]
        contact_i = contact_by_frame.get(action_i.frame)

        if action_i.is_synthetic or contact_i is None:
            # Passthrough: keep existing pid, advance state from each partial.
            existing_pid = action_i.player_track_id
            existing_team = team_assignments.get(existing_pid)
            for cum_score, partial, state in beam:
                if existing_team is not None:
                    new_state = _derive_state_after(action_i, existing_team, state)
                else:
                    # pid=-1 or unmapped: use sentinel so the next real
                    # action is unconstrained (see _is_valid_candidate).
                    new_state = RallyState(
                        expected_team=None,
                        count_consecutive_same_team=-1,
                        last_was_block=False,
                        serving_team=None,
                    )
                next_beam.append((cum_score, partial + [existing_pid], new_state))
        else:
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


def joint_attribute(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int],
    serving_team: int | None,
    beam_width: int = 50,
) -> list[ClassifiedAction]:
    """Joint rule-aware action attribution over a single rally.

    Reuses existing proximity ranking (Contact.player_candidates) as the
    soft signal. Enforces R1-R5 as hard constraints via beam search.
    Mutates each ClassifiedAction in place to rewrite ``player_track_id``
    per the beam-search-best assignment. Returns the same list (for
    chainable use).

    If no rule-valid assignment exists, returns ``actions`` UNCHANGED and
    emits a single WARN log line. Never silently corrupts attribution.

    Parameters
    ----------
    actions, contacts
        Lists of actions and contacts for the rally. Contacts are indexed
        by frame; list lengths need not match (synthetic actions have no
        corresponding contact entry).
    team_assignments
        Map from player track id to team (0=near=A, 1=far=B).
    serving_team
        The team that opened the rally (0 or 1). If None, the first
        SERVE's chosen pid seeds it.
    beam_width
        Number of partial assignments retained per step. Default 50.
    """
    if not actions:
        return actions

    assignment = _beam_search(
        actions, contacts, team_assignments,
        serving_team=serving_team, beam_width=beam_width,
    )
    if assignment is None:
        logger.warning(
            "joint_attribute fallback: no rule-valid assignment for "
            "rally with %d actions; preserving input attributions",
            len(actions),
        )
        return actions

    for i, action in enumerate(actions):
        action.player_track_id = assignment[i]
    return actions
