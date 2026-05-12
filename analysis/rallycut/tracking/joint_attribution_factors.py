"""Pure factor functions for the Joint Attribution PGM.

Each function takes typed inputs and returns a log-likelihood (float).
No global state; no I/O. Trivially testable.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

import math

from rallycut.tracking.joint_attribution import (
    State,
    is_absent,
)
from rallycut.tracking.joint_attribution_weights import FactorWeights

_LARGE_NEGATIVE = -1000.0  # used when a state is impossible (e.g., player absent from candidates)


def unary_proximity(
    state: State,
    candidates: list[tuple[int, float]],
    pid_to_raw_id: dict[int, int],
    weights: FactorWeights,
) -> float:
    """Log-likelihood from playerCandidates rank ordering.
    Lower rank (1 = best) gets higher score."""
    if is_absent(state):
        return 0.0
    pid = int(state[1:])  # "P3" -> 3
    raw_id = pid_to_raw_id.get(pid)
    if raw_id is None:
        return _LARGE_NEGATIVE
    for rank, (cand_raw_id, _dist) in enumerate(candidates, start=1):
        if cand_raw_id == raw_id:
            return math.log(1.0 / (rank + 1)) * weights.w_proximity
    return _LARGE_NEGATIVE


def unary_distance(
    state: State,
    candidates: list[tuple[int, float]],
    pid_to_raw_id: dict[int, int],
    weights: FactorWeights,
    team_assignments: dict[int, str] | None = None,
) -> float:
    """Log-likelihood from playerCandidates distance.
    For tracked players: -distance * w_dist (closer = less penalty).
    For absent states: +nearest_team_distance * w_dist_team (far team -> absent
    is more plausible, so less penalized / higher score).
    """
    if is_absent(state):
        if team_assignments is None:
            return 0.0
        team = "A" if state == "ABSENT_TEAM_A" else "B"
        # Find nearest tracked player on this team
        team_pids = [pid for pid, t in team_assignments.items() if t == team]
        team_raw_ids = [pid_to_raw_id.get(pid) for pid in team_pids]
        team_raw_ids = [r for r in team_raw_ids if r is not None]
        team_distances = [
            d for raw_id, d in candidates if raw_id in team_raw_ids
        ]
        if not team_distances:
            return 0.0  # team has no tracked players in candidates; absent is "free"
        nearest = min(team_distances)
        # Positive: far nearest player -> absent is less penalized (semantic in test).
        return nearest * weights.w_dist_team
    # Tracked player
    pid = int(state[1:])
    raw_id = pid_to_raw_id.get(pid)
    if raw_id is None:
        return _LARGE_NEGATIVE
    for cand_raw_id, dist in candidates:
        if cand_raw_id == raw_id:
            return -dist * weights.w_dist
    return _LARGE_NEGATIVE


def unary_action_prior(
    state: State,
    initial_pid: int | None,
    weights: FactorWeights,
) -> float:
    """Log-likelihood from action_classifier's initial PID as soft prior."""
    if is_absent(state):
        return math.log(0.05) * weights.w_prior
    if initial_pid is None:
        # Uniform over 4 players
        return math.log(0.25) * weights.w_prior
    pid = int(state[1:])
    if pid == initial_pid:
        return math.log(0.6) * weights.w_prior
    return math.log(0.1) * weights.w_prior


def _state_team(state: State, team_assignments: dict[int, str]) -> str | None:
    """Resolve a state to its team letter, or None if unresolvable."""
    if state == "ABSENT_TEAM_A":
        return "A"
    if state == "ABSENT_TEAM_B":
        return "B"
    pid = int(state[1:])
    return team_assignments.get(pid)


def pairwise_no_back_to_back(
    state_t: State,
    state_t1: State,
    action_t: str | None,
    action_t1: str | None,
    weights: FactorWeights,
) -> float:
    """Penalty if consecutive contacts have the same player.
    Exception: block-attack by the same player is allowed."""
    if is_absent(state_t) or is_absent(state_t1):
        return 0.0
    if state_t != state_t1:
        return 0.0
    # Block-attack exception
    if action_t == "block" and action_t1 == "attack":
        return 0.0
    return -weights.w_back_to_back


def pairwise_alternation(
    state_t: State,
    state_t1: State,
    net_crossed: bool,
    team_assignments: dict[int, str],
    weights: FactorWeights,
) -> float:
    """Two soft rules folded into one factor:
    - Same team across a net crossing: penalty (should alternate)
    - Different team without a net crossing: penalty (impossible play)"""
    team_t = _state_team(state_t, team_assignments)
    team_t1 = _state_team(state_t1, team_assignments)
    if team_t is None or team_t1 is None:
        return 0.0
    same_team = team_t == team_t1
    if net_crossed and same_team:
        return -weights.w_alternation
    if not net_crossed and not same_team:
        return -weights.w_team_consistency
    return 0.0


_ABSENT_SERVER_SMALL_PENALTY = -0.5  # spec: small penalty when first is ABSENT_TEAM_<serving_team>


def pairwise_absent_pair(
    state_t: State,
    state_t1: State,
    weights: FactorWeights,
) -> float:
    """Penalty for two consecutive ABSENT_* states."""
    if is_absent(state_t) and is_absent(state_t1):
        return -weights.w_absent_pair
    return 0.0


def higher_3_contact_per_side(
    states: tuple[State, ...],
    net_crossings: tuple[bool, ...],
    team_assignments: dict[int, str],
    weights: FactorWeights,
) -> float:
    """Penalty per same-team contact beyond the 3rd in a row without a crossing.

    `net_crossings[i]` is True iff a net crossing occurred between states[i]
    and states[i+1]. Length is len(states) - 1.
    """
    assert len(net_crossings) == len(states) - 1
    penalty = 0.0
    streak = 1
    streak_team: str | None = _state_team(states[0], team_assignments)
    for i in range(1, len(states)):
        team_i = _state_team(states[i], team_assignments)
        if net_crossings[i - 1] or team_i != streak_team:
            streak = 1
            streak_team = team_i
            continue
        streak += 1
        if streak > 3:
            penalty -= weights.w_3_contact
    return penalty


def higher_serve_first(
    first_state: State,
    serving_team: str | None,
    team_assignments: dict[int, str],
    weights: FactorWeights,
) -> float:
    """Penalty if first contact's team doesn't match serving_team.

    Special case: if first_state is ABSENT_TEAM_<serving_team>, small penalty
    (off-screen server matches the serving team but isn't tracked).
    """
    if serving_team is None:
        return 0.0
    if first_state == f"ABSENT_TEAM_{serving_team}":
        return _ABSENT_SERVER_SMALL_PENALTY
    first_team = _state_team(first_state, team_assignments)
    if first_team is None:
        return 0.0
    if first_team == serving_team:
        return 0.0
    return -weights.w_serve_first
