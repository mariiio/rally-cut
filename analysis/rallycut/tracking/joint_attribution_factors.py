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
