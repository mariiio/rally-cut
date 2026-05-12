"""Unit tests for joint-attribution factor functions.

Each test is a small truth table for one factor: explicit inputs +
expected log-likelihood output. Pure-function testing.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

import math

from rallycut.tracking.joint_attribution_factors import (
    unary_action_prior,
    unary_distance,
    unary_proximity,
)
from rallycut.tracking.joint_attribution_weights import DEFAULT_WEIGHTS

# ---- unary_proximity ----

def test_unary_proximity_rank_1_best() -> None:
    """Player at rank 1 in candidates should get the highest proximity score."""
    candidates = [(1, 0.05), (2, 0.10), (3, 0.20), (4, 0.50)]  # (raw_id, distance)
    rank_1 = unary_proximity(
        "P1", candidates, pid_to_raw_id={1: 1, 2: 2, 3: 3, 4: 4}, weights=DEFAULT_WEIGHTS
    )
    rank_2 = unary_proximity(
        "P2", candidates, pid_to_raw_id={1: 1, 2: 2, 3: 3, 4: 4}, weights=DEFAULT_WEIGHTS
    )
    assert rank_1 > rank_2


def test_unary_proximity_player_not_in_candidates() -> None:
    """Player absent from candidates returns a large negative score."""
    candidates = [(1, 0.05)]
    score = unary_proximity("P4", candidates, pid_to_raw_id={1: 1, 4: 4}, weights=DEFAULT_WEIGHTS)
    assert score < -100  # large negative


def test_unary_proximity_absent_state_returns_zero() -> None:
    """ABSENT_TEAM_X states don't claim proximity; return 0."""
    candidates = [(1, 0.05)]
    score = unary_proximity(
        "ABSENT_TEAM_A", candidates, pid_to_raw_id={1: 1}, weights=DEFAULT_WEIGHTS
    )
    assert score == 0.0


# ---- unary_distance ----

def test_unary_distance_closer_is_better() -> None:
    candidates = [(1, 0.05), (2, 0.20)]
    pid_to_raw = {1: 1, 2: 2}
    near = unary_distance("P1", candidates, pid_to_raw, weights=DEFAULT_WEIGHTS)
    far = unary_distance("P2", candidates, pid_to_raw, weights=DEFAULT_WEIGHTS)
    assert near > far


def test_unary_distance_absent_uses_team_distance() -> None:
    """ABSENT_TEAM_X penalty proportional to team's NEAREST tracked player.
    If team's nearest player is far, absent is less penalized."""
    candidates = [(1, 0.05), (3, 0.30)]
    pid_to_raw = {1: 1, 3: 3}
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    # team A's nearest: P1 at 0.05 (close) -> ABSENT_TEAM_A is heavily penalized
    abs_a = unary_distance("ABSENT_TEAM_A", candidates, pid_to_raw,
                           weights=DEFAULT_WEIGHTS, team_assignments=team_assignments)
    # team B's nearest: P3 at 0.30 (far) -> ABSENT_TEAM_B is less penalized
    abs_b = unary_distance("ABSENT_TEAM_B", candidates, pid_to_raw,
                           weights=DEFAULT_WEIGHTS, team_assignments=team_assignments)
    assert abs_b > abs_a  # less penalty for team-B-absent


# ---- unary_action_prior ----

def test_unary_action_prior_matching_initial_pid() -> None:
    """If state matches the initial classifier PID, return log(0.6)*w."""
    score = unary_action_prior("P1", initial_pid=1, weights=DEFAULT_WEIGHTS)
    assert math.isclose(score, math.log(0.6) * DEFAULT_WEIGHTS.w_prior, abs_tol=1e-9)


def test_unary_action_prior_non_matching_player() -> None:
    score = unary_action_prior("P2", initial_pid=1, weights=DEFAULT_WEIGHTS)
    assert math.isclose(score, math.log(0.1) * DEFAULT_WEIGHTS.w_prior, abs_tol=1e-9)


def test_unary_action_prior_absent_state() -> None:
    """Absent states get a very small prior unless evidence is strong."""
    score = unary_action_prior("ABSENT_TEAM_A", initial_pid=1, weights=DEFAULT_WEIGHTS)
    assert math.isclose(score, math.log(0.05) * DEFAULT_WEIGHTS.w_prior, abs_tol=1e-9)


def test_unary_action_prior_no_initial_pid() -> None:
    """When initial_pid is None (action_classifier abstained), prior is uniform."""
    score = unary_action_prior("P1", initial_pid=None, weights=DEFAULT_WEIGHTS)
    expected = math.log(0.25) * DEFAULT_WEIGHTS.w_prior  # 1/4 if no info
    assert math.isclose(score, expected, abs_tol=1e-9)
