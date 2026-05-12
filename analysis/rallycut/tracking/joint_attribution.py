"""Joint Attribution PGM — replaces reattribute_players with per-rally
joint MAP inference over 6 states per contact (4 players + 2 absent-team).

Public API: joint_attribute_rally(rally) -> RallyAttribution

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal

# State type: a tracked player (1-4) OR an absent-team marker.
# Represented as a string for easy serialization and pattern-matching.
State = str  # "P1" | "P2" | "P3" | "P4" | "ABSENT_TEAM_A" | "ABSENT_TEAM_B"

ABSENT_STATES = ("ABSENT_TEAM_A", "ABSENT_TEAM_B")


def is_absent(state: State) -> bool:
    """True if state is one of the ABSENT_TEAM_* markers."""
    return state in ABSENT_STATES


def absent_state_for_team(team: str) -> State:
    """Map team letter ('A' or 'B') to the corresponding absent state."""
    if team == "A":
        return "ABSENT_TEAM_A"
    if team == "B":
        return "ABSENT_TEAM_B"
    raise ValueError(f"unknown team: {team}")


@dataclass(frozen=True)
class RallyContext:
    """Bundle of per-rally inputs the PGM needs.

    Built by the caller (e.g., redetect_all_actions, reattribute-actions CLI)
    from existing data sources: contacts_json, actions_json (initial PIDs),
    team_assignments, serving_team, and optionally visual profiles.
    """
    rally_id: str
    contacts: list[dict[str, Any]]
    """Each contact: {frame, ballX, ballY, playerCandidates, courtSide, ...}
    Same shape as contacts_json.contacts[*]."""

    initial_actions: list[dict[str, Any]]
    """Each action: {frame, action, playerTrackId, confidence, ...}
    Initial output from classify_rally_actions; provides per-contact
    action labels and the soft-prior PID for unary action_prior factors."""

    team_assignments: dict[int, str]
    """Map: canonical PID (1-4) -> team letter ('A' or 'B'). From the
    cross-rally matcher."""

    serving_team: str | None
    """'A' or 'B' or None (if unknown). From the rally's matcher state."""

    visual_profiles: dict[int, list[float]] | None = None
    """Optional: PID -> embedding vector for cross-rally visual similarity.
    None disables the w_visual factor for this rally."""


@dataclass(frozen=True)
class RallyAttribution:
    """Result of joint_attribute_rally."""
    map: tuple[State, ...]
    """The MAP joint assignment, one State per contact."""

    score: float
    """Joint log-likelihood of the MAP assignment."""

    marginals: list[dict[State, float]]
    """Per-contact posterior marginals over states. Indexed by contact position.
    Computed by exp(score - max_score) and normalized per-contact."""

    fallback_used: Literal["exhaustive", "coordinate_descent"]
    """Which inference method was actually used (exhaustive for N<=8,
    coordinate_descent for N>8)."""


# Weights are safe to import at module top: joint_attribution_weights does not
# depend on this module. The factor functions live in joint_attribution_factors
# which DOES depend on this module's State / is_absent symbols, so those imports
# happen inside the functions below to avoid a circular import.
from rallycut.tracking.joint_attribution_weights import (  # noqa: E402
    DEFAULT_WEIGHTS,
    FactorWeights,
)


def build_state_domain(team_assignments: dict[int, str]) -> tuple[State, ...]:
    """Return the per-contact state domain: 4 player states + 2 absent-team states.
    Always returns the full domain regardless of team_assignments completeness;
    factor functions handle missing teams gracefully."""
    return ("P1", "P2", "P3", "P4", "ABSENT_TEAM_A", "ABSENT_TEAM_B")


def build_pid_to_raw_id(rally: RallyContext) -> dict[int, int]:
    """Build the canonical PID -> raw track id mapping from the rally's
    initial actions. Used to convert State (e.g., "P1") to candidate raw ids
    in playerCandidates lists.

    Falls back to identity mapping (1->1, 2->2, ...) if no actions or
    initial PIDs are present.
    """
    mapping: dict[int, int] = {}
    for action in rally.initial_actions:
        pid = action.get("playerTrackId")
        raw = action.get("rawTrackId")  # may be present from action_classifier
        if pid is None:
            continue
        if raw is None:
            # Identity fallback
            mapping[int(pid)] = int(pid)
        else:
            mapping[int(pid)] = int(raw)
    # Ensure all 4 PIDs have an entry (identity fallback for missing)
    for pid in (1, 2, 3, 4):
        mapping.setdefault(pid, pid)
    return mapping


def build_evidence(
    rally: RallyContext, weights: FactorWeights = DEFAULT_WEIGHTS,
) -> list[dict[State, float]]:
    """For each contact in the rally, compute a dict mapping each state
    to its summed unary log-likelihood (sum of all unary factor contributions).

    Returns a list of dicts of length len(rally.contacts).
    """
    # Local import: see module-level note about circular import avoidance.
    from rallycut.tracking.joint_attribution_factors import (
        unary_action_prior,
        unary_distance,
        unary_proximity,
    )

    pid_to_raw = build_pid_to_raw_id(rally)
    state_domain = build_state_domain(rally.team_assignments)
    evidence: list[dict[State, float]] = []
    # Pair contacts with their initial actions (by frame proximity)
    actions_by_frame = {a["frame"]: a for a in rally.initial_actions}
    for contact in rally.contacts:
        scores: dict[State, float] = {}
        candidates = [
            (int(tid), float(d) if d is not None else 1.0)
            for tid, d in (contact.get("playerCandidates") or [])
        ]
        # Find action at this contact's frame (within ±5)
        cf = contact["frame"]
        nearest_action: dict[str, Any] | None = None
        nearest_d: float = math.inf
        for af, a in actions_by_frame.items():
            if abs(af - cf) <= 5 and abs(af - cf) < nearest_d:
                nearest_d, nearest_action = abs(af - cf), a
        initial_pid = nearest_action.get("playerTrackId") if nearest_action else None
        for state in state_domain:
            score = (
                unary_proximity(state, candidates, pid_to_raw, weights)
                + unary_distance(state, candidates, pid_to_raw, weights, rally.team_assignments)
                + unary_action_prior(state, initial_pid, weights)
            )
            scores[state] = score
        evidence.append(scores)
    return evidence


_EXHAUSTIVE_THRESHOLD = 6  # exhaustive for N <= 6 contacts; coordinate-descent fallback for larger.
# Threshold lowered from 8 to 6 after profiling on the 409-rally action-GT corpus showed
# 28% of rallies have N >= 7 (exhaustive >1s/rally; spec budget 50ms/rally avg).
# Per-corpus distribution: 71.6% at N <= 6 (exhaustive, fast); 28.4% at N >= 7 (coord descent).
# TODO(Phase B): numpy-vectorize the inner enumeration to raise the threshold safely.


def _net_crossings_for(contacts: list[dict[str, Any]]) -> tuple[bool, ...]:
    """For each consecutive contact pair, True if courtSide flipped between them.
    "unknown" courtSide is treated as no-information (preserves the prior side)."""
    sides: list[str | None] = []
    last_known: str | None = None
    for c in contacts:
        side = c.get("courtSide")
        if side in ("near", "far"):
            last_known = side
            sides.append(side)
        else:
            sides.append(last_known)
    crossings: list[bool] = []
    for i in range(len(sides) - 1):
        if sides[i] is None or sides[i + 1] is None:
            crossings.append(False)
        else:
            crossings.append(sides[i] != sides[i + 1])
    return tuple(crossings)


def score_joint_config(
    config: tuple[State, ...],
    rally: RallyContext,
    weights: FactorWeights = DEFAULT_WEIGHTS,
    evidence: list[dict[State, float]] | None = None,
    net_crossings: tuple[bool, ...] | None = None,
) -> float:
    """Compute joint log-likelihood for a configuration.

    Sums all unary, pairwise, and higher-order factor contributions.
    `evidence` and `net_crossings` are precomputed once per rally; pass
    them in for efficiency in the inner enumeration loop.
    """
    # Local import: see module-level note about circular import avoidance.
    from rallycut.tracking.joint_attribution_factors import (
        higher_3_contact_per_side,
        higher_serve_first,
        pairwise_absent_pair,
        pairwise_alternation,
        pairwise_no_back_to_back,
    )

    if evidence is None:
        evidence = build_evidence(rally, weights)
    if net_crossings is None:
        net_crossings = _net_crossings_for(rally.contacts)

    score = 0.0
    # Unary
    for t, state in enumerate(config):
        score += evidence[t][state]
    # Pairwise
    for t in range(len(config) - 1):
        action_t = (
            rally.initial_actions[t].get("action")
            if t < len(rally.initial_actions) else None
        )
        action_t1 = (
            rally.initial_actions[t + 1].get("action")
            if t + 1 < len(rally.initial_actions) else None
        )
        score += pairwise_no_back_to_back(
            config[t], config[t + 1], action_t, action_t1, weights,
        )
        score += pairwise_alternation(
            config[t], config[t + 1], net_crossings[t],
            rally.team_assignments, weights,
        )
        score += pairwise_absent_pair(config[t], config[t + 1], weights)
    # Higher-order
    score += higher_3_contact_per_side(
        config, net_crossings, rally.team_assignments, weights,
    )
    score += higher_serve_first(
        config[0], rally.serving_team, rally.team_assignments, weights,
    )
    return score


def _logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _exhaustive_map(
    rally: RallyContext, weights: FactorWeights,
) -> tuple[tuple[State, ...], float, list[dict[State, float]]]:
    """Enumerate all 6^N configurations; return MAP + score + marginals."""
    n = len(rally.contacts)
    state_domain = build_state_domain(rally.team_assignments)
    evidence = build_evidence(rally, weights)
    net_crossings = _net_crossings_for(rally.contacts)

    best_score = -math.inf
    best_config: tuple[State, ...] | None = None
    # Per-(contact, state) logsumexp accumulator for marginals.
    per_contact_state_log_sum: list[dict[State, float]] = [
        {state: -math.inf for state in state_domain} for _ in range(n)
    ]

    for config in itertools.product(state_domain, repeat=n):
        score = score_joint_config(config, rally, weights, evidence, net_crossings)
        if score > best_score:
            best_score, best_config = score, config
        # Accumulate marginals (logsumexp per (contact_t, state))
        for t, state in enumerate(config):
            cur = per_contact_state_log_sum[t][state]
            per_contact_state_log_sum[t][state] = _logaddexp(cur, score)

    assert best_config is not None
    # Normalize marginals per contact
    marginals: list[dict[State, float]] = []
    for t in range(n):
        log_sum_total = -math.inf
        for state in state_domain:
            log_sum_total = _logaddexp(log_sum_total, per_contact_state_log_sum[t][state])
        marginal: dict[State, float] = {}
        for state in state_domain:
            log_p = per_contact_state_log_sum[t][state] - log_sum_total
            marginal[state] = math.exp(log_p) if log_p > -50 else 0.0
        # Renormalize to handle floating-point drift
        total = sum(marginal.values())
        if total > 0:
            marginal = {s: p / total for s, p in marginal.items()}
        marginals.append(marginal)

    return best_config, best_score, marginals


def _coordinate_descent_map(
    rally: RallyContext, weights: FactorWeights, max_iter: int = 5,
) -> tuple[tuple[State, ...], float, list[dict[State, float]]]:
    """Coordinate-descent fallback for N > 8 contacts.

    Initialize from initial_actions PIDs; iteratively re-assign each contact
    to maximize the joint score given others. Approximate; finds local MAP.
    Marginals are not computed; returned as uniform-ish (each MAP state at 1.0).
    """
    n = len(rally.contacts)
    state_domain = build_state_domain(rally.team_assignments)
    evidence = build_evidence(rally, weights)
    net_crossings = _net_crossings_for(rally.contacts)

    # Initialize from initial_actions (or first state in the domain if missing)
    config: list[State] = [state_domain[0] for _ in range(n)]
    for t in range(n):
        if t < len(rally.initial_actions):
            pid = rally.initial_actions[t].get("playerTrackId")
            if pid is not None and 1 <= int(pid) <= 4:
                config[t] = f"P{int(pid)}"

    score = score_joint_config(tuple(config), rally, weights, evidence, net_crossings)
    for _ in range(max_iter):
        improved = False
        for t in range(n):
            best_state, best_score = config[t], score
            for state in state_domain:
                if state == config[t]:
                    continue
                new_config = config[:t] + [state] + config[t + 1:]
                new_score = score_joint_config(
                    tuple(new_config), rally, weights, evidence, net_crossings,
                )
                if new_score > best_score:
                    best_state, best_score = state, new_score
            if best_state != config[t]:
                config[t] = best_state
                score = best_score
                improved = True
        if not improved:
            break

    final = tuple(config)
    marginals = [
        {state: (1.0 if state == final[t] else 0.0) for state in state_domain}
        for t in range(n)
    ]
    return final, score, marginals


def joint_attribute_rally(
    rally: RallyContext, weights: FactorWeights = DEFAULT_WEIGHTS,
) -> RallyAttribution:
    """Public API: compute joint MAP attribution for a rally.

    Uses exhaustive enumeration when N <= 8 contacts; coordinate-descent
    fallback for larger rallies. Returns the MAP assignment, joint score,
    and per-contact marginals.
    """
    n = len(rally.contacts)
    if n == 0:
        return RallyAttribution(
            map=(), score=0.0, marginals=[], fallback_used="exhaustive",
        )
    if n <= _EXHAUSTIVE_THRESHOLD:
        config, score, marginals = _exhaustive_map(rally, weights)
        return RallyAttribution(
            map=config, score=score, marginals=marginals, fallback_used="exhaustive",
        )
    config, score, marginals = _coordinate_descent_map(rally, weights)
    return RallyAttribution(
        map=config, score=score, marginals=marginals, fallback_used="coordinate_descent",
    )
