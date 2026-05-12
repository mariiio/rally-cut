"""Joint Attribution PGM — replaces reattribute_players with per-rally
joint MAP inference over 6 states per contact (4 players + 2 absent-team).

Public API: joint_attribute_rally(rally) -> RallyAttribution

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

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
