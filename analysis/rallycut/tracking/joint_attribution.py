"""Joint Attribution PGM — replaces reattribute_players with per-rally
joint MAP inference over 6 states per contact (4 players + 2 absent-team).

Public API: joint_attribute_rally(rally) -> RallyAttribution

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

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
