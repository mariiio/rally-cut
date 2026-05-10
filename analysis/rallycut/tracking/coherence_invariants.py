"""Coherence (game-rule) invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-coherence-invariants`) wires them in alongside the
existing PID-structural audit.

Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams.
  C-3: First action of a rally is `serve`.

Skip semantics: each rule has explicit skip conditions. Additionally, the
orchestrator (`run_all`) excludes rallies that fail any I-1 / I-3 / I-6
PID invariant — those are upstream issues that would produce noisy
downstream coherence violations.

Spec: docs/superpowers/specs/2026-05-10-coherence-invariants-v1-design.md
"""

from __future__ import annotations

from typing import Any

from rallycut.tracking.pid_invariants import Violation


def _team_for_action(
    action: dict[str, Any], team_assignments: dict[str, str]
) -> str | None:
    """Resolve an action to its team label ('A'/'B'). None if undeterminable."""
    tid = action.get("playerTrackId")
    if tid is None:
        return None
    label = team_assignments.get(str(tid))
    if label not in ("A", "B"):
        return None
    return label


def _actions_sorted_by_frame(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Defensive sort by frame — actions in DB may be unordered."""
    return sorted(actions, key=lambda a: int(a.get("frame", 0)))


def check_c1_three_contact_rule(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-1: a team can have at most 3 consecutive contacts before crossing."""
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    current_team: str | None = None
    current_frames: list[int] = []

    for action in sorted_actions:
        team = _team_for_action(action, team_assignments)
        if team is None:
            return []  # defensive — orchestrator should have skipped
        frame = int(action.get("frame", 0))
        if team == current_team:
            current_frames.append(frame)
        else:
            current_team = team
            current_frames = [frame]
        if len(current_frames) == 4:
            # First time we hit 4 consecutive — emit one violation per
            # offending sequence (don't keep emitting at 5, 6, etc.).
            violations.append(
                Violation(
                    invariant="C-1",
                    rally_id=rally_id,
                    detail=(
                        f"team {current_team} had 4 consecutive contacts "
                        f"at frames {current_frames}; max is 3"
                    ),
                )
            )
    return violations
