"""Coherence-driven contact recovery (Sub-2.B).

Pure logic for `rallycut recover-missed-contacts <video-id>`. Sibling to
`coherence_invariants.py` (Sub-2.A, detection-only). Re-runs contact
detection inside the gap window implied by each coherence violation,
filters candidates with conservative two-signal gates plus a hard team-
match check, and accepts a recovery iff the rally's coherence-violation
count strictly decreases.

Spec: docs/superpowers/specs/2026-05-10-coherence-driven-contact-recovery-design.md
"""

from __future__ import annotations

from dataclasses import dataclass

_POSSESSION_END_ACTIONS = frozenset({"attack", "serve"})


@dataclass(frozen=True)
class Gap:
    """A window in the rally where a missing contact may live.

    Attributes:
        rule: Coherence rule code that produced this gap ("C-1" / "C-2" / "C-3").
        lo: Lower frame bound (inclusive). For C-3 this is the rally's first
            tracked frame; otherwise it's the prior action's frame.
        hi: Upper frame bound (inclusive). For C-1/C-2 this is the next
            same-team action's frame; for C-3 it's the first detected
            action's frame.
        expected_team: "A" or "B" — the team whose contact's absence
            produced the violation. None for C-3 (any team can serve).
        expected_action: "serve" if the rule forces a specific action type
            (C-3 only). None otherwise; the recovery uses MS-TCN++ argmax.
    """

    rule: str
    lo: int
    hi: int
    expected_team: str | None
    expected_action: str | None


def _team_for(action: dict, team_assignments: dict[str, str]) -> str | None:
    tid = action.get("playerTrackId")
    if tid is None:
        return None
    label = team_assignments.get(str(tid))
    return label if label in ("A", "B") else None


def _other(team: str) -> str:
    return "B" if team == "A" else "A"


def derive_gaps_from_actions(
    *,
    actions: list[dict],
    team_assignments: dict[str, str],
    rally_start_frame: int,
) -> list[Gap]:
    """Walk the actions list once and emit a Gap per coherence-rule firing.

    Mirrors `coherence_invariants.check_c{1,2,3}_*` but emits the gap window
    instead of a Violation. Defensive: returns [] if any action is missing a
    valid team label (matches the orchestrator skip semantics in
    `coherence_invariants.run_all`).
    """
    if not actions:
        return []
    sorted_actions = sorted(actions, key=lambda a: int(a.get("frame", 0)))
    gaps: list[Gap] = []

    # C-3
    first = sorted_actions[0]
    if str(first.get("action", "")) != "serve":
        gaps.append(
            Gap(rule="C-3", lo=rally_start_frame, hi=int(first["frame"]),
                expected_team=None, expected_action="serve")
        )

    if len(sorted_actions) < 2:
        return gaps

    # C-1 + C-2 require team labels for every action.
    teams = [_team_for(a, team_assignments) for a in sorted_actions]
    if any(t is None for t in teams):
        return gaps  # skip C-1/C-2 (orchestrator-equivalent)

    # C-1: emit when we hit the 4th consecutive same-team action.
    streak_team: str | None = None
    streak_frames: list[int] = []
    for a, t in zip(sorted_actions, teams):
        f = int(a["frame"])
        if t == streak_team:
            streak_frames.append(f)
        else:
            streak_team = t
            streak_frames = [f]
        if len(streak_frames) == 4:
            assert streak_team is not None
            gaps.append(
                Gap(rule="C-1", lo=streak_frames[2], hi=streak_frames[3],
                    expected_team=_other(streak_team), expected_action=None)
            )

    # C-2: scan for possession-end transitions that didn't actually transfer.
    contacts_in_possession = 0
    for i, (a, t) in enumerate(zip(sorted_actions, teams)):
        if i == 0:
            contacts_in_possession = 1
            continue
        prev = sorted_actions[i - 1]
        prev_team = teams[i - 1]
        prev_action = str(prev.get("action", ""))
        possession_ended = (
            prev_action in _POSSESSION_END_ACTIONS or contacts_in_possession >= 3
        )
        if possession_ended:
            if t == prev_team:
                # Same team continued after possession end — gap fires.
                assert prev_team is not None
                gaps.append(
                    Gap(rule="C-2", lo=int(prev["frame"]), hi=int(a["frame"]),
                        expected_team=_other(prev_team), expected_action=None)
                )
            contacts_in_possession = 1
        else:
            contacts_in_possession = 1 if t != prev_team else contacts_in_possession + 1

    return gaps
