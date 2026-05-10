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

from dataclasses import dataclass, field
from typing import Any, cast

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

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


def _team_for(action: dict[str, Any], team_assignments: dict[str, str]) -> str | None:
    tid = action.get("playerTrackId")
    if tid is None:
        return None
    label = team_assignments.get(str(tid))
    return label if label in ("A", "B") else None


def _other(team: str) -> str:
    return "B" if team == "A" else "A"


def derive_gaps_from_actions(
    *,
    actions: list[dict[str, Any]],
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


@dataclass
class RallyInputs:
    """Everything detect_contacts + get_sequence_probs need for one rally.

    Loaded from the rally's persisted `player_tracks` row + parent rally row.
    """

    rally_id: str
    video_id: str
    rally_start_frame: int
    fps: float
    frame_count: int
    court_split_y: float | None
    ball_positions: list[BallPosition] = field(default_factory=list)
    player_positions: list[PlayerPosition] = field(default_factory=list)
    actions_json: dict[str, Any] = field(default_factory=dict)
    # team_assignments stored two ways for downstream convenience:
    #  - str-keyed "A"/"B" (matches actions_json + coherence_invariants)
    #  - int-keyed 0/1 (matches detect_contacts + get_sequence_probs API)
    team_assignments_str: dict[str, str] = field(default_factory=dict)
    team_assignments_int: dict[int, int] = field(default_factory=dict)
    primary_track_ids: list[int] = field(default_factory=list)


def _ball_position_from_dict(d: dict[str, Any]) -> BallPosition:
    return BallPosition(
        frame_number=int(d.get("frameNumber", d.get("frame", 0))),
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        confidence=float(d.get("confidence", 0.0)),
        motion_energy=float(d.get("motionEnergy", 0.0)),
    )


def _player_position_from_dict(d: dict[str, Any]) -> PlayerPosition:
    return PlayerPosition(
        frame_number=int(d.get("frameNumber", 0)),
        track_id=int(d.get("trackId", -1)),
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        width=float(d.get("width", 0.0)),
        height=float(d.get("height", 0.0)),
        confidence=float(d.get("confidence", 0.0)),
        keypoints=d.get("keypoints"),
    )


def load_rally_inputs(rally_id: str) -> RallyInputs:
    """Load all per-rally inputs needed for recovery from the DB."""
    from rallycut.evaluation.db import get_connection

    query = """
        SELECT
            r.id, r.video_id, r.start_ms,
            pt.fps, pt.frame_count, pt.court_split_y,
            pt.ball_positions_json, pt.positions_json, pt.actions_json,
            pt.primary_track_ids
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [rally_id])
            row = cur.fetchone()
    if row is None:
        raise ValueError(f"No player_tracks row for rally {rally_id}")

    rid = cast(Any, row[0])
    vid = cast(Any, row[1])
    _start_ms = cast(Any, row[2])
    fps = cast(Any, row[3])
    frame_count = cast(Any, row[4])
    court_split_y = cast(Any, row[5])
    bp_json = cast(Any, row[6])
    pp_json = cast(Any, row[7])
    actions_json = cast(Any, row[8])
    primary_raw = cast(Any, row[9])

    ball_positions = [
        _ball_position_from_dict(b) for b in (bp_json or [])
        if isinstance(b, dict)
    ]
    player_positions = [
        _player_position_from_dict(p) for p in (pp_json or [])
        if isinstance(p, dict)
    ]
    aj = cast(dict[str, Any], actions_json or {})
    ta_str_raw = aj.get("teamAssignments")
    ta_str: dict[str, str] = (
        cast(dict[str, str], ta_str_raw)
        if isinstance(ta_str_raw, dict)
        else {}
    )
    ta_int: dict[int, int] = {}
    for k, v in ta_str.items():
        if v == "A":
            ta_int[int(k)] = 0
        elif v == "B":
            ta_int[int(k)] = 1

    primary = [int(t) for t in (primary_raw or [])]

    return RallyInputs(
        rally_id=str(rid),
        video_id=str(vid),
        rally_start_frame=0,  # actions/ball/positions are rally-relative
        fps=float(fps or 30.0),
        frame_count=int(frame_count or 0),
        court_split_y=(float(court_split_y) if court_split_y is not None else None),
        ball_positions=ball_positions,
        player_positions=player_positions,
        actions_json=aj,
        team_assignments_str=ta_str,
        team_assignments_int=ta_int,
        primary_track_ids=primary,
    )
