"""PID-attribution invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-pid-invariants`) and the panel eval script wire them in.

Invariants:
  I-1: len(primary_track_ids) == 4 (or 0 if filter disabled)
  I-2: every trackId in positionsJson ∈ primary_track_ids
  I-3: every action's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-4: every contact's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-5: trackToPlayer is total over primary_track_ids (each maps to PID 1..4)
  I-6: team_assignments is total over primary_track_ids (team in {A, B})
  I-7: post-mapping, every action's mapped_track_id ∈ {1..4} ∪ {-1}
       (closed by silent-skip in compute_match_stats — see commit history)
  I-8: when 4 primary tracks all have team labels, partition must be 2A + 2B
       (beach volleyball is definitionally 2v2 — fires on classify_teams
       errors caused by late arrivers or Y-ambiguous starts)

Spec: docs/superpowers/specs/2026-05-08-pid-leverage-audit-sub1-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from rallycut.evaluation.tracking.db import get_connection


@dataclass(frozen=True)
class Violation:
    invariant: str
    rally_id: str
    detail: str
    severity: Literal["error", "warn"] = "error"


def check_i1_primary_set_size(
    *,
    rally_id: str,
    primary_track_ids: list[int],
) -> list[Violation]:
    """I-1: primary_track_ids must have exactly 4 entries (or 0 if filter disabled)."""
    n = len(primary_track_ids)
    if n in (0, 4):
        return []
    return [
        Violation(
            invariant="I-1",
            rally_id=rally_id,
            detail=f"primary_track_ids has size {n}, expected 4 (or 0 if filter disabled)",
        )
    ]


def check_i2_positions_in_primary(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    positions_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-2: every trackId in positions_json must be in primary_track_ids."""
    if not positions_json or not primary_track_ids:
        return []
    primary = set(primary_track_ids)
    offenders: set[int] = set()
    for p in positions_json:
        tid = p.get("trackId")
        if tid is None:
            continue
        if int(tid) not in primary:
            offenders.add(int(tid))
    return [
        Violation(
            invariant="I-2",
            rally_id=rally_id,
            detail=f"positions_json contains non-primary trackId={tid} (primary={sorted(primary)})",
        )
        for tid in sorted(offenders)
    ]


def check_i3_action_attribution(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    actions_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-3: every action's playerTrackId must be in primary ∪ {-1}."""
    if not actions_json or not primary_track_ids:
        return []
    allowed = set(primary_track_ids) | {-1}
    violations: list[Violation] = []
    for idx, a in enumerate(actions_json):
        tid = a.get("playerTrackId")
        if tid is None:
            continue
        if int(tid) not in allowed:
            violations.append(
                Violation(
                    invariant="I-3",
                    rally_id=rally_id,
                    detail=(
                        f"action[{idx}] playerTrackId={tid} not in primary "
                        f"{sorted(primary_track_ids)} ∪ {{-1}} "
                        f"(action={a.get('action')!r}, frame={a.get('frame')})"
                    ),
                )
            )
    return violations


def check_i4_contact_attribution(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    contacts_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-4: every contact's playerTrackId must be in primary ∪ {-1}."""
    if not contacts_json or not primary_track_ids:
        return []
    allowed = set(primary_track_ids) | {-1}
    violations: list[Violation] = []
    for idx, c in enumerate(contacts_json):
        tid = c.get("playerTrackId")
        if tid is None:
            continue
        if int(tid) not in allowed:
            violations.append(
                Violation(
                    invariant="I-4",
                    rally_id=rally_id,
                    detail=(
                        f"contact[{idx}] playerTrackId={tid} not in primary "
                        f"{sorted(primary_track_ids)} ∪ {{-1}} (frame={c.get('frame')})"
                    ),
                )
            )
    return violations


def check_i5_track_to_player_total(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    track_to_player: dict[str, int] | None,
) -> list[Violation]:
    """I-5: trackToPlayer must map every primary_track_id to a PID in {1..4}."""
    if not primary_track_ids:
        return []
    mapping = track_to_player or {}
    # Normalize keys: JSON serializes int keys as strings.
    normalized = {int(k): int(v) for k, v in mapping.items()}
    violations: list[Violation] = []
    for tid in primary_track_ids:
        if tid not in normalized:
            violations.append(
                Violation(
                    invariant="I-5",
                    rally_id=rally_id,
                    detail=f"primary track {tid} missing from trackToPlayer (have keys {sorted(normalized.keys())})",
                )
            )
            continue
        pid = normalized[tid]
        if pid not in (1, 2, 3, 4):
            violations.append(
                Violation(
                    invariant="I-5",
                    rally_id=rally_id,
                    detail=f"trackToPlayer[{tid}]=pid={pid}, expected 1..4",
                )
            )
    return violations


def check_i6_team_assignments_total(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    team_assignments: dict[str, str] | None,
) -> list[Violation]:
    """I-6: team_assignments must label every primary_track_id with team A or B."""
    if not primary_track_ids:
        return []
    mapping = team_assignments or {}
    normalized = {int(k): str(v) for k, v in mapping.items()}
    violations: list[Violation] = []
    for tid in primary_track_ids:
        if tid not in normalized:
            violations.append(
                Violation(
                    invariant="I-6",
                    rally_id=rally_id,
                    detail=f"primary track {tid} missing from team_assignments (have keys {sorted(normalized.keys())})",
                )
            )
            continue
        team = normalized[tid]
        if team not in ("A", "B"):
            violations.append(
                Violation(
                    invariant="I-6",
                    rally_id=rally_id,
                    detail=f"team_assignments[{tid}]={team!r}, expected 'A' or 'B'",
                )
            )
    return violations


def check_i7_stats_canonical_pid(
    *,
    rally_id: str,
    mapped_track_ids: list[int],
) -> list[Violation]:
    """I-7: post-mapping, every action's mapped_track_id must be in {1..4} ∪ {-1}."""
    allowed = {1, 2, 3, 4, -1}
    offenders: set[int] = set()
    for tid in mapped_track_ids:
        if int(tid) not in allowed:
            offenders.add(int(tid))
    return [
        Violation(
            invariant="I-7",
            rally_id=rally_id,
            detail=f"post-mapping mapped_track_id={tid} not in {{1..4}} ∪ {{-1}}",
        )
        for tid in sorted(offenders)
    ]


def check_i8_team_partition_is_2v2(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    team_assignments: dict[str, str] | None,
) -> list[Violation]:
    """I-8: when 4 primary tracks all have team labels, partition must be 2A + 2B.

    Beach volleyball is definitionally 2v2: any rally with 4 detected on-court
    players must have exactly 2 on team A and 2 on team B. A rally where the
    label distribution is 1A+3B, 3A+1B, etc., reflects a classification error
    from `classify_teams` (typically caused by late-arriving tracks or by
    Y-ambiguous rally starts where two players sit on opposite sides of the
    court split by a small margin).

    Skips when:
      - len(primary_track_ids) != 4 — caught by I-1; not enough tracks for a
        2v2 partition (e.g., a player was occluded the entire rally, or an
        off-frame server never re-entered the frame).
      - team_assignments is None or empty — caught by I-6; no labels to check.
      - Any primary track lacks a label — caught by I-6.
      - Any primary's label is not in {"A", "B"} — caught by I-6.

    These skips are intentional: I-8 only fires on cases that survive the
    earlier structural checks. The remaining violations are CORRECTNESS
    failures, not data-quality artifacts. A future producer-side fix in
    `classify_teams` should handle late arrivers (use appearance signal when
    Y data is sparse) and Y-ambiguous starts (use bbox-size or color
    tiebreak); until then, I-8 is the visibility surface.
    """
    if len(primary_track_ids) != 4 or not team_assignments:
        return []
    primary_set = {int(t) for t in primary_track_ids}
    labels: list[str] = []
    for tid in primary_set:
        label = team_assignments.get(str(tid))
        if label not in ("A", "B"):
            # Missing or invalid label — I-6 will catch this.
            return []
        labels.append(label)
    a_count = labels.count("A")
    b_count = labels.count("B")
    if a_count == 2 and b_count == 2:
        return []
    partition_view = {
        str(t): team_assignments.get(str(t)) for t in sorted(primary_set)
    }
    return [
        Violation(
            invariant="I-8",
            rally_id=rally_id,
            detail=(
                f"team partition is {a_count}A+{b_count}B; "
                f"beach volleyball requires 2A+2B "
                f"(team_assignments for primary {sorted(primary_set)} = {partition_view})"
            ),
        )
    ]


def run_all(*, video_id: str) -> list[Violation]:
    """Run all 7 PID invariants against a video's persisted state.

    Loads rallies + player_tracks for `video_id`, plus the video's
    match_analysis_json. Calls each check_iN function and aggregates results.
    """
    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.primary_track_ids,
            pt.positions_json,
            pt.actions_json,
            pt.contacts_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND (r.status = 'CONFIRMED' OR r.status IS NULL)
        ORDER BY r.start_ms
    """
    video_query = "SELECT match_analysis_json FROM videos WHERE id = %s"

    violations: list[Violation] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rally_rows = cur.fetchall()
            cur.execute(video_query, [video_id])
            video_row = cur.fetchone()

    match_analysis: dict[str, Any] = {}
    if video_row and video_row[0]:
        match_analysis = video_row[0] if isinstance(video_row[0], dict) else {}

    rally_to_track_to_player: dict[str, dict[str, int]] = {}
    if isinstance(match_analysis.get("rallies"), list):
        for entry in match_analysis["rallies"]:
            # Some videos persist rally entries with snake_case keys
            # (rally_id / track_to_player) instead of camelCase. Accept either.
            rid = entry.get("rallyId") or entry.get("rally_id")
            ttp = entry.get("trackToPlayer") or entry.get("track_to_player")
            if rid and isinstance(ttp, dict):
                rally_to_track_to_player[rid] = ttp

    for row in rally_rows:
        rally_id = cast(str, row[0])
        primary_raw = row[1]
        positions_json = row[2]
        actions_json = row[3]
        contacts_json = row[4]
        primary_track_ids: list[int] = []
        if isinstance(primary_raw, list):
            primary_track_ids = [int(t) for t in primary_raw]

        actions_list = None
        team_assignments = None
        if isinstance(actions_json, dict):
            actions_list = actions_json.get("actions")
            team_assignments = actions_json.get("teamAssignments")

        track_to_player = rally_to_track_to_player.get(rally_id)

        violations.extend(
            check_i1_primary_set_size(
                rally_id=rally_id, primary_track_ids=primary_track_ids,
            )
        )
        violations.extend(
            check_i2_positions_in_primary(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                positions_json=positions_json if isinstance(positions_json, list) else None,
            )
        )
        violations.extend(
            check_i3_action_attribution(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                actions_json=actions_list,
            )
        )
        violations.extend(
            check_i4_contact_attribution(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                contacts_json=contacts_json if isinstance(contacts_json, list) else None,
            )
        )
        violations.extend(
            check_i5_track_to_player_total(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                track_to_player=track_to_player,
            )
        )
        violations.extend(
            check_i6_team_assignments_total(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                team_assignments=team_assignments,
            )
        )
        violations.extend(
            check_i8_team_partition_is_2v2(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                team_assignments=team_assignments,
            )
        )
        # I-7 is checked using mapped_track_ids derived from actions + track_to_player.
        # An unmapped raw track_id falls through to itself, so we rebuild the same
        # mapping logic compute_match_stats uses.
        if actions_list and track_to_player:
            normalized_ttp = {int(k): int(v) for k, v in track_to_player.items()}
            mapped_track_ids: list[int] = []
            for a in actions_list:
                tid = a.get("playerTrackId")
                if tid is None:
                    continue
                tid_int = int(tid)
                if tid_int == -1:
                    mapped_track_ids.append(-1)
                elif tid_int in normalized_ttp:
                    mapped_track_ids.append(normalized_ttp[tid_int])
                else:
                    mapped_track_ids.append(tid_int)  # fall-through (current bug)
            violations.extend(
                check_i7_stats_canonical_pid(
                    rally_id=rally_id, mapped_track_ids=mapped_track_ids,
                )
            )

    return violations
