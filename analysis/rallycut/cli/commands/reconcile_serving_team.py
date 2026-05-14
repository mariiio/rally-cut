"""CLI: rallycut reconcile-serving-team <video-id> [--dry-run]

Stage 2 of upstream team-fix.

The rally-level `servingTeam` is computed by `_find_serving_team_by_formation`
from player positions at rally start (before any ball is in play). This is a
ROBUST signal — it doesn't depend on contact detection, picker output, or any
downstream attribution. When it disagrees with the serve action's `team`
(derived from picker + teamAssignments), the picker is the one we should
question.

This command:
  1. Reads each rally's `servingTeam` (formation) and `actions[0].team` (picker).
  2. For mismatches with both signals well-defined:
     - Find primary tracks at the serve frame on the formation-predicted team.
     - Re-attribute the serve to the nearest one (by image distance).
  3. Writes the corrected serve attribution.

After applying, re-run:
  - rallycut reattribute-actions <video-id>
to propagate the corrected serve through the chain logic.

Conservative gates:
  - Skip rallies where `servingTeam` is None or the serve action is missing.
  - Skip when the serve action's `team` is "unknown" or absent (the team
    couldn't be derived even from the wrong picker pick).
  - Skip when no primary track on the formation team has a bbox within ±5
    frames of the serve contact (the formation team isn't tracked there).
  - Skip when the serve is synthetic AND has no server identity (player_track_id < 0).
"""

from __future__ import annotations

import json
import math
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection

console = Console()

FRAME_TOLERANCE = 5


def _player_to_ball_dist_2d(p: dict[str, Any], ball_x: float, ball_y: float) -> float:
    """Image-space distance from player to ball, using bbox upper-quarter.

    Mirrors `contact_detector._player_to_ball_dist`'s bbox fallback (used when
    wrist keypoints are unavailable). The upper-quarter (`y - height*0.25`)
    approximates head/shoulder position — closer to where contact happens
    than the bbox center, and robust to bboxes whose lower portion extends
    off-screen (a common pattern at frame edges)."""
    px = float(p.get("x", 0))
    py = float(p.get("y", 0)) - float(p.get("height", 0)) * 0.25
    return math.hypot(px - ball_x, py - ball_y)


def _find_team_candidate_at_frame(
    positions: list[dict[str, Any]],
    target_frame: int,
    target_team: str,
    team_assignments: dict[str, str],
    primary_track_ids: list[int],
    ball_x: float,
    ball_y: float,
) -> tuple[int, float] | None:
    """Among primary tracks on `target_team`, return the nearest-to-ball one
    in the [target_frame-5, target_frame+30] window. Returns (track_id, dist)
    or None if no candidate."""
    candidates: dict[int, tuple[float, int]] = {}  # tid -> (dist, frame_offset)
    primary_set = {int(t) for t in primary_track_ids}
    for p in positions:
        tid = int(p.get("trackId", -1))
        if tid not in primary_set:
            continue
        if team_assignments.get(str(tid)) != target_team:
            continue
        f = int(p.get("frameNumber", -10**9))
        offset = f - target_frame
        if offset < -FRAME_TOLERANCE or offset > 30:
            continue
        d = _player_to_ball_dist_2d(p, ball_x, ball_y)
        prior = candidates.get(tid)
        if prior is None or abs(offset) < abs(prior[1]):
            candidates[tid] = (d, offset)
    if not candidates:
        return None
    # Pick the candidate with smallest distance.
    best_tid = min(candidates.items(), key=lambda kv: kv[1][0])
    return best_tid[0], best_tid[1][0]


def reconcile_serving_team_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Re-attribute serves where formation servingTeam disagrees with picker."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Reconciling serve attributions for video {video_id}…[/dim]"
        )

    rally_query = """
        SELECT
            r.id AS rally_id,
            r."order" AS rally_order,
            pt.primary_track_ids,
            pt.positions_json,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s AND r.status = 'CONFIRMED'
        ORDER BY r."order"
    """

    n_fixed = 0
    n_no_mismatch = 0
    n_skipped = 0
    n_no_candidate = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        for row in rows:
            rally_id = cast(str, row[0])
            rally_order = cast(int, row[1])
            primary_raw = row[2]
            positions_json = row[3]
            actions_json = row[4]
            if not isinstance(primary_raw, list) or not isinstance(actions_json, dict):
                n_skipped += 1
                continue
            ta = actions_json.get("teamAssignments")
            if not isinstance(ta, dict):
                n_skipped += 1
                continue
            serving_team = actions_json.get("servingTeam")
            actions = actions_json.get("actions") or []
            if not actions:
                n_skipped += 1
                continue
            # Find the SERVE action (usually the first).
            serve_idx = -1
            for i, a in enumerate(actions):
                if isinstance(a, dict) and a.get("action", "").upper() == "SERVE":
                    serve_idx = i
                    break
            if serve_idx < 0:
                n_skipped += 1
                continue
            serve = actions[serve_idx]
            serve_team = serve.get("team")
            serve_tid = int(serve.get("playerTrackId", -1))
            if (
                serving_team not in ("A", "B")
                or serve_team not in ("A", "B")
                or serving_team == serve_team
            ):
                n_no_mismatch += 1
                continue
            # We have a mismatch. servingTeam (formation) != serve.team (picker).
            primary = [int(t) for t in primary_raw]
            positions = positions_json if isinstance(positions_json, list) else []
            ball_x = float(serve.get("ballX") or 0)
            ball_y = float(serve.get("ballY") or 0)
            serve_frame = int(serve.get("frame", 0))

            result = _find_team_candidate_at_frame(
                positions=positions,
                target_frame=serve_frame,
                target_team=serving_team,
                team_assignments=ta,
                primary_track_ids=primary,
                ball_x=ball_x,
                ball_y=ball_y,
            )
            if result is None:
                n_no_candidate += 1
                if not quiet:
                    console.print(
                        f"  [yellow][no-cand][/yellow] rally r{rally_order+1} "
                        f"(id={rally_id[:8]}): no primary track on team {serving_team} "
                        f"near serve frame {serve_frame}"
                    )
                continue
            new_tid, new_dist = result
            if new_tid == serve_tid:
                n_no_mismatch += 1
                continue

            if dry_run:
                if not quiet:
                    console.print(
                        f"  [cyan][DRY][/cyan]    rally r{rally_order+1} "
                        f"(id={rally_id[:8]}): servingTeam={serving_team} but "
                        f"serve P{serve_tid} team={serve_team} → swap to "
                        f"P{new_tid} (dist={new_dist:.3f})"
                    )
                n_fixed += 1
                continue

            # Apply: rewrite serve's playerTrackId and team.
            new_actions = list(actions)
            new_serve = dict(serve)
            new_serve["playerTrackId"] = new_tid
            new_serve["team"] = serving_team
            new_actions[serve_idx] = new_serve
            new_actions_json = dict(actions_json)
            new_actions_json["actions"] = new_actions

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s "
                        "WHERE rally_id = %s",
                        [json.dumps(new_actions_json), rally_id],
                    )
                conn.commit()
                n_fixed += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]    rally r{rally_order+1} "
                        f"(id={rally_id[:8]}): serve P{serve_tid} team={serve_team} → "
                        f"P{new_tid} team={serving_team} (dist={new_dist:.3f})"
                    )
            except Exception as exc:
                conn.rollback()
                console.print(
                    f"  [red][err][/red] rally r{rally_order+1}: write failed: {exc}"
                )

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] "
        f"{n_fixed} fixed · {n_no_mismatch} no-mismatch · "
        f"{n_no_candidate} no-candidate · {n_skipped} skipped"
    )
    if not dry_run and n_fixed > 0:
        console.print(
            f"\n[bold yellow]Next step:[/bold yellow] re-run\n"
            f"  rallycut reattribute-actions {video_id}\n"
            f"to propagate the corrected serve through the chain logic."
        )
