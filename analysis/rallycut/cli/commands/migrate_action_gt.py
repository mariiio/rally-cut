"""Backfill rally_action_ground_truth from PlayerTrack.action_ground_truth_json."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import typer
from rich.console import Console

from rallycut.evaluation.db import get_connection

ACTION_MAP: dict[str, str] = {
    "serve": "SERVE",
    "receive": "RECEIVE",
    "set": "SET",
    "attack": "ATTACK",
    "block": "BLOCK",
    "dig": "DIG",
}

console = Console()


def _bbox_at(
    raw_positions: list[dict[str, Any]] | None,
    frame: int,
    track_id: int,
) -> dict[str, float] | None:
    if not raw_positions:
        return None
    for p in raw_positions:
        if p.get("frameNumber") == frame and p.get("trackId") == track_id:
            bbox = p.get("bbox")
            if isinstance(bbox, dict) and all(k in bbox for k in ("x1", "y1", "x2", "y2")):
                return {
                    "x1": float(bbox["x1"]),
                    "y1": float(bbox["y1"]),
                    "x2": float(bbox["x2"]),
                    "y2": float(bbox["y2"]),
                }
            x, y, w, h = p.get("x"), p.get("y"), p.get("width"), p.get("height")
            if x is not None and y is not None and w is not None and h is not None:
                return {
                    "x1": float(x),
                    "y1": float(y),
                    "x2": float(x) + float(w),
                    "y2": float(y) + float(h),
                }
    return None


def _ball_at(
    ball_positions: list[dict[str, Any]] | None,
    frame: int,
) -> tuple[float, float] | None:
    if not ball_positions:
        return None
    for b in ball_positions:
        if b.get("frameNumber") == frame:
            x, y = b.get("x"), b.get("y")
            if x is not None and y is not None:
                return float(x), float(y)
    return None


def _team_for(
    track_id: int,
    primary_track_ids: list[int] | None,
    team_assignments: dict[str, str] | None,
) -> str | None:
    if not primary_track_ids or track_id not in primary_track_ids:
        return None
    if not team_assignments:
        return None
    t = team_assignments.get(str(track_id))
    return t if t in ("A", "B") else None


def build_label_row(
    label: dict[str, Any],
    raw_positions: list[dict[str, Any]] | None,
    positions: list[dict[str, Any]] | None,
    ball_positions: list[dict[str, Any]] | None,
    primary_track_ids: list[int] | None,
    team_assignments: dict[str, str] | None,
) -> dict[str, Any] | None:
    frame = label.get("frame")
    action_raw = label.get("action")
    if frame is None or not isinstance(action_raw, str):
        return None
    action_enum = ACTION_MAP.get(action_raw.lower())
    if action_enum is None:
        return None

    track_id = label.get("trackId")
    player_track_id = label.get("playerTrackId")
    ball = _ball_at(ball_positions, frame)

    # Stage 1: try raw_positions by trackId (raw BoT-SORT id, preferred when present).
    bbox: dict[str, float] | None = None
    resolved_track_id: int | None = None
    if track_id is not None:
        bbox = _bbox_at(raw_positions, frame, int(track_id))
        if bbox is not None:
            resolved_track_id = int(track_id)

    # Stage 2: fall back to positions by playerTrackId (canonical pid, post-remap).
    # Either when trackId was absent OR when raw_positions lookup didn't find a bbox.
    if bbox is None and player_track_id is not None:
        bbox = _bbox_at(positions, frame, int(player_track_id))
        if bbox is not None:
            resolved_track_id = int(player_track_id)

    if bbox is not None:
        resolved_source = "SNAPSHOT_EXACT"
    else:
        resolved_source = "UNRESOLVED"

    # Team: prefer the resolved id when we found a bbox; else try whichever id is present.
    team_id_for_lookup = resolved_track_id if resolved_track_id is not None else (
        int(track_id) if track_id is not None else (int(player_track_id) if player_track_id is not None else None)
    )
    team = (
        _team_for(team_id_for_lookup, primary_track_ids, team_assignments)
        if team_id_for_lookup is not None else None
    )

    # snapshot_track_id: preserve the labeler's hint. Prefer trackId (raw) over playerTrackId (canonical).
    snapshot_track_id_value = (
        int(track_id) if track_id is not None
        else (int(player_track_id) if player_track_id is not None else None)
    )

    return {
        "frame": int(frame),
        "action": action_enum,
        "snapshot_bbox_x1": bbox["x1"] if bbox else None,
        "snapshot_bbox_y1": bbox["y1"] if bbox else None,
        "snapshot_bbox_x2": bbox["x2"] if bbox else None,
        "snapshot_bbox_y2": bbox["y2"] if bbox else None,
        "snapshot_ball_x": ball[0] if ball else label.get("ballX"),
        "snapshot_ball_y": ball[1] if ball else label.get("ballY"),
        "snapshot_team": team,
        "snapshot_track_id": snapshot_track_id_value,
        "resolved_track_id": resolved_track_id,
        "resolved_source": resolved_source,
    }


def backfill_video(conn: psycopg.Connection, video_id: str) -> dict[str, int]:
    report: dict[str, int] = {
        "snapshot_exact": 0,
        "unresolved": 0,
        "already_present": 0,
        "skipped_no_gt": 0,
    }

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pt.rally_id,
                   pt.action_ground_truth_json,
                   pt.raw_positions_json,
                   pt.positions_json,
                   pt.ball_positions_json,
                   pt.primary_track_ids,
                   pt.actions_json
              FROM player_tracks pt
              JOIN rallies r ON r.id = pt.rally_id
             WHERE r.video_id = %s AND pt.action_ground_truth_json IS NOT NULL
            """,
            (video_id,),
        )
        tracks = cur.fetchall()

    for rally_id, gt, raw, positions, ball, primary, actions_json in tracks:
        if not gt:
            report["skipped_no_gt"] += 1
            continue
        team_assignments: dict[str, str] | None = None
        if actions_json and isinstance(actions_json, dict):
            ta = actions_json.get("teamAssignments")
            if isinstance(ta, dict):
                team_assignments = ta
        for label in gt:
            row = build_label_row(label, raw, positions, ball, primary, team_assignments)
            if row is None:
                continue
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rally_action_ground_truth (
                        id, rally_id, frame, action,
                        snapshot_bbox_x1, snapshot_bbox_y1, snapshot_bbox_x2, snapshot_bbox_y2,
                        snapshot_ball_x, snapshot_ball_y, snapshot_team, snapshot_track_id,
                        resolved_track_id, resolved_at, resolved_source,
                        created_at, updated_at, created_by
                    ) VALUES (
                        gen_random_uuid(), %s::uuid, %s, %s::"ActionLabel",
                        %s, %s, %s, %s,
                        %s, %s, %s::"ServingTeam", %s,
                        %s, NOW(), %s::"ResolveSource",
                        NOW(), NOW(), NULL
                    )
                    ON CONFLICT (rally_id, frame, action) DO NOTHING
                    RETURNING id
                    """,
                    (
                        rally_id,
                        row["frame"],
                        row["action"],
                        row["snapshot_bbox_x1"],
                        row["snapshot_bbox_y1"],
                        row["snapshot_bbox_x2"],
                        row["snapshot_bbox_y2"],
                        row["snapshot_ball_x"],
                        row["snapshot_ball_y"],
                        row["snapshot_team"],
                        row["snapshot_track_id"],
                        row["resolved_track_id"],
                        row["resolved_source"],
                    ),
                )
                inserted = cur.fetchone()
                if inserted is None:
                    report["already_present"] += 1
                elif row["resolved_source"] == "SNAPSHOT_EXACT":
                    report["snapshot_exact"] += 1
                else:
                    report["unresolved"] += 1
        conn.commit()

    return report


def dump_backup(conn: psycopg.Connection, out_path: str | Path) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rally_id, action_ground_truth_json FROM player_tracks"
            " WHERE action_ground_truth_json IS NOT NULL"
        )
        rows = cur.fetchall()
    payload = {str(rally_id): gt for rally_id, gt in rows}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    return len(payload)


def migrate_action_gt_cmd(
    video_id: str | None = typer.Option(None, "--video-id", help="Single video UUID to backfill"),
    all_videos: bool = typer.Option(False, "--all", help="Backfill every video with GT JSON"),
    backup: bool = typer.Option(
        False, "--backup", help="Write pre-cutover JSON dump before backfilling"
    ),
) -> None:
    """Backfill rally_action_ground_truth from PlayerTrack.action_ground_truth_json."""
    if not video_id and not all_videos:
        raise typer.BadParameter("Specify --video-id or --all")

    with get_connection() as conn:
        if backup:
            ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            backup_path = Path(f"backups/action_gt_pre_cutover_{ts}.json")
            n = dump_backup(conn, backup_path)
            console.print(f"[backup] wrote {n} rallies to {backup_path}")

        video_ids: list[str]
        if video_id:
            video_ids = [video_id]
        else:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT r.video_id FROM player_tracks pt
                      JOIN rallies r ON r.id = pt.rally_id
                     WHERE pt.action_ground_truth_json IS NOT NULL
                    """
                )
                video_ids = [str(r[0]) for r in cur.fetchall()]

        total: dict[str, int] = {
            "snapshot_exact": 0,
            "unresolved": 0,
            "already_present": 0,
            "skipped_no_gt": 0,
        }
        for i, vid in enumerate(video_ids, 1):
            r = backfill_video(conn, str(vid))
            for k, v in r.items():
                total[k] += v
            console.print(f"[{i}/{len(video_ids)}] {vid}: {r}")

        console.print(f"[done] totals: {total}")
