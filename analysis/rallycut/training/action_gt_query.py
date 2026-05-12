"""Single source of truth for reading rally_action_ground_truth.

Every analysis script that used to query the legacy player_tracks GT column
should import from here. Each label dict carries the legacy field names so most
consumers don't need to change beyond the import + load call.
"""
from __future__ import annotations

from typing import Any

import psycopg


def load_for_rallies(
    conn: psycopg.Connection[tuple[Any, ...]],
    rally_ids: list[str],
    *,
    include_unresolved: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Return rallyId -> list of label dicts ordered by frame.

    By default only rows with resolved_track_id IS NOT NULL are returned
    (training/eval should never train on unresolved labels). Set
    include_unresolved=True for audit/integrity scripts that need to see
    everything.
    """
    action_reverse = {
        "SERVE": "serve", "RECEIVE": "receive", "SET": "set",
        "ATTACK": "attack", "BLOCK": "block", "DIG": "dig",
    }
    where = "gt.rally_id = ANY(%s)"
    if not include_unresolved:
        where += " AND gt.resolved_track_id IS NOT NULL"

    out: dict[str, list[dict[str, Any]]] = {}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT gt.rally_id, gt.frame, gt.action,
                   gt.resolved_track_id, gt.snapshot_track_id,
                   gt.snapshot_ball_x, gt.snapshot_ball_y, gt.snapshot_team,
                   gt.snapshot_bbox_x1, gt.snapshot_bbox_y1,
                   gt.snapshot_bbox_x2, gt.snapshot_bbox_y2,
                   gt.resolved_source
              FROM rally_action_ground_truth gt
             WHERE {where}
             ORDER BY gt.rally_id, gt.frame
            """,
            (rally_ids,),
        )
        for row in cur.fetchall():
            (rid, frame, action_enum, resolved_tid, snap_tid,
             ball_x, ball_y, team,
             bx1, by1, bx2, by2,
             resolved_source) = row
            bbox = (
                [float(bx1), float(by1), float(bx2), float(by2)]
                if bx1 is not None else None
            )
            label = {
                "frame": int(frame),
                "action": action_reverse.get(action_enum, str(action_enum).lower()),
                # Legacy-compatible names so existing consumers don't break.
                "trackId": int(snap_tid) if snap_tid is not None else None,
                "playerTrackId": int(resolved_tid) if resolved_tid is not None else None,
                "ballX": float(ball_x) if ball_x is not None else None,
                "ballY": float(ball_y) if ball_y is not None else None,
                # New fields for callers that want them.
                "snapshotBbox": bbox,
                "snapshotTeam": team,
                "resolvedSource": resolved_source,
            }
            out.setdefault(str(rid), []).append(label)
    return out


def load_for_videos(
    conn: psycopg.Connection[tuple[Any, ...]],
    video_ids: list[str],
    *,
    include_unresolved: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Convenience wrapper that resolves video_ids -> rally_ids first."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM rallies WHERE video_id = ANY(%s)",
            (video_ids,),
        )
        rally_ids = [str(row[0]) for row in cur.fetchall()]
    return load_for_rallies(conn, rally_ids, include_unresolved=include_unresolved)
