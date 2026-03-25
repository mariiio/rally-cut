"""Re-run contact detection + action classification for all tracked rallies.

Uses stored ball/player positions (from tracking) to re-detect contacts with
the current pipeline (classifier + threshold) and re-classify actions. Saves
updated contacts_json and actions_json back to the database.

This fixes stale stored actions from older pipeline versions.

Usage:
    cd analysis
    uv run python scripts/redetect_all_actions.py                # Dry run
    uv run python scripts/redetect_all_actions.py --apply        # Write to DB
    uv run python scripts/redetect_all_actions.py --video <id>   # Single video
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, cast

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-detect contacts + actions for all rallies")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB (default: dry run)")
    parser.add_argument("--video", type=str, help="Only process this video ID")
    args = parser.parse_args()

    # Load match team assignments and rally data
    where_clauses = ["pt.ball_positions_json IS NOT NULL"]
    params: list[str] = []
    if args.video:
        where_clauses.append("r.video_id = %s")
        params.append(args.video)
    where_sql = " AND ".join(where_clauses)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT v.id, v.match_analysis_json FROM videos v "
                "WHERE v.match_analysis_json IS NOT NULL"
            )
            match_teams_by_rally: dict[str, dict[int, int]] = {}
            for vid, mj_raw in cur.fetchall():
                mj = cast(dict[str, Any], mj_raw)
                if not mj or "team_assignments" not in mj:
                    continue
                for rid, teams in mj.get("team_assignments", {}).items():
                    if teams.get("confidence", 0) >= 0.70:
                        match_teams_by_rally[rid] = {
                            int(tid): t for tid, t in teams.get("assignments", {}).items()
                        }

            cur.execute(f"""
                SELECT r.id, r.video_id, pt.id as pt_id,
                       pt.ball_positions_json, pt.positions_json,
                       pt.frame_count, pt.court_split_y
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE {where_sql}
                ORDER BY r.video_id, r.start_ms
            """, params)
            rows = cur.fetchall()

    print(f"Found {len(rows)} rallies with ball positions")
    if not args.apply:
        print("  DRY RUN — use --apply to write changes to DB\n")

    # Load court calibrations
    calibrators: dict[str, CourtCalibrator | None] = {}

    t_start = time.monotonic()
    updated = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(rows):
        rally_id = str(row[0])
        video_id = str(row[1])
        pt_id = cast(int, row[2])
        ball_json = cast(list[dict[str, Any]], row[3])
        positions_json = cast(list[dict[str, Any]] | None, row[4])
        frame_count = cast(int | None, row[5])
        court_split_y = cast(float | None, row[6])

        # Load court calibration (cached per video)
        if video_id not in calibrators:
            corners = load_court_calibration(video_id)
            if corners and len(corners) == 4:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])
                calibrators[video_id] = cal
            else:
                calibrators[video_id] = None

        # Convert DB dicts to typed objects
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in ball_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        if not ball_positions:
            skipped += 1
            continue

        player_positions = []
        if positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in positions_json
            ]

        match_teams = match_teams_by_rally.get(rally_id)

        try:
            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=court_split_y,
                frame_count=frame_count or None,
                court_calibrator=calibrators.get(video_id),
                team_assignments=match_teams,
            )

            rally_actions = classify_rally_actions(
                contacts, rally_id,
                use_classifier=True,
                match_team_assignments=match_teams,
            )

            # Serialize
            new_contacts_json = contacts.to_dict()
            new_actions_json = {
                "actions": [a.to_dict() for a in rally_actions.actions],
            }

            n_contacts = len(contacts.contacts)
            n_actions = len(rally_actions.actions)

            if args.apply:
                with get_connection() as wconn:
                    with wconn.cursor() as wcur:
                        wcur.execute(
                            "UPDATE player_tracks SET contacts_json = %s, actions_json = %s WHERE id = %s",
                            (json.dumps(new_contacts_json), json.dumps(new_actions_json), pt_id),
                        )
                    wconn.commit()

            updated += 1
            elapsed = time.monotonic() - t_start
            print(
                f"  [{i+1}/{len(rows)}] {rally_id[:8]}: "
                f"{n_contacts} contacts, {n_actions} actions ({elapsed:.1f}s)"
            )

        except Exception as e:
            errors += 1
            print(f"  ERROR {rally_id[:8]}: {e}")

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {updated} updated, {skipped} skipped, {errors} errors ({elapsed:.1f}s)")
    if not args.apply:
        print("  DRY RUN — no changes written. Use --apply to write.")


if __name__ == "__main__":
    main()
