"""Populate playerCandidates in stored contacts for existing tracked rallies.

Re-runs contact detection to generate player_candidates (with court-space
ranking when calibration available), then updates contacts_json in the DB.

Usage:
    cd analysis
    uv run python scripts/populate_player_candidates.py
    uv run python scripts/populate_player_candidates.py --dry-run
    uv run python scripts/populate_player_candidates.py --video <video-id>
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, cast

from rich.console import Console

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate playerCandidates in DB")
    parser.add_argument("--dry-run", action="store_true", help="Don't update DB")
    parser.add_argument("--video", type=str, help="Specific video ID")
    args = parser.parse_args()

    # Load all tracked rallies with contacts
    where_clause = ""
    query_params: list[str] = []
    if args.video:
        where_clause = "AND r.video_id = %s"
        query_params = [args.video]

    query = f"""
        SELECT r.id, r.video_id,
               pt.id as pt_id,
               pt.ball_positions_json,
               pt.positions_json,
               pt.contacts_json,
               pt.actions_json,
               pt.court_split_y,
               pt.frame_count
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.contacts_json IS NOT NULL
          AND pt.positions_json IS NOT NULL
          AND pt.ball_positions_json IS NOT NULL
          {where_clause}
        ORDER BY r.video_id, r.start_ms
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, query_params)
            rows = cur.fetchall()

    console.print(f"[bold]Populating playerCandidates[/bold] for {len(rows)} rallies")

    # Load calibrators per video
    video_ids = {str(row[1]) for row in rows}
    calibrators: dict[str, CourtCalibrator | None] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    n_calibrated = sum(1 for c in calibrators.values() if c is not None)
    console.print(f"  Court calibration: {n_calibrated}/{len(video_ids)} videos\n")

    updated = 0
    skipped = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        rally_id = str(row[0])
        video_id = str(row[1])
        pt_id = cast(int, row[2])
        bp_json = cast(list[dict[str, Any]] | None, row[3])
        pos_json = cast(list[dict[str, Any]] | None, row[4])
        contacts_json = cast(dict[str, Any] | None, row[5])
        actions_json = cast(dict[str, Any] | None, row[6])
        court_split_y = row[7]
        frame_count = row[8]

        # Check if already populated
        if contacts_json:
            existing_contacts = contacts_json.get("contacts", [])
            if existing_contacts and "playerCandidates" in existing_contacts[0]:
                if existing_contacts[0]["playerCandidates"]:
                    skipped += 1
                    continue

        # Parse ball positions
        ball_positions: list[BallPosition] = []
        if bp_json:
            for bp in bp_json:
                ball_positions.append(BallPosition(
                    frame_number=bp.get("frameNumber", 0),
                    x=bp.get("x", 0.0),
                    y=bp.get("y", 0.0),
                    confidence=bp.get("confidence", 0.0),
                ))

        # Parse player positions
        player_positions: list[PlayerPosition] = []
        if pos_json:
            for p in pos_json:
                player_positions.append(PlayerPosition(
                    frame_number=p.get("frameNumber", 0),
                    track_id=p.get("trackId", -1),
                    x=p.get("x", 0.0),
                    y=p.get("y", 0.0),
                    width=p.get("width", 0.0),
                    height=p.get("height", 0.0),
                    confidence=p.get("confidence", 0.0),
                ))

        # Parse team assignments from actions
        team_assignments: dict[int, int] | None = None
        if actions_json and "teamAssignments" in actions_json:
            ta_raw = actions_json["teamAssignments"]
            team_assignments = {}
            for tid_str, team_label in ta_raw.items():
                team_int = 0 if team_label == "A" else 1 if team_label == "B" else -1
                if team_int >= 0:
                    team_assignments[int(tid_str)] = team_int

        # Re-run contact detection
        calibrator = calibrators.get(video_id)
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=court_split_y,
            frame_count=frame_count,
            team_assignments=team_assignments,
            court_calibrator=calibrator,
        )

        # Serialize new contacts
        new_contacts_json = contact_seq.to_dict()

        # Verify candidates populated
        n_with_candidates = sum(
            1 for c in new_contacts_json.get("contacts", [])
            if c.get("playerCandidates")
        )
        n_contacts = len(new_contacts_json.get("contacts", []))

        cal_str = "+cal" if calibrator else ""
        console.print(
            f"  [{i + 1}/{len(rows)}] {rally_id[:8]}: "
            f"{n_contacts} contacts, {n_with_candidates} with candidates{cal_str}"
        )

        if not args.dry_run:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE player_tracks SET contacts_json = %s WHERE id = %s",
                        [json.dumps(new_contacts_json), pt_id],
                    )
                conn.commit()

        updated += 1

    elapsed = time.time() - t0
    console.print(
        f"\n[green]Done[/green]: {updated} updated, {skipped} already had candidates "
        f"({elapsed:.1f}s)"
    )
    if args.dry_run:
        console.print("[yellow]Dry run — no DB changes made[/yellow]")


if __name__ == "__main__":
    main()
