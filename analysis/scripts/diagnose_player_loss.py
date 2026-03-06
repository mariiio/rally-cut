#!/usr/bin/env python3
"""Diagnose rallies where player filtering results in <4 primary tracks.

Loads stored tracking data from DB and re-runs the filter logic to count
how many rallies end up with fewer than 4 primary players.

Usage:
    uv run python scripts/diagnose_player_loss.py --all
    uv run python scripts/diagnose_player_loss.py --all --only-3   # Only show <4
    uv run python scripts/diagnose_player_loss.py --all --verbose  # Show filter details
"""

import argparse
import logging
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_filter import (
    CourtFilterConfig,
    PlayerFilterConfig,
    compute_ball_proximity_scores,
    compute_track_stats,
    detect_referee_tracks,
    identify_primary_tracks,
)
from rallycut.tracking.player_tracker import PlayerPosition


def load_all_tracked_rallies() -> list[dict[str, Any]]:
    """Load all rallies with tracking data from DB."""
    query = """
        SELECT
            r.id as rally_id,
            r.video_id,
            v.filename,
            r.start_ms,
            r.end_ms,
            pt.positions_json,
            pt.primary_track_ids,
            pt.ball_positions_json,
            pt.fps
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.positions_json IS NOT NULL
          AND pt.status = 'COMPLETED'
          AND r.rejection_reason IS NULL
        ORDER BY v.filename, r.start_ms
    """
    results = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            for row in rows:
                (
                    rally_id, video_id, filename, start_ms, end_ms,
                    positions_json, primary_track_ids, ball_positions_json,
                    fps,
                ) = row
                results.append({
                    "rally_id": rally_id,
                    "video_id": video_id,
                    "filename": filename,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "positions_json": positions_json,
                    "primary_track_ids": primary_track_ids,
                    "ball_positions_json": ball_positions_json,
                    "fps": fps or 30.0,
                })

    return results


def rerun_filter(rally: dict[str, Any]) -> tuple[set[int], set[int]]:
    """Re-run filter logic on stored positions, return (new_primary, referee_tracks)."""
    pos_json = cast(list[dict[str, Any]], rally["positions_json"])
    positions = [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            confidence=p["confidence"],
        )
        for p in pos_json
    ]

    # Parse ball positions
    ball_positions: list[BallPosition] = []
    bp_json = cast(list[dict[str, Any]] | None, rally["ball_positions_json"])
    if bp_json:
        for bp in bp_json:
            ball_positions.append(BallPosition(
                frame_number=bp["frameNumber"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp["confidence"],
            ))

    # Compute total frames from positions
    frames = {p.frame_number for p in positions}
    total_frames = max(frames) - min(frames) + 1 if frames else 1

    config = PlayerFilterConfig()
    court_config = CourtFilterConfig()

    track_stats = compute_track_stats(positions, total_frames)

    # Compute ball proximity scores
    if ball_positions:
        prox_scores = compute_ball_proximity_scores(positions, ball_positions, config)
        for tid, score in prox_scores.items():
            if tid in track_stats:
                track_stats[tid].ball_proximity_score = score

    referee_tracks = detect_referee_tracks(track_stats, ball_positions, config)
    primary = identify_primary_tracks(track_stats, config, court_config, referee_tracks)

    return primary, referee_tracks


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose player loss in tracked rallies")
    parser.add_argument("--all", action="store_true", required=True, help="Analyze all tracked rallies")
    parser.add_argument("--only-3", action="store_true", help="Only show rallies with <4 primary tracks")
    parser.add_argument("--verbose", action="store_true", help="Show filter details")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="  %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="  %(message)s")

    print("Loading tracked rallies from DB...")
    rallies = load_all_tracked_rallies()
    print(f"Found {len(rallies)} tracked rallies\n")

    total = 0
    under4_stored = 0
    under4_new = 0
    readmitted = 0

    print(f"{'Rally ID':<40} {'Video':<20} {'Stored':>6} {'New':>6} {'Refs':>5} {'Notes'}")
    print("-" * 100)

    for rally in rallies:
        total += 1
        stored_ids = rally["primary_track_ids"] or []
        stored_count = len(set(stored_ids))

        new_primary, referee_tracks = rerun_filter(rally)
        new_count = len(new_primary)

        if stored_count < 4:
            under4_stored += 1
        if new_count < 4:
            under4_new += 1

        notes = []
        if stored_count < 4:
            notes.append(f"STORED<4")
        if new_count < 4:
            notes.append(f"NEW<4")
        if new_count > stored_count:
            notes.append(f"+{new_count - stored_count} recovered")
            readmitted += 1
        if referee_tracks:
            notes.append(f"refs={sorted(referee_tracks)}")

        if args.only_3 and stored_count >= 4 and new_count >= 4:
            continue

        filename = (rally["filename"] or "?")[:20]
        notes_str = ", ".join(notes) if notes else ""
        print(f"{rally['rally_id']:<40} {filename:<20} {stored_count:>6} {new_count:>6} {len(referee_tracks):>5}  {notes_str}")

    print("-" * 100)
    print(f"\nSummary ({total} rallies):")
    print(f"  Stored <4 players: {under4_stored} ({under4_stored*100/total:.1f}%)")
    print(f"  New    <4 players: {under4_new} ({under4_new*100/total:.1f}%)")
    print(f"  Recovered (new > stored): {readmitted}")
    if under4_stored > under4_new:
        print(f"  Improvement: {under4_stored - under4_new} fewer rallies with <4 players")
    elif under4_stored == under4_new:
        print(f"  No change in <4 player count")
    else:
        print(f"  REGRESSION: {under4_new - under4_stored} more rallies with <4 players")


if __name__ == "__main__":
    main()
