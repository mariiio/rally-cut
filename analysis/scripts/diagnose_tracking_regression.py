#!/usr/bin/env python3
"""Diagnose tracking regression by retracking specific rallies.

Portable across commits: queries DB directly, no project-specific evaluation imports.

Run from analysis/ directory:
    uv run python scripts/diagnose_tracking_regression.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# Enable detailed tracking logs
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
for name in [
    "rallycut.tracking.player_tracker",
    "rallycut.tracking.player_filter",
    "rallycut.tracking.spatial_consistency",
    "rallycut.tracking.tracklet_link",
]:
    logging.getLogger(name).setLevel(logging.DEBUG)
for name in ["ultralytics", "boxmot", "PIL", "urllib3", "botocore", "boto3", "s3transfer"]:
    logging.getLogger(name).setLevel(logging.WARNING)


RALLIES = [
    ("27edd9d4-6d44-4802-bdc2-b4e23093e599", "7ba71fc4-R4-flickering"),
    ("f2e47246-7041-4afa-9581-b11d4ac3989d", "ce4c67a1-R5-bg-steal"),
]

DB_DSN = "host=localhost port=5436 dbname=rallycut user=postgres password=postgres"


def query_rally(rally_id: str) -> dict | None:
    """Query rally info directly from PostgreSQL."""
    import psycopg
    conn = psycopg.connect(DB_DSN)
    cur = conn.cursor()
    cur.execute(
        """SELECT r.start_ms, r.end_ms, v.s3_key, v.content_hash
           FROM rallies r JOIN videos v ON v.id = r.video_id
           WHERE r.id = %s""",
        (rally_id,),
    )
    row = cur.fetchone()
    cur.close()

    # Also get calibration (column may not exist in older schema)
    cal_row = None
    try:
        cur2 = conn.cursor()
        cur2.execute(
            """SELECT calibration FROM videos v
               JOIN rallies r ON r.video_id = v.id
               WHERE r.id = %s""",
            (rally_id,),
        )
        cal_row = cur2.fetchone()
        cur2.close()
    except Exception:
        conn.rollback()
    conn.close()

    if row is None:
        return None
    return {
        "start_ms": row[0],
        "end_ms": row[1],
        "s3_key": row[2],
        "content_hash": row[3],
        "calibration": cal_row[0] if cal_row and cal_row[0] else None,
    }


def resolve_video(s3_key: str, content_hash: str) -> Path:
    """Resolve video from MinIO/S3 cache."""
    from rallycut.evaluation.video_resolver import VideoResolver
    resolver = VideoResolver()
    return resolver.resolve(s3_key, content_hash)


def retrack_rally(rally_id: str, label: str) -> None:
    print(f"\n{'='*70}")
    print(f"RETRACKING: {label} ({rally_id})")
    print(f"{'='*70}\n")

    info = query_rally(rally_id)
    if info is None:
        print(f"ERROR: Rally {rally_id} not found")
        return

    video_path = resolve_video(info["s3_key"], info["content_hash"])

    # Build calibrator
    calibrator = None
    if info["calibration"]:
        try:
            from rallycut.court.calibration import CourtCalibrator
            corners = info["calibration"]
            if isinstance(corners, str):
                corners = json.loads(corners)
            if isinstance(corners, list) and len(corners) >= 4:
                calibrator = CourtCalibrator()
                image_corners = [(c["x"], c["y"]) for c in corners[:4]]
                calibrator.calibrate(image_corners)
        except Exception as e:
            print(f"Warning: calibration failed: {e}")

    from rallycut.tracking.ball_tracker import create_ball_tracker
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    tracker = PlayerTracker()
    config = PlayerFilterConfig()

    print(f"Video: {video_path.name}")
    print(f"Rally: {info['start_ms']}ms - {info['end_ms']}ms")
    print(f"Calibration: {'yes' if calibrator else 'no'}")

    # Run ball tracking first (matches app pipeline)
    print("Running ball tracking...")
    ball_tracker = create_ball_tracker(model="wasb")
    ball_result = ball_tracker.track_video(
        video_path, start_ms=info["start_ms"], end_ms=info["end_ms"]
    )
    ball_positions = ball_result.positions
    n_ball = sum(1 for bp in ball_positions if bp.confidence > 0)
    total_ball = len(ball_positions)
    print(f"Ball detection: {n_ball}/{total_ball} ({100*n_ball/total_ball:.0f}%)" if total_ball else "No ball positions")
    print()

    result = tracker.track_video(
        video_path=video_path,
        start_ms=info["start_ms"],
        end_ms=info["end_ms"],
        filter_enabled=True,
        filter_config=config,
        court_calibrator=calibrator,
        ball_positions=ball_positions,
    )

    # Summarize
    positions = result.positions
    track_ids = set(p.track_id for p in positions)
    primary = result.primary_track_ids or set()

    print(f"\n--- RESULTS: {label} ---")
    print(f"Total positions: {len(positions)}")
    print(f"Unique track IDs: {sorted(track_ids)}")
    print(f"Primary tracks: {sorted(primary)}")

    frames_per_track: dict[int, list[int]] = defaultdict(list)
    for p in positions:
        frames_per_track[p.track_id].append(p.frame_number)

    print(f"\nPer-track frame counts:")
    for tid in sorted(frames_per_track):
        frames = sorted(frames_per_track[tid])
        is_primary = "PRIMARY" if tid in primary else ""
        print(
            f"  Track {tid:3d}: {len(frames):4d} frames "
            f"(range {frames[0]}-{frames[-1]}) {is_primary}"
        )

    # Check for over-tracking
    frame_tracks: dict[int, set[int]] = defaultdict(set)
    for p in positions:
        frame_tracks[p.frame_number].add(p.track_id)

    overtrack_frames = [f for f, tids in frame_tracks.items() if len(tids) > 4]
    if overtrack_frames:
        print(f"\nFrames with >4 tracks: {len(overtrack_frames)}")
        for f in sorted(overtrack_frames)[:5]:
            print(f"  Frame {f}: tracks {sorted(frame_tracks[f])}")

    # Quality report
    if result.quality_report:
        qr = result.quality_report
        print(f"\nQuality report:")
        import dataclasses as _dc
        if _dc.is_dataclass(qr):
            for f in _dc.fields(qr):
                print(f"  {f.name}: {getattr(qr, f.name)}")
        elif isinstance(qr, dict):
            for k, v in sorted(qr.items()):
                print(f"  {k}: {v}")

    # Save for comparison
    out_path = Path(f"/tmp/tracking_diag_{label}.json")
    data = [
        {"track_id": p.track_id, "frame": p.frame_number,
         "x": round(p.x, 4), "y": round(p.y, 4)}
        for p in positions
    ]
    out_path.write_text(json.dumps(data, indent=2))
    print(f"\nSaved {len(data)} positions to {out_path}")


def main():
    for rally_id, label in RALLIES:
        retrack_rally(rally_id, label)


if __name__ == "__main__":
    main()
