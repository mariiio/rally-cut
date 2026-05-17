#!/usr/bin/env python3
"""Backfill pose keypoints into positions_json for the trusted corpus.

Reads each rally's existing positions_json + ball_positions_json from DB,
runs enrich_positions_with_pose on contact-frame neighborhoods, writes the
updated positions_json (with keypoints) back to DB.

Avoids re-tracking. Required before training the v2 scorer that uses pose
features at inference time.

Usage:
    cd analysis
    uv run python scripts/backfill_pose_keypoints_2026_05_15.py             # dry-run
    uv run python scripts/backfill_pose_keypoints_2026_05_15.py --apply     # write to DB
    uv run python scripts/backfill_pose_keypoints_2026_05_15.py --video keke --apply
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg

from rallycut.tracking.pose_attribution.pose_cache import (
    _get_pose_model, enrich_positions_with_pose,
)
from rallycut.tracking.player_tracker import PlayerPosition

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
VIDEOS_DIR = Path("/tmp/rca_videos")
WINDOW_HALF = 5


def positions_from_json(positions_json: list[dict]) -> list[PlayerPosition]:
    out = []
    for p in positions_json:
        pos = PlayerPosition(
            frame_number=int(p.get("frameNumber", -1)),
            track_id=int(p.get("trackId", -1)),
            x=float(p.get("x", 0)),
            y=float(p.get("y", 0)),
            width=float(p.get("width", 0)),
            height=float(p.get("height", 0)),
            confidence=float(p.get("confidence", 0)),
        )
        if "keypoints" in p and p["keypoints"]:
            pos.keypoints = p["keypoints"]
        if "embedding" in p and p["embedding"]:
            pos.embedding = p["embedding"]
        out.append(pos)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Write to DB")
    parser.add_argument("--video", type=str, help="Single video codename")
    args = parser.parse_args()

    codenames = (args.video,) if args.video else TRUSTED_CODENAMES
    print(f"Backfilling pose keypoints for {len(codenames)} video(s)…", flush=True)
    print(f"  dry_run: {not args.apply}", flush=True)
    print(f"  videos_dir: {VIDEOS_DIR}", flush=True)

    print("Loading pose model…", flush=True)
    pose_model = _get_pose_model()
    if pose_model is None:
        print("ERROR: pose model unavailable", flush=True)
        return 1
    print(f"Pose model: {type(pose_model).__name__}", flush=True)

    t_start = time.time()
    total_rallies = 0
    total_enriched = 0
    total_written = 0
    with psycopg.connect(DB_DSN) as conn:
        for codename in codenames:
            video_path = VIDEOS_DIR / f"{codename}.mp4"
            if not video_path.exists():
                print(f"  [{codename}] video missing at {video_path} — skipping", flush=True)
                continue
            cur = conn.execute("""
                SELECT r.id, r."order", r.start_ms,
                       pt.positions_json, pt.contacts_json
                FROM videos v JOIN rallies r ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.name = %s AND r.status = 'CONFIRMED'
                  AND pt.positions_json IS NOT NULL
                ORDER BY r."order"
            """, [codename])
            rallies = cur.fetchall()
            print(f"  [{codename}] {len(rallies)} rallies", flush=True)
            n_enriched_video = 0
            for rid, order, start_ms, positions_json_raw, contacts_json_raw in rallies:
                positions = positions_from_json(
                    positions_json_raw if isinstance(positions_json_raw, list) else []
                )
                if not positions:
                    continue
                # Get contact frames (use detected contacts; if missing, fall back
                # to all frames in the rally)
                cj = contacts_json_raw if isinstance(contacts_json_raw, dict) else (
                    json.loads(contacts_json_raw) if isinstance(contacts_json_raw, str) else {}
                )
                contacts = cj.get("contacts") or []
                contact_frames = sorted({int(c.get("frame", -1)) for c in contacts if c.get("frame", -1) >= 0})
                if not contact_frames:
                    # No contacts — skip pose enrichment for this rally
                    continue
                n_enriched = enrich_positions_with_pose(
                    positions, str(video_path), contact_frames,
                    rally_start_ms=int(start_ms or 0),
                    window_half=WINDOW_HALF,
                    pose_model=pose_model,
                )
                if n_enriched == 0:
                    continue
                n_enriched_video += n_enriched
                # Serialize back to JSON
                new_positions_json = [p.to_dict() for p in positions]
                if args.apply:
                    with conn.cursor() as wcur:
                        wcur.execute(
                            "UPDATE player_tracks SET positions_json = %s WHERE rally_id = %s",
                            [json.dumps(new_positions_json), rid],
                        )
                    conn.commit()
                    total_written += 1
                total_rallies += 1
                total_enriched += n_enriched
            print(f"  [{codename}] enriched {n_enriched_video} positions", flush=True)

    elapsed = time.time() - t_start
    print(flush=True)
    print(f"Total: {total_rallies} rallies processed", flush=True)
    print(f"       {total_enriched} (frame, track) positions enriched with keypoints", flush=True)
    if args.apply:
        print(f"       {total_written} rallies written to DB", flush=True)
    else:
        print("       DRY RUN — nothing written", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
