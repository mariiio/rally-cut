#!/usr/bin/env python3
"""L4: ball-tracking accuracy at contact probe.

Prerequisite: GT ball positions from analysis/training_datasets/beach_v11/
tracking_ground_truth.json. Reports overlap with trusted-32 upfront.

Schema (found 2026-05-20):
  Top-level: {"rallies": [...], "stats": {...}}
  Each rally: {
    "video_content_hash": str,   # sha-like hash of video file
    "rally_start_ms": int,
    "rally_end_ms": int,
    "ground_truth_json": {
      "positions": [             # players + ball mixed
        {"frameNumber": int,     # absolute frame in full video, at 30 fps
         "label": "ball" | "player_1" | ...,
         "x": float,            # normalized 0-1
         "y": float,
         "confidence": float,
         "trackId": int,
         ...},
        ...
      ],
      "frameCount": int,
      "videoWidth": int,
      "videoHeight": int,
    }
  }

Mapping to DB:
  Rally is matched by (video.content_hash, rally.start_ms, rally.end_ms)
  with a ±5000 ms tolerance window.
  DB ball_positions_json uses rally-relative frame numbers (starts at 0).
  GT frameNumber = DB frameNumber + int(rally_start_ms / 1000 * 30)

For wrong-attribution contacts in the overlap subset:
  Oracle: substitute GT ball position at contact frame, re-score.
  Realistic: ball-confidence-at-contact correlation with attribution error.

Output: reports/upstream_bottleneck_2026_05_20/L4.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    DB_DSN,
    TRUSTED_32,
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

BALL_GT_PATH = (
    ANALYSIS_DIR / "training_datasets" / "beach_v11" / "tracking_ground_truth.json"
)
OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_raw_gt() -> list[dict[str, Any]]:
    if not BALL_GT_PATH.exists():
        print(f"WARNING: {BALL_GT_PATH} not found", flush=True)
        return []
    data = json.loads(BALL_GT_PATH.read_text())
    return data.get("rallies", []) if isinstance(data, dict) else []


def _match_gt_to_db(
    gt_items: list[dict[str, Any]],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Return rally_id -> {db_relative_frame: (x, y)} for matched GT rallies.

    Matching: (video.content_hash, rally.start_ms) within ±5000 ms.
    Frame conversion: gt_frameNumber = db_frame + round(start_ms / 1000 * 30)
    """
    if not gt_items:
        return {}

    # Fetch all trusted-32 rallies from DB
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.content_hash, r.id, r.start_ms, r.end_ms
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            WHERE v.name = ANY(%s)
            """,
            [list(TRUSTED_32)],
        )
        db_rows = cur.fetchall()

    by_hash: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for chash, rid, start_ms, end_ms in db_rows:
        by_hash[str(chash)].append((str(rid), int(start_ms), int(end_ms)))

    out: dict[str, dict[int, tuple[float, float]]] = {}

    for item in gt_items:
        ch = item.get("video_content_hash", "")
        gt_start = int(item.get("rally_start_ms", 0))
        gt_end = int(item.get("rally_end_ms", 0))
        candidates = by_hash.get(ch, [])
        if not candidates:
            continue

        best: tuple[str, int, int] | None = None
        best_dist = float("inf")
        for rid, ds, de in candidates:
            dist = abs(ds - gt_start) + abs(de - gt_end)
            if dist < best_dist:
                best_dist = dist
                best = (rid, ds, de)

        if best is None or best_dist > 5000:
            continue

        rid, db_start_ms, _ = best
        # Frame offset: GT frames are absolute at 30 fps; DB frames are rally-relative.
        frame_offset = round(db_start_ms / 1000.0 * 30)

        positions = item.get("ground_truth_json", {}).get("positions", [])
        per_frame: dict[int, tuple[float, float]] = {}
        for pos in positions:
            if pos.get("label") != "ball":
                continue
            gt_fn = pos.get("frameNumber")
            x = pos.get("x")
            y = pos.get("y")
            if gt_fn is None or x is None or y is None:
                continue
            db_frame = int(gt_fn) - frame_offset
            per_frame[db_frame] = (float(x), float(y))

        if per_frame:
            out[rid] = per_frame

    return out


def load_ball_gt() -> dict[str, dict[int, tuple[float, float]]]:
    """Return rally_id -> {db_relative_frame: (ball_x, ball_y)}."""
    gt_items = _load_raw_gt()
    return _match_gt_to_db(gt_items)


def main() -> int:
    print("Loading ball GT...", flush=True)
    ball_gt = load_ball_gt()
    print(f"  {len(ball_gt)} rallies with ball GT matched to trusted-32", flush=True)

    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    overlap_rows = [r for r in rows if r.rally_id in ball_gt]
    print(f"  {len(overlap_rows)} in ball-GT overlap", flush=True)

    if len(overlap_rows) < 30:
        print(
            f"  WARNING: overlap n={len(overlap_rows)} < 30; L4 oracle "
            f"ceiling under-sampled. Reporting partial corpus.",
            flush=True,
        )

    oracle_recoveries = 0
    total_attempted = 0
    confidence_data: list[tuple[float, bool]] = []

    for i, row in enumerate(overlap_rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue

        # Find closest contact to action_frame (within ±3 frames)
        contact = next(
            (
                c
                for c in rally["contacts"]
                if abs(int(c.get("frame", -1)) - row.action_frame) <= 3
            ),
            None,
        )
        if contact is None:
            continue

        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        if not cand_tids:
            continue

        # Look up GT ball position at or near the action frame
        frames_by_rally = ball_gt[row.rally_id]
        gt_ball: tuple[float, float] | None = None
        oracle_pick: int | None = None
        best_dist = 3
        for df in range(-3, 4):
            candidate_pos = frames_by_rally.get(row.action_frame + df)
            if candidate_pos is not None and abs(df) <= best_dist:
                best_dist = abs(df)
                gt_ball = candidate_pos

        if gt_ball is not None:
            total_attempted += 1
            oracle_pick = rescore_contact(
                rally,
                contact,
                row.action_type,
                cand_tids,
                expected_team=None,
                ball_position_override=gt_ball,
            )
            if oracle_pick == row.gt_pid:
                oracle_recoveries += 1

        # Confidence analysis: DB ball position confidence at contact frame.
        # Only record when oracle was attempted (gt_ball available).
        if gt_ball is not None:
            ball_at_frame = next(
                (
                    b
                    for b in rally["ball_positions"]
                    if abs(int(b.get("frameNumber", -1)) - row.action_frame) <= 1
                ),
                None,
            )
            if ball_at_frame is not None:
                conf = float(ball_at_frame.get("confidence", 0))
                was_recovered = oracle_pick == row.gt_pid
                confidence_data.append((conf, was_recovered))

        if (i + 1) % 10 == 0 or (i + 1) == len(overlap_rows):
            print(
                f"  [{i+1}/{len(overlap_rows)}] processed "
                f"(oracle attempted={total_attempted}, recovered={oracle_recoveries})",
                flush=True,
            )

    # Summarize confidence correlation
    if confidence_data:
        rec_confs = [c for c, r in confidence_data if r]
        not_rec_confs = [c for c, r in confidence_data if not r]
        avg_conf_recovered = sum(rec_confs) / max(len(rec_confs), 1)
        avg_conf_not_recovered = sum(not_rec_confs) / max(len(not_rec_confs), 1)
    else:
        avg_conf_recovered = 0.0
        avg_conf_not_recovered = 0.0

    out = {
        "n_ball_gt_rallies": len(ball_gt),
        "n_total_wrong": len(rows),
        "n_overlap": len(overlap_rows),
        "n_oracle_attempted": total_attempted,
        "oracle_recoveries": oracle_recoveries,
        "avg_ball_confidence_recovered": avg_conf_recovered,
        "avg_ball_confidence_not_recovered": avg_conf_not_recovered,
        "partial_corpus_caveat": len(overlap_rows) < 30,
    }
    out_path = OUT_DIR / "L4.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}", flush=True)
    print(
        f"  oracle recoveries on attempted: {oracle_recoveries}/{total_attempted} "
        f"(overlap n={len(overlap_rows)})",
        flush=True,
    )
    if avg_conf_recovered or avg_conf_not_recovered:
        print(
            f"  avg ball confidence — recovered: {avg_conf_recovered:.3f} "
            f"vs not recovered: {avg_conf_not_recovered:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
