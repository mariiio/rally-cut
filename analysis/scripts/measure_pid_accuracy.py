"""Measure cross-rally PID assignment accuracy against ground truth.

Uses `videos.player_matching_gt_json` as the source of truth. For each
labeled bbox `{cx, cy, w, h, frame, playerId}`, find the production
track whose position at that frame is closest to the GT bbox center,
look up that track's assigned PID in `match_analysis_json`, and score
whether `matcher_pid == gt_player_id`.

Reports per-rally and overall accuracy. Mismatch lines name the
matcher's mistake at frame granularity, which makes it easy to
classify (within-team swap / cross-team swap / side-switch flip).

Usage:
    uv run python scripts/measure_pid_accuracy.py <video_id>

This script is the canonical objective measurement for cross-rally
identity work. Every code change targeting matcher accuracy should
report a delta in this script's overall percentage.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection, load_rallies_for_video


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    args = p.parse_args()

    rallies = load_rallies_for_video(args.video_id)
    rid_to_rally = {r.rally_id: r for r in rallies}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT player_matching_gt_json, match_analysis_json "
                "FROM videos WHERE id = %s",
                [args.video_id],
            )
            row = cur.fetchone()
            if not row:
                sys.exit(f"video not found: {args.video_id}")
            gt: dict[str, Any] = cast(dict, row[0]) if row[0] else {}
            ma: dict[str, Any] = cast(dict, row[1]) if row[1] else {}
            if isinstance(gt, str):
                gt = json.loads(gt)
            if isinstance(ma, str):
                ma = json.loads(ma)

    if not gt.get("rallies"):
        sys.exit("no player_matching_gt_json — run GT labeling first")

    # Per-rally tracker_id → final PID. Prefer appliedFullMapping when
    # available (post-remap), else fall back to trackToPlayer.
    afm_by_rally: dict[str, dict[int, int]] = {}
    for entry in ma.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id")
        if not rid:
            continue
        src = entry.get("appliedFullMapping") or entry.get("trackToPlayer") or {}
        afm_by_rally[rid] = {
            int(k): int(v) for k, v in src.items() if int(k) > 0
        }

    # Pre-remap raw positions per rally — needed because GT bboxes are
    # in the original video coordinate space and we need to find the
    # original tracker id at the labeled frame.
    positions_by_rally: dict[str, list[dict[str, Any]]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pt.rally_id, pt.pre_remap_state_json FROM player_tracks pt "
                "JOIN rallies r ON pt.rally_id = r.id WHERE r.video_id = %s",
                [args.video_id],
            )
            for rid, snap in cur.fetchall():
                if snap is None:
                    continue
                if isinstance(snap, dict):
                    snap_dict: dict[str, Any] = snap
                else:
                    snap_dict = json.loads(cast(str, snap))
                positions_by_rally[str(rid)] = snap_dict.get("positions") or []

    total = 0
    correct = 0
    per_rally: dict[str, tuple[int, int]] = {}
    mismatches: list[tuple[str, int, int, int]] = []

    for rid, entry in (gt.get("rallies", {}) or {}).items():
        afm = afm_by_rally.get(rid, {})
        pos = positions_by_rally.get(rid, [])
        if not pos or not afm or rid not in rid_to_rally:
            continue
        pos_by_frame: dict[int, list[dict[str, Any]]] = {}
        for q in pos:
            fn = q.get("frameNumber")
            if fn is None:
                continue
            pos_by_frame.setdefault(int(fn), []).append(q)
        rally_total = 0
        rally_correct = 0
        for lbl in entry.get("labels", []):
            gt_pid = int(lbl["playerId"])
            cx, cy = float(lbl["cx"]), float(lbl["cy"])
            gt_frame = int(lbl["frame"])
            candidates = pos_by_frame.get(gt_frame, [])
            if not candidates:
                for delta in range(1, 10):
                    candidates = (
                        pos_by_frame.get(gt_frame - delta, [])
                        + pos_by_frame.get(gt_frame + delta, [])
                    )
                    if candidates:
                        break
            if not candidates:
                continue
            closest = min(
                candidates,
                key=lambda p: (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2,
            )
            track_id = int(closest["trackId"])
            matcher_pid = afm.get(track_id)
            if matcher_pid is None:
                continue
            rally_total += 1
            total += 1
            if matcher_pid == gt_pid:
                rally_correct += 1
                correct += 1
            else:
                mismatches.append((rid[:8], gt_frame, gt_pid, matcher_pid))
        per_rally[rid[:8]] = (rally_correct, rally_total)

    print("Per-rally PID accuracy:")
    for rid_short, (c, t) in sorted(per_rally.items()):
        pct = (100 * c / t) if t else 0
        marker = "" if c == t else "  ←"
        print(f"  {rid_short}: {c}/{t} ({pct:.0f}%){marker}")
    if total:
        print(f"\nOVERALL: {correct}/{total} = {(100 * correct / total):.1f}%")
    else:
        print("\nNo GT labels matched any tracks.")
    if mismatches:
        print("\nMismatches (rally, frame, gt_pid → matcher_pid):")
        for m in mismatches:
            print(f"  {m[0]} frame={m[1]}: GT=PID{m[2]}, matcher=PID{m[3]}")


if __name__ == "__main__":
    main()
