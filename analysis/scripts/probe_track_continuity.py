"""Probe track-id continuity in player tracking output.

Tests the hypothesis that a single post-filter track_id in `positions_json`
is composed of frames that originated from multiple raw BoT-SORT tracks
in `raw_positions_json` — i.e., the player_filter stitched two distinct
physical players' detections into one filtered track.

Method:
  - Load positions_json (post-filter, post-remap) and raw_positions_json
    (pre-filter, BoT-SORT raw IDs) for one rally.
  - For each post-filter track_id, walk its frames in order. At each frame,
    find the raw track whose bbox center is closest to the filtered bbox
    center. That raw track is the "evidence track" for the filtered track
    at that frame.
  - Report: filtered_track → ordered list of (frame_range, evidence_raw_track,
    centroid). If a filtered track flips between distinct evidence tracks
    mid-rally, that is direct evidence of stitching.
  - Also report large frame-to-frame spatial jumps in positions_json
    (likely teleports / discontinuities).

Read-only DB access. No data is written.

Usage:
    cd analysis
    uv run python scripts/probe_track_continuity.py --rally <rally-uuid>
    uv run python scripts/probe_track_continuity.py --rally efdbf6b2-54a8-4444-8135-17d91baf977b
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any

from rallycut.evaluation.db import get_connection

# A "spatial jump" in normalized coords: bbox-center distance between
# consecutive frames of the same track exceeds this threshold.
JUMP_THRESHOLD = 0.10  # 10% of frame width — flags clear teleports

# Match radius for assigning a filtered-track frame to a raw track:
# the bbox center of the closest raw detection must be within this distance.
MATCH_RADIUS = 0.05  # 5% of frame width — generous, BoT-SORT raw should be very close


def _center(p: dict[str, Any]) -> tuple[float, float]:
    return (p["x"] + p["width"] / 2.0, p["y"] + p["height"] / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _frame_range(frames: list[int]) -> str:
    if not frames:
        return "(empty)"
    return f"{frames[0]}-{frames[-1]} ({len(frames)} frames)"


def _runs_strict(positions: list[dict[str, Any]], evidence: dict[int, int | None]) -> list[dict[str, Any]]:
    """Build runs of (frame, evidence_raw_track) using actual frameNumbers, not positional approximation."""
    if not positions:
        return []
    sorted_pos = sorted(positions, key=lambda p: p["frameNumber"])
    runs: list[dict[str, Any]] = []
    cur_track = evidence.get(sorted_pos[0]["frameNumber"])
    cur_start = sorted_pos[0]["frameNumber"]
    cur_end = sorted_pos[0]["frameNumber"]
    cur_count = 1
    for p in sorted_pos[1:]:
        f = p["frameNumber"]
        ev = evidence.get(f)
        if ev == cur_track:
            cur_end = f
            cur_count += 1
        else:
            runs.append({
                "raw_track": cur_track,
                "start_frame": cur_start,
                "end_frame": cur_end,
                "count": cur_count,
            })
            cur_track = ev
            cur_start = f
            cur_end = f
            cur_count = 1
    runs.append({
        "raw_track": cur_track,
        "start_frame": cur_start,
        "end_frame": cur_end,
        "count": cur_count,
    })
    return runs


def analyze_rally(
    rally_id: str,
    positions: list[dict[str, Any]],
    raw_positions: list[dict[str, Any]],
    primary_track_ids: list[int],
) -> dict[str, Any]:
    # Group filtered positions by trackId
    by_filtered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in positions:
        by_filtered[p["trackId"]].append(p)
    for tid in by_filtered:
        by_filtered[tid].sort(key=lambda p: p["frameNumber"])

    # Group raw positions by frameNumber (so we can search per frame)
    raw_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in raw_positions:
        raw_by_frame[p["frameNumber"]].append(p)

    # All raw track ids and their frame range
    raw_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in raw_positions:
        raw_by_track[p["trackId"]].append(p)
    for tid in raw_by_track:
        raw_by_track[tid].sort(key=lambda p: p["frameNumber"])

    rally_report: dict[str, Any] = {
        "rally_id": rally_id,
        "primary_track_ids": list(primary_track_ids),
        "total_frames": len({p["frameNumber"] for p in positions}),
        "filtered_track_ids": sorted(by_filtered.keys()),
        "raw_track_count": len(raw_by_track),
        "raw_track_summary": {
            int(tid): {
                "frame_count": len(frames),
                "frame_range": [frames[0]["frameNumber"], frames[-1]["frameNumber"]],
            }
            for tid, frames in sorted(raw_by_track.items())
        },
        "filtered_tracks": [],
    }

    print(f"\n=== Rally {rally_id} ===")
    print(f"  primary_track_ids: {primary_track_ids}")
    print(f"  positions_json: {len(positions)} entries, {len(by_filtered)} distinct trackIds: {sorted(by_filtered.keys())}")
    print(f"  raw_positions_json: {len(raw_positions)} entries, {len(raw_by_track)} distinct raw trackIds")
    print(f"  Raw tracks: {sorted(raw_by_track.keys())}")
    print()

    for filtered_tid in sorted(by_filtered.keys()):
        frames_data = by_filtered[filtered_tid]
        evidence: dict[int, int | None] = {}  # frameNumber -> closest raw track id
        match_distances: list[float] = []
        no_match_count = 0

        # 1. Spatial-jump detection
        jumps: list[dict[str, Any]] = []
        for i in range(len(frames_data) - 1):
            cur = frames_data[i]
            nxt = frames_data[i + 1]
            d = _dist(_center(cur), _center(nxt))
            if d > JUMP_THRESHOLD:
                jumps.append({
                    "frame_pair": [cur["frameNumber"], nxt["frameNumber"]],
                    "frame_gap": nxt["frameNumber"] - cur["frameNumber"],
                    "from_xy": [round(cur["x"], 4), round(cur["y"], 4)],
                    "to_xy": [round(nxt["x"], 4), round(nxt["y"], 4)],
                    "distance": round(d, 4),
                })

        # 2. Per-frame raw-track association
        for fp in frames_data:
            fnum = fp["frameNumber"]
            fc = _center(fp)
            best_tid: int | None = None
            best_d = float("inf")
            for rp in raw_by_frame.get(fnum, []):
                d = _dist(fc, _center(rp))
                if d < best_d:
                    best_d = d
                    best_tid = rp["trackId"]
            if best_tid is not None and best_d <= MATCH_RADIUS:
                evidence[fnum] = best_tid
                match_distances.append(best_d)
            else:
                evidence[fnum] = None
                no_match_count += 1

        # 3. Group into runs by evidence raw track
        runs = _runs_strict(frames_data, evidence)
        unique_evidence_tracks = {r["raw_track"] for r in runs if r["raw_track"] is not None}

        is_primary = filtered_tid in primary_track_ids
        is_chimera = len(unique_evidence_tracks) > 1
        avg_match_d = (sum(match_distances) / len(match_distances)) if match_distances else None

        track_report = {
            "filtered_track_id": filtered_tid,
            "is_primary": is_primary,
            "frame_count": len(frames_data),
            "frame_range": [frames_data[0]["frameNumber"], frames_data[-1]["frameNumber"]],
            "spatial_jumps": jumps,
            "evidence_raw_tracks": sorted(unique_evidence_tracks),
            "is_chimera": is_chimera,
            "no_raw_match_count": no_match_count,
            "avg_match_distance": round(avg_match_d, 5) if avg_match_d is not None else None,
            "runs": runs,
        }
        rally_report["filtered_tracks"].append(track_report)

        # Print per-track summary as we go
        marker = "[PRIMARY]" if is_primary else "         "
        chimera_flag = " ⚠ CHIMERA" if is_chimera else ""
        print(f"  {marker} filtered T{filtered_tid:>3}: "
              f"{len(frames_data):>3} frames "
              f"[{frames_data[0]['frameNumber']:>3}-{frames_data[-1]['frameNumber']:>3}], "
              f"evidence raw=[{','.join(str(t) for t in sorted(unique_evidence_tracks))}], "
              f"jumps={len(jumps)}, no_match={no_match_count}{chimera_flag}")
        if is_chimera:
            for r in runs:
                if r["count"] > 0:
                    print(f"      run: raw_track={r['raw_track']} frames {r['start_frame']}-{r['end_frame']} ({r['count']} frames)")
        if jumps:
            for j in jumps:
                print(f"      jump: frames {j['frame_pair'][0]}->{j['frame_pair'][1]} "
                      f"({j['frame_gap']}-frame gap), "
                      f"{j['from_xy']} -> {j['to_xy']}, dist={j['distance']:.3f}")

    return rally_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe track continuity / stitching for one rally")
    parser.add_argument("--rally", required=True, help="rally UUID")
    parser.add_argument("--json", action="store_true", help="emit full JSON to stdout after summary")
    args = parser.parse_args()

    rally_id = args.rally

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    pt.rally_id,
                    pt.primary_track_ids,
                    pt.positions_json,
                    pt.raw_positions_json,
                    pt.frame_count
                FROM player_tracks pt
                WHERE pt.rally_id = %s
                """,
                (rally_id,),
            )
            row = cur.fetchone()

    if row is None:
        print(f"ERROR: rally_id {rally_id} not found in player_tracks", file=sys.stderr)
        return 1

    _, primary, positions, raw_positions, frame_count = row
    if not positions:
        print(f"ERROR: rally {rally_id} has empty positions_json", file=sys.stderr)
        return 1
    if not raw_positions:
        print(f"WARN: rally {rally_id} has no raw_positions_json — chimera detection will be limited", file=sys.stderr)
        raw_positions = []

    print(f"Loaded rally {rally_id}: {frame_count} frames, "
          f"{len(positions)} positions, {len(raw_positions)} raw_positions, "
          f"primary_track_ids={primary}")

    report = analyze_rally(rally_id, positions, raw_positions, primary or [])

    chimeras = [t for t in report["filtered_tracks"] if t["is_chimera"]]
    print()
    print(f"=== SUMMARY for {rally_id} ===")
    print(f"  Filtered tracks total: {len(report['filtered_tracks'])}")
    print(f"  Chimera tracks (multi-raw-evidence): {len(chimeras)}")
    if chimeras:
        for c in chimeras:
            print(f"    * filtered T{c['filtered_track_id']} ← raw {c['evidence_raw_tracks']} "
                  f"(primary={c['is_primary']})")

    if args.json:
        print()
        print("=== FULL JSON REPORT ===")
        print(json.dumps(report, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
