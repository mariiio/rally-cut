"""Diagnose bbox height discontinuities in player tracks that suggest identity swaps.

In beach volleyball filmed from a fixed tripod behind the baseline:
  - Near-side players: bbox height ~0.25-0.38 (large, close to camera)
  - Far-side players:  bbox height ~0.10-0.17 (small, far from camera)

When BoT-SORT loses a track during occlusion and revives it on the wrong player,
there is a dramatic height discontinuity. This script finds those gaps and checks
for complementary cross-track discontinuities (Track A: big→small, Track B: small→big
at the same gap), which are strong evidence of a swap.

Usage:
    uv run python scripts/diagnose_height_discontinuity.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rallycut.evaluation.db import get_connection

# ── Tuning constants ─────────────────────────────────────────────────────────
WINDOW = 10              # frames before/after gap to average height
MAX_FRAME_GAP_SMALL = 2  # gaps ≤ this are not "gaps" (normal missing frames)
HEIGHT_CHANGE_THRESH = 0.30   # 30% relative change to flag a discontinuity
CROSS_MATCH_TOL = 0.30        # within 30% for a complementary height pair
GAP_PROXIMITY = 30            # gaps whose start/end overlap within this many frames are considered "same gap"


def _avg_height(positions: list[dict[str, Any]], frame_min: int, frame_max: int) -> float | None:
    """Average bbox height for positions whose frameNumber is in [frame_min, frame_max]."""
    heights = [
        p["height"]
        for p in positions
        if frame_min <= p["frameNumber"] <= frame_max
    ]
    return sum(heights) / len(heights) if heights else None


def _heights_cross_match(
    pre_a: float,
    post_a: float,
    pre_b: float,
    post_b: float,
    tol: float,
) -> bool:
    """Return True if A and B have complementary (swapped) height discontinuities.

    Track A: pre_a (big) → post_a (small)
    Track B: pre_b (small) → post_b (big)
    Cross-check: pre_A ≈ post_B  AND  pre_B ≈ post_A
    """
    def approx(v1: float, v2: float) -> bool:
        ref = max(v1, v2)
        return ref > 0 and abs(v1 - v2) / ref <= tol

    return approx(pre_a, post_b) and approx(pre_b, post_a)


def _rel_change(before: float, after: float) -> float:
    ref = max(before, after)
    return abs(before - after) / ref if ref > 0 else 0.0


def analyze_rally(
    rally_id: str,
    video_filename: str,
    positions_json: list[dict[str, Any]],
    primary_track_ids: list[int],
) -> list[dict[str, Any]]:
    """Analyze one rally for height discontinuities across gaps.

    Returns a list of flagged discontinuity events.
    """
    if not positions_json or not primary_track_ids:
        return []

    # Group positions by track ID, sorted by frame
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in positions_json:
        tid = p.get("trackId")
        if tid is not None:
            by_track[tid].append(p)

    for tid in by_track:
        by_track[tid].sort(key=lambda p: p["frameNumber"])

    # For each primary track, find gaps and measure height before/after
    track_gaps: dict[int, list[dict[str, Any]]] = {}  # track_id → list of gap events

    for tid in primary_track_ids:
        if tid not in by_track:
            continue
        frames = by_track[tid]
        if len(frames) < 2:
            continue

        gaps: list[dict[str, Any]] = []
        for i in range(len(frames) - 1):
            curr_frame = frames[i]["frameNumber"]
            next_frame = frames[i + 1]["frameNumber"]
            gap_size = next_frame - curr_frame - 1

            if gap_size <= MAX_FRAME_GAP_SMALL:
                continue  # not a meaningful gap

            # Compute average height in window before and after gap
            pre_start = max(frames[0]["frameNumber"], curr_frame - WINDOW + 1)
            pre_end = curr_frame
            post_start = next_frame
            post_end = min(frames[-1]["frameNumber"], next_frame + WINDOW - 1)

            h_before = _avg_height(frames, pre_start, pre_end)
            h_after = _avg_height(frames, post_start, post_end)

            if h_before is None or h_after is None:
                continue

            change = _rel_change(h_before, h_after)
            if change < HEIGHT_CHANGE_THRESH:
                continue

            gaps.append({
                "gap_start": curr_frame + 1,   # first missing frame
                "gap_end": next_frame - 1,      # last missing frame
                "gap_size": gap_size,
                "h_before": h_before,
                "h_after": h_after,
                "rel_change": change,
            })

        if gaps:
            track_gaps[tid] = gaps

    if not track_gaps:
        return []

    # Cross-match: look for complementary discontinuities between two tracks
    flagged: list[dict[str, Any]] = []
    primary_list = [tid for tid in primary_track_ids if tid in track_gaps]

    # Check every pair of primary tracks
    matched_pairs: set[tuple[int, int]] = set()

    for i, tid_a in enumerate(primary_list):
        for tid_b in primary_list[i + 1:]:
            for gap_a in track_gaps[tid_a]:
                for gap_b in track_gaps[tid_b]:
                    # Gaps are "at the same location" if they overlap or are within GAP_PROXIMITY frames
                    overlap = (
                        gap_a["gap_start"] <= gap_b["gap_end"] + GAP_PROXIMITY
                        and gap_b["gap_start"] <= gap_a["gap_end"] + GAP_PROXIMITY
                    )
                    if not overlap:
                        continue
                    # Same gap window — check complementary heights
                    cross = _heights_cross_match(
                        gap_a["h_before"], gap_a["h_after"],
                        gap_b["h_before"], gap_b["h_after"],
                        CROSS_MATCH_TOL,
                    )
                    flagged.append({
                        "rally_id": rally_id,
                        "video": video_filename,
                        "track_a": tid_a,
                        "track_b": tid_b,
                        "gap_frame_start": min(gap_a["gap_start"], gap_b["gap_start"]),
                        "gap_frame_end": max(gap_a["gap_end"], gap_b["gap_end"]),
                        "h_before_a": round(gap_a["h_before"], 4),
                        "h_after_a": round(gap_a["h_after"], 4),
                        "h_before_b": round(gap_b["h_before"], 4),
                        "h_after_b": round(gap_b["h_after"], 4),
                        "change_a": round(gap_a["rel_change"], 3),
                        "change_b": round(gap_b["rel_change"], 3),
                        "cross_match": cross,
                    })
                    matched_pairs.add((tid_a, tid_b))

    # Also emit unmatched single-track discontinuities (no complementary gap found)
    matched_tracks: set[int] = set()
    for a, b in matched_pairs:
        matched_tracks.add(a)
        matched_tracks.add(b)

    for tid, gaps in track_gaps.items():
        if tid in matched_tracks:
            continue
        for gap in gaps:
            flagged.append({
                "rally_id": rally_id,
                "video": video_filename,
                "track_a": tid,
                "track_b": None,
                "gap_frame_start": gap["gap_start"],
                "gap_frame_end": gap["gap_end"],
                "h_before_a": round(gap["h_before"], 4),
                "h_after_a": round(gap["h_after"], 4),
                "h_before_b": None,
                "h_after_b": None,
                "change_a": round(gap["rel_change"], 3),
                "change_b": None,
                "cross_match": False,
            })

    return flagged


def main() -> None:
    print("Connecting to database...")

    query = """
        SELECT
            r.id              AS rally_id,
            v.filename        AS video_filename,
            v.fps             AS video_fps,
            pt.positions_json,
            pt.primary_track_ids
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v         ON v.id = r.video_id
        WHERE pt.positions_json IS NOT NULL
          AND pt.primary_track_ids IS NOT NULL
          AND jsonb_array_length(pt.primary_track_ids) > 0
        ORDER BY v.filename, r.start_ms
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    print(f"Loaded {len(rows)} rallies with player tracking data.\n")

    all_flagged: list[dict[str, Any]] = []
    for idx, (rally_id, video_filename, video_fps, positions_json, primary_track_ids) in enumerate(rows):
        short_vid = Path(video_filename).stem if video_filename else rally_id[:8]
        print(f"  [{idx+1:3d}/{len(rows)}] rally={rally_id[:8]} video={short_vid}", end="\r", flush=True)

        if not positions_json or not primary_track_ids:
            continue

        # psycopg returns JSONB as Python objects directly
        pos_list: list[dict[str, Any]] = (
            positions_json if isinstance(positions_json, list) else json.loads(positions_json)
        )
        primary_ids: list[int] = (
            primary_track_ids if isinstance(primary_track_ids, list) else json.loads(primary_track_ids)
        )

        events = analyze_rally(rally_id, video_filename or "", pos_list, primary_ids)
        all_flagged.extend(events)

    print(f"\nDone. Found {len(all_flagged)} discontinuity events total.\n")

    if not all_flagged:
        print("No height discontinuities found. All tracks look stable across gaps.")
        return

    # ── Print results ─────────────────────────────────────────────────────────
    cross_matches = [e for e in all_flagged if e["cross_match"]]
    single_discon = [e for e in all_flagged if not e["cross_match"]]

    print("=" * 100)
    print("CROSS-MATCHED SWAPS (Track A: big→small, Track B: small→big at same gap)")
    print("These are the strongest evidence of identity swaps between near/far players")
    print("=" * 100)

    if not cross_matches:
        print("  (none found)")
    else:
        hdr = (
            f"{'Rally':<10} {'Video':<20} {'TrkA':>5} {'TrkB':>5} "
            f"{'GapStart':>9} {'GapEnd':>7} "
            f"{'preA':>7} {'postA':>7} {'preB':>7} {'postB':>7} "
            f"{'chgA':>6} {'chgB':>6}"
        )
        print(hdr)
        print("-" * 100)
        for e in sorted(cross_matches, key=lambda x: (x["video"], x["gap_frame_start"])):
            vid_short = Path(e["video"]).stem[:20] if e["video"] else e["rally_id"][:8]
            print(
                f"{e['rally_id'][:8]:<10} {vid_short:<20} "
                f"{e['track_a']:>5} {e['track_b']:>5} "
                f"{e['gap_frame_start']:>9} {e['gap_frame_end']:>7} "
                f"{e['h_before_a']:>7.4f} {e['h_after_a']:>7.4f} "
                f"{e['h_before_b']:>7.4f} {e['h_after_b']:>7.4f} "
                f"{e['change_a']:>6.1%} {e['change_b']:>6.1%}"
            )

    print()
    print("=" * 100)
    print("SINGLE-TRACK DISCONTINUITIES (height jump, no complementary partner found)")
    print("=" * 100)

    if not single_discon:
        print("  (none)")
    else:
        hdr = (
            f"{'Rally':<10} {'Video':<20} {'Trk':>5} "
            f"{'GapStart':>9} {'GapEnd':>7} {'GapSz':>6} "
            f"{'hBefore':>8} {'hAfter':>8} {'Change':>7}"
        )
        print(hdr)
        print("-" * 90)
        for e in sorted(single_discon, key=lambda x: (-x["change_a"], x["video"])):
            vid_short = Path(e["video"]).stem[:20] if e["video"] else e["rally_id"][:8]
            gap_sz = e["gap_frame_end"] - e["gap_frame_start"] + 1
            print(
                f"{e['rally_id'][:8]:<10} {vid_short:<20} {e['track_a']:>5} "
                f"{e['gap_frame_start']:>9} {e['gap_frame_end']:>7} {gap_sz:>6} "
                f"{e['h_before_a']:>8.4f} {e['h_after_a']:>8.4f} {e['change_a']:>7.1%}"
            )

    # ── Summary by video ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY BY VIDEO (cross-matched potential swaps)")
    print("=" * 70)
    by_video: dict[str, int] = defaultdict(int)
    for e in cross_matches:
        vid = Path(e["video"]).stem if e["video"] else "unknown"
        by_video[vid] += 1
    if not by_video:
        print("  (no cross-matched swaps)")
    else:
        for vid, count in sorted(by_video.items(), key=lambda x: -x[1]):
            print(f"  {vid:<30} {count:>3} potential swap(s)")

    print()
    print(f"Total:  {len(cross_matches)} cross-matched (potential swaps)")
    print(f"        {len(single_discon)} single-track height jumps")
    print(f"        {len(all_flagged)} total discontinuity events")


if __name__ == "__main__":
    main()
