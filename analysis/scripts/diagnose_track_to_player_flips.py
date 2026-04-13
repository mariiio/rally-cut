"""Diagnose phantom trackToPlayer flips across rallies.

For each GT video, checks consecutive rally pairs for cases where the
near-side player IDs changed (e.g. {1,2} → {3,4}) without sideSwitchDetected
being set. Cross-references with GT sideSwitches to confirm these are phantom
flips (not real switches).

Usage:
    uv run python scripts/diagnose_track_to_player_flips.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402


@dataclass
class RallyInfo:
    rally_id: str
    rally_index: int
    track_to_player: dict[int, int]  # track_id -> player_id
    side_switch_detected: bool
    near_player_ids: set[int]  # player IDs on the near side (high Y)


@dataclass
class VideoResult:
    video_id: str
    n_rallies: int
    n_flips: int  # phantom flips (no sideSwitchDetected)
    n_real_switches_gt: int  # GT side switches
    n_real_switches_detected: int  # pipeline sideSwitchDetected count
    flip_indices: list[int]  # rally indices where phantom flips occur


def _load_gt_videos() -> dict[str, set[int]]:
    """Load GT video IDs and their sideSwitches."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json
            FROM videos
            WHERE player_matching_gt_json IS NOT NULL
        """)
        result: dict[str, set[int]] = {}
        for vid, gt_json in cur.fetchall():
            if isinstance(gt_json, dict):
                sw = gt_json.get("sideSwitches", gt_json.get("side_switches", []))
                result[vid] = set(sw) if sw else set()
        return result


def _load_match_analysis(video_ids: set[str]) -> dict[str, list[dict]]:
    """Load match_analysis_json rallies for given videos."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, match_analysis_json
            FROM videos
            WHERE id IN ({placeholders})
              AND match_analysis_json IS NOT NULL
        """, list(video_ids))
        result: dict[str, list[dict]] = {}
        for vid, ma_json in cur.fetchall():
            if isinstance(ma_json, dict):
                rallies = ma_json.get("rallies", [])
                if isinstance(rallies, list):
                    result[vid] = rallies
        return result


def _load_rally_positions(
    video_ids: set[str],
) -> dict[str, tuple[list[dict], float | None]]:
    """Load positions_json and court_split_y for all rallies in given videos.

    Returns {rally_id: (positions_json, court_split_y)}.
    """
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT pt.rally_id, pt.positions_json, pt.court_split_y
            FROM player_tracks pt
            JOIN rallies r ON r.id = pt.rally_id
            WHERE r.video_id IN ({placeholders})
        """, list(video_ids))
        result: dict[str, tuple[list[dict], float | None]] = {}
        for rid, pj, split_y in cur.fetchall():
            result[rid] = (pj or [], split_y)
        return result


def _determine_near_player_ids(
    track_to_player: dict[str | int, int],
    positions: list[dict],
    court_split_y: float | None,
) -> set[int]:
    """Determine which player IDs are on the near side using Y positions.

    Near side = higher Y (closer to camera). Uses relative splitting:
    sort tracks by avg Y, take the top 2 as "near". This avoids depending
    on court_split_y which can be wrong for narrow-angle cameras.
    """
    if not track_to_player or not positions:
        return set()

    # Compute average Y per track (only tracks in t2p)
    t2p_tids = {int(k) for k in track_to_player}
    track_ys: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        tid = p.get("trackId")
        y = p.get("y")
        if tid is not None and y is not None and int(tid) in t2p_tids:
            track_ys[int(tid)].append(float(y))

    if len(track_ys) < 4:
        return set()

    track_avg_y: dict[int, float] = {
        tid: float(np.mean(ys)) for tid, ys in track_ys.items()
    }

    # Sort tracks by avg Y descending (highest Y = near court)
    sorted_tracks = sorted(track_avg_y.items(), key=lambda x: x[1], reverse=True)

    # Top 2 by Y = near side
    near_tids = {sorted_tracks[0][0], sorted_tracks[1][0]}

    # Map to player IDs
    near_pids: set[int] = set()
    for tid_str, pid in track_to_player.items():
        tid = int(tid_str)
        if tid in near_tids:
            near_pids.add(int(pid))

    return near_pids


def analyze_video(
    video_id: str,
    rallies: list[dict],
    rally_positions: dict[str, tuple[list[dict], float | None]],
    gt_switches: set[int],
) -> VideoResult:
    """Analyze one video for phantom flips."""
    rally_infos: list[RallyInfo] = []

    for i, entry in enumerate(rallies):
        rid = entry.get("rallyId") or entry.get("rally_id", "")
        t2p = entry.get("trackToPlayer") or entry.get("track_to_player", {})
        ssd = entry.get("sideSwitchDetected") or entry.get(
            "side_switch_detected", False
        )

        positions, split_y = rally_positions.get(rid, ([], None))
        near_pids = _determine_near_player_ids(t2p, positions, split_y)

        rally_infos.append(RallyInfo(
            rally_id=rid,
            rally_index=i,
            track_to_player={int(k): int(v) for k, v in t2p.items()} if t2p else {},
            side_switch_detected=bool(ssd),
            near_player_ids=near_pids,
        ))

    # Count detected switches
    n_detected = sum(1 for r in rally_infos if r.side_switch_detected)

    # Detect phantom flips
    flips: list[int] = []
    for i in range(1, len(rally_infos)):
        prev = rally_infos[i - 1]
        curr = rally_infos[i]

        if not prev.near_player_ids or not curr.near_player_ids:
            continue

        # Check if near-side team changed
        prev_team = 0 if any(p <= 2 for p in prev.near_player_ids) else 1
        curr_team = 0 if any(p <= 2 for p in curr.near_player_ids) else 1

        if prev_team != curr_team:
            # Team on near side changed — is this accounted for?
            if curr.side_switch_detected:
                continue  # Pipeline detected a switch, OK
            if i in gt_switches:
                continue  # GT says there's a switch here

            # Phantom flip: team changed without detection or GT switch
            flips.append(i)

    return VideoResult(
        video_id=video_id,
        n_rallies=len(rally_infos),
        n_flips=len(flips),
        n_real_switches_gt=len(gt_switches),
        n_real_switches_detected=n_detected,
        flip_indices=flips,
    )


def main() -> int:
    print("Loading GT videos...")
    gt_videos = _load_gt_videos()
    print(f"Found {len(gt_videos)} videos with player_matching_gt_json")

    print("Loading match_analysis_json...")
    match_data = _load_match_analysis(set(gt_videos.keys()))
    print(f"Found {len(match_data)} videos with match_analysis_json")

    print("Loading rally positions...")
    rally_positions = _load_rally_positions(set(gt_videos.keys()))
    print(f"Loaded positions for {len(rally_positions)} rallies")

    # Analyze each video
    results: list[VideoResult] = []
    for vid in sorted(gt_videos.keys()):
        if vid not in match_data:
            continue
        result = analyze_video(
            vid, match_data[vid], rally_positions, gt_videos[vid]
        )
        results.append(result)

    # Report
    print(f"\n{'=' * 75}")
    print(f"{'video_id':10s}  {'rallies':>7s}  {'flips':>5s}  {'rate':>6s}  "
          f"{'gt_sw':>5s}  {'det_sw':>6s}  {'flip_indices'}")
    print("-" * 75)

    total_rallies = 0
    total_flips = 0
    affected_videos = 0

    for r in sorted(results, key=lambda x: x.n_flips, reverse=True):
        total_rallies += r.n_rallies
        total_flips += r.n_flips
        if r.n_flips > 0:
            affected_videos += 1

        rate = r.n_flips / max(r.n_rallies - 1, 1) * 100
        idx_str = ",".join(str(i) for i in r.flip_indices[:10])
        if len(r.flip_indices) > 10:
            idx_str += "..."
        print(f"{r.video_id[:10]:10s}  {r.n_rallies:7d}  {r.n_flips:5d}  "
              f"{rate:5.1f}%  {r.n_real_switches_gt:5d}  "
              f"{r.n_real_switches_detected:6d}  {idx_str}")

    print("-" * 75)
    rate = total_flips / max(total_rallies - len(results), 1) * 100
    print(f"{'TOTAL':10s}  {total_rallies:7d}  {total_flips:5d}  {rate:5.1f}%")
    print(f"\nAffected videos: {affected_videos}/{len(results)}")
    print(f"Total phantom flips: {total_flips}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
