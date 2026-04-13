"""Diagnose and optionally repair GT sideSwitches in player_matching_gt_json.

Derives expected side switches from actual player positions (which team is
physically on each side) and compares with stored GT sideSwitches. Reports
discrepancies and optionally updates the DB.

The GT sideSwitches were initially templated from match_analysis_json's
sideSwitchDetected flags, which can be wrong due to trackToPlayer phantom
flips. This script uses position data (always correct) to find the truth.

Usage:
    uv run python scripts/repair_gt_side_switches.py               # dry-run
    uv run python scripts/repair_gt_side_switches.py --apply        # update DB
    uv run python scripts/repair_gt_side_switches.py --video abc123 # one video
"""

from __future__ import annotations

import argparse
import json
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
class VideoSwitchDiag:
    video_id: str
    n_rallies: int
    gt_switches: list[int]
    derived_switches: list[int]
    matches: bool


def _load_videos(
    video_prefix: str | None = None,
) -> list[tuple[str, dict]]:
    """Load videos with player_matching_gt_json."""
    with get_connection() as conn, conn.cursor() as cur:
        if video_prefix:
            cur.execute("""
                SELECT id, player_matching_gt_json FROM videos
                WHERE player_matching_gt_json IS NOT NULL
                  AND id::text LIKE %s
            """, [f"{video_prefix}%"])
        else:
            cur.execute("""
                SELECT id, player_matching_gt_json FROM videos
                WHERE player_matching_gt_json IS NOT NULL
            """)
        return [(vid, gt) for vid, gt in cur.fetchall() if isinstance(gt, dict)]


def _load_rally_order(video_ids: set[str]) -> dict[str, list[str]]:
    """Load rally IDs per video in chronological order."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, video_id FROM rallies
            WHERE video_id IN ({placeholders})
            ORDER BY video_id, start_ms
        """, list(video_ids))
        result: dict[str, list[str]] = {}
        for rid, vid in cur.fetchall():
            result.setdefault(vid, []).append(rid)
        return result


def _load_positions(video_ids: set[str]) -> dict[str, tuple[list[dict], float | None]]:
    """Load positions and court_split_y for rallies."""
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
        return {rid: (pj or [], sy) for rid, pj, sy in cur.fetchall()}


def _near_team_label(
    positions: list[dict],
    t2p: dict[str | int, int] | None,
) -> int | None:
    """Determine which team (0 or 1) is on the near side.

    Team 0 = player IDs 1,2. Team 1 = player IDs 3,4.
    Returns 0 if team 0 players are on near side (higher Y),
    1 if team 1 players are on near side, None if ambiguous.
    """
    if not positions:
        return None

    # Compute avg Y per track
    track_ys: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        tid = p.get("trackId")
        y = p.get("y")
        if tid is not None and y is not None:
            track_ys[int(tid)].append(float(y))

    if len(track_ys) < 4:
        return None

    track_avg_y = {tid: float(np.mean(ys)) for tid, ys in track_ys.items()}

    if t2p is None:
        return None

    # Get player IDs with their average Y
    pid_y: dict[int, float] = {}
    for tid_str, pid in t2p.items():
        tid = int(tid_str)
        if tid in track_avg_y:
            pid_y[int(pid)] = track_avg_y[tid]

    if len(pid_y) < 4:
        return None

    # Average Y for each team
    team0_y = np.mean([pid_y[p] for p in [1, 2] if p in pid_y])
    team1_y = np.mean([pid_y[p] for p in [3, 4] if p in pid_y])

    # Tighter than localize_team_near's 0.03 threshold because this operates
    # on raw pid→Y data (no track_to_player indirection causing phantom flips).
    if abs(team0_y - team1_y) < 0.005:
        return None  # Too close to tell

    return 0 if team0_y > team1_y else 1


def derive_switches(
    rally_ids: list[str],
    positions_by_rally: dict[str, tuple[list[dict], float | None]],
    t2p_by_rally: dict[str, dict[str, int]],
) -> list[int]:
    """Derive side switch indices from position data.

    A switch occurs when the near-side team changes between consecutive rallies.
    """
    switches: list[int] = []
    prev_team: int | None = None

    for idx, rid in enumerate(rally_ids):
        pos_data = positions_by_rally.get(rid)
        t2p = t2p_by_rally.get(rid)
        if pos_data is None:
            continue

        positions, _split_y = pos_data
        team = _near_team_label(positions, t2p)

        if team is not None:
            if prev_team is not None and team != prev_team:
                switches.append(idx)
            prev_team = team

    return switches


def analyze_video(
    video_id: str,
    gt_json: dict,
    rally_ids: list[str],
    positions_by_rally: dict[str, tuple[list[dict], float | None]],
    ma_rallies: list[dict] | None,
) -> VideoSwitchDiag:
    """Analyze one video's side switches."""
    gt_switches = list(gt_json.get("sideSwitches", gt_json.get("side_switches", [])))

    # Build trackToPlayer from match_analysis_json
    t2p_by_rally: dict[str, dict[str, int]] = {}
    if ma_rallies:
        for entry in ma_rallies:
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            t2p = entry.get("trackToPlayer") or entry.get("track_to_player", {})
            if rid and t2p:
                t2p_by_rally[rid] = t2p

    derived = derive_switches(rally_ids, positions_by_rally, t2p_by_rally)

    return VideoSwitchDiag(
        video_id=video_id,
        n_rallies=len(rally_ids),
        gt_switches=gt_switches,
        derived_switches=derived,
        matches=(gt_switches == derived),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair GT sideSwitches")
    parser.add_argument("--apply", action="store_true", help="Update DB (dry-run by default)")
    parser.add_argument("--video", type=str, default=None, help="Process one video")
    args = parser.parse_args()

    print("Loading GT videos...")
    videos = _load_videos(args.video)
    print(f"Found {len(videos)} videos")

    video_ids = {v[0] for v in videos}

    print("Loading rally order...")
    rally_order = _load_rally_order(video_ids)

    print("Loading positions...")
    positions = _load_positions(video_ids)

    print("Loading match_analysis_json...")
    ma_by_video: dict[str, list[dict]] = {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, match_analysis_json FROM videos
            WHERE id IN ({placeholders}) AND match_analysis_json IS NOT NULL
        """, list(video_ids))
        for vid, ma in cur.fetchall():
            if isinstance(ma, dict):
                ma_by_video[vid] = ma.get("rallies", [])

    # Analyze
    results: list[VideoSwitchDiag] = []
    for vid, gt_json in sorted(videos, key=lambda x: x[0]):
        rids = rally_order.get(vid, [])
        if not rids:
            continue
        diag = analyze_video(vid, gt_json, rids, positions, ma_by_video.get(vid))
        results.append(diag)

    # Report
    print(f"\n{'=' * 80}")
    print(f"{'video_id':10s}  {'rallies':>7s}  {'gt_sw':>12s}  {'derived_sw':>12s}  {'match':>5s}")
    print("-" * 80)

    n_mismatch = 0
    for r in results:
        match_str = "OK" if r.matches else "DIFF"
        if not r.matches:
            n_mismatch += 1
        print(f"{r.video_id[:10]:10s}  {r.n_rallies:7d}  {str(r.gt_switches):>12s}  "
              f"{str(r.derived_switches):>12s}  {match_str:>5s}")

    print("-" * 80)
    print(f"Mismatches: {n_mismatch}/{len(results)}")

    if not args.apply:
        if n_mismatch > 0:
            print("\nRun with --apply to update the DB.")
        return 0

    # Apply fixes
    n_updated = 0
    with get_connection() as conn, conn.cursor() as cur:
        for r in results:
            if r.matches:
                continue
            # Read current GT JSON
            cur.execute("SELECT player_matching_gt_json FROM videos WHERE id = %s", [r.video_id])
            row = cur.fetchone()
            if not row or not row[0]:
                continue
            gt = dict(row[0])

            old_sw = gt.get("sideSwitches", gt.get("side_switches", []))
            gt["sideSwitches"] = r.derived_switches
            # Remove snake_case variant if present
            gt.pop("side_switches", None)

            cur.execute(
                "UPDATE videos SET player_matching_gt_json = %s WHERE id = %s",
                [json.dumps(gt), r.video_id],
            )
            n_updated += 1
            print(f"  Updated {r.video_id[:10]}: {old_sw} → {r.derived_switches}")
        conn.commit()

    print(f"\nUpdated {n_updated} videos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
