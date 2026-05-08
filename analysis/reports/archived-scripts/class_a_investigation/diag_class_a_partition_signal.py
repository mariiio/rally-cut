"""Test the 'bad partition signal' hypothesis on all 4 Class A cases.

For each case (5c756c41 r02, 7d77980f r16, 7d77980f r18, dd042609 r10),
compute:
- y_mean per primary track (from pre_remap positions)
- y_range across all 4 primaries (max y_mean - min y_mean): tight range
  means y-side signal is weak

A 'tight' y-range (<~0.10) on a 4-primary rally means the y-side
classifier is fitting noise, and the team-pair guard's proposed
partition is unreliable.

Then compares the y-side partition (per track_court_sides if probe
sidecar exists, else by-the-y-mean) against the GT-correct partition
from Phase 2.
"""

from __future__ import annotations

import json
from pathlib import Path

from rallycut.evaluation.tracking.db import get_connection


CASES = [
    # (video_prefix, video_uuid, rally_idx_1based, primary_tids, gt_team_assignment, label)
    ("5c756c41", "5c756c41-1cc1-4486-a95c-97398912cfbe",
     2, [1, 2, 3, 6],
     {1: "?", 2: "near (GT P4)", 3: "?", 6: "?"},
     "5c756c41 r02 (8c49e480)"),
    ("7d77980f", "7d77980f-3006-40e0-adc0-db491a5bb659",
     16, [1, 2, 3, 4],
     {1: "far (GT P4)", 2: "near (GT P1)", 3: "near (GT P2)", 4: "far (GT P3)"},
     "7d77980f r16 (613fde44)"),
    ("7d77980f", "7d77980f-3006-40e0-adc0-db491a5bb659",
     18, [1, 2, 4, 3],
     {1: "far (GT P4)", 2: "near (GT P1)", 3: "near (GT P3)", 4: "far (GT P2)"},
     "7d77980f r18 (1f9ce33a)"),
    ("dd042609", "dd042609-e22e-4f60-83ed-038897c88c32",
     10, [1, 2, 3, 16],
     {1: "far (GT P4)", 2: "near (GT P2)", 3: "far (GT P3)", 16: "near (GT P1)"},
     "dd042609 r10 (f811cc63)"),
]


def list_rallies(video_id: str) -> list[str]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id::text FROM rallies WHERE video_id = %s "
            "AND status='CONFIRMED' ORDER BY start_ms",
            [video_id],
        )
        return [r[0] for r in cur.fetchall()]


def fetch_pre_remap(rally_id: str):
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pt.pre_remap_state_json, pt.raw_positions_json "
            "FROM player_tracks pt WHERE rally_id = %s",
            [rally_id],
        )
        row = cur.fetchone()
        if not row:
            return None
        pre = row[0] or {}
        raw = row[1] or {}
        positions = pre.get("positions") if pre else None
        if not positions:
            positions = raw.get("positions") if raw else None
        return positions or []


def main() -> None:
    for prefix, vid, idx_1, primaries, gt_team, label in CASES:
        print(f"\n{'=' * 70}")
        print(f"=== {label} ===")
        print(f"{'=' * 70}")

        rallies = list_rallies(vid)
        if idx_1 - 1 >= len(rallies):
            print(f"  rally idx out of range ({len(rallies)} rallies)")
            continue
        rally_id = rallies[idx_1 - 1]
        positions = fetch_pre_remap(rally_id)
        if not positions:
            print(f"  no pre_remap positions for rally {rally_id[:8]}")
            continue

        # Y-stats per primary track
        by_track: dict[int, list[tuple[int, float, float]]] = {}
        for p in positions:
            tid = int(p["trackId"])
            if tid <= 0:
                continue
            by_track.setdefault(tid, []).append(
                (int(p["frameNumber"]), float(p["x"]), float(p["y"]))
            )

        y_means = {}
        for tid in primaries:
            pts = by_track.get(tid, [])
            if not pts:
                print(f"  T{tid}: no positions")
                continue
            y_mean = sum(p[2] for p in pts) / len(pts)
            x_mean = sum(p[1] for p in pts) / len(pts)
            y_means[tid] = y_mean
            gt_label = gt_team.get(tid, "?")
            print(f"  T{tid:3d}: n={len(pts):4d}  y_mean={y_mean:.3f}  "
                  f"x_mean={x_mean:.3f}  GT={gt_label}")

        if y_means:
            y_range = max(y_means.values()) - min(y_means.values())
            print(f"\n  y_mean spread across primaries: {y_range:.3f}")
            if y_range < 0.10:
                print(f"  ⚠  TIGHT — y-side classifier likely fitting noise.")
            else:
                print(f"  y-spread is normal (>=0.10).")

            # Y-side partition (top 2 by y vs bottom 2)
            sorted_pids = sorted(y_means.items(), key=lambda x: x[1])
            far_team = [t for t, _ in sorted_pids[:2]]
            near_team = [t for t, _ in sorted_pids[2:]]
            print(f"  Y-sort partition: far={far_team}, near={near_team}")


if __name__ == "__main__":
    main()
