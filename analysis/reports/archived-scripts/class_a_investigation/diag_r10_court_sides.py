"""Inspect dd042609 r10's track_court_sides + sides_by_bbox to verify
the team-partition signal.

Phase 2 diagnostic showed r10's matcher places T3 on near team (PID1)
but GT puts T3 on far team (PID3). If `track_court_sides` and
`sides_by_bbox` both say T3 is near-side, the team-pair guard will
lock that wrong partition into the cost-matrix-Hungarian step.

Reads pre_remap_state from DB so we see the same data the matcher saw.
"""

from __future__ import annotations

from collections import Counter

from rallycut.evaluation.tracking.db import get_connection


VID = "dd042609-e22e-4f60-83ed-038897c88c32"
RALLY = "f811cc63-cb68-49a2-93ef-88b6b79b682c"


def main() -> None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pt.pre_remap_state_json, pt.raw_positions_json, "
            "v.match_analysis_json "
            "FROM player_tracks pt "
            "JOIN rallies r ON pt.rally_id = r.id "
            "JOIN videos v ON r.video_id = v.id "
            "WHERE pt.rally_id = %s AND v.id = %s",
            [RALLY, VID],
        )
        row = cur.fetchone()
        if not row:
            raise SystemExit(f"no player_track for rally {RALLY}")
        pre = row[0] or {}
        raw = row[1] or {}
        ma = row[2] or {}

    # Find r10's match_analysis entry
    rally_entry = None
    for e in ma.get("rallies", []) or []:
        if e.get("rallyId") == RALLY:
            rally_entry = e
            break

    print(f"=== dd042609 r10 / {RALLY[:8]} ===")
    if rally_entry:
        print(f"top tracks (from match_analysis): {rally_entry.get('topTracks')}")
        print(f"trackToPlayer: {rally_entry.get('trackToPlayer')}")
        print(f"appliedFullMapping: {rally_entry.get('appliedFullMapping')}")
        anchor = rally_entry.get("assignmentAnchor") or {}
        print(f"assignmentAnchor: {anchor}")
        print(f"trackCourtSides: {rally_entry.get('trackCourtSides')}")
        print(f"sidesByBbox: {rally_entry.get('sidesByBbox')}")
        print(f"sidesFromCalibration: {rally_entry.get('sidesFromCalibration')}")
        print(f"teamAssignments: {rally_entry.get('teamAssignments')}")
    else:
        print("no match_analysis entry for r10")

    # Per-frame: where do tracks 1, 2, 3, 16 spend most of their time
    # (Y-position relative to court center)?
    pos = pre.get("positions", []) or []
    if not pos:
        pos = raw.get("positions", []) or []
    by_track: dict[int, list[tuple[int, float, float]]] = {}
    for p in pos:
        tid = int(p["trackId"])
        if tid <= 0:
            continue
        by_track.setdefault(tid, []).append(
            (int(p["frameNumber"]), float(p["x"]), float(p["y"]))
        )

    print()
    print(f"Per-track Y stats (n, y_mean, y_min, y_max, x_mean):")
    for tid in sorted(by_track):
        pts = by_track[tid]
        if not pts:
            continue
        ys = [p[2] for p in pts]
        xs = [p[1] for p in pts]
        n = len(pts)
        y_mean = sum(ys) / n
        x_mean = sum(xs) / n
        y_min = min(ys)
        y_max = max(ys)
        side_label = "near (y > 0.5)" if y_mean > 0.5 else "far (y <= 0.5)"
        marker = ""
        if tid in [1, 2, 3, 16]:
            marker = "  <-- PRIMARY"
        print(f"  T{tid:3d}: n={n:4d}, y_mean={y_mean:.3f}, "
              f"y_range=[{y_min:.3f}, {y_max:.3f}], x_mean={x_mean:.3f} "
              f"({side_label}){marker}")


if __name__ == "__main__":
    main()
