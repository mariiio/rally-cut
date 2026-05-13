"""One-shot fleet-wide PID-invariant audit.

Runs `pid_invariants.run_all` against every video that has at least one
tracked rally, tabulates per-video and per-invariant counts.

Usage: cd analysis && uv run python scripts/fleet_pid_audit.py
"""

from __future__ import annotations

import sys
from collections import Counter

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.pid_invariants import run_all


def list_audit_videos() -> list[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT v.id
                FROM videos v
                JOIN rallies r ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE pt.actions_json IS NOT NULL
                ORDER BY v.id
                """
            )
            return [row[0] for row in cur.fetchall()]


def main() -> int:
    video_ids = list_audit_videos()
    n = len(video_ids)
    print(f"Auditing {n} videos with tracking data...\n")

    rows: list[tuple[str, int, Counter[str]]] = []
    fleet_counter: Counter[str] = Counter()
    n_clean = 0

    for i, vid in enumerate(video_ids, start=1):
        try:
            violations, _stale = run_all(video_id=vid)
        except Exception as exc:
            print(f"[{i}/{n}] {vid[:8]}: ERROR {exc}", flush=True)
            continue

        per_invariant: Counter[str] = Counter(v.invariant for v in violations)
        total = len(violations)
        if total == 0:
            n_clean += 1
        rows.append((vid, total, per_invariant))
        fleet_counter.update(per_invariant)

        # Progress line: vid_short total per_invariant
        bits = " ".join(f"{k}={per_invariant[k]}" for k in sorted(per_invariant))
        print(f"[{i}/{n}] {vid[:8]}: {total:4d} {bits}", flush=True)

    # Summary
    print("\n" + "=" * 70)
    print(f"Fleet summary: {n} videos audited, {n_clean} clean ({n_clean*100//n}%)")
    print(f"Total violations across fleet: {sum(fleet_counter.values())}")
    print("Per-invariant fleet counts:")
    for inv in sorted(fleet_counter):
        print(f"  {inv}: {fleet_counter[inv]}")

    # Top dirty videos
    print("\nTop 10 dirty videos (by total violations):")
    rows_sorted = sorted(rows, key=lambda r: -r[1])
    for vid, total, per in rows_sorted[:10]:
        bits = ", ".join(f"{k}={per[k]}" for k in sorted(per))
        print(f"  {vid[:8]}: {total} ({bits})")

    # Per-invariant distribution: how many videos have at least one of each
    print("\nVideos affected per invariant:")
    for inv in sorted(fleet_counter):
        n_videos_affected = sum(1 for _, _, p in rows if p.get(inv, 0) > 0)
        print(f"  {inv}: {n_videos_affected}/{n} videos")

    return 0


if __name__ == "__main__":
    sys.exit(main())
