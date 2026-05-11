"""Prep visual-check list for off-screen-server diagnostic.

Reads reports/offscreen_server_diagnostic.json and prints, for each
candidate cluster, the (video, rally, abs-time of GT serve, abs-time of
MS-TCN++ peak, abs-time of first detected contact) — so the user can
spot-check the rally in the player and verify whether the serve really
was off-screen.

Also includes the "late_real_serve_no_off_screen" cluster (the cases
where the per-frame gate flagged a player-close-to-ball signal early on,
but the GT-vs-pred gap is still wide) so we can decide whether the
player-empty gate is too restrictive.
"""

from __future__ import annotations

import json
from pathlib import Path

from rallycut.evaluation.tracking.db import get_connection

DIAG_PATH = Path("reports/offscreen_server_diagnostic.json")


def main() -> None:
    with open(DIAG_PATH) as f:
        diag = json.load(f)
    rows = diag["rows"]
    rally_ids = [r["rally_id"] for r in rows]
    meta: dict[str, tuple[int, float]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id, r.start_ms, pt.fps
                   FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.id = ANY(%s)""",
                [rally_ids],
            )
            for rid, ms, fps in cur.fetchall():
                meta[rid] = (int(ms or 0), float(fps or 30.0))

    clusters_of_interest = [
        "off_screen_candidate",
        "late_real_serve_no_off_screen",
    ]
    for cluster in clusters_of_interest:
        cases = [r for r in rows if r["cluster"] == cluster]
        if not cases:
            continue
        print(f"\n## Cluster: {cluster}  ({len(cases)} rallies)\n")
        print(f"{'video':<8} {'rally':<10} "
              f"{'gt_t':>7} {'peak_t':>7} {'pred_t':>7} "
              f"{'gt_f':>5} {'peak_f':>6} {'pred_f':>6} "
              f"{'peak_p':>6} {'tracks':>6} {'pdist':>6}")
        print("-" * 96)
        for r in cases:
            ms, fps = meta.get(r["rally_id"], (0, 30.0))
            base = ms / 1000.0
            gt_t = base + r["gt_first_f"] / fps
            peak_t = base + r["peak_f"] / fps if r["peak_f"] >= 0 else 0.0
            pred_t = base + r["first_pred_f"] / fps if r["first_pred_f"] >= 0 else 0.0
            tracks = r["tracks_n"]
            pdist_disp = "inf" if r["pdist"] == float("inf") else f"{r['pdist']:.3f}"
            print(
                f"{r['video'][:8]:<8} {r['rally_id'][:8]:<10} "
                f"{gt_t:>7.2f} {peak_t:>7.2f} {pred_t:>7.2f} "
                f"{r['gt_first_f']:>5} {r['peak_f']:>6} {r['first_pred_f']:>6} "
                f"{r['peak_p']:>6.2f} {tracks:>6} {pdist_disp:>6}"
            )


if __name__ == "__main__":
    main()
