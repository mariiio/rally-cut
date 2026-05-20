"""Probe X-J: per-video comparison of estimate_net_position vs user GT.

User just labeled the visible net top in the editor for every video
they cared about (stored in `videos.court_calibration_net_top_y`).
This probe reads that GT and compares it against the per-rally `netY`
values produced by `contact_detector.estimate_net_position` (stored
in `player_tracks.contacts_json.netY`).

For each labeled video:
  - GT net top y (one value)
  - n rallies with a stored netY
  - median + mean of per-rally netY
  - std-dev across rallies (how stable is the estimate?)
  - delta = (estimate − GT). Negative = estimate above GT, positive = below.
  - per-rally |delta_i| max — worst single-rally divergence

Sorted by |median − GT| descending so the worst-systematic cases lead
the output. Helps decide whether the failure mode is:
  - systematic bias (algorithm)
  - high per-video variance with bias near 0 (noisy estimator, would
    aggregate well)
  - one-off outliers (specific bug or pathological ball trajectories)
"""
from __future__ import annotations

import json
import statistics
import sys

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"


def main() -> int:
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.id::text, v.name, COALESCE(v.fps, 30) AS fps,
                   v.court_calibration_net_top_y AS gt
            FROM videos v
            WHERE v.court_calibration_net_top_y IS NOT NULL
            ORDER BY v.name
            """,
        )
        videos = cur.fetchall()

        rows = []
        for vid, vname, fps, gt in videos:
            cur = conn.execute(
                """
                SELECT pt.contacts_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s
                  AND pt.contacts_json IS NOT NULL
                """,
                (vid,),
            )
            net_ys = []
            for (cj,) in cur:
                cj = cj if isinstance(cj, dict) else json.loads(cj or '{}')
                ny = cj.get("netY")
                if ny is not None and ny > 0:
                    net_ys.append(float(ny))
            if not net_ys:
                rows.append({
                    "video": vname, "fps": fps, "gt": gt,
                    "n": 0, "median": None, "mean": None, "std": None,
                    "delta_median": None, "max_abs_delta": None,
                })
                continue
            median = statistics.median(net_ys)
            mean = statistics.mean(net_ys)
            std = statistics.pstdev(net_ys) if len(net_ys) > 1 else 0.0
            delta_median = median - gt
            max_abs_delta = max(abs(y - gt) for y in net_ys)
            rows.append({
                "video": vname, "fps": fps, "gt": gt,
                "n": len(net_ys), "median": median, "mean": mean, "std": std,
                "delta_median": delta_median, "max_abs_delta": max_abs_delta,
            })

    # Sort by |delta_median| descending (worst systematic bias first); None
    # rows (no per-rally data) at the end.
    rows.sort(key=lambda r: (
        0 if r["delta_median"] is None else 1,
        -abs(r["delta_median"] or 0),
    ))

    print(f"Loaded GT for {len(videos)} videos.\n", flush=True)
    print(
        f"{'video':<14} {'fps':>5} {'n':>4} {'gt':>6} {'median':>7} "
        f"{'mean':>7} {'std':>6} {'Δmed':>7} {'maxΔ':>6}",
        flush=True,
    )
    for r in rows:
        if r["median"] is None:
            print(f"{r['video']:<14} {r['fps']:>5.1f} {0:>4} {r['gt']:>6.3f}  (no rallies)", flush=True)
            continue
        print(
            f"{r['video']:<14} {r['fps']:>5.1f} {r['n']:>4} {r['gt']:>6.3f} "
            f"{r['median']:>7.3f} {r['mean']:>7.3f} {r['std']:>6.3f} "
            f"{r['delta_median']:+7.3f} {r['max_abs_delta']:>6.3f}",
            flush=True,
        )

    # Aggregate stats across labeled fleet
    valid = [r for r in rows if r["delta_median"] is not None]
    if not valid:
        return 0
    deltas = [r["delta_median"] for r in valid]
    abs_deltas = [abs(d) for d in deltas]
    stds = [r["std"] for r in valid]
    print()
    print("=== Aggregate across labeled fleet ===", flush=True)
    print(f"  n videos with rallies:  {len(valid)} / {len(videos)}", flush=True)
    print(f"  mean signed delta:     {statistics.mean(deltas):+.3f}", flush=True)
    print(f"  median signed delta:   {statistics.median(deltas):+.3f}", flush=True)
    print(f"  mean |delta|:          {statistics.mean(abs_deltas):.3f}", flush=True)
    print(f"  median |delta|:        {statistics.median(abs_deltas):.3f}", flush=True)
    print(f"  worst |delta|:         {max(abs_deltas):.3f}", flush=True)
    print(f"  videos with |delta| > 0.05:  {sum(1 for d in abs_deltas if d > 0.05)}", flush=True)
    print(f"  videos with |delta| > 0.10:  {sum(1 for d in abs_deltas if d > 0.10)}", flush=True)
    print(f"  videos with |delta| > 0.15:  {sum(1 for d in abs_deltas if d > 0.15)}", flush=True)
    print(f"  median per-video stddev:  {statistics.median(stds):.3f} "
          f"(per-rally variance within same video)", flush=True)

    # FPS split
    print()
    print("=== By FPS ===", flush=True)
    for fps_bucket, label in [(45, "≤45 fps"), (45.001, ">45 fps")]:
        sub = [r for r in valid if (
            r["fps"] <= 45 if label == "≤45 fps" else r["fps"] > 45
        )]
        if not sub:
            continue
        abs_d = [abs(r["delta_median"]) for r in sub]
        print(
            f"  {label}: n={len(sub)}, "
            f"mean |delta|={statistics.mean(abs_d):.3f}, "
            f"median |delta|={statistics.median(abs_d):.3f}, "
            f"worst={max(abs_d):.3f}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
