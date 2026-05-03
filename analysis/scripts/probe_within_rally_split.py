"""Probe within-rally track split: do T_a and T_b cover the same physical
position over time, or do they spatially separate at every frame?

For each rally on a video, finds pairs of primary tracks where:
  - frame ranges overlap substantially
  - per-frame distance between (track_a position, track_b position) is
    SMALL (i.e. they're tracking the same player) OR LARGE (they're
    different players with similar mean positions).

If small per-frame distance over many overlapping frames → within-rally
track split (Pattern A).

If large per-frame distance → just two players who happen to mean to
similar positions; not a split.

Usage:
    uv run python scripts/probe_within_rally_split.py <video_id>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import cast

from rallycut.evaluation.tracking.db import get_connection, load_rallies_for_video


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--rally", default=None,
                   help="Restrict to one rally (8-char prefix or full UUID)")
    args = p.parse_args()

    rallies = load_rallies_for_video(args.video_id)
    chronological = sorted(rallies, key=lambda r: r.start_ms)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pt.rally_id::text, pt.pre_remap_state_json "
            "FROM player_tracks pt JOIN rallies r ON pt.rally_id = r.id "
            "WHERE r.video_id = %s",
            [args.video_id],
        )
        snaps = {
            str(rid): (snap if isinstance(snap, dict) else json.loads(cast(str, snap)))
            for rid, snap in cur.fetchall() if snap is not None
        }

    for idx, rally in enumerate(chronological):
        rid = rally.rally_id
        if args.rally and not rid.startswith(args.rally) and args.rally != rid:
            continue
        snap = snaps.get(rid)
        if snap is None:
            continue
        positions = snap.get("positions") or []
        primary = [int(t) for t in (snap.get("primaryTrackIds") or []) if int(t) > 0]
        if len(primary) < 2:
            continue

        # frame -> {track_id: (x, y)}
        per_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
        for q in positions:
            tid = int(q.get("trackId", -1))
            fn = q.get("frameNumber")
            if tid <= 0 or fn is None or tid not in primary:
                continue
            per_frame[int(fn)][tid] = (float(q["x"]), float(q["y"]))

        print(f"\n## Rally {idx + 1}: {rid[:8]} primary={primary}")
        print(
            "| Track A | Track B | Overlapping frames | Mean dist | "
            "p10 dist | p90 dist | Same-player? |"
        )
        print("|---|---|---|---|---|---|---|")

        for i, ta in enumerate(primary):
            for tb in primary[i + 1:]:
                dists = []
                for fn, posdict in per_frame.items():
                    if ta in posdict and tb in posdict:
                        ax, ay = posdict[ta]
                        bx, by = posdict[tb]
                        d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                        dists.append(d)
                if not dists:
                    continue
                dists.sort()
                mean_d = sum(dists) / len(dists)
                p10 = dists[int(len(dists) * 0.10)]
                p90 = dists[int(len(dists) * 0.90)]
                # Same-player heuristic: more than half of overlapping frames
                # have distance < 0.05 (50 px on a 1000px-wide image-normalized
                # coordinate space — roughly within bbox-half-width).
                close_frac = sum(1 for d in dists if d < 0.05) / len(dists)
                verdict = "?"
                if close_frac > 0.5:
                    verdict = "**LIKELY SAME PLAYER**"
                elif p10 > 0.10:
                    verdict = "different (always far)"
                else:
                    verdict = "ambiguous"
                print(
                    f"| T{ta} | T{tb} | {len(dists)} | {mean_d:.3f} | "
                    f"{p10:.3f} | {p90:.3f} | {verdict} |"
                )


if __name__ == "__main__":
    main()
