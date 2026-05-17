#!/usr/bin/env python3
"""Diagnose chimera-track prevalence on keke (and trusted-29 fleet).

For each rally, compute pairwise track overlap: % of frames where two
primary tracks' bbox centers are within EPS of each other. High overlap
== same physical player tracked under two IDs (chimera).

Output: per-rally chimera candidates, prevalence summary.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)

EPS = 0.04         # bbox-center proximity threshold (normalized coords)
MIN_OVERLAP_PCT = 0.50  # ≥50% of co-present frames must be within EPS


def main(focus_video: str | None = None) -> int:
    codenames = (focus_video,) if focus_video else TRUSTED_29
    print(f"Diagnosing chimera prevalence on {len(codenames)} video(s) "
          f"(EPS={EPS}, MIN_OVERLAP={MIN_OVERLAP_PCT*100:.0f}%)…",
          flush=True)
    total_rallies = 0
    chimera_rallies = 0
    fleet_chimeras: list[tuple[str, int, int, int, float]] = []  # (video, order, t1, t2, overlap_pct)
    with psycopg.connect(DB) as conn:
        cur = conn.execute("""
            SELECT v.name, r."order", r.id, pt.primary_track_ids, pt.positions_json
            FROM rallies r JOIN videos v ON r.video_id=v.id
            JOIN player_tracks pt ON pt.rally_id=r.id
            WHERE v.name = ANY(%s) AND r.status='CONFIRMED'
              AND pt.positions_json IS NOT NULL
            ORDER BY v.name, r."order"
        """, [list(codenames)])
        for video, order, rid, primary, pj in cur.fetchall():
            total_rallies += 1
            if not isinstance(primary, list):
                continue
            if isinstance(pj, str):
                pj = json.loads(pj)
            positions = pj if isinstance(pj, list) else []
            primary_tids = [int(t) for t in primary]
            # frame -> tid -> (cx, cy)
            by_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
            for p in positions:
                tid = int(p.get("trackId", -1))
                if tid not in primary_tids:
                    continue
                f = int(p.get("frameNumber", -1))
                cx = float(p.get("x", 0)) + float(p.get("width", 0)) / 2
                cy = float(p.get("y", 0)) + float(p.get("height", 0)) / 2
                by_frame[f][tid] = (cx, cy)
            # Pairwise overlap analysis
            pair_co_present: dict[tuple[int, int], int] = defaultdict(int)
            pair_close: dict[tuple[int, int], int] = defaultdict(int)
            for f, tids in by_frame.items():
                items = sorted(tids.items())
                for i in range(len(items)):
                    for j in range(i + 1, len(items)):
                        t1, c1 = items[i]
                        t2, c2 = items[j]
                        key = (t1, t2)
                        pair_co_present[key] += 1
                        if math.hypot(c1[0] - c2[0], c1[1] - c2[1]) < EPS:
                            pair_close[key] += 1
            this_chimeras: list[tuple[int, int, float, int]] = []
            for key, n_co in pair_co_present.items():
                if n_co < 20:
                    continue
                n_close = pair_close.get(key, 0)
                pct = n_close / n_co
                if pct >= MIN_OVERLAP_PCT:
                    t1, t2 = key
                    this_chimeras.append((t1, t2, pct, n_co))
                    fleet_chimeras.append((video, order, t1, t2, pct))
            if this_chimeras:
                chimera_rallies += 1
                print(f"  {video:6s} r{order:2d} ({rid[:8]}): "
                      f"primary={primary_tids}  chimera candidates:",
                      flush=True)
                for t1, t2, pct, n_co in this_chimeras:
                    print(f"    track {t1} <-> {t2}: {pct*100:.0f}% of {n_co} co-present frames within {EPS} normalized",
                          flush=True)

    print(flush=True)
    print(f"Summary: {chimera_rallies}/{total_rallies} rallies have chimera-candidate pairs "
          f"({chimera_rallies/max(1,total_rallies)*100:.1f}%)", flush=True)
    if fleet_chimeras and not focus_video:
        per_video: dict[str, int] = defaultdict(int)
        for v, _o, _t1, _t2, _p in fleet_chimeras:
            per_video[v] += 1
        print()
        print("Per-video chimera rally counts (>0):", flush=True)
        for v in sorted(per_video, key=lambda x: -per_video[x]):
            print(f"  {v:6s}: {per_video[v]} chimera rallies", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
