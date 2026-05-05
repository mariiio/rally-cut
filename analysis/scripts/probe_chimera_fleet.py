"""Fleet-wide chimera survey: are out-of-range rallies chimera-heavy?

Reuses the per-rally chimera detection from probe_track_continuity.py.
For each rally in two cohorts (AFFECTED = primary_track_ids has any id > 4,
CONTROL = sample of in-range rallies), reports:
  - count of filtered tracks total
  - count of chimera tracks (multi-raw-evidence)
  - chimera rate per rally

Read-only DB. Per-rally progress logging.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any

from rallycut.evaluation.db import get_connection

MATCH_RADIUS = 0.05


def _center(p: dict[str, Any]) -> tuple[float, float]:
    return (p["x"] + p["width"] / 2.0, p["y"] + p["height"] / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def chimera_count(
    positions: list[dict[str, Any]],
    raw_positions: list[dict[str, Any]],
) -> tuple[int, int, dict[int, list[int]]]:
    """Returns (n_filtered_tracks, n_chimera_tracks, per_track_evidence_set)."""
    by_filtered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in positions:
        by_filtered[p["trackId"]].append(p)

    raw_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in raw_positions:
        raw_by_frame[p["frameNumber"]].append(p)

    per_track_evidence: dict[int, list[int]] = {}
    chimera_n = 0
    for ftid, frames in by_filtered.items():
        evidence_set: set[int] = set()
        for fp in frames:
            fnum = fp["frameNumber"]
            fc = _center(fp)
            best_tid: int | None = None
            best_d = float("inf")
            for rp in raw_by_frame.get(fnum, []):
                d = _dist(fc, _center(rp))
                if d < best_d:
                    best_d = d
                    best_tid = rp["trackId"]
            if best_tid is not None and best_d <= MATCH_RADIUS:
                evidence_set.add(best_tid)
        per_track_evidence[ftid] = sorted(evidence_set)
        if len(evidence_set) > 1:
            chimera_n += 1

    return len(by_filtered), chimera_n, per_track_evidence


def fetch_cohort(
    conn: Any,
    cohort: str,
    sample_size: int,
) -> list[tuple[str, list[int], list[dict[str, Any]], list[dict[str, Any]]]]:
    """cohort: 'AFFECTED' or 'CONTROL'. Returns rows (rally_id, primary_ids, positions, raw_positions)."""
    if cohort == "AFFECTED":
        sql = """
            SELECT pt.rally_id, pt.primary_track_ids, pt.positions_json, pt.raw_positions_json
            FROM player_tracks pt
            WHERE pt.primary_track_ids IS NOT NULL
              AND pt.positions_json IS NOT NULL
              AND pt.raw_positions_json IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(pt.primary_track_ids) AS x(val)
                WHERE x.val::int > 4
              )
            ORDER BY pt.rally_id
        """
        params: tuple[object, ...] = ()
    else:
        sql = """
            SELECT pt.rally_id, pt.primary_track_ids, pt.positions_json, pt.raw_positions_json
            FROM player_tracks pt
            WHERE pt.primary_track_ids IS NOT NULL
              AND pt.positions_json IS NOT NULL
              AND pt.raw_positions_json IS NOT NULL
              AND NOT EXISTS (
                SELECT 1 FROM jsonb_array_elements_text(pt.primary_track_ids) AS x(val)
                WHERE x.val::int > 4
              )
            ORDER BY pt.rally_id
            LIMIT %s
        """
        params = (sample_size,)

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fleet-wide chimera survey")
    parser.add_argument("--control-sample", type=int, default=30,
                        help="Number of in-range control rallies to sample")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Write per-rally results to this JSON path")
    args = parser.parse_args()

    print("Connecting to DB...")
    with get_connection() as conn:
        print("Fetching AFFECTED cohort (all 22)...")
        affected = fetch_cohort(conn, "AFFECTED", 0)
        print(f"  Got {len(affected)} affected rallies")
        print(f"Fetching CONTROL cohort (sample of {args.control_sample})...")
        control = fetch_cohort(conn, "CONTROL", args.control_sample)
        print(f"  Got {len(control)} control rallies")

    results = []
    for cohort_name, rows in [("AFFECTED", affected), ("CONTROL", control)]:
        print(f"\n=== Processing {cohort_name} cohort ({len(rows)} rallies) ===")
        for i, row in enumerate(rows, 1):
            rally_id, primary_ids, positions, raw_positions = row
            n_filtered, n_chimera, evidence = chimera_count(positions, raw_positions)
            chimera_rate = n_chimera / n_filtered if n_filtered else 0.0
            has_outlier = any(int(t) > 4 for t in (primary_ids or []))
            chimera_filtered_tracks = [
                ftid for ftid, ev in evidence.items() if len(ev) > 1
            ]
            results.append({
                "cohort": cohort_name,
                "rally_id": rally_id,
                "primary_track_ids": primary_ids,
                "n_filtered": n_filtered,
                "n_chimera": n_chimera,
                "chimera_rate": chimera_rate,
                "chimera_filtered_tracks": sorted(chimera_filtered_tracks),
                "evidence_per_track": {str(k): v for k, v in evidence.items()},
            })
            print(f"  [{i:>3}/{len(rows)}] {rally_id[:8]} "
                  f"primaries={primary_ids} "
                  f"filtered={n_filtered} chimeras={n_chimera} "
                  f"rate={chimera_rate:.0%} "
                  f"{'OUT' if has_outlier else '   '}")

    # Aggregates
    print("\n=== AGGREGATE ===")
    for cohort_name in ["AFFECTED", "CONTROL"]:
        cohort = [r for r in results if r["cohort"] == cohort_name]
        if not cohort:
            continue
        total_filt = sum(r["n_filtered"] for r in cohort)
        total_chim = sum(r["n_chimera"] for r in cohort)
        rallies_with_chimera = sum(1 for r in cohort if r["n_chimera"] > 0)
        print(f"  {cohort_name}: {len(cohort)} rallies")
        print(f"    Total filtered tracks: {total_filt}")
        print(f"    Total chimera tracks: {total_chim} ({total_chim/total_filt:.1%})")
        print(f"    Rallies with ≥1 chimera: {rallies_with_chimera} "
              f"({rallies_with_chimera/len(cohort):.0%})")
        # Distribution
        rate_buckets = defaultdict(int)
        for r in cohort:
            if r["n_chimera"] == 0:
                rate_buckets["0_clean"] += 1
            elif r["n_chimera"] == 1:
                rate_buckets["1_chimera"] += 1
            elif r["n_chimera"] == 2:
                rate_buckets["2_chimeras"] += 1
            else:
                rate_buckets["3+_chimeras"] += 1
        for bucket in ["0_clean", "1_chimera", "2_chimeras", "3+_chimeras"]:
            print(f"    {bucket}: {rate_buckets[bucket]}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nWrote per-rally results → {args.json_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
