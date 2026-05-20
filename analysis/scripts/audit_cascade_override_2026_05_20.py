#!/usr/bin/env python3
"""Driver: re-run redetect_all_actions on the 51 trusted-32 rallies that
contain at least one rank_1 flip-target contact, with CASCADE_TRACE_OUT
set so that per-rally cascade traces are written to disk.

Reads `reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv`, filters
to `gt_rank == 1` rows (the 28 flip-targets where the v2 scorer's top-1
already equals GT). Computes the set of distinct rally_ids from those
rows, then for each rally:

  1. Sets CASCADE_TRACE_OUT to reports/cascade_override_audit_2026_05_20/traces/
  2. Looks up video name for the rally
  3. Invokes redetect_all_actions.py via subprocess with --video <UUID> --apply

Per-rally trace JSON ends up in CASCADE_TRACE_OUT named `{rally_id}.trace.json`.
Idempotent: re-running clears prior traces.

The redetect script takes a video UUID (not name); this driver resolves
names → UUIDs via the videos table.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import psycopg

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

IN_CSV = Path("reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv")
OUT_DIR = Path("reports/cascade_override_audit_2026_05_20")
TRACE_DIR = OUT_DIR / "traces"


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run probe_scorer_rank2_ceiling first.",
              file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    # Clear prior traces (idempotency).
    for p in TRACE_DIR.glob("*.trace.json"):
        p.unlink()

    rank1_rows = [
        r for r in csv.DictReader(open(IN_CSV))
        if r["gt_rank"] == "1"
    ]
    rallies_to_videos: dict[str, set[str]] = defaultdict(set)
    for r in rank1_rows:
        rallies_to_videos[r["video"]].add(r["rally_id"])
    videos = sorted(rallies_to_videos)
    n_rallies = sum(len(s) for s in rallies_to_videos.values())
    print(f"flip-targets: {len(rank1_rows)}; rallies: {n_rallies}; "
          f"videos: {len(videos)}", flush=True)

    # Resolve video names -> UUIDs (redetect_all_actions.py --video takes UUID)
    name_to_uuid: dict[str, str] = {}
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            "SELECT name, id FROM videos WHERE name = ANY(%s)",
            [list(videos)],
        )
        for name, uid in cur.fetchall():
            name_to_uuid[str(name)] = str(uid)
    missing_videos = [v for v in videos if v not in name_to_uuid]
    if missing_videos:
        print(f"ERROR: video names not resolvable to UUIDs: {missing_videos}",
              file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["CASCADE_TRACE_OUT"] = str(TRACE_DIR.resolve())
    env["USE_DYNAMIC_ATTRIBUTION_SCORER"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    for i, vname in enumerate(videos, start=1):
        vuid = name_to_uuid[vname]
        print(f"[{i}/{len(videos)}] redetect video={vname} ({vuid}) "
              f"({len(rallies_to_videos[vname])} affected rallies)",
              flush=True)
        rc = subprocess.call(
            ["uv", "run", "python", "-u", "scripts/redetect_all_actions.py",
             "--video", vuid, "--apply"],
            env=env,
        )
        if rc != 0:
            print(f"  WARNING: redetect failed for {vname} (rc={rc})",
                  flush=True)

    n_traces = len(list(TRACE_DIR.glob("*.trace.json")))
    print(f"\nWrote {n_traces} trace files -> {TRACE_DIR}", flush=True)
    # Sanity: ensure every flip-target rally has a trace
    target_ids: set[str] = set()
    for s in rallies_to_videos.values():
        target_ids |= s
    have_ids = {p.stem.replace(".trace", "") for p in TRACE_DIR.glob("*.trace.json")}
    missing = target_ids - have_ids
    if missing:
        print(f"WARNING: {len(missing)} target rallies missing traces:",
              flush=True)
        for rid in sorted(missing):
            print(f"  {rid}", flush=True)
        return 2
    print(f"All {len(target_ids)} target rallies have traces.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
