"""Inspect dd042609 r10's team-partition signals from probe sidecar.

Reads track_court_sides from the probe's track_stats_input snapshot
to determine whether the team-pair guard saw a clean 2v2 partition
(and if so, what it looked like) vs falling back to unconstrained
Hungarian.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    probe_dir = Path(__file__).resolve().parent.parent / "reports" / "profile_drift_probe"
    sidecar = sorted(probe_dir.glob("dd042609_*.json"))[-1]
    print(f"Sidecar: {sidecar}")
    data = json.loads(sidecar.read_text())

    rally_ids = data["rally_ids"]
    snapshot = data.get("track_stats_input", [])

    # r10 = rally idx 9
    r10_idx = None
    for i, rid in enumerate(rally_ids):
        if rid.startswith("f811cc63"):
            r10_idx = i
            break

    print(f"r10_idx: {r10_idx}")
    print(f"track_stats_input has {len(snapshot)} entries")

    r10_snap = None
    for entry in snapshot:
        if entry.get("rally_idx") == r10_idx:
            r10_snap = entry
            break

    if r10_snap is None:
        print("No track_stats snapshot for r10 — sidecar may not have captured input snapshots.")
        # Show what we DID capture per rally
        for entry in snapshot[:3]:
            print(f"  ridx {entry.get('rally_idx')}: keys={list(entry.keys())}")
        return

    print()
    print(f"r10 top_tracks: {r10_snap.get('top_tracks')}")
    print(f"r10 track_court_sides: {r10_snap.get('track_court_sides')}")
    if "track_stats" in r10_snap:
        print(f"r10 track_stats keys: {list(r10_snap.get('track_stats', {}).keys())}")


if __name__ == "__main__":
    main()
