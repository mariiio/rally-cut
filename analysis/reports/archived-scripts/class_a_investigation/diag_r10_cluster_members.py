"""Trace dd042609 r10's cluster members and check if upstream rally
assignments are consistent.

For each cluster (1-4), list which (rally_idx, track_id) pairs were
members at the final iteration. The cost matrix at r10 is built from
appearance similarity to these members. If members are predominantly
correct (per cross-rally consensus), the cost matrix is signal — and
r10 is feature-space ceiling. If members are mixed/wrong, the cost
matrix is contaminated and a matcher fix is plausible.

Also reports cross-rally consensus on each track-id-as-physical-player
by checking what PID the matcher gave each "long-track" tid across
rallies. (Within-rally tids reset per rally, so we only correlate by
appearance, not tid — this approximates by checking the modal cluster
per primary tid across the video.)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def main() -> None:
    probe_dir = Path(__file__).resolve().parent.parent / "reports" / "profile_drift_probe"
    sidecar = sorted(probe_dir.glob("dd042609_*.json"))[-1]
    print(f"Sidecar: {sidecar}")
    data = json.loads(sidecar.read_text())

    rally_ids = data["rally_ids"]
    iter_records = data["iter_records"]

    # Find r10 idx
    r10_idx = None
    for i, rid in enumerate(rally_ids):
        if rid.startswith("f811cc63"):
            r10_idx = i
            break
    print(f"r10 rally_idx: {r10_idx}")

    # Get final assignments per rally (last iter for each rally_idx)
    final_by_rally: dict[int, dict[int, int]] = {}
    for rec in iter_records:
        ri = rec["rally_idx"]
        if ri not in final_by_rally or rec["iteration"] > _last_iter_seen.get(ri, -1):
            final_by_rally[ri] = {int(k): v for k, v in rec["assignment"].items()}
            _last_iter_seen[ri] = rec["iteration"]

    # Build cluster members from final assignments (excluding r10 itself).
    # Cluster member = (rally_idx, track_id, cluster_id).
    cluster_members: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}
    for ri, asg in final_by_rally.items():
        if ri == r10_idx:
            continue
        for tid, cid in asg.items():
            cluster_members.setdefault(cid, []).append((ri, tid))

    print()
    print(f"Final cluster members (excluding r10):")
    for cid in sorted(cluster_members):
        mems = cluster_members[cid]
        print(f"  C{cid}: {len(mems)} members, "
              f"first 6: {mems[:6]}")

    # Cross-rally consensus check: for each rally, what's its 'team partition'
    # — which 2 tracks got near-team clusters {1,2} vs which got far {3,4}?
    print()
    print(f"Per-rally team-partition (track -> 'near' if cluster in [1,2] else 'far'):")
    for ri in sorted(final_by_rally.keys()):
        asg = final_by_rally[ri]
        rid_short = rally_ids[ri][:8] if ri < len(rally_ids) else "?"
        team_parts = {tid: ("near" if cid in (1, 2) else "far") for tid, cid in asg.items()}
        marker = "  <-- r10" if ri == r10_idx else ""
        print(f"  rally {ri:2d} ({rid_short}): {team_parts}{marker}")


_last_iter_seen: dict[int, int] = {}


if __name__ == "__main__":
    main()
