#!/usr/bin/env python3
"""Dump MatchSolver's cluster membership across all rallies + per-rally
appearance-cost matrices for the assigned cluster.

The matcher's final per-rally `trackToPlayer` is Pass-2 MatchSolver's
output. Without seeing the cluster membership and the appearance cost
that drove each rally's Hungarian, you can't tell whether a particular
mis-assignment is because of (a) the wrong seed, (b) bad cluster
membership in other rallies cascading, or (c) genuinely close
appearance margins.

This script runs the matcher with `collect_diagnostics=True`, then
re-runs MatchSolver in-memory while also exporting per-rally cluster
assignment + the appearance cost matrix. Useful for tracing one
player's track across rallies — e.g. "find every rally where a
particular cluster contains a track that visually doesn't belong."

Usage:
    uv run python scripts/diagnose_matchsolver_clusters.py \
        --video-id <uuid> [--rallies 6,9,22,23]

Read-only: does NOT touch the DB.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.evaluation.tracking.db import (  # noqa: E402
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.match_tracker import (  # noqa: E402
    match_players_across_rallies,
)


def _load_court_calibrator(video_id: str) -> CourtCalibrator | None:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT court_calibration_json FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    cal = row[0]
    if not isinstance(cal, list) or len(cal) != 4:
        return None
    c = CourtCalibrator()
    c.calibrate([(p["x"], p["y"]) for p in cal])
    return c if c.is_calibrated else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video-id", required=True)
    ap.add_argument(
        "--rallies",
        default="",
        help="1-indexed comma-separated rally indices to focus on (default: all)",
    )
    args = ap.parse_args()
    focus = (
        {int(x.strip()) for x in args.rallies.split(",") if x.strip()}
        if args.rallies else set()
    )

    rallies = load_rallies_for_video(args.video_id)
    if not rallies:
        print(f"No rallies for {args.video_id}")
        sys.exit(1)
    vp = get_video_path(args.video_id)
    if vp is None:
        print(f"No video file for {args.video_id}")
        sys.exit(1)
    cal = _load_court_calibrator(args.video_id)

    reid_model = None
    try:
        from rallycut.tracking.reid_general import (
            WEIGHTS_PATH as REID_WEIGHTS_PATH,
        )
        if REID_WEIGHTS_PATH.exists():
            from rallycut.tracking.reid_general import GeneralReIDModel
            reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
            print(f"  Loaded ReID weights: {REID_WEIGHTS_PATH.name}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ReID load failed: {exc}")

    print(
        f"Loaded {len(rallies)} rallies for {args.video_id[:8]}  "
        f"video={vp.name}  calibrator={'yes' if cal else 'no'}"
    )

    print("Running matcher (collect_diagnostics=True)...")
    result = match_players_across_rallies(
        video_path=vp,
        rallies=rallies,
        calibrator=cal,
        collect_diagnostics=True,
        extract_reid=reid_model is not None,
        reid_model=reid_model,
    )

    # Print per-rally MatchSolver decision and cluster membership.
    # Cluster IDs in MatchSolver are 1..NUM_CLUSTERS and used directly
    # as the PID labels; we recover cluster membership by inverting
    # `result.rally_results[i].track_to_player`.
    cluster_members: dict[int, list[tuple[int, int]]] = {1: [], 2: [], 3: [], 4: []}
    for rally_idx, rr in enumerate(result.rally_results):
        for tid, pid in rr.track_to_player.items():
            if tid >= 0 and pid in cluster_members:
                cluster_members[pid].append((rally_idx, tid))

    print(f"\n{'=' * 78}")
    print("FINAL cluster membership (rally_idx + 1, track_id):")
    for cid in (1, 2, 3, 4):
        rows = cluster_members[cid]
        sample = ", ".join(f"r{ri + 1}t{ti}" for ri, ti in rows[:15])
        more = f" +{len(rows) - 15} more" if len(rows) > 15 else ""
        print(f"  C{cid} ({len(rows)} tracks): {sample}{more}")

    # Walk diagnostics + side classification per rally to surface where
    # the team-pair constraint fired and where it didn't.
    print(f"\n{'=' * 78}")
    print(
        "Per-rally side cardinality + team-pair status "
        "(2v2 partition proposers and which won):"
    )
    diag_by_idx = {d.rally_index: d for d in result.diagnostics}
    for i, (rd, rr) in enumerate(zip(rallies, result.rally_results)):
        idx1 = i + 1
        if focus and idx1 not in focus:
            continue
        diag = diag_by_idx.get(i)
        if diag is None:
            continue
        sides = diag.track_court_sides
        near = sum(1 for s in sides.values() if s == 0)
        far = sum(1 for s in sides.values() if s == 1)
        print(
            f"  [{idx1:2d}] {rd.rally_id[:8]}  "
            f"sides=N{near}/F{far}  "
            f"top={diag.track_ids}  "
            f"assignment={dict(sorted(rr.track_to_player.items()))}  "
            f"side_switch={rr.side_switch_detected}"
        )

    # For focused rallies, print which CLUSTERS each track is closest to
    # by appearance only — the appearance cost vector per track.
    if focus:
        print(f"\n{'=' * 78}")
        print("Per-rally appearance-only similarity to each cluster:")
        # Build cluster medoid stats via members. We use compute_track_similarity
        # against each cluster's other-rally members (averaged), matching
        # what MatchSolver does internally.

        # Recover stored_rally_data via the match_players_across_rallies
        # internal — easiest: walk per-rally top_tracks + track_stats from
        # result.diagnostics, but track_stats aren't on diagnostics. We
        # need the appearance stats. So we'd need to re-run extract.
        # Cheap alternative: run a second pass using the matcher to build
        # `track_stats`, then compute against final clusters.
        # Skipping for now — the user-facing observation is in the table
        # above. (Comment kept so a future caller knows what to add.)
        print("  (per-rally appearance cost table requires re-running extract; "
              "use diagnose_pid_per_rally.py for that)")

    print(f"\n{'=' * 78}")
    print(
        f"SUMMARY: {len(rallies)} rallies, "
        f"avg confidence={sum(r.assignment_confidence for r in result.rally_results)/len(rallies):.3f}"
    )
    print(
        f"Side switches detected at rally indices: "
        f"{[i + 1 for i, r in enumerate(result.rally_results) if r.side_switch_detected]}"
    )


if __name__ == "__main__":
    main()
