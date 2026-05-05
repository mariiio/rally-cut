"""Ablation probe: which player_filter pass is responsible for filtered T7's chimera on r19?

Re-runs the player_tracker on a single rally with cumulative post-processing
passes enabled (K=0..7), then for each K, attaches each filtered track to its
raw-evidence set (closest-bbox match against raw_positions_json) and reports:

  - filtered track ids in the result
  - which of them are chimeras (multi-raw-evidence)
  - the raw-evidence runs for chimeras

By comparing K vs K-1, we can localize which pass introduced the chimera
composition we observed in production for r19's filtered T7.

Per-K result is JSON-serialized to reports/per_pass_chimera/<rally_short>/k<K>.json.

Usage:
    cd analysis
    uv run python scripts/probe_per_pass_chimera.py --rally efdbf6b2-54a8-4444-8135-17d91baf977b
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Suppress noisy library logs
logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s: %(message)s")
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.ERROR)
logging.getLogger("rallycut.tracking.ball_features").setLevel(logging.ERROR)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "per_pass_chimera"

# Pipeline passes in execution order (matches diagnose_per_pass_swaps.py).
# K=0 → all passes skipped; K=7 → all passes enabled (production default).
PASSES: list[tuple[str, str]] = [
    ("ENFORCE_SPATIAL_CONSISTENCY", "0"),
    ("FIX_HEIGHT_SWAPS", "0a"),
    ("SPLIT_TRACKS_BY_COLOR", "0b"),
    ("RELINK_SPATIAL_SPLITS", "0b2"),
    ("RELINK_PRIMARY_FRAGMENTS", "0b3"),
    ("LINK_TRACKLETS_BY_APPEARANCE", "0c"),
    ("STABILIZE_TRACK_IDS", "1"),
]

MATCH_RADIUS = 0.05


def _set_skip_flags(passes_enabled: int) -> None:
    """Set SKIP_* env vars to enable passes 0..K-1, skip K..6."""
    for idx, (suffix, _step) in enumerate(PASSES):
        flag = f"SKIP_{suffix}"
        if idx >= passes_enabled:
            os.environ[flag] = "1"
        else:
            os.environ.pop(flag, None)
    os.environ.pop("SKIP_ALL_MERGE_PASSES", None)


def _center(p: dict[str, Any]) -> tuple[float, float]:
    return (p["x"] + p["width"] / 2.0, p["y"] + p["height"] / 2.0)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _runs_evidence(positions: list[dict[str, Any]], raw_positions: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """For each filtered track id in `positions`, compute evidence runs against raw_positions."""
    by_filtered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in positions:
        by_filtered[p["trackId"]].append(p)
    for tid in by_filtered:
        by_filtered[tid].sort(key=lambda p: p["frameNumber"])

    raw_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for p in raw_positions:
        raw_by_frame[p["frameNumber"]].append(p)

    out: dict[int, dict[str, Any]] = {}
    for tid, frames in by_filtered.items():
        evidence_set: set[int] = set()
        runs: list[dict[str, Any]] = []
        cur_track: int | None = None
        cur_start = -1
        cur_end = -1
        cur_count = 0
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
            ev = best_tid if (best_tid is not None and best_d <= MATCH_RADIUS) else None
            if ev is not None:
                evidence_set.add(ev)
            if cur_track is None:
                cur_track = ev
                cur_start = fnum
                cur_end = fnum
                cur_count = 1
            elif ev == cur_track:
                cur_end = fnum
                cur_count += 1
            else:
                runs.append({"raw_track": cur_track, "start": cur_start, "end": cur_end, "count": cur_count})
                cur_track = ev
                cur_start = fnum
                cur_end = fnum
                cur_count = 1
        if cur_count > 0:
            runs.append({"raw_track": cur_track, "start": cur_start, "end": cur_end, "count": cur_count})

        out[tid] = {
            "frame_count": len(frames),
            "frame_range": [frames[0]["frameNumber"], frames[-1]["frameNumber"]],
            "evidence_raw_tracks": sorted(evidence_set),
            "is_chimera": len(evidence_set) > 1,
            "runs": runs,
        }
    return out


def _retrack(rally_id: str) -> dict[str, Any]:
    """Retrack a single rally with current SKIP env state, return positions + primary_ids."""
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import get_rally_info, get_video_path
    from rallycut.tracking.ball_tracker import create_ball_tracker
    from rallycut.tracking.player_tracker import PlayerTracker

    info = get_rally_info(rally_id)
    if info is None:
        raise RuntimeError(f"Rally {rally_id} not found")
    video_path = get_video_path(info.video_id)
    if video_path is None:
        raise RuntimeError(f"Video {info.video_id} could not be resolved")

    calibrator = None
    if info.calibration_json and len(info.calibration_json) == 4:
        calibrator = CourtCalibrator()
        image_corners = [(c["x"], c["y"]) for c in info.calibration_json]
        calibrator.calibrate(image_corners)

    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path, start_ms=info.start_ms, end_ms=info.end_ms,
    )
    ball_positions = ball_result.positions

    tracker = PlayerTracker()
    result = tracker.track_video(
        video_path=video_path,
        start_ms=info.start_ms,
        end_ms=info.end_ms,
        stride=1,
        filter_enabled=True,
        court_calibrator=calibrator,
        ball_positions=ball_positions,
    )

    positions = [
        {
            "x": p.x, "y": p.y, "width": p.width, "height": p.height,
            "trackId": p.track_id, "frameNumber": p.frame_number,
            "confidence": p.confidence,
        }
        for p in result.positions
    ]
    # Capture this run's own raw positions (pre-filter BoT-SORT output) so
    # chimera detection matches against the same BoT-SORT run that produced
    # the filtered output. Using DB raw_positions_json from a different
    # original tracking run would conflate "chimera" with "ID renumbering
    # across BoT-SORT runs".
    raw_positions = [
        {
            "x": p.x, "y": p.y, "width": p.width, "height": p.height,
            "trackId": p.track_id, "frameNumber": p.frame_number,
            "confidence": p.confidence,
        }
        for p in result.raw_positions
    ]
    return {
        "primary_track_ids": list(result.primary_track_ids),
        "frame_count": result.frame_count,
        "positions": positions,
        "raw_positions": raw_positions,
    }


def _load_raw_positions(rally_id: str) -> list[dict[str, Any]]:
    from rallycut.evaluation.db import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT raw_positions_json FROM player_tracks WHERE rally_id = %s",
                (rally_id,),
            )
            row = cur.fetchone()
    if row is None or row[0] is None:
        return []
    return list(row[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-pass chimera ablation")
    parser.add_argument("--rally", required=True)
    parser.add_argument("--ks", type=str, default=None,
                        help="Comma-separated K values to run (default: all 0..7)")
    args = parser.parse_args()

    rally_id = args.rally
    rally_short = rally_id.split("-")[0]
    out_root = OUT_DIR / rally_short
    out_root.mkdir(parents=True, exist_ok=True)

    # Default: K=0..7 (8 cells)
    ks = list(range(len(PASSES) + 1)) if args.ks is None else [int(k) for k in args.ks.split(",")]

    print(f"Rally: {rally_id}")
    print(f"Running K = {ks}")
    print()

    summary_rows: list[str] = []
    summary_rows.append(
        f"{'K':>2} {'pass_added':<32} {'primary_ids':<24} {'#filtered':>9} {'#chimeras':>9} {'chimera tracks ← raw evidence'}"
    )

    for k in ks:
        _set_skip_flags(k)
        passes_str = ", ".join([PASSES[i][0] for i in range(k)]) or "<none>"
        print(f"=== K={k}  enabled: {passes_str} ===")
        sys.stdout.flush()
        try:
            result = _retrack(rally_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            continue

        # Use this run's OWN raw_positions for chimera detection
        ev = _runs_evidence(result["positions"], result["raw_positions"])
        chimera_tids = sorted([t for t, info in ev.items() if info["is_chimera"]])
        n_filtered = len(ev)
        n_chimeras = len(chimera_tids)

        # Print per-K summary
        primary_ids = result["primary_track_ids"]
        primary_str = str(primary_ids)
        chim_summary_parts = []
        for tid in chimera_tids:
            info = ev[tid]
            chim_summary_parts.append(f"T{tid}←{info['evidence_raw_tracks']}")
        chim_summary = "  ".join(chim_summary_parts) or "(none)"
        line = (
            f"{k:>2} +{(PASSES[k-1][0] if k > 0 else '<baseline>'):<31} "
            f"{primary_str:<24} {n_filtered:>9} {n_chimeras:>9} {chim_summary}"
        )
        print(line)
        summary_rows.append(line)

        # Persist per-K full report
        out_path = out_root / f"k{k}.json"
        out_path.write_text(json.dumps({
            "rally_id": rally_id,
            "k": k,
            "passes_enabled": [PASSES[i][0] for i in range(k)],
            "primary_track_ids": primary_ids,
            "frame_count": result["frame_count"],
            "filtered_tracks": {str(tid): info for tid, info in ev.items()},
        }, indent=2, default=str))
        print(f"  → wrote {out_path.relative_to(ANALYSIS_ROOT)}\n")
        sys.stdout.flush()

    print()
    print("=== SUMMARY ===")
    for row in summary_rows:
        print(row)
    print()
    print("Interpretation:")
    print("  - The K where #chimeras goes UP (vs K-1) = the pass that introduced the chimera")
    print("  - The K where filtered T7-style out-of-range track APPEARS = pass that admits the chimera primary")

    return 0


if __name__ == "__main__":
    sys.exit(main())
