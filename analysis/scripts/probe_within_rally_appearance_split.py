"""Decision-gate probe for the within-rally ID-switch detector design.

For a given rally, split each primary track into N=3 contiguous time
windows; compute appearance features per window; measure pairwise
appearance distance BETWEEN windows of the SAME track and BETWEEN
windows of DIFFERENT tracks. Use the existing matcher's similarity
function so the numbers are directly comparable to what the matcher
sees.

Decision criteria for shipping an appearance-based ID-switch detector:
  - For a known ID-switch track (e.g. 09553ef1's T1), the inter-window
    distance within that track should be COMPARABLE TO or LARGER THAN
    the typical inter-track distance. If yes → appearance signal is
    strong enough; build the detector. If no (within-track distances
    are clearly smaller than inter-track) → appearance approach won't
    fire on this case; need a different signal.

Usage:
    uv run python scripts/probe_within_rally_appearance_split.py \\
        <video_id> <rally_id_prefix> [--num-windows 3]
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.match_tracker import extract_rally_appearances
from rallycut.tracking.player_features import (
    TrackAppearanceStats,
    compute_track_similarity,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.reid_general import GeneralReIDModel, WEIGHTS_PATH


@dataclass
class WindowStats:
    track_id: int
    window_idx: int
    frame_range: tuple[int, int]
    n_frames: int
    stats: TrackAppearanceStats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("rally_prefix")
    ap.add_argument("--num-windows", type=int, default=3)
    ap.add_argument("--num-samples-per-window", type=int, default=8)
    args = ap.parse_args()

    rallies = load_rallies_for_video(args.video_id)
    rally = next(
        (r for r in rallies if r.rally_id.startswith(args.rally_prefix)), None,
    )
    if rally is None:
        sys.exit(f"no rally matching prefix {args.rally_prefix}")

    print(f"Rally {rally.rally_id[:8]}, {rally.end_ms - rally.start_ms}ms, "
          f"{len(rally.positions)} positions, "
          f"primary={rally.primary_track_ids}")

    # Group positions by track.
    by_tid: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in rally.positions:
        if p.track_id < 0:
            continue
        by_tid[p.track_id].append(p)

    primary = [t for t in rally.primary_track_ids if t in by_tid]

    # Build per-track per-window position subsets.
    per_track_windows: dict[int, list[list[PlayerPosition]]] = {}
    for tid in primary:
        pts = sorted(by_tid[tid], key=lambda p: p.frame_number)
        if len(pts) < args.num_windows * args.num_samples_per_window:
            print(f"  T{tid}: too few frames ({len(pts)}) — skipping")
            continue
        window_size = len(pts) // args.num_windows
        windows = []
        for i in range(args.num_windows):
            start = i * window_size
            end = start + window_size if i < args.num_windows - 1 else len(pts)
            windows.append(pts[start:end])
        per_track_windows[tid] = windows

    if not per_track_windows:
        sys.exit("no primary tracks survived window analysis")

    # Extract per-window appearance stats by calling extract_rally_appearances
    # on each window's positions independently. This reuses the matcher's
    # canonical aggregation so the numbers we measure are exactly what the
    # matcher would see.
    reid = GeneralReIDModel(weights_path=WEIGHTS_PATH)
    video_path = get_video_path(args.video_id)
    if video_path is None:
        sys.exit("video file not resolvable")

    all_window_stats: list[WindowStats] = []
    for tid, windows in per_track_windows.items():
        for w_idx, w_positions in enumerate(windows):
            # extract_rally_appearances expects positions for ALL tracks
            # but only computes stats for those in primary_track_ids. We
            # restrict to this single track + window.
            ts = extract_rally_appearances(
                video_path=video_path,
                positions=w_positions,
                primary_track_ids=[tid],
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                num_samples=args.num_samples_per_window,
                extract_reid=True,
                reid_model=reid,
            )
            if tid not in ts:
                print(f"  T{tid} window {w_idx}: extract failed")
                continue
            all_window_stats.append(WindowStats(
                track_id=tid,
                window_idx=w_idx,
                frame_range=(
                    w_positions[0].frame_number,
                    w_positions[-1].frame_number,
                ),
                n_frames=len(w_positions),
                stats=ts[tid],
            ))

    # Pairwise distance matrix (cost from compute_track_similarity).
    n = len(all_window_stats)
    if n < 2:
        sys.exit("not enough window stats for pairwise comparison")

    print(f"\n{n} window-stats extracted:")
    for ws in all_window_stats:
        print(f"  T{ws.track_id} W{ws.window_idx}: "
              f"frames {ws.frame_range[0]}-{ws.frame_range[1]} ({ws.n_frames})")

    print("\n=== Pairwise appearance cost matrix (lower = more similar) ===")
    print("            ", end="")
    for ws in all_window_stats:
        print(f"  T{ws.track_id}W{ws.window_idx} ", end="")
    print()

    cost_matrix = np.zeros((n, n))
    for i, ws_i in enumerate(all_window_stats):
        print(f"  T{ws_i.track_id}W{ws_i.window_idx}     ", end="")
        for j, ws_j in enumerate(all_window_stats):
            if i == j:
                print(f"   .    ", end="")
                cost_matrix[i, j] = 0.0
                continue
            cost = compute_track_similarity(
                ws_i.stats, ws_j.stats, reid_blend=0.5,
            )
            cost_matrix[i, j] = cast(float, cost)
            print(f" {cost:5.3f} ", end="")
        print()

    # Within-track inter-window vs across-track inter-window cost summary.
    intra_track: dict[int, list[float]] = defaultdict(list)
    inter_track: list[float] = []
    for i, ws_i in enumerate(all_window_stats):
        for j, ws_j in enumerate(all_window_stats):
            if j <= i:
                continue
            if ws_i.track_id == ws_j.track_id:
                intra_track[ws_i.track_id].append(cost_matrix[i, j])
            else:
                inter_track.append(cost_matrix[i, j])

    print("\n=== Within-track inter-window cost (per-track) ===")
    for tid in sorted(intra_track):
        vs = intra_track[tid]
        print(f"  T{tid}: max={max(vs):.3f}, mean={np.mean(vs):.3f}, n={len(vs)}")

    print(f"\n=== Across-track inter-window cost (all pairs) ===")
    if inter_track:
        print(f"  min={min(inter_track):.3f}, "
              f"mean={np.mean(inter_track):.3f}, "
              f"median={np.median(inter_track):.3f}, "
              f"max={max(inter_track):.3f}, n={len(inter_track)}")

    # Decision verdict.
    print("\n=== DECISION GATE ===")
    if not inter_track:
        print("  Insufficient inter-track data — inconclusive.")
        return
    inter_min = min(inter_track)
    inter_median = float(np.median(inter_track))
    for tid in sorted(intra_track):
        intra_max = max(intra_track[tid])
        ratio_to_median = intra_max / inter_median if inter_median else 0.0
        if intra_max >= inter_min:
            verdict = (
                f"intra-window max {intra_max:.3f} ≥ inter-track min "
                f"{inter_min:.3f} — STRONG signal: appearance detector "
                f"would fire on T{tid}"
            )
        elif intra_max >= 0.5 * inter_median:
            verdict = (
                f"intra-window max {intra_max:.3f} = "
                f"{ratio_to_median:.2f}× inter-track median — "
                f"BORDERLINE signal on T{tid}"
            )
        else:
            verdict = (
                f"intra-window max {intra_max:.3f} = "
                f"{ratio_to_median:.2f}× inter-track median — "
                f"WEAK signal on T{tid}: appearance approach unlikely "
                f"to fire on this track"
            )
        print(f"  T{tid}: {verdict}")


if __name__ == "__main__":
    main()
