"""Slow-drift bisect diagnostic.

For each panel rally tagged `slow_drift` by the verdict tool, identify the
worst-shifting PID, find its assigned track, bisect that track's frames in
half, and compare the two halves' appearance features (HSV histograms +
position centroid) to:
  (a) each other (are halves distinguishable?)
  (b) the OTHER 3 PIDs' tracks in the same rally (does either half match
      a different PID better?)

This is the falsifiable test: if halves of a slow-drift track are
distinguishable AND one half best-matches a different PID, an HSV-only
within-track split would fix the rally. If halves look identical, HSV
splitting can't help — the drift is positional only.

Usage: uv run python scripts/probe_slow_drift_bisect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.player_features import (
    HS_BINS,
    HS_RANGES,
    extract_appearance_features,
)


# Panel slow_drift rallies (from verdict tool).
TARGETS = [
    {
        "video_id": "b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
        "rally_idx": 9,           # r10 (0-indexed 9)
        "worst_pid": 3,           # PID with half-shift=0.21
        "shift": 0.21,
        "xrange_overlap": 0.56,
    },
    {
        "video_id": "5c756c41-1cc1-4486-a95c-97398912cfbe",
        "rally_idx": 6,           # r07 (0-indexed 6)
        "worst_pid": 4,           # PID with half-shift=0.58
        "shift": 0.58,
        "xrange_overlap": 0.51,
    },
]


def _resolve_rally_id(video_id: str, rally_idx: int) -> tuple[str, list[dict]] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id, pt.positions_json FROM rallies r
                   JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s AND pt.positions_json IS NOT NULL
                   ORDER BY r.start_ms""",
                [video_id],
            )
            rows = cur.fetchall()
    if rally_idx >= len(rows):
        return None
    rid, pos = rows[rally_idx]
    return str(rid), pos if isinstance(pos, list) else []


def _hist_chi2(a: np.ndarray, b: np.ndarray) -> float:
    """Chi-square distance between two normalized histograms."""
    eps = 1e-9
    return float(np.sum((a - b) ** 2 / (a + b + eps)))


def _track_for_pid(positions: list[dict], pid: int) -> int | None:
    """Find the track_id that maps to this pid (must be unique on remapped data)."""
    counts: dict[int, int] = {}
    for p in positions:
        if p.get("trackId") == pid:
            counts[pid] = counts.get(pid, 0) + 1
    if pid not in counts:
        return None
    # On remapped positions, trackId IS the pid (1-4). So we extract crops
    # for THIS pid from the original frames; "track" is implicit.
    return pid


def _extract_halved_features(
    video_path: Path,
    positions_for_pid: list[dict],
    rally_start_ms: int,
    fps: float,
) -> tuple[np.ndarray | None, np.ndarray | None, dict]:
    """Pull frames for this PID's positions, split by frame_number median,
    return (lower_hist_first_half, lower_hist_second_half, debug_info)."""
    if not positions_for_pid:
        return None, None, {"error": "no positions"}
    sorted_pos = sorted(positions_for_pid, key=lambda p: p.get("frameNumber", 0))
    n = len(sorted_pos)
    half = n // 2
    first = sorted_pos[:half]
    second = sorted_pos[half:]
    debug = {
        "n_total": n,
        "first_n": len(first),
        "second_n": len(second),
        "first_x_mean": float(np.mean([p["x"] for p in first])) if first else None,
        "second_x_mean": float(np.mean([p["x"] for p in second])) if second else None,
        "first_y_mean": float(np.mean([p["y"] for p in first])) if first else None,
        "second_y_mean": float(np.mean([p["y"] for p in second])) if second else None,
    }

    pass  # local capture per call inside _avg_hist

    def _avg_hist(samples: list[dict], max_frames: int = 12) -> np.ndarray | None:
        if not samples:
            return None
        step = max(1, len(samples) // max_frames)
        chosen = samples[::step][:max_frames]
        accum = None
        count = 0
        local_cap = cv2.VideoCapture(str(video_path))
        if not local_cap.isOpened():
            return None
        local_fps = local_cap.get(cv2.CAP_PROP_FPS) or fps
        fw = int(local_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(local_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for p in chosen:
            frame_no = int(p.get("frameNumber", 0))
            ms = rally_start_ms + (frame_no * 1000 / local_fps)
            local_cap.set(cv2.CAP_PROP_POS_MSEC, ms)
            ok, frame = local_cap.read()
            if not ok or frame is None:
                continue
            cx, cy = float(p.get("x", 0)), float(p.get("y", 0))
            w, h = float(p.get("width", 0)), float(p.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            features = extract_appearance_features(
                frame=frame,
                track_id=int(p.get("trackId", 0)),
                frame_number=frame_no,
                bbox=(cx, cy, w, h),
                frame_width=fw,
                frame_height=fh,
            )
            if features.lower_body_hist is not None:
                if accum is None:
                    accum = features.lower_body_hist.astype(np.float32)
                else:
                    accum = accum + features.lower_body_hist
                count += 1
        local_cap.release()
        if accum is None or count == 0:
            return None
        return accum / count

    h_first = _avg_hist(first)
    h_second = _avg_hist(second)
    return h_first, h_second, debug


def main() -> None:
    for target in TARGETS:
        print(f"\n{'='*70}")
        print(f"Rally: {target['video_id'][:8]} idx={target['rally_idx']} "
              f"(slow_drift PID{target['worst_pid']}, shift={target['shift']})")
        print(f"{'='*70}")

        video_path = get_video_path(target["video_id"])
        if video_path is None:
            print("  no video path")
            continue

        rid_pos = _resolve_rally_id(target["video_id"], target["rally_idx"])
        if rid_pos is None:
            print("  rally not found")
            continue
        rally_id, positions = rid_pos
        print(f"  rally_id: {rally_id[:8]}, positions: {len(positions)}")

        # Get rally start_ms
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT start_ms FROM rallies WHERE id = %s", [rally_id])
                row = cur.fetchone()
                start_ms = int(row[0]) if row else 0

        # Group positions by PID (positions_json is post-remap, so trackId == pid)
        by_pid: dict[int, list[dict]] = {}
        for p in positions:
            pid = p.get("trackId")
            if pid in (1, 2, 3, 4):
                by_pid.setdefault(pid, []).append(p)

        print(f"  positions by PID: {{pid: count for pid, count in sorted({{k: len(v) for k, v in by_pid.items()}}.items())}}")
        for pid, lst in sorted(by_pid.items()):
            print(f"    PID{pid}: {len(lst)} positions")

        # Bisect the worst PID's positions
        worst_pid = target["worst_pid"]
        worst_positions = by_pid.get(worst_pid, [])
        if len(worst_positions) < 24:
            print(f"  PID{worst_pid} too few positions ({len(worst_positions)}) — skip")
            continue

        h_first, h_second, debug = _extract_halved_features(
            video_path, worst_positions, start_ms, fps=30.0,
        )
        print(f"  bisect debug: {debug}")
        if h_first is None or h_second is None:
            print("  feature extraction failed")
            continue

        # Inter-half distance
        inter_half = _hist_chi2(h_first, h_second)
        print(f"  worst-PID half-vs-half lower_hist chi2: {inter_half:.4f}")

        # Compare each half to OTHER PIDs' aggregated histograms
        other_hists: dict[int, np.ndarray] = {}
        for other_pid, other_pos in by_pid.items():
            if other_pid == worst_pid or len(other_pos) < 12:
                continue
            h_other, _, _ = _extract_halved_features(
                video_path, other_pos, start_ms, fps=30.0,
            )
            if h_other is not None:
                # Use first half of the other PID as its aggregated profile
                other_hists[other_pid] = h_other

        print(f"  comparing to {len(other_hists)} other PIDs:")
        for other_pid, h_other in sorted(other_hists.items()):
            d_first = _hist_chi2(h_first, h_other)
            d_second = _hist_chi2(h_second, h_other)
            tag1 = " *" if d_first < inter_half / 2 else ""
            tag2 = " *" if d_second < inter_half / 2 else ""
            print(f"    PID{other_pid}: first-half chi2={d_first:.4f}{tag1}, "
                  f"second-half chi2={d_second:.4f}{tag2}")

        # Verdict
        if inter_half < 0.05:
            print(f"  VERDICT: halves INDISTINGUISHABLE (chi2={inter_half:.4f}). "
                  "HSV split CANNOT help; pose-anchored or other signal needed.")
        elif inter_half >= 0.10:
            min_other_first = min((_hist_chi2(h_first, h) for h in other_hists.values()), default=999.0)
            min_other_second = min((_hist_chi2(h_second, h) for h in other_hists.values()), default=999.0)
            if min_other_first < inter_half or min_other_second < inter_half:
                print(f"  VERDICT: halves DISTINGUISHABLE (chi2={inter_half:.4f}); "
                      f"min other-PID distance ({min_other_first:.4f}, {min_other_second:.4f}) "
                      "< inter-half — at least one half best-matches a different PID. "
                      "HSV split is VIABLE.")
            else:
                print(f"  VERDICT: halves DIFFER (chi2={inter_half:.4f}) but neither best-matches "
                      "another PID — drift is appearance-noise, not identity. HSV split unsafe.")
        else:
            print(f"  VERDICT: halves marginally different (chi2={inter_half:.4f}); inconclusive.")


if __name__ == "__main__":
    main()
