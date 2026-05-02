"""Measure cross-rally PID assignment accuracy against ground truth.

Uses `videos.player_matching_gt_json` as the source of truth. For each
labeled bbox `{cx, cy, w, h, frame, playerId}`, find the production
track whose position at that frame is closest to the GT bbox center,
look up that track's assigned PID in `match_analysis_json`.

REPORTS TWO METRICS:

1. **Direct accuracy**: matcher_pid == gt_pid as-is. Measures both
   "matcher tracks identity correctly" AND "matcher's canonical PID
   numbering matches the user's GT labeling convention."

2. **Permuted accuracy**: matcher_pid mapped through the optimal
   global permutation matcher_pid → gt_pid (Hungarian over the
   confusion matrix), then compared. Isolates "matcher tracks identity
   correctly" from canonical-numbering choices. A high permuted
   accuracy with low direct accuracy means the matcher works fine but
   its canonical convention disagrees with how GT crops were labeled
   on this video — a labeling-convention issue, not a matcher bug.

The two metrics together let you tell apart:
- Real matcher regressions (permuted accuracy drops)
- Canonical-convention mismatches (direct drops, permuted holds)
- A mix (both move)

Usage:
    uv run python scripts/measure_pid_accuracy.py <video_id>

This script is the canonical objective measurement for cross-rally
identity work. Every code change targeting matcher accuracy should
report a delta on BOTH metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.evaluation.tracking.db import get_connection, load_rallies_for_video


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    args = p.parse_args()

    rallies = load_rallies_for_video(args.video_id)
    rid_to_rally = {r.rally_id: r for r in rallies}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT player_matching_gt_json, match_analysis_json "
                "FROM videos WHERE id = %s",
                [args.video_id],
            )
            row = cur.fetchone()
            if not row:
                sys.exit(f"video not found: {args.video_id}")
            gt: dict[str, Any] = cast(dict, row[0]) if row[0] else {}
            ma: dict[str, Any] = cast(dict, row[1]) if row[1] else {}
            if isinstance(gt, str):
                gt = json.loads(gt)
            if isinstance(ma, str):
                ma = json.loads(ma)

    if not gt.get("rallies"):
        sys.exit("no player_matching_gt_json — run GT labeling first")

    # Per-rally tracker_id → final PID. Prefer appliedFullMapping when
    # available (post-remap), else fall back to trackToPlayer.
    afm_by_rally: dict[str, dict[int, int]] = {}
    for entry in ma.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id")
        if not rid:
            continue
        src = entry.get("appliedFullMapping") or entry.get("trackToPlayer") or {}
        afm_by_rally[rid] = {
            int(k): int(v) for k, v in src.items() if int(k) > 0
        }

    # Pre-remap raw positions per rally — needed because GT bboxes are
    # in the original video coordinate space and we need to find the
    # original tracker id at the labeled frame.
    positions_by_rally: dict[str, list[dict[str, Any]]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pt.rally_id, pt.pre_remap_state_json FROM player_tracks pt "
                "JOIN rallies r ON pt.rally_id = r.id WHERE r.video_id = %s",
                [args.video_id],
            )
            for rid, snap in cur.fetchall():
                if snap is None:
                    continue
                if isinstance(snap, dict):
                    snap_dict: dict[str, Any] = snap
                else:
                    snap_dict = json.loads(cast(str, snap))
                positions_by_rally[str(rid)] = snap_dict.get("positions") or []

    # Pass 1: collect every (rally, frame, gt_pid, matcher_pid) sample
    # exactly once. The DIRECT metric and the PERMUTED metric are both
    # computed from this set so they're comparable.
    samples: list[tuple[str, int, int, int]] = []
    for rid, entry in (gt.get("rallies", {}) or {}).items():
        afm = afm_by_rally.get(rid, {})
        pos = positions_by_rally.get(rid, [])
        if not pos or not afm or rid not in rid_to_rally:
            continue
        pos_by_frame: dict[int, list[dict[str, Any]]] = {}
        for q in pos:
            fn = q.get("frameNumber")
            if fn is None:
                continue
            pos_by_frame.setdefault(int(fn), []).append(q)
        for lbl in entry.get("labels", []):
            gt_pid = int(lbl["playerId"])
            cx, cy = float(lbl["cx"]), float(lbl["cy"])
            gt_frame = int(lbl["frame"])
            candidates = pos_by_frame.get(gt_frame, [])
            if not candidates:
                for delta in range(1, 10):
                    candidates = (
                        pos_by_frame.get(gt_frame - delta, [])
                        + pos_by_frame.get(gt_frame + delta, [])
                    )
                    if candidates:
                        break
            if not candidates:
                continue
            closest = min(
                candidates,
                key=lambda p: (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2,
            )
            track_id = int(closest["trackId"])
            matcher_pid = afm.get(track_id)
            if matcher_pid is None:
                continue
            samples.append((rid[:8], gt_frame, gt_pid, matcher_pid))

    if not samples:
        print("No GT labels matched any tracks.")
        return

    total = len(samples)

    # --- Direct metric: matcher_pid == gt_pid as-is. -----------------
    direct_per_rally: dict[str, tuple[int, int]] = {}
    direct_correct = 0
    direct_mismatches: list[tuple[str, int, int, int]] = []
    for rid_short, gt_frame, gt_pid, matcher_pid in samples:
        c, t = direct_per_rally.get(rid_short, (0, 0))
        ok = matcher_pid == gt_pid
        direct_per_rally[rid_short] = (c + (1 if ok else 0), t + 1)
        if ok:
            direct_correct += 1
        else:
            direct_mismatches.append(
                (rid_short, gt_frame, gt_pid, matcher_pid),
            )

    # --- Permuted metric: best global matcher_pid → gt_pid mapping. --
    # Build confusion matrix of size 4×4 (PIDs 1..4), Hungarian-match,
    # then re-score with the optimal permutation applied.
    pids = sorted({gt_pid for _, _, gt_pid, _ in samples}
                  | {mp for _, _, _, mp in samples
                     if 1 <= mp <= 4})
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    n = max(4, len(pids))
    confusion = np.zeros((n, n), dtype=int)
    for _, _, gt_pid, matcher_pid in samples:
        if matcher_pid not in pid_to_idx or gt_pid not in pid_to_idx:
            continue
        confusion[pid_to_idx[matcher_pid], pid_to_idx[gt_pid]] += 1
    # Hungarian maximizes hits → minimizes negative confusion.
    row_ind, col_ind = linear_sum_assignment(-confusion)
    perm: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(pids) and c < len(pids):
            perm[pids[r]] = pids[c]

    permuted_correct = 0
    permuted_per_rally: dict[str, tuple[int, int]] = {}
    permuted_mismatches: list[tuple[str, int, int, int]] = []
    for rid_short, gt_frame, gt_pid, matcher_pid in samples:
        mapped = perm.get(matcher_pid, matcher_pid)
        c, t = permuted_per_rally.get(rid_short, (0, 0))
        ok = mapped == gt_pid
        permuted_per_rally[rid_short] = (c + (1 if ok else 0), t + 1)
        if ok:
            permuted_correct += 1
        else:
            permuted_mismatches.append(
                (rid_short, gt_frame, gt_pid, matcher_pid),
            )

    # --- Reporting ---------------------------------------------------
    direct_pct = 100 * direct_correct / total
    permuted_pct = 100 * permuted_correct / total
    is_identity = all(perm.get(p, p) == p for p in pids)
    canonical_gap = permuted_pct - direct_pct

    print("=== DIRECT (matcher_pid as-is vs gt_pid) ===")
    print("Per-rally:")
    for rid_short, (c, t) in sorted(direct_per_rally.items()):
        pct = (100 * c / t) if t else 0
        marker = "" if c == t else "  ←"
        print(f"  {rid_short}: {c}/{t} ({pct:.0f}%){marker}")
    print(f"OVERALL: {direct_correct}/{total} = {direct_pct:.1f}%")

    print("\n=== PERMUTED (best matcher→gt mapping applied) ===")
    print(f"Optimal permutation matcher_pid → gt_pid: "
          f"{ {p: perm.get(p, p) for p in pids} }")
    print("Per-rally:")
    for rid_short, (c, t) in sorted(permuted_per_rally.items()):
        pct = (100 * c / t) if t else 0
        marker = "" if c == t else "  ←"
        print(f"  {rid_short}: {c}/{t} ({pct:.0f}%){marker}")
    print(f"OVERALL: {permuted_correct}/{total} = {permuted_pct:.1f}%")

    print("\n=== INTERPRETATION ===")
    if is_identity:
        print("• Optimal permutation is identity — matcher's canonical "
              "convention agrees with GT labeling on this video.")
        print(f"• Residual error ({total - direct_correct}/{total}) is "
              "real identity-tracking error.")
    else:
        print(f"• Permuted accuracy = {permuted_pct:.1f}% measures the "
              "matcher's identity-tracking quality WITHOUT canonical "
              "labeling penalty.")
        print(f"• Direct = {direct_pct:.1f}%; canonicalization gap "
              f"({canonical_gap:+.1f}pp) is the cost of the matcher's "
              "convention disagreeing with GT labeling on this video.")
        if permuted_pct >= 95.0 and direct_pct < 70.0:
            print("• Strong signal: matcher is doing its job — every "
                  "real identity-tracking error has been absorbed; the "
                  "remaining error IS the canonical-convention "
                  "mismatch, not a matcher bug.")
    if permuted_mismatches:
        print(f"\nResidual mismatches AFTER permutation "
              f"({len(permuted_mismatches)} of {total} samples) — these "
              f"are the real identity-tracking errors:")
        for rs, fr, gp, mp in permuted_mismatches:
            mapped = perm.get(mp, mp)
            print(f"  {rs} frame={fr}: GT=PID{gp}, "
                  f"matcher=PID{mp} → permuted PID{mapped}")


if __name__ == "__main__":
    main()
