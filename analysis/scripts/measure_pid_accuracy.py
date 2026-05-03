"""Measure cross-rally PID assignment accuracy against ground truth.

Uses `videos.player_matching_gt_json` as the source of truth. For each
labeled bbox `{cx, cy, w, h, frame, playerId}`, find the production
track whose position at that frame is closest to the GT bbox center,
look up that track's assigned PID in `match_analysis_json`.

REPORTS TWO METRICS:

1. **PERMUTED accuracy** (load-bearing): matcher_pid mapped through
   the optimal global permutation matcher_pid → gt_pid (Hungarian
   over the confusion matrix), then compared. Measures whether the
   matcher tracks identity correctly INDEPENDENT of canonical PID
   numbering. **This is the quality metric** — the user's GT
   labeling rule (confirmed 2026-05-02) is "same players same IDs
   across rallies, no specific convention beyond that," so canonical
   label disagreement isn't a quality issue.

2. **Direct accuracy** (diagnostic only): matcher_pid == gt_pid as-is.
   Useful only as a sanity-check to detect bug-introducing changes:
   if a code change unintentionally flips canonical labels on a video
   without affecting identity tracking, DIRECT will show a sudden
   gap from PERMUTED. Otherwise ignore — direct vs permuted gap is
   not a defect.

A code change is good iff PERMUTED holds or improves.

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
    print(f"• QUALITY METRIC: permuted accuracy = {permuted_pct:.1f}% "
          f"({permuted_correct}/{total}). User's labeling rule "
          "(2026-05-02) is internal consistency only, no canonical "
          "convention — so canonical label disagreement is NOT a "
          "defect. PERMUTED is the number that should hold or "
          "improve across changes.")
    if is_identity:
        print("• Direct == permuted (identity permutation). Matcher's "
              "blind-mode convention happens to align with GT labels "
              "on this video; nothing to read into it.")
    else:
        print(f"• Direct = {direct_pct:.1f}% differs from permuted by "
              f"{canonical_gap:+.1f}pp. Canonical labels disagree on "
              f"this video; not a defect — see permuted.")
    if permuted_mismatches:
        print(f"\nResidual mismatches AFTER permutation "
              f"({len(permuted_mismatches)} of {total} samples) — these "
              f"are the real identity-tracking errors:")
        for rs, fr, gp, mp in permuted_mismatches:
            mapped = perm.get(mp, mp)
            print(f"  {rs} frame={fr}: GT=PID{gp}, "
                  f"matcher=PID{mp} → permuted PID{mapped}")

    # --- ID-stability metric (distinct matcher-PIDs per GT player) ---
    # Per-physical-player measure: for each GT-identified player,
    # count how many DISTINCT matcher-PIDs end up labeling their
    # bboxes across all GT-labeled frames in the video. 1 = perfectly
    # stable (same matcher-PID always). >1 = the matcher gives the
    # same physical player different PIDs at different times — what
    # the user perceives as "PID flicker" or "1 and 4 swap" patterns.
    #
    # This is the user-facing analog to standard MOT IDSW: standard
    # IDSW counts per-track transitions, but in our post-remap world
    # each track ID has a single PID, so the flicker happens at the
    # PHYSICAL PLAYER level when BoT-SORT splits a player into two
    # tracks with different PIDs.
    #
    # Interpretation:
    #   1.0 = perfectly stable (every GT player has one matcher PID)
    #   1.X = average of X distinct PIDs per player; the further from
    #         1.0, the more the matcher disagrees with itself about
    #         which PID a given physical player should have.
    print("\n=== ID-stability per GT player ===")
    print("(How many distinct matcher PIDs label each physical player "
          "across the whole video. 1 = stable; >1 = flicker / cross-"
          "rally inconsistency.)")
    pids_per_gt_player: dict[int, set[int]] = {}
    samples_per_gt_player: dict[int, int] = {}
    for _rid_short, _frame, gt_pid, matcher_pid in samples:
        pids_per_gt_player.setdefault(gt_pid, set()).add(matcher_pid)
        samples_per_gt_player[gt_pid] = (
            samples_per_gt_player.get(gt_pid, 0) + 1
        )
    total_distinct = 0
    for gt_pid in sorted(pids_per_gt_player):
        n_distinct = len(pids_per_gt_player[gt_pid])
        n_samples = samples_per_gt_player[gt_pid]
        marker = "" if n_distinct == 1 else "  ←"
        print(f"  GT PID{gt_pid}: {n_distinct} distinct matcher PID(s) "
              f"({sorted(pids_per_gt_player[gt_pid])}) across "
              f"{n_samples} samples{marker}")
        total_distinct += n_distinct
    if pids_per_gt_player:
        avg = total_distinct / len(pids_per_gt_player)
        print(f"AVERAGE distinct PIDs per GT player: {avg:.2f} "
              f"(1.00 = perfectly stable; ≥1.5 = significant flicker)")


if __name__ == "__main__":
    main()
