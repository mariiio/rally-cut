"""
Layer 2 — Shadow assignment on full production fleet.

Builds per-video per-PID gallery from each video's anchor rallies, then
predicts each rally's PIDs (including 3-track partial-cardinality cases)
and compares to the current `appliedFullMapping`.

Output:
  - Console: per-video summary (n rallies, n disagreements, partial-cardinality cases)
  - Disagreement report at reports/identity_first_shadow/disagreements.txt
    formatted for visual inspection (1-indexed rally numbers, per-video groups,
    actual vs predicted side-by-side).

Read-only. Same gallery primitive validated in Layer 1.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.evaluation.tracking.db import get_connection

# Reuse Layer 1 primitives
from shadow_loro_gallery_validation import (  # type: ignore
    TrackFeature,
    _aggregate,
    cost,
    _is_anchor_rally,
    _track_to_pid_for_top,
)


REPORT_DIR = Path("reports/identity_first_shadow")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RallyResult:
    rally_idx: int  # 0-based
    rally_id: str
    n_top: int
    side_switch: bool
    is_anchor: bool
    actual_afm: dict[int, int]   # track_id -> pid (only meaningful entries)
    predicted_afm: dict[int, int]
    cost_summary: list[float]    # per-track cost of predicted assignment
    matches: bool


def _build_gallery_from_anchors(
    sp_rallies: list[dict],
    rallies_meta: list[dict],
    exclude_idx: int | None = None,
) -> dict[int, TrackFeature] | None:
    """Aggregate per-PID features across all anchor rallies (optionally excluding one)."""
    per_pid_features: dict[int, list[TrackFeature]] = defaultdict(list)
    for i, (r, sp) in enumerate(zip(rallies_meta, sp_rallies)):
        if exclude_idx is not None and i == exclude_idx:
            continue
        if not _is_anchor_rally(r, sp):
            continue
        top = sp["top_tracks"]
        track_to_pid = _track_to_pid_for_top(r, top)
        if len(track_to_pid) != 4:
            continue
        for tid in top:
            ts = sp["track_stats"].get(str(tid))
            if not ts:
                continue
            f = TrackFeature.from_track_stats(ts)
            if f is None:
                continue
            pid = track_to_pid[tid]
            per_pid_features[pid].append(f)
    gallery = {pid: _aggregate(feats) for pid, feats in per_pid_features.items()}
    if any(g is None for g in gallery.values()):
        return None
    if not gallery or len(gallery) < 2:
        return None
    return gallery


def _predict_rally(
    gallery: dict[int, TrackFeature],
    sp_rally: dict,
) -> tuple[dict[int, int], list[float]] | None:
    """Predict PIDs for a rally's primary tracks via rectangular Hungarian.

    Returns (track_id -> pid) and per-track cost of the predicted assignment.
    Handles N tracks vs 4 PIDs (rectangular) — partial-cardinality rallies
    naturally produce sparse output (4-N PIDs unassigned).
    """
    top = sp_rally.get("top_tracks", [])
    track_stats = sp_rally.get("track_stats", {})
    if not top:
        return None

    track_features: list[tuple[int, TrackFeature]] = []
    for tid in top:
        ts = track_stats.get(str(tid))
        if not ts:
            continue
        f = TrackFeature.from_track_stats(ts)
        if f is None:
            continue
        track_features.append((tid, f))
    if not track_features:
        return None

    available_pids = sorted(gallery.keys())
    n_tracks = len(track_features)
    n_pids = len(available_pids)

    cost_mat = np.zeros((n_tracks, n_pids), dtype=np.float32)
    for ti, (_, feat) in enumerate(track_features):
        for pj, pid in enumerate(available_pids):
            cost_mat[ti, pj] = cost(feat, gallery[pid])

    # Rectangular Hungarian: each track gets exactly one PID; if n_tracks < n_pids,
    # then n_pids - n_tracks PIDs are left unassigned.
    row_ind, col_ind = linear_sum_assignment(cost_mat)
    predicted = {}
    cost_per_track = []
    for ri, ci in zip(row_ind, col_ind):
        tid = track_features[ri][0]
        pid = available_pids[ci]
        predicted[tid] = pid
        cost_per_track.append(float(cost_mat[ri, ci]))
    return predicted, cost_per_track


def shadow_video(video_id: str, name: str) -> dict | None:
    with get_connection() as c, c.cursor() as cur:
        cur.execute("SELECT match_analysis_json FROM videos WHERE id = %s", (video_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return None
    maj = row[0]
    rallies = maj.get("rallies", [])
    sp_rallies = maj.get("rallyScratchpad", {}).get("rallies", [])
    if not rallies or len(rallies) != len(sp_rallies):
        return None

    results: list[RallyResult] = []
    for i, (r, sp) in enumerate(zip(rallies, sp_rallies)):
        # Build gallery excluding this rally (LORO-style for anchors,
        # full gallery for non-anchors)
        gallery = _build_gallery_from_anchors(sp_rallies, rallies, exclude_idx=i)
        if gallery is None or set(gallery.keys()) != {1, 2, 3, 4}:
            # Cannot evaluate this rally — gallery insufficient
            continue
        pred = _predict_rally(gallery, sp)
        if pred is None:
            continue
        predicted_afm, costs = pred

        top = sp.get("top_tracks", [])
        actual_afm = _track_to_pid_for_top(r, top)
        is_anchor = _is_anchor_rally(r, sp)
        matches = predicted_afm == actual_afm

        results.append(RallyResult(
            rally_idx=i,
            rally_id=r.get("rallyId", "?"),
            n_top=len(top),
            side_switch=bool(r.get("sideSwitchDetected", False)),
            is_anchor=is_anchor,
            actual_afm=actual_afm,
            predicted_afm=predicted_afm,
            cost_summary=costs,
            matches=matches,
        ))

    if not results:
        return None

    n_total = len(results)
    n_match = sum(1 for r in results if r.matches)
    n_anchor = sum(1 for r in results if r.is_anchor)
    n_partial = sum(1 for r in results if not r.is_anchor)
    n_disagreements = n_total - n_match
    n_disagreements_anchor = sum(1 for r in results if r.is_anchor and not r.matches)
    n_disagreements_partial = sum(1 for r in results if not r.is_anchor and not r.matches)

    return {
        "video_id": video_id,
        "name": name,
        "n_total": n_total,
        "n_match": n_match,
        "n_anchor": n_anchor,
        "n_partial": n_partial,
        "n_disagreements": n_disagreements,
        "n_disagreements_anchor": n_disagreements_anchor,
        "n_disagreements_partial": n_disagreements_partial,
        "results": results,
    }


def main() -> None:
    print("=== Layer 2: Shadow assignment on full fleet ===\n")

    # Load all videos with match_analysis_json
    with get_connection() as c, c.cursor() as cur:
        cur.execute("""
            SELECT id, name FROM videos
            WHERE match_analysis_json IS NOT NULL
            ORDER BY name
        """)
        videos = cur.fetchall()

    print(f"Found {len(videos)} videos with match_analysis_json\n")

    summaries = []
    all_disagreements_anchor: list[tuple[str, str, RallyResult]] = []
    all_disagreements_partial: list[tuple[str, str, RallyResult]] = []

    for vid, name in videos:
        s = shadow_video(vid, name or "?")
        if s is None:
            continue
        summaries.append(s)
        for r in s["results"]:
            if not r.matches:
                if r.is_anchor:
                    all_disagreements_anchor.append((vid, name or "?", r))
                else:
                    all_disagreements_partial.append((vid, name or "?", r))

    # Per-video summary
    print(f"{'name':<10} {'video':<10} {'rallies':<9} {'anchor':<8} {'partial':<8} "
          f"{'match':<7} {'disagree':<9} {'pct':<7}")
    for s in summaries:
        pct = (s["n_match"] / s["n_total"] * 100) if s["n_total"] else 0.0
        print(f"{s['name']:<10} {s['video_id'][:8]:<10} {s['n_total']:<9} "
              f"{s['n_anchor']:<8} {s['n_partial']:<8} {s['n_match']:<7} "
              f"{s['n_disagreements']:<9} {pct:<7.1f}")

    total_rallies = sum(s["n_total"] for s in summaries)
    total_match = sum(s["n_match"] for s in summaries)
    total_anchor = sum(s["n_anchor"] for s in summaries)
    total_partial = sum(s["n_partial"] for s in summaries)
    total_dis_anchor = sum(s["n_disagreements_anchor"] for s in summaries)
    total_dis_partial = sum(s["n_disagreements_partial"] for s in summaries)
    overall_pct = (total_match / total_rallies * 100) if total_rallies else 0.0

    print(f"\nFLEET TOTAL: {total_rallies} rallies, {total_match} match, "
          f"{total_rallies - total_match} disagreements ({overall_pct:.1f}% match)")
    print(f"  Anchor rallies (4-track):  {total_anchor} total, {total_dis_anchor} disagree")
    print(f"  Partial rallies (<4 track): {total_partial} total, {total_dis_partial} disagree")

    # Write disagreement report for visual inspection
    report_path = REPORT_DIR / "disagreements.txt"
    with open(report_path, "w") as f:
        f.write("# Identity-first matcher: shadow vs current AFM disagreements\n")
        f.write("# Format: each line is a rally where the proposed gallery-based\n")
        f.write("# matcher would predict different PIDs than the currently-stored AFM.\n")
        f.write("# Visual inspection: open the editor for each rally and confirm which is correct.\n#\n")
        f.write(f"# Total fleet: {total_rallies} rallies, {len(all_disagreements_anchor) + len(all_disagreements_partial)} disagreements\n")
        f.write(f"# - {len(all_disagreements_anchor)} on anchor (4-track) rallies\n")
        f.write(f"# - {len(all_disagreements_partial)} on partial (<4-track) rallies — primary fix target\n\n")

        f.write("=" * 80 + "\n")
        f.write("PARTIAL-CARDINALITY RALLIES (likely fix candidates)\n")
        f.write("=" * 80 + "\n\n")

        # Group partial disagreements by video
        by_video_partial = defaultdict(list)
        for vid, name, r in all_disagreements_partial:
            by_video_partial[(vid, name)].append(r)

        for (vid, name), rs in sorted(by_video_partial.items(), key=lambda x: x[0][1]):
            f.write(f"## {name} ({vid[:8]})\n")
            for r in sorted(rs, key=lambda x: x.rally_idx):
                f.write(f"  rally #{r.rally_idx + 1} ({r.rally_id[:8]}) "
                        f"n_top={r.n_top}{' [SW]' if r.side_switch else ''}\n")
                f.write(f"    current:   {r.actual_afm}\n")
                f.write(f"    proposed:  {r.predicted_afm}\n")
                f.write(f"    cost/track: {[f'{c:.2f}' for c in r.cost_summary]}\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ANCHOR (4-TRACK) DISAGREEMENTS — possible matcher errors caught by gallery\n")
        f.write("=" * 80 + "\n\n")

        by_video_anchor = defaultdict(list)
        for vid, name, r in all_disagreements_anchor:
            by_video_anchor[(vid, name)].append(r)

        for (vid, name), rs in sorted(by_video_anchor.items(), key=lambda x: x[0][1]):
            f.write(f"## {name} ({vid[:8]})\n")
            for r in sorted(rs, key=lambda x: x.rally_idx):
                f.write(f"  rally #{r.rally_idx + 1} ({r.rally_id[:8]}) "
                        f"n_top={r.n_top}{' [SW]' if r.side_switch else ''}\n")
                f.write(f"    current:   {r.actual_afm}\n")
                f.write(f"    proposed:  {r.predicted_afm}\n")
                f.write(f"    cost/track: {[f'{c:.2f}' for c in r.cost_summary]}\n\n")

    print(f"\nFull report: {report_path}")
    print(f"  partial-cardinality disagreements: {len(all_disagreements_partial)}")
    print(f"  anchor (4-track) disagreements:    {len(all_disagreements_anchor)}")

    # Targeted summary for user-flagged videos
    print("\n=== Targeted: known user-reported videos ===")
    targeted = {
        "dd042609": "tutu (Bug D — server occlusion)",
        "1a5da176": "moma (Bug C r4 within-team P3↔P4)",
        "4f2bd66a": "lulu (Bug C r9)",
        "b026dc6c": "lili (Bug C r10/r17)",
        "073cb11b": "wuwu (Pattern E — recovered)",
        "2d105b7b": "pipi (Pattern E — partial)",
    }
    for s in summaries:
        if s["video_id"][:8] in targeted:
            print(f"\n{targeted[s['video_id'][:8]]}:")
            for r in s["results"]:
                if not r.matches:
                    flag = "PARTIAL" if not r.is_anchor else "ANCHOR"
                    print(f"  rally #{r.rally_idx + 1} ({r.rally_id[:8]}) [{flag}] n_top={r.n_top}")
                    print(f"    current:  {r.actual_afm}")
                    print(f"    proposed: {r.predicted_afm}")


if __name__ == "__main__":
    main()
