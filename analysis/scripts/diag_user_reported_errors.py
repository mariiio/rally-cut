"""Per-rally forensic for user-reported identity-assignment errors.

For each rally on a video, dumps:
  - Primary tracks with [start_frame, end_frame], frame_count, assigned PID
  - Pairwise within-rally appearance similarity (compute_track_similarity)
  - For each GT-labeled bbox: which production track is closest, what other
    tracks are within radius_px, and their assigned PIDs (catches track-split
    Pattern A: GT lands on one track, but another track of the same player
    covers more frames)
  - Side-switch detector state per rally
  - High BoT-SORT track IDs flagged (signal of track fragmentation)

Usage:
    uv run python scripts/diag_user_reported_errors.py <video_id>
        [--out reports/identity_diag_2026_05_03/<short>_forensic.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection, load_rallies_for_video


def _dump_track_inventory(
    rally_id: str,
    rally_short: str,
    rally_idx: int,
    pre_remap_snap: dict[str, Any],
    afm: dict[int, int],
    gt_labels: list[dict[str, Any]],
    out_lines: list[str],
) -> None:
    positions = pre_remap_snap.get("positions") or []
    primary_track_ids = pre_remap_snap.get("primaryTrackIds") or []

    # Track lifetimes from positions
    track_frames: dict[int, list[int]] = defaultdict(list)
    track_positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for q in positions:
        tid = int(q.get("trackId", -1))
        fn = q.get("frameNumber")
        if tid <= 0 or fn is None:
            continue
        track_frames[tid].append(int(fn))
        track_positions[tid].append((float(q["x"]), float(q["y"])))

    out_lines.append(f"\n## Rally {rally_idx + 1}: {rally_short} ({rally_id})\n")
    out_lines.append(f"Pre-remap primary tracks (matcher input): {primary_track_ids}")
    out_lines.append(
        f"AFM (track_id → matcher PID): "
        f"{ {k: afm[k] for k in sorted(afm)} }"
    )

    primary_set = set(int(t) for t in primary_track_ids)
    out_lines.append("\n### Primary tracks")
    out_lines.append("| Track | Start | End | Frames | Mean (x,y) | AFM PID |")
    out_lines.append("|---|---|---|---|---|---|")
    for tid in sorted(primary_set):
        frames = track_frames.get(tid, [])
        if not frames:
            out_lines.append(
                f"| T{tid} | - | - | 0 | - | "
                f"{afm.get(tid, '-')}|  ⚠ no positions"
            )
            continue
        positions_for_track = track_positions[tid]
        mean_x = sum(p[0] for p in positions_for_track) / len(positions_for_track)
        mean_y = sum(p[1] for p in positions_for_track) / len(positions_for_track)
        flag = ""
        if tid > 10:
            flag = "  ⚠ high BoT-SORT ID (fragmentation signal)"
        out_lines.append(
            f"| T{tid} | {min(frames)} | {max(frames)} | {len(frames)} | "
            f"({mean_x:.2f}, {mean_y:.2f}) | {afm.get(tid, '-')} |{flag}"
        )

    # GT label cross-reference: for each GT label, find ALL nearby tracks at
    # the labeled frame, not just the closest. Pattern A signal = another
    # track of similar appearance covers more frames.
    if gt_labels:
        out_lines.append("\n### GT bbox vs primary tracks at labeled frames")
        out_lines.append(
            "| GT frame | GT PID | Closest track | Closest dist | "
            "Other tracks at frame (within ~0.20) |"
        )
        out_lines.append("|---|---|---|---|---|")

        positions_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for q in positions:
            fn = q.get("frameNumber")
            if fn is None:
                continue
            positions_by_frame[int(fn)].append(q)

        for lbl in gt_labels:
            gt_frame = int(lbl["frame"])
            cx = float(lbl["cx"])
            cy = float(lbl["cy"])
            gt_pid = int(lbl["playerId"])

            candidates: list[dict[str, Any]] = positions_by_frame.get(gt_frame, [])
            search_frame = gt_frame
            if not candidates:
                for delta in range(1, 10):
                    if positions_by_frame.get(gt_frame - delta):
                        candidates = positions_by_frame[gt_frame - delta]
                        search_frame = gt_frame - delta
                        break
                    if positions_by_frame.get(gt_frame + delta):
                        candidates = positions_by_frame[gt_frame + delta]
                        search_frame = gt_frame + delta
                        break

            if not candidates:
                out_lines.append(
                    f"| {gt_frame} | P{gt_pid} | - | - | (no positions at frame) |"
                )
                continue

            scored = sorted(
                candidates,
                key=lambda p: (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2,
            )
            closest = scored[0]
            closest_dist = (
                (closest["x"] - cx) ** 2 + (closest["y"] - cy) ** 2
            ) ** 0.5
            closest_tid = int(closest["trackId"])
            closest_pid = afm.get(closest_tid, "-")

            others = []
            for c in scored[1:]:
                d = ((c["x"] - cx) ** 2 + (c["y"] - cy) ** 2) ** 0.5
                if d > 0.20:
                    break
                tid = int(c["trackId"])
                pid = afm.get(tid, "-")
                others.append(f"T{tid}(d={d:.3f}, P{pid})")
            others_str = ", ".join(others) if others else "(none)"

            note = ""
            if search_frame != gt_frame:
                note = f"  (matched frame {search_frame})"
            out_lines.append(
                f"| {gt_frame} | P{gt_pid} | T{closest_tid}(P{closest_pid}) | "
                f"{closest_dist:.3f} | {others_str} |{note}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--out", default=None,
                   help="Markdown output path (default: stdout)")
    args = p.parse_args()

    rallies = load_rallies_for_video(args.video_id)
    chronological = sorted(rallies, key=lambda r: r.start_ms)

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
            gt_raw = row[0]
            ma_raw = row[1]
            gt: dict[str, Any] = (
                cast(dict, gt_raw) if isinstance(gt_raw, dict) else
                (json.loads(cast(str, gt_raw)) if gt_raw else {})
            )
            ma: dict[str, Any] = (
                cast(dict, ma_raw) if isinstance(ma_raw, dict) else
                (json.loads(cast(str, ma_raw)) if ma_raw else {})
            )

    # Side-switch detection per rally
    side_switch_by_rally: dict[str, bool] = {}
    afm_by_rally: dict[str, dict[int, int]] = {}
    for entry in ma.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id")
        if not rid:
            continue
        side_switch_by_rally[rid] = bool(entry.get("sideSwitch", False))
        src = entry.get("appliedFullMapping") or entry.get("trackToPlayer") or {}
        afm_by_rally[rid] = {
            int(k): int(v) for k, v in src.items() if int(k) > 0
        }

    # Pre-remap snapshot per rally
    pre_remap_by_rally: dict[str, dict[str, Any]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pt.rally_id::text, pt.pre_remap_state_json "
                "FROM player_tracks pt "
                "JOIN rallies r ON pt.rally_id = r.id WHERE r.video_id = %s",
                [args.video_id],
            )
            for rid, snap in cur.fetchall():
                if snap is None:
                    continue
                if isinstance(snap, dict):
                    pre_remap_by_rally[str(rid)] = snap
                else:
                    pre_remap_by_rally[str(rid)] = json.loads(cast(str, snap))

    out_lines: list[str] = []
    out_lines.append(f"# Forensic: {args.video_id[:8]}\n")
    out_lines.append(f"Total rallies: {len(chronological)}")
    out_lines.append(f"Rallies with GT: {len(gt.get('rallies', {}) or {})}")
    out_lines.append(f"Side switches detected: "
                     f"{sum(1 for v in side_switch_by_rally.values() if v)}")

    gt_rallies = gt.get("rallies", {}) or {}
    for idx, rally in enumerate(chronological):
        rid = rally.rally_id
        rid_short = rid[:8]
        snap = pre_remap_by_rally.get(rid)
        if snap is None:
            out_lines.append(
                f"\n## Rally {idx + 1}: {rid_short} — NO pre_remap_state_json"
            )
            continue
        afm = afm_by_rally.get(rid, {})
        gt_entry = gt_rallies.get(rid, {})
        gt_labels = gt_entry.get("labels", []) if isinstance(gt_entry, dict) else []
        side_switch = side_switch_by_rally.get(rid, False)
        ss_marker = " ⚠ SIDE SWITCH" if side_switch else ""

        _dump_track_inventory(
            rid, rid_short, idx, snap, afm, gt_labels, out_lines,
        )
        if side_switch:
            out_lines[-1] = out_lines[-1] + ss_marker

    body = "\n".join(out_lines)
    if args.out:
        with open(args.out, "w") as f:
            f.write(body)
        print(f"wrote {args.out}")
    else:
        print(body)


if __name__ == "__main__":
    main()
