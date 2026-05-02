"""Pose body-proportion 1-NN probe on GT-labeled frames.

For every GT-labeled bbox in `videos.player_matching_gt_json`, extract the
crop, run YOLO-Pose, compute body-proportion ratios. Then build a per-PID
median profile using ALL OTHER labeled crops (leave-one-out), and check
whether the labeled crop's nearest median PID (in body-proportion space)
equals the GT PID.

This isolates the within-team teammate-discrimination question: if pose
features can pick PID4 over PID2 on `7094136a` and `8c49e480` GT crops,
the production matcher's residual errors are addressable via pose.

Usage:
    uv run python scripts/probe_pose_within_team.py <video_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import cast

import cv2
import numpy as np

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.pose_anchored_features import (
    _frame_kps_to_crop_kps,
    compute_body_proportions,
    get_pose_model,
    run_pose_on_frame,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    args = ap.parse_args()

    rallies = {r.rally_id: r for r in load_rallies_for_video(args.video_id)}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT player_matching_gt_json FROM videos WHERE id = %s",
            [args.video_id],
        )
        row = cur.fetchone()
        if not row or not row[0]:
            sys.exit("no player_matching_gt_json")
        gt = row[0] if isinstance(row[0], dict) else json.loads(cast(str, row[0]))

    video_path = get_video_path(args.video_id)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_model = get_pose_model()

    # (pid, rally_short, frame, body_props) — body_props is None when pose failed.
    samples: list[tuple[int, str, int, dict[str, float] | None]] = []

    for rid, entry in (gt.get("rallies", {}) or {}).items():
        rally = rallies.get(rid)
        if rally is None:
            continue
        labels = entry.get("labels", [])
        if not labels:
            continue
        for lbl in labels:
            pid = int(lbl["playerId"])
            cx, cy = float(lbl["cx"]), float(lbl["cy"])
            w, h = float(lbl["w"]), float(lbl["h"])
            frame_no = int(lbl["frame"])
            ms = rally.start_ms + (frame_no * 1000 / fps)
            cap.set(cv2.CAP_PROP_POS_MSEC, int(ms))
            ok, frame = cap.read()
            if not ok or frame is None:
                samples.append((pid, rid[:8], frame_no, None))
                continue

            x1n, y1n = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
            x2n, y2n = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
            pose_res = run_pose_on_frame(
                frame, {pid: (x1n, y1n, x2n, y2n)}, pose_model=pose_model,
            )
            entry_pose = pose_res.by_track.get(pid)
            if entry_pose is None:
                samples.append((pid, rid[:8], frame_no, None))
                continue
            kps_pixel, _ = entry_pose
            x1p = int(x1n * fw); y1p = int(y1n * fh)
            x2p = int(x2n * fw); y2p = int(y2n * fh)
            kps_crop = _frame_kps_to_crop_kps(kps_pixel, (x1p, y1p, x2p, y2p))
            props = compute_body_proportions(kps_crop)
            samples.append((pid, rid[:8], frame_no, props))

    cap.release()
    print(f"Extracted {len(samples)} samples; "
          f"{sum(1 for s in samples if s[3])} with body-props")

    # 1-NN leave-one-out: for each sample with body-props, build a per-PID
    # median over all OTHER samples, compute the pidwise L1 distance, and
    # take argmin.
    by_pid: dict[int, list[dict[str, float]]] = {}
    for pid, _, _, props in samples:
        if props is not None:
            by_pid.setdefault(pid, []).append(props)
    pid_keys = sorted(by_pid.keys())
    print(f"PIDs labeled: {pid_keys}; "
          f"counts: { {p: len(by_pid[p]) for p in pid_keys} }")

    pid_key_sets: list[set[str]] = []
    for samples_for_pid in by_pid.values():
        keys: set[str] = set()
        for d in samples_for_pid:
            keys.update(d.keys())
        pid_key_sets.append(keys)
    common_keys = sorted(set.intersection(*pid_key_sets)) if pid_key_sets else []
    print(f"Common ratio keys across PIDs: {common_keys}")

    if not common_keys:
        sys.exit("No ratio keys shared across all PIDs — can't run 1-NN.")

    correct_overall = 0
    total_overall = 0
    per_pid_correct: dict[int, tuple[int, int]] = {p: (0, 0) for p in pid_keys}
    detail: list[tuple[str, int, int, int, dict[int, float]]] = []

    for i, (gt_pid, rally_short, frame, props) in enumerate(samples):
        if props is None:
            continue
        # Build LOO medians per PID.
        medians: dict[int, dict[str, float]] = {}
        for pid in pid_keys:
            others = [
                p for j, (op, _, _, p) in enumerate(samples)
                if op == pid and j != i and p is not None
            ]
            if not others:
                continue
            m = {
                k: float(np.median([d[k] for d in others if k in d]))
                for k in common_keys
                if any(k in d for d in others)
            }
            medians[pid] = m
        # L1 distance using common keys present in both.
        dists: dict[int, float] = {}
        for pid, m in medians.items():
            keys = [k for k in common_keys if k in props and k in m]
            if not keys:
                continue
            dists[pid] = float(np.mean([abs(props[k] - m[k]) for k in keys]))
        if not dists:
            continue
        pred = min(dists, key=lambda p: dists[p])
        ok = pred == gt_pid
        total_overall += 1
        c, t = per_pid_correct[gt_pid]
        per_pid_correct[gt_pid] = (c + (1 if ok else 0), t + 1)
        if ok:
            correct_overall += 1
        detail.append((rally_short, frame, gt_pid, pred, dists))

    print()
    print("=== Per-PID 1-NN accuracy (leave-one-out, body-proportion L1) ===")
    for pid in pid_keys:
        c, t = per_pid_correct[pid]
        pct = (100 * c / t) if t else 0
        print(f"  PID{pid}: {c}/{t} ({pct:.0f}%)")
    if total_overall:
        print(f"  OVERALL: {correct_overall}/{total_overall} = "
              f"{(100 * correct_overall / total_overall):.1f}%")

    # Targeted: print the 4 known mismatches.
    print()
    print("=== Mismatches the production matcher had on this video ===")
    target_frames = {
        ("7094136a", 250): 2, ("7094136a", 198): 4,
        ("8c49e480", 682): 4, ("8c49e480", 758): 2,
    }
    for (rs, fr), gt_pid in sorted(target_frames.items()):
        match = next(
            (d for d in detail if d[0] == rs and d[1] == fr and d[2] == gt_pid),
            None,
        )
        if match is None:
            print(f"  {rs} frame={fr} GT=P{gt_pid}: no pose extracted")
            continue
        _, _, _, pred, dists = match
        ok = "OK" if pred == gt_pid else "WRONG"
        dlist = ", ".join(f"P{p}={dists[p]:.3f}" for p in sorted(dists))
        print(f"  {rs} frame={fr} GT=P{gt_pid} → predicted P{pred} [{ok}] "
              f"{{ {dlist} }}")


if __name__ == "__main__":
    main()
