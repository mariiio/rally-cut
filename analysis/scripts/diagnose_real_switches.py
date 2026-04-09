"""Diagnose all real identity switches: find exact frames, tracks, GT assignments, and context."""

from __future__ import annotations

import cv2
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rallycut.evaluation.tracking.db import load_labeled_rallies, get_video_path
from rallycut.evaluation.tracking.metrics import smart_interpolate_gt

# Same constants as compute_identity_metrics
_OVERLAP_IOU_THRESHOLD = 0.05
_MIN_SEGMENT_FRAMES = 5


def _compute_iou(a: tuple, b: tuple) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1, ax2, ay2 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx1, by1, bx2, by2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass
class SwitchDetail:
    rally_id: str
    pred_track_id: int
    gt_before: int
    gt_after: int
    switch_frame: int  # first frame of new GT assignment
    segment_before_frames: int
    segment_after_frames: int
    segment_before_range: tuple[int, int]
    segment_after_range: tuple[int, int]
    overlap_frames_between: list[int]  # overlap frames in between segments


def find_switches(rally, gt, pred_by_frame, matches_by_frame) -> list[SwitchDetail]:
    """Replicate compute_identity_metrics logic but capture switch details."""
    all_frames = sorted(matches_by_frame.keys())
    if not all_frames:
        return []

    # Per-frame: pred→gt mapping + overlap flag
    frame_info: list[tuple[int, dict[int, int], bool]] = []
    for frame in all_frames:
        pred_boxes = pred_by_frame.get(frame, [])
        matches = matches_by_frame[frame]
        p2g = {pred_id: gt_id for gt_id, pred_id in matches}

        is_overlap = False
        for i in range(len(pred_boxes)):
            for j in range(i + 1, len(pred_boxes)):
                iou = _compute_iou(
                    (pred_boxes[i][1], pred_boxes[i][2], pred_boxes[i][3], pred_boxes[i][4]),
                    (pred_boxes[j][1], pred_boxes[j][2], pred_boxes[j][3], pred_boxes[j][4]),
                )
                if iou > _OVERLAP_IOU_THRESHOLD:
                    is_overlap = True
                    break
            if is_overlap:
                break
        frame_info.append((frame, p2g, is_overlap))

    # Collect all pred IDs
    pred_ids: set[int] = set()
    for _, p2g, _ in frame_info:
        pred_ids.update(p2g.keys())

    switches = []
    for pred_id in sorted(pred_ids):
        # Non-overlap assignments for this pred
        clean: list[tuple[int, int]] = []
        overlap_frames: list[int] = []
        for frame, p2g, is_overlap in frame_info:
            if pred_id in p2g:
                if not is_overlap:
                    clean.append((frame, p2g[pred_id]))
                else:
                    overlap_frames.append(frame)

        if len(clean) < _MIN_SEGMENT_FRAMES:
            continue

        # Build segments of consistent GT assignment
        segments: list[tuple[int, int, int, int]] = []  # (gt_id, count, start_frame, end_frame)
        seg_gt = clean[0][1]
        seg_count = 1
        seg_start = clean[0][0]
        seg_end = clean[0][0]
        for i in range(1, len(clean)):
            frame, gt_id = clean[i]
            if gt_id == seg_gt:
                seg_count += 1
                seg_end = frame
            else:
                segments.append((seg_gt, seg_count, seg_start, seg_end))
                seg_gt = gt_id
                seg_count = 1
                seg_start = frame
                seg_end = frame
        segments.append((seg_gt, seg_count, seg_start, seg_end))

        # Filter to real segments
        real_segs = [s for s in segments if s[1] >= _MIN_SEGMENT_FRAMES]
        if len(real_segs) <= 1:
            continue

        # Find switches
        for i in range(1, len(real_segs)):
            if real_segs[i][0] != real_segs[i - 1][0]:
                # Find overlap frames between these two segments
                between_start = real_segs[i - 1][3]
                between_end = real_segs[i][2]
                overlaps_between = [f for f in overlap_frames if between_start <= f <= between_end]

                switches.append(SwitchDetail(
                    rally_id=rally.rally_id,
                    pred_track_id=pred_id,
                    gt_before=real_segs[i - 1][0],
                    gt_after=real_segs[i][0],
                    switch_frame=real_segs[i][2],
                    segment_before_frames=real_segs[i - 1][1],
                    segment_after_frames=real_segs[i][1],
                    segment_before_range=(real_segs[i - 1][2], real_segs[i - 1][3]),
                    segment_after_range=(real_segs[i][2], real_segs[i][3]),
                    overlap_frames_between=overlaps_between,
                ))
                break  # Only first switch per pred track (same as metrics.py)

    return switches


def _match_detections(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Simple Hungarian matching by IoU."""
    from scipy.optimize import linear_sum_assignment

    if not gt_boxes or not pred_boxes:
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    cost = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            cost[i, j] = 1.0 - _compute_iou(gt[1:], pred[1:])

    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1.0 - iou_threshold:
            matches.append((gt_boxes[r][0], pred_boxes[c][0]))

    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}
    unmatched_gt = [i for i in range(len(gt_boxes)) if gt_boxes[i][0] not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if pred_boxes[i][0] not in matched_pred]
    return matches, unmatched_gt, unmatched_pred


def extract_switch_frame(video_path: str, start_ms: float, frame_num: int) -> np.ndarray | None:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_ms)
    for i in range(frame_num + 1):
        ret, img = cap.read()
        if not ret:
            cap.release()
            return None
    cap.release()
    return img


def main() -> None:
    rallies = load_labeled_rallies()
    rally_map = {r.rally_id[:8]: r for r in rallies}

    print(f"Loaded {len(rallies)} labeled rallies\n")
    print("=" * 100)
    print("DIAGNOSING ALL REAL IDENTITY SWITCHES")
    print("=" * 100)

    all_switches: list[tuple[SwitchDetail, object]] = []

    for rally in rallies:
        gt = smart_interpolate_gt(rally.ground_truth, rally.predictions, rally.predictions.frame_count)

        gt_by_frame: dict[int, list] = defaultdict(list)
        pred_by_frame: dict[int, list] = defaultdict(list)

        for p in gt.player_positions:
            gt_by_frame[p.frame_number].append(
                (p.track_id, p.x, p.y, p.width, p.height)
            )
        for p in rally.predictions.positions:
            pred_by_frame[p.frame_number].append(
                (p.track_id, p.x, p.y, p.width, p.height)
            )

        # Build matches
        matches_by_frame: dict[int, list[tuple[int, int]]] = {}
        all_frames = sorted(set(list(gt_by_frame.keys()) + list(pred_by_frame.keys())))
        for frame in all_frames:
            gt_boxes = gt_by_frame.get(frame, [])
            pred_boxes = pred_by_frame.get(frame, [])
            if gt_boxes and pred_boxes:
                m, _, _ = _match_detections(gt_boxes, pred_boxes, 0.5)
                matches_by_frame[frame] = m

        switches = find_switches(rally, gt, pred_by_frame, matches_by_frame)
        for s in switches:
            all_switches.append((s, rally))

    print(f"\nFound {len(all_switches)} real identity switches across {len(rallies)} rallies\n")

    for idx, (sw, rally) in enumerate(all_switches, 1):
        print(f"\n{'─' * 80}")
        print(f"SWITCH {idx}/{len(all_switches)}")
        print(f"{'─' * 80}")
        print(f"  Rally:        {sw.rally_id[:8]} (video: {rally.video_id[:8]})")
        print(f"  Pred track:   p{sw.pred_track_id}")
        print(f"  GT change:    GT{sw.gt_before} → GT{sw.gt_after}")
        is_cross_team = abs(sw.gt_before - sw.gt_after) > 1  # rough heuristic
        print(f"  Type:         {'cross-team (likely)' if is_cross_team else 'same-team (likely)'}")
        print(f"  Switch frame: {sw.switch_frame}")
        print(f"  Before:       frames {sw.segment_before_range[0]}-{sw.segment_before_range[1]} ({sw.segment_before_frames} clean frames)")
        print(f"  After:        frames {sw.segment_after_range[0]}-{sw.segment_after_range[1]} ({sw.segment_after_frames} clean frames)")
        gap = sw.segment_after_range[0] - sw.segment_before_range[1]
        print(f"  Gap:          {gap} frames ({len(sw.overlap_frames_between)} overlap frames in gap)")

        # Get pred track positions around the switch
        gt_interp = smart_interpolate_gt(rally.ground_truth, rally.predictions, rally.predictions.frame_count)
        pred_by_frame_local: dict[int, list] = defaultdict(list)
        for p in rally.predictions.positions:
            pred_by_frame_local[p.frame_number].append(
                (p.track_id, p.x, p.y, p.width, p.height)
            )

        # Show positions around switch
        print(f"\n  Pred track p{sw.pred_track_id} positions around switch:")
        context_start = max(0, sw.switch_frame - 10)
        context_end = sw.switch_frame + 10
        for frame in range(context_start, context_end + 1):
            preds = pred_by_frame_local.get(frame, [])
            track_pos = [p for p in preds if p[0] == sw.pred_track_id]
            if track_pos:
                _, x, y, w, h = track_pos[0]
                marker = " <-- SWITCH" if frame == sw.switch_frame else ""
                print(f"    frame {frame:4d}: ({x:.3f}, {y:.3f}) {w:.3f}x{h:.3f}{marker}")

        # Show ALL pred tracks around switch to see what's happening
        print(f"\n  All pred tracks around switch frame {sw.switch_frame}:")
        for frame in [sw.switch_frame - 5, sw.switch_frame - 2, sw.switch_frame, sw.switch_frame + 2, sw.switch_frame + 5]:
            preds = pred_by_frame_local.get(frame, [])
            if preds:
                tracks_str = "  ".join(
                    f"p{p[0]}:({p[1]:.3f},{p[2]:.3f})"
                    for p in sorted(preds, key=lambda x: x[0])
                )
                print(f"    frame {frame:4d}: {tracks_str}")

        # Show GT positions around switch
        gt_by_frame_local: dict[int, list] = defaultdict(list)
        for p in gt_interp.player_positions:
            gt_by_frame_local[p.frame_number].append(
                (p.track_id, p.x, p.y, p.width, p.height)
            )

        print(f"\n  GT{sw.gt_before} and GT{sw.gt_after} positions around switch:")
        for frame in range(context_start, context_end + 1):
            gts = gt_by_frame_local.get(frame, [])
            relevant = [g for g in gts if g[0] in (sw.gt_before, sw.gt_after)]
            if relevant:
                parts = []
                for g in sorted(relevant, key=lambda x: x[0]):
                    parts.append(f"GT{g[0]}:({g[1]:.3f},{g[2]:.3f})")
                marker = " <-- SWITCH" if frame == sw.switch_frame else ""
                print(f"    frame {frame:4d}: {'  '.join(parts)}{marker}")

        # Compute distance between the two GT players at switch point
        gt_at_switch = gt_by_frame_local.get(sw.switch_frame, [])
        gt_before_pos = [g for g in gt_at_switch if g[0] == sw.gt_before]
        gt_after_pos = [g for g in gt_at_switch if g[0] == sw.gt_after]
        if gt_before_pos and gt_after_pos:
            dx = gt_before_pos[0][1] - gt_after_pos[0][1]
            dy = gt_before_pos[0][2] - gt_after_pos[0][2]
            dist = (dx ** 2 + dy ** 2) ** 0.5
            print(f"\n  Distance between GT{sw.gt_before} and GT{sw.gt_after} at switch: {dist:.3f} ({dist*1920:.0f}px at 1920w)")
        else:
            print(f"\n  Could not compute GT distance at switch frame")

        # Try to save visualization
        video_path = get_video_path(rally.video_id)
        if video_path:
            out_dir = Path("analysis/outputs/switch_diagnosis")
            out_dir.mkdir(parents=True, exist_ok=True)

            frames_to_grab = [
                sw.switch_frame - 15,
                sw.switch_frame - 5,
                sw.switch_frame,
                sw.switch_frame + 5,
                sw.switch_frame + 15,
            ]
            labels = ["15 BEFORE", "5 BEFORE", "SWITCH", "5 AFTER", "15 AFTER"]

            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_MSEC, rally.start_ms)
            grabbed: dict[int, np.ndarray] = {}
            max_frame = max(frames_to_grab) + 1
            for i in range(max_frame):
                ret, img = cap.read()
                if not ret:
                    break
                if i in frames_to_grab:
                    grabbed[i] = img
            cap.release()

            if grabbed:
                h, w = list(grabbed.values())[0].shape[:2]
                panels = []
                for fn, label in zip(frames_to_grab, labels):
                    if fn not in grabbed:
                        continue
                    img = grabbed[fn].copy()

                    # Draw pred tracks
                    preds = pred_by_frame_local.get(fn, [])
                    for p in preds:
                        pid, px, py, pw, ph = p
                        x1 = int((px - pw / 2) * w)
                        y1 = int((py - ph / 2) * h)
                        x2 = int((px + pw / 2) * w)
                        y2 = int((py + ph / 2) * h)
                        color = (0, 255, 0) if pid == sw.pred_track_id else (128, 128, 128)
                        thickness = 4 if pid == sw.pred_track_id else 2
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(img, f"p{pid}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

                    # Draw GT positions for the two involved GTs
                    gts = gt_by_frame_local.get(fn, [])
                    for g in gts:
                        if g[0] in (sw.gt_before, sw.gt_after):
                            gid, gx, gy, gw, gh = g
                            cx = int(gx * w)
                            cy = int(gy * h)
                            color = (0, 0, 255) if gid == sw.gt_before else (255, 0, 0)
                            cv2.circle(img, (cx, cy), 15, color, 3)
                            cv2.putText(img, f"GT{gid}", (cx + 20, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

                    # Label
                    is_switch = fn == sw.switch_frame
                    bg_color = (0, 200, 200) if is_switch else (40, 40, 40)
                    cv2.rectangle(img, (0, 0), (500, 50), bg_color, -1)
                    cv2.putText(img, f"{label} (frame {fn})", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0) if is_switch else (255, 255, 255), 3)

                    # Scale down to fit
                    scale = 960 / w
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    panels.append(img)

                if panels:
                    # Title bar
                    pw = panels[0].shape[1]
                    title = np.zeros((60, pw, 3), dtype=np.uint8)
                    cv2.putText(title, f"Switch {idx}: {sw.rally_id[:8]} p{sw.pred_track_id} GT{sw.gt_before}->GT{sw.gt_after}",
                                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
                    final = np.vstack([title] + panels)
                    out_path = out_dir / f"switch_{idx}_{sw.rally_id[:8]}_p{sw.pred_track_id}.png"
                    cv2.imwrite(str(out_path), final)
                    print(f"\n  Saved: {out_path}")

    # Summary
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total real identity switches: {len(all_switches)}")

    if all_switches:
        gaps = [sw.segment_after_range[0] - sw.segment_before_range[1] for sw, _ in all_switches]
        overlaps = [len(sw.overlap_frames_between) for sw, _ in all_switches]
        print(f"Frame gaps at switch point: {gaps}")
        print(f"Overlap frames in gap: {overlaps}")

        print(f"\nPer-switch summary:")
        print(f"{'#':>3} {'Rally':>10} {'Track':>6} {'GT':>12} {'Frame':>6} {'Gap':>5} {'Overlaps':>9} {'GT dist':>8}")
        for idx, (sw, rally) in enumerate(all_switches, 1):
            gap = sw.segment_after_range[0] - sw.segment_before_range[1]

            gt_interp2 = smart_interpolate_gt(rally.ground_truth, rally.predictions, rally.predictions.frame_count)
            gt_by_f: dict[int, list] = defaultdict(list)
            for p in gt_interp2.player_positions:
                gt_by_f[p.frame_number].append((p.track_id, p.x, p.y))
            gt_at = gt_by_f.get(sw.switch_frame, [])
            gb = [g for g in gt_at if g[0] == sw.gt_before]
            ga = [g for g in gt_at if g[0] == sw.gt_after]
            if gb and ga:
                dist = ((gb[0][1]-ga[0][1])**2 + (gb[0][2]-ga[0][2])**2)**0.5
                dist_str = f"{dist:.3f}"
            else:
                dist_str = "N/A"

            print(f"{idx:3d} {sw.rally_id[:8]:>10} p{sw.pred_track_id:<4d} GT{sw.gt_before}→GT{sw.gt_after:>3} {sw.switch_frame:6d} {gap:5d} {len(sw.overlap_frames_between):9d} {dist_str:>8}")


if __name__ == "__main__":
    main()
