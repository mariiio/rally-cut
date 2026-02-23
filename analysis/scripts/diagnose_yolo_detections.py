#!/usr/bin/env python3
"""Diagnose YOLO raw detections for a specific rally.

Runs YOLO11s@1280 on each frame with multiple confidence thresholds to
understand why only 3 of 4 players are detected in a given rally.

Key questions answered:
1. Is the 4th player detectable at all (low conf = 0.01)?
2. At what confidence level does it disappear?
3. Are overlapping detections (high NMS IoU) causing suppression?
4. How many survive the _filter_detections post-processing (aspect ratio, min area, cap)?

Usage:
    cd /Users/mario/Personal/Projects/RallyCut/analysis
    uv run python scripts/diagnose_yolo_detections.py \\
        --rally fad29c31-6e2a-4a8d-86f1-9064b2f1f425
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Add analysis root to path for local dev
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Confidence thresholds to evaluate
CONF_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# Default tracking confidence (what BoT-SORT uses in the pipeline)
DEFAULT_TRACKING_CONF = 0.15  # pipeline default (see player_tracker.py DEFAULT_CONFIDENCE)
DEFAULT_NMS_IOU = 0.45         # pipeline default (see player_tracker.py DEFAULT_IOU)

# Aggressive (low-thresh) run to see ALL candidates
AGGRESSIVE_CONF = 0.01
AGGRESSIVE_NMS_IOU = 0.90      # Very permissive NMS — see more overlaps

PERSON_CLASS_ID = 0
IMGSZ = 1280
YOLO_MODEL = "yolo11s"


def _compute_iou(box_a: tuple[float, float, float, float],
                 box_b: tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) absolute boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_rally_metadata(rally_id: str) -> dict:
    """Load rally start_ms, end_ms, video_id from DB."""
    from rallycut.evaluation.db import get_connection

    query = """
        SELECT r.id, r.video_id, r.start_ms, r.end_ms,
               v.fps, v.width, v.height
        FROM rallies r
        JOIN videos v ON v.id = r.video_id
        WHERE r.id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [rally_id])
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Rally {rally_id} not found in DB")
            rid, video_id, start_ms, end_ms, fps, width, height = row
            return {
                "rally_id": str(rid),
                "video_id": str(video_id),
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "fps": float(fps) if fps else 30.0,
                "width": int(width) if width else 1920,
                "height": int(height) if height else 1080,
            }


def get_video_path_for_rally(video_id: str) -> Path:
    """Resolve video to local path (download from S3 if needed)."""
    from rallycut.evaluation.tracking.db import get_video_path

    path = get_video_path(video_id)
    if path is None:
        raise FileNotFoundError(
            f"Video {video_id} not available locally and could not be downloaded"
        )
    return path


def run_yolo_on_frames(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    fps: float,
    conf: float,
    nms_iou: float,
    max_frames: int | None = None,
    stride: int = 1,
) -> list[dict]:
    """Run YOLO inference on rally frames.

    Returns a list of per-frame dicts with:
        frame_idx      : rally-relative frame index (0-based)
        abs_frame_idx  : absolute video frame index
        detections     : list of {"conf": float, "xyxy": (x1,y1,x2,y2), "cx": float, "cy": float}
    """
    from ultralytics import YOLO

    model_filename = f"{YOLO_MODEL}.pt"
    model = YOLO(model_filename)

    start_frame = int(start_ms / 1000.0 * fps)
    end_frame = int(end_ms / 1000.0 * fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # Seeking is intentional here — this diagnostic script runs YOLO on
    # arbitrary rally ranges, not full videos, so sequential read is impractical.
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    results_per_frame: list[dict] = []
    rally_frame_idx = 0
    abs_frame_idx = start_frame

    try:
        while abs_frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            if rally_frame_idx % stride == 0:
                yolo_results = model.predict(
                    frame,
                    conf=conf,
                    iou=nms_iou,
                    imgsz=IMGSZ,
                    classes=[PERSON_CLASS_ID],
                    verbose=False,
                )

                detections: list[dict] = []
                if yolo_results and len(yolo_results) > 0:
                    result = yolo_results[0]
                    if result.boxes is not None and len(result.boxes.xyxy) > 0:
                        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                        boxes_conf = result.boxes.conf.cpu().numpy()
                        for i in range(len(boxes_xyxy)):
                            x1, y1, x2, y2 = boxes_xyxy[i]
                            cx = (x1 + x2) / 2 / w
                            cy = (y1 + y2) / 2 / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            detections.append({
                                "conf": float(boxes_conf[i]),
                                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                                "cx": float(cx),
                                "cy": float(cy),
                                "width": float(bw),
                                "height": float(bh),
                            })

                results_per_frame.append({
                    "frame_idx": rally_frame_idx,
                    "abs_frame_idx": abs_frame_idx,
                    "detections": detections,
                })

                if max_frames is not None and len(results_per_frame) >= max_frames:
                    break

            rally_frame_idx += 1
            abs_frame_idx += 1
    finally:
        cap.release()

    return results_per_frame


def filter_by_threshold(detections: list[dict], conf_thresh: float) -> list[dict]:
    """Return only detections above a given confidence threshold."""
    return [d for d in detections if d["conf"] >= conf_thresh]


def find_overlapping_pairs(detections: list[dict], iou_thresh: float = 0.30) -> list[tuple[int, int]]:
    """Find pairs of detections with IoU > iou_thresh (candidates for NMS suppression)."""
    overlaps = []
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            iou = _compute_iou(detections[i]["xyxy"], detections[j]["xyxy"])
            if iou >= iou_thresh:
                overlaps.append((i, j))
    return overlaps


def simulate_filter_detections(
    detections: list[dict],
    max_per_frame: int = 8,
) -> list[dict]:
    """Simulate PlayerTracker._filter_detections (aspect ratio, min area, cap).

    Filters:
    1. Aspect ratio: height > width (persons are taller than wide; allows up to 1.5x)
    2. Zone-dependent min area: min_area = 0.0005 + 0.0025 * cy
    3. Detection count cap: keep top max_per_frame by confidence
    """
    filtered = []
    for d in detections:
        w = d["width"]
        h = d["height"]
        cy = d["cy"]

        # Aspect ratio: reject if width > 1.5 * height
        if w > 1.5 * h:
            continue

        # Zone-dependent minimum area
        min_area = 0.0005 + 0.0025 * cy
        if w * h < min_area:
            continue

        filtered.append(d)

    # Cap by confidence
    if len(filtered) > max_per_frame:
        filtered.sort(key=lambda d: d["conf"], reverse=True)
        filtered = filtered[:max_per_frame]

    return filtered


def analyze_frames(
    frames_default: list[dict],
    frames_aggressive: list[dict],
) -> dict:
    """Compute summary statistics comparing default vs aggressive thresholds.

    Returns a rich statistics dict.
    """
    total_frames = len(frames_default)
    assert len(frames_aggressive) == total_frames, \
        "Default and aggressive frame lists must have same length"

    # Per-threshold detection count distributions at aggressive-conf YOLO run
    # (because aggressive captures all candidates before confidence filtering)
    count_dist_by_thresh: dict[float, Counter] = {t: Counter() for t in CONF_THRESHOLDS}
    frames_gaining_4th_player: dict[float, int] = {t: 0 for t in CONF_THRESHOLDS}
    # Track confidence range of "4th player" detections
    fourth_player_conf_samples: list[float] = []

    # Summary across all frames
    default_count_dist: Counter = Counter()
    aggressive_count_dist: Counter = Counter()

    # Frames where default gives 3 detections but aggressive gives 4+
    frames_3_to_4_aggressive = 0
    frames_3_default = 0

    for frame_def, frame_agg in zip(frames_default, frames_aggressive):
        n_def = len(frame_def["detections"])
        n_agg = len(frame_agg["detections"])

        default_count_dist[n_def] += 1
        aggressive_count_dist[n_agg] += 1

        if n_def == 3:
            frames_3_default += 1
            if n_agg >= 4:
                frames_3_to_4_aggressive += 1

        # For each threshold, count how many dets pass
        agg_dets = frame_agg["detections"]
        for thresh in CONF_THRESHOLDS:
            n_at_thresh = sum(1 for d in agg_dets if d["conf"] >= thresh)
            count_dist_by_thresh[thresh][n_at_thresh] += 1

            # How many frames go from <4 at this threshold to 4+ at lower threshold?
            n_def_at_thresh = sum(1 for d in frame_def["detections"] if d["conf"] >= thresh)
            if n_def_at_thresh < 4 and n_agg >= 4:
                frames_gaining_4th_player[thresh] += 1

        # Collect confidence of the marginal (4th+) detection when 4+ at aggressive
        if n_agg >= 4:
            sorted_by_conf = sorted(agg_dets, key=lambda d: d["conf"])
            # If default sees fewer than 4, the "lost" ones are the low-conf ones
            if n_def < 4:
                marginal_dets = sorted_by_conf[:max(0, 4 - n_def)]
                for d in marginal_dets:
                    fourth_player_conf_samples.append(d["conf"])

    # Post-filter (_filter_detections) analysis on the default run
    postfilter_count_dist: Counter = Counter()
    for frame_def in frames_default:
        post = simulate_filter_detections(frame_def["detections"])
        postfilter_count_dist[len(post)] += 1

    # Overlap analysis (default run, NMS IoU = 0.45)
    overlap_frames_default = 0
    for frame_def in frames_default:
        if len(frame_def["detections"]) >= 2:
            pairs = find_overlapping_pairs(frame_def["detections"], iou_thresh=0.30)
            if pairs:
                overlap_frames_default += 1

    # Overlap analysis (aggressive run, permissive NMS — more overlaps visible)
    overlap_frames_aggressive = 0
    for frame_agg in frames_aggressive:
        if len(frame_agg["detections"]) >= 2:
            pairs = find_overlapping_pairs(frame_agg["detections"], iou_thresh=0.30)
            if pairs:
                overlap_frames_aggressive += 1

    return {
        "total_frames": total_frames,
        "default_count_distribution": dict(default_count_dist),
        "aggressive_count_distribution": dict(aggressive_count_dist),
        "count_distribution_by_threshold": {
            thresh: dict(dist) for thresh, dist in count_dist_by_thresh.items()
        },
        "frames_3_default": frames_3_default,
        "frames_3_to_4_aggressive": frames_3_to_4_aggressive,
        "frames_gaining_4th_player_by_threshold": frames_gaining_4th_player,
        "fourth_player_conf_samples_count": len(fourth_player_conf_samples),
        "fourth_player_conf_min": float(min(fourth_player_conf_samples)) if fourth_player_conf_samples else None,
        "fourth_player_conf_max": float(max(fourth_player_conf_samples)) if fourth_player_conf_samples else None,
        "fourth_player_conf_mean": float(np.mean(fourth_player_conf_samples)) if fourth_player_conf_samples else None,
        "fourth_player_conf_p25": float(np.percentile(fourth_player_conf_samples, 25)) if fourth_player_conf_samples else None,
        "fourth_player_conf_p50": float(np.percentile(fourth_player_conf_samples, 50)) if fourth_player_conf_samples else None,
        "fourth_player_conf_p75": float(np.percentile(fourth_player_conf_samples, 75)) if fourth_player_conf_samples else None,
        "overlap_frames_default": overlap_frames_default,
        "overlap_frames_aggressive": overlap_frames_aggressive,
        "postfilter_count_distribution": dict(postfilter_count_dist),
    }


def print_report(meta: dict, stats: dict, frames_def: list[dict], frames_agg: list[dict]) -> None:
    """Print a human-readable diagnostic report."""
    rally_id = meta["rally_id"]
    video_id = meta["video_id"]
    start_ms = meta["start_ms"]
    end_ms = meta["end_ms"]
    duration_s = (end_ms - start_ms) / 1000.0
    total = stats["total_frames"]

    print(f"\n{'='*70}")
    print("YOLO Detection Diagnostic Report")
    print(f"{'='*70}")
    print(f"Rally:   {rally_id}")
    print(f"Video:   {video_id}")
    print(f"Duration: {duration_s:.1f}s  ({total} frames processed)")
    print(f"Model:   {YOLO_MODEL}@{IMGSZ}")
    print()

    # --- Default run ---
    print(f"PIPELINE THRESHOLDS (conf={DEFAULT_TRACKING_CONF}, NMS_IoU={DEFAULT_NMS_IOU})")
    print("-" * 50)
    d_dist = stats["default_count_distribution"]
    total_frames = stats["total_frames"]
    for n in sorted(d_dist.keys()):
        pct = 100.0 * d_dist[n] / total_frames
        bar = "#" * int(pct / 2)
        print(f"  {n} players: {d_dist[n]:5d} frames ({pct:5.1f}%)  {bar}")
    pct_4plus_def = 100.0 * sum(v for k, v in d_dist.items() if k >= 4) / total_frames
    pct_3_def = 100.0 * d_dist.get(3, 0) / total_frames
    pct_lt3_def = 100.0 * sum(v for k, v in d_dist.items() if k < 3) / total_frames
    print(f"  -> 4+ players: {pct_4plus_def:.1f}%  |  3 players: {pct_3_def:.1f}%  |  <3 players: {pct_lt3_def:.1f}%")
    print(f"  -> Frames with overlapping dets (IoU>0.30): {stats['overlap_frames_default']} ({100.0*stats['overlap_frames_default']/total_frames:.1f}%)")
    print()

    # --- Aggressive run ---
    print(f"AGGRESSIVE THRESHOLDS (conf={AGGRESSIVE_CONF}, NMS_IoU={AGGRESSIVE_NMS_IOU})")
    print("-" * 50)
    a_dist = stats["aggressive_count_distribution"]
    for n in sorted(a_dist.keys()):
        pct = 100.0 * a_dist[n] / total_frames
        bar = "#" * int(pct / 2)
        print(f"  {n} players: {a_dist[n]:5d} frames ({pct:5.1f}%)  {bar}")
    pct_4plus_agg = 100.0 * sum(v for k, v in a_dist.items() if k >= 4) / total_frames
    pct_3_agg = 100.0 * a_dist.get(3, 0) / total_frames
    print(f"  -> 4+ players: {pct_4plus_agg:.1f}%  |  3 players: {pct_3_agg:.1f}%")
    print(f"  -> Frames with overlapping dets (IoU>0.30): {stats['overlap_frames_aggressive']} ({100.0*stats['overlap_frames_aggressive']/total_frames:.1f}%)")
    print()

    # --- Post-filter (_filter_detections) ---
    print("AFTER _filter_detections (aspect ratio + min area + cap@8) on default run")
    print("-" * 50)
    pf_dist = stats["postfilter_count_distribution"]
    for n in sorted(pf_dist.keys()):
        pct = 100.0 * pf_dist[n] / total_frames
        bar = "#" * int(pct / 2)
        print(f"  {n} players: {pf_dist[n]:5d} frames ({pct:5.1f}%)  {bar}")
    pct_4plus_pf = 100.0 * sum(v for k, v in pf_dist.items() if k >= 4) / total_frames
    pct_3_pf = 100.0 * pf_dist.get(3, 0) / total_frames
    pct_lt3_pf = 100.0 * sum(v for k, v in pf_dist.items() if k < 3) / total_frames
    print(f"  -> 4+ players: {pct_4plus_pf:.1f}%  |  3 players: {pct_3_pf:.1f}%  |  <3 players: {pct_lt3_pf:.1f}%")
    print()

    # --- 3->4 analysis ---
    frames_3_def = stats["frames_3_default"]
    frames_3_to_4 = stats["frames_3_to_4_aggressive"]
    print("3->4 PLAYER TRANSITION ANALYSIS")
    print("-" * 50)
    print(f"  Frames with exactly 3 players (default): {frames_3_def} ({100.0*frames_3_def/total_frames:.1f}%)")
    if frames_3_def > 0:
        print(f"  Of those, gain 4th player at conf=0.01:  {frames_3_to_4} ({100.0*frames_3_to_4/frames_3_def:.1f}%)")
    print()
    print("  Detection count at aggressive conf=0.01, grouped by threshold:")
    print(f"  {'Threshold':>10} {'4+ frames':>12} {'% of total':>12}")
    print(f"  {'-'*36}")
    thr_data = stats["count_distribution_by_threshold"]
    for thresh in CONF_THRESHOLDS:
        dist = thr_data[thresh]
        n_4plus = sum(v for k, v in dist.items() if k >= 4)
        pct = 100.0 * n_4plus / total_frames
        print(f"  {thresh:>10.2f} {n_4plus:>12d} {pct:>11.1f}%")
    print()

    # --- 4th player confidence range ---
    print("4TH PLAYER CONFIDENCE RANGE (when 4+ visible at conf=0.01 but fewer at default)")
    print("-" * 50)
    n_samples = stats["fourth_player_conf_samples_count"]
    if n_samples == 0:
        print("  (no frames where aggressive detects 4+ but default detects fewer)")
    else:
        print(f"  Samples: {n_samples}")
        print(f"  Min:  {stats['fourth_player_conf_min']:.4f}")
        print(f"  P25:  {stats['fourth_player_conf_p25']:.4f}")
        print(f"  P50:  {stats['fourth_player_conf_p50']:.4f}")
        print(f"  P75:  {stats['fourth_player_conf_p75']:.4f}")
        print(f"  Max:  {stats['fourth_player_conf_max']:.4f}")
        print(f"  Mean: {stats['fourth_player_conf_mean']:.4f}")

        # Confidence histogram in brackets
        samples_arr = []
        for frame_agg, frame_def in zip(frames_agg, frames_def):
            n_agg = len(frame_agg["detections"])
            n_def = len(frame_def["detections"])
            if n_agg >= 4 and n_def < 4:
                agg_dets = sorted(frame_agg["detections"], key=lambda d: d["conf"])
                marginal = agg_dets[:max(0, 4 - n_def)]
                samples_arr.extend(d["conf"] for d in marginal)

        if samples_arr:
            buckets = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
            hist = Counter()
            for c in samples_arr:
                for i in range(len(buckets) - 1):
                    if buckets[i] <= c < buckets[i + 1]:
                        hist[(buckets[i], buckets[i + 1])] += 1
                        break
            print()
            print("  Confidence distribution of marginal (4th) detections:")
            for i in range(len(buckets) - 1):
                lo, hi = buckets[i], buckets[i + 1]
                count = hist.get((lo, hi), 0)
                pct = 100.0 * count / max(len(samples_arr), 1)
                bar = "#" * int(pct / 3)
                print(f"    [{lo:.2f}-{hi:.2f}): {count:5d} ({pct:5.1f}%)  {bar}")

    print()

    # --- Spatial analysis of where 4th player appears ---
    print("SPATIAL ANALYSIS: Where is the 4th player?")
    print("-" * 50)
    # For frames where aggressive gets 4+ and default gets <4,
    # show the Y position of the lost detection
    fourth_y_positions = []
    fourth_x_positions = []
    for frame_agg, frame_def in zip(frames_agg, frames_def):
        n_agg = len(frame_agg["detections"])
        n_def = len(frame_def["detections"])
        if n_agg >= 4 and n_def < 4:
            agg_dets = sorted(frame_agg["detections"], key=lambda d: d["conf"])
            marginal = agg_dets[:max(0, 4 - n_def)]
            for d in marginal:
                fourth_y_positions.append(d["cy"])
                fourth_x_positions.append(d["cx"])

    if fourth_y_positions:
        arr_y = np.array(fourth_y_positions)
        arr_x = np.array(fourth_x_positions)
        print(f"  Lost detections (at conf=0.01 vs pipeline default): {len(arr_y)}")
        print("  Y position (0=top/far, 1=bottom/near):")
        print(f"    Mean={arr_y.mean():.3f}  Min={arr_y.min():.3f}  Max={arr_y.max():.3f}")
        print(f"    P25={np.percentile(arr_y,25):.3f}  P75={np.percentile(arr_y,75):.3f}")
        print("  X position (0=left, 1=right):")
        print(f"    Mean={arr_x.mean():.3f}  Min={arr_x.min():.3f}  Max={arr_x.max():.3f}")

        # Where in Y does the 4th player appear? (near vs far)
        near = (arr_y > 0.58).sum()
        mid = ((arr_y >= 0.48) & (arr_y <= 0.58)).sum()
        far = (arr_y < 0.48).sum()
        total_pts = len(arr_y)
        print("  Court zone:")
        print(f"    Far  (y<0.48):       {far:4d} ({100.0*far/total_pts:.1f}%)")
        print(f"    Mid  (0.48-0.58):    {mid:4d} ({100.0*mid/total_pts:.1f}%)")
        print(f"    Near (y>0.58):       {near:4d} ({100.0*near/total_pts:.1f}%)")
    else:
        print("  No frames where aggressive detects 4+ but default detects fewer.")
    print()

    # --- Interpretation ---
    print("INTERPRETATION")
    print("-" * 50)
    pct_detectable_at_min = 100.0 * sum(v for k, v in a_dist.items() if k >= 4) / total_frames
    if pct_detectable_at_min < 10:
        print("  CONCLUSION: The 4th player is RARELY detectable even at conf=0.01.")
        print("  Cause: Player is outside YOLO's detection range (too small, occluded,")
        print("         or outside the frame). Lowering confidence threshold will not help.")
    elif pct_detectable_at_min >= 50:
        if frames_3_def > 0 and frames_3_to_4 / frames_3_def > 0.5:
            print("  CONCLUSION: The 4th player IS detectable but is FILTERED BY CONFIDENCE.")
            print("  Fix: Lower conf threshold from 0.15 toward 0.05-0.10.")
            if stats.get("fourth_player_conf_p50") is not None:
                p50 = stats["fourth_player_conf_p50"]
                print(f"  Suggested conf threshold: {p50:.2f} (median of 4th player dets)")
        else:
            print("  CONCLUSION: The 4th player is mostly detectable. Issue may be in")
            print("  post-processing filters (aspect ratio, min area, detection cap).")
    else:
        print("  CONCLUSION: Mixed — the 4th player is sometimes detectable.")
        print("  Part of the problem is detection difficulty; part may be threshold-related.")
        if stats.get("fourth_player_conf_p50") is not None:
            p50 = stats["fourth_player_conf_p50"]
            print(f"  4th player median conf when detectable: {p50:.3f}")

    print(f"\n{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose YOLO raw detections for a rally"
    )
    parser.add_argument(
        "--rally", required=True,
        help="Rally ID (full UUID, e.g. fad29c31-6e2a-4a8d-86f1-9064b2f1f425)"
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Frame stride (1=all frames, 3=every 3rd for speed)"
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum frames to analyze (for quick tests)"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Expand partial rally ID if needed
    rally_id = args.rally

    print(f"Loading rally metadata for: {rally_id}")
    meta = load_rally_metadata(rally_id)
    print(f"  Video ID:  {meta['video_id']}")
    print(f"  Duration:  {(meta['end_ms']-meta['start_ms'])/1000:.1f}s")
    print(f"  FPS:       {meta['fps']:.1f}")

    print("Resolving video path...")
    video_path = get_video_path_for_rally(meta["video_id"])
    print(f"  Video:     {video_path}")

    total_raw_frames = int((meta["end_ms"] - meta["start_ms"]) / 1000.0 * meta["fps"])
    frames_to_process = (total_raw_frames + args.stride - 1) // args.stride
    if args.max_frames is not None:
        frames_to_process = min(frames_to_process, args.max_frames)
    print(f"  Frames:    {total_raw_frames} raw → {frames_to_process} to process (stride={args.stride})")

    # --- Run 1: Default pipeline thresholds ---
    print(f"\nRun 1: Pipeline defaults (conf={DEFAULT_TRACKING_CONF}, NMS_IoU={DEFAULT_NMS_IOU})...")
    frames_default = run_yolo_on_frames(
        video_path=video_path,
        start_ms=meta["start_ms"],
        end_ms=meta["end_ms"],
        fps=meta["fps"],
        conf=DEFAULT_TRACKING_CONF,
        nms_iou=DEFAULT_NMS_IOU,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    print(f"  Processed {len(frames_default)} frames")

    # --- Run 2: Aggressive thresholds ---
    print(f"Run 2: Aggressive (conf={AGGRESSIVE_CONF}, NMS_IoU={AGGRESSIVE_NMS_IOU})...")
    frames_aggressive = run_yolo_on_frames(
        video_path=video_path,
        start_ms=meta["start_ms"],
        end_ms=meta["end_ms"],
        fps=meta["fps"],
        conf=AGGRESSIVE_CONF,
        nms_iou=AGGRESSIVE_NMS_IOU,
        max_frames=args.max_frames,
        stride=args.stride,
    )
    print(f"  Processed {len(frames_aggressive)} frames")

    # --- Analyze ---
    stats = analyze_frames(frames_default, frames_aggressive)

    # --- Report ---
    print_report(meta, stats, frames_default, frames_aggressive)


if __name__ == "__main__":
    main()
