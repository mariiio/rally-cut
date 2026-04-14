"""Phase 2: Pose-based serve toss detection.

Detects the server by finding the player with arm-above-shoulder motion
(the serve toss) in the first ~3 seconds of each rally. Uses existing
keypoints from positions_json when available, or runs YOLO-pose on demand.

Usage:
  uv run python scripts/eval_pose_serve_detection.py
  uv run python scripts/eval_pose_serve_detection.py --coverage-only   # just check keypoint coverage
  uv run python scripts/eval_pose_serve_detection.py --extract         # run YOLO-pose for missing rallies
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval_score_tracking import (  # noqa: E402
    RallyData,
    evaluate_simple,
    load_score_gt,
    print_result,
)
from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

# COCO keypoint indices
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

# How many early frames to analyze (covers ~3s at 30fps)
EARLY_FRAMES = 90


# ── Pose data extraction ────────────────────────────────────────────────


@dataclass
class PlayerPoseSequence:
    track_id: int
    court_side: str  # "near" or "far"
    frames: list[int]
    keypoints: list[list[list[float]]]  # list of [17][3] per frame
    bbox_centers: list[tuple[float, float]]  # (cx, cy) per frame


def _extract_pose_from_positions(
    rally: RallyData,
    max_frame: int = EARLY_FRAMES,
) -> list[PlayerPoseSequence]:
    """Extract pose sequences from existing positions_json keypoints."""
    if not rally.positions or rally.court_split_y is None:
        return []

    by_track: dict[int, dict[int, dict]] = defaultdict(dict)
    for p in rally.positions:
        fn = p.get("frameNumber", 0)
        tid = p.get("trackId")
        if tid is None or tid < 0 or fn > max_frame:
            continue
        by_track[tid][fn] = p

    # Classify court sides via median Y
    track_median_y: dict[int, float] = {}
    for tid, frames in by_track.items():
        ys = [p.get("y", 0) + p.get("height", 0) / 2 for p in frames.values()]
        if ys:
            track_median_y[tid] = float(np.median(ys))

    results = []
    for tid, frames in by_track.items():
        kp_frames = {fn: p for fn, p in frames.items() if p.get("keypoints")}
        if not kp_frames:
            continue
        median_y = track_median_y.get(tid, 0)
        side = "near" if median_y > rally.court_split_y else "far"

        sorted_fns = sorted(kp_frames.keys())
        results.append(PlayerPoseSequence(
            track_id=tid,
            court_side=side,
            frames=sorted_fns,
            keypoints=[kp_frames[fn]["keypoints"] for fn in sorted_fns],
            bbox_centers=[
                (kp_frames[fn].get("x", 0) + kp_frames[fn].get("width", 0) / 2,
                 kp_frames[fn].get("y", 0) + kp_frames[fn].get("height", 0) / 2)
                for fn in sorted_fns
            ],
        ))
    return results


def _run_yolo_pose_on_rally(
    video_path: Path,
    rally: RallyData,
    pose_model: object,
    max_frame: int = EARLY_FRAMES,
) -> list[PlayerPoseSequence]:
    """Run YOLO-pose on early frames and match to tracks via position overlap."""
    import cv2

    if rally.court_split_y is None:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    abs_start_frame = int(rally.start_ms / 1000 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_start_frame)

    # Build position lookup for IoU matching
    pos_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for p in rally.positions:
        fn = p.get("frameNumber", 0)
        tid = p.get("trackId")
        if tid is None or tid < 0 or fn > max_frame:
            continue
        cx, cy = p["x"], p["y"]
        w, h = p["width"], p["height"]
        pos_by_frame.setdefault(fn, []).append((tid, (cx - w/2, cy - h/2, cx + w/2, cy + h/2)))

    # Track median Y for court side
    track_ys: dict[int, list[float]] = defaultdict(list)
    for p in rally.positions:
        tid = p.get("trackId")
        if tid is not None and tid >= 0:
            track_ys[tid].append(p.get("y", 0) + p.get("height", 0) / 2)

    track_median_y = {tid: float(np.median(ys)) for tid, ys in track_ys.items() if ys}

    # Collect keypoints per track
    track_data: dict[int, dict[str, list]] = defaultdict(lambda: {
        "frames": [], "keypoints": [], "bbox_centers": []
    })

    stride = 3  # Process every 3rd frame for speed
    for rally_frame in range(0, max_frame, stride):
        ret, frame = cap.read()
        if not ret:
            break
        if rally_frame % stride != 0:
            continue

        results = pose_model(frame, verbose=False)
        if not results or not results[0].keypoints:
            continue

        result = results[0]
        kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)

        players = pos_by_frame.get(rally_frame, [])
        if not players:
            continue

        for det_idx in range(len(boxes)):
            # Normalize bbox
            x1, y1, x2, y2 = boxes[det_idx]
            det_norm = (x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)

            # Find best IoU match with tracked players
            best_iou = 0.0
            best_tid = -1
            for tid, player_box in players:
                iou = _iou(det_norm, player_box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou < 0.3 or best_tid < 0:
                continue

            kps = kps_all[det_idx]  # (17, 3)
            # Normalize keypoints
            kps_norm = kps.copy()
            kps_norm[:, 0] /= img_w
            kps_norm[:, 1] /= img_h

            track_data[best_tid]["frames"].append(rally_frame)
            track_data[best_tid]["keypoints"].append(kps_norm.tolist())
            cx = (det_norm[0] + det_norm[2]) / 2
            cy = (det_norm[1] + det_norm[3]) / 2
            track_data[best_tid]["bbox_centers"].append((cx, cy))

    cap.release()

    results_out = []
    for tid, data in track_data.items():
        median_y = track_median_y.get(tid, 0)
        side = "near" if median_y > rally.court_split_y else "far"
        results_out.append(PlayerPoseSequence(
            track_id=tid,
            court_side=side,
            frames=data["frames"],
            keypoints=data["keypoints"],
            bbox_centers=data["bbox_centers"],
        ))
    return results_out


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Toss features ───────────────────────────────────────────────────────


@dataclass
class TossFeatures:
    track_id: int
    court_side: str
    max_wrist_above_shoulder: float  # max across all frames, positive = arm up
    toss_frame_count: int  # frames with wrist significantly above shoulder
    max_elbow_above_shoulder: float
    baseline_proximity: float  # how close to baseline (0 = at baseline, 1 = at net)
    n_frames: int


def _compute_toss_features(
    seq: PlayerPoseSequence,
    court_split_y: float,
) -> TossFeatures:
    """Compute toss-related features from a player's pose sequence."""
    max_wrist_above = 0.0
    max_elbow_above = 0.0
    toss_count = 0
    threshold = 0.03  # normalized image space

    for kps in seq.keypoints:
        if len(kps) < 11:
            continue
        l_sh = kps[LEFT_SHOULDER]
        r_sh = kps[RIGHT_SHOULDER]
        l_wr = kps[LEFT_WRIST]
        r_wr = kps[RIGHT_WRIST]
        l_el = kps[LEFT_ELBOW]
        r_el = kps[RIGHT_ELBOW]

        # Wrist above shoulder (in image: lower y = higher position)
        wrist_above = 0.0
        if l_sh[2] > 0.3 and l_wr[2] > 0.3:
            wrist_above = max(wrist_above, l_sh[1] - l_wr[1])
        if r_sh[2] > 0.3 and r_wr[2] > 0.3:
            wrist_above = max(wrist_above, r_sh[1] - r_wr[1])

        max_wrist_above = max(max_wrist_above, wrist_above)
        if wrist_above > threshold:
            toss_count += 1

        # Elbow above shoulder
        elbow_above = 0.0
        if l_sh[2] > 0.3 and l_el[2] > 0.3:
            elbow_above = max(elbow_above, l_sh[1] - l_el[1])
        if r_sh[2] > 0.3 and r_el[2] > 0.3:
            elbow_above = max(elbow_above, r_sh[1] - r_el[1])
        max_elbow_above = max(max_elbow_above, elbow_above)

    # Baseline proximity: how far from center toward baseline
    # Near baseline = high y, far baseline = low y
    if seq.bbox_centers:
        median_cy = float(np.median([c[1] for c in seq.bbox_centers]))
        if seq.court_side == "near":
            # Near baseline is at bottom of image (y=1)
            baseline_proximity = median_cy  # higher = closer to baseline
        else:
            # Far baseline is at top of image (y=0)
            baseline_proximity = 1.0 - median_cy  # higher = closer to baseline
    else:
        baseline_proximity = 0.5

    return TossFeatures(
        track_id=seq.track_id,
        court_side=seq.court_side,
        max_wrist_above_shoulder=max_wrist_above,
        toss_frame_count=toss_count,
        max_elbow_above_shoulder=max_elbow_above,
        baseline_proximity=baseline_proximity,
        n_frames=len(seq.frames),
    )


# ── Serve-side prediction ───────────────────────────────────────────────


def _side_to_team(side: str, flipped: bool) -> str:
    base = "A" if side == "near" else "B"
    if flipped:
        return "B" if base == "A" else "A"
    return base


def predict_by_toss(
    rally: RallyData,
    pose_sequences: list[PlayerPoseSequence] | None = None,
) -> str | None:
    """Predict serving team by finding the player with the strongest toss signal.

    Strategy:
    1. Find the player with the strongest arm-above-shoulder signal.
    2. Their court side → team (with side flip correction).

    Tiebreaking: toss_frame_count > max_wrist_above > baseline_proximity.
    """
    if pose_sequences is None:
        pose_sequences = _extract_pose_from_positions(rally)

    if not pose_sequences or rally.court_split_y is None:
        return None

    features = [_compute_toss_features(seq, rally.court_split_y) for seq in pose_sequences]
    features = [f for f in features if f.n_frames >= 3]  # need minimum data

    if not features:
        return None

    # Primary: max_wrist_above_shoulder
    best = max(features, key=lambda f: (
        f.max_wrist_above_shoulder,
        f.toss_frame_count,
        f.baseline_proximity,
    ))

    # If no clear toss signal, fall back to baseline proximity
    if best.max_wrist_above_shoulder < 0.02:
        # No toss detected — try baseline proximity heuristic
        # The player closest to baseline is likely the server
        best = max(features, key=lambda f: f.baseline_proximity)

    return _side_to_team(best.court_side, rally.side_flipped)


def predict_by_baseline_only(rally: RallyData) -> str | None:
    """Predict by finding the player closest to the baseline (no toss needed)."""
    if not rally.positions or rally.court_split_y is None:
        return None

    by_track: dict[int, list[float]] = defaultdict(list)
    for p in rally.positions:
        fn = p.get("frameNumber", 0)
        tid = p.get("trackId")
        if tid is None or tid < 0 or fn > EARLY_FRAMES:
            continue
        by_track[tid].append(p.get("y", 0) + p.get("height", 0))  # foot position

    if not by_track:
        return None

    # Classify sides
    track_sides: dict[int, str] = {}
    for tid, ys in by_track.items():
        median_y = float(np.median(ys))
        track_sides[tid] = "near" if median_y > rally.court_split_y else "far"

    # Find the track with most extreme foot position (closest to baseline)
    best_tid = None
    best_extremity = -1.0
    for tid, ys in by_track.items():
        median_y = float(np.median(ys))
        side = track_sides[tid]
        if side == "near":
            extremity = median_y  # closer to bottom = closer to baseline
        else:
            extremity = 1.0 - median_y  # closer to top = closer to baseline
        if extremity > best_extremity:
            best_extremity = extremity
            best_tid = tid

    if best_tid is None:
        return None

    return _side_to_team(track_sides[best_tid], rally.side_flipped)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Pose-based serve toss detection")
    parser.add_argument("--coverage-only", action="store_true",
                        help="Only check keypoint coverage, don't evaluate")
    parser.add_argument("--extract", action="store_true",
                        help="Run YOLO-pose on rallies missing keypoints")
    args = parser.parse_args()

    print("Loading score GT...")
    video_rallies = load_score_gt()
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} rallies across {len(video_rallies)} videos")

    # Coverage check
    has_kp = 0
    no_kp = 0
    for rs in video_rallies.values():
        for r in rs:
            seqs = _extract_pose_from_positions(r)
            if seqs:
                has_kp += 1
            else:
                no_kp += 1
    print(f"\nKeypoint coverage (frames 0-{EARLY_FRAMES}):")
    print(f"  Has keypoints: {has_kp}/{total} ({has_kp/total*100:.1f}%)")
    print(f"  No keypoints:  {no_kp}/{total} ({no_kp/total*100:.1f}%)")

    if args.coverage_only:
        return 0

    # Evaluate on all rallies (using existing keypoints where available)
    print("\n--- Evaluating with existing keypoints ---")

    # Toss-based predictor
    result_toss = evaluate_simple(
        "pose_toss (existing_kp)",
        predict_by_toss,
        video_rallies,
    )
    print_result(result_toss)

    # Baseline-proximity predictor (no keypoints needed)
    result_baseline = evaluate_simple(
        "baseline_proximity",
        predict_by_baseline_only,
        video_rallies,
    )
    print_result(result_baseline)

    # If --extract, run YOLO-pose on missing rallies
    if args.extract:
        print("\n--- Running YOLO-pose extraction ---")
        _run_extraction_and_eval(video_rallies)

    return 0


def _run_extraction_and_eval(video_rallies: dict[str, list[RallyData]]) -> None:
    """Run YOLO-pose on rallies missing keypoints, then evaluate."""
    from rallycut.evaluation.video_resolver import VideoResolver

    # Resolve videos
    resolver = VideoResolver()
    video_paths: dict[str, Path] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, s3_key, content_hash FROM videos
            WHERE id = ANY(%s)
        """, [list(video_rallies.keys())])
        for vid, s3_key, content_hash in cur.fetchall():
            if not s3_key or not content_hash:
                continue
            try:
                video_paths[vid] = resolver.resolve(s3_key, content_hash)
            except Exception as e:
                print(f"  WARN: {vid[:8]}: {e}")

    # Load YOLO-pose model once
    from ultralytics import YOLO
    pose_model = YOLO("yolo11s-pose.pt")

    # Extract and evaluate
    all_predictions: dict[str, str | None] = {}
    total = sum(len(rs) for rs in video_rallies.values())
    done = 0

    for vid, rallies in sorted(video_rallies.items()):
        vpath = video_paths.get(vid)
        for r in rallies:
            done += 1
            # Try existing keypoints first
            seqs = _extract_pose_from_positions(r)
            if not seqs and vpath is not None:
                # Run YOLO-pose on demand
                seqs = _run_yolo_pose_on_rally(vpath, r, pose_model)

            pred = predict_by_toss(r, pose_sequences=seqs)
            all_predictions[r.rally_id] = pred

            if done % 20 == 0:
                print(f"  [{done}/{total}] {r.rally_id[:8]} → {pred}")

    # Build predictor from cached predictions
    def predict(rally: RallyData) -> str | None:
        return all_predictions.get(rally.rally_id)

    result = evaluate_simple("pose_toss (with_extraction)", predict, video_rallies)
    print_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
