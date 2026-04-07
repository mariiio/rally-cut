"""YOLO-Pose keypoint extraction and disk caching.

Runs YOLO-Pose on full video frames at contact windows, matches pose
detections to existing player tracks via bbox IoU, and caches results
per rally as compressed numpy archives.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO 17-keypoint indices
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_HIP = 11
KPT_RIGHT_HIP = 12

CACHE_DIR = Path("training_data/pose_cache")

# Lazily-loaded singleton YOLO-Pose model. Loaded once per process so the
# tracking service can reuse it across rallies without reload overhead.
_POSE_MODEL: object | None = None


def _get_pose_model() -> object:
    """Return the singleton YOLO-Pose model, loading it on first call."""
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO

        _POSE_MODEL = YOLO("yolo11s-pose.pt")
    return _POSE_MODEL


def _bbox_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def _player_pos_to_xyxy(
    pos: dict,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float]:
    """Convert PlayerPosition dict (normalized center + wh) to pixel (x1, y1, x2, y2)."""
    cx = pos["x"] * img_w
    cy = pos["y"] * img_h
    w = pos["width"] * img_w
    h = pos["height"] * img_h
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def extract_pose_for_rally(
    video_path: str,
    rally_start_ms: int,
    fps: float,
    contact_frames: list[int],
    positions_json: list[dict],
    window_half: int = 10,
    iou_threshold: float = 0.3,
    pose_model: object | None = None,
) -> dict[str, np.ndarray]:
    """Extract YOLO-Pose keypoints for contact windows and match to tracks.

    Args:
        video_path: Path to the video file.
        rally_start_ms: Rally start time in milliseconds.
        fps: Video FPS.
        contact_frames: Rally-relative frame numbers of contacts.
        positions_json: Player tracking positions for the rally.
        window_half: Half-window size around each contact (default ±10).
        iou_threshold: Minimum IoU to match pose detection to player track.
        pose_model: Pre-loaded YOLO pose model. If None, loads yolo11s-pose.

    Returns:
        Dict with keys 'frames', 'track_ids', 'keypoints', 'bboxes'
        suitable for np.savez_compressed.
    """
    if pose_model is None:
        from ultralytics import YOLO
        pose_model = YOLO("yolo11s-pose.pt")

    # Determine which frames we need
    needed_frames: set[int] = set()
    for cf in contact_frames:
        for offset in range(-window_half, window_half + 1):
            f = cf + offset
            if f >= 0:
                needed_frames.add(f)

    if not needed_frames:
        return _empty_result()

    # Build player position lookup: frame -> list of (track_id, xyxy_norm)
    player_by_frame: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for pp in positions_json:
        fn = pp["frameNumber"]
        if fn in needed_frames:
            tid = pp["trackId"]
            if tid < 0:
                continue
            # Store normalized xyxy
            cx, cy = pp["x"], pp["y"]
            w, h = pp["width"], pp["height"]
            xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            player_by_frame.setdefault(fn, []).append((tid, xyxy))

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return _empty_result()

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Compute absolute frame offset
    abs_offset = int(rally_start_ms / 1000 * fps)

    # Sort needed frames for sequential reading
    sorted_frames = sorted(needed_frames)

    all_frames: list[int] = []
    all_track_ids: list[int] = []
    all_keypoints: list[np.ndarray] = []  # each (17, 3)
    all_bboxes: list[np.ndarray] = []  # each (4,)

    # Seek to first needed frame
    first_abs = abs_offset + sorted_frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_abs)

    current_abs = first_abs

    for rally_frame in sorted_frames:
        target_abs = abs_offset + rally_frame

        # Skip frames if needed
        while current_abs < target_abs:
            cap.grab()
            current_abs += 1

        ret, frame = cap.read()
        if not ret:
            current_abs += 1
            continue
        current_abs += 1

        # Run YOLO-Pose on full frame
        results = pose_model.predict(frame, verbose=False, imgsz=1280)  # type: ignore[attr-defined]
        if not results or len(results) == 0:
            continue

        result = results[0]
        if result.keypoints is None or result.boxes is None:
            continue

        # Get pose detections
        kps_all = result.keypoints.data.cpu().numpy()  # (N_det, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()  # (N_det, 4) pixel coords

        if len(kps_all) == 0:
            continue

        # Normalize boxes to 0-1
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= img_w
        boxes_norm[:, [1, 3]] /= img_h

        # Match each pose detection to player tracks via IoU
        players = player_by_frame.get(rally_frame, [])
        if not players:
            continue

        for det_idx in range(len(kps_all)):
            det_box = (
                boxes_norm[det_idx, 0],
                boxes_norm[det_idx, 1],
                boxes_norm[det_idx, 2],
                boxes_norm[det_idx, 3],
            )

            best_iou = 0.0
            best_tid = -1

            for tid, track_box in players:
                iou = _bbox_iou(det_box, track_box)
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_iou >= iou_threshold and best_tid >= 0:
                # Normalize keypoints to 0-1
                kps = kps_all[det_idx].copy()  # (17, 3)
                kps[:, 0] /= img_w
                kps[:, 1] /= img_h

                all_frames.append(rally_frame)
                all_track_ids.append(best_tid)
                all_keypoints.append(kps.astype(np.float32))
                all_bboxes.append(boxes_norm[det_idx].astype(np.float32))

    cap.release()

    if not all_frames:
        return _empty_result()

    return {
        "frames": np.array(all_frames, dtype=np.int32),
        "track_ids": np.array(all_track_ids, dtype=np.int32),
        "keypoints": np.stack(all_keypoints),  # (N, 17, 3)
        "bboxes": np.stack(all_bboxes),  # (N, 4)
    }


def _empty_result() -> dict[str, np.ndarray]:
    return {
        "frames": np.array([], dtype=np.int32),
        "track_ids": np.array([], dtype=np.int32),
        "keypoints": np.zeros((0, 17, 3), dtype=np.float32),
        "bboxes": np.zeros((0, 4), dtype=np.float32),
    }


def save_pose_cache(
    rally_id: str,
    data: dict[str, np.ndarray],
    cache_dir: Path = CACHE_DIR,
) -> Path:
    """Save pose data to compressed numpy archive."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{rally_id}.npz"
    np.savez_compressed(path, **data)  # type: ignore[arg-type]
    return path


def load_pose_cache(
    rally_id: str,
    cache_dir: Path = CACHE_DIR,
) -> dict[str, np.ndarray] | None:
    """Load cached pose data for a rally. Returns None if not cached."""
    path = cache_dir / f"{rally_id}.npz"
    if not path.exists():
        return None
    data = np.load(path)
    return {k: data[k] for k in data.files}


def enrich_positions_with_pose(
    player_positions: list,  # list[PlayerPosition] — typed as list to avoid circular import
    video_path: str,
    contact_frames: list[int],
    rally_start_ms: int = 0,
    window_half: int = 10,
    iou_threshold: float = 0.3,
    batch_size: int = 8,
    imgsz: int = 1280,
    pose_model: object | None = None,
) -> int:
    """Run YOLO-Pose around contact frames and inject keypoints into PlayerPositions.

    Production-grade pose enrichment for the tracking service. Reads only frames
    within ±window_half of each contact, runs batched YOLO-Pose inference, matches
    detections to existing tracks via bbox IoU, and mutates `PlayerPosition.keypoints`
    in place. The returned count is the number of (frame, track) tuples enriched.

    Args:
        player_positions: PlayerPosition objects from player tracking.
            Mutated in place: `.keypoints` is set on matched (frame, track) pairs.
        video_path: Path to the video file the positions were tracked from.
        contact_frames: Rally-relative frame numbers of contact candidates.
        rally_start_ms: Start time of the rally segment in the video (0 if the
            video is already a per-rally segment).
        window_half: Half-window of frames to enrich around each contact.
        iou_threshold: Minimum bbox IoU to match a pose detection to a track.
        batch_size: YOLO-Pose batch size for GPU inference.
        imgsz: Inference resolution.
        pose_model: Optional pre-loaded model. Defaults to the singleton.

    Returns:
        Number of (frame, track_id) entries enriched with keypoints.
    """
    if not contact_frames or not player_positions:
        return 0

    if pose_model is None:
        pose_model = _get_pose_model()

    # Determine which rally-relative frames we need
    needed: set[int] = set()
    for cf in contact_frames:
        for offset in range(-window_half, window_half + 1):
            f = cf + offset
            if f >= 0:
                needed.add(f)
    if not needed:
        return 0

    # Build (frame -> list of PlayerPosition) lookup, restricted to needed frames
    pos_by_frame: dict[int, list] = {}
    for pp in player_positions:
        if pp.frame_number in needed and pp.track_id >= 0:
            pos_by_frame.setdefault(pp.frame_number, []).append(pp)

    if not pos_by_frame:
        return 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("enrich_positions_with_pose: cannot open %s", video_path)
        return 0

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    abs_offset = int(rally_start_ms / 1000 * fps)

    sorted_frames = sorted(pos_by_frame.keys())
    first_abs = abs_offset + sorted_frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_abs)
    current_abs = first_abs

    n_enriched = 0
    batch: list[tuple[int, np.ndarray]] = []

    def _flush_batch() -> int:
        if not batch:
            return 0
        frames_imgs = [f for _, f in batch]
        rally_frames_in_batch = [rf for rf, _ in batch]
        try:
            results = pose_model.predict(  # type: ignore[attr-defined]
                frames_imgs, verbose=False, imgsz=imgsz,
            )
        except Exception as exc:
            logger.warning("YOLO-Pose batch failed: %s", exc)
            return 0
        enriched = 0
        for result, rally_frame in zip(results, rally_frames_in_batch):
            if result.keypoints is None or result.boxes is None:
                continue
            kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
            boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) pixel
            if len(kps_all) == 0:
                continue
            boxes_norm = boxes.copy()
            boxes_norm[:, [0, 2]] /= img_w
            boxes_norm[:, [1, 3]] /= img_h

            for pp in pos_by_frame.get(rally_frame, []):
                cx, cy = pp.x, pp.y
                w, h = pp.width, pp.height
                pp_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                best_iou = 0.0
                best_idx = -1
                for det_idx in range(len(boxes_norm)):
                    iou = _bbox_iou(pp_box, tuple(boxes_norm[det_idx]))
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = det_idx
                if best_iou >= iou_threshold and best_idx >= 0:
                    kps = kps_all[best_idx].copy()
                    kps[:, 0] /= img_w
                    kps[:, 1] /= img_h
                    pp.keypoints = kps.tolist()
                    enriched += 1
        return enriched

    for rally_frame in sorted_frames:
        target_abs = abs_offset + rally_frame
        while current_abs < target_abs:
            cap.grab()
            current_abs += 1
        ret, frame = cap.read()
        if not ret:
            current_abs += 1
            continue
        current_abs += 1
        batch.append((rally_frame, frame))
        if len(batch) >= batch_size:
            n_enriched += _flush_batch()
            batch = []

    n_enriched += _flush_batch()
    cap.release()
    return n_enriched
