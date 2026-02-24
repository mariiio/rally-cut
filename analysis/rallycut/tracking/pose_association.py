"""Pose-based keypoint association for BoT-SORT.

Adds lower-body keypoint distance as a penalty to BoT-SORT's cost matrix when
bounding boxes overlap. During net interactions, players' bboxes merge into
similar regions making IoU ambiguous. Lower-body keypoints (hips, knees, ankles)
remain spatially separated because players are on opposite sides of the net.

Uses a dual-model approach: yolo11s handles detection+tracking (preserving
detection quality), while yolo11s-pose runs as a secondary model ONLY on
frames where bboxes overlap (minimizing performance cost).

Uses the same monkey-patch pattern as team_aware_tracker.py — wrapping
BoT-SORT's get_dists() to add an additive penalty. Stacks with team-aware
penalty (addition is commutative).

Data flow per frame:
1. yolo11s runs detection+tracking as normal
2. Check if any tracked bboxes overlap (IoU > threshold)
3. If overlap: run yolo11s-pose on the same frame, match keypoints to
   tracked boxes by IoU, update per-track EMA centroids AND store
   per-detection keypoints for the get_dists penalty
4. Patched get_dists() uses stored centroids for next frame's matching
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# COCO pose keypoint indices for lower body (hips, knees, ankles)
LOWER_BODY_INDICES = [11, 12, 13, 14, 15, 16]

# Pose model name (matches YOLO_MODELS entry in player_tracker.py)
POSE_MODEL_NAME = "yolo11s-pose.pt"


@dataclass
class PoseAssociationConfig:
    """Configuration for pose-based keypoint association in BoT-SORT."""

    enabled: bool = False
    kp_weight: float = 0.15  # Max penalty added to cost matrix
    overlap_iou_thresh: float = 0.10  # Only apply when bboxes overlap
    min_visible_kps: int = 2  # Min lower-body keypoints with conf > threshold
    kp_conf_thresh: float = 0.3  # Keypoint confidence gate
    ema_alpha: float = 0.7  # Smoothing for per-track centroid (higher = more recent)
    distance_scale: float = 3.0  # Scale factor: penalty = min(dist * scale, kp_weight)


def lower_body_centroid(
    kpt_xy: np.ndarray,
    kpt_conf: np.ndarray,
    min_visible: int = 2,
    conf_thresh: float = 0.3,
) -> np.ndarray | None:
    """Compute centroid of visible lower-body keypoints.

    Args:
        kpt_xy: (17, 2) keypoint coordinates (normalized 0-1).
        kpt_conf: (17,) confidence scores.
        min_visible: Minimum visible keypoints required.
        conf_thresh: Minimum confidence for a keypoint to be visible.

    Returns:
        (2,) centroid array [x, y] or None if insufficient visible keypoints.
    """
    visible_pts = []
    for idx in LOWER_BODY_INDICES:
        if kpt_conf[idx] >= conf_thresh:
            visible_pts.append(kpt_xy[idx])
    if len(visible_pts) < min_visible:
        return None
    centroid: np.ndarray = np.mean(visible_pts, axis=0).astype(np.float64)
    return centroid


def _xyxy_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute IoU between two xyxy bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def has_overlapping_bboxes(
    positions: list[Any],
    iou_thresh: float = 0.10,
) -> bool:
    """Check if any pair of tracked positions have overlapping bboxes.

    Args:
        positions: List of PlayerPosition objects with x, y, width, height.
        iou_thresh: IoU threshold for overlap detection.

    Returns:
        True if any pair has IoU > threshold.
    """
    n = len(positions)
    if n < 2:
        return False

    # Convert to xyxy for IoU computation (normalized coords)
    boxes = []
    for p in positions:
        if p.track_id < 0:
            continue
        x1 = p.x - p.width / 2
        y1 = p.y - p.height / 2
        x2 = p.x + p.width / 2
        y2 = p.y + p.height / 2
        boxes.append(np.array([x1, y1, x2, y2]))

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _xyxy_iou(boxes[i], boxes[j]) > iou_thresh:
                return True
    return False


class PoseKeypointTracker:
    """Tracks per-player lower-body keypoint centroids for association penalty.

    Maintains EMA-smoothed centroids per track_id. Uses a secondary pose model
    that runs ONLY on frames with overlapping bboxes.
    """

    def __init__(self, config: PoseAssociationConfig | None = None) -> None:
        self.config = config or PoseAssociationConfig(enabled=True)
        # EMA-smoothed lower-body centroid per track_id: (2,) normalized [x, y]
        self._track_centroids: dict[int, np.ndarray] = {}
        # Detection keypoints for current frame (set before get_dists runs)
        self._detection_keypoints: list[np.ndarray | None] | None = None
        # Secondary pose model (lazy-loaded on first overlap)
        self._pose_model: Any = None

    @property
    def track_centroids(self) -> dict[int, np.ndarray]:
        """Current per-track centroids (read-only copy)."""
        return dict(self._track_centroids)

    def clear_detection_keypoints(self) -> None:
        """Clear stored detection keypoints to prevent stale data."""
        self._detection_keypoints = None

    def _load_pose_model(self) -> Any:
        """Lazy-load the secondary pose model."""
        if self._pose_model is not None:
            return self._pose_model

        from ultralytics import YOLO

        logger.info("Loading secondary pose model: %s", POSE_MODEL_NAME)
        self._pose_model = YOLO(POSE_MODEL_NAME)
        return self._pose_model

    def run_pose_on_frame(
        self,
        frame: np.ndarray,
        positions: list[Any],
        video_width: int,
        video_height: int,
    ) -> None:
        """Run secondary pose model, update centroids AND store keypoints.

        Single pose model call that does both:
        1. Updates per-track EMA centroids (for penalty on future frames)
        2. Stores per-detection keypoints (for penalty on current frame's
           get_dists, which runs on the NEXT model.track() call)

        Only call on frames where has_overlapping_bboxes() is True.

        Args:
            frame: BGR image (original, not ROI-masked).
            positions: Tracked PlayerPosition objects for this frame.
            video_width: Video frame width in pixels.
            video_height: Video frame height in pixels.
        """
        pose_model = self._load_pose_model()

        # Run pose model (detection only, no tracking)
        pose_results = pose_model(
            frame,
            conf=0.15,
            imgsz=1280,
            classes=[0],  # person only
            verbose=False,
        )

        result = pose_results[0]
        if (
            result.keypoints is None
            or result.keypoints.xyn is None
            or result.boxes is None
        ):
            self._detection_keypoints = None
            return

        pose_boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (M, 4) pixel coords
        kpts_xyn = result.keypoints.xyn.cpu().numpy()  # (M, 17, 2)
        kpts_conf = result.keypoints.conf.cpu().numpy()  # (M, 17)

        # Match each tracked position to best-overlapping pose detection
        # and build both centroid updates and detection keypoint list
        centroids: list[np.ndarray | None] = []
        for p in positions:
            if p.track_id < 0:
                centroids.append(None)
                continue

            # Convert tracked bbox to pixel xyxy for matching
            track_xyxy = np.array([
                (p.x - p.width / 2) * video_width,
                (p.y - p.height / 2) * video_height,
                (p.x + p.width / 2) * video_width,
                (p.y + p.height / 2) * video_height,
            ])

            # Find best-matching pose detection
            best_iou = 0.3  # minimum IoU to accept a match
            best_idx = -1
            for k in range(len(pose_boxes_xyxy)):
                iou = _xyxy_iou(track_xyxy, pose_boxes_xyxy[k])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = k

            if best_idx < 0:
                centroids.append(None)
                continue

            centroid = lower_body_centroid(
                kpts_xyn[best_idx],
                kpts_conf[best_idx],
                min_visible=self.config.min_visible_kps,
                conf_thresh=self.config.kp_conf_thresh,
            )
            centroids.append(centroid)

            if centroid is None:
                continue

            # EMA update for track centroid
            alpha = self.config.ema_alpha
            tid = p.track_id
            if tid in self._track_centroids:
                self._track_centroids[tid] = (
                    alpha * centroid + (1 - alpha) * self._track_centroids[tid]
                )
            else:
                self._track_centroids[tid] = centroid

        # Store for get_dists penalty on next tracking frame
        self._detection_keypoints = centroids

    def compute_penalty_matrix(
        self,
        tracks: list[Any],
        detections: list[Any],
        img_height: int,
    ) -> np.ndarray:
        """Compute keypoint distance penalty for overlapping (track, det) pairs.

        For each pair with bbox IoU > overlap_iou_thresh:
        - Looks up track's stored centroid and detection's keypoint centroid
        - Computes euclidean distance in normalized coordinates
        - Penalty = min(distance * distance_scale, kp_weight)

        Large keypoint distance -> large penalty (detection is far from where
        the track's lower body should be).

        Args:
            tracks: STrack objects from BoT-SORT (have .tlwh, .track_id).
            detections: STrack objects (detections, have .tlwh, .idx).
            img_height: Image height in pixels.

        Returns:
            (n_tracks, n_dets) penalty matrix. Zero where no penalty applies.
        """
        n_tracks = len(tracks)
        n_dets = len(detections)
        penalty = np.zeros((n_tracks, n_dets), dtype=np.float64)

        if n_tracks == 0 or n_dets == 0:
            return penalty

        det_keypoints = self._detection_keypoints
        if det_keypoints is None:
            return penalty

        for i, track in enumerate(tracks):
            tid = track.track_id
            if tid not in self._track_centroids:
                continue
            track_centroid = self._track_centroids[tid]

            # Track predicted bbox in tlwh (top-left x, top-left y, w, h)
            track_tlwh = track.tlwh

            for j, det in enumerate(detections):
                # Get detection keypoint centroid
                det_idx = int(det.idx) if hasattr(det, "idx") else j
                if det_idx >= len(det_keypoints) or det_keypoints[det_idx] is None:
                    continue
                det_centroid = det_keypoints[det_idx]

                # Compute bbox IoU to check overlap
                det_tlwh = det.tlwh
                iou = self._tlwh_iou(track_tlwh, det_tlwh)
                if iou < self.config.overlap_iou_thresh:
                    continue

                # Euclidean distance between centroids (normalized coords)
                dist = float(np.linalg.norm(track_centroid - det_centroid))

                penalty[i, j] = min(
                    dist * self.config.distance_scale,
                    self.config.kp_weight,
                )

        return penalty

    @staticmethod
    def _tlwh_iou(tlwh1: np.ndarray, tlwh2: np.ndarray) -> float:
        """Compute IoU between two tlwh bounding boxes."""
        x1 = max(tlwh1[0], tlwh2[0])
        y1 = max(tlwh1[1], tlwh2[1])
        x2 = min(tlwh1[0] + tlwh1[2], tlwh2[0] + tlwh2[2])
        y2 = min(tlwh1[1] + tlwh1[3], tlwh2[1] + tlwh2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = tlwh1[2] * tlwh1[3]
        area2 = tlwh2[2] * tlwh2[3]
        union = area1 + area2 - inter
        return float(inter / union) if union > 0 else 0.0


def patch_botsort_with_pose(
    botsort_instance: Any,
    pose_tracker: PoseKeypointTracker,
    img_height: int,
) -> None:
    """Monkey-patch BoT-SORT's get_dists to add keypoint distance penalty.

    Wraps the existing get_dists (which may already be team-aware-patched)
    with an additional keypoint penalty. Safe to stack with team_aware_tracker.

    Args:
        botsort_instance: A BOTSORT tracker instance from ultralytics.
        pose_tracker: PoseKeypointTracker with centroid state.
        img_height: Image height in pixels for normalization.
    """
    if not hasattr(botsort_instance, "get_dists"):
        logger.warning("BoT-SORT instance has no get_dists method, skipping pose patch")
        return

    # Store original (may already be team-aware-wrapped)
    current_get_dists = botsort_instance.get_dists
    # Keep a reference to restore later
    if not hasattr(botsort_instance, "_original_get_dists_pose"):
        botsort_instance._original_get_dists_pose = current_get_dists

    def patched_get_dists(
        tracks: list[Any], detections: list[Any]
    ) -> Any:
        dists = current_get_dists(tracks, detections)
        penalty = pose_tracker.compute_penalty_matrix(
            tracks, detections, img_height
        )
        return dists + penalty

    botsort_instance.get_dists = patched_get_dists
    logger.debug("Patched BoT-SORT get_dists with pose keypoint penalty")


def unpatch_botsort_pose(botsort_instance: Any) -> None:
    """Restore BoT-SORT's get_dists to the state before pose patching.

    Args:
        botsort_instance: A BOTSORT tracker instance that was patched.
    """
    if hasattr(botsort_instance, "_original_get_dists_pose"):
        botsort_instance.get_dists = botsort_instance._original_get_dists_pose
        del botsort_instance._original_get_dists_pose
        logger.debug("Unpatched BoT-SORT pose keypoint penalty")
