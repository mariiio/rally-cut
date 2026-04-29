"""Pose-keypoint-anchored appearance feature extraction.

Workstream 1 + 2 (shipped 2026-04-29). Replaces the production rule-based
clothing mask (skin+sand HSV exclusion + central-50% spatial filter) with
YOLO-Pose-derived body-region polygons:
    - Upper band: torso quadrilateral {LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_HIP, LEFT_HIP}.
    - Lower band: per-leg rectangles along LEFT_HIP→LEFT_KNEE / RIGHT_HIP→RIGHT_KNEE,
                  ORed (avoids the inter-leg sand gap that a hip-knee quadrilateral
                  scoops up in athletic stance).

Also computes body-proportion ratios (Workstream 2) from the same pose
keypoints. Per-track median ratios are stored on `TrackAppearanceStats` and
blended into `compute_track_similarity` cost via env BODY_PROP_ALPHA (default 0.3).

Activation:
    USE_POSE_ANCHORED_FEATURES=1 — caller (extract_rally_appearances) routes
    feature extraction through this module instead of the legacy whole-bbox path.
    Default: legacy behavior unchanged.

7-rally panel verdict (jojo r10 + 2d105b7b r06 + jojo r07/r08 + 90266c1d r05/r06
+ wawa r09): 4/7 within−cross-margin PASSes vs 3/7 under production. The 3
remaining failures (jojo r07/r08, wawa r09) are upstream within-rally swap bugs
(BoT-SORT track containing frames from two physical players) and are not
solvable at the cross-rally feature level — they require within-rally repair.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rallycut.tracking.player_features import (
    PlayerAppearanceFeatures,
    TrackAppearanceStats,
    _build_clothing_mask,
    _extract_dominant_color,
    _extract_hs_histogram,
    _extract_skin_tone,
    _extract_v_histogram,
)
from rallycut.tracking.pose_attribution.pose_cache import (
    KPT_LEFT_HIP,
    KPT_LEFT_SHOULDER,
    KPT_RIGHT_HIP,
    KPT_RIGHT_SHOULDER,
    _bbox_iou,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# COCO-17 indices not exported by pose_cache.
KPT_NOSE = 0
KPT_LEFT_KNEE = 13
KPT_RIGHT_KNEE = 14
KPT_LEFT_ANKLE = 15
KPT_RIGHT_ANKLE = 16

# Pose-IoU threshold for matching pose detection → track bbox. Matches
# `pose_cache.extract_pose_for_rally` defaults.
POSE_IOU_THRESHOLD = 0.3
KPT_VIS_THRESHOLD = 0.3
# YOLO-pose inference resolution. 1280 matches production
# `enrich_positions_with_pose` default and is necessary at 1920x1080 to
# pick up far-court players (960 misses ~25%; 640 misses ~50%).
POSE_IMGSZ = 1280

# Per-leg thigh rectangle geometry (locked 2026-04-29 after probe).
LEG_RECTANGLE_HALF_WIDTH_FRAC = 0.25
LEG_RECTANGLE_HIP_KNEE_EXTENT = 1.0  # full hip-knee; extent=0.6 was probed and reverted

# Polygon-area floor below which the variant falls back to whole-bbox.
MIN_POLYGON_AREA_PX = 80


# -----------------------------------------------------------------------------
# Activation gate
# -----------------------------------------------------------------------------


def is_pose_anchored_enabled() -> bool:
    """Production caller checks this before routing through the pose path."""
    return os.environ.get("USE_POSE_ANCHORED_FEATURES", "0") == "1"


# -----------------------------------------------------------------------------
# YOLO-Pose model singleton
# -----------------------------------------------------------------------------


_POSE_MODEL: object | None = None


def _resolve_pose_weights_path() -> Path:
    """Find yolo11s-pose.pt. Same convention as pose_cache._get_pose_model."""
    here = Path(__file__).resolve()
    # walk up to <analysis>/ directory (contains rallycut/ + scripts/ + the .pt)
    for parent in [here.parent, *here.parents]:
        candidate = parent / "yolo11s-pose.pt"
        if candidate.exists():
            return candidate
    # ultralytics will download if path is bare filename — fall back.
    return Path("yolo11s-pose.pt")


def get_pose_model() -> object:
    """Singleton YOLO-Pose model. Lazy-loaded once per process."""
    global _POSE_MODEL
    if _POSE_MODEL is None:
        from ultralytics import YOLO
        weights_path = _resolve_pose_weights_path()
        _POSE_MODEL = YOLO(str(weights_path))
        logger.info("loaded yolo11s-pose from %s", weights_path)
    return _POSE_MODEL


# -----------------------------------------------------------------------------
# Per-frame pose inference
# -----------------------------------------------------------------------------


@dataclass
class FramePoseResult:
    """Pose detections matched to track bboxes for one frame."""

    by_track: dict[int, tuple[np.ndarray, tuple[float, float, float, float]]] = (
        field(default_factory=dict)
    )


def run_pose_on_frame(
    frame_bgr: np.ndarray,
    bboxes_norm_by_tid: dict[int, tuple[float, float, float, float]],
    pose_model: object | None = None,
) -> FramePoseResult:
    """Run YOLO-Pose once on a full frame, IoU-match each detection to a track.

    Args:
        frame_bgr: BGR frame.
        bboxes_norm_by_tid: track_id → (x1, y1, x2, y2) normalized 0-1.
        pose_model: optional pre-loaded YOLO model; defaults to the singleton.

    Returns:
        FramePoseResult with `by_track[tid] = (kps_pixel, det_box_norm)` for
        each track that got an IoU-matched detection.
    """
    if pose_model is None:
        pose_model = get_pose_model()
    h, w = frame_bgr.shape[:2]
    try:
        results = pose_model.predict(  # type: ignore[attr-defined]
            frame_bgr, verbose=False, imgsz=POSE_IMGSZ,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("YOLO-Pose predict failed: %s", exc)
        return FramePoseResult()
    if not results:
        return FramePoseResult()
    result = results[0]
    if result.keypoints is None or result.boxes is None:
        return FramePoseResult()

    kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3) pixel
    boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4) pixel xyxy
    if len(kps_all) == 0:
        return FramePoseResult()

    boxes_norm = boxes.copy()
    boxes_norm[:, [0, 2]] /= w
    boxes_norm[:, [1, 3]] /= h

    out: dict[int, tuple[np.ndarray, tuple[float, float, float, float]]] = {}
    for tid, track_xyxy in bboxes_norm_by_tid.items():
        best_iou = 0.0
        best_idx = -1
        for i in range(len(boxes_norm)):
            det_box = (
                float(boxes_norm[i, 0]), float(boxes_norm[i, 1]),
                float(boxes_norm[i, 2]), float(boxes_norm[i, 3]),
            )
            iou = _bbox_iou(track_xyxy, det_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= POSE_IOU_THRESHOLD and best_idx >= 0:
            out[tid] = (
                kps_all[best_idx].astype(np.float32),
                (
                    float(boxes_norm[best_idx, 0]),
                    float(boxes_norm[best_idx, 1]),
                    float(boxes_norm[best_idx, 2]),
                    float(boxes_norm[best_idx, 3]),
                ),
            )
    return FramePoseResult(by_track=out)


# -----------------------------------------------------------------------------
# Polygon-mask construction
# -----------------------------------------------------------------------------


def _frame_kps_to_crop_kps(
    kps_pixel: np.ndarray, bbox_pixel: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, _, _ = bbox_pixel
    out = kps_pixel.copy()
    out[:, 0] -= x1
    out[:, 1] -= y1
    return out


def _polygon_mask(
    crop_shape_hw: tuple[int, int],
    kps_crop: np.ndarray,
    indices: list[int],
) -> tuple[np.ndarray, int]:
    h, w = crop_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts: list[tuple[int, int]] = []
    for i in indices:
        kx, ky, kv = kps_crop[i]
        if kv < KPT_VIS_THRESHOLD:
            return mask, 0
        ix = int(round(max(0.0, min(float(w - 1), kx))))
        iy = int(round(max(0.0, min(float(h - 1), ky))))
        pts.append((ix, iy))
    if len(pts) < 3:
        return mask, 0
    poly = np.array(pts, dtype=np.int32)
    cv2.fillConvexPoly(mask, poly, (255,))
    return mask, int((mask > 0).sum())


def _leg_rectangle_mask(
    crop_shape_hw: tuple[int, int],
    kps_crop: np.ndarray,
    hip_idx: int,
    knee_idx: int,
    half_width_frac: float = LEG_RECTANGLE_HALF_WIDTH_FRAC,
    hip_knee_extent: float = LEG_RECTANGLE_HIP_KNEE_EXTENT,
) -> tuple[np.ndarray, int]:
    """Rectangle aligned to hip→(extent·knee) axis for one leg."""
    h, w = crop_shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if (kps_crop[hip_idx, 2] < KPT_VIS_THRESHOLD or
            kps_crop[knee_idx, 2] < KPT_VIS_THRESHOLD):
        return mask, 0
    if (kps_crop[KPT_LEFT_HIP, 2] >= KPT_VIS_THRESHOLD and
            kps_crop[KPT_RIGHT_HIP, 2] >= KPT_VIS_THRESHOLD):
        hip_width = float(np.linalg.norm(
            kps_crop[KPT_LEFT_HIP, :2] - kps_crop[KPT_RIGHT_HIP, :2],
        ))
    else:
        hip_width = w * 0.2
    half_w = max(2.0, hip_width * half_width_frac)

    hip = kps_crop[hip_idx, :2].astype(np.float32)
    knee = kps_crop[knee_idx, :2].astype(np.float32)
    leg_vec = knee - hip
    leg_len = float(np.linalg.norm(leg_vec))
    if leg_len < 1.0:
        return mask, 0
    end = hip + leg_vec * float(hip_knee_extent)
    perp = np.array([-leg_vec[1], leg_vec[0]], dtype=np.float32) / leg_len

    p1 = hip - perp * half_w
    p2 = hip + perp * half_w
    p3 = end + perp * half_w
    p4 = end - perp * half_w
    poly = np.array([p1, p2, p3, p4], dtype=np.float32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    cv2.fillConvexPoly(mask, poly.astype(np.int32), (255,))
    return mask, int((mask > 0).sum())


def _build_pose_region_masks(
    crop_shape_hw: tuple[int, int],
    kps_crop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Returns (upper_torso_mask, lower_thighs_mask, upper_area_px, lower_area_px)."""
    upper_mask, upper_area = _polygon_mask(
        crop_shape_hw, kps_crop,
        [KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER, KPT_RIGHT_HIP, KPT_LEFT_HIP],
    )
    left_thigh, _ = _leg_rectangle_mask(
        crop_shape_hw, kps_crop, KPT_LEFT_HIP, KPT_LEFT_KNEE,
    )
    right_thigh, _ = _leg_rectangle_mask(
        crop_shape_hw, kps_crop, KPT_RIGHT_HIP, KPT_RIGHT_KNEE,
    )
    lower_mask = np.asarray(cv2.bitwise_or(left_thigh, right_thigh), dtype=np.uint8)
    return upper_mask, lower_mask, upper_area, int((lower_mask > 0).sum())


# -----------------------------------------------------------------------------
# Body-proportion ratios (Workstream 2)
# -----------------------------------------------------------------------------


def _kp(kps_crop: np.ndarray, idx: int) -> tuple[float, float] | None:
    x, y, c = kps_crop[idx]
    if c < KPT_VIS_THRESHOLD:
        return None
    return float(x), float(y)


def _euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def compute_body_proportions(kps_crop: np.ndarray) -> dict[str, float] | None:
    """Scale-invariant body-proportion ratios from one frame's pose."""
    ls = _kp(kps_crop, KPT_LEFT_SHOULDER)
    rs = _kp(kps_crop, KPT_RIGHT_SHOULDER)
    lh = _kp(kps_crop, KPT_LEFT_HIP)
    rh = _kp(kps_crop, KPT_RIGHT_HIP)
    if ls is None or rs is None or lh is None or rh is None:
        return None
    shoulder_width = _euclid(ls, rs)
    hip_width = _euclid(lh, rh)
    if shoulder_width < 2.0 or hip_width < 2.0:
        return None
    shoulder_mid = _midpoint(ls, rs)
    hip_mid = _midpoint(lh, rh)
    torso_length = _euclid(shoulder_mid, hip_mid)
    if torso_length < 2.0:
        return None

    out: dict[str, float] = {"shoulder_to_hip": shoulder_width / hip_width}
    lk = _kp(kps_crop, KPT_LEFT_KNEE)
    rk = _kp(kps_crop, KPT_RIGHT_KNEE)
    if lk is not None and rk is not None:
        knee_mid = _midpoint(lk, rk)
        thigh_len = _euclid(hip_mid, knee_mid)
        if thigh_len > 2.0:
            out["torso_to_thigh"] = torso_length / thigh_len
    la = _kp(kps_crop, KPT_LEFT_ANKLE)
    ra = _kp(kps_crop, KPT_RIGHT_ANKLE)
    if la is not None and ra is not None:
        ankle_mid = _midpoint(la, ra)
        leg_full = _euclid(hip_mid, ankle_mid)
        if leg_full > 2.0:
            out["torso_to_full_leg"] = torso_length / leg_full
            out["shoulder_to_height"] = shoulder_width / (torso_length + leg_full)
    nose = _kp(kps_crop, KPT_NOSE)
    if nose is not None:
        nose_to_shoulder = _euclid(nose, shoulder_mid)
        if nose_to_shoulder > 2.0:
            out["head_to_shoulder"] = nose_to_shoulder / shoulder_width
    return out


def aggregate_body_proportions(
    per_frame: list[dict[str, float] | None],
    min_frames_per_ratio: int = 3,
) -> dict[str, float]:
    """Per-ratio median across pose-available frames; drops sparse ratios."""
    by_key: dict[str, list[float]] = {}
    for d in per_frame:
        if d is None:
            continue
        for k, v in d.items():
            by_key.setdefault(k, []).append(v)
    return {
        k: float(np.median(vs))
        for k, vs in by_key.items()
        if len(vs) >= min_frames_per_ratio
    }


# -----------------------------------------------------------------------------
# Per-frame extraction (production-style)
# -----------------------------------------------------------------------------


def extract_pose_anchored_features(
    frame: NDArray[np.uint8],
    track_id: int,
    frame_number: int,
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    pose_data: tuple[np.ndarray, tuple[float, float, float, float]] | None,
) -> tuple[PlayerAppearanceFeatures, dict[str, float] | None]:
    """Drop-in pose-anchored alternative to extract_appearance_features.

    Returns (appearance_features, body_props_dict_or_none). When pose data is
    unavailable for this (frame, track), falls back to the legacy clothing_mask
    + vertical-band masks so the per-frame feature is never None for any track.

    Args:
        frame: BGR full-frame image.
        track_id, frame_number: identifiers (preserved on the returned features).
        bbox: normalized (cx, cy, w, h).
        frame_width, frame_height: full-frame dimensions in pixels.
        pose_data: (kps_pixel, det_box_norm) from FramePoseResult.by_track,
            or None if no IoU-matched pose detection for this track.

    Returns:
        (PlayerAppearanceFeatures populated as in production, body_proportions
        dict if pose data was usable, else None).
    """
    cx, cy, w, h = bbox
    x1 = max(0, int((cx - w / 2) * frame_width))
    y1 = max(0, int((cy - h / 2) * frame_height))
    x2 = min(frame_width, int((cx + w / 2) * frame_width))
    y2 = min(frame_height, int((cy + h / 2) * frame_height))

    features = PlayerAppearanceFeatures(
        track_id=track_id, frame_number=frame_number,
    )
    if x2 - x1 < 8 or y2 - y1 < 16:
        return features, None

    crop_bgr = np.asarray(frame[y1:y2, x1:x2], dtype=np.uint8)
    if crop_bgr.size == 0:
        return features, None

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    bh, _bw = hsv.shape[:2]

    # Skin tone (top 40%) — production rule, unchanged.
    upper_for_skin = hsv[: int(bh * 0.4), :]
    skin_tone, skin_count = _extract_skin_tone(upper_for_skin)
    features.skin_tone_hsv = skin_tone
    features.skin_pixel_count = skin_count

    # Head hist (top 15%, no mask) — production rule, unchanged.
    head_height = int(bh * 0.15)
    if head_height >= 5:
        features.head_hist = _extract_hs_histogram(hsv[:head_height, :])

    # Body region masks. Try pose polygons first; fall back to legacy
    # vertical-band slice + clothing_mask if pose unavailable / polygon too small.
    body_props: dict[str, float] | None = None
    use_pose = False
    upper_mask: np.ndarray | None = None
    lower_mask: np.ndarray | None = None

    if pose_data is not None:
        kps_pixel, _det_box = pose_data
        kps_crop = _frame_kps_to_crop_kps(kps_pixel, (x1, y1, x2, y2))
        u_mask, l_mask, u_area, l_area = _build_pose_region_masks(
            (bh, hsv.shape[1]), kps_crop,
        )
        if u_area >= MIN_POLYGON_AREA_PX or l_area >= MIN_POLYGON_AREA_PX:
            upper_mask = u_mask
            lower_mask = l_mask
            use_pose = True
        body_props = compute_body_proportions(kps_crop)

    if not use_pose:
        # Legacy fallback: production clothing_mask + vertical bands.
        clothing = _build_clothing_mask(hsv)
        upper_top = int(bh * 0.20)
        upper_bot = int(bh * 0.55)
        lower_top = int(bh * 0.50)
        lower_bot = int(bh * 0.78)
        upper_band = np.zeros_like(clothing)
        upper_band[upper_top:upper_bot, :] = 255
        lower_band = np.zeros_like(clothing)
        lower_band[lower_top:lower_bot, :] = 255
        upper_mask = np.asarray(
            cv2.bitwise_and(clothing, upper_band), dtype=np.uint8,
        )
        lower_mask = np.asarray(
            cv2.bitwise_and(clothing, lower_band), dtype=np.uint8,
        )

    features.upper_body_hist = _extract_hs_histogram(hsv, upper_mask)
    features.upper_body_v_hist = _extract_v_histogram(hsv, upper_mask)
    features.lower_body_hist = _extract_hs_histogram(hsv, lower_mask)
    features.lower_body_v_hist = _extract_v_histogram(hsv, lower_mask)
    features.dominant_color_hsv = _extract_dominant_color(hsv, lower_mask)

    return features, body_props


def populate_track_body_proportions(
    stats: TrackAppearanceStats,
    per_frame_props: list[dict[str, float] | None],
) -> None:
    """Set stats.body_proportions to per-track median across pose-available frames."""
    agg = aggregate_body_proportions(per_frame_props)
    if agg:
        stats.body_proportions = agg
