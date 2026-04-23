"""Automatic net-line estimator for fixed-camera beach volleyball.

Uses the 6-keypoint YOLO court model:
  - 4 court corners → camera calibration via solvePnP
  - 2 center points at net-sideline intersections → direct net-base anchors

Net-top is computed by Y-shifting the camera-projected net-top so the
camera-predicted net-base matches the observed keypoint centers. This
bypasses the focal-length-amplification-at-z=2.43 failure mode that
plagues manual-homography-only calibration.

Tier 1 optimizations (2026-04-23 scoping session):
  (1) multi-frame aggregation: weighted median over N sampled frames
  (2) per-video cache keyed on (video_id, model_mtime, n_frames)
  (3) per-sideline confidence gating: if one center has conf < threshold,
      mirror Y from the higher-confidence side using corner geometry
  (4) sanity validation: flag detections (via ``warnings`` field) where the
      net-base fraction between far/near baselines is outside
      [SANITY_BASE_LOW, SANITY_BASE_HIGH]. Caller decides whether to trust.

Entry point: ``estimate_net_line(video_path) -> NetLine | None``.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from rallycut.court.camera_model import calibrate_camera, project_3d_to_image
from rallycut.court.keypoint_detector import (
    CourtKeypointDetector,
    FrameKeypoints,
)

logger = logging.getLogger(__name__)

COURT_WIDTH_M = 8.0
COURT_LENGTH_M = 16.0
NET_Y_M = 8.0
NET_HEIGHT_M = 2.43

DEFAULT_COURT_CORNERS = [
    (0.0, 0.0),
    (COURT_WIDTH_M, 0.0),
    (COURT_WIDTH_M, COURT_LENGTH_M),
    (0.0, COURT_LENGTH_M),
]

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

CENTER_CONF_MIN = 0.30     # Per-sideline center-point confidence gate
KEYPOINT_AGG_N = 30        # Frames to aggregate per video
SANITY_BASE_LOW = 0.12     # net-base must sit ≥ this fraction above far baseline
SANITY_BASE_HIGH = 0.85    # and ≤ this fraction above near baseline

CACHE_DIR_ENV = "RALLYCUT_NET_LINE_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rallycut" / "net_line"


# --------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------


@dataclass
class NetLine:
    """Auto-detected net line in normalized image coordinates."""

    # Net BASE line endpoints at the two sidelines (z=0 ground plane).
    base_left_xy: tuple[float, float]
    base_right_xy: tuple[float, float]
    # Net TOP line endpoints (z=NET_HEIGHT_M).
    top_left_xy: tuple[float, float]
    top_right_xy: tuple[float, float]
    # Confidence: min of the two center-keypoint confidences, median-aggregated.
    confidence: float
    # Per-sideline provenance: "keypoint" | "mirrored" | "fallback".
    left_source: str
    right_source: str
    # Reproj error of the underlying camera model (pixels).
    reproj_err_px: float
    focal_px: float
    # How many frames contributed to the aggregation.
    n_frames_used: int
    # Optional warning tags (e.g. "sanity_failed", "single_frame").
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> NetLine:
        return cls(
            base_left_xy=tuple(d["base_left_xy"]),
            base_right_xy=tuple(d["base_right_xy"]),
            top_left_xy=tuple(d["top_left_xy"]),
            top_right_xy=tuple(d["top_right_xy"]),
            confidence=float(d["confidence"]),
            left_source=d["left_source"],
            right_source=d["right_source"],
            reproj_err_px=float(d["reproj_err_px"]),
            focal_px=float(d["focal_px"]),
            n_frames_used=int(d["n_frames_used"]),
            warnings=list(d.get("warnings", [])),
        )


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------


def _cache_dir() -> Path:
    d = Path(os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(video_key: str, model_path: Path, n_frames: int) -> str:
    mtime = int(model_path.stat().st_mtime) if model_path.exists() else 0
    return f"{video_key}__m{mtime}__n{n_frames}.json"


def _cache_load(video_key: str, model_path: Path, n_frames: int) -> NetLine | None:
    path = _cache_dir() / _cache_key(video_key, model_path, n_frames)
    if not path.exists():
        return None
    try:
        return NetLine.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError):
        return None


def _cache_store(video_key: str, model_path: Path, n_frames: int, nl: NetLine) -> None:
    path = _cache_dir() / _cache_key(video_key, model_path, n_frames)
    path.write_text(json.dumps(nl.to_dict(), indent=2))


# --------------------------------------------------------------------------
# Multi-frame aggregation
# --------------------------------------------------------------------------


def _weighted_median_xy(
    xs: np.ndarray, ys: np.ndarray, weights: np.ndarray,
) -> tuple[float, float]:
    """Weighted median of (x, y) points — returned per-axis independently."""
    if len(xs) == 0:
        return 0.0, 0.0
    total = float(weights.sum())
    if total <= 0:
        return float(np.median(xs)), float(np.median(ys))

    def _wmed(vals: np.ndarray) -> float:
        order = np.argsort(vals)
        cum = np.cumsum(weights[order])
        idx = int(np.searchsorted(cum, total / 2.0))
        idx = min(idx, len(vals) - 1)
        return float(vals[order][idx])

    return _wmed(xs), _wmed(ys)


def _aggregate_keypoints(frame_results: list[FrameKeypoints]) -> dict | None:
    """Weighted-median-aggregate corners + center points across N frames.

    Returns dict with 'corners' (4), 'center_left', 'center_right' plus
    'corner_confs', 'center_confs', 'n_frames'. Corners/centers are (x, y)
    normalized. Returns None if no frame produced a valid 6-keypoint detection.
    """
    valid = [
        fk for fk in frame_results
        if fk is not None and fk.center_points is not None and len(fk.center_points) == 2
    ]
    if not valid:
        return None

    corners_out = []
    corner_confs_out = []
    for i in range(4):
        xs = np.array([fk.corners[i]["x"] for fk in valid], dtype=np.float64)
        ys = np.array([fk.corners[i]["y"] for fk in valid], dtype=np.float64)
        w = np.array([fk.kpt_confidences[i] for fk in valid], dtype=np.float64)
        mx, my = _weighted_median_xy(xs, ys, w)
        corners_out.append((mx, my))
        corner_confs_out.append(float(np.median(w)))

    center_out = []
    center_confs_out = []
    for i in range(2):
        # ``valid`` is filtered to only FrameKeypoints with 2 center_points, so
        # these attributes are never None here — narrow for type-checker.
        xs = np.array(
            [fk.center_points[i]["x"] for fk in valid if fk.center_points is not None],
            dtype=np.float64,
        )
        ys = np.array(
            [fk.center_points[i]["y"] for fk in valid if fk.center_points is not None],
            dtype=np.float64,
        )
        w = np.array(
            [
                (fk.center_confidences[i] if fk.center_confidences else 0.0)
                for fk in valid
            ],
            dtype=np.float64,
        )
        mx, my = _weighted_median_xy(xs, ys, w)
        center_out.append((mx, my))
        center_confs_out.append(float(np.median(w)))

    return {
        "corners": corners_out,
        "center_left": center_out[0],
        "center_right": center_out[1],
        "corner_confs": corner_confs_out,
        "center_confs": center_confs_out,
        "n_frames": len(valid),
    }


# --------------------------------------------------------------------------
# Sanity + confidence gating
# --------------------------------------------------------------------------


def _sanity_ok(base_l: tuple[float, float], base_r: tuple[float, float],
               corners: list[tuple[float, float]]) -> bool:
    """Reject net-base detections that fall outside a sensible band between
    the far baseline (corners 2, 3) and near baseline (corners 0, 1)."""
    near_y = 0.5 * (corners[0][1] + corners[1][1])
    far_y = 0.5 * (corners[2][1] + corners[3][1])
    if near_y <= far_y:
        return False  # degenerate geometry
    span = near_y - far_y
    base_y_mean = 0.5 * (base_l[1] + base_r[1])
    frac = (base_y_mean - far_y) / span
    return SANITY_BASE_LOW <= frac <= SANITY_BASE_HIGH


def _apply_confidence_gating(
    base_l: tuple[float, float],
    base_r: tuple[float, float],
    conf_l: float,
    conf_r: float,
    corners: list[tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], str, str]:
    """If one center point is low-conf, mirror its Y from the high-conf side.

    X is kept at the corner's sideline X (0 and W extrapolated onto the
    line through the two baselines at court-y=8). Returns the gated
    (base_l, base_r, left_source, right_source).
    """
    lsrc, rsrc = "keypoint", "keypoint"
    if conf_l < CENTER_CONF_MIN and conf_r >= CENTER_CONF_MIN:
        # Mirror L from R: use right-side Y adjusted by the perspective ratio
        # between the two sidelines in the 4-corner homography.
        y_ratio = (corners[3][1] - corners[0][1]) / max(
            1e-6, corners[2][1] - corners[1][1]
        )
        base_l = (base_l[0], base_r[1] + (corners[3][1] - corners[2][1]) * 0.5 * y_ratio)
        lsrc = "mirrored"
    elif conf_r < CENTER_CONF_MIN and conf_l >= CENTER_CONF_MIN:
        y_ratio = (corners[2][1] - corners[1][1]) / max(
            1e-6, corners[3][1] - corners[0][1]
        )
        base_r = (base_r[0], base_l[1] + (corners[2][1] - corners[3][1]) * 0.5 * y_ratio)
        rsrc = "mirrored"
    return base_l, base_r, lsrc, rsrc


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _sample_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = int(2.0 * fps)
    start = min(skip, total // 4)
    end = max(start + 1, total - skip)
    step = max(1, (end - start) // n_frames)
    indices = list(range(start, end, step))[:n_frames]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def estimate_net_line(
    video_path: str | Path,
    *,
    image_width: int,
    image_height: int,
    video_key: str | None = None,
    n_frames: int = KEYPOINT_AGG_N,
    detector: CourtKeypointDetector | None = None,
    use_cache: bool = True,
) -> NetLine | None:
    """Estimate a net line for the video via keypoint aggregation.

    Args:
        video_path: path to a locally-readable video file (proxy or processed).
        image_width / image_height: frame dimensions (used to seed calibrate_camera).
        video_key: cache key (typically DB video id). Skips cache if None.
        n_frames: frames to sample for aggregation. 30 is the tested sweet spot.
        detector: optional pre-initialised CourtKeypointDetector.
        use_cache: set False to force recompute.
    """
    video_path = Path(video_path)
    if detector is None:
        detector = CourtKeypointDetector()
        if not detector.model_exists:
            logger.warning("court keypoint model not available")
            return None

    model_path = detector._model_path  # noqa: SLF001
    if use_cache and video_key:
        cached = _cache_load(video_key, model_path, n_frames)
        if cached:
            return cached

    frames = _sample_frames(video_path, n_frames)
    if not frames:
        return None

    frame_results: list[FrameKeypoints] = []
    for f in frames:
        fk = detector._detect_frame(f)  # noqa: SLF001
        if fk is not None:
            frame_results.append(fk)

    if not frame_results:
        return None

    agg = _aggregate_keypoints(frame_results)
    if agg is None:
        return None

    warnings: list[str] = []
    if agg["n_frames"] < 5:
        warnings.append("few_frames_aggregated")

    # Build camera from aggregated corners
    cam = calibrate_camera(agg["corners"], DEFAULT_COURT_CORNERS, image_width, image_height)
    if cam is None:
        return None

    base_l = agg["center_left"]
    base_r = agg["center_right"]
    conf_l, conf_r = agg["center_confs"]

    # Confidence gating
    base_l, base_r, lsrc, rsrc = _apply_confidence_gating(
        base_l, base_r, conf_l, conf_r, agg["corners"],
    )
    if lsrc == "mirrored":
        warnings.append("left_mirrored")
    if rsrc == "mirrored":
        warnings.append("right_mirrored")

    # Sanity validation
    if not _sanity_ok(base_l, base_r, agg["corners"]):
        warnings.append("sanity_failed")
        # Still return — caller can decide whether to trust. Confidence flag handles it.

    # Camera-projected net-top (for shift delta) + net-base (for reference).
    hom_base_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, 0.0]))
    hom_base_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, 0.0]))
    hom_top_l = project_3d_to_image(cam, np.array([0.0, NET_Y_M, NET_HEIGHT_M]))
    hom_top_r = project_3d_to_image(cam, np.array([COURT_WIDTH_M, NET_Y_M, NET_HEIGHT_M]))

    shift_l = base_l[1] - hom_base_l[1]
    shift_r = base_r[1] - hom_base_r[1]
    top_l = (hom_top_l[0], hom_top_l[1] + shift_l)
    top_r = (hom_top_r[0], hom_top_r[1] + shift_r)

    nl = NetLine(
        base_left_xy=base_l,
        base_right_xy=base_r,
        top_left_xy=top_l,
        top_right_xy=top_r,
        confidence=min(conf_l, conf_r) if lsrc == "keypoint" and rsrc == "keypoint" else 0.5 * (conf_l + conf_r),
        left_source=lsrc,
        right_source=rsrc,
        reproj_err_px=float(cam.reprojection_error),
        focal_px=float(cam.focal_length_px),
        n_frames_used=int(agg["n_frames"]),
        warnings=warnings,
    )
    if use_cache and video_key:
        _cache_store(video_key, model_path, n_frames, nl)
    return nl


# --------------------------------------------------------------------------
# Convenience: download-and-estimate helper for S3-backed videos
# --------------------------------------------------------------------------


def estimate_net_line_from_s3(
    s3_key: str,
    *,
    video_id: str,
    image_width: int,
    image_height: int,
    n_frames: int = KEYPOINT_AGG_N,
    detector: CourtKeypointDetector | None = None,
    use_cache: bool = True,
    duration_s: float = 30.0,
) -> NetLine | None:
    """Download a small video segment from S3 and run the estimator."""
    if use_cache:
        cached = _cache_load(
            video_id,
            (detector or CourtKeypointDetector())._model_path,  # noqa: SLF001
            n_frames,
        )
        if cached:
            return cached

    import boto3

    endpoint = os.environ.get("S3_ENDPOINT") or "http://localhost:9000"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.environ.get("AWS_REGION", "us-east-1"),
    )
    bucket = os.environ.get("S3_BUCKET_NAME", "rallycut-dev")
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": s3_key},
        ExpiresIn=3600,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", url, "-t", f"{duration_s:.1f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
            "-an", str(tmp_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            logger.warning("ffmpeg download failed: %s", r.stderr.strip())
            return None
        return estimate_net_line(
            tmp_path,
            image_width=image_width,
            image_height=image_height,
            video_key=video_id,
            n_frames=n_frames,
            detector=detector,
            use_cache=use_cache,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
