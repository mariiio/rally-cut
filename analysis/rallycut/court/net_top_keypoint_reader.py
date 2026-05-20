"""v9 SOTA net-top reader — directly reads net-top tape position from the
8-keypoint court keypoint detector (no solvePnP, no ridge regression).

The 8-keypoint court model outputs:
  0..3  : court corners
  4..5  : net-base center points (sideline x net midline, on ground)
  6..7  : net-top tape at each post in image space (v9)

This module aggregates keypoints 6 + 7 across N sampled frames per
video and returns a `NetTopLine` struct that mirrors the shape of
`net_line_estimator.NetLine` (so the contact_detector cascade can
swap them transparently). Includes a per-video disk cache keyed on
(video_id, model_mtime, n_frames) — same convention as the v8
`net_line_estimator` cache.

When the loaded keypoint model emits fewer than 8 keypoints (i.e. an
older 6-kpt or 4-kpt model), this reader returns `None` and the caller
falls through to the v8 `estimate_net_line()` solvePnP path. So the
module degrades cleanly until the v9 8-kpt model is installed.

Entry point: `read_net_top(video_path, ...) -> NetTopLine | None`.

Used by `contact_detector._prepare_candidates` as the new top of the
cascade:
    NetTopLine (v9) > NetLine midpoint (v8) > M4 ridge (v7) > v6 traj.
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

from rallycut.court.keypoint_detector import CourtKeypointDetector, FrameKeypoints

logger = logging.getLogger(__name__)

# v9 model is held at a DEDICATED path, separate from the 6-keypoint
# production calibration model (`court_keypoint_best.pt`). Loading the
# v9 model into the default `CourtKeypointDetector` would replace the
# 6-kpt model that production court calibration depends on; the v9 net-
# top training overfits ~0.034 normalized y of center-point regression
# vs the original yolo11m-trained 6-kpt model. Keeping them as two
# separate files isolates the change to this module.
V9_MODEL_PATH = (
    Path(__file__).parent.parent.parent
    / "weights" / "court_keypoint" / "court_keypoint_v9_8kpt.pt"
)

# Per-endpoint minimum confidence to trust the YOLO net-top keypoint.
# Lower than CENTER_CONF_MIN (0.30) because net-top is a thinner, smaller
# feature than the net-base posts and the trainer has less signal per pixel.
NET_TOP_CONF_MIN = 0.20
KEYPOINT_AGG_N = 30
SANITY_NET_Y_LOW = 0.05   # net-top y must be ≥5% from frame top
SANITY_NET_Y_HIGH = 0.95  # ≤95% from frame top (i.e. ≥5% from frame bottom)

CACHE_DIR_ENV = "RALLYCUT_NET_TOP_KPT_CACHE_DIR"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "rallycut" / "net_top_kpt"


@dataclass
class NetTopLine:
    """Auto-detected net-top line in normalized image coordinates.

    Mirrors the `NetLine` shape from `net_line_estimator` so the
    contact_detector cascade can use either type behind the same
    `net_line` parameter — the L/R midpoint computation is identical.
    """

    # Net top line endpoints (the visible tape at the posts).
    top_left_xy: tuple[float, float]
    top_right_xy: tuple[float, float]
    # Per-endpoint confidence + provenance.
    left_confidence: float
    right_confidence: float
    confidence: float  # min of left/right
    left_source: str   # "keypoint" or "fallback"
    right_source: str
    # Diagnostic fields.
    n_frames_used: int
    warnings: list[str]

    # Compatibility shim: contact_detector's cascade reads
    # `net_line.warnings` and `net_line.top_left_xy[1]` /
    # `net_line.top_right_xy[1]`. Those fields exist above so the
    # struct drops in where `NetLine` was expected.

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> NetTopLine:
        return cls(
            top_left_xy=tuple(d["top_left_xy"]),
            top_right_xy=tuple(d["top_right_xy"]),
            left_confidence=float(d["left_confidence"]),
            right_confidence=float(d["right_confidence"]),
            confidence=float(d["confidence"]),
            left_source=d["left_source"],
            right_source=d["right_source"],
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


def _cache_load(video_key: str, model_path: Path, n_frames: int) -> NetTopLine | None:
    path = _cache_dir() / _cache_key(video_key, model_path, n_frames)
    if not path.exists():
        return None
    try:
        return NetTopLine.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, KeyError):
        return None


def _cache_store(video_key: str, model_path: Path, n_frames: int, nl: NetTopLine) -> None:
    path = _cache_dir() / _cache_key(video_key, model_path, n_frames)
    path.write_text(json.dumps(nl.to_dict(), indent=2))


# --------------------------------------------------------------------------
# Multi-frame aggregation
# --------------------------------------------------------------------------


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    total = float(weights.sum())
    if total <= 0:
        return float(np.median(values))
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    idx = int(np.searchsorted(cum, total / 2.0))
    idx = min(idx, len(values) - 1)
    return float(values[order][idx])


def _aggregate_net_top(frame_results: list[FrameKeypoints]) -> dict | None:
    """Weighted-median-aggregate net-top kp 6, 7 across N frames.

    Returns dict with `top_left`, `top_right`, `left_confidence`,
    `right_confidence`, `n_frames`. Returns None if no frame produced
    valid net-top output (i.e. the loaded model is not 8-kpt, or every
    frame had no detection).
    """
    valid = [
        fk for fk in frame_results
        if fk is not None
        and fk.net_top_points is not None
        and len(fk.net_top_points) == 2
        and fk.net_top_confidences is not None
        and len(fk.net_top_confidences) == 2
    ]
    if not valid:
        return None

    top_out = []
    conf_out = []
    for i in range(2):
        xs = np.array(
            [fk.net_top_points[i]["x"] for fk in valid if fk.net_top_points is not None],
            dtype=np.float64,
        )
        ys = np.array(
            [fk.net_top_points[i]["y"] for fk in valid if fk.net_top_points is not None],
            dtype=np.float64,
        )
        w = np.array(
            [
                (fk.net_top_confidences[i] if fk.net_top_confidences else 0.0)
                for fk in valid
            ],
            dtype=np.float64,
        )
        mx = _weighted_median(xs, w)
        my = _weighted_median(ys, w)
        top_out.append((float(mx), float(my)))
        conf_out.append(float(np.median(w)))

    return {
        "top_left": top_out[0],
        "top_right": top_out[1],
        "left_confidence": conf_out[0],
        "right_confidence": conf_out[1],
        "n_frames": len(valid),
    }


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


def read_net_top(
    video_path: str | Path,
    *,
    video_key: str | None = None,
    n_frames: int = KEYPOINT_AGG_N,
    detector: CourtKeypointDetector | None = None,
    use_cache: bool = True,
) -> NetTopLine | None:
    """Read the net top line directly from the 8-keypoint court model.

    Args:
        video_path: path to a locally-readable video file (proxy is fine).
        video_key: cache key (typically DB video id). Skips cache if None.
        n_frames: frames to sample for aggregation. 30 is the tested
            sweet spot, matching `net_line_estimator`.
        detector: optional pre-initialised CourtKeypointDetector.
        use_cache: set False to force recompute.

    Returns:
        NetTopLine, or None when the loaded model is not 8-kpt /
        no usable detections / sanity gate failed. Callers cascade
        to `estimate_net_line()` (v8 solvePnP) when None.
    """
    video_path = Path(video_path)
    if detector is None:
        # Load the v9-specific 8-kpt model, not the default 6-kpt model
        # that production calibration uses.
        detector = CourtKeypointDetector(model_path=V9_MODEL_PATH)
        if not detector.model_exists:
            logger.warning(
                "v9 8-kpt model not available at %s — net-top reader unavailable",
                V9_MODEL_PATH,
            )
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

    agg = _aggregate_net_top(frame_results)
    if agg is None:
        # 6-kpt or 4-kpt model — net-top output not present.
        return None

    warnings: list[str] = []
    if agg["n_frames"] < 5:
        warnings.append("few_frames_aggregated")

    left_conf = agg["left_confidence"]
    right_conf = agg["right_confidence"]
    left_source = "keypoint" if left_conf >= NET_TOP_CONF_MIN else "fallback"
    right_source = "keypoint" if right_conf >= NET_TOP_CONF_MIN else "fallback"

    # Sanity gate: net-top y must be in a plausible band. If wildly out,
    # fall back via warning (caller decides) — same idiom as v8 NetLine.
    top_l = agg["top_left"]
    top_r = agg["top_right"]
    mid_y = (top_l[1] + top_r[1]) / 2.0
    if not (SANITY_NET_Y_LOW <= mid_y <= SANITY_NET_Y_HIGH):
        warnings.append("sanity_failed")

    nl = NetTopLine(
        top_left_xy=(float(top_l[0]), float(top_l[1])),
        top_right_xy=(float(top_r[0]), float(top_r[1])),
        left_confidence=float(left_conf),
        right_confidence=float(right_conf),
        confidence=float(min(left_conf, right_conf)),
        left_source=left_source,
        right_source=right_source,
        n_frames_used=int(agg["n_frames"]),
        warnings=warnings,
    )
    if use_cache and video_key:
        _cache_store(video_key, model_path, n_frames, nl)
    return nl


# --------------------------------------------------------------------------
# Convenience: download-and-read helper for S3-backed videos
# --------------------------------------------------------------------------


def read_net_top_from_s3(
    s3_key: str,
    *,
    video_id: str,
    n_frames: int = KEYPOINT_AGG_N,
    detector: CourtKeypointDetector | None = None,
    use_cache: bool = True,
    duration_s: float = 30.0,
) -> NetTopLine | None:
    """Download a small video segment from S3 and read the net top.

    Mirrors `net_line_estimator.estimate_net_line_from_s3` so the
    fleet-refresh script can swap one for the other with a one-line
    change.
    """
    if use_cache:
        cached = _cache_load(
            video_id,
            (detector or CourtKeypointDetector(model_path=V9_MODEL_PATH))._model_path,  # noqa: SLF001
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
        return read_net_top(
            tmp_path,
            video_key=video_id,
            n_frames=n_frames,
            detector=detector,
            use_cache=use_cache,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
