"""Top-level preflight runner.

Loads inputs once (metadata, sampled frames, court corners, person detections),
then runs each check and merges their results into a QualityReport.

`_load_video_inputs` is deliberately extracted so unit tests can patch it.

Note: the CLIP-based beach-VB classifier (beach_vb_classifier.py) is intentionally
NOT wired in here for A1. The "wrong sport / wrong angle" case is already covered by
the court-confidence branch of `camera_geometry`. CLIP will be re-evaluated in
Project C once open-clip is available in the runtime image.
"""
from __future__ import annotations

from rallycut.quality.camera_distance import check_camera_distance
from rallycut.quality.camera_geometry import check_camera_geometry
from rallycut.quality.crowd_density import check_crowd_density
from rallycut.quality.metadata import check_brightness, check_metadata
from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.types import QualityReport


def _load_video_inputs(video_path: str, sample_seconds: int):
    """Return (metadata, sampled_frames, court_corners, person_detections, court_bbox).

    Integration-only: wraps ffprobe, frame sampling, court-keypoint inference,
    and a fast YOLO pass. Heavy; unit tests patch this.

    Note: CLIP zero-shot scoring was removed for A1 — the CLIP result is dropped
    from this tuple. The beach-VB classifier (beach_vb_classifier.py) is preserved
    for Project C but is not called here.
    """
    # All heavy imports are function-local so the module loads for unit tests
    # even when these heavy deps aren't available.
    import logging
    import os

    import cv2
    import numpy as np

    from rallycut.core.video import Video
    from rallycut.court.detector import CourtDetectionConfig, CourtDetector
    from rallycut.quality.camera_distance import Detection
    from rallycut.quality.camera_geometry import CourtCorners
    from rallycut.quality.metadata import VideoMetadata

    logger = logging.getLogger(__name__)

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    meta = VideoMetadata.from_ffprobe(video_path)

    # ── 2. Sample ~10 frames evenly across the first sample_seconds ──────────
    n_frames = 10
    with Video(video_path) as vid:
        fps = vid.info.fps
        total_frames = vid.info.frame_count
        max_frame = min(total_frames, int(fps * sample_seconds)) if fps > 0 else total_frames
        max_frame = max(max_frame, 1)
        step = max(1, max_frame // n_frames)
        frames: list[np.ndarray] = []
        for _, frame in vid.iter_frames(start_frame=0, end_frame=max_frame, step=step):
            frames.append(frame)
            if len(frames) >= n_frames:
                break

    # ── 3. Court corners via CourtDetector (keypoint → classical fallback) ───
    # Convert sample_seconds to end_frame for the detector
    end_frame_for_court: int | None = None
    if fps and fps > 0:
        end_frame_for_court = min(
            int(fps * sample_seconds),
            int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT)),
        ) or None

    detector = CourtDetector(CourtDetectionConfig())
    raw_result = detector.detect(
        video_path,
        start_frame=0,
        end_frame=end_frame_for_court,
    )

    # CourtDetectionResult.corners order: [near-left, near-right, far-right, far-left]
    # CourtCorners convention: tl=top-left (far-left), tr=top-right (far-right),
    #                          br=bottom-right (near-right), bl=bottom-left (near-left)
    if raw_result.corners and len(raw_result.corners) == 4:
        c = raw_result.corners
        corners = CourtCorners(
            tl=(c[3]["x"], c[3]["y"]),  # far-left  → top-left
            tr=(c[2]["x"], c[2]["y"]),  # far-right → top-right
            br=(c[1]["x"], c[1]["y"]),  # near-right → bottom-right
            bl=(c[0]["x"], c[0]["y"]),  # near-left  → bottom-left
            confidence=raw_result.confidence,
        )
    else:
        # No court detected — return zero-confidence corners at image edges
        corners = CourtCorners(
            tl=(0.0, 0.0), tr=(1.0, 0.0),
            br=(1.0, 1.0), bl=(0.0, 1.0),
            confidence=0.0,
        )

    # ── 4. Person detections via YOLO (one pass over sampled frames) ─────────
    # Disable YOLO telemetry / auto-update noise
    os.environ.setdefault("YOLO_AUTOCHECK", "False")
    from ultralytics import YOLO  # noqa: PLC0415

    # yolo11s.pt lives in analysis/ (project root for the analysis package)
    analysis_root = os.path.join(os.path.dirname(__file__), "..", "..")
    yolo_path = os.path.normpath(os.path.join(analysis_root, "yolo11s.pt"))
    if not os.path.exists(yolo_path):
        yolo_path = "yolo11s.pt"  # fallback: ultralytics auto-download

    yolo = YOLO(yolo_path)
    PERSON_CLASS = 0

    per_frame_dets: list[list[Detection]] = []
    for frame in frames:
        result = yolo.predict(frame, classes=[PERSON_CLASS], verbose=False, imgsz=640)[0]
        frame_dets: list[Detection] = []
        if result.boxes is not None and len(result.boxes):
            h_img, w_img = frame.shape[:2]
            for box in result.boxes:
                cls = int(box.cls[0])
                if cls != PERSON_CLASS:
                    continue
                # xywhn = normalized center-x, center-y, width, height
                xywhn = box.xywhn[0].cpu().numpy()
                frame_dets.append(Detection(
                    x=float(xywhn[0]),
                    y=float(xywhn[1]),
                    w=float(xywhn[2]),
                    h=float(xywhn[3]),
                ))
        per_frame_dets.append(frame_dets)

    # ── 5. Court bbox from corners ────────────────────────────────────────────
    court_bbox = (
        min(corners.tl[0], corners.bl[0]),
        min(corners.tl[1], corners.tr[1]),
        max(corners.tr[0], corners.br[0]),
        max(corners.bl[1], corners.br[1]),
    )

    return meta, frames, corners, per_frame_dets, court_bbox


def run_full_preflight(video_path: str, sample_seconds: int = 60) -> QualityReport:
    meta, frames, corners, dets, court_bbox = _load_video_inputs(
        video_path, sample_seconds=sample_seconds
    )
    results = [
        check_metadata(meta),
        check_brightness(frames),
        check_camera_geometry(corners),
        check_camera_distance(dets),
        check_crowd_density(dets, court_bbox),
        check_shakiness(frames),
    ]
    return QualityReport.from_checks(
        results, source="preflight", sample_seconds=sample_seconds, duration_ms=int(meta.duration_s * 1000)
    )
