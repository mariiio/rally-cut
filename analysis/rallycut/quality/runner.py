"""Top-level preflight runner.

Loads inputs once (metadata, sampled frames, court corners, person detections,
beach-VB probabilities), then runs each check and merges their results into a
QualityReport.

`_load_video_inputs` is deliberately extracted so unit tests can patch it.
"""
from __future__ import annotations

from rallycut.quality.beach_vb_classifier import classify_is_beach_vb
from rallycut.quality.camera_distance import check_camera_distance
from rallycut.quality.camera_geometry import check_camera_geometry
from rallycut.quality.crowd_density import check_crowd_density
from rallycut.quality.metadata import check_brightness, check_metadata
from rallycut.quality.shakiness import check_shakiness
from rallycut.quality.types import QualityReport


def _load_video_inputs(video_path: str, sample_seconds: int):
    """Return (metadata, sampled_frames, court_corners, person_detections, court_bbox, clip_probs).

    Integration-only: wraps ffprobe, frame sampling, court-keypoint inference,
    a fast YOLO pass, and CLIP zero-shot scoring. Heavy; unit tests patch this.
    """
    # All heavy imports are function-local so the module loads for unit tests
    # even when video_io / keypoint_detector / yolo_person aren't available.
    from rallycut.detection.video_io import sample_frames  # existing
    from rallycut.court.keypoint_detector import detect_court_corners  # existing
    from rallycut.detection.yolo_person import detect_persons_in_frames  # existing (fast alias)
    from rallycut.quality.beach_vb_classifier import embed_and_score_frames
    from rallycut.quality.camera_geometry import CourtCorners
    from rallycut.quality.metadata import VideoMetadata

    meta = VideoMetadata.from_ffprobe(video_path)
    frames = sample_frames(video_path, n=10, max_seconds=sample_seconds)

    raw_corners = detect_court_corners(video_path, max_seconds=sample_seconds)
    corners = CourtCorners(
        tl=raw_corners.tl, tr=raw_corners.tr, br=raw_corners.br, bl=raw_corners.bl,
        confidence=raw_corners.confidence,
    )

    per_frame_dets = detect_persons_in_frames(frames)
    court_bbox = (
        min(corners.tl[0], corners.bl[0]),
        min(corners.tl[1], corners.tr[1]),
        max(corners.tr[0], corners.br[0]),
        max(corners.bl[1], corners.br[1]),
    )

    clip_probs = embed_and_score_frames(frames[:5])

    return meta, frames, corners, per_frame_dets, court_bbox, clip_probs


def run_full_preflight(video_path: str, sample_seconds: int = 60) -> QualityReport:
    meta, frames, corners, dets, court_bbox, clip_probs = _load_video_inputs(
        video_path, sample_seconds=sample_seconds
    )
    results = [
        check_metadata(meta),
        check_brightness(frames),
        check_camera_geometry(corners),
        check_camera_distance(dets),
        check_crowd_density(dets, court_bbox),
        check_shakiness(frames),
        classify_is_beach_vb(clip_probs),
    ]
    return QualityReport.from_checks(
        results, source="preflight", sample_seconds=sample_seconds, duration_ms=int(meta.duration_s * 1000)
    )
