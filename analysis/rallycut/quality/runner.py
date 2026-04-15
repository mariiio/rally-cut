"""Top-level preflight runner.

Loads video metadata + court corners, then runs the two surviving preflight
checks (metadata invariants + court geometry) and merges their results into a
QualityReport.

Post-validation (2026-04-15) the pipeline dropped `camera_too_far`,
`crowded_scene`, `shaky_camera`, `too_dark`, `overexposed`, and
`video_rotated` — all had zero or negative calibration lift and never
correlated with pipeline failure. With them gone, preflight no longer needs
YOLO person-detection or dense frame sampling; the court-keypoint detector
is the only heavy step.

`_load_video_inputs` is deliberately extracted so unit tests can patch it.

Note: the CLIP-based beach-VB classifier (beach_vb_classifier.py) is intentionally
NOT wired in here for A1. The "wrong sport / wrong angle" case is already covered by
the court-confidence branch of `camera_geometry`. CLIP will be re-evaluated in
Project C once open-clip is available in the runtime image.
"""
from __future__ import annotations

from rallycut.quality.camera_geometry import CourtCorners, check_camera_geometry
from rallycut.quality.metadata import VideoMetadata, check_metadata
from rallycut.quality.types import QualityReport


def _load_video_inputs(
    video_path: str, sample_seconds: int
) -> tuple[VideoMetadata, CourtCorners]:
    """Return (metadata, court_corners).

    Integration-only: wraps ffprobe + court-keypoint inference. Unit tests
    patch this.
    """
    from rallycut.core.video import Video
    from rallycut.court.detector import CourtDetectionConfig, CourtDetector

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    meta = VideoMetadata.from_ffprobe(video_path)

    # ── 2. Convert sample_seconds to end_frame for the court detector ────────
    # Using the Video wrapper (not cv2.VideoCapture) avoids the descriptor leak
    # pre-review commits had.
    end_frame_for_court: int | None = None
    with Video(video_path) as vid:
        fps = vid.info.fps
        total_frames = vid.info.frame_count
    if fps and fps > 0:
        end_frame_for_court = min(int(fps * sample_seconds), total_frames) or None

    # ── 3. Court corners via CourtDetector (keypoint → classical fallback) ───
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

    return meta, corners


def run_full_preflight(video_path: str, sample_seconds: int = 60) -> QualityReport:
    meta, corners = _load_video_inputs(video_path, sample_seconds=sample_seconds)
    results = [
        check_metadata(meta),
        check_camera_geometry(corners),
    ]
    return QualityReport.from_checks(
        results, source="preflight", sample_seconds=sample_seconds, duration_ms=int(meta.duration_s * 1000)
    )
