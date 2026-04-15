from unittest.mock import patch

from rallycut.quality.runner import run_full_preflight
from rallycut.quality.metadata import VideoMetadata
from rallycut.quality.camera_geometry import CourtCorners


def test_runner_produces_serializable_report_with_good_video(tmp_path):
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    corners = CourtCorners(tl=(0.3, 0.4), tr=(0.7, 0.4), br=(0.8, 0.8), bl=(0.2, 0.8), confidence=0.9)

    with patch("rallycut.quality.runner._load_video_inputs", return_value=(meta, corners)):
        report = run_full_preflight("/tmp/fake.mp4", sample_seconds=60)

    d = report.to_dict()
    assert d["version"] == 2
    assert d["preflight"]["sampleSeconds"] == 60
    assert d["issues"] == []  # all checks should be silent on good inputs


def test_runner_surfaces_block_issue_for_bad_video(tmp_path):
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    bad_corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.1)

    with patch("rallycut.quality.runner._load_video_inputs", return_value=(meta, bad_corners)):
        report = run_full_preflight("/tmp/fake.mp4", sample_seconds=60)

    d = report.to_dict()
    assert any(i["id"] == "wrong_angle_or_not_volleyball" and i["tier"] == "block" for i in d["issues"])
