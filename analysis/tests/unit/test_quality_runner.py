from unittest.mock import patch

import numpy as np

from rallycut.quality.runner import run_full_preflight
from rallycut.quality.metadata import VideoMetadata
from rallycut.quality.camera_geometry import CourtCorners
from rallycut.quality.beach_vb_classifier import BeachVBProbabilities


def test_runner_produces_serializable_report_with_good_video(tmp_path):
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    corners = CourtCorners(tl=(0.3, 0.4), tr=(0.7, 0.4), br=(0.8, 0.8), bl=(0.2, 0.8), confidence=0.9)
    probs = [BeachVBProbabilities(0.93, 0.04, 0.03)] * 5

    with patch("rallycut.quality.runner._load_video_inputs", return_value=(meta, frames, corners, [[], [], []], (0.2, 0.3, 0.8, 0.9), probs)):
        report = run_full_preflight("/tmp/fake.mp4", sample_seconds=60)

    d = report.to_dict()
    assert d["version"] == 2
    assert d["preflight"]["sampleSeconds"] == 60
    assert d["issues"] == []  # all checks should be silent on good inputs


def test_runner_surfaces_block_issue_for_bad_video(tmp_path):
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    frames = [np.full((120, 160, 3), 128, dtype=np.uint8) for _ in range(10)]
    bad_corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.1)
    probs = [BeachVBProbabilities(0.93, 0.04, 0.03)] * 5

    with patch("rallycut.quality.runner._load_video_inputs", return_value=(meta, frames, bad_corners, [[], [], []], (0.2, 0.3, 0.8, 0.9), probs)):
        report = run_full_preflight("/tmp/fake.mp4", sample_seconds=60)

    d = report.to_dict()
    assert any(i["id"] == "wrong_angle_or_not_volleyball" and i["tier"] == "block" for i in d["issues"])
