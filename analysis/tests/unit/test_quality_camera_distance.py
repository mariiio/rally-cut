from rallycut.quality.camera_distance import check_camera_distance, Detection
from rallycut.quality.types import Tier


def test_players_large_in_frame_passes():
    # bbox height ~0.3 of frame — normal
    dets = [[Detection(x=0.5, y=0.5, w=0.1, h=0.3) for _ in range(4)] for _ in range(10)]
    result = check_camera_distance(dets)
    assert result.issues == []


def test_players_very_small_gates():
    # bbox height 0.05 — far away
    dets = [[Detection(x=0.5, y=0.5, w=0.02, h=0.05) for _ in range(4)] for _ in range(10)]
    result = check_camera_distance(dets)
    issue = next(i for i in result.issues if i.id == "camera_too_far")
    assert issue.tier == Tier.GATE


def test_empty_detections_produces_no_issue():
    # If no detections at all, this check can't decide — silent
    result = check_camera_distance([[]] * 10)
    assert result.issues == []
