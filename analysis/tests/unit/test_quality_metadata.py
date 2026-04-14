from rallycut.quality.metadata import check_metadata, VideoMetadata
from rallycut.quality.types import Tier


def test_short_video_hard_blocks():
    meta = VideoMetadata(duration_s=5, width=1920, height=1080, fps=30)
    result = check_metadata(meta)
    ids = {i.id for i in result.issues}
    assert "video_too_short" in ids
    blocks = [i for i in result.issues if i.tier == Tier.BLOCK]
    assert len(blocks) == 1


def test_low_resolution_soft_gates():
    meta = VideoMetadata(duration_s=120, width=640, height=480, fps=30)
    result = check_metadata(meta)
    issue = next(i for i in result.issues if i.id == "resolution_too_low")
    assert issue.tier == Tier.GATE


def test_low_fps_soft_gates():
    meta = VideoMetadata(duration_s=120, width=1920, height=1080, fps=15)
    result = check_metadata(meta)
    issue = next(i for i in result.issues if i.id == "fps_too_low")
    assert issue.tier == Tier.GATE


def test_normal_metadata_produces_no_issues():
    meta = VideoMetadata(duration_s=600, width=1920, height=1080, fps=30)
    result = check_metadata(meta)
    assert result.issues == []


def test_duration_is_bounded_below_for_severity():
    # A 0-length video should still produce a block, not crash
    meta = VideoMetadata(duration_s=0, width=1920, height=1080, fps=30)
    result = check_metadata(meta)
    assert any(i.id == "video_too_short" for i in result.issues)
