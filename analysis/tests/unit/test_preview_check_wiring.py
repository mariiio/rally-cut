"""Confirm preview-check merges court-geometry + beach-VB issues."""
from unittest.mock import patch

from rallycut.quality.camera_geometry import CourtCorners
from rallycut.quality.types import Tier


def _good_corners() -> CourtCorners:
    return CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )


def test_preview_check_fires_beach_vb_block_when_scores_low(tmp_path, monkeypatch):
    from rallycut.cli.commands import preview_check as pc

    # Fake frame files — preview-check only needs the first to exist
    (tmp_path / "frame_0.jpg").write_bytes(b"\x00" * 16)

    with patch.object(pc, "_detect_corners_from_frame", return_value=_good_corners()), \
         patch.object(pc, "_score_beach_vb_for_frames", return_value=[0.1, 0.15, 0.12, 0.2, 0.18]):
        report = pc._run(tmp_path, width=640, height=360, duration_s=60.0)

    ids = {i.id for i in report.issues}
    assert "not_beach_volleyball" in ids
    block = next(i for i in report.issues if i.id == "not_beach_volleyball")
    assert block.tier == Tier.BLOCK


def test_preview_check_passes_when_scores_high(tmp_path):
    from rallycut.cli.commands import preview_check as pc

    (tmp_path / "frame_0.jpg").write_bytes(b"\x00" * 16)

    with patch.object(pc, "_detect_corners_from_frame", return_value=_good_corners()), \
         patch.object(pc, "_score_beach_vb_for_frames", return_value=[0.95, 0.97, 0.96, 0.98, 0.94]):
        report = pc._run(tmp_path, width=640, height=360, duration_s=60.0)

    assert report.issues == []
