import math

from rallycut.quality.camera_geometry import check_camera_geometry, CourtCorners
from rallycut.quality.types import Tier


def _square_corners() -> CourtCorners:
    # Perfect straight-on court, centered
    return CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )


def test_straight_court_passes():
    result = check_camera_geometry(_square_corners())
    assert result.issues == []


def test_tilted_court_produces_advisory():
    # Rotate baseline by 10 degrees
    angle = math.radians(10)
    c = math.cos(angle); s = math.sin(angle)
    def rot(p):
        return (0.5 + (p[0]-0.5)*c - (p[1]-0.5)*s, 0.5 + (p[0]-0.5)*s + (p[1]-0.5)*c)
    sq = _square_corners()
    corners = CourtCorners(
        tl=rot(sq.tl), tr=rot(sq.tr), br=rot(sq.br), bl=rot(sq.bl),
        confidence=0.9,
    )
    result = check_camera_geometry(corners)
    issue = next(i for i in result.issues if i.id == "video_rotated")
    assert issue.tier == Tier.ADVISORY


def test_no_court_hard_blocks():
    corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.1)
    result = check_camera_geometry(corners)
    issue = next(i for i in result.issues if i.id == "wrong_angle_or_not_volleyball")
    assert issue.tier == Tier.BLOCK


def test_side_view_hard_blocks():
    # Baselines nearly vertical (camera is beside the court, not behind the baseline)
    corners = CourtCorners(
        tl=(0.4, 0.2), tr=(0.45, 0.8),
        br=(0.55, 0.8), bl=(0.5, 0.2),
        confidence=0.9,
    )
    result = check_camera_geometry(corners)
    issue = next(i for i in result.issues if i.id == "wrong_angle_or_not_volleyball")
    assert issue.tier == Tier.BLOCK
