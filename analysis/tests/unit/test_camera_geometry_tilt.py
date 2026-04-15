import math

from rallycut.quality.camera_geometry import CourtCorners, baseline_tilt_deg


def test_straight_baseline_is_zero_tilt():
    corners = CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )
    assert baseline_tilt_deg(corners) == 0.0


def test_10deg_baseline_returns_10deg():
    angle = math.radians(10)
    c, s = math.cos(angle), math.sin(angle)
    def rot(p):
        return (0.5 + (p[0] - 0.5) * c - (p[1] - 0.5) * s,
                0.5 + (p[0] - 0.5) * s + (p[1] - 0.5) * c)
    base = CourtCorners(
        tl=(0.3, 0.4), tr=(0.7, 0.4),
        br=(0.8, 0.8), bl=(0.2, 0.8),
        confidence=0.9,
    )
    tilted = CourtCorners(
        tl=rot(base.tl), tr=rot(base.tr), br=rot(base.br), bl=rot(base.bl),
        confidence=0.9,
    )
    assert abs(baseline_tilt_deg(tilted) - 10.0) < 0.1


def test_degenerate_corners_return_zero():
    corners = CourtCorners(tl=(0, 0), tr=(0, 0), br=(0, 0), bl=(0, 0), confidence=0.0)
    assert baseline_tilt_deg(corners) == 0.0
