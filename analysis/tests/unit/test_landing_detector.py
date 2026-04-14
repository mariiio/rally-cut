"""Tests for landing detector validation logic.

Regression test for the inverted opposite-half validation bug introduced
in commit f538bf5. The bug named `court_pos[1] > HALF_COURT_M` as
"target_on_near" when it is actually the far side (by the convention
documented in ``action_classifier.py:442``: near baseline at cy=0, far
baseline at cy=16). As a result the validation dropped correct
opposite-half landings and accepted wrong same-half / off-court ones.
"""

from __future__ import annotations

import pytest

from rallycut.court.calibration import CourtCalibrator
from rallycut.statistics.landing_detector import (
    HALF_COURT_M,
    _lands_on_opposite_half,
    detect_rally_landings,
)
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
)
from rallycut.tracking.ball_tracker import BallPosition

# Image corners are arbitrary here — we build a calibrator whose image_to_court
# is the identity on normalised inputs by mapping full-frame corners to court
# corners. That means a raw position dict with x,y in [0,1] maps directly to
# (x * COURT_WIDTH, y * COURT_LENGTH) in metres — easy to reason about.
_IMAGE_CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


@pytest.fixture
def calibrator() -> CourtCalibrator:
    cal = CourtCalibrator()
    cal.calibrate(_IMAGE_CORNERS)
    return cal


def _pos(track_id: int, frame: int, court_y_m: float) -> dict:
    """Build a raw position dict whose feet project to court_y_m metres.

    Court length is 16m; calibrator maps normalised image y to court_y = y*16.
    The detector estimates feet from bbox bottom-center: (x, y + h/2).
    We set y such that y + h/2 == court_y_m / 16.
    """
    h = 0.05
    return {
        "trackId": track_id,
        "frameNumber": frame,
        "x": 0.5,
        "y": (court_y_m / 16.0) - h / 2.0,
        "width": 0.05,
        "height": h,
        "confidence": 0.9,
    }


def _ball(frame: int, x: float = 0.5, y: float = 0.5) -> BallPosition:
    return BallPosition(frame_number=frame, x=x, y=y, confidence=0.9)


class TestLandsOnOppositeHalf:
    """Direct tests for the opposite-half predicate.

    Pins the court convention so the validation cannot silently invert
    again: Y=0 is near baseline, Y=HALF_COURT_M is the net, Y>HALF_COURT_M
    is the far half.
    """

    def test_near_source_far_target_is_opposite(self):
        assert _lands_on_opposite_half("near", HALF_COURT_M + 0.1) is True

    def test_near_source_near_target_is_same_half(self):
        assert _lands_on_opposite_half("near", HALF_COURT_M - 0.1) is False

    def test_far_source_near_target_is_opposite(self):
        assert _lands_on_opposite_half("far", HALF_COURT_M - 0.1) is True

    def test_far_source_far_target_is_same_half(self):
        assert _lands_on_opposite_half("far", HALF_COURT_M + 0.1) is False

    def test_unknown_source_is_rejected(self):
        # We can't validate an unknown source side, so default to drop.
        assert _lands_on_opposite_half("unknown", 2.0) is False
        assert _lands_on_opposite_half("unknown", 14.0) is False

    def test_net_boundary_counts_as_near(self):
        # Exact net line is treated as near (>, not >=). Matches the
        # documented convention.
        assert _lands_on_opposite_half("far", HALF_COURT_M) is True
        assert _lands_on_opposite_half("near", HALF_COURT_M) is False


def test_near_server_far_receiver_is_valid(calibrator):
    """Near-side server (team A) hitting receiver on far side should land."""
    serve = ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=10, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=1, court_side="near", confidence=0.9, team="A",
    )
    receive = ClassifiedAction(
        action_type=ActionType.RECEIVE,
        frame=25, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=2, court_side="far", confidence=0.9, team="B",
    )
    ra = RallyActions(actions=[serve, receive], rally_id="r1")

    positions_raw = [
        _pos(1, 10, court_y_m=2.0),   # server near (y<8)
        _pos(2, 25, court_y_m=12.0),  # receiver far (y>8)
    ]
    ball = [_ball(f) for f in range(5, 40)]

    landings = detect_rally_landings(
        ra, ball, calibrator, 1920, 1080, positions_raw=positions_raw,
    )
    serve_landings = [lp for lp in landings if lp.action_type == "serve"]
    assert len(serve_landings) == 1
    assert serve_landings[0].court_y is not None
    assert serve_landings[0].court_y > 8.0  # landed on far half


def test_far_server_near_receiver_is_valid(calibrator):
    """Far-side server (team B) hitting receiver on near side should land."""
    serve = ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=10, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=1, court_side="far", confidence=0.9, team="B",
    )
    receive = ClassifiedAction(
        action_type=ActionType.RECEIVE,
        frame=25, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=2, court_side="near", confidence=0.9, team="A",
    )
    ra = RallyActions(actions=[serve, receive], rally_id="r2")

    positions_raw = [
        _pos(1, 10, court_y_m=14.0),  # server far
        _pos(2, 25, court_y_m=3.0),   # receiver near
    ]
    ball = [_ball(f) for f in range(5, 40)]

    landings = detect_rally_landings(
        ra, ball, calibrator, 1920, 1080, positions_raw=positions_raw,
    )
    serve_landings = [lp for lp in landings if lp.action_type == "serve"]
    assert len(serve_landings) == 1
    assert serve_landings[0].court_y is not None
    assert serve_landings[0].court_y < 8.0  # landed on near half


def test_same_half_landing_is_rejected(calibrator):
    """Near-side server with receiver *also* projected to near half must drop.

    This covers the misattribution case the validation was added for.
    """
    serve = ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=10, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=1, court_side="near", confidence=0.9, team="A",
    )
    receive = ClassifiedAction(
        action_type=ActionType.RECEIVE,
        frame=25, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=2, court_side="near", confidence=0.9, team="A",
    )
    ra = RallyActions(actions=[serve, receive], rally_id="r3")

    positions_raw = [
        _pos(1, 10, court_y_m=2.0),   # server near
        _pos(2, 25, court_y_m=3.0),   # "receiver" *also* near — misattribution
    ]
    ball = [_ball(f) for f in range(5, 40)]

    landings = detect_rally_landings(
        ra, ball, calibrator, 1920, 1080, positions_raw=positions_raw,
    )
    assert [lp for lp in landings if lp.action_type == "serve"] == []


def test_mid_rally_attack_valid_opposite_half(calibrator):
    """Attack by near-side team with next-contact on far half should land."""
    attack = ClassifiedAction(
        action_type=ActionType.ATTACK,
        frame=40, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=1, court_side="near", confidence=0.9, team="A",
    )
    dig = ClassifiedAction(
        action_type=ActionType.DIG,
        frame=55, ball_x=0.5, ball_y=0.5, velocity=0.0,
        player_track_id=3, court_side="far", confidence=0.9, team="B",
    )
    ra = RallyActions(actions=[attack, dig], rally_id="r4")

    positions_raw = [
        _pos(1, 40, court_y_m=5.0),   # attacker near
        _pos(3, 55, court_y_m=11.0),  # defender far
    ]
    ball = [_ball(f) for f in range(30, 70)]

    landings = detect_rally_landings(
        ra, ball, calibrator, 1920, 1080, positions_raw=positions_raw,
    )
    attack_landings = [lp for lp in landings if lp.action_type == "attack"]
    assert len(attack_landings) == 1
    assert attack_landings[0].court_y is not None
    assert attack_landings[0].court_y > 8.0
