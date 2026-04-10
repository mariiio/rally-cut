"""Unit tests for ``rallycut.statistics.play_annotations``."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from rallycut.statistics.play_annotations import (
    CROSS_MAX_DEG,
    LINE_MAX_DEG,
    annotate_rally_actions,
    classify_attack_direction_from_xy,
    x_to_zone,
)
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
)
from rallycut.tracking.ball_tracker import BallPosition

# --------------------------------------------------------------------------- #
# Pure helpers.
# --------------------------------------------------------------------------- #


class TestXToZone:
    def test_clamps_below(self) -> None:
        assert x_to_zone(-1.0) == 1

    def test_clamps_above(self) -> None:
        assert x_to_zone(100.0) == 5

    @pytest.mark.parametrize(
        ("x_m", "expected"),
        [
            (0.01, 1),
            (1.59, 1),
            (1.6, 2),
            (3.19, 2),
            (3.2, 3),
            (4.8, 4),
            (6.4, 5),
            (7.99, 5),
        ],
    )
    def test_boundaries(self, x_m: float, expected: int) -> None:
        assert x_to_zone(x_m) == expected

    def test_custom_num_zones(self) -> None:
        # 4 zones → 2m each
        assert x_to_zone(0.1, num_zones=4) == 1
        assert x_to_zone(2.1, num_zones=4) == 2
        assert x_to_zone(6.1, num_zones=4) == 4

    def test_far_side_inversion(self) -> None:
        # From camera's left (x=0.5 → zone 1), far-side team sees zone 5.
        assert x_to_zone(0.5, far_side=True) == 5
        # Center stays center.
        assert x_to_zone(4.0, far_side=True) == 3
        # Camera's right (zone 5) → far-side team's zone 1.
        assert x_to_zone(7.5, far_side=True) == 1

    def test_invalid_num_zones(self) -> None:
        with pytest.raises(ValueError):
            x_to_zone(1.0, num_zones=0)


class TestClassifyAttackDirection:
    def test_line_straight_across(self) -> None:
        # Small |dx| relative to large |dy| → line.
        assert classify_attack_direction_from_xy(4.0, 6.0, 4.2, 12.0) == "line"

    def test_cross_moderate_angle(self) -> None:
        # ~25° angle → within (LINE_MAX_DEG, CROSS_MAX_DEG) → cross.
        assert classify_attack_direction_from_xy(4.0, 6.0, 6.0, 10.0) == "cross"

    def test_cut_sharp_angle(self) -> None:
        # ~60° from straight-across → cut.
        assert classify_attack_direction_from_xy(4.0, 6.0, 8.0, 8.0) == "cut"

    def test_zero_dy_is_unknown(self) -> None:
        assert classify_attack_direction_from_xy(4.0, 8.0, 5.0, 8.0) == "unknown"

    def test_line_boundary(self) -> None:
        # Just under LINE_MAX_DEG — still line.
        import math
        just_under = math.tan(math.radians(LINE_MAX_DEG - 0.5))
        dy = 10.0
        dx = just_under * dy
        assert classify_attack_direction_from_xy(0.0, 0.0, dx, dy) == "line"

    def test_cut_boundary(self) -> None:
        import math
        over = math.tan(math.radians(CROSS_MAX_DEG + 0.5))
        dy = 10.0
        dx = over * dy
        assert classify_attack_direction_from_xy(0.0, 0.0, dx, dy) == "cut"


# --------------------------------------------------------------------------- #
# Integration — annotate_rally_actions on synthetic rallies.
# --------------------------------------------------------------------------- #


@dataclass
class _FakeCalibrator:
    """Stand-in for ``CourtCalibrator``. Maps normalized image (u,v) ↦
    court (x_m, y_m) with ``x_m = u * 8`` and ``y_m = v * 16`` so test
    fixtures can specify coordinates directly in normalized-image space.
    """

    calibrated: bool = True

    @property
    def is_calibrated(self) -> bool:
        return self.calibrated

    def image_to_court(
        self, pt: tuple[float, float], _w: int, _h: int
    ) -> tuple[float, float]:
        u, v = pt
        return (u * 8.0, v * 16.0)


def _mk_action(
    action_type: ActionType, frame: int, *, ball_x: float = 0.5, ball_y: float = 0.5,
    player_track_id: int = 1,
) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=ball_x,
        ball_y=ball_y,
        velocity=0.0,
        player_track_id=player_track_id,
        court_side="near",
        confidence=1.0,
    )


def _ball(frame: int, x: float, y: float) -> BallPosition:
    return BallPosition(frame_number=frame, x=x, y=y, confidence=1.0)


class TestAnnotateRallyActions:
    def test_no_calibrator_is_noop(self) -> None:
        actions = RallyActions(actions=[_mk_action(ActionType.ATTACK, 10)])
        stats = annotate_rally_actions(actions, [], [], calibrator=None)
        assert stats.attacks_total == 0
        assert actions.actions[0].attack_direction is None

    def test_uncalibrated_calibrator_is_noop(self) -> None:
        cal = _FakeCalibrator(calibrated=False)
        actions = RallyActions(actions=[_mk_action(ActionType.ATTACK, 10)])
        stats = annotate_rally_actions(actions, [], [], calibrator=cal)
        assert stats.attacks_total == 0

    def test_attack_direction_line(self) -> None:
        # Attacker at court (4, 2); next-contact player at (4.1, 14).
        # Small dx, big dy → line.  Now uses player feet, not ball.
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 112, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        positions_raw = [
            # Attacker feet at (0.5, 0.175) → court (4, 2.8)
            # (y=0.125 center + height=0.1 → feet at y=0.125+0.05=0.175)
            {"frameNumber": 100, "trackId": atk_tid, "x": 0.5, "y": 0.125,
             "width": 0.05, "height": 0.1},
            # Next-contact player feet at (0.5125, 0.925) → court (4.1, 14.8)
            {"frameNumber": 112, "trackId": dig_tid, "x": 0.5125, "y": 0.875,
             "width": 0.05, "height": 0.1},
        ]
        stats = annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        assert stats.attacks_total == 1
        assert stats.attacks_annotated == 1
        assert atk.attack_direction == "line"

    def test_attack_direction_cut(self) -> None:
        # Attacker at court (1, 2.8); next-contact player at (7, 6).
        # Large dx relative to moderate dy → cut.
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 200, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 220, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        positions_raw = [
            {"frameNumber": 200, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 220, "trackId": dig_tid, "x": 0.875, "y": 0.325,
             "width": 0.05, "height": 0.1},
        ]
        stats = annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        assert stats.attacks_annotated == 1
        assert atk.attack_direction == "cut"

    def test_set_zones_populated(self) -> None:
        # Origin = setter feet court-x, dest = attacker feet court-x.
        cal = _FakeCalibrator()
        setter_tid = 7
        attacker_tid = 9
        set_action = _mk_action(
            ActionType.SET, 50, player_track_id=setter_tid,
        )
        attack = _mk_action(
            ActionType.ATTACK, 80, player_track_id=attacker_tid,
        )
        rally = RallyActions(actions=[set_action, attack])
        # Setter feet at (0.1, 0.25) → court x = 0.8 → zone 1
        # Attacker feet at (0.875, 0.55) → court x = 7.0 → zone 5
        positions_raw = [
            {"frameNumber": 50, "trackId": setter_tid, "x": 0.1, "y": 0.2,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 80, "trackId": attacker_tid, "x": 0.875, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        stats = annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        assert stats.sets_total == 1
        assert stats.sets_annotated == 1
        assert set_action.set_origin_zone == 1
        assert set_action.set_dest_zone == 5

    def test_set_without_following_attack_is_skipped(self) -> None:
        cal = _FakeCalibrator()
        s = _mk_action(ActionType.SET, 50, player_track_id=7)
        rally = RallyActions(actions=[s])
        stats = annotate_rally_actions(rally, [], [], calibrator=cal)
        assert stats.sets_total == 1
        assert stats.sets_annotated == 0
        assert s.set_origin_zone is None
        assert s.set_dest_zone is None

    def test_action_zone_on_all_actions(self) -> None:
        cal = _FakeCalibrator()
        serve = _mk_action(ActionType.SERVE, 10, player_track_id=1)
        serve.court_side = "near"
        dig = _mk_action(ActionType.DIG, 50, player_track_id=2)
        dig.court_side = "far"
        rally = RallyActions(actions=[serve, dig])
        # Player 1 feet at (0.1, 0.55) → court x=0.8 → near zone 1
        # Player 2 feet at (0.1, 0.55) → court x=0.8 → far zone 5 (inverted)
        positions_raw = [
            {"frameNumber": 10, "trackId": 1, "x": 0.1, "y": 0.5,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 50, "trackId": 2, "x": 0.1, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        assert serve.action_zone == 1  # near side, left → zone 1
        assert dig.action_zone == 5    # far side, camera-left → team-right → zone 5

    def test_to_dict_emits_action_zone(self) -> None:
        a = _mk_action(ActionType.DIG, 10)
        a.action_zone = 3
        assert a.to_dict()["actionZone"] == 3

    def test_to_dict_omits_none_annotations(self) -> None:
        a = _mk_action(ActionType.ATTACK, 10)
        d = a.to_dict()
        assert "attackDirection" not in d
        assert "setOriginZone" not in d
        assert "setDestZone" not in d

    def test_to_dict_emits_annotations_when_set(self) -> None:
        a = _mk_action(ActionType.ATTACK, 10)
        a.attack_direction = "cross"
        s = _mk_action(ActionType.SET, 20)
        s.set_origin_zone = 3
        s.set_dest_zone = 4
        assert a.to_dict()["attackDirection"] == "cross"
        assert s.to_dict()["setOriginZone"] == 3
        assert s.to_dict()["setDestZone"] == 4
