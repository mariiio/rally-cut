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
        # Attack contact at court (4, 2); ball flies to (4.1, 14). Small dx,
        # big dy → line.
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 100, ball_x=0.5, ball_y=0.125)
        rally = RallyActions(actions=[atk])
        ball_positions = [
            _ball(100, 0.5, 0.125),   # court (4, 2)  contact
            _ball(106, 0.5125, 0.5),  # court (4.1, 8)  at net
            _ball(112, 0.5125, 0.875),  # court (4.1, 14) landing on far side
        ]
        stats = annotate_rally_actions(rally, ball_positions, [], calibrator=cal)
        assert stats.attacks_total == 1
        assert stats.attacks_annotated == 1
        assert atk.attack_direction == "line"

    def test_attack_direction_cut(self) -> None:
        # Attack contact at court (1, 2); ball flies to (7, 6). Large dx
        # relative to moderate dy → cut.
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 200, ball_x=0.125, ball_y=0.125)
        rally = RallyActions(actions=[atk])
        ball_positions = [
            _ball(200, 0.125, 0.125),  # court (1, 2)
            _ball(220, 0.875, 0.375),  # court (7, 6)  — dt=20 triggers fallback
        ]
        stats = annotate_rally_actions(rally, ball_positions, [], calibrator=cal)
        assert stats.attacks_annotated == 1
        assert atk.attack_direction == "cut"

    def test_set_zones_populated(self) -> None:
        cal = _FakeCalibrator()
        setter_tid = 7
        set_action = _mk_action(
            ActionType.SET, 50, player_track_id=setter_tid,
        )
        attack = _mk_action(
            ActionType.ATTACK, 80, player_track_id=9,
        )
        rally = RallyActions(actions=[set_action, attack])
        # Setter at normalized (0.1, 0.2) → court x = 0.8 → zone 1
        positions_raw = [
            {"frameNumber": 50, "trackId": setter_tid, "x": 0.1, "y": 0.2,
             "width": 0.05, "height": 0.1, "confidence": 1.0},
        ]
        ball_positions = [
            # Ball at attack contact: (0.875, 0.5) → court (7, 8) → zone 5
            _ball(80, 0.875, 0.5),
        ]
        stats = annotate_rally_actions(rally, ball_positions, positions_raw, calibrator=cal)
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
