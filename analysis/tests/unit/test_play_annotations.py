"""Unit tests for ``rallycut.statistics.play_annotations``."""

from __future__ import annotations

import math
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
        just_under = math.tan(math.radians(LINE_MAX_DEG - 0.5))
        dy = 10.0
        dx = just_under * dy
        assert classify_attack_direction_from_xy(0.0, 0.0, dx, dy) == "line"

    def test_cut_boundary(self) -> None:
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

    def test_attack_direction_fallback_ball_trajectory(self) -> None:
        # Attack is the last action — no next-contact player. Fallback
        # uses ball image-space delta. Ball moves mostly horizontally
        # (dx_img large, dy_img small) → maps to a cut-like angle.
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 300, player_track_id=1)
        rally = RallyActions(actions=[atk])
        positions_raw = [
            {"frameNumber": 300, "trackId": 1, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        ball_positions = [
            _ball(300, 0.5, 0.5),
            _ball(315, 0.8, 0.52),  # dt=15, mostly horizontal in image
        ]
        stats = annotate_rally_actions(rally, ball_positions, positions_raw, calibrator=cal)
        assert stats.attacks_annotated == 1
        assert atk.attack_direction is not None
        assert atk.attack_direction != "unknown"

    def test_attack_negative_track_id_skipped(self) -> None:
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=-1)
        rally = RallyActions(actions=[atk])
        stats = annotate_rally_actions(rally, [], [], calibrator=cal)
        assert stats.attacks_total == 1
        assert stats.attacks_annotated == 0
        assert atk.attack_direction is None
        assert atk.action_zone is None

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


# --------------------------------------------------------------------------- #
# Fix A — pre-contact window for airborne attackers.
# --------------------------------------------------------------------------- #


def _kp(x: float, y: float, conf: float = 0.9) -> list[float]:
    return [x, y, conf]


def _pose17(
    ankle_l: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ankle_r: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    """Build a COCO-17 keypoints array with only ankles set."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[15] = list(ankle_l)
    kps[16] = list(ankle_r)
    return kps


class TestPreContactWindow:
    """Fix A — attack feet taken from the takeoff window, not contact frame."""

    def test_pre_contact_window_uses_pre_jump_position(self) -> None:
        # Attacker airborne at contact frame 100: bbox feet at court x=4.0.
        # Pre-contact window (frames 88–97): bbox feet at court x=1.0,
        # which is where the attack truly originated. The annotator
        # should use the pre-contact median, not the airborne frame.
        # Next-contact player is at court x=4.0 → dx from 1.0 is a
        # cross/cut; dx from 4.0 is a line.
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 115, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        positions_raw = [
            # Pre-contact window samples — attacker standing at court
            # x ≈ 1.0 (u=0.125), feet at v=0.175 → court y ≈ 2.8.
            {"frameNumber": 90, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 93, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 96, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            # Contact frame — airborne, feet artifact at court x ≈ 4.0
            # (u=0.5). If the annotator used this frame, the direction
            # would collapse to line.
            {"frameNumber": 100, "trackId": atk_tid, "x": 0.5, "y": 0.125,
             "width": 0.05, "height": 0.1},
            # Defender pre-stance at court x ≈ 4.0, court y ≈ 10.
            {"frameNumber": 112, "trackId": dig_tid, "x": 0.5, "y": 0.575,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 115, "trackId": dig_tid, "x": 0.5, "y": 0.575,
             "width": 0.05, "height": 0.1},
        ]
        stats = annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        assert stats.attacks_annotated == 1
        # dx ≈ 3.0, dy ≈ 7.2 → angle ≈ 23° → cross.
        assert atk.attack_direction == "cross"

    def test_pre_contact_window_falls_back_when_empty(self) -> None:
        # No positions in the pre-contact window: the annotator should
        # fall back to the ±3 frame nearest lookup and still annotate.
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 112, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        positions_raw = [
            # Only the contact frame is available — window [88, 97]
            # is empty, fallback to ±3 finds frame 100 (d=0).
            {"frameNumber": 100, "trackId": atk_tid, "x": 0.5, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 112, "trackId": dig_tid, "x": 0.5125, "y": 0.875,
             "width": 0.05, "height": 0.1},
        ]
        stats = annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        assert stats.attacks_annotated == 1
        # Small dx ≈ 0.1, large dy ≈ 12 → line.
        assert atk.attack_direction == "line"

    def test_pre_contact_window_takes_median(self) -> None:
        # One outlier position in the window shouldn't dominate the
        # feet estimate — the median rejects it.
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 115, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        positions_raw = [
            # Three in-window samples all at court x ≈ 1.0, plus one
            # outlier at u=0.9 (court x=7.2). Median across four is the
            # average of the two middle values: (0.125 + 0.125)/2.
            {"frameNumber": 90, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 93, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 96, "trackId": atk_tid, "x": 0.9, "y": 0.125,
             "width": 0.05, "height": 0.1},
            {"frameNumber": 97, "trackId": atk_tid, "x": 0.125, "y": 0.125,
             "width": 0.05, "height": 0.1},
            # Defender at court x ≈ 4.0, court y ≈ 10.
            {"frameNumber": 115, "trackId": dig_tid, "x": 0.5, "y": 0.575,
             "width": 0.05, "height": 0.1},
        ]
        annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        # Median x = 0.125 → court x = 1.0, landing at 4.0, dx=3.0,
        # dy=7.2 → cross. If the outlier dominated we'd see a
        # different label.
        assert atk.attack_direction == "cross"


# --------------------------------------------------------------------------- #
# Fix B — ankle-based feet with bbox fallback.
# --------------------------------------------------------------------------- #


class TestAnkleFeet:
    def test_attack_uses_ankle_midpoint_when_confident(self) -> None:
        # Attacker bbox-foot is at court x=4.0 (airborne artifact), but
        # ankle midpoint is at court x=1.0. With confident ankles, the
        # annotator should use the ankle value.
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 115, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        # Populate the pre-contact window with bbox far from ankles so
        # we know ankles are what's being used.
        ankle_pose = _pose17(
            ankle_l=(0.12, 0.18, 0.95),
            ankle_r=(0.13, 0.18, 0.95),
        )
        positions_raw = [
            {"frameNumber": 93, "trackId": atk_tid, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1, "keypoints": ankle_pose},
            {"frameNumber": 95, "trackId": atk_tid, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1, "keypoints": ankle_pose},
            {"frameNumber": 97, "trackId": atk_tid, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1, "keypoints": ankle_pose},
            {"frameNumber": 115, "trackId": dig_tid, "x": 0.5, "y": 0.575,
             "width": 0.05, "height": 0.1},
        ]
        annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        # Ankle midpoint u=(0.12+0.13)/2=0.125 → court x=1.0.
        # If bbox were used the attacker court x would be 4.0 (u=0.5).
        # dx from 1.0 → 4.0 is 3.0; direction = cross.
        assert atk.attack_direction == "cross"

    def test_attack_falls_back_to_bbox_when_ankle_low_confidence(self) -> None:
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 115, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        # Ankles with conf 0.1 → below threshold, ignored. bbox is used.
        low_conf_pose = _pose17(
            ankle_l=(0.12, 0.18, 0.1),
            ankle_r=(0.13, 0.18, 0.1),
        )
        positions_raw = [
            {"frameNumber": 95, "trackId": atk_tid, "x": 0.5, "y": 0.125,
             "width": 0.05, "height": 0.1, "keypoints": low_conf_pose},
            {"frameNumber": 115, "trackId": dig_tid, "x": 0.5125, "y": 0.875,
             "width": 0.05, "height": 0.1},
        ]
        annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        # bbox-foot u=0.5 → court x=4.0 (not 1.0 which would be
        # ankles). dx ≈ 0.1 → line.
        assert atk.attack_direction == "line"

    def test_attack_uses_single_confident_ankle(self) -> None:
        cal = _FakeCalibrator()
        atk_tid = 1
        dig_tid = 2
        atk = _mk_action(ActionType.ATTACK, 100, player_track_id=atk_tid)
        dig = _mk_action(ActionType.DIG, 115, player_track_id=dig_tid)
        rally = RallyActions(actions=[atk, dig])
        # Only the left ankle passes threshold. Use it directly (no
        # midpoint).
        pose = _pose17(
            ankle_l=(0.12, 0.18, 0.9),
            ankle_r=(0.5, 0.18, 0.1),
        )
        positions_raw = [
            {"frameNumber": 95, "trackId": atk_tid, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1, "keypoints": pose},
            {"frameNumber": 115, "trackId": dig_tid, "x": 0.5, "y": 0.575,
             "width": 0.05, "height": 0.1},
        ]
        annotate_rally_actions(rally, [], positions_raw, calibrator=cal)
        # Left ankle u=0.12 → court x=0.96; defender at 4.0 → dx=3.04,
        # dy ≈ 7.12 → cross.
        assert atk.attack_direction == "cross"


# --------------------------------------------------------------------------- #
# Fix C — homography-based ball fallback for terminal attacks.
# --------------------------------------------------------------------------- #


class TestBallHomographyFallback:
    def test_fallback_terminal_attack_annotates(self) -> None:
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 300, player_track_id=1)
        rally = RallyActions(actions=[atk])
        positions_raw = [
            {"frameNumber": 300, "trackId": 1, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        ball_positions = [
            _ball(300, 0.5, 0.5),
            _ball(315, 0.8, 0.52),
        ]
        stats = annotate_rally_actions(
            rally, ball_positions, positions_raw, calibrator=cal
        )
        assert stats.attacks_annotated == 1
        assert atk.attack_direction == "cut"

    def test_fallback_projects_ball_through_homography(self) -> None:
        # Direct regression test for Fix C: the old image-scale math
        # passed ball coords to ``dx_img * COURT_WIDTH_M`` without ever
        # calling ``image_to_court`` on the ball. The new code projects
        # both contact and landing ball coords through the homography.
        # A recording calibrator captures every projection call so we
        # can assert the ball values flowed through.
        calls: list[tuple[float, float]] = []

        @dataclass
        class _RecordingCalibrator:
            calibrated: bool = True

            @property
            def is_calibrated(self) -> bool:
                return self.calibrated

            def image_to_court(
                self, pt: tuple[float, float], _w: int, _h: int
            ) -> tuple[float, float]:
                calls.append((float(pt[0]), float(pt[1])))
                return (pt[0] * 8.0, pt[1] * 16.0)

        cal = _RecordingCalibrator()
        atk = _mk_action(ActionType.ATTACK, 300, player_track_id=1)
        rally = RallyActions(actions=[atk])
        positions_raw = [
            {"frameNumber": 300, "trackId": 1, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        ball_positions = [
            _ball(300, 0.42, 0.31),
            _ball(315, 0.77, 0.63),
        ]
        annotate_rally_actions(rally, ball_positions, positions_raw, calibrator=cal)
        assert (0.42, 0.31) in calls
        assert (0.77, 0.63) in calls

    def test_fallback_lowered_min_dt_finds_short_attack_landing(self) -> None:
        # Ball lands dt=10 out (was rejected by previous min_dt=15).
        # New min_dt=10 admits this sample.
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 300, player_track_id=1)
        rally = RallyActions(actions=[atk])
        positions_raw = [
            {"frameNumber": 300, "trackId": 1, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        ball_positions = [
            _ball(300, 0.5, 0.5),
            _ball(310, 0.55, 0.8),  # dt=10
        ]
        stats = annotate_rally_actions(
            rally, ball_positions, positions_raw, calibrator=cal
        )
        assert stats.attacks_annotated == 1
        assert atk.attack_direction is not None
        assert atk.attack_direction != "unknown"

    def test_fallback_caps_max_dt_to_avoid_next_contact(self) -> None:
        # Only ball sample is dt=30 — beyond max_dt=25. Annotator
        # should skip the fallback and leave attack_direction unset.
        cal = _FakeCalibrator()
        atk = _mk_action(ActionType.ATTACK, 300, player_track_id=1)
        rally = RallyActions(actions=[atk])
        positions_raw = [
            {"frameNumber": 300, "trackId": 1, "x": 0.5, "y": 0.5,
             "width": 0.05, "height": 0.1},
        ]
        ball_positions = [
            _ball(300, 0.5, 0.5),
            _ball(330, 0.8, 0.8),  # dt=30, too far
        ]
        stats = annotate_rally_actions(
            rally, ball_positions, positions_raw, calibrator=cal
        )
        assert stats.attacks_total == 1
        assert stats.attacks_annotated == 0
        assert atk.attack_direction is None
