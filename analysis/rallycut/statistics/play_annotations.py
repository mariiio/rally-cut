"""Play-level annotations for classified rally actions.

Adds two product-visible stats that depend only on the ball trajectory,
court homography, and player positions — not on player attribution:

    1. ``attack_direction`` — ``"line"`` / ``"cross"`` / ``"cut"`` / ``"unknown"``
       per attack contact. Computed from the court-space angle between the
       attack contact and the ball's landing point (first sample past the
       net, or ≥20 frames out, whichever comes first).

    2. ``set_origin_zone`` / ``set_dest_zone`` — integers 1–5 spanning the
       8m court width left→right (team-relative: far-side zones are
       inverted). Origin is the **setter's feet** court-x at the set
       contact frame. Destination is the **attacker's feet** court-x at
       the next attack contact frame.

Both stats are strictly additive: the helpers mutate optional fields on
existing ``ClassifiedAction`` instances. Actions without enough data to
compute a value keep the default ``None``, and ``ClassifiedAction.to_dict``
omits ``None`` fields for backward compatibility with stored production
JSON.

Neither stat depends on player identity (attribution), so both are safe
to ship while the attribution bottleneck remains open.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rallycut.tracking.action_classifier import RallyActions
    from rallycut.tracking.ball_tracker import BallPosition


# Attack direction thresholds (degrees), calibrated from the 2026-04-10
# stats-pack pre-work diagnostic on 479 production attacks across the
# 62-video action-GT eval set (see
# ``analysis/outputs/stats_pack_diagnostic.json``). Produced a
# 66% line / 24% cross / 8% cut distribution on that set.
LINE_MAX_DEG = 15.0
CROSS_MAX_DEG = 40.0

# Beach volleyball court dimensions (meters). Must match
# ``rallycut.court.calibration.COURT_WIDTH/COURT_LENGTH``.
COURT_WIDTH_M = 8.0
COURT_LENGTH_M = 16.0
NET_Y_M = COURT_LENGTH_M / 2.0  # midline perpendicular to net

# Maximum frame radius for nearest-ball / nearest-player lookups.
_NEAREST_RADIUS = 3

# Pre-contact window for ATTACK feet snapshots. Beach attackers are almost
# always airborne at the attack contact frame — their bbox bottom and even
# their ankle keypoints are mid-air, so "y + height/2" misplaces their
# ground position. Sampling their feet in the short takeoff window
# ``[contact-12, contact-3]`` and taking the median gives a much more
# stable estimate of *where the attack originated on the court*. At 30
# FPS this spans ~300 ms before peak contact, which is typical takeoff
# timing for a beach volleyball spike.
_PRE_CONTACT_WINDOW_LOW = 12  # start this many frames before contact
_PRE_CONTACT_WINDOW_HIGH = 3  # stop this many frames before contact

# COCO-17 pose indices for ankles, matching ``PlayerPosition.keypoints``
# which is populated by the YOLO-Pose detector. When both ankles have
# confidence ≥ ``_MIN_ANKLE_CONF`` we use their midpoint as the feet
# estimate instead of the bbox bottom-center, which is biased upward for
# torso-biased boxes.
_COCO_LEFT_ANKLE = 15
_COCO_RIGHT_ANKLE = 16
_MIN_ANKLE_CONF = 0.5

# Terminal-attack ball-trajectory fallback window. The primary attacker
# landing proxy is the next-contact player's feet; when there is no next
# contact (point scored, last action in rally), we fall back to the
# ball's position some frames later projected through the homography.
# Lowered from 15 so short attacks that end quickly still find a landing
# sample, capped at 25 so we don't accidentally pick up the next real
# contact's ball position.
_FALLBACK_MIN_DT = 10
_FALLBACK_MAX_DT = 25


# --------------------------------------------------------------------------- #
# Pure helpers — easy to unit test without the full pipeline.
# --------------------------------------------------------------------------- #


def x_to_zone(x_m: float, num_zones: int = 5, *, far_side: bool = False) -> int:
    """Bin a court-x coordinate in meters into a zone 1..num_zones.

    Clamps values outside the court width to the nearest edge zone so
    small projection errors don't produce out-of-band zone IDs. Zones
    are left→right across the 8m court width from the CAMERA's perspective.

    When ``far_side=True``, the zone is flipped to be from the team's
    perspective: zone 1 = the team's left (camera's right), zone 5 =
    the team's right (camera's left). This matches how volleyball zones
    are conventionally numbered relative to the team, not the observer.
    """
    if num_zones <= 0:
        raise ValueError("num_zones must be positive")
    width = COURT_WIDTH_M / num_zones
    # Add a small epsilon so floating-point representations of exact
    # multiples of ``width`` (e.g., 4.8 = 3 × 1.6) don't round down to the
    # previous zone.
    z = int((x_m + 1e-9) // width) + 1
    if z < 1:
        z = 1
    elif z > num_zones:
        z = num_zones
    if far_side:
        z = num_zones + 1 - z
    return z


def classify_attack_direction_from_xy(
    contact_x_m: float,
    contact_y_m: float,
    landing_x_m: float,
    landing_y_m: float,
) -> str:
    """Classify an attack as line/cross/cut from court-space endpoints.

    ``y`` is the axis perpendicular to the net. ``x`` is across the net
    (court width). An attack with small |dx|/|dy| is a line shot; large
    |dx| relative to |dy| is a cut.
    """
    dx = landing_x_m - contact_x_m
    dy = landing_y_m - contact_y_m
    if abs(dy) < 1e-6:
        return "unknown"
    angle_deg = abs(math.degrees(math.atan2(dx, abs(dy))))
    if angle_deg < LINE_MAX_DEG:
        return "line"
    if angle_deg < CROSS_MAX_DEG:
        return "cross"
    return "cut"


# --------------------------------------------------------------------------- #
# Projection / nearest-frame helpers.
# --------------------------------------------------------------------------- #


def _nearest_ball(
    ball_positions: list[BallPosition], target_frame: int
) -> BallPosition | None:
    best: BallPosition | None = None
    best_d = _NEAREST_RADIUS + 1
    for bp in ball_positions:
        d = abs(bp.frame_number - target_frame)
        if d <= _NEAREST_RADIUS and d < best_d:
            best = bp
            best_d = d
    return best


def _ankle_midpoint_xy(pos: dict) -> tuple[float, float] | None:
    """Image-space ankle midpoint from COCO-17 keypoints, if confident.

    Prefers the midpoint of both ankles when both clear
    ``_MIN_ANKLE_CONF``; falls back to a single confident ankle;
    returns ``None`` when neither ankle is usable or keypoints are
    missing. All coordinates are normalized 0-1 matching
    ``PlayerPosition.keypoints``.
    """
    kps = pos.get("keypoints")
    if not kps or len(kps) <= _COCO_RIGHT_ANKLE:
        return None
    try:
        left = kps[_COCO_LEFT_ANKLE]
        right = kps[_COCO_RIGHT_ANKLE]
        lx, ly, lc = float(left[0]), float(left[1]), float(left[2])
        rx, ry, rc = float(right[0]), float(right[1]), float(right[2])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    left_ok = lc >= _MIN_ANKLE_CONF
    right_ok = rc >= _MIN_ANKLE_CONF
    if left_ok and right_ok:
        return ((lx + rx) / 2.0, (ly + ry) / 2.0)
    if left_ok:
        return (lx, ly)
    if right_ok:
        return (rx, ry)
    return None


def _feet_image_xy(pos: dict) -> tuple[float, float] | None:
    """Image-space feet estimate for a raw position dict.

    Prefers ankle keypoints when confident, else falls back to
    bbox bottom-center ``(x, y + height/2)``. The bbox fallback is still
    correct when the player is standing (takeoff window), and is the
    only signal we have on tracks without pose keypoints — roughly 60%
    of production ``player_tracks`` as of 2026-04-11.
    """
    ankle_xy = _ankle_midpoint_xy(pos)
    if ankle_xy is not None:
        return ankle_xy
    try:
        px = float(pos["x"])
        py = float(pos["y"]) + float(pos.get("height", 0.10)) / 2.0
        return (px, py)
    except (KeyError, TypeError, ValueError):
        return None


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _player_feet_court_xy(
    positions_raw: list[dict],
    track_id: int,
    frame: int,
    calibrator: Any,
    *,
    pre_contact_window: bool = False,
) -> tuple[float, float] | None:
    """Return a player's court (x, y) projected from their FEET position.

    Feet are computed via :func:`_feet_image_xy` — ankle midpoint when
    COCO-17 keypoints are confident, else bbox bottom-center. Projecting
    a point on the court ground plane is required for the planar
    homography to be accurate; projecting bbox CENTER (~1 m above floor)
    gives wildly wrong court positions on most camera angles.

    When ``pre_contact_window=True`` (attack path), aggregate the feet
    position over the takeoff window ``[frame-12, frame-3]`` as a robust
    median. This sidesteps the airborne-attacker problem where the
    player's feet at contact frame are mid-air and don't represent
    their ground position on the court. If no positions exist in the
    window (e.g., sparse tracking), fall back to the usual
    ``±_NEAREST_RADIUS`` nearest-frame lookup.
    """
    best_img: tuple[float, float] | None = None

    if pre_contact_window:
        low = frame - _PRE_CONTACT_WINDOW_LOW
        high = frame - _PRE_CONTACT_WINDOW_HIGH
        xs: list[float] = []
        ys: list[float] = []
        for pp in positions_raw:
            try:
                if int(pp["trackId"]) != track_id:
                    continue
                fno = int(pp["frameNumber"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (low <= fno <= high):
                continue
            pt = _feet_image_xy(pp)
            if pt is not None:
                xs.append(pt[0])
                ys.append(pt[1])
        if xs and ys:
            best_img = (_median(xs), _median(ys))

    if best_img is None:
        best_d = _NEAREST_RADIUS + 1
        for pp in positions_raw:
            try:
                if int(pp["trackId"]) != track_id:
                    continue
                d = abs(int(pp["frameNumber"]) - frame)
            except (KeyError, TypeError, ValueError):
                continue
            if d <= _NEAREST_RADIUS and d < best_d:
                pt = _feet_image_xy(pp)
                if pt is not None:
                    best_img = pt
                    best_d = d

    if best_img is None:
        return None
    try:
        pc = calibrator.image_to_court(best_img, 1, 1)
    except Exception:  # noqa: BLE001
        return None
    return (float(pc[0]), float(pc[1]))


# --------------------------------------------------------------------------- #
# Top-level post-processor.
# --------------------------------------------------------------------------- #


@dataclass
class AnnotationStats:
    """Counters describing what the annotator was able to populate."""

    attacks_total: int = 0
    attacks_annotated: int = 0
    sets_total: int = 0
    sets_annotated: int = 0


def annotate_rally_actions(
    rally_actions: RallyActions,
    ball_positions: list[BallPosition],
    positions_raw: list[dict],
    calibrator: Any,
) -> AnnotationStats:
    """Mutate ``rally_actions.actions`` in place with new play stats.

    Adds ``attack_direction`` on each ATTACK and
    ``set_origin_zone`` / ``set_dest_zone`` on each SET. Silently skips
    actions missing the inputs they need; writes nothing when the
    calibrator is absent or uncalibrated.
    """
    from rallycut.tracking.action_classifier import ActionType  # local import to avoid cycle

    stats = AnnotationStats()
    if calibrator is None or not getattr(calibrator, "is_calibrated", False):
        return stats
    if not rally_actions.actions:
        return stats

    actions_sorted = sorted(
        rally_actions.actions, key=lambda a: a.frame
    )

    # 1. action_zone on ALL actions — player feet court-x, team-relative.
    for a in actions_sorted:
        if a.player_track_id < 0:
            continue
        pc = _player_feet_court_xy(
            positions_raw, a.player_track_id, a.frame, calibrator
        )
        if pc is not None:
            is_far = a.court_side == "far"
            a.action_zone = x_to_zone(pc[0], far_side=is_far)

    # 2. Attack direction — use PLAYER feet positions (ground plane) for
    # both the attacker and the next-contact player. Both queries use
    # the pre-contact window so airborne contact frames don't skew the
    # feet estimate. Fallback to ball homography projection when no
    # next-contact player is available (last action in rally, point
    # scored, etc.).
    for i, a in enumerate(actions_sorted):
        if a.action_type != ActionType.ATTACK:
            continue
        stats.attacks_total += 1
        if a.player_track_id < 0:
            continue
        attacker_court = _player_feet_court_xy(
            positions_raw, a.player_track_id, a.frame, calibrator,
            pre_contact_window=True,
        )
        if attacker_court is None:
            continue
        # Primary: next-contact player's feet (pre-contact window too —
        # the defender's pre-dig stance is a better court proxy than
        # their dive-landing feet).
        landing_court: tuple[float, float] | None = None
        for b in actions_sorted[i + 1 :]:
            # Skip UNKNOWN only — any real action type (serve, dig,
            # block, etc.) can be the first contact after an attack.
            if b.action_type in (ActionType.UNKNOWN,):
                continue
            if b.player_track_id < 0:
                continue
            bc = _player_feet_court_xy(
                positions_raw, b.player_track_id, b.frame, calibrator,
                pre_contact_window=True,
            )
            if bc is not None:
                landing_court = bc
                break
        # Fallback: project ball image positions through the homography.
        # Fires on terminal attacks (last action in rally / point
        # scored). The homography is correct for points on the ground
        # plane; the ball is not on the ground, but projecting both
        # endpoints through the same homography keeps them on the same
        # (biased) surface, so the angle dx/dy is internally consistent.
        # Scaling image-space delta by court meters (the previous
        # implementation) was geometrically wrong under any perspective
        # warp. We deliberately override ``attacker_court`` with the
        # ball CONTACT projection rather than keeping the feet-based
        # value: mixing a ground-plane attacker point with a
        # ball-projected landing point introduces a systematic vertical
        # offset into the angle calculation. Both-from-ball keeps them
        # on the same surface even if that surface is slightly above
        # the ground.
        #
        # Assumes ``ball_positions`` is sorted by ``frame_number`` so
        # the ``break`` on ``dt > _FALLBACK_MAX_DT`` short-circuits
        # correctly; this matches how ``_parse_ball`` materializes the
        # stored ``ball_positions_json`` and how WASB writes frames.
        if landing_court is None:
            contact_bp = _nearest_ball(ball_positions, a.frame)
            if contact_bp is not None:
                landing_bp: BallPosition | None = None
                for bp in ball_positions:
                    dt = bp.frame_number - a.frame
                    if dt < _FALLBACK_MIN_DT:
                        continue
                    if dt > _FALLBACK_MAX_DT:
                        break
                    landing_bp = bp
                    break
                if landing_bp is not None:
                    try:
                        c_pt = calibrator.image_to_court(
                            (float(contact_bp.x), float(contact_bp.y)), 1, 1
                        )
                        l_pt = calibrator.image_to_court(
                            (float(landing_bp.x), float(landing_bp.y)), 1, 1
                        )
                        attacker_court = (float(c_pt[0]), float(c_pt[1]))
                        landing_court = (float(l_pt[0]), float(l_pt[1]))
                    except Exception:  # noqa: BLE001
                        pass
        if landing_court is None:
            continue
        a.attack_direction = classify_attack_direction_from_xy(
            attacker_court[0], attacker_court[1],
            landing_court[0], landing_court[1],
        )
        stats.attacks_annotated += 1

    # 3. Set zones — origin = setter feet court-x, dest = attacker feet
    # court-x. Both use player feet (ground plane), not ball positions.
    # Zones are team-relative (far-side inverted).
    for i, s in enumerate(actions_sorted):
        if s.action_type != ActionType.SET:
            continue
        stats.sets_total += 1
        next_attack = None
        for b in actions_sorted[i + 1 :]:
            if b.action_type == ActionType.ATTACK:
                next_attack = b
                break
        if next_attack is None:
            continue

        is_far = s.court_side == "far"
        origin_zone: int | None = None
        if s.player_track_id >= 0:
            setter_court = _player_feet_court_xy(
                positions_raw, s.player_track_id, s.frame, calibrator
            )
            if setter_court is not None:
                origin_zone = x_to_zone(setter_court[0], far_side=is_far)

        dest_zone: int | None = None
        if next_attack.player_track_id >= 0:
            atk_far = next_attack.court_side == "far"
            attacker_court = _player_feet_court_xy(
                positions_raw, next_attack.player_track_id,
                next_attack.frame, calibrator,
            )
            if attacker_court is not None:
                dest_zone = x_to_zone(attacker_court[0], far_side=atk_far)

        if origin_zone is not None and dest_zone is not None:
            s.set_origin_zone = origin_zone
            s.set_dest_zone = dest_zone
            stats.sets_annotated += 1

    return stats
