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


def _player_feet_court_xy(
    positions_raw: list[dict],
    track_id: int,
    frame: int,
    calibrator: Any,
) -> tuple[float, float] | None:
    """Return a player's court (x, y) projected from their FEET position.

    Uses the bottom-center of the bounding box ``(x, y + height/2)``
    instead of the bbox center, so the projected point lies on the court
    ground plane and the planar homography is accurate. Projecting the
    bbox center (chest-height ≈ 1 m above court) gives wildly wrong
    court positions on most camera angles — see the 2026-04-10 stats-pack
    session debug dump.
    """
    best_img: tuple[float, float] | None = None
    best_d = _NEAREST_RADIUS + 1
    for pp in positions_raw:
        try:
            if int(pp["trackId"]) != track_id:
                continue
            d = abs(int(pp["frameNumber"]) - frame)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= _NEAREST_RADIUS and d < best_d:
            try:
                px = float(pp["x"])
                py = float(pp["y"]) + float(pp.get("height", 0.10)) / 2.0
                best_img = (px, py)
                best_d = d
            except (KeyError, TypeError, ValueError):
                continue
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
    # both the attacker and the next-contact player. Fallback to ball
    # image-space trajectory when no next-contact player is available
    # (last action in rally, point scored, etc.).
    for i, a in enumerate(actions_sorted):
        if a.action_type != ActionType.ATTACK:
            continue
        stats.attacks_total += 1
        if a.player_track_id < 0:
            continue
        attacker_court = _player_feet_court_xy(
            positions_raw, a.player_track_id, a.frame, calibrator
        )
        if attacker_court is None:
            continue
        # Primary: next-contact player's feet.
        landing_court: tuple[float, float] | None = None
        for b in actions_sorted[i + 1 :]:
            # Skip UNKNOWN only — any real action type (serve, dig,
            # block, etc.) can be the first contact after an attack.
            if b.action_type in (ActionType.UNKNOWN,):
                continue
            if b.player_track_id < 0:
                continue
            bc = _player_feet_court_xy(
                positions_raw, b.player_track_id, b.frame, calibrator
            )
            if bc is not None:
                landing_court = bc
                break
        # Fallback: ball image-space direction (no homography). Fires
        # on terminal attacks (last action in rally / point scored).
        # Systematically biased by perspective warp — treat labels from
        # this path as approximate, not ground truth.
        if landing_court is None:
            contact_bp = _nearest_ball(ball_positions, a.frame)
            if contact_bp is not None:
                landing_bp = None
                for bp in ball_positions:
                    dt = bp.frame_number - a.frame
                    if dt >= 15:
                        landing_bp = bp
                        break
                if landing_bp is not None:
                    # Use image-space delta scaled to court-like axes.
                    # x_img → court_x (width), y_img → court_y (length).
                    dx_img = landing_bp.x - contact_bp.x
                    dy_img = landing_bp.y - contact_bp.y
                    landing_court = (
                        attacker_court[0] + dx_img * COURT_WIDTH_M,
                        attacker_court[1] + dy_img * COURT_LENGTH_M,
                    )
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
