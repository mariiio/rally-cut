"""Play-level annotations for classified rally actions.

Adds two product-visible stats that depend only on the ball trajectory,
court homography, and player positions — not on player attribution:

    1. ``attack_direction`` — ``"line"`` / ``"cross"`` / ``"cut"`` / ``"unknown"``
       per attack contact. Computed from the court-space angle between the
       attack contact and the ball's landing point (first sample past the
       net, or ≥20 frames out, whichever comes first).

    2. ``set_origin_zone`` / ``set_dest_zone`` — integers 1–5 spanning the
       8m court width left→right. Origin is the **setter's** court-x at
       the set contact frame. Destination is the **ball's** court-x at
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

# Look-ahead cap for locating an attack's landing point. 1.5s at 30fps is
# longer than any plausible physically-realised attack flight.
_LANDING_MAX_FRAMES = 45
# Fallback horizon: if no net crossing is observed, use whatever sample
# sits ≥ this many frames after contact. Ensures we don't emit "unknown"
# on attacks where the ball gets dug before crossing.
_LANDING_FALLBACK_FRAMES = 20

# Maximum frame radius for nearest-ball / nearest-player lookups.
_NEAREST_RADIUS = 3


# --------------------------------------------------------------------------- #
# Pure helpers — easy to unit test without the full pipeline.
# --------------------------------------------------------------------------- #


def x_to_zone(x_m: float, num_zones: int = 5) -> int:
    """Bin a court-x coordinate in meters into a zone 1..num_zones.

    Clamps values outside the court width to the nearest edge zone so
    small projection errors don't produce out-of-band zone IDs. Zones
    are left→right across the 8m court width.
    """
    if num_zones <= 0:
        raise ValueError("num_zones must be positive")
    width = COURT_WIDTH_M / num_zones
    # Add a small epsilon so floating-point representations of exact
    # multiples of ``width`` (e.g., 4.8 = 3 × 1.6) don't round down to the
    # previous zone.
    z = int((x_m + 1e-9) // width) + 1
    if z < 1:
        return 1
    if z > num_zones:
        return num_zones
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


def _ball_court_xy(
    calibrator: Any, bp: BallPosition
) -> tuple[float, float] | None:
    try:
        pt = calibrator.image_to_court((bp.x, bp.y), 1, 1)
    except Exception:  # noqa: BLE001
        return None
    return (float(pt[0]), float(pt[1]))


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


def _find_landing(
    ball_positions: list[BallPosition],
    calibrator: Any,
    contact_frame: int,
    contact_y_court: float,
) -> tuple[float, float] | None:
    """Find the ball's (x, y) in court coords shortly after contact.

    First sample that crosses the net midline, or first sample at least
    ``_LANDING_FALLBACK_FRAMES`` after contact — whichever comes first.
    """
    for bp in ball_positions:
        dt = bp.frame_number - contact_frame
        if dt <= 0:
            continue
        if dt > _LANDING_MAX_FRAMES:
            break
        p = _ball_court_xy(calibrator, bp)
        if p is None:
            continue
        crossed = (
            (contact_y_court < NET_Y_M and p[1] >= NET_Y_M)
            or (contact_y_court > NET_Y_M and p[1] <= NET_Y_M)
        )
        if crossed or dt >= _LANDING_FALLBACK_FRAMES:
            return p
    return None


def _setter_court_x(
    positions_raw: list[dict],
    setter_tid: int,
    set_frame: int,
    calibrator: Any,
) -> float | None:
    """Return the setter's court-x at set_frame via nearest-frame lookup."""
    best_img: tuple[float, float] | None = None
    best_d = _NEAREST_RADIUS + 1
    for pp in positions_raw:
        try:
            if int(pp["trackId"]) != setter_tid:
                continue
            d = abs(int(pp["frameNumber"]) - set_frame)
        except (KeyError, TypeError, ValueError):
            continue
        if d <= _NEAREST_RADIUS and d < best_d:
            try:
                best_img = (float(pp["x"]), float(pp["y"]))
                best_d = d
            except (KeyError, TypeError, ValueError):
                continue
    if best_img is None:
        return None
    try:
        pc = calibrator.image_to_court(best_img, 1, 1)
    except Exception:  # noqa: BLE001
        return None
    return float(pc[0])


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

    # Attack direction.
    for a in actions_sorted:
        if a.action_type != ActionType.ATTACK:
            continue
        stats.attacks_total += 1
        contact_bp = _nearest_ball(ball_positions, a.frame)
        if contact_bp is None:
            continue
        pc = _ball_court_xy(calibrator, contact_bp)
        if pc is None:
            continue
        pl = _find_landing(ball_positions, calibrator, a.frame, pc[1])
        if pl is None:
            continue
        a.attack_direction = classify_attack_direction_from_xy(
            pc[0], pc[1], pl[0], pl[1]
        )
        stats.attacks_annotated += 1

    # Set zones (origin = setter x, dest = ball x at next attack contact).
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

        origin_zone: int | None = None
        if s.player_track_id >= 0:
            setter_x = _setter_court_x(
                positions_raw, s.player_track_id, s.frame, calibrator
            )
            if setter_x is not None:
                origin_zone = x_to_zone(setter_x)

        dest_zone: int | None = None
        atk_bp = _nearest_ball(ball_positions, next_attack.frame)
        if atk_bp is not None:
            pb = _ball_court_xy(calibrator, atk_bp)
            if pb is not None:
                dest_zone = x_to_zone(pb[0])

        if origin_zone is not None and dest_zone is not None:
            s.set_origin_zone = origin_zone
            s.set_dest_zone = dest_zone
            stats.sets_annotated += 1

    return stats
