"""Ball contact detection from trajectory inflection points.

Detects moments when a player contacts the ball by analyzing the ball
trajectory for sharp direction changes (inflection points). Each detected
contact is attributed to the nearest player.

This module provides a clean, standalone API for contact detection,
building on the lower-level contact detection in ball_features.py.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_classifier import ContactClassifier
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Minimum confidence to treat a ball position as a real detection.
# VballNet confidence is bimodal: either 0.0 (no detection) or >=0.3 (confident).
_CONFIDENCE_THRESHOLD = 0.3

# Cached default classifier (loaded once from disk on first use)
_default_classifier_cache: dict[str, ContactClassifier | None] = {}


def _get_default_classifier() -> ContactClassifier | None:
    """Load and cache the default contact classifier from disk.

    Returns None if no trained model exists at the default path.
    """
    if "default" not in _default_classifier_cache:
        from rallycut.tracking.contact_classifier import load_contact_classifier

        clf = load_contact_classifier()
        _default_classifier_cache["default"] = clf
        if clf is not None:
            logger.info("Auto-loaded contact classifier from default path")
    return _default_classifier_cache["default"]


@dataclass
class ContactDetectionConfig:
    """Configuration for trajectory-based contact detection."""

    # Velocity peak detection
    min_peak_velocity: float = 0.008  # Min velocity for a contact peak
    min_peak_prominence: float = 0.003  # Min prominence above neighbors
    smoothing_window: int = 5  # Frames for velocity smoothing
    min_peak_distance_frames: int = 12  # Min frames between contacts (~0.4s @ 30fps)

    # Direction change thresholds
    min_direction_change_deg: float = 30.0  # Min angle change to confirm contact
    direction_check_frames: int = 8  # Frames before/after to check direction

    # Inflection detection
    enable_inflection_detection: bool = True
    min_inflection_angle_deg: float = 15.0  # Min angle for inflection candidate
    inflection_check_frames: int = 5  # Frames before/after for inflection check

    # Noise spike filter
    enable_noise_filter: bool = True
    noise_spike_max_jump: float = 0.20  # Max distance to predecessor/successor

    # Player proximity
    player_contact_radius: float = 0.15  # Max distance (normalized) for attribution
    player_search_frames: int = 5  # Search ±N frames for nearby player

    # High-velocity contacts (lenient validation)
    high_velocity_threshold: float = 0.025  # Auto-accept above this velocity

    # Warmup filter: skip candidates in the first N frames (ball tracking warmup)
    warmup_skip_frames: int = 10  # Skip first ~0.33s of ball tracking (avoid warmup noise)

    # Minimum velocity for any candidate (floor for inflection/reversal candidates)
    min_candidate_velocity: float = 0.005  # Below this, direction change is likely noise

    # Parabolic arc breakpoint detection
    enable_parabolic_detection: bool = True
    parabolic_window_frames: int = 12  # Sliding window size for parabolic fit
    parabolic_stride: int = 3  # Window slide step
    parabolic_min_residual: float = 0.015  # Min residual peak to flag breakpoint
    parabolic_min_prominence: float = 0.008  # Min prominence for residual peaks

    # Player-proximity candidate refinement
    enable_proximity_candidates: bool = True
    proximity_search_window: int = 8  # Search ±N frames around each candidate

    # Court position (baselines used by ball_features.py serve detection)
    baseline_y_near: float = 0.82  # Near baseline Y threshold
    baseline_y_far: float = 0.18  # Far baseline Y threshold
    serve_window_frames: int = 60  # Serve must occur in first N frames (~2s)


@dataclass
class Contact:
    """A detected ball contact event."""

    frame: int  # Frame number of contact
    ball_x: float  # Ball X position (normalized 0-1)
    ball_y: float  # Ball Y position (normalized 0-1)
    velocity: float  # Ball velocity at contact (normalized units/frame)
    direction_change_deg: float  # Trajectory direction change in degrees

    # Player attribution
    player_track_id: int = -1  # Track ID of contacting player (-1 = unknown)
    player_distance: float = float("inf")  # Distance to nearest player

    # Court context
    court_side: str = "unknown"  # "near", "far", or "unknown"
    is_at_net: bool = False  # True if contact near net area

    # Validation
    is_validated: bool = False  # True if contact passed validation checks
    confidence: float = 0.0  # Classifier confidence (0-1), set by Phase 3 classifier
    arc_fit_residual: float = 0.0  # Parabolic arc fit residual at this frame

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "ballX": self.ball_x,
            "ballY": self.ball_y,
            "velocity": self.velocity,
            "directionChangeDeg": self.direction_change_deg,
            "playerTrackId": self.player_track_id,
            "playerDistance": self.player_distance if math.isfinite(self.player_distance) else None,
            "courtSide": self.court_side,
            "isAtNet": self.is_at_net,
            "isValidated": self.is_validated,
            "confidence": self.confidence,
            "arcFitResidual": self.arc_fit_residual,
        }


@dataclass
class ContactSequence:
    """A sequence of contacts detected within a rally."""

    contacts: list[Contact] = field(default_factory=list)
    net_y: float = 0.50  # Estimated net Y position
    rally_start_frame: int = 0
    ball_positions: list[BallPosition] = field(default_factory=list)

    @property
    def num_contacts(self) -> int:
        return len(self.contacts)

    @property
    def serve_contact(self) -> Contact | None:
        """Get the first contact (likely serve)."""
        return self.contacts[0] if self.contacts else None

    def contacts_on_side(self, side: str) -> list[Contact]:
        """Get contacts on a specific court side."""
        return [c for c in self.contacts if c.court_side == side]

    def to_dict(self) -> dict:
        return {
            "numContacts": self.num_contacts,
            "netY": self.net_y,
            "rallyStartFrame": self.rally_start_frame,
            "contacts": [c.to_dict() for c in self.contacts],
        }


def _compute_velocities(
    ball_positions: list[BallPosition],
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> dict[int, tuple[float, float, float]]:
    """Compute ball velocity at each frame.

    Returns:
        Dict mapping frame_number to (velocity, vx, vy) in normalized units/frame.
    """
    confident = [
        bp for bp in ball_positions
        if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 2:
        return {}

    velocities: dict[int, tuple[float, float, float]] = {}

    for i in range(1, len(confident)):
        prev = confident[i - 1]
        curr = confident[i]

        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap <= 0 or frame_gap > 5:
            continue

        dx = (curr.x - prev.x) / frame_gap
        dy = (curr.y - prev.y) / frame_gap
        speed = math.sqrt(dx * dx + dy * dy)

        velocities[curr.frame_number] = (speed, dx, dy)

    return velocities


def _smooth_signal(values: list[float], window: int) -> list[float]:
    """Apply moving average smoothing."""
    if len(values) < window:
        return values

    half_w = window // 2
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def _compute_direction_change(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
    check_frames: int = 5,
) -> float:
    """Compute trajectory direction change at a frame in degrees (0-180)."""
    before_frame = None
    after_frame = None

    for offset in range(1, check_frames + 1):
        if before_frame is None and (frame - offset) in ball_by_frame:
            before_frame = frame - offset
        if after_frame is None and (frame + offset) in ball_by_frame:
            after_frame = frame + offset
        if before_frame is not None and after_frame is not None:
            break

    if before_frame is None or after_frame is None:
        return 0.0

    bp_before = ball_by_frame[before_frame]
    bp_at = ball_by_frame.get(frame)
    bp_after = ball_by_frame[after_frame]

    if bp_at is None:
        t = (frame - before_frame) / (after_frame - before_frame)
        at_x = bp_before.x + t * (bp_after.x - bp_before.x)
        at_y = bp_before.y + t * (bp_after.y - bp_before.y)
    else:
        at_x, at_y = bp_at.x, bp_at.y

    vec_in = (at_x - bp_before.x, at_y - bp_before.y)
    vec_out = (bp_after.x - at_x, bp_after.y - at_y)

    mag_in = math.sqrt(vec_in[0] ** 2 + vec_in[1] ** 2)
    mag_out = math.sqrt(vec_out[0] ** 2 + vec_out[1] ** 2)

    if mag_in < 1e-6 or mag_out < 1e-6:
        return 0.0

    dot = vec_in[0] * vec_out[0] + vec_in[1] * vec_out[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag_in * mag_out)))

    return float(np.degrees(np.arccos(cos_angle)))


def _find_nearest_player(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 5,
) -> tuple[int, float, float]:
    """Find nearest player to ball at given frame.

    Returns:
        (track_id, distance, player_center_y). track_id=-1 if no player found.
        player_center_y is the bbox center Y (for court side determination).
    """
    best_track_id = -1
    best_distance = float("inf")
    best_player_y = 0.5

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        # Use upper-quarter of bbox (torso/arms where volleyball contacts happen)
        player_x = p.x
        player_y = p.y - p.height * 0.25

        dist = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

        if dist < best_distance:
            best_distance = dist
            best_track_id = p.track_id
            best_player_y = p.y  # bbox center Y for court side

    return best_track_id, best_distance, best_player_y


def _filter_noise_spikes(
    ball_positions: list[BallPosition],
    max_jump: float,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[BallPosition]:
    """Zero out noise spikes where ball jumps far from both predecessor and successor.

    VballNet produces single-frame false positives that jump to player positions.
    If a position is far from BOTH its predecessor and successor, it's a spike.
    """
    confident = [
        bp for bp in ball_positions if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 3:
        return ball_positions

    # Build set of spike frame numbers
    spike_frames: set[int] = set()

    for i in range(1, len(confident) - 1):
        prev_bp = confident[i - 1]
        curr_bp = confident[i]
        next_bp = confident[i + 1]

        dist_prev = math.sqrt(
            (curr_bp.x - prev_bp.x) ** 2 + (curr_bp.y - prev_bp.y) ** 2
        )
        dist_next = math.sqrt(
            (curr_bp.x - next_bp.x) ** 2 + (curr_bp.y - next_bp.y) ** 2
        )

        if dist_prev > max_jump and dist_next > max_jump:
            spike_frames.add(curr_bp.frame_number)

    if not spike_frames:
        return ball_positions

    logger.debug(f"Noise filter: zeroing {len(spike_frames)} spike frames")

    # Return new list with spike frames zeroed (confidence=0.0)
    return [
        replace(bp, confidence=0.0) if bp.frame_number in spike_frames else bp
        for bp in ball_positions
    ]


def _find_inflection_candidates(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    min_angle_deg: float,
    check_frames: int,
    min_distance_frames: int,
) -> list[int]:
    """Find trajectory inflection points (direction changes) as contact candidates.

    Returns sorted list of candidate frame numbers.
    """
    if len(confident_frames) < 3:
        return []

    # Compute angle at each confident frame
    frame_angles: list[tuple[int, float]] = []
    for frame in confident_frames:
        angle = _compute_direction_change(ball_by_frame, frame, check_frames)
        if angle >= min_angle_deg:
            frame_angles.append((frame, angle))

    if not frame_angles:
        return []

    # Enforce min distance: when two candidates are close, keep largest angle
    frame_angles.sort(key=lambda x: x[0])
    angle_by_frame = dict(frame_angles)
    selected: list[int] = []

    for frame, angle in frame_angles:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif angle > angle_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _find_velocity_reversal_candidates(
    velocities: dict[int, tuple[float, float, float]],
    frames: list[int],
    min_distance_frames: int,
) -> list[int]:
    """Find frames where ball velocity reverses direction.

    A real contact reverses the ball's velocity direction. This catches contacts
    that velocity peaks miss (e.g., a dig that barely changes speed but reverses
    direction).

    Returns:
        Sorted list of candidate frame numbers where velocity reverses.
    """
    if len(frames) < 3:
        return []

    reversal_frames: list[tuple[int, float]] = []

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        if curr_frame - prev_frame > 5:
            continue

        prev_vel = velocities.get(prev_frame)
        curr_vel = velocities.get(curr_frame)
        if prev_vel is None or curr_vel is None:
            continue

        _, prev_vx, prev_vy = prev_vel
        _, curr_vx, curr_vy = curr_vel

        # Dot product of consecutive velocity vectors
        dot = prev_vx * curr_vx + prev_vy * curr_vy
        if dot < 0:
            # Magnitude of the reversal (more negative = sharper reversal)
            reversal_frames.append((curr_frame, -dot))

    if not reversal_frames:
        return []

    # Enforce min distance: keep strongest reversal when close
    reversal_frames.sort(key=lambda x: x[0])
    selected: list[int] = []
    strength_by_frame = dict(reversal_frames)

    for frame, strength in reversal_frames:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif strength > strength_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _find_parabolic_breakpoints(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    window_frames: int = 12,
    stride: int = 3,
    min_residual: float = 0.015,
    min_prominence: float = 0.008,
    min_distance_frames: int = 12,
) -> tuple[list[int], dict[int, float]]:
    """Find contact candidates by detecting transitions between parabolic arcs.

    A volleyball in free flight follows a parabolic arc (gravity). Each contact
    creates a transition between arcs. Fit parabolas to overlapping windows and
    compute per-frame fitting residuals. Residual peaks indicate contacts.

    Key insight: arc apexes (current FP source) lie ON the parabola and have
    LOW residuals. Contacts BREAK the parabola and have HIGH residuals.

    Args:
        ball_by_frame: Ball positions indexed by frame number.
        confident_frames: Sorted frame numbers with confident detections.
        window_frames: Size of sliding window for parabolic fit.
        stride: Step size for sliding window.
        min_residual: Minimum residual to consider as breakpoint.
        min_prominence: Minimum peak prominence.
        min_distance_frames: Minimum distance between breakpoints.

    Returns:
        Tuple of (breakpoint_frames, residual_by_frame) where residual_by_frame
        maps frame numbers to their arc fit residuals.
    """
    from scipy.signal import find_peaks

    if len(confident_frames) < window_frames:
        return [], {}

    # Build arrays of (frame, x, y) for confident positions
    frames_arr = np.array(confident_frames, dtype=np.float64)
    x_arr = np.array([ball_by_frame[f].x for f in confident_frames])
    y_arr = np.array([ball_by_frame[f].y for f in confident_frames])

    # Per-frame residual accumulator (sum of residuals, count of windows)
    residual_sum: dict[int, float] = {}
    residual_count: dict[int, int] = {}

    # Slide window across the trajectory
    for start_idx in range(0, len(confident_frames) - window_frames + 1, stride):
        end_idx = start_idx + window_frames
        win_frames = frames_arr[start_idx:end_idx]
        win_x = x_arr[start_idx:end_idx]
        win_y = y_arr[start_idx:end_idx]

        # Normalize frame numbers to [0, 1] for numerical stability
        t = win_frames - win_frames[0]
        t_range = t[-1]
        if t_range < 1.0:
            continue
        t = t / t_range

        # Fit degree-2 polynomial (parabola) to x(t) and y(t)
        try:
            px = np.polyfit(t, win_x, 2)
            py = np.polyfit(t, win_y, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Compute residuals per frame
        x_pred = np.polyval(px, t)
        y_pred = np.polyval(py, t)
        residuals = np.sqrt((win_x - x_pred) ** 2 + (win_y - y_pred) ** 2)

        for i in range(window_frames):
            frame = confident_frames[start_idx + i]
            residual_sum[frame] = residual_sum.get(frame, 0.0) + float(residuals[i])
            residual_count[frame] = residual_count.get(frame, 0) + 1

    # Average residuals across all windows that included each frame
    residual_by_frame: dict[int, float] = {}
    for frame in confident_frames:
        if frame in residual_sum and residual_count.get(frame, 0) > 0:
            residual_by_frame[frame] = residual_sum[frame] / residual_count[frame]

    if not residual_by_frame:
        return [], {}

    # Extract peaks from the residual signal
    ordered_frames = sorted(residual_by_frame.keys())
    residual_signal = [residual_by_frame[f] for f in ordered_frames]

    if len(residual_signal) < 3:
        return [], residual_by_frame

    peak_indices, _ = find_peaks(
        residual_signal,
        height=min_residual,
        prominence=min_prominence,
        distance=max(1, min_distance_frames // stride),
    )

    breakpoint_frames = [ordered_frames[idx] for idx in peak_indices]

    # Enforce min_distance_frames between breakpoints
    if len(breakpoint_frames) > 1:
        filtered = [breakpoint_frames[0]]
        for frame in breakpoint_frames[1:]:
            if frame - filtered[-1] >= min_distance_frames:
                filtered.append(frame)
            elif residual_by_frame.get(frame, 0) > residual_by_frame.get(filtered[-1], 0):
                filtered[-1] = frame
        breakpoint_frames = filtered

    logger.debug(
        f"Parabolic breakpoints: {len(breakpoint_frames)} from {len(confident_frames)} frames"
    )

    return breakpoint_frames, residual_by_frame


def _compute_acceleration(
    velocities: dict[int, tuple[float, float, float]],
    frame: int,
    window: int = 3,
) -> float:
    """Compute acceleration (velocity change magnitude) near a frame.

    Returns the maximum absolute speed change between consecutive frames
    within ±window of the target frame. Contacts cause sudden speed changes.
    """
    nearby_frames = sorted(f for f in velocities if abs(f - frame) <= window)
    if len(nearby_frames) < 2:
        return 0.0

    max_accel = 0.0
    for i in range(1, len(nearby_frames)):
        prev_f = nearby_frames[i - 1]
        curr_f = nearby_frames[i]
        gap = curr_f - prev_f
        if gap <= 0 or gap > 5:
            continue
        speed_prev = velocities[prev_f][0]
        speed_curr = velocities[curr_f][0]
        accel = abs(speed_curr - speed_prev) / gap
        if accel > max_accel:
            max_accel = accel

    return max_accel


def _compute_trajectory_curvature(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
    window: int = 5,
) -> float:
    """Compute trajectory curvature (inverse radius) near a frame.

    Uses three points: before, at, and after the candidate frame.
    High curvature = sharp bend (contact). Low curvature = smooth arc (free flight).
    Returns curvature as 1/radius, or 0.0 if not enough data.
    """
    before_frame = None
    after_frame = None

    for offset in range(1, window + 1):
        if before_frame is None and (frame - offset) in ball_by_frame:
            before_frame = frame - offset
        if after_frame is None and (frame + offset) in ball_by_frame:
            after_frame = frame + offset
        if before_frame is not None and after_frame is not None:
            break

    if before_frame is None or after_frame is None:
        return 0.0

    bp_at = ball_by_frame.get(frame)
    if bp_at is None:
        return 0.0

    bp_before = ball_by_frame[before_frame]
    bp_after = ball_by_frame[after_frame]

    # Menger curvature: 4*area / (|AB|*|BC|*|CA|)
    ax, ay = bp_before.x, bp_before.y
    bx, by = bp_at.x, bp_at.y
    cx, cy = bp_after.x, bp_after.y

    # Twice the signed area of triangle
    twice_area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))

    ab = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    bc = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    ca = math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)

    denom = ab * bc * ca
    if denom < 1e-12:
        return 0.0

    return twice_area / denom


def _check_net_crossing(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
    net_y: float,
    window: int = 5,
) -> bool:
    """Check if the ball crosses net_y within ±window frames of the given frame."""
    nearby = sorted(f for f in ball_by_frame if abs(f - frame) <= window)
    if len(nearby) < 2:
        return False

    for i in range(1, len(nearby)):
        prev_y = ball_by_frame[nearby[i - 1]].y
        curr_y = ball_by_frame[nearby[i]].y
        if (prev_y < net_y <= curr_y) or (curr_y < net_y <= prev_y):
            return True

    return False


def _find_net_crossing_candidates(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    net_y: float,
    min_distance_frames: int,
) -> list[int]:
    """Find candidates where ball crosses net Y position.

    When the ball Y crosses net_y between consecutive confident frames,
    a contact likely occurred (attack/serve crossing the net). The frame
    just before the crossing is marked as a candidate.

    Returns:
        Sorted list of candidate frame numbers.
    """
    if len(confident_frames) < 2:
        return []

    crossing_frames: list[tuple[int, float]] = []

    for i in range(1, len(confident_frames)):
        prev_frame = confident_frames[i - 1]
        curr_frame = confident_frames[i]

        # Only consider consecutive-ish frames (skip large gaps)
        if curr_frame - prev_frame > 5:
            continue

        prev_y = ball_by_frame[prev_frame].y
        curr_y = ball_by_frame[curr_frame].y

        # Check if net_y is crossed (ball goes from one side to the other)
        if (prev_y < net_y <= curr_y) or (curr_y < net_y <= prev_y):
            # Magnitude of crossing = distance traveled across net
            crossing_mag = abs(curr_y - prev_y)
            # Use the frame just before crossing
            crossing_frames.append((prev_frame, crossing_mag))

    if not crossing_frames:
        return []

    # Enforce min distance: keep strongest crossing when close
    crossing_frames.sort(key=lambda x: x[0])
    selected: list[int] = []
    mag_by_frame = dict(crossing_frames)

    for frame, mag in crossing_frames:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif mag > mag_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _merge_candidates(
    velocity_peak_frames: list[int],
    inflection_frames: list[int],
    min_distance_frames: int,
) -> list[int]:
    """Merge velocity peak and inflection candidates, preferring velocity peaks.

    Velocity peaks are kept as-is. Inflection candidates are added only if
    no existing candidate is within min_distance_frames.
    """
    merged = set(velocity_peak_frames)

    for frame in inflection_frames:
        too_close = any(
            abs(frame - existing) < min_distance_frames
            for existing in merged
        )
        if not too_close:
            merged.add(frame)

    return sorted(merged)


def estimate_net_position(
    ball_positions: list[BallPosition],
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> float:
    """Estimate net Y position from ball trajectory Y extrema midpoint.

    Uses the midpoint between local Y minima (far-side arc peaks) and local
    Y maxima (near-side arc peaks) for more robust estimation than simple
    direction-reversal median.

    Returns:
        Estimated net Y position (normalized 0-1). Returns 0.5 as fallback.
    """
    confident = [
        bp for bp in ball_positions if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 10:
        return 0.5

    # Find local Y minima and maxima using a sliding window
    window = 5
    local_minima_ys: list[float] = []
    local_maxima_ys: list[float] = []

    for i in range(window, len(confident) - window):
        curr_y = confident[i].y
        neighbors_before = [confident[j].y for j in range(i - window, i)]
        neighbors_after = [confident[j].y for j in range(i + 1, i + window + 1)]

        if all(curr_y <= y for y in neighbors_before) and all(
            curr_y <= y for y in neighbors_after
        ):
            local_minima_ys.append(curr_y)
        elif all(curr_y >= y for y in neighbors_before) and all(
            curr_y >= y for y in neighbors_after
        ):
            local_maxima_ys.append(curr_y)

    if local_minima_ys and local_maxima_ys:
        return (float(np.median(local_minima_ys)) + float(np.median(local_maxima_ys))) / 2

    # Fallback: Y range midpoint
    y_values = [bp.y for bp in confident]
    return (min(y_values) + max(y_values)) / 2


def _find_proximity_frame(
    frame: int,
    ball_by_frame: dict[int, BallPosition],
    player_positions: list[PlayerPosition],
    search_window: int = 8,
    player_search_frames: int = 3,
    max_distance: float = 0.15,
) -> int | None:
    """Find frame of minimum player-ball distance within search window.

    Trajectory-based candidates (velocity peaks, inflections) detect the EFFECT
    of a contact, which lags by several frames. The actual contact occurs when
    ball touches the player — the frame of minimum player-ball distance.

    Returns:
        Frame number with minimum distance, or None if no player within
        max_distance in the search window.
    """
    best_frame = None
    best_distance = max_distance

    for offset in range(-search_window, search_window + 1):
        f = frame + offset
        ball = ball_by_frame.get(f)
        if ball is None:
            continue
        _, dist, _ = _find_nearest_player(
            f, ball.x, ball.y, player_positions,
            search_frames=player_search_frames,
        )
        if dist < best_distance:
            best_distance = dist
            best_frame = f

    return best_frame


def _deduplicate_contacts(
    contacts: list[Contact], min_distance: int
) -> list[Contact]:
    """Remove duplicate contacts within min_distance frames, keeping higher confidence."""
    if not contacts:
        return contacts

    # Sort by confidence descending (keep best)
    sorted_contacts = sorted(contacts, key=lambda c: c.confidence, reverse=True)
    result: list[Contact] = []
    used_frames: list[int] = []

    for contact in sorted_contacts:
        if any(abs(contact.frame - f) < min_distance for f in used_frames):
            continue
        result.append(contact)
        used_frames.append(contact.frame)

    # Return in frame order
    return sorted(result, key=lambda c: c.frame)


def detect_contacts(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition] | None = None,
    config: ContactDetectionConfig | None = None,
    net_y: float | None = None,
    frame_count: int | None = None,
    classifier: ContactClassifier | None = None,
    use_classifier: bool = True,
) -> ContactSequence:
    """Detect ball contacts from trajectory inflection points and velocity peaks.

    Algorithm:
    1. Pre-filter noise spikes (single-frame false positives)
    2. Estimate net position (or use provided net_y)
    3. Compute smoothed ball velocity signal
    4. Find velocity peak candidates (local maxima)
    5. Find inflection candidates (direction changes)
    5c. Find parabolic arc breakpoint candidates
    6. Merge candidates (velocity peaks preferred)
    7. Validate each candidate (classifier or hand-tuned gates), attribute player

    Args:
        ball_positions: Ball tracking positions.
        player_positions: Player tracking positions (optional but recommended).
        config: Detection configuration.
        net_y: Explicit net Y position override. If provided, skips auto-estimation.
            Pass court_split_y from player tracking for more reliable court side
            classification.
        frame_count: Total rally frames. If provided, candidates beyond this frame
            are suppressed (post-rally ball pickup/warmdown).
        classifier: Optional trained ContactClassifier. When provided, replaces the
            hand-tuned 3-tier validation gates with learned predictions.
        use_classifier: When True (default) and no explicit classifier is provided,
            auto-loads the default classifier from disk if available. Set to False
            to force hand-tuned validation gates.

    Returns:
        ContactSequence with all detected contacts.
    """
    # Auto-load classifier if not explicitly provided
    if classifier is None and use_classifier:
        classifier = _get_default_classifier()
    from scipy.signal import find_peaks

    cfg = config or ContactDetectionConfig()

    if not ball_positions:
        return ContactSequence()

    # Step 1: Pre-filter noise spikes
    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(
            ball_positions, cfg.noise_spike_max_jump
        )

    # Step 2: Estimate net position
    if net_y is not None:
        estimated_net_y = net_y
    else:
        estimated_net_y = estimate_net_position(ball_positions)

    # Step 3: Compute velocities from filtered positions
    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return ContactSequence(net_y=estimated_net_y)

    # Sort frames and smooth velocity
    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return ContactSequence(net_y=estimated_net_y)

    speeds = [velocities[f][0] for f in frames]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)

    # Step 4: Find velocity peak candidates
    peak_indices, _ = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )
    velocity_peak_frames = [frames[idx] for idx in peak_indices]

    # Index ball positions by frame (only confident ones)
    ball_by_frame: dict[int, BallPosition] = {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    first_frame = frames[0]

    # Step 5a: Find inflection candidates
    confident_frames = sorted(ball_by_frame.keys())
    inflection_frames: list[int] = []
    if cfg.enable_inflection_detection:
        inflection_frames = _find_inflection_candidates(
            ball_by_frame,
            confident_frames,
            min_angle_deg=cfg.min_inflection_angle_deg,
            check_frames=cfg.inflection_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    # Step 5b: Find velocity reversal candidates
    reversal_frames = _find_velocity_reversal_candidates(
        velocities, frames, cfg.min_peak_distance_frames
    )

    # Step 5c: Find parabolic arc breakpoint candidates
    parabolic_frames: list[int] = []
    residual_by_frame: dict[int, float] = {}
    if cfg.enable_parabolic_detection:
        parabolic_frames, residual_by_frame = _find_parabolic_breakpoints(
            ball_by_frame,
            confident_frames,
            window_frames=cfg.parabolic_window_frames,
            stride=cfg.parabolic_stride,
            min_residual=cfg.parabolic_min_residual,
            min_prominence=cfg.parabolic_min_prominence,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    # Step 5d: Find net-crossing candidates
    net_crossing_frames = _find_net_crossing_candidates(
        ball_by_frame, confident_frames, estimated_net_y, cfg.min_peak_distance_frames
    )

    # Step 6: Merge all candidates.
    # Priority: velocity peaks > inflections > reversals > parabolic > net-crossing.
    # _merge_candidates keeps the first arg's frames, adding second arg's only
    # if no existing candidate is within min_distance_frames.
    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    traditional_candidates = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )
    # Add parabolic breakpoints (catches soft touches missed by velocity/inflection)
    with_parabolic = _merge_candidates(
        traditional_candidates, parabolic_frames, cfg.min_peak_distance_frames
    )
    # Add net-crossing candidates (lowest priority — fills gaps from other detectors)
    candidate_frames = _merge_candidates(
        with_parabolic, net_crossing_frames, cfg.min_peak_distance_frames
    )

    if not candidate_frames:
        return ContactSequence(net_y=estimated_net_y)

    # Step 6b: Generate player-proximity candidates
    # Trajectory-based candidates detect the EFFECT of a contact (velocity peak,
    # inflection, etc.) which lags the actual contact by several frames. Proximity
    # candidates are generated at the frame of minimum player-ball distance near
    # each standard candidate — closer to the actual contact moment.
    n_proximity = 0
    if player_positions and cfg.enable_proximity_candidates:
        candidate_set = set(candidate_frames)
        proximity_frames: list[int] = []
        for frame in candidate_frames:
            prox = _find_proximity_frame(
                frame, ball_by_frame, player_positions,
                search_window=cfg.proximity_search_window,
                player_search_frames=cfg.player_search_frames,
                max_distance=cfg.player_contact_radius,
            )
            if prox is not None and prox != frame and prox not in candidate_set:
                proximity_frames.append(prox)
                candidate_set.add(prox)
        if proximity_frames:
            n_proximity = len(proximity_frames)
            candidate_frames = sorted(candidate_set)

    # Build sets for source tracking (used by classifier features)
    velocity_peak_set = set(velocity_peak_frames)
    inflection_set = set(inflection_frames)
    parabolic_set = set(parabolic_frames)

    # Build velocity lookup for any frame
    velocity_lookup = dict(zip(frames, smoothed))

    contacts: list[Contact] = []
    prev_candidate_frame = 0  # Track ALL candidates, not just accepted ones

    for frame in candidate_frames:
        # Skip warmup period (ball tracking produces false detections early)
        if frame - first_frame < cfg.warmup_skip_frames:
            continue

        # Skip post-rally candidates (ball pickup, warmdown)
        if frame_count is not None and frame_count > 0 and frame > frame_count:
            continue

        # Get velocity (may be 0 for inflection-only candidates)
        velocity = velocity_lookup.get(frame, 0.0)

        # Get ball position at candidate frame
        ball = ball_by_frame.get(frame)
        if ball is None:
            for offset in [-1, 1, -2, 2, -3, 3]:
                ball = ball_by_frame.get(frame + offset)
                if ball is not None:
                    break
        if ball is None:
            continue

        # Compute direction change
        direction_change = _compute_direction_change(
            ball_by_frame, frame, cfg.direction_check_frames
        )

        # Find nearest player
        if player_positions:
            track_id, player_dist, _player_y = _find_nearest_player(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_search_frames,
            )
        else:
            track_id = -1
            player_dist = float("inf")

        # Velocity floor: skip very low velocity candidates (tracking noise)
        if velocity < cfg.min_candidate_velocity:
            continue

        # Determine court side from ball position relative to net
        court_side = "far" if ball.y < estimated_net_y else "near"

        # Check if at net
        net_zone = 0.08  # ±8% of screen around net
        is_at_net = abs(ball.y - estimated_net_y) < net_zone

        has_player = player_dist <= cfg.player_contact_radius
        has_direction_change = direction_change >= cfg.min_direction_change_deg
        arc_residual = residual_by_frame.get(frame, 0.0)

        # Compute new trajectory features
        acceleration = _compute_acceleration(velocities, frame, window=3)
        curvature = _compute_trajectory_curvature(ball_by_frame, frame, window=5)
        is_net_cross = _check_net_crossing(
            ball_by_frame, frame, estimated_net_y, window=5
        )

        # Compute frames since last candidate (accepted or rejected).
        # Must match training script semantics (prev_frame tracks all candidates).
        frames_since_last = (
            frame - prev_candidate_frame if prev_candidate_frame > 0 else 0
        )
        prev_candidate_frame = frame

        # Determine which source detected this candidate
        is_vel_peak = frame in velocity_peak_set
        is_infl = frame in inflection_set
        is_para = frame in parabolic_set

        if classifier is not None and classifier.is_trained:
            # Phase 3: Use learned classifier
            from rallycut.tracking.contact_classifier import CandidateFeatures

            features = CandidateFeatures(
                frame=frame,
                velocity=velocity,
                direction_change_deg=direction_change,
                arc_fit_residual=arc_residual,
                acceleration=acceleration,
                trajectory_curvature=curvature,
                player_distance=player_dist,
                has_player=has_player,
                ball_x=ball.x,
                ball_y=ball.y,
                ball_y_relative_net=ball.y - estimated_net_y,
                is_at_net=is_at_net,
                is_net_crossing=is_net_cross,
                frames_since_last=frames_since_last,
                is_velocity_peak=is_vel_peak,
                is_inflection=is_infl,
                is_parabolic=is_para,
            )
            results = classifier.predict([features])
            is_validated, confidence = results[0]
        else:
            # Fallback: Hand-tuned 3-tier validation gates
            # Tier 1: Strong signal — high velocity + direction change (definitive)
            # Tier 2: High velocity + player nearby (serves/spikes without direction
            #         change, e.g. when ball tracking has gaps around the contact)
            # Tier 3: Player confirmed — player nearby + trajectory signal
            is_high_velocity = velocity >= cfg.high_velocity_threshold
            is_strong = is_high_velocity and has_direction_change
            is_fast_with_player = is_high_velocity and has_player
            is_player_confirmed = has_player and (
                has_direction_change or velocity >= cfg.min_peak_velocity
            )
            is_validated = is_strong or is_fast_with_player or is_player_confirmed
            confidence = 0.0

        if not is_validated:
            continue

        contacts.append(Contact(
            frame=frame,
            ball_x=ball.x,
            ball_y=ball.y,
            velocity=velocity,
            direction_change_deg=direction_change,
            player_track_id=track_id,
            player_distance=player_dist,
            court_side=court_side,
            is_at_net=is_at_net,
            is_validated=is_validated,
            confidence=confidence,
            arc_fit_residual=arc_residual,
        ))

    # Deduplicate contacts from proximity + standard candidates at similar frames
    pre_dedup = len(contacts)
    contacts = _deduplicate_contacts(contacts, cfg.min_peak_distance_frames)

    logger.info(
        f"Detected {len(contacts)} contacts "
        f"({len(velocity_peak_frames)} vel peaks + "
        f"{len(inflection_frames)} inflections + "
        f"{len(parabolic_frames)} parabolic + "
        f"{len(net_crossing_frames)} net-cross + "
        f"{n_proximity} proximity → "
        f"{len(candidate_frames)} candidates"
        f"{f', {pre_dedup - len(contacts)} deduped' if pre_dedup > len(contacts) else ''}"
        f", net_y={estimated_net_y:.3f})"
    )

    # Pass confident ball positions for net-crossing detection in action classifier
    confident_positions = [
        bp for bp in ball_positions if bp.confidence >= _CONFIDENCE_THRESHOLD
    ]

    return ContactSequence(
        contacts=contacts,
        net_y=estimated_net_y,
        rally_start_frame=first_frame,
        ball_positions=confident_positions,
    )
