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
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Minimum confidence to treat a ball position as a real detection.
# VballNet confidence is bimodal: either 0.0 (no detection) or >=0.3 (confident).
_CONFIDENCE_THRESHOLD = 0.3


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
    player_search_frames: int = 3  # Search ±N frames for nearby player

    # High-velocity contacts (lenient validation)
    high_velocity_threshold: float = 0.025  # Auto-accept above this velocity

    # Warmup filter: skip candidates in the first N frames (ball tracking warmup)
    warmup_skip_frames: int = 20  # Skip first ~0.7s (all GT serves at frame 42+)

    # Court position
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
        }


@dataclass
class ContactSequence:
    """A sequence of contacts detected within a rally."""

    contacts: list[Contact] = field(default_factory=list)
    net_y: float = 0.50  # Estimated net Y position
    rally_start_frame: int = 0

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
    search_frames: int = 3,
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


def detect_contacts(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition] | None = None,
    config: ContactDetectionConfig | None = None,
    net_y: float | None = None,
) -> ContactSequence:
    """Detect ball contacts from trajectory inflection points and velocity peaks.

    Algorithm:
    1. Pre-filter noise spikes (single-frame false positives)
    2. Estimate net position (or use provided net_y)
    3. Compute smoothed ball velocity signal
    4. Find velocity peak candidates (local maxima)
    5. Find inflection candidates (direction changes)
    6. Merge candidates (velocity peaks preferred)
    7. Validate each candidate, attribute player, classify side

    Args:
        ball_positions: Ball tracking positions.
        player_positions: Player tracking positions (optional but recommended).
        config: Detection configuration.
        net_y: Explicit net Y position override. If provided, skips auto-estimation.
            Pass court_split_y from player tracking for more reliable court side
            classification.

    Returns:
        ContactSequence with all detected contacts.
    """
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
    inflection_frames: list[int] = []
    if cfg.enable_inflection_detection:
        confident_frames = sorted(ball_by_frame.keys())
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

    # Step 6: Merge all candidates.
    # Priority: velocity peaks > inflections > reversals.
    # _merge_candidates keeps the first arg's frames, adding second arg's only
    # if no existing candidate is within min_distance_frames.
    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    candidate_frames = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )

    if not candidate_frames:
        return ContactSequence(net_y=estimated_net_y)

    # Build velocity lookup for any frame
    velocity_lookup = dict(zip(frames, smoothed))

    contacts: list[Contact] = []

    for frame in candidate_frames:
        # Skip warmup period (ball tracking produces false detections early)
        if frame - first_frame < cfg.warmup_skip_frames:
            continue

        # Get velocity (may be 0 for inflection-only candidates)
        velocity = velocity_lookup.get(frame, 0.0)

        # Get ball position at candidate frame
        ball = ball_by_frame.get(frame)
        if ball is None:
            for offset in [-1, 1, -2, 2]:
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

        # Validate contact using compound gates to reduce false positives.
        # Tier 1: Strong signal — high velocity + direction change (definitive)
        # Tier 2: High velocity + player nearby (serves/spikes without direction
        #         change, e.g. when ball tracking has gaps around the contact)
        # Tier 3: Player confirmed — player nearby + trajectory signal
        has_player = player_dist <= cfg.player_contact_radius
        has_direction_change = direction_change >= cfg.min_direction_change_deg
        is_high_velocity = velocity >= cfg.high_velocity_threshold
        is_strong = is_high_velocity and has_direction_change
        is_fast_with_player = is_high_velocity and has_player
        is_player_confirmed = has_player and (
            has_direction_change or velocity >= cfg.min_peak_velocity
        )
        is_validated = is_strong or is_fast_with_player or is_player_confirmed

        if not is_validated:
            continue

        # Determine court side from ball position relative to net
        if ball.y >= cfg.baseline_y_near:
            court_side = "near"
        elif ball.y <= cfg.baseline_y_far:
            court_side = "far"
        elif ball.y < estimated_net_y:
            court_side = "far"
        else:
            court_side = "near"

        # Check if at net
        net_zone = 0.08  # ±8% of screen around net
        is_at_net = abs(ball.y - estimated_net_y) < net_zone

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
        ))

    logger.info(
        f"Detected {len(contacts)} contacts "
        f"({len(velocity_peak_frames)} velocity peaks + "
        f"{len(inflection_frames)} inflections → "
        f"{len(candidate_frames)} candidates, net_y={estimated_net_y:.3f})"
    )

    return ContactSequence(
        contacts=contacts,
        net_y=estimated_net_y,
        rally_start_frame=first_frame,
    )
