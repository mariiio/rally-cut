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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class ContactDetectionConfig:
    """Configuration for trajectory-based contact detection."""

    # Velocity peak detection
    min_peak_velocity: float = 0.012  # Min velocity for a contact peak
    min_peak_prominence: float = 0.006  # Min prominence above neighbors
    smoothing_window: int = 5  # Frames for velocity smoothing
    min_peak_distance_frames: int = 12  # Min frames between contacts (~0.4s @ 30fps)

    # Direction change thresholds
    min_direction_change_deg: float = 30.0  # Min angle change to confirm contact
    direction_check_frames: int = 5  # Frames before/after to check direction

    # Player proximity
    player_contact_radius: float = 0.15  # Max distance (normalized) for attribution
    player_search_frames: int = 3  # Search ±N frames for nearby player

    # High-velocity contacts (lenient validation)
    high_velocity_threshold: float = 0.025  # Auto-accept above this velocity

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
            "playerDistance": self.player_distance,
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
    confidence_threshold: float = 0.3,
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
) -> tuple[int, float]:
    """Find nearest player to ball at given frame.

    Returns:
        (track_id, distance). track_id=-1 if no player found.
    """
    best_track_id = -1
    best_distance = float("inf")

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        # Use bottom-center of bbox (feet position approximation)
        player_x = p.x
        player_y = p.y + p.height / 2

        dist = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

        if dist < best_distance:
            best_distance = dist
            best_track_id = p.track_id

    return best_track_id, best_distance


def estimate_net_position(
    ball_positions: list[BallPosition],
    confidence_threshold: float = 0.3,
) -> float:
    """Estimate net Y position from ball trajectory direction changes.

    Returns:
        Estimated net Y position (normalized 0-1). Returns 0.5 as fallback.
    """
    confident = [
        bp for bp in ball_positions if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 10:
        return 0.5

    direction_change_ys: list[float] = []

    for i in range(2, len(confident)):
        curr = confident[i]
        prev = confident[i - 1]
        prev_prev = confident[i - 2]

        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap <= 0 or frame_gap > 5:
            continue

        dy = curr.y - prev.y
        prev_dy = prev.y - prev_prev.y

        if (dy > 0 and prev_dy < -0.005) or (dy < 0 and prev_dy > 0.005):
            direction_change_ys.append(prev.y)

    if direction_change_ys:
        return float(np.median(direction_change_ys))

    # Fallback: Y range midpoint
    y_values = [bp.y for bp in confident]
    return (min(y_values) + max(y_values)) / 2


def detect_contacts(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition] | None = None,
    config: ContactDetectionConfig | None = None,
) -> ContactSequence:
    """Detect ball contacts from trajectory inflection points.

    Algorithm:
    1. Compute smoothed ball velocity signal
    2. Find velocity peaks (local maxima) = candidate contacts
    3. Validate each peak:
       - Direction change at peak >= threshold, OR
       - Player is nearby (within contact_radius), OR
       - High velocity (auto-accept)
    4. Attribute each contact to nearest player
    5. Classify court side based on ball Y vs net Y

    Args:
        ball_positions: Ball tracking positions.
        player_positions: Player tracking positions (optional but recommended).
        config: Detection configuration.

    Returns:
        ContactSequence with all detected contacts.
    """
    from scipy.signal import find_peaks

    cfg = config or ContactDetectionConfig()

    if not ball_positions:
        return ContactSequence()

    # Estimate net position
    net_y = estimate_net_position(ball_positions)

    # Compute velocities
    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return ContactSequence(net_y=net_y)

    # Sort frames and smooth velocity
    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return ContactSequence(net_y=net_y)

    speeds = [velocities[f][0] for f in frames]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)

    # Find peaks
    peak_indices, _ = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )

    if len(peak_indices) == 0:
        return ContactSequence(net_y=net_y)

    # Index ball positions by frame
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions}
    first_frame = frames[0]

    contacts: list[Contact] = []

    for idx in peak_indices:
        frame = frames[idx]
        velocity = smoothed[idx]

        # Get ball position at peak
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
            track_id, player_dist = _find_nearest_player(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_search_frames,
            )
        else:
            track_id = -1
            player_dist = float("inf")

        # Validate contact
        has_player = player_dist <= cfg.player_contact_radius
        has_direction_change = direction_change >= cfg.min_direction_change_deg
        is_high_velocity = velocity >= cfg.high_velocity_threshold
        is_validated = has_player or has_direction_change or is_high_velocity

        if not is_validated:
            continue

        # Determine court side
        if ball.y >= cfg.baseline_y_near:
            court_side = "near"
        elif ball.y <= cfg.baseline_y_far:
            court_side = "far"
        elif ball.y < net_y:
            court_side = "far"
        else:
            court_side = "near"

        # Check if at net
        net_zone = 0.08  # ±8% of screen around net
        is_at_net = abs(ball.y - net_y) < net_zone

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
        f"Detected {len(contacts)} contacts from {len(peak_indices)} peaks "
        f"(net_y={net_y:.3f})"
    )

    return ContactSequence(
        contacts=contacts,
        net_y=net_y,
        rally_start_frame=first_frame,
    )
