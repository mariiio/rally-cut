"""
Ball feature extraction for rally segment validation and player tracking.

Provides:
- Aggregate ball visibility and trajectory features for rally validation
- Ball phase detection (serve, attack, defense, transition)
- Phase-weighted ball proximity scoring
- Ball reactivity score (player movement correlation with ball)
- Server detection (identify which player served)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class SegmentBallFeatures:
    """Aggregate ball features for a video segment."""

    detection_rate: float  # % frames with ball detected (conf > threshold)
    mean_confidence: float  # Average detection confidence
    trajectory_variance: float  # Ball position variance (high = active play)
    velocity_mean: float  # Mean ball velocity (normalized units/frame)
    velocity_variance: float  # Velocity variance

    def is_valid_rally(
        self,
        min_detection_rate: float = 0.2,
        min_trajectory_variance: float = 0.03,
    ) -> bool:
        """
        Check if features indicate valid rally.

        Rally segments have:
        - High detection rate (ball visible most of the time)
        - High trajectory variance (ball moving around court)

        Dead time has:
        - Low/no detection (ball not visible or held)
        - Low variance (ball stationary or held by player)

        Args:
            min_detection_rate: Minimum % of frames with detected ball.
                Beach uses lower threshold (0.2) due to outdoor conditions.
            min_trajectory_variance: Minimum position variance threshold.

        Returns:
            True if features indicate valid rally.
        """
        return (
            self.detection_rate >= min_detection_rate
            and self.trajectory_variance >= min_trajectory_variance
        )


def compute_ball_features(
    positions: list[BallPosition],
    total_frames: int,
    confidence_threshold: float = 0.5,
) -> SegmentBallFeatures:
    """
    Compute aggregate ball features from tracking positions.

    Args:
        positions: List of ball positions from tracking.
        total_frames: Total number of frames in segment.
        confidence_threshold: Minimum confidence for detection to count.

    Returns:
        SegmentBallFeatures with aggregate statistics.
    """
    if not positions or total_frames == 0:
        return SegmentBallFeatures(
            detection_rate=0.0,
            mean_confidence=0.0,
            trajectory_variance=0.0,
            velocity_mean=0.0,
            velocity_variance=0.0,
        )

    # Filter to confident detections
    confident = [p for p in positions if p.confidence >= confidence_threshold]

    # Detection rate
    detection_rate = len(confident) / total_frames if total_frames > 0 else 0.0

    # Mean confidence (over all positions, not just confident ones)
    mean_confidence = np.mean([p.confidence for p in positions]) if positions else 0.0

    if len(confident) < 2:
        # Not enough detections for trajectory analysis
        return SegmentBallFeatures(
            detection_rate=detection_rate,
            mean_confidence=float(mean_confidence),
            trajectory_variance=0.0,
            velocity_mean=0.0,
            velocity_variance=0.0,
        )

    # Extract positions as arrays
    xs = np.array([p.x for p in confident])
    ys = np.array([p.y for p in confident])

    # Trajectory variance (combined x,y variance)
    trajectory_variance = float(np.var(xs) + np.var(ys))

    # Compute velocities between consecutive frames
    # Note: positions may not be consecutive frames, so velocity is approximate
    velocities = []
    for i in range(1, len(confident)):
        dx = confident[i].x - confident[i - 1].x
        dy = confident[i].y - confident[i - 1].y
        # Frame gap for normalization
        frame_gap = max(1, confident[i].frame_number - confident[i - 1].frame_number)
        velocity = np.sqrt(dx**2 + dy**2) / frame_gap
        velocities.append(velocity)

    if velocities:
        velocity_mean = float(np.mean(velocities))
        velocity_variance = float(np.var(velocities))
    else:
        velocity_mean = 0.0
        velocity_variance = 0.0

    return SegmentBallFeatures(
        detection_rate=detection_rate,
        mean_confidence=float(mean_confidence),
        trajectory_variance=trajectory_variance,
        velocity_mean=velocity_mean,
        velocity_variance=velocity_variance,
    )


def compute_ball_features_subsampled(
    positions: list[BallPosition],
    sample_rate: int = 3,
    confidence_threshold: float = 0.5,
) -> SegmentBallFeatures:
    """
    Compute features from subsampled positions (for faster validation).

    Args:
        positions: List of ball positions from tracking.
        sample_rate: Only use every Nth position.
        confidence_threshold: Minimum confidence for detection to count.

    Returns:
        SegmentBallFeatures with aggregate statistics.
    """
    if sample_rate <= 1:
        return compute_ball_features(positions, len(positions), confidence_threshold)

    # Subsample positions
    sampled = positions[::sample_rate]

    # Estimate total frames from subsampling
    estimated_frames = len(sampled) if sampled else 0

    return compute_ball_features(sampled, estimated_frames, confidence_threshold)


# =============================================================================
# Ball Phase Detection and Player Tracking Features
# =============================================================================


class BallPhase(Enum):
    """Ball trajectory phase during volleyball play (legacy names for API compatibility)."""

    SERVE = "serve"  # Initial hit from baseline
    ATTACK = "attack"  # Spike/hit near net
    DEFENSE = "defense"  # Dig/receive (legacy, use RECEIVE/DIG instead)
    TRANSITION = "transition"  # Set/pass (legacy, use SET instead)
    UNKNOWN = "unknown"


class VolleyballPhase(Enum):
    """Refined phase classification using proper volleyball terminology."""

    SERVE = "serve"      # Initial hit from baseline
    RECEIVE = "receive"  # First contact after serve (pass/dig)
    SET = "set"          # Second contact (setter touch)
    ATTACK = "attack"    # Third contact (spike/hit)
    DIG = "dig"          # Defensive save after attack
    UNKNOWN = "unknown"

    def to_ball_phase(self) -> BallPhase:
        """Convert to legacy BallPhase for API compatibility."""
        mapping = {
            VolleyballPhase.SERVE: BallPhase.SERVE,
            VolleyballPhase.RECEIVE: BallPhase.DEFENSE,  # First touch after serve
            VolleyballPhase.SET: BallPhase.TRANSITION,   # Setting touch
            VolleyballPhase.ATTACK: BallPhase.ATTACK,    # Spike/hit
            VolleyballPhase.DIG: BallPhase.DEFENSE,      # Defensive save
            VolleyballPhase.UNKNOWN: BallPhase.UNKNOWN,
        }
        return mapping.get(self, BallPhase.UNKNOWN)


@dataclass
class ContactDetectionConfig:
    """Configuration for ball contact detection with player validation."""

    # Player proximity validation
    contact_radius: float = 0.15  # Player must be within 15% of frame
    contact_validation_frames: int = 3  # Check ±N frames for nearby player

    # Trajectory validation
    min_direction_change: float = 30.0  # Degrees - trajectory must change direction
    direction_check_frames: int = 5  # Frames before/after to check direction

    # Minimum velocity to be considered a contact
    min_contact_velocity: float = 0.012  # Lower than peak threshold for soft touches


@dataclass
class BallPhaseConfig:
    """Configuration for ball phase detection using peak detection.

    Ball contacts are detected as velocity peaks (local maxima) in the
    smoothed velocity signal. Each peak represents one ball contact.
    """

    # Peak detection parameters
    min_peak_velocity: float = 0.012  # Lower to catch soft touches
    min_peak_prominence: float = 0.006  # Lower for softer contacts
    smoothing_window: int = 5  # Frames for velocity smoothing (moving average)
    min_peak_distance_frames: int = 12  # Longer minimum between contacts (~0.4s at 30fps)

    # Velocity thresholds for classification (after peak is detected)
    high_velocity_threshold: float = 0.025  # Serve or attack
    medium_velocity_threshold: float = 0.015  # Defense/dig

    # Court position thresholds (normalized Y, 0=far baseline, 1=near baseline)
    baseline_y_near: float = 0.82  # Near baseline threshold
    baseline_y_far: float = 0.18  # Far baseline threshold
    net_y_range: tuple[float, float] = (0.38, 0.62)  # Net area

    # Server detection
    serve_window_frames: int = 60  # First 2 seconds at 30fps - serve must occur here
    serve_proximity_radius: float = 0.18  # Player must be this close to ball
    min_serve_confidence: float = 0.4  # Minimum confidence for server ID

    # Contact validation (new)
    contact_config: ContactDetectionConfig | None = None

    # Dynamic net detection
    use_dynamic_net: bool = True  # Estimate net from ball trajectory


@dataclass
class BallPhaseEvent:
    """A detected ball phase event."""

    phase: BallPhase
    frame_start: int
    frame_end: int
    velocity: float  # Max velocity during phase
    ball_position: tuple[float, float]  # Position at phase start


@dataclass
class BallContactEvent:
    """A validated ball contact event with player information."""

    frame: int  # Frame of contact
    velocity: float  # Ball velocity at contact
    ball_x: float  # Ball X position
    ball_y: float  # Ball Y position
    player_track_id: int  # Track ID of player who made contact (-1 if unknown)
    player_distance: float  # Distance to nearest player
    direction_change: float  # Trajectory direction change in degrees
    is_validated: bool  # True if contact passed player proximity validation


@dataclass
class ServerDetectionResult:
    """Result of server detection."""

    track_id: int  # Track ID of detected server (-1 if not detected)
    confidence: float  # Confidence score (0-1)
    serve_frame: int  # Frame when serve was detected
    serve_velocity: float  # Velocity of serve
    is_near_court: bool  # True if server was on near side


class RallyState(Enum):
    """State machine states for rally flow."""

    SERVE_PENDING = "serve_pending"  # Waiting for serve
    SERVE_IN_FLIGHT = "serve_in_flight"  # Ball traveling after serve
    POSSESSION_TEAM_0 = "possession_0"  # Serving team has ball
    POSSESSION_TEAM_1 = "possession_1"  # Receiving team has ball


@dataclass
class RallyFlowModel:
    """State machine for volleyball rally phases.

    Tracks game flow to accurately classify ball contacts:
    - First contact = SERVE (from baseline)
    - First contact after serve = RECEIVE
    - Second contact = SET
    - Third contact = ATTACK
    - After ball crosses net, reset contact count and use DIG for first touch
    """

    state: RallyState = RallyState.SERVE_PENDING
    possession_team: int = 0  # 0 = serving team (near baseline), 1 = receiving (far)
    contact_count: int = 0  # Contacts since possession change (0=first touch)
    net_y: float = 0.50  # Dynamic net position (normalized Y)
    serve_frame: int = -1  # Frame when serve occurred
    last_contact_frame: int = -1  # Frame of last contact
    last_ball_y: float = 0.5  # Last known ball Y position

    def reset(self) -> None:
        """Reset state for new rally."""
        self.state = RallyState.SERVE_PENDING
        self.possession_team = 0
        self.contact_count = 0
        self.serve_frame = -1
        self.last_contact_frame = -1
        self.last_ball_y = 0.5

    def _ball_crossed_net(self, ball_y: float) -> bool:
        """Check if ball has crossed the net based on Y position."""
        if self.possession_team == 0:
            # Serving team is near (high Y), ball crosses when Y < net_y
            return ball_y < self.net_y and self.last_ball_y >= self.net_y
        else:
            # Receiving team is far (low Y), ball crosses when Y > net_y
            return ball_y > self.net_y and self.last_ball_y <= self.net_y

    def process_contact(
        self,
        contact: BallContactEvent,
        is_serve_window: bool = False,
        baseline_y_near: float = 0.82,
        baseline_y_far: float = 0.18,
    ) -> VolleyballPhase:
        """
        Process a ball contact and classify it.

        Args:
            contact: The contact event to classify.
            is_serve_window: True if contact is within serve window (first ~2s).
            baseline_y_near: Y threshold for near baseline.
            baseline_y_far: Y threshold for far baseline.

        Returns:
            VolleyballPhase classification for this contact.
        """
        ball_y = contact.ball_y

        # Check if ball crossed net since last contact
        if self.last_contact_frame >= 0 and self._ball_crossed_net(ball_y):
            # Possession change
            self.possession_team = 1 - self.possession_team
            self.contact_count = 0
            logger.debug(
                f"Ball crossed net at frame {contact.frame}, "
                f"possession -> team {self.possession_team}"
            )

        # Update last ball position
        self.last_ball_y = ball_y
        self.last_contact_frame = contact.frame

        # State machine transitions
        if self.state == RallyState.SERVE_PENDING:
            # First contact in rally
            is_near_baseline = ball_y >= baseline_y_near
            is_far_baseline = ball_y <= baseline_y_far

            if is_serve_window and (is_near_baseline or is_far_baseline):
                # This is the serve
                self.state = RallyState.SERVE_IN_FLIGHT
                self.serve_frame = contact.frame
                self.possession_team = 0 if is_near_baseline else 1
                logger.debug(
                    f"Serve detected at frame {contact.frame}, "
                    f"team {self.possession_team} (Y={ball_y:.2f})"
                )
                return VolleyballPhase.SERVE
            else:
                # Contact before serve window ends but not at baseline
                # Could be a replay or partial rally - treat as unknown
                return VolleyballPhase.UNKNOWN

        elif self.state == RallyState.SERVE_IN_FLIGHT:
            # First contact after serve = RECEIVE
            self.state = (
                RallyState.POSSESSION_TEAM_1
                if self.possession_team == 0
                else RallyState.POSSESSION_TEAM_0
            )
            self.possession_team = 1 - self.possession_team  # Receiving team
            self.contact_count = 1
            logger.debug(
                f"Receive at frame {contact.frame}, possession -> team {self.possession_team}"
            )
            return VolleyballPhase.RECEIVE

        else:
            # Normal possession state
            self.contact_count += 1

            # After ball crosses net, first touch is DIG
            if self.contact_count == 1:
                return VolleyballPhase.DIG

            # Second touch = SET
            if self.contact_count == 2:
                return VolleyballPhase.SET

            # Third touch = ATTACK
            if self.contact_count >= 3:
                # Reset count for next sequence (but keep possession until net cross)
                return VolleyballPhase.ATTACK

            return VolleyballPhase.UNKNOWN


# Phase weights for ball proximity scoring
PHASE_WEIGHTS = {
    BallPhase.SERVE: 0.30,  # Strong signal for server ID
    BallPhase.ATTACK: 0.25,  # Good signal for attacker
    BallPhase.DEFENSE: 0.20,  # Normal (multiple defenders)
    BallPhase.TRANSITION: 0.15,  # Less discriminative
    BallPhase.UNKNOWN: 0.10,  # Minimal weight
}

# Weights for volleyball-specific phases (used internally)
VOLLEYBALL_PHASE_WEIGHTS = {
    VolleyballPhase.SERVE: 0.30,   # Strong signal for server ID
    VolleyballPhase.RECEIVE: 0.20, # First touch after serve
    VolleyballPhase.SET: 0.15,     # Setter touch
    VolleyballPhase.ATTACK: 0.25,  # Spike/hit
    VolleyballPhase.DIG: 0.20,     # Defensive save
    VolleyballPhase.UNKNOWN: 0.10, # Minimal weight
}


def compute_ball_velocities(
    ball_positions: list[BallPosition],
    confidence_threshold: float = 0.40,
) -> dict[int, float]:
    """
    Compute ball velocity at each frame.

    Args:
        ball_positions: Ball positions from tracking.
        confidence_threshold: Minimum confidence for valid positions.

    Returns:
        Dictionary mapping frame_number to velocity (normalized units/frame).
    """
    # Filter to confident positions and sort by frame
    confident = [
        bp for bp in ball_positions
        if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 2:
        return {}

    velocities: dict[int, float] = {}

    for i in range(1, len(confident)):
        prev = confident[i - 1]
        curr = confident[i]

        # Skip if frames are not consecutive (large gap) or same frame
        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap <= 0 or frame_gap > 5:  # Same frame or more than ~0.15s gap at 30fps
            continue

        # Compute velocity (distance / frames)
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        distance = (dx * dx + dy * dy) ** 0.5
        velocity = distance / frame_gap

        velocities[curr.frame_number] = velocity

    return velocities


def _smooth_velocities(
    frames: list[int],
    velocities: dict[int, float],
    window: int,
) -> list[float]:
    """Apply moving average smoothing to velocity signal.

    Args:
        frames: Sorted list of frame numbers.
        velocities: Frame number to velocity mapping.
        window: Smoothing window size.

    Returns:
        Smoothed velocity values aligned with frames.
    """
    raw = [velocities.get(f, 0.0) for f in frames]

    if len(raw) < window:
        return raw

    # Simple moving average
    smoothed = []
    half_w = window // 2

    for i in range(len(raw)):
        start = max(0, i - half_w)
        end = min(len(raw), i + half_w + 1)
        smoothed.append(sum(raw[start:end]) / (end - start))

    return smoothed


def estimate_net_from_ball_trajectory(
    ball_positions: list[BallPosition],
    confidence_threshold: float = 0.4,
) -> float:
    """
    Estimate net Y position from ball trajectory direction changes.

    When ball crosses the net, Y direction typically reverses or changes
    significantly. The net position is estimated as the median Y of these
    direction change points.

    Args:
        ball_positions: Ball positions from tracking.
        confidence_threshold: Minimum confidence for valid positions.

    Returns:
        Estimated net Y position (normalized 0-1). Returns 0.5 if insufficient data.
    """
    # Filter and sort positions
    confident = [
        bp for bp in ball_positions if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 10:
        logger.debug("Not enough ball positions for net estimation, using 0.5")
        return 0.5

    # Find Y direction changes
    direction_change_ys: list[float] = []
    prev_dy = 0.0

    for i in range(2, len(confident)):
        prev = confident[i - 1]
        curr = confident[i]
        prev_prev = confident[i - 2]

        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap <= 0 or frame_gap > 5:
            continue

        dy = curr.y - prev.y
        prev_dy = prev.y - prev_prev.y

        # Direction reversal: sign change or significant slowdown
        if (dy > 0 and prev_dy < -0.005) or (dy < 0 and prev_dy > 0.005):
            direction_change_ys.append(prev.y)

    if direction_change_ys:
        net_y = float(np.median(direction_change_ys))
        logger.debug(
            f"Estimated net Y from {len(direction_change_ys)} direction changes: {net_y:.3f}"
        )
        return net_y

    # Fallback: use Y range midpoint
    y_values = [bp.y for bp in confident]
    net_y = (min(y_values) + max(y_values)) / 2
    logger.debug(f"Net Y from trajectory midpoint: {net_y:.3f}")
    return net_y


def _find_nearest_player(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 3,
) -> tuple[int, float]:
    """
    Find the nearest player to the ball at a given frame.

    Args:
        frame: Frame number to search around.
        ball_x: Ball X position (normalized).
        ball_y: Ball Y position (normalized).
        player_positions: All player positions.
        search_frames: Number of frames to search ±.

    Returns:
        Tuple of (track_id, distance). track_id is -1 if no player found.
    """
    best_track_id = -1
    best_distance = float("inf")

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        # Use bottom-center of bbox (feet position)
        player_x = p.x
        player_y = p.y + p.height / 2

        dist = ((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2) ** 0.5

        if dist < best_distance:
            best_distance = dist
            best_track_id = p.track_id

    return best_track_id, best_distance


def _compute_direction_change(
    ball_positions: list[BallPosition],
    frame: int,
    check_frames: int = 5,
) -> float:
    """
    Compute trajectory direction change at a given frame.

    Args:
        ball_positions: Ball positions (must be sorted by frame).
        frame: Frame to compute direction change at.
        check_frames: Frames before/after to compute direction.

    Returns:
        Direction change in degrees (0-180).
    """
    # Index positions by frame
    by_frame = {bp.frame_number: bp for bp in ball_positions}

    # Find positions before and after
    before_frame = None
    after_frame = None

    for offset in range(1, check_frames + 1):
        if before_frame is None and (frame - offset) in by_frame:
            before_frame = frame - offset
        if after_frame is None and (frame + offset) in by_frame:
            after_frame = frame + offset

        if before_frame is not None and after_frame is not None:
            break

    if before_frame is None or after_frame is None:
        return 0.0

    bp_before = by_frame[before_frame]
    bp_at = by_frame.get(frame)
    bp_after = by_frame[after_frame]

    if bp_at is None:
        # Use interpolated position
        t = (frame - before_frame) / (after_frame - before_frame)
        at_x = bp_before.x + t * (bp_after.x - bp_before.x)
        at_y = bp_before.y + t * (bp_after.y - bp_before.y)
    else:
        at_x, at_y = bp_at.x, bp_at.y

    # Compute direction vectors
    vec_in = (at_x - bp_before.x, at_y - bp_before.y)
    vec_out = (bp_after.x - at_x, bp_after.y - at_y)

    # Normalize and compute angle
    mag_in = (vec_in[0] ** 2 + vec_in[1] ** 2) ** 0.5
    mag_out = (vec_out[0] ** 2 + vec_out[1] ** 2) ** 0.5

    if mag_in < 1e-6 or mag_out < 1e-6:
        return 0.0

    dot = vec_in[0] * vec_out[0] + vec_in[1] * vec_out[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag_in * mag_out)))
    angle_rad = np.arccos(cos_angle)

    return float(np.degrees(angle_rad))


def detect_ball_contacts(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition] | None,
    config: BallPhaseConfig,
) -> list[BallContactEvent]:
    """
    Detect ball contacts with player proximity validation.

    For each velocity peak:
    1. Verify a player is within contact_radius
    2. Verify ball trajectory changes direction
    3. Filter peaks that fail validation

    Args:
        ball_positions: Ball positions from tracking.
        player_positions: Player positions for proximity validation (optional).
        config: Detection configuration.

    Returns:
        List of validated contact events, sorted by frame.
    """
    from scipy.signal import find_peaks

    contact_cfg = config.contact_config or ContactDetectionConfig()
    velocities = compute_ball_velocities(ball_positions)

    if not velocities:
        return []

    # Sort ball positions for direction computation
    sorted_balls = sorted(ball_positions, key=lambda bp: bp.frame_number)

    # Get sorted frames and smooth velocity
    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return []

    smoothed = _smooth_velocities(frames, velocities, config.smoothing_window)

    # Find peaks
    peak_indices, _ = find_peaks(
        smoothed,
        height=config.min_peak_velocity,
        prominence=config.min_peak_prominence,
        distance=config.min_peak_distance_frames,
    )

    if len(peak_indices) == 0:
        return []

    # Index ball positions by frame
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions}

    contacts: list[BallContactEvent] = []

    for idx in peak_indices:
        frame = frames[idx]
        velocity = smoothed[idx]
        ball = ball_by_frame.get(frame)

        if ball is None:
            # Try nearby frames
            for offset in [-1, 1, -2, 2]:
                ball = ball_by_frame.get(frame + offset)
                if ball is not None:
                    break

        if ball is None:
            continue

        # Find nearest player
        if player_positions:
            track_id, player_dist = _find_nearest_player(
                frame, ball.x, ball.y, player_positions,
                search_frames=contact_cfg.contact_validation_frames,
            )
        else:
            track_id = -1
            player_dist = float("inf")

        # Compute direction change
        direction_change = _compute_direction_change(
            sorted_balls, frame, contact_cfg.direction_check_frames
        )

        # Validate contact
        has_player_nearby = player_dist <= contact_cfg.contact_radius
        has_direction_change = direction_change >= contact_cfg.min_direction_change

        # Contact is validated if:
        # - Player is nearby, OR
        # - Significant direction change (ball bounced off something)
        # For high velocity contacts, we're more lenient
        is_high_velocity = velocity >= config.high_velocity_threshold
        is_validated = has_player_nearby or has_direction_change or is_high_velocity

        if not is_validated:
            logger.debug(
                f"Contact at frame {frame} failed validation: "
                f"player_dist={player_dist:.3f}, dir_change={direction_change:.1f}°"
            )
            continue

        contacts.append(BallContactEvent(
            frame=frame,
            velocity=velocity,
            ball_x=ball.x,
            ball_y=ball.y,
            player_track_id=track_id,
            player_distance=player_dist,
            direction_change=direction_change,
            is_validated=is_validated,
        ))

    logger.debug(f"Detected {len(contacts)} validated contacts from {len(peak_indices)} peaks")
    return contacts


def detect_ball_phases(
    ball_positions: list[BallPosition],
    config: BallPhaseConfig | None = None,
) -> list[BallPhaseEvent]:
    """
    Detect ball contact events using velocity peak detection.

    Each velocity peak (local maximum) represents one ball contact.
    Peaks are classified based on velocity magnitude and court position:
    - First peak in serve window + near baseline = SERVE
    - High velocity near net = ATTACK
    - High velocity elsewhere = DEFENSE (hard hit)
    - Medium velocity = DEFENSE (dig/receive)
    - Low velocity = TRANSITION (set/pass)

    Args:
        ball_positions: Ball positions from tracking.
        config: Configuration for phase detection.

    Returns:
        List of detected contact events, one per ball contact.
    """
    from scipy.signal import find_peaks

    cfg = config or BallPhaseConfig()
    velocities = compute_ball_velocities(ball_positions)

    if not velocities:
        return []

    # Get sorted frames and smooth velocity signal
    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return []

    smoothed = _smooth_velocities(frames, velocities, cfg.smoothing_window)

    # Find peaks (local maxima) in smoothed velocity
    peak_indices, properties = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )

    if len(peak_indices) == 0:
        logger.debug("No velocity peaks detected")
        return []

    # Index ball positions by frame for classification
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions}

    # Get the first frame to determine serve window
    first_frame = frames[0]

    events: list[BallPhaseEvent] = []
    serve_detected = False

    for idx in peak_indices:
        frame = frames[idx]
        velocity = smoothed[idx]
        ball = ball_by_frame.get(frame)

        if ball is None:
            # Try nearby frames
            for offset in [-1, 1, -2, 2]:
                ball = ball_by_frame.get(frame + offset)
                if ball is not None:
                    break

        if ball is None:
            continue

        # Classify the contact based on velocity and position
        phase = BallPhase.UNKNOWN
        is_in_serve_window = (frame - first_frame) < cfg.serve_window_frames
        is_near_baseline = ball.y >= cfg.baseline_y_near or ball.y <= cfg.baseline_y_far
        is_near_net = cfg.net_y_range[0] <= ball.y <= cfg.net_y_range[1]

        if velocity >= cfg.high_velocity_threshold:
            # High velocity contact
            if not serve_detected and is_in_serve_window and is_near_baseline:
                phase = BallPhase.SERVE
                serve_detected = True
            elif is_near_net:
                phase = BallPhase.ATTACK
            else:
                # High velocity away from net - could be attack or hard defense
                phase = BallPhase.ATTACK
        elif velocity >= cfg.medium_velocity_threshold:
            # Medium velocity - defense/dig
            phase = BallPhase.DEFENSE
        else:
            # Low velocity - set/pass
            phase = BallPhase.TRANSITION

        # Estimate frame range around peak (±10 frames ~ contact duration)
        frame_start = max(first_frame, frame - 10)
        frame_end = min(frames[-1], frame + 10)

        events.append(BallPhaseEvent(
            phase=phase,
            frame_start=frame_start,
            frame_end=frame_end,
            velocity=velocity,
            ball_position=(ball.x, ball.y),
        ))

    if events:
        phase_counts: dict[str, int] = {}
        for e in events:
            phase_counts[e.phase.value] = phase_counts.get(e.phase.value, 0) + 1
        logger.debug(f"Detected {len(events)} ball contacts: {phase_counts}")

    return events


def detect_ball_phases_v2(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition] | None = None,
    config: BallPhaseConfig | None = None,
) -> list[BallPhaseEvent]:
    """
    Detect ball phases using game-flow-aware state machine.

    This improved algorithm:
    1. Validates contacts with player proximity
    2. Uses volleyball game flow (SERVE → RECEIVE → SET → ATTACK pattern)
    3. Tracks possession changes when ball crosses net
    4. Generates non-overlapping phase segments

    Args:
        ball_positions: Ball positions from tracking.
        player_positions: Player positions for contact validation (optional but recommended).
        config: Configuration for phase detection.

    Returns:
        List of detected phase events, one per contact, non-overlapping.
    """
    cfg = config or BallPhaseConfig()

    if not ball_positions:
        return []

    # Detect net position dynamically
    if cfg.use_dynamic_net:
        net_y = estimate_net_from_ball_trajectory(ball_positions)
    else:
        net_y = (cfg.net_y_range[0] + cfg.net_y_range[1]) / 2

    # Detect validated contacts
    contacts = detect_ball_contacts(ball_positions, player_positions, cfg)

    if not contacts:
        # Fall back to legacy detection if no contacts validated
        logger.debug("No validated contacts, falling back to legacy detection")
        return detect_ball_phases(ball_positions, cfg)

    # Sort contacts by frame
    contacts.sort(key=lambda c: c.frame)

    # Initialize state machine
    flow = RallyFlowModel(net_y=net_y)

    # Get first frame for serve window calculation
    first_frame = contacts[0].frame if contacts else 0

    events: list[BallPhaseEvent] = []

    for i, contact in enumerate(contacts):
        # Check if in serve window
        is_serve_window = (contact.frame - first_frame) < cfg.serve_window_frames

        # Process contact through state machine
        volleyball_phase = flow.process_contact(
            contact,
            is_serve_window=is_serve_window,
            baseline_y_near=cfg.baseline_y_near,
            baseline_y_far=cfg.baseline_y_far,
        )

        # Convert to legacy BallPhase for API compatibility
        ball_phase = volleyball_phase.to_ball_phase()

        # Compute frame range: from this contact to next contact (non-overlapping)
        frame_start = contact.frame
        if i + 1 < len(contacts):
            # End just before next contact
            frame_end = contacts[i + 1].frame - 1
        else:
            # Last contact: extend to last ball position
            last_frame = max(bp.frame_number for bp in ball_positions)
            frame_end = last_frame

        # Ensure valid range
        frame_end = max(frame_start, frame_end)

        events.append(BallPhaseEvent(
            phase=ball_phase,
            frame_start=frame_start,
            frame_end=frame_end,
            velocity=contact.velocity,
            ball_position=(contact.ball_x, contact.ball_y),
        ))

    if events:
        phase_counts: dict[str, int] = {}
        for e in events:
            phase_counts[e.phase.value] = phase_counts.get(e.phase.value, 0) + 1
        logger.info(
            f"Detected {len(events)} ball phases (v2 state machine): {phase_counts}, "
            f"net_y={net_y:.3f}"
        )

    return events


def compute_phase_weighted_proximity(
    track_positions: list[tuple[int, float, float]],  # (frame, x, y)
    ball_positions: list[BallPosition],
    phases: list[BallPhaseEvent],
    proximity_radius: float = 0.20,
) -> float:
    """
    Compute ball proximity score weighted by ball phase.

    Players near the ball during serve/attack phases get higher weight
    than during transition phases.

    Args:
        track_positions: List of (frame, x, y) tuples for a single track.
        ball_positions: Ball positions from tracking.
        phases: Detected ball phases.
        proximity_radius: Radius for considering "near ball".

    Returns:
        Weighted proximity score (0-1).
    """
    if not track_positions or not ball_positions:
        return 0.0

    # Index ball positions and phases by frame
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions}

    # Build frame -> phase mapping
    frame_to_phase: dict[int, BallPhase] = {}
    for phase_event in phases:
        for f in range(phase_event.frame_start, phase_event.frame_end + 1):
            frame_to_phase[f] = phase_event.phase

    weighted_near = 0.0
    total_weight = 0.0

    for frame, px, py in track_positions:
        ball = ball_by_frame.get(frame)
        if ball is None:
            continue

        # Get phase weight
        phase = frame_to_phase.get(frame, BallPhase.UNKNOWN)
        weight = PHASE_WEIGHTS.get(phase, 0.10)

        # Check proximity
        dist = ((px - ball.x) ** 2 + (py - ball.y) ** 2) ** 0.5
        if dist <= proximity_radius:
            weighted_near += weight

        total_weight += weight

    return weighted_near / total_weight if total_weight > 0 else 0.0


def compute_ball_reactivity(
    track_positions: list[tuple[int, float, float]],  # (frame, x, y)
    ball_positions: list[BallPosition],
    min_ball_velocity: float = 0.02,
) -> float:
    """
    Compute ball reactivity score for a track.

    Measures correlation between player movement and ball movement.
    Players react to ball (move when ball moves), non-players don't.

    Args:
        track_positions: List of (frame, x, y) tuples for a single track.
        ball_positions: Ball positions from tracking.
        min_ball_velocity: Minimum ball velocity to consider "ball moving".

    Returns:
        Reactivity score (0-1). Higher = more reactive to ball.
    """
    if len(track_positions) < 3 or len(ball_positions) < 3:
        return 0.0

    # Sort by frame
    track_positions = sorted(track_positions, key=lambda x: x[0])

    # Compute ball velocities
    velocities = compute_ball_velocities(ball_positions)
    if not velocities:
        return 0.0

    # Compute player velocities
    player_velocities: dict[int, float] = {}
    for i in range(1, len(track_positions)):
        prev_frame, prev_x, prev_y = track_positions[i - 1]
        curr_frame, curr_x, curr_y = track_positions[i]

        frame_gap = curr_frame - prev_frame
        if frame_gap <= 0 or frame_gap > 5:
            continue

        dx = curr_x - prev_x
        dy = curr_y - prev_y
        player_velocity = ((dx * dx + dy * dy) ** 0.5) / frame_gap
        player_velocities[curr_frame] = player_velocity

    # Compute correlation between player movement and ball movement
    # When ball moves (high velocity), player should also move
    reactions: list[float] = []

    for frame, ball_vel in velocities.items():
        if ball_vel < min_ball_velocity:
            continue

        # Look for player velocity in nearby frames (reaction window)
        for offset in range(-3, 4):  # -100ms to +100ms at 30fps
            check_frame = frame + offset
            if check_frame in player_velocities:
                player_vel = player_velocities[check_frame]
                # Score: player moves when ball moves
                reaction = min(player_vel / ball_vel, 1.0) if ball_vel > 0 else 0.0
                reactions.append(reaction)
                break

    if not reactions:
        return 0.0

    return float(np.mean(reactions))


def detect_server(
    player_positions: list[PlayerPosition],
    ball_positions: list[BallPosition],
    rally_start_frame: int = 0,
    config: BallPhaseConfig | None = None,
    calibrator: CourtCalibrator | None = None,
) -> ServerDetectionResult:
    """
    Detect which player served the ball.

    Algorithm:
    1. Find first significant velocity peak in serve window (first ~2s)
    2. Find player closest to ball at peak time
    3. Verify player is near baseline (if calibration available)

    Args:
        player_positions: All player positions from tracking.
        ball_positions: Ball positions from tracking.
        rally_start_frame: Frame where rally starts.
        config: Configuration for server detection.
        calibrator: Optional court calibrator for baseline verification.

    Returns:
        ServerDetectionResult with detected server info.
    """
    from scipy.signal import find_peaks

    cfg = config or BallPhaseConfig()

    # No detection possible
    result = ServerDetectionResult(
        track_id=-1,
        confidence=0.0,
        serve_frame=-1,
        serve_velocity=0.0,
        is_near_court=False,
    )

    if not player_positions or not ball_positions:
        return result

    # Compute ball velocities
    velocities = compute_ball_velocities(ball_positions)
    if not velocities:
        return result

    # Get sorted frames and smooth velocity signal
    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return result

    smoothed = _smooth_velocities(frames, velocities, cfg.smoothing_window)

    # Find peaks in serve window
    serve_end_frame = rally_start_frame + cfg.serve_window_frames

    # Find peaks with lower threshold for serve detection
    peak_indices, properties = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence * 0.8,  # Slightly lower for serve
        distance=cfg.min_peak_distance_frames,
    )

    # Find first peak in serve window
    serve_frame = -1
    max_velocity = 0.0

    for idx in peak_indices:
        frame = frames[idx]
        if frame < rally_start_frame or frame > serve_end_frame:
            continue
        # Take the first peak in serve window
        serve_frame = frame
        max_velocity = smoothed[idx]
        break

    if serve_frame < 0:
        logger.debug("No serve velocity peak detected in serve window")
        return result

    # Get ball position at serve frame
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions}
    ball = ball_by_frame.get(serve_frame)
    if ball is None:
        # Try nearby frames
        for offset in range(-2, 3):
            ball = ball_by_frame.get(serve_frame + offset)
            if ball is not None:
                break

    if ball is None:
        logger.debug(f"No ball position at serve frame {serve_frame}")
        return result

    # Find player closest to ball at serve time
    # Look at frames around serve time
    candidates: dict[int, list[float]] = {}  # track_id -> distances

    for p in player_positions:
        if abs(p.frame_number - serve_frame) > 5:  # Within ~0.15s
            continue

        dist = ((p.x - ball.x) ** 2 + (p.y - ball.y) ** 2) ** 0.5

        if p.track_id not in candidates:
            candidates[p.track_id] = []
        candidates[p.track_id].append(dist)

    if not candidates:
        logger.debug("No players found near serve")
        return result

    # Find track with minimum average distance
    best_track = -1
    best_distance = float("inf")

    for track_id, distances in candidates.items():
        avg_dist = sum(distances) / len(distances)
        if avg_dist < best_distance:
            best_distance = avg_dist
            best_track = track_id

    if best_distance > cfg.serve_proximity_radius:
        logger.debug(f"Closest player too far from ball: {best_distance:.3f}")
        return result

    # Compute confidence based on distance
    confidence = 1.0 - (best_distance / cfg.serve_proximity_radius)
    confidence = max(0.0, min(1.0, confidence))

    # Check if player is near baseline (if calibration available)
    is_near_court = ball.y >= cfg.baseline_y_near

    # Verify with calibration if available
    if calibrator is not None and calibrator.is_calibrated:
        # Get player position at serve frame
        for p in player_positions:
            if p.track_id == best_track and abs(p.frame_number - serve_frame) <= 2:
                try:
                    court_point = calibrator.image_to_court((p.x, p.y + p.height / 2), 1, 1)
                    # Check if near baseline (Y close to 0 or 16m)
                    is_near_baseline = court_point[1] < 2.0 or court_point[1] > 14.0
                    if is_near_baseline:
                        confidence = min(confidence + 0.1, 1.0)
                    else:
                        confidence = max(confidence - 0.2, 0.0)
                    is_near_court = court_point[1] > 8.0  # Near court is Y > 8m
                except (RuntimeError, ValueError):
                    pass
                break

    if confidence < cfg.min_serve_confidence:
        logger.debug(f"Server confidence too low: {confidence:.2f}")
        return result

    logger.info(
        f"Detected server: track {best_track} at frame {serve_frame} "
        f"(velocity={max_velocity:.3f}, confidence={confidence:.2f})"
    )

    return ServerDetectionResult(
        track_id=best_track,
        confidence=confidence,
        serve_frame=serve_frame,
        serve_velocity=max_velocity,
        is_near_court=is_near_court,
    )


def compute_ball_zone_score(
    track_positions: list[tuple[int, float, float]],  # (frame, x, y)
    ball_positions: list[BallPosition],
    phases: list[BallPhaseEvent],
) -> float:
    """
    Compute score based on player position relative to ball trajectory zones.

    Players should be in defensive positions during defense phases,
    and near the ball during attack phases.

    Args:
        track_positions: List of (frame, x, y) tuples for a single track.
        ball_positions: Ball positions from tracking.
        phases: Detected ball phases.

    Returns:
        Zone score (0-1). Higher = better court positioning.
    """
    if not track_positions or not phases:
        return 0.0

    # Index positions by frame
    positions_by_frame = {f: (x, y) for f, x, y in track_positions}
    ball_by_frame = {bp.frame_number: bp for bp in ball_positions}

    zone_scores: list[float] = []

    for phase in phases:
        for frame in range(phase.frame_start, phase.frame_end + 1):
            if frame not in positions_by_frame or frame not in ball_by_frame:
                continue

            px, py = positions_by_frame[frame]
            ball = ball_by_frame[frame]

            score = 0.0

            if phase.phase == BallPhase.SERVE:
                # During serve, server should be near baseline
                # Other players should be in ready position
                pass  # Handled by server detection

            elif phase.phase == BallPhase.ATTACK:
                # During attack, players should be watching the attacker
                # Good positioning is being ready to defend
                dist_to_ball = ((px - ball.x) ** 2 + (py - ball.y) ** 2) ** 0.5
                score = 1.0 - min(dist_to_ball / 0.5, 1.0)  # Close is better

            elif phase.phase == BallPhase.DEFENSE:
                # During defense, players spread out to cover court
                # Good positioning is being in defensive zone
                score = 0.5  # Neutral score

            elif phase.phase == BallPhase.TRANSITION:
                # During transition, ball is being set
                # Players should be positioning for attack/defense
                score = 0.5  # Neutral score

            zone_scores.append(score)

    return float(np.mean(zone_scores)) if zone_scores else 0.0
