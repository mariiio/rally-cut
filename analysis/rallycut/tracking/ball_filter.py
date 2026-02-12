"""
Temporal filtering for ball tracking.

Two modes: raw mode (default) passes through VballNet positions with segment
pruning + interpolation; Kalman mode applies full Kalman filter pipeline for
smoother trajectories at the cost of detection rate.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class BallFilterConfig:
    """Configuration for ball temporal filtering.

    Default mode (enable_kalman=False): Raw VballNet positions are kept as-is,
    with segment pruning, blip removal, and interpolation applied. Testing
    showed raw positions have higher match rate than Kalman-smoothed output
    because the Kalman filter smooths toward false detections instead of
    rejecting them. Current: 78.6% detection, 47.1% match, 56.6px mean error.

    Kalman mode (enable_kalman=True): Full Kalman filter pipeline with
    Mahalanobis gating, re-acquisition guard, exit detection, and outlier
    removal. Produces smoother trajectories with lower mean error but at the
    cost of detection rate and match rate. Use for visualization overlays
    where smooth trajectories are preferred over coverage.
    """

    # Kalman filter toggle
    # When False (default), skip Kalman filtering and use raw VballNet positions
    # directly with segment pruning + interpolation. This maximizes detection
    # and match rate. When True, apply full Kalman pipeline for smooth trajectories.
    enable_kalman: bool = False

    # Kalman filter parameters (only used when enable_kalman=True)
    process_noise_position: float = 0.001  # Low: ball position is predictable
    process_noise_velocity: float = 0.01  # Higher: velocity changes on hits
    measurement_noise: float = 0.005  # Trust measurements reasonably

    # Confidence thresholds
    min_confidence_for_update: float = 0.3  # Below this, use prediction only

    # Lag compensation for VballNet model bias (only used when enable_kalman=True)
    # Note: Grid search showed lag_frames=0 performs best
    enable_lag_compensation: bool = True
    lag_frames: int = 0  # Frames to extrapolate forward (0 = no extrapolation)

    # Jump detection (only used when enable_kalman=True)
    # Mahalanobis gating uses Kalman innovation covariance for adaptive rejection
    mahalanobis_threshold: float = 5.99
    # Hard velocity limit as absolute backstop (50% of screen per frame)
    max_velocity: float = 0.5

    # Re-acquisition guard (only used when enable_kalman=True)
    reacquisition_threshold: int = 8  # Prediction-only frames before tentative mode
    reacquisition_required: int = 3  # Consistent detections needed to re-acquire
    reacquisition_radius: float = 0.05  # Max spread of consistent detections (5% of screen)

    # Occlusion handling (only used when enable_kalman=True)
    max_occlusion_frames: int = 30  # ~1s at 30fps before losing track

    # Bidirectional smoothing (only used when enable_kalman=True)
    enable_bidirectional: bool = False

    # Interpolation for missing frames
    enable_interpolation: bool = True
    max_interpolation_gap: int = 10  # Max frames to interpolate (larger gaps left empty)
    interpolated_confidence: float = 0.5  # Confidence assigned to interpolated positions

    # Out-of-frame exit detection (only used when enable_kalman=True)
    enable_exit_detection: bool = True
    exit_edge_margin: float = 0.05  # 5% of screen - ball near edge
    exit_opposite_side_margin: float = 0.3  # Must be within 30% of exit edge

    # Trajectory segment pruning (post-processing)
    # VballNet outputs consistent false detections at rally start/end.
    # Pruning splits trajectory at large jumps, discards short fragments,
    # but recovers short segments spatially close to anchor segments
    # (real trajectory fragments between interleaved false positives).
    enable_segment_pruning: bool = True
    segment_jump_threshold: float = 0.20  # 20% of screen to split segments
    min_segment_frames: int = 15  # Segments shorter than this are discarded
    min_output_confidence: float = 0.05  # Drop positions below this confidence

    # Oscillation pruning (detects cluster-based player-locking after ball exits)
    # VballNet can lock onto two players and alternate with high confidence
    # after the ball leaves the frame. The pattern is cluster-based: positions
    # stay near player B for 2-5 frames, jump to player A for 1-2 frames, then
    # back. Detection uses spatial clustering: find two poles (furthest-apart
    # positions) in each window, assign positions to nearest pole, and count
    # transitions between clusters.
    enable_oscillation_pruning: bool = True
    min_oscillation_frames: int = 12  # Sliding window size for cluster transition rate
    oscillation_reversal_rate: float = 0.25  # Cluster transition rate threshold
    oscillation_min_displacement: float = 0.03  # Min pole distance (3% of screen)

    # Exit ghost removal (detects false detections after ball exits frame)
    # When ball approaches screen edge with consistent velocity and then
    # reverses direction, subsequent positions are ghosts from player-locking.
    enable_exit_ghost_removal: bool = True
    exit_edge_zone: float = 0.10  # 10% of screen — zone where exit approach is checked
    exit_approach_frames: int = 3  # Min consecutive frames approaching edge
    exit_min_approach_speed: float = 0.008  # Min per-frame speed toward edge (~0.8% of screen)

    # Outlier removal (removes flickering and edge artifacts)
    # In raw mode, runs after segment pruning to clean within real segments.
    # In Kalman mode, runs before segment pruning to clean Kalman artifacts.
    enable_outlier_removal: bool = True
    edge_margin: float = 0.02  # 2% of screen = ~38px on 1920px
    max_trajectory_deviation: float = 0.08  # 8% of screen = ~154px on 1920px
    min_neighbors_for_outlier: int = 2

    # Trajectory blip removal (catches multi-frame false positives)
    # VballNet can briefly lock onto a player position for 2-5 frames mid-trajectory.
    # Single-frame outlier detection misses these because consecutive false positives
    # validate each other. This step checks each position against distant trajectory
    # context (positions ≥5 frames away) to detect deviations from the overall path.
    enable_blip_removal: bool = True
    blip_context_min_frames: int = 5  # Min frame distance for context neighbors
    blip_max_deviation: float = 0.15  # 15% of screen = ~288px on 1920px
    blip_max_context_gap: int = 30  # Max total context gap (~1s at 30fps)

    # Outlier removal tuning
    outlier_min_speed: float = 0.02  # 2% of screen/frame — below this, skip reversal check

    # Motion energy filter (removes false positives at stationary positions)
    # Real ball in flight creates temporal intensity change. False positives
    # at player positions have low motion energy because players move slowly
    # relative to the ball.
    enable_motion_energy_filter: bool = True
    motion_energy_threshold: float = 0.02  # Below this = suspicious (reduce conf to 0)


class BallTemporalFilter:
    """Ball tracking filter with raw and Kalman modes.

    Raw mode (default): Passes through VballNet positions with segment pruning
    and interpolation. Maximizes detection rate and match rate.

    Kalman mode: Full Kalman filter pipeline with Mahalanobis gating,
    re-acquisition guard, exit detection, and outlier removal.
    State vector: [x, y, vx, vy]. Produces smoother trajectories.
    """

    def __init__(self, config: BallFilterConfig | None = None):
        self.config = config or BallFilterConfig()
        self._state: np.ndarray | None = None  # [x, y, vx, vy]
        self._covariance: np.ndarray | None = None  # 4x4 covariance matrix
        self._frames_since_confident: int = 0
        self._initialized: bool = False

        # Re-acquisition guard state
        self._in_tentative_mode: bool = False
        self._tentative_buffer: list[tuple[float, float, float]] = []  # (x, y, confidence)

        # Exit detection state
        self._exited: bool = False
        self._exit_edge: str | None = None  # "left", "right", "top", "bottom"

        # State transition matrix (constant velocity model)
        # x' = x + vx, y' = y + vy, vx' = vx, vy' = vy
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix (we observe position only)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self._Q = np.diag([
            self.config.process_noise_position,
            self.config.process_noise_position,
            self.config.process_noise_velocity,
            self.config.process_noise_velocity,
        ])

        # Measurement noise covariance
        self._R = np.eye(2) * self.config.measurement_noise

    def reset(self) -> None:
        """Reset filter state for a new video."""
        self._state = None
        self._covariance = None
        self._frames_since_confident = 0
        self._initialized = False
        self._in_tentative_mode = False
        self._tentative_buffer = []
        self._exited = False
        self._exit_edge = None

    def _initialize(self, x: float, y: float) -> None:
        """Initialize filter state from first valid measurement."""
        self._state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        # High initial covariance for position, higher for velocity (unknown)
        self._covariance = np.diag([0.01, 0.01, 0.1, 0.1])
        self._initialized = True
        self._frames_since_confident = 0

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict next state using motion model."""
        assert self._state is not None and self._covariance is not None

        # State prediction: x' = F @ x
        predicted_state = self._F @ self._state

        # Covariance prediction: P' = F @ P @ F.T + Q
        predicted_cov = self._F @ self._covariance @ self._F.T + self._Q

        return predicted_state, predicted_cov

    def _is_valid_measurement(
        self,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        new_x: float,
        new_y: float,
    ) -> bool:
        """Check if measurement is plausible using Mahalanobis distance gating.

        Uses the Kalman filter's innovation covariance for adaptive rejection:
        - When filter is confident (tight P) → small gate → rejects flickering
        - When filter is uncertain (wide P after occlusion) → large gate → accepts fast moves
        Also enforces a hard max_velocity limit as absolute backstop.
        """
        # Hard velocity limit (absolute backstop)
        pred_x, pred_y = predicted_state[0], predicted_state[1]
        distance = np.sqrt((new_x - pred_x) ** 2 + (new_y - pred_y) ** 2)
        if distance > self.config.max_velocity:
            return False

        # Mahalanobis distance gating
        z = np.array([new_x, new_y])
        innovation = z - self._H @ predicted_state
        innov_cov = self._H @ predicted_cov @ self._H.T + self._R
        try:
            innov_cov_inv = np.linalg.inv(innov_cov)
        except np.linalg.LinAlgError:
            # Singular covariance - fall back to distance check only
            return True
        mahalanobis_sq = float(innovation.T @ innov_cov_inv @ innovation)

        return mahalanobis_sq <= self.config.mahalanobis_threshold

    def _check_tentative_consistent(self) -> bool:
        """Check if buffered tentative detections are spatially consistent."""
        if len(self._tentative_buffer) < self.config.reacquisition_required:
            return False

        xs = [p[0] for p in self._tentative_buffer]
        ys = [p[1] for p in self._tentative_buffer]

        # Check if all detections are within radius of each other
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                dist = np.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
                if dist > self.config.reacquisition_radius:
                    return False
        return True

    def _detect_exit(self, x: float, y: float, vx: float, vy: float) -> None:
        """Detect if ball is exiting the frame based on position and velocity.

        Once exit is detected, the state persists until cleared by re-acquisition.
        This prevents false detections from resetting the exit state.
        """
        if not self.config.enable_exit_detection:
            return

        # Don't clear exit state - it persists until re-acquisition
        if self._exited:
            return

        margin = self.config.exit_edge_margin

        # Check each edge: ball near edge AND moving toward it
        if x < margin and vx < 0:
            self._exited = True
            self._exit_edge = "left"
        elif x > (1 - margin) and vx > 0:
            self._exited = True
            self._exit_edge = "right"
        elif y < margin and vy < 0:
            self._exited = True
            self._exit_edge = "top"
        elif y > (1 - margin) and vy > 0:
            self._exited = True
            self._exit_edge = "bottom"

    def _is_suppressed_reacquisition(self, x: float, y: float) -> bool:
        """Check if a re-acquisition should be suppressed due to exit detection.

        After ball exits one side, suppress detections from the opposite side
        (likely false positives from audience/equipment).
        """
        if not self._exited or self._exit_edge is None:
            return False

        margin = self.config.exit_opposite_side_margin

        # Suppress detections from opposite side of where ball exited
        if self._exit_edge == "left" and x > (1 - margin):
            return True
        if self._exit_edge == "right" and x < margin:
            return True
        if self._exit_edge == "top" and y > (1 - margin):
            return True
        if self._exit_edge == "bottom" and y < margin:
            return True

        return False

    def _update(
        self,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman filter update step."""
        # Innovation (measurement residual)
        y = z - self._H @ predicted_state

        # Innovation covariance
        innov_cov = self._H @ predicted_cov @ self._H.T + self._R

        # Kalman gain
        kalman_gain = predicted_cov @ self._H.T @ np.linalg.inv(innov_cov)

        # State update
        updated_state = predicted_state + kalman_gain @ y

        # Covariance update (Joseph form for numerical stability)
        i_kh = np.eye(4) - kalman_gain @ self._H
        updated_cov = i_kh @ predicted_cov @ i_kh.T + kalman_gain @ self._R @ kalman_gain.T

        return updated_state, updated_cov

    def update(self, position: "BallPosition") -> "BallPosition":
        """
        Process a single ball position through the filter.

        Args:
            position: Raw ball position from detector

        Returns:
            Filtered ball position with smoothed coordinates
        """
        x, y, confidence = position.x, position.y, position.confidence
        is_confident = confidence >= self.config.min_confidence_for_update

        # Initialization guard: require consistent detections before first track
        # Prevents a single false positive from starting the filter at the wrong place
        if not self._initialized:
            if is_confident:
                self._tentative_buffer.append((x, y, confidence))
                if self._check_tentative_consistent():
                    mean_x = np.mean([p[0] for p in self._tentative_buffer])
                    mean_y = np.mean([p[1] for p in self._tentative_buffer])
                    self._initialize(float(mean_x), float(mean_y))
                    self._tentative_buffer = []
                    logger.debug(
                        f"Frame {position.frame_number}: Initialized at "
                        f"({mean_x:.3f},{mean_y:.3f}) after "
                        f"{self.config.reacquisition_required} consistent detections"
                    )
                    return self._create_output(position)
            # Not initialized yet, return as-is
            return position

        # Predict step
        predicted_state, predicted_cov = self._predict()

        # After exit: force tentative mode immediately (don't wait for threshold)
        # This prevents false detections from hijacking the track after ball leaves frame
        if self._exited and not self._in_tentative_mode:
            self._in_tentative_mode = True
            self._tentative_buffer = []
            logger.debug(
                f"Frame {position.frame_number}: Entering tentative mode "
                f"(ball exited {self._exit_edge})"
            )

        # Tentative mode: require consistent detections before re-acquiring
        if self._in_tentative_mode and is_confident:
            # Suppress opposite-side re-acquisitions after exit
            if self._is_suppressed_reacquisition(x, y):
                logger.debug(
                    f"Frame {position.frame_number}: Suppressed re-acquisition "
                    f"(exit_edge={self._exit_edge}, pos={x:.3f},{y:.3f})"
                )
                self._state = predicted_state
                self._covariance = predicted_cov
                self._frames_since_confident += 1
                return self._create_output(position)

            # Buffer detection for consistency check
            self._tentative_buffer.append((x, y, confidence))

            if self._check_tentative_consistent():
                # Re-acquire: re-initialize from mean of consistent detections
                mean_x = np.mean([p[0] for p in self._tentative_buffer])
                mean_y = np.mean([p[1] for p in self._tentative_buffer])
                self._initialize(float(mean_x), float(mean_y))
                self._in_tentative_mode = False
                self._tentative_buffer = []
                self._exited = False
                self._exit_edge = None
                logger.debug(
                    f"Frame {position.frame_number}: Re-acquired track at "
                    f"({mean_x:.3f},{mean_y:.3f}) after "
                    f"{self.config.reacquisition_required} consistent detections"
                )
                return self._create_output(position)

            # Not enough consistent detections yet - use prediction only
            self._state = predicted_state
            self._covariance = predicted_cov
            self._frames_since_confident += 1
            return self._create_output(position)

        is_valid = is_confident and self._is_valid_measurement(
            predicted_state, predicted_cov, x, y,
        )

        if is_valid:
            # Measurement update
            z = np.array([x, y])
            self._state, self._covariance = self._update(predicted_state, predicted_cov, z)
            self._frames_since_confident = 0
            self._in_tentative_mode = False
            self._tentative_buffer = []
            # Check for exit after successful update
            self._detect_exit(
                self._state[0], self._state[1], self._state[2], self._state[3],
            )
        else:
            # Prediction only (occlusion or invalid jump)
            self._state = predicted_state
            self._covariance = predicted_cov
            self._frames_since_confident += 1

            # Enter tentative mode after enough prediction-only frames
            if (
                self._frames_since_confident >= self.config.reacquisition_threshold
                and not self._in_tentative_mode
            ):
                self._in_tentative_mode = True
                self._tentative_buffer = []
                logger.debug(
                    f"Frame {position.frame_number}: Entering tentative mode "
                    f"after {self._frames_since_confident} prediction-only frames"
                )

            if not is_confident:
                logger.debug(
                    f"Frame {position.frame_number}: Low confidence ({confidence:.2f}), "
                    f"using prediction only"
                )
            else:
                logger.debug(
                    f"Frame {position.frame_number}: Measurement rejected "
                    f"(pred={predicted_state[0]:.3f},{predicted_state[1]:.3f} "
                    f"meas={x:.3f},{y:.3f})"
                )

        return self._create_output(position)

    def _create_output(self, original: "BallPosition") -> "BallPosition":
        """Create output position with lag compensation and confidence decay."""
        from rallycut.tracking.ball_tracker import BallPosition

        assert self._state is not None

        x, y = self._state[0], self._state[1]
        vx, vy = self._state[2], self._state[3]

        # Apply lag compensation by extrapolating forward
        if self.config.enable_lag_compensation:
            x += vx * self.config.lag_frames
            y += vy * self.config.lag_frames

        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        # Decay confidence during occlusion
        confidence = original.confidence
        if self._frames_since_confident > 0:
            decay = 1.0 - (self._frames_since_confident / self.config.max_occlusion_frames)
            confidence = max(0.0, confidence * decay)

            # If we've been occluded too long, mark as lost
            if self._frames_since_confident >= self.config.max_occlusion_frames:
                confidence = 0.0

        return BallPosition(
            frame_number=original.frame_number,
            x=float(x),
            y=float(y),
            confidence=confidence,
        )

    def _motion_energy_filter(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Remove false positives with low motion energy.

        A real ball in flight creates high temporal intensity change at its
        position (it wasn't there before, or it leaves after). False positive
        detections at player/court positions have low motion energy because
        those regions are relatively static.

        Zeroes the confidence of positions where motion_energy is below threshold.
        """
        from rallycut.tracking.ball_tracker import BallPosition

        threshold = self.config.motion_energy_threshold
        removed = 0
        result = []

        for p in positions:
            if p.confidence > 0 and p.motion_energy > 0 and p.motion_energy < threshold:
                # Low motion energy = likely false positive at static position
                result.append(BallPosition(
                    frame_number=p.frame_number,
                    x=p.x,
                    y=p.y,
                    confidence=0.0,
                    motion_energy=p.motion_energy,
                ))
                removed += 1
            else:
                result.append(p)

        if removed > 0:
            logger.info(f"Motion energy filter: zeroed {removed} low-energy positions")

        return result

    def filter_batch(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """
        Filter a complete list of ball positions.

        When enable_kalman=False (default), applies only segment pruning and
        interpolation to raw positions — preserving VballNet's native accuracy.

        When enable_kalman=True, runs the full Kalman filter pipeline with
        outlier removal for smoother trajectories.

        Args:
            positions: List of raw ball positions from detector

        Returns:
            List of filtered ball positions
        """
        if not positions:
            return []

        # Sort by frame number to ensure temporal order
        sorted_positions = sorted(positions, key=lambda p: p.frame_number)

        if self.config.enable_kalman:
            filtered = self._run_kalman_pipeline(sorted_positions)
        else:
            # Raw mode: keep VballNet positions as-is
            filtered = list(sorted_positions)

        # Track counts for logging
        input_count = len(filtered)
        outlier_count = 0
        pruned_count = 0
        exit_ghost_count = 0
        blip_count = 0
        oscillation_count = 0
        interp_count = 0
        motion_energy_count = 0

        if self.config.enable_kalman:
            # Kalman mode: outlier removal first (cleans Kalman artifacts),
            # then segment pruning
            if self.config.enable_outlier_removal:
                filtered = self._remove_outliers(filtered)
                outlier_count = input_count - len(filtered)

            after_outlier_count = len(filtered)

            if self.config.enable_segment_pruning:
                filtered = self._prune_segments(filtered)
                pruned_count = after_outlier_count - len(filtered)
        else:
            # Raw mode pipeline:
            # 0. motion energy filter (remove FP at static positions)
            # 1. detect exit ghost frame ranges (on raw data, before pruning)
            # 2. segment pruning (splits at jumps, discards short fragments)
            # 3. apply exit ghost removal (remove ghost ranges from pruned data)
            #    Two-phase: detect on raw to see edge-approach evidence that
            #    segment pruning would discard, apply to pruned to avoid cascade.
            # 4. oscillation pruning (trims A→B→A→B tails from player-locking)
            # 5. outlier removal (cleans flickering within real segments)
            # 6. re-prune (after outlier removal may fragment segments)
            if self.config.enable_motion_energy_filter:
                before_me = sum(1 for p in filtered if p.confidence > 0)
                filtered = self._motion_energy_filter(filtered)
                after_me = sum(1 for p in filtered if p.confidence > 0)
                motion_energy_count = before_me - after_me

            ghost_ranges: list[tuple[int, int]] = []
            if self.config.enable_exit_ghost_removal:
                ghost_ranges = self._detect_exit_ghost_ranges(filtered)

            if self.config.enable_segment_pruning:
                filtered = self._prune_segments(
                    filtered, ghost_ranges=ghost_ranges
                )
                pruned_count = input_count - len(filtered)

            if ghost_ranges:
                before_exit = len(filtered)
                ghost_frames: set[int] = set()
                for start, end in ghost_ranges:
                    for p in filtered:
                        if start <= p.frame_number <= end:
                            ghost_frames.add(p.frame_number)
                if ghost_frames:
                    filtered = [
                        p for p in filtered if p.frame_number not in ghost_frames
                    ]
                    exit_ghost_count = before_exit - len(filtered)
                    logger.info(
                        f"Exit ghost removal: removed {exit_ghost_count} "
                        f"ghost positions from pruned trajectory"
                    )

            after_prune_count = len(filtered)

            if self.config.enable_oscillation_pruning:
                filtered = self._prune_oscillating(filtered)
                oscillation_count = after_prune_count - len(filtered)

            after_oscillation_count = len(filtered)

            if self.config.enable_outlier_removal:
                filtered = self._remove_outliers(filtered)
                outlier_count = after_oscillation_count - len(filtered)

            if self.config.enable_blip_removal:
                before_blip = len(filtered)
                filtered = self._remove_trajectory_blips(filtered)
                blip_count = before_blip - len(filtered)

            # Re-prune after outlier/blip removal: removal can fragment segments,
            # exposing short false sub-segments and hovering patterns that were hidden
            # inside longer segments during the first pass.
            if outlier_count > 0 or blip_count > 0:
                before_reprune = len(filtered)
                if self.config.enable_oscillation_pruning:
                    filtered = self._prune_oscillating(filtered)
                if self.config.enable_segment_pruning:
                    filtered = self._prune_segments(filtered)
                reprune_count = before_reprune - len(filtered)
                if reprune_count > 0:
                    pruned_count += reprune_count

        # Interpolation: fill small gaps
        before_interp_count = len(filtered)
        if self.config.enable_interpolation:
            filtered = self._interpolate_missing(filtered)
            interp_count = len(filtered) - before_interp_count

        # Log summary
        if filtered:
            mode = "kalman" if self.config.enable_kalman else "raw"
            parts = [f"Ball filter ({mode}): {input_count} positions"]
            if motion_energy_count > 0:
                parts.append(f"-{motion_energy_count} low-energy")
            if outlier_count > 0:
                parts.append(f"-{outlier_count} outliers")
            if pruned_count > 0:
                parts.append(f"-{pruned_count} pruned")
            if exit_ghost_count > 0:
                parts.append(f"-{exit_ghost_count} exit ghosts")
            if blip_count > 0:
                parts.append(f"-{blip_count} blips")
            if oscillation_count > 0:
                parts.append(f"-{oscillation_count} oscillating")
            if interp_count > 0:
                parts.append(f"+{interp_count} interpolated")
            logger.info(", ".join(parts))

        return filtered

    def _run_kalman_pipeline(
        self,
        sorted_positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Run full Kalman filter pipeline (forward pass + optional bidirectional).

        Args:
            sorted_positions: Positions sorted by frame number.

        Returns:
            Kalman-filtered positions.
        """
        self.reset()

        filtered = []
        for pos in sorted_positions:
            filtered_pos = self.update(pos)
            filtered.append(filtered_pos)

        if self.config.enable_bidirectional and len(filtered) > 1:
            filtered = self._smooth_bidirectional(sorted_positions, filtered)

        return filtered

    def _smooth_bidirectional(
        self,
        raw_positions: list["BallPosition"],
        forward_filtered: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Apply RTS (Rauch-Tung-Striebel) smoother for bidirectional smoothing.

        This combines forward and backward Kalman filter passes to produce
        optimal zero-lag estimates. Only suitable for offline batch processing.

        The algorithm:
        1. Forward pass already done in _run_kalman_pipeline
        2. Backward pass to compute smoothed estimates

        Args:
            raw_positions: Original raw positions (sorted by frame)
            forward_filtered: Results from forward pass

        Returns:
            Smoothed ball positions with zero lag
        """
        from rallycut.tracking.ball_tracker import BallPosition

        if len(forward_filtered) < 2:
            return forward_filtered

        # Re-run forward pass storing all states and covariances
        self.reset()
        n = len(raw_positions)

        # Storage for forward pass
        forward_states: list[np.ndarray] = []
        forward_covs: list[np.ndarray] = []
        predicted_states: list[np.ndarray] = []
        predicted_covs: list[np.ndarray] = []

        for pos in raw_positions:
            x, y, confidence = pos.x, pos.y, pos.confidence

            if not self._initialized:
                if confidence >= self.config.min_confidence_for_update:
                    self._initialize(x, y)
                    forward_states.append(self._state.copy())  # type: ignore
                    forward_covs.append(self._covariance.copy())  # type: ignore
                    predicted_states.append(self._state.copy())  # type: ignore
                    predicted_covs.append(self._covariance.copy())  # type: ignore
                else:
                    # Not initialized yet - use dummy values
                    dummy_state = np.array([x, y, 0.0, 0.0])
                    dummy_cov = np.eye(4) * 1.0
                    forward_states.append(dummy_state)
                    forward_covs.append(dummy_cov)
                    predicted_states.append(dummy_state)
                    predicted_covs.append(dummy_cov)
                continue

            # Predict
            pred_state, pred_cov = self._predict()
            predicted_states.append(pred_state.copy())
            predicted_covs.append(pred_cov.copy())

            # Update
            is_confident = confidence >= self.config.min_confidence_for_update
            is_valid = is_confident and self._is_valid_measurement(
                pred_state, pred_cov, x, y,
            )

            if is_valid:
                z = np.array([x, y])
                self._state, self._covariance = self._update(pred_state, pred_cov, z)
            else:
                self._state = pred_state
                self._covariance = pred_cov

            forward_states.append(self._state.copy())
            forward_covs.append(self._covariance.copy())

        # Backward smoothing pass (RTS smoother)
        smoothed_states: list[np.ndarray] = [np.zeros(4) for _ in range(n)]
        smoothed_states[n - 1] = forward_states[n - 1]

        for k in range(n - 2, -1, -1):
            # Compute smoother gain
            # C_k = P_k @ F^T @ inv(P_{k+1|k})
            try:
                smoother_gain = (
                    forward_covs[k]
                    @ self._F.T
                    @ np.linalg.inv(predicted_covs[k + 1])
                )
            except np.linalg.LinAlgError:
                # Singular matrix - use forward estimate
                smoothed_states[k] = forward_states[k]
                continue

            # Smoothed state
            # x_k|n = x_k|k + C_k @ (x_{k+1|n} - x_{k+1|k})
            smoothed_states[k] = (
                forward_states[k]
                + smoother_gain @ (smoothed_states[k + 1] - predicted_states[k + 1])
            )

        # Convert smoothed states back to BallPosition
        smoothed_positions: list[BallPosition] = []
        for i, pos in enumerate(raw_positions):
            state = smoothed_states[i]
            x_smooth = float(max(0.0, min(1.0, state[0])))
            y_smooth = float(max(0.0, min(1.0, state[1])))

            smoothed_positions.append(
                BallPosition(
                    frame_number=pos.frame_number,
                    x=x_smooth,
                    y=y_smooth,
                    confidence=pos.confidence,  # Keep original confidence
                )
            )

        return smoothed_positions

    def _interpolate_missing(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Interpolate missing frames with linear interpolation.

        Fills gaps where ball detection failed by linearly interpolating
        between known positions. Only interpolates gaps up to max_interpolation_gap
        frames to avoid creating fake trajectories across scene cuts.

        Args:
            positions: List of ball positions (sorted by frame number)

        Returns:
            List with interpolated positions added for missing frames
        """
        from rallycut.tracking.ball_tracker import BallPosition

        if not positions:
            return []

        # Build map of confident positions
        # In raw mode, use min_output_confidence (0.05) since VballNet positions
        # are already filtered. min_confidence_for_update (0.3) is for Kalman mode
        # where low-confidence observations should be prediction-only.
        min_conf = (
            self.config.min_output_confidence
            if not self.config.enable_kalman
            else self.config.min_confidence_for_update
        )
        pos_by_frame = {
            p.frame_number: p for p in positions if p.confidence >= min_conf
        }

        if not pos_by_frame:
            return positions

        frames = sorted(pos_by_frame.keys())
        result = list(positions)  # Start with original positions
        interpolated_count = 0

        # Find and fill gaps
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            gap = f2 - f1

            # Only interpolate small gaps
            if gap > 1 and gap <= self.config.max_interpolation_gap:
                p1, p2 = pos_by_frame[f1], pos_by_frame[f2]

                for f in range(f1 + 1, f2):
                    # Linear interpolation factor
                    t = (f - f1) / gap

                    interp_pos = BallPosition(
                        frame_number=f,
                        x=p1.x + t * (p2.x - p1.x),
                        y=p1.y + t * (p2.y - p1.y),
                        confidence=self.config.interpolated_confidence,
                    )
                    result.append(interp_pos)
                    interpolated_count += 1

        if interpolated_count > 0:
            logger.debug(f"Interpolated {interpolated_count} missing frames")
            # Re-sort by frame number after adding interpolated positions
            result.sort(key=lambda p: p.frame_number)

        return result

    def _prune_segments(
        self,
        positions: list["BallPosition"],
        ghost_ranges: list[tuple[int, int]] | None = None,
    ) -> list["BallPosition"]:
        """Remove short disconnected segments from the trajectory.

        VballNet often outputs consistent false detections at the start and end
        of rallies (before it has enough temporal context, or after the ball
        leaves the frame). These form short trajectory segments that are
        spatially disconnected from the main ball trajectory.

        VballNet also interleaves single-frame false positives (jumping to
        player positions) within real trajectory regions. This creates many
        tiny real-trajectory fragments separated by false jumps. To handle
        this, short segments that are spatially close to an anchor (long)
        segment are kept rather than discarded.

        This method:
        1. Drops very low confidence positions (VballNet "no detection" placeholders)
        2. Splits the trajectory into segments at large position jumps
        3. Identifies anchor segments (long enough to be reliable trajectory)
        4. Keeps non-anchor segments whose centroid is close to an anchor endpoint
        5. Discards remaining segments (false detections)
        """
        if len(positions) < 2:
            return positions

        # Step 1: Drop positions below minimum output confidence
        # VballNet outputs (0.5, 0.5) at conf=0.0 for frames without detection
        min_conf = self.config.min_output_confidence
        confident = [p for p in positions if p.confidence >= min_conf]
        if not confident:
            return []

        dropped = len(positions) - len(confident)
        if dropped > 0:
            logger.debug(f"Dropped {dropped} positions below confidence {min_conf}")

        # Step 2: Split into segments at large jumps or gaps
        threshold = self.config.segment_jump_threshold
        segments: list[list[BallPosition]] = [[confident[0]]]

        for i in range(1, len(confident)):
            prev = confident[i - 1]
            curr = confident[i]
            frame_gap = curr.frame_number - prev.frame_number
            dist = np.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2)

            # Split on large position jump (normalized by gap for velocity)
            # or large frame gap (likely different context)
            if dist > threshold or frame_gap > 15:
                segments.append([curr])
            else:
                segments[-1].append(curr)

        if len(segments) <= 1:
            return confident

        # Step 3: Identify anchor segments (long enough to be reliable)
        min_len = self.config.min_segment_frames
        anchor_indices: set[int] = set()
        for i, seg in enumerate(segments):
            if len(seg) >= min_len:
                anchor_indices.add(i)

        # Step 3b: Exclude anchors that overlap with ghost ranges.
        # Ghost segments (false detections after ball exits frame) can be long
        # enough to qualify as anchors, but they shouldn't rescue nearby short
        # segments from being pruned — those short segments are typically false
        # positives at player positions that only appear "near" the ghost anchor.
        # Only exclude if the non-ghost portion is too short to be an anchor
        # on its own — otherwise the real portion should still act as anchor.
        if ghost_ranges:
            ghost_anchors: set[int] = set()
            for i in anchor_indices:
                seg = segments[i]
                non_ghost_count = sum(
                    1
                    for p in seg
                    if not any(
                        g_start <= p.frame_number <= g_end
                        for g_start, g_end in ghost_ranges
                    )
                )
                if non_ghost_count < min_len:
                    ghost_anchors.add(i)
            if ghost_anchors:
                anchor_indices -= ghost_anchors
                logger.debug(
                    f"Excluded {len(ghost_anchors)} ghost-overlapping "
                    f"segments from anchors (non-ghost portion < {min_len})"
                )

        # Step 3c: Remove false start/tail anchors (VballNet warmup/cooldown).
        # VballNet's temporal context warmup produces false detections at rally
        # start that can be long enough to qualify as anchors (>min_segment_frames).
        # If the first anchor is much shorter than the second AND spatially
        # disconnected, it's a warmup artifact. Same for the last anchor.
        sorted_anchors = sorted(anchor_indices)
        if len(sorted_anchors) >= 2:
            # Check first anchor (false start)
            first_idx = sorted_anchors[0]
            second_idx = sorted_anchors[1]
            first_seg = segments[first_idx]
            second_seg = segments[second_idx]
            if (
                len(first_seg) < len(second_seg) / 3
                and np.sqrt(
                    (first_seg[-1].x - second_seg[0].x) ** 2
                    + (first_seg[-1].y - second_seg[0].y) ** 2
                )
                > threshold
            ):
                anchor_indices.discard(first_idx)
                logger.info(
                    f"Segment pruning: removed false start anchor "
                    f"[{first_seg[0].frame_number}-{first_seg[-1].frame_number}] "
                    f"({len(first_seg)} frames, next anchor has {len(second_seg)})"
                )

            # Check last anchor (false tail)
            last_idx = sorted_anchors[-1]
            second_last_idx = sorted_anchors[-2]
            # Re-check in case first anchor removal changed things
            if last_idx in anchor_indices and second_last_idx in anchor_indices:
                last_seg = segments[last_idx]
                second_last_seg = segments[second_last_idx]
                if (
                    len(last_seg) < len(second_last_seg) / 3
                    and np.sqrt(
                        (last_seg[0].x - second_last_seg[-1].x) ** 2
                        + (last_seg[0].y - second_last_seg[-1].y) ** 2
                    )
                    > threshold
                ):
                    anchor_indices.discard(last_idx)
                    logger.info(
                        f"Segment pruning: removed false tail anchor "
                        f"[{last_seg[0].frame_number}-{last_seg[-1].frame_number}] "
                        f"({len(last_seg)} frames, prev anchor has "
                        f"{len(second_last_seg)})"
                    )

        # Step 4: Keep short segments whose centroid is close to an anchor endpoint.
        # These are real trajectory fragments between interleaved false positives.
        # Use half the jump threshold as proximity — tight enough to exclude
        # false positives (which jump to player positions 30-50% away) while
        # keeping real trajectory fragments (typically <5% from anchor).
        # Also require temporal proximity: after a large gap (ball exited frame),
        # VballNet can restart at a player position that happens to be spatially
        # near the last anchor endpoint. These shouldn't be recovered.
        proximity = threshold / 2
        max_recovery_gap = self.config.max_interpolation_gap * 3
        kept: list[BallPosition] = []
        removed_count = 0
        kept_info: list[str] = []
        removed_info: list[str] = []
        recovered_count = 0

        for i, seg in enumerate(segments):
            tag = f"[{seg[0].frame_number}-{seg[-1].frame_number}]({len(seg)})"

            if i in anchor_indices:
                kept.extend(seg)
                kept_info.append(tag)
                continue

            # Short segment: check proximity to nearest anchor endpoints
            centroid_x = float(np.mean([p.x for p in seg]))
            centroid_y = float(np.mean([p.y for p in seg]))

            close_to_anchor = False

            # Check previous anchor (end position)
            for j in range(i - 1, -1, -1):
                if j in anchor_indices:
                    ref = segments[j][-1]
                    frame_gap = seg[0].frame_number - ref.frame_number
                    if frame_gap > max_recovery_gap:
                        break
                    dist = np.sqrt(
                        (centroid_x - ref.x) ** 2 + (centroid_y - ref.y) ** 2
                    )
                    if dist < proximity:
                        close_to_anchor = True
                    break

            # Check next anchor (start position)
            if not close_to_anchor:
                for j in range(i + 1, len(segments)):
                    if j in anchor_indices:
                        ref = segments[j][0]
                        frame_gap = ref.frame_number - seg[-1].frame_number
                        if frame_gap > max_recovery_gap:
                            break
                        dist = np.sqrt(
                            (centroid_x - ref.x) ** 2 + (centroid_y - ref.y) ** 2
                        )
                        if dist < proximity:
                            close_to_anchor = True
                        break

            if close_to_anchor:
                kept.extend(seg)
                kept_info.append(tag + "*")  # * marks recovered segments
                recovered_count += len(seg)
            else:
                removed_count += len(seg)
                removed_info.append(tag)

        if removed_count > 0 or recovered_count > 0:
            parts = [
                f"Segment pruning: kept {len(kept_info)} segments "
                f"({', '.join(kept_info)})"
            ]
            if removed_count > 0:
                parts.append(
                    f"removed {removed_count} positions from "
                    f"{len(removed_info)} short segments"
                )
            if recovered_count > 0:
                parts.append(f"recovered {recovered_count} near-anchor positions")
            logger.info(", ".join(parts))

        return kept if kept else confident  # Fall back to all if nothing survives

    def _prune_oscillating(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Trim sustained oscillation from trajectory tails using cluster detection.

        VballNet can lock onto two players and alternate between them with high
        confidence after the ball exits the frame. The pattern is cluster-based:
        positions stay near player B for 2-5 frames, jump to player A for 1-2
        frames, then back. Per-frame displacement is tiny within each cluster,
        so displacement-reversal detection misses this pattern entirely.

        Also detects single-cluster hovering: after a large gap (ball exited
        frame), VballNet can lock onto a single player position and produce
        many frames within a tiny radius. Detected by checking short segments
        (≤3x window) after a large gap: if the first window positions all lie
        within segment_jump_threshold/4 of their centroid, the segment is dropped.

        Algorithm (cluster transition detection):
        1. Split into contiguous segments (gap > 5 frames)
        2. For each segment after a large gap, check for hovering (all
           positions within a small radius of centroid → drop entire segment)
        3. For each remaining segment, slide a window across positions:
           a. Find two poles: the pair of positions with maximum distance
           b. If pole distance < min_displacement: skip (jitter, not oscillation)
           c. Assign each position to nearest pole (binary cluster label)
           d. Count transitions (cluster[i] != cluster[i+1])
           e. If transition_rate >= threshold: trim from window start onward
        """
        if len(positions) < self.config.min_oscillation_frames + 2:
            return positions

        min_pole_dist = self.config.oscillation_min_displacement
        window = self.config.min_oscillation_frames
        rate_threshold = self.config.oscillation_reversal_rate

        # Step 1: Split into contiguous segments (gap > 5 frames)
        segments: list[list[BallPosition]] = [[positions[0]]]
        for i in range(1, len(positions)):
            if positions[i].frame_number - positions[i - 1].frame_number > 5:
                segments.append([positions[i]])
            else:
                segments[-1].append(positions[i])

        result: list[BallPosition] = []
        max_gap = self.config.max_interpolation_gap
        hover_radius = self.config.segment_jump_threshold / 4
        prev_end_frame: int | None = None

        for seg in segments:
            # Hovering detection: single-player lock-on after ball exits frame.
            # If segment follows a large gap and all positions cluster within
            # a tiny radius, it's VballNet locked onto a stationary player.
            # Only flag short segments — long ones that start slow are likely
            # real (ball gradually accelerating after serve/bounce).
            gap = (
                seg[0].frame_number - prev_end_frame
                if prev_end_frame is not None
                else 0
            )
            if gap > max_gap and window <= len(seg) <= window * 3:
                first_w = seg[:window]
                cx = float(np.mean([p.x for p in first_w]))
                cy = float(np.mean([p.y for p in first_w]))
                max_spread = float(
                    max(
                        np.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
                        for p in first_w
                    )
                )
                if max_spread < hover_radius:
                    frame_range = (
                        f"[{seg[0].frame_number}-{seg[-1].frame_number}]"
                    )
                    logger.info(
                        f"Oscillation pruning: dropped hovering segment "
                        f"{frame_range} ({len(seg)} frames, "
                        f"spread={max_spread:.4f}, gap={gap})"
                    )
                    prev_end_frame = seg[-1].frame_number
                    continue

            if len(seg) < window + 2:
                result.extend(seg)
                prev_end_frame = seg[-1].frame_number
                continue

            n = len(seg)
            trim_idx = None  # Index into seg where trimming starts

            # Slide a window across the segment
            for start in range(n - window + 1):
                w = seg[start : start + window]

                # Find two poles: pair with maximum distance
                max_dist = 0.0
                pole_a = (w[0].x, w[0].y)
                pole_b = (w[0].x, w[0].y)
                for i in range(len(w)):
                    for j in range(i + 1, len(w)):
                        d = np.sqrt((w[i].x - w[j].x) ** 2 + (w[i].y - w[j].y) ** 2)
                        if d > max_dist:
                            max_dist = d
                            pole_a = (w[i].x, w[i].y)
                            pole_b = (w[j].x, w[j].y)

                # Skip if poles are too close (jitter, not oscillation)
                if max_dist < min_pole_dist:
                    continue

                # Assign each position to nearest pole, tracking distance
                labels = []
                dists_a: list[float] = []  # distances to pole_a for cluster 0
                dists_b: list[float] = []  # distances to pole_b for cluster 1
                for p in w:
                    da = np.sqrt((p.x - pole_a[0]) ** 2 + (p.y - pole_a[1]) ** 2)
                    db = np.sqrt((p.x - pole_b[0]) ** 2 + (p.y - pole_b[1]) ** 2)
                    if da <= db:
                        labels.append(0)
                        dists_a.append(da)
                    else:
                        labels.append(1)
                        dists_b.append(db)

                # Both clusters must have at least 3 positions to be oscillation.
                # A single spike + geometric artifacts won't reach 3.
                if len(dists_a) < 3 or len(dists_b) < 3:
                    continue

                # Clusters must be compact: in real oscillation, VballNet locks
                # onto fixed player positions so within-cluster spread is tiny
                # (<1% of screen). In a ball bounce passing through the midpoint,
                # cluster members span a wide trajectory arc. Use half the pole
                # distance as the max allowed spread per cluster.
                max_cluster_spread = min_pole_dist / 2
                if max(dists_a) > max_cluster_spread or max(dists_b) > max_cluster_spread:
                    continue

                # Count transitions between clusters
                transitions = sum(
                    1 for k in range(len(labels) - 1) if labels[k] != labels[k + 1]
                )
                rate = transitions / (len(labels) - 1)

                if rate >= rate_threshold:
                    trim_idx = start
                    break

            if trim_idx is not None:
                trimmed = len(seg) - trim_idx
                result.extend(seg[:trim_idx])
                frame_range = f"[{seg[trim_idx].frame_number}-{seg[-1].frame_number}]"
                logger.info(
                    f"Oscillation pruning: trimmed {trimmed} frames {frame_range} "
                    f"from segment [{seg[0].frame_number}-{seg[-1].frame_number}]"
                )
            else:
                result.extend(seg)

            prev_end_frame = seg[-1].frame_number

        return result

    def _detect_exit_ghost_ranges(
        self,
        positions: list["BallPosition"],
    ) -> list[tuple[int, int]]:
        """Detect frame ranges containing ghost detections after ball exits.

        Scans positions for the exit ghost pattern: ball approaching a screen edge
        with consistent velocity, then reversing direction. Only the last approach
        frame must be in the edge zone — the ball approaches from further away.
        Returns frame ranges [start, end] where detections are ghosts.

        Ghost range continues until:
        - A gap > max_interpolation_gap frames (break in detections), OR
        - A position returns to the exit edge zone (ball re-entering from same edge)

        This is the detection phase only — does not modify positions. Called on
        raw data before segment pruning to preserve edge-approach evidence.
        """
        if len(positions) < self.config.exit_approach_frames + 1:
            return []

        approach_n = self.config.exit_approach_frames
        edge_zone = self.config.exit_edge_zone
        min_speed = self.config.exit_min_approach_speed
        max_gap = self.config.max_interpolation_gap

        ranges: list[tuple[int, int]] = []
        in_ghost_region = False
        exit_edge_name: str | None = None
        ghost_start_frame: int = 0
        last_ghost_frame: int = 0

        for i in range(approach_n, len(positions)):
            # If we're in a ghost region, keep marking until termination
            if in_ghost_region:
                frame_gap = positions[i].frame_number - positions[i - 1].frame_number
                curr = positions[i]

                # Terminate at frame gap
                if frame_gap > max_gap:
                    ranges.append((ghost_start_frame, last_ghost_frame))
                    in_ghost_region = False
                    exit_edge_name = None
                    continue

                # Terminate when position returns to exit edge zone
                # (ball re-entering from the same edge it left)
                in_edge = False
                if exit_edge_name == "top" and curr.y < edge_zone:
                    in_edge = True
                elif exit_edge_name == "bottom" and curr.y > 1 - edge_zone:
                    in_edge = True
                elif exit_edge_name == "left" and curr.x < edge_zone:
                    in_edge = True
                elif exit_edge_name == "right" and curr.x > 1 - edge_zone:
                    in_edge = True

                if in_edge:
                    ranges.append((ghost_start_frame, last_ghost_frame))
                    in_ghost_region = False
                    exit_edge_name = None
                    logger.debug(
                        f"Exit ghost: terminated at f={curr.frame_number} "
                        f"(position returned to edge zone)"
                    )
                    continue

                last_ghost_frame = curr.frame_number
                continue

            # Check that approach frames are contiguous (no large gaps)
            approach = positions[i - approach_n : i]
            has_gap = False
            for j in range(1, len(approach)):
                if approach[j].frame_number - approach[j - 1].frame_number > max_gap:
                    has_gap = True
                    break
            if has_gap:
                continue

            curr = positions[i]

            # Check each edge direction
            # Only the LAST approach frame must be in the edge zone — the ball
            # approaches from further away, what matters is that it reached the
            # edge. Consistent approach velocity is checked separately below.
            edges: list[tuple[str, float, list[float]]] = []
            last_approach = approach[-1]

            # Top edge
            if last_approach.y < edge_zone:
                velocities = [
                    approach[j].y - approach[j - 1].y for j in range(1, len(approach))
                ]
                reversal_vel = curr.y - last_approach.y
                edges.append(("top", reversal_vel, velocities))

            # Bottom edge
            if last_approach.y > 1 - edge_zone:
                velocities = [
                    approach[j].y - approach[j - 1].y for j in range(1, len(approach))
                ]
                reversal_vel = curr.y - last_approach.y
                edges.append(("bottom", reversal_vel, velocities))

            # Left edge
            if last_approach.x < edge_zone:
                velocities = [
                    approach[j].x - approach[j - 1].x for j in range(1, len(approach))
                ]
                reversal_vel = curr.x - last_approach.x
                edges.append(("left", reversal_vel, velocities))

            # Right edge
            if last_approach.x > 1 - edge_zone:
                velocities = [
                    approach[j].x - approach[j - 1].x for j in range(1, len(approach))
                ]
                reversal_vel = curr.x - last_approach.x
                edges.append(("right", reversal_vel, velocities))

            for edge_name, reversal_vel, velocities in edges:
                # Check consistent approach velocity toward edge
                if edge_name == "top":
                    approaching = all(v < -min_speed for v in velocities)
                    reversed = reversal_vel > min_speed
                elif edge_name == "bottom":
                    approaching = all(v > min_speed for v in velocities)
                    reversed = reversal_vel < -min_speed
                elif edge_name == "left":
                    approaching = all(v < -min_speed for v in velocities)
                    reversed = reversal_vel > min_speed
                else:  # right
                    approaching = all(v > min_speed for v in velocities)
                    reversed = reversal_vel < -min_speed

                if approaching and reversed:
                    in_ghost_region = True
                    exit_edge_name = edge_name
                    ghost_start_frame = curr.frame_number
                    last_ghost_frame = curr.frame_number
                    logger.info(
                        f"Exit ghost: ball exited {edge_name} at "
                        f"f={approach[-1].frame_number}, marking "
                        f"f={curr.frame_number}+ as ghosts"
                    )
                    break

        # Close any open ghost region at end of trajectory
        if in_ghost_region:
            ranges.append((ghost_start_frame, last_ghost_frame))

        return ranges

    def _remove_outliers(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Remove outlier positions that are likely detection failures.

        Identifies and removes positions that:
        1. Are at screen edges (common failure mode where model outputs 0,0 or 1,1)
        2. Deviate significantly from the trajectory defined by neighbors

        Args:
            positions: List of ball positions (sorted by frame number)

        Returns:
            List with outlier positions removed
        """
        if len(positions) < 3:
            return positions

        # Build position map by frame
        pos_by_frame = {p.frame_number: p for p in positions}
        frames = sorted(pos_by_frame.keys())

        outlier_frames: set[int] = set()

        for i, frame in enumerate(frames):
            pos = pos_by_frame[frame]

            # Check 1: Edge detection (positions at screen boundaries)
            margin = self.config.edge_margin
            is_edge = (
                pos.x < margin or pos.x > (1 - margin) or
                pos.y < margin or pos.y > (1 - margin)
            )

            if is_edge:
                # Only mark as outlier if confidence is low-medium
                # High confidence edge positions might be legitimate (ball at edge)
                if pos.confidence < 0.8:
                    outlier_frames.add(frame)
                    continue

            # Check 2: Trajectory consistency with neighbors
            # Find neighbors (previous and next positions)
            prev_pos = None
            next_pos = None

            # Look for previous neighbor (up to 5 frames back)
            for j in range(i - 1, max(-1, i - 6), -1):
                if j >= 0 and frames[j] not in outlier_frames:
                    prev_pos = pos_by_frame[frames[j]]
                    prev_gap = frame - frames[j]
                    break

            # Look for next neighbor (up to 5 frames ahead)
            for j in range(i + 1, min(len(frames), i + 6)):
                if frames[j] not in outlier_frames:
                    next_pos = pos_by_frame[frames[j]]
                    next_gap = frames[j] - frame
                    break

            # Need at least min_neighbors for trajectory check
            neighbor_count = (1 if prev_pos else 0) + (1 if next_pos else 0)
            if neighbor_count < self.config.min_neighbors_for_outlier:
                continue

            # Interpolate expected position from neighbors
            if prev_pos and next_pos:
                # Both neighbors available - interpolate
                total_gap = prev_gap + next_gap
                t = prev_gap / total_gap
                expected_x = prev_pos.x + t * (next_pos.x - prev_pos.x)
                expected_y = prev_pos.y + t * (next_pos.y - prev_pos.y)
            elif prev_pos:
                # Only previous - use it directly (no extrapolation)
                expected_x, expected_y = prev_pos.x, prev_pos.y
            else:
                # Only next - use it directly
                expected_x, expected_y = next_pos.x, next_pos.y  # type: ignore

            # Compute deviation from expected position
            deviation = np.sqrt(
                (pos.x - expected_x) ** 2 + (pos.y - expected_y) ** 2
            )

            # Mark as outlier if deviation exceeds threshold
            # Scale threshold by confidence (lower confidence = stricter check)
            conf_factor = 0.5 + 0.5 * pos.confidence  # 0.5 to 1.0
            threshold = self.config.max_trajectory_deviation * conf_factor

            if deviation > threshold:
                outlier_frames.add(frame)
                logger.debug(
                    f"Frame {frame}: Outlier detected (deviation={deviation:.3f}, "
                    f"threshold={threshold:.3f}, conf={pos.confidence:.2f})"
                )
                continue

            # Check 3: Velocity reversal detection (A→B→A flickering pattern)
            # If we have both neighbors, check if velocity reverses sharply
            if prev_pos and next_pos:
                # Velocity into this point
                v_in_x = pos.x - prev_pos.x
                v_in_y = pos.y - prev_pos.y
                # Velocity out of this point
                v_out_x = next_pos.x - pos.x
                v_out_y = next_pos.y - pos.y

                speed_in = np.sqrt(v_in_x ** 2 + v_in_y ** 2)
                speed_out = np.sqrt(v_out_x ** 2 + v_out_y ** 2)

                # Only check reversal if both speeds are significant
                min_speed = self.config.outlier_min_speed
                if speed_in > min_speed and speed_out > min_speed:
                    # Cosine of angle between velocity vectors
                    dot = v_in_x * v_out_x + v_in_y * v_out_y
                    cos_angle = dot / (speed_in * speed_out)

                    # Sharp reversal (cos < -0.5 means angle > 120 degrees)
                    if cos_angle < -0.5:
                        outlier_frames.add(frame)
                        logger.debug(
                            f"Frame {frame}: Velocity reversal detected "
                            f"(cos_angle={cos_angle:.2f}, speed_in={speed_in:.3f}, "
                            f"speed_out={speed_out:.3f})"
                        )

        # Remove outliers
        if outlier_frames:
            logger.info(f"Removed {len(outlier_frames)} outlier positions")
            return [p for p in positions if p.frame_number not in outlier_frames]

        return positions

    def _find_blip_context(
        self,
        idx: int,
        frames: list[int],
        pos_by_frame: dict[int, "BallPosition"],
        min_dist: int,
        exclude: set[int],
    ) -> tuple["BallPosition | None", int, "BallPosition | None", int]:
        """Find distant context positions for blip detection.

        Searches backward and forward from idx for the nearest position that is
        at least min_dist frame numbers away, skipping any indices in exclude.

        Returns:
            (prev_ctx, prev_gap, next_ctx, next_gap) where:
            - prev_ctx: nearest previous context position (or None)
            - prev_gap: frame distance to prev_ctx
            - next_ctx: nearest next context position (or None)
            - next_gap: frame distance to next_ctx
        """
        current_frame = frames[idx]
        prev_ctx = None
        prev_gap = 0
        next_ctx = None
        next_gap = 0

        # Search backward for previous context
        for j in range(idx - 1, -1, -1):
            if j in exclude:
                continue
            if current_frame - frames[j] >= min_dist:
                prev_ctx = pos_by_frame[frames[j]]
                prev_gap = current_frame - frames[j]
                break

        # Search forward for next context
        for j in range(idx + 1, len(frames)):
            if j in exclude:
                continue
            if frames[j] - current_frame >= min_dist:
                next_ctx = pos_by_frame[frames[j]]
                next_gap = frames[j] - current_frame
                break

        return prev_ctx, prev_gap, next_ctx, next_gap

    def _blip_deviation(
        self,
        pos: "BallPosition",
        prev_ctx: "BallPosition",
        next_ctx: "BallPosition",
        prev_gap: int,
        next_gap: int,
    ) -> float:
        """Compute deviation of a position from the line between two context positions.

        Linearly interpolates between prev_ctx and next_ctx based on frame
        distance ratios, then returns the Euclidean distance from pos to the
        interpolated point.
        """
        total_gap = prev_gap + next_gap
        t = prev_gap / total_gap
        expected_x = prev_ctx.x + t * (next_ctx.x - prev_ctx.x)
        expected_y = prev_ctx.y + t * (next_ctx.y - prev_ctx.y)
        return float(np.sqrt((pos.x - expected_x) ** 2 + (pos.y - expected_y) ** 2))

    def _remove_trajectory_blips(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """Remove multi-frame trajectory blips using distant context.

        VballNet can briefly lock onto a player position for 2-5 consecutive
        frames mid-trajectory. Single-frame outlier detection misses these
        because the consecutive false positives validate each other as neighbors.

        Two-phase approach to avoid false positives on real bounces:
        1. Flag positions that deviate from distant trajectory context
        2. Only remove CLUSTERS of ≥2 consecutive flagged frames with compact
           internal spread — real bounces have spread, blips are tightly clustered
        """
        if len(positions) < 3:
            return positions

        min_dist = self.config.blip_context_min_frames
        max_dev = self.config.blip_max_deviation
        max_ctx_gap = self.config.blip_max_context_gap
        # Blip cluster must be spatially compact (within 5% of screen).
        # Real bounces spread along a curve; VballNet player-locking blips
        # cluster tightly at a fixed position (~1% noise).
        max_blip_spread = 0.05

        pos_by_frame = {p.frame_number: p for p in positions}
        frames = sorted(pos_by_frame.keys())

        # Phase 1a: Flag suspect positions deviating from distant context
        suspect_indices: set[int] = set()

        for i, frame in enumerate(frames):
            prev_ctx, prev_gap, next_ctx, next_gap = self._find_blip_context(
                i, frames, pos_by_frame, min_dist, exclude=set()
            )
            # Skip if context is too far — linear interpolation becomes unreliable
            if not prev_ctx or not next_ctx or prev_gap + next_gap > max_ctx_gap:
                continue

            deviation = self._blip_deviation(
                pos_by_frame[frame], prev_ctx, next_ctx, prev_gap, next_gap
            )
            conf_factor = 0.5 + 0.5 * pos_by_frame[frame].confidence
            if deviation > max_dev * conf_factor:
                suspect_indices.add(i)

        if not suspect_indices:
            return positions

        # Phase 1b: Re-evaluate suspects with clean context (skip other suspects)
        # Prevents blip positions from contaminating nearby real frame context
        confirmed: set[int] = set()
        for i in suspect_indices:
            prev_ctx, prev_gap, next_ctx, next_gap = self._find_blip_context(
                i, frames, pos_by_frame, min_dist, exclude=suspect_indices
            )
            # Skip if context is too far — linear interpolation becomes unreliable
            if not prev_ctx or not next_ctx or prev_gap + next_gap > max_ctx_gap:
                continue

            deviation = self._blip_deviation(
                pos_by_frame[frames[i]], prev_ctx, next_ctx, prev_gap, next_gap
            )
            conf_factor = 0.5 + 0.5 * pos_by_frame[frames[i]].confidence
            if deviation > max_dev * conf_factor:
                confirmed.add(i)

        if not confirmed:
            return positions

        suspect_indices = confirmed

        # Phase 2: Group consecutive suspects into runs, keep only compact clusters
        blip_frames: set[int] = set()
        sorted_suspects = sorted(suspect_indices)
        runs: list[list[int]] = [[sorted_suspects[0]]]
        for idx in sorted_suspects[1:]:
            if idx == runs[-1][-1] + 1:
                runs[-1].append(idx)
            else:
                runs.append([idx])

        for run in runs:
            # Require ≥2 consecutive suspect frames (single deviations are
            # real trajectory changes like bounces)
            if len(run) < 2:
                continue

            # Check cluster compactness — blips are at a fixed player position,
            # real bounces spread along a curve.
            # Scale spread tolerance with cluster length: longer blips have
            # transitional frames as tracker moves to/from wrong position,
            # creating more spread even though Phase 1 confirmed deviation.
            effective_spread = min(
                max_blip_spread + 0.01 * max(0, len(run) - 2),
                3 * max_blip_spread,
            )
            cluster_positions = [pos_by_frame[frames[i]] for i in run]
            cx = float(np.mean([p.x for p in cluster_positions]))
            cy = float(np.mean([p.y for p in cluster_positions]))
            spread = float(max(
                np.sqrt((p.x - cx) ** 2 + (p.y - cy) ** 2)
                for p in cluster_positions
            ))

            if spread <= effective_spread:
                for i in run:
                    blip_frames.add(frames[i])
                frame_range = f"[{frames[run[0]]}-{frames[run[-1]]}]"
                logger.info(
                    f"Blip removal: removed {len(run)} frames {frame_range} "
                    f"(spread={spread:.4f})"
                )

        if blip_frames:
            return [
                p for p in positions if p.frame_number not in blip_frames
            ]

        return positions
