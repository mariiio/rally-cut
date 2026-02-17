"""
Player filtering to detect only court players using multiple signals.

Filters out spectators, referees, and other non-playing persons using:
1. Bounding box size (court players are closer = larger boxes)
2. Track stability (court players appear consistently across frames)
3. Expected player count (4 for beach volleyball 2v2)
4. Ball trajectory play area (optional refinement)
5. Court position (requires calibration) - hard filter for court presence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator

logger = logging.getLogger(__name__)


@dataclass
class TrackStats:
    """Statistics for a single track across all frames."""

    track_id: int
    frame_count: int  # Number of frames this track appears in
    total_frames: int  # Total frames in video
    avg_bbox_area: float  # Average bbox area across appearances
    avg_confidence: float  # Average detection confidence
    ball_proximity_score: float = 0.0  # Fraction of appearances near ball
    first_frame: int = 0  # First frame where track appears
    last_frame: int = 0  # Last frame where track appears
    position_spread: float = 0.0  # Movement spread (geometric mean of X/Y std dev, normalized 0-1)

    # Court position stats (requires calibration, set by analyze_tracks)
    court_presence_ratio: float = 0.0  # Fraction of positions inside court bounds
    interior_ratio: float = 0.0  # Fraction of positions in court interior
    has_court_stats: bool = False  # True if court stats were computed

    # Referee detection stats (set by compute_track_stats or detect_referee_tracks)
    avg_x: float = 0.5  # Average X position (0-1, for sideline detection)
    avg_y: float = 0.5  # Average Y position (0-1)
    x_std: float = 0.0  # X position standard deviation
    y_std: float = 0.0  # Y position standard deviation
    is_likely_referee: bool = False  # Set by detect_referee_tracks

    @property
    def presence_rate(self) -> float:
        """Fraction of frames this track appears in."""
        if self.total_frames == 0:
            return 0.0
        return self.frame_count / self.total_frames

    @property
    def stability_score(self) -> float:
        """Combined score for track stability (higher = more stable).

        Uses default weights. For configurable weights, use compute_stability_score().
        """
        # Weighted combination of presence, size, and ball proximity
        # Weights match PlayerFilterConfig defaults for consistency
        bbox_area_scalar = 10.0
        score = (
            self.presence_rate * 0.40  # presence_weight
            + min(self.avg_bbox_area * bbox_area_scalar, 1.0) * 0.35  # bbox_area_weight
            + self.ball_proximity_score * 0.25  # ball_proximity_weight
        )
        return score

    @property
    def movement_ratio(self) -> float:
        """Ratio of X movement to Y movement.

        Referees tend to move parallel to the net (high X variance, low Y variance).
        Players move in all directions.
        """
        if self.y_std < 0.001:
            return float("inf") if self.x_std > 0.001 else 1.0
        return self.x_std / self.y_std


@dataclass
class PlayerFilterConfig:
    """Configuration for player filtering (beach volleyball 2v2).

    Beach volleyball-optimized thresholds tuned for:
    - Far players appearing smaller in camera view
    - Short rallies (3-5 seconds typical)
    - Defensive positioning requiring larger ball proximity radius
    """

    # Bbox size filtering - minimum size to be considered a court player
    # These thresholds filter out distant spectators while keeping court players
    # Lower values to catch far-side players who appear smaller
    min_bbox_area: float = 0.003  # Min 0.3% of frame area (was 0.5%)
    min_bbox_height: float = 0.08  # Min 8% of frame height (was 10%)

    # Top-K selection - keep only the 4 largest detections per frame
    # Beach volleyball is 2v2 = 4 players on court
    max_players: int = 4
    players_per_team: int = 2  # 2v2 beach volleyball

    # Two-team filtering - split court and select per team
    # This ensures players from both teams are selected
    # Camera is always behind baseline: teams split by Y (near/far, horizontal line)
    use_two_team_filter: bool = True

    # Track stability filtering - prefer tracks that appear consistently
    # Lower thresholds for short rallies (3-5s at 30fps = 90-150 frames)
    min_presence_rate: float = 0.20  # Track must appear in 20%+ of frames (was 30%)
    min_stability_score: float = 0.20  # Minimum combined stability score (was 0.25)

    # Stability score weights (for compute_stability_score)
    # Rebalanced: more weight on bbox area and ball proximity (more discriminative)
    presence_weight: float = 0.40  # Weight for presence rate (was 0.5)
    bbox_area_weight: float = 0.35  # Weight for bbox area (was 0.3)
    ball_proximity_weight: float = 0.25  # Weight for ball proximity (was 0.2)

    # Ball proximity - tracks near ball get stability boost
    # Larger radius for defensive positioning where players spread out
    ball_proximity_radius: float = 0.20  # 20% of frame size (was 15%)

    # Position spread requirement for primary tracks (filters out stationary objects)
    # Players move around the court; stationary things (referees, net posts) don't
    # Spread is geometric mean of X/Y std dev (normalized 0-1)
    # Lower threshold for short rallies with limited movement
    min_position_spread_for_primary: float = 0.015  # Base movement threshold (was 0.018)

    # Combined filter: tracks with BOTH low spread AND low ball proximity are filtered
    # This catches stationary objects while keeping players who don't move much but engage with ball
    # Tighter filter: require more ball proximity if stationary
    min_ball_proximity_for_stationary: float = 0.05  # If spread < threshold, need 5%+ ball proximity (was 3%)

    # Ball trajectory thresholds (secondary filter)
    # Higher thresholds to reduce false positives
    ball_confidence_threshold: float = 0.40  # Higher threshold (was 0.35)
    min_ball_points: int = 15  # More points for robust hull (was 10)
    hull_padding: float = 0.20  # 20% padding for edge plays (was 15%)
    min_ball_detection_rate: float = 0.30  # Unchanged

    # Track ID stabilization - merge tracks when one ends and another starts nearby
    # Slightly longer gap tolerance for track merging
    stabilize_track_ids: bool = True
    max_gap_frames: int = 90  # Max frames between track end and new track start (~3s at 30fps)
    max_merge_distance: float = 0.20  # Max position distance for merging (20% of frame)
    merge_distance_per_frame: float = 0.005  # Additional distance allowed per frame of gap (~0.15m/frame at 16m court)

    # Two-team hysteresis - prevents flickering when players cross midcourt boundary
    two_team_hysteresis: float = 0.03  # Players must cross boundary + hysteresis to switch teams

    # Referee detection heuristics
    # Note: This should be tighter than primary_sideline_threshold since referees stand
    # at the very edge of the frame, not just outside the typical play area
    referee_sideline_threshold: float = 0.08  # Track avg X near sidelines (x <= 0.08 or x >= 0.92)

    # Primary track sideline filter - wider than referee detection
    # Excludes tracks on far edges of frame (likely spectators/equipment)
    # Set wider for cameras angled toward court edges where players may appear near frame edges
    primary_sideline_threshold: float = 0.05  # Tracks with avg_x < this or > (1-this) are excluded
    referee_movement_ratio_min: float = 1.5  # X/Y movement ratio (referees move parallel to net)
    referee_ball_proximity_max: float = 0.12  # Referees occasionally near ball trajectory
    referee_y_std_max: float = 0.04  # Referees don't move up/down court like players (< 4%)


@dataclass
class CourtFilterConfig:
    """Configuration for court-based filtering using homography projection.

    Beach volleyball court is 8m x 16m. Players must be on or near the court.
    Asymmetric margins account for:
    - Sidelines: 2m for ball chases
    - Baselines: 4m for jump serves and defensive positioning
    """

    # Asymmetric court margins (meters)
    # Sideline margin: small to exclude referees standing on the side
    # Baseline margin: larger for jump serves
    court_margin_sideline: float = 0.5  # X-direction - tight to exclude sideline refs
    court_margin_baseline: float = 3.0  # Y-direction margin for jump serves

    # Hard filter threshold: tracks with less court presence are definitively not players
    # Set to 50% - players should be on court most of the time
    min_court_presence_ratio: float = 0.50  # 50% minimum

    # Interior definition: points must be this far from edges to count as "interior"
    inner_court_margin: float = 1.0  # meters


@dataclass
class CourtPositionStats:
    """Court position statistics for a track (requires calibration).

    Computed by projecting player positions to court coordinates.
    """

    track_id: int
    court_presence_ratio: float  # Fraction of positions inside court bounds
    interior_ratio: float  # Fraction of positions in court interior (not margins)
    avg_court_x: float  # Average X position in court coords (meters)
    avg_court_y: float  # Average Y position in court coords (meters)
    total_positions: int  # Number of positions projected


def compute_stability_score(stats: TrackStats, config: PlayerFilterConfig) -> float:
    """Compute stability score for a track using configurable weights.

    Args:
        stats: Track statistics.
        config: Filter configuration with weight parameters.

    Returns:
        Stability score (0-1, higher = more stable).
    """
    # Bbox area scalar: normalizes bbox area to 0-1 range
    # 10.0 works well for typical volleyball videos (players occupy ~5-10% of frame)
    bbox_area_scalar = 10.0

    # Weighted combination of presence, size, and ball proximity
    score = (
        stats.presence_rate * config.presence_weight
        + min(stats.avg_bbox_area * bbox_area_scalar, 1.0) * config.bbox_area_weight
        + stats.ball_proximity_score * config.ball_proximity_weight
    )
    return score


@dataclass
class CombinedScoreWeights:
    """Weights for combined player scoring.

    Different weight configurations for different data availability scenarios.
    """

    length: float = 0.10
    court_presence: float = 0.20
    interior_ratio: float = 0.15
    ball_proximity: float = 0.20
    ball_reactivity: float = 0.15
    ball_zone: float = 0.10
    position_spread: float = 0.10


# Pre-defined weight configurations for different data scenarios
WEIGHTS_FULL_DATA = CombinedScoreWeights(
    length=0.10,
    court_presence=0.20,
    interior_ratio=0.15,
    ball_proximity=0.20,
    ball_reactivity=0.15,
    ball_zone=0.10,
    position_spread=0.10,
)

WEIGHTS_NO_CALIBRATION = CombinedScoreWeights(
    length=0.20,
    court_presence=0.0,  # No calibration
    interior_ratio=0.0,  # No calibration
    ball_proximity=0.30,
    ball_reactivity=0.25,
    ball_zone=0.15,
    position_spread=0.10,
)

WEIGHTS_NO_BALL = CombinedScoreWeights(
    length=0.20,
    court_presence=0.30,
    interior_ratio=0.25,
    ball_proximity=0.0,  # No ball data
    ball_reactivity=0.0,  # No ball data
    ball_zone=0.0,  # No ball data
    position_spread=0.25,
)

WEIGHTS_MINIMAL = CombinedScoreWeights(
    length=1.0,  # Only length available
    court_presence=0.0,
    interior_ratio=0.0,
    ball_proximity=0.0,
    ball_reactivity=0.0,
    ball_zone=0.0,
    position_spread=0.0,
)


@dataclass
class CombinedScoreInputs:
    """Inputs for combined player scoring.

    All scores should be normalized to 0-1 range.
    """

    track_id: int
    length_score: float = 0.0  # Fraction of frames track is present
    court_presence: float = 0.0  # Fraction of positions inside court bounds
    interior_ratio: float = 0.0  # Fraction of positions in court interior
    ball_proximity: float = 0.0  # Fraction of appearances near ball
    ball_reactivity: float = 0.0  # Correlation with ball movement
    ball_zone: float = 0.0  # Position relative to ball trajectory zones
    position_spread: float = 0.0  # Movement spread (normalized)

    # Data availability flags
    has_calibration: bool = False
    has_ball_data: bool = False


def compute_combined_score(
    inputs: CombinedScoreInputs,
    weights: CombinedScoreWeights | None = None,
) -> float:
    """
    Compute combined player identification score.

    Uses different weight configurations based on data availability:
    - Full data (calibration + ball): All features weighted
    - No calibration: Ball-focused scoring
    - No ball data: Court-focused scoring
    - Neither: Length only

    Args:
        inputs: Score inputs for a single track.
        weights: Optional custom weights (auto-selects if None).

    Returns:
        Combined score (0-1, higher = more likely to be a player).
    """
    # Auto-select weights based on data availability
    if weights is None:
        if inputs.has_calibration and inputs.has_ball_data:
            weights = WEIGHTS_FULL_DATA
        elif inputs.has_ball_data:
            weights = WEIGHTS_NO_CALIBRATION
        elif inputs.has_calibration:
            weights = WEIGHTS_NO_BALL
        else:
            weights = WEIGHTS_MINIMAL

    score = (
        weights.length * inputs.length_score
        + weights.court_presence * inputs.court_presence
        + weights.interior_ratio * inputs.interior_ratio
        + weights.ball_proximity * inputs.ball_proximity
        + weights.ball_reactivity * inputs.ball_reactivity
        + weights.ball_zone * inputs.ball_zone
        + weights.position_spread * inputs.position_spread
    )

    return min(1.0, max(0.0, score))


def build_score_inputs_from_track_stats(
    stats: TrackStats,
    has_calibration: bool = False,
    has_ball_data: bool = False,
    ball_reactivity: float = 0.0,
    ball_zone: float = 0.0,
) -> CombinedScoreInputs:
    """
    Build CombinedScoreInputs from TrackStats.

    Args:
        stats: Track statistics from compute_track_stats().
        has_calibration: Whether court calibration is available.
        has_ball_data: Whether ball tracking data is available.
        ball_reactivity: Ball reactivity score (from ball_features.py).
        ball_zone: Ball zone score (from ball_features.py).

    Returns:
        CombinedScoreInputs ready for compute_combined_score().
    """
    # Normalize position spread to 0-1 range
    # Typical player spread: 0.02-0.10, use 0.10 as max
    spread_normalized = min(stats.position_spread / 0.10, 1.0)

    return CombinedScoreInputs(
        track_id=stats.track_id,
        length_score=stats.presence_rate,
        court_presence=stats.court_presence_ratio if stats.has_court_stats else 0.0,
        interior_ratio=stats.interior_ratio if stats.has_court_stats else 0.0,
        ball_proximity=stats.ball_proximity_score,
        ball_reactivity=ball_reactivity,
        ball_zone=ball_zone,
        position_spread=spread_normalized,
        has_calibration=has_calibration,
        has_ball_data=has_ball_data,
    )


def compute_court_position_stats(
    all_positions: list[PlayerPosition],
    calibrator: CourtCalibrator,
    court_config: CourtFilterConfig | None = None,
) -> dict[int, CourtPositionStats]:
    """
    Compute court position statistics for each track using homography projection.

    Projects player feet positions (bbox bottom-center) to court coordinates
    and computes per-track statistics about court presence.

    Args:
        all_positions: All player positions from all frames.
        calibrator: Calibrated court calibrator with homography.
        court_config: Court filter configuration (uses defaults if None).

    Returns:
        Dictionary mapping track_id to CourtPositionStats.
    """
    # Import here to avoid circular import, calibrator type already checked via duck typing
    from rallycut.court.calibration import CourtCalibrator as CourtCalibratorClass

    if not isinstance(calibrator, CourtCalibratorClass) or not calibrator.is_calibrated:
        logger.debug("Calibrator not available or not calibrated, skipping court position stats")
        return {}

    config = court_config or CourtFilterConfig()

    # Group court coordinates by track ID
    track_court_positions: dict[int, list[tuple[float, float]]] = {}

    for p in all_positions:
        if p.track_id < 0:
            continue

        # Use feet position (bbox bottom-center) for court projection
        # This is more accurate than bbox center for floor position
        feet_x = p.x  # Normalized image X
        feet_y = p.y + p.height / 2  # Bottom of bbox

        try:
            # Project to court coordinates (need dummy width/height, coords are normalized)
            court_point = calibrator.image_to_court((feet_x, feet_y), 1, 1)

            if p.track_id not in track_court_positions:
                track_court_positions[p.track_id] = []

            track_court_positions[p.track_id].append(court_point)
        except (RuntimeError, ValueError) as e:
            logger.debug(f"Failed to project position for track {p.track_id}: {e}")
            continue

    # Compute stats for each track
    stats: dict[int, CourtPositionStats] = {}

    for track_id, court_points in track_court_positions.items():
        if not court_points:
            continue

        # Count positions inside court bounds and interior
        in_court_count = 0
        in_interior_count = 0
        court_x_sum = 0.0
        court_y_sum = 0.0

        for cx, cy in court_points:
            court_x_sum += cx
            court_y_sum += cy

            # Check if in court bounds (with asymmetric margins)
            if calibrator.is_point_in_court_with_margin(
                (cx, cy),
                sideline_margin=config.court_margin_sideline,
                baseline_margin=config.court_margin_baseline,
            ):
                in_court_count += 1

            # Check if in court interior
            if calibrator.is_point_in_court_interior(
                (cx, cy),
                interior_margin=config.inner_court_margin,
            ):
                in_interior_count += 1

        total = len(court_points)
        stats[track_id] = CourtPositionStats(
            track_id=track_id,
            court_presence_ratio=in_court_count / total,
            interior_ratio=in_interior_count / total,
            avg_court_x=court_x_sum / total,
            avg_court_y=court_y_sum / total,
            total_positions=total,
        )

    if stats:
        avg_presence = sum(s.court_presence_ratio for s in stats.values()) / len(stats)
        logger.debug(
            f"Court position stats: {len(stats)} tracks, "
            f"avg court presence={avg_presence:.2f}"
        )

    return stats


def filter_by_bbox_size(
    players: list[PlayerPosition],
    config: PlayerFilterConfig,
) -> list[PlayerPosition]:
    """
    Filter players by bounding box size.

    Court players are closer to the camera, so they have larger bounding boxes.
    Spectators in background have smaller boxes.

    Args:
        players: List of player detections.
        config: Filter configuration.

    Returns:
        Players with bboxes above minimum size thresholds.
    """
    filtered = []
    for p in players:
        bbox_area = p.width * p.height
        if bbox_area >= config.min_bbox_area and p.height >= config.min_bbox_height:
            filtered.append(p)

    if len(filtered) < len(players):
        logger.debug(
            f"Bbox size filter: {len(players)} -> {len(filtered)} "
            f"(removed {len(players) - len(filtered)} small detections)"
        )

    return filtered


def select_top_k_by_size(
    players: list[PlayerPosition],
    k: int,
) -> list[PlayerPosition]:
    """
    Select top K players by bounding box area.

    For beach volleyball, we expect 4 players. Taking the largest K
    detections reliably filters out background people.

    Args:
        players: List of player detections.
        k: Maximum number of players to keep.

    Returns:
        Top K players by bbox area.
    """
    if len(players) <= k:
        return players

    # Sort by bbox area (descending) and take top K
    sorted_players = sorted(
        players,
        key=lambda p: p.width * p.height,
        reverse=True,
    )

    logger.debug(
        f"Top-K selection: {len(players)} -> {k} "
        f"(removed {len(players) - k} smaller detections)"
    )

    return sorted_players[:k]


def compute_court_split(
    ball_positions: list[BallPosition],
    config: PlayerFilterConfig,
    player_positions: list[PlayerPosition] | None = None,
) -> float | None:
    """
    Compute the Y-coordinate that splits the court into near/far teams.

    Camera is always behind baseline, so teams are split by a horizontal line.

    Primary method: Use player positions to find the net. Players cluster on
    opposite sides of the net, creating a gap with minimum player density.

    Fallback: Use ball trajectory center if player data is insufficient.

    Args:
        ball_positions: Ball positions from tracking.
        config: Filter configuration.
        player_positions: All player positions (preferred for split calculation).

    Returns:
        Y-coordinate that splits the court (0-1 normalized), or None if insufficient data.
    """
    # Primary method: find net by clustering players by bbox size
    # Near team = larger bboxes (closer to camera), Far team = smaller bboxes
    if player_positions and len(player_positions) >= 20:
        split_y = _find_net_from_bbox_clustering(
            player_positions, config.players_per_team
        )
        if split_y is not None:
            logger.debug(f"Court split from bbox clustering: y={split_y:.3f}")
            return split_y

    # Secondary method: use ball trajectory direction changes
    # Ball crosses net multiple times during a rally
    if ball_positions:
        split_y = _find_net_from_ball_crossings(ball_positions)
        if split_y is not None:
            logger.debug(f"Court split from ball crossings: y={split_y:.3f}")
            return split_y

    # Fallback: use ball trajectory center
    confident = [
        p.y for p in ball_positions
        if p.confidence >= config.ball_confidence_threshold
    ]

    if len(confident) < config.min_ball_points:
        return None

    # Use ball trajectory Y-range center as court center (net position)
    y_center = (max(confident) + min(confident)) / 2

    logger.debug(f"Court split from ball trajectory center: y={y_center:.3f}")

    return y_center


def _find_net_from_ball_crossings(
    ball_positions: list[BallPosition],
    confidence_threshold: float = 0.4,
) -> float | None:
    """
    Find net Y position from ball trajectory direction changes.

    When ball crosses net, Y direction typically changes. The net position
    is estimated as the median Y of direction change points.

    Args:
        ball_positions: Ball positions from tracking.
        confidence_threshold: Minimum confidence for valid positions.

    Returns:
        Estimated net Y position (0-1), or None if insufficient data.
    """
    # Filter and sort
    confident = [bp for bp in ball_positions if bp.confidence >= confidence_threshold]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 10:
        return None

    # Find Y direction reversals
    crossing_ys: list[float] = []
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

        # Direction reversal (ball went up, now going down or vice versa)
        if (dy > 0.005 and prev_dy < -0.005) or (dy < -0.005 and prev_dy > 0.005):
            crossing_ys.append(prev.y)

    if len(crossing_ys) >= 2:
        return float(np.median(crossing_ys))

    return None


def identify_teams_by_court_side(
    track_stats: dict[int, TrackStats],
    net_y: float,
    min_side_fraction: float = 0.60,
) -> tuple[set[int], set[int]]:
    """
    Identify teams by where players spend most of their time.

    Beach volleyball players stay primarily on one side of the net.
    A player belongs to a team if they spend 60%+ of their time on that side.

    Args:
        track_stats: Statistics for each track (must have avg_y).
        net_y: Y-coordinate of the net (0-1).
        min_side_fraction: Minimum fraction of time on one side (default 60%).

    Returns:
        Tuple of (near_team, far_team) sets of track IDs.
        Near team: y > net_y (closer to camera)
        Far team: y <= net_y (further from camera)
    """
    near_team: set[int] = set()
    far_team: set[int] = set()

    for track_id, stats in track_stats.items():
        if stats.is_likely_referee:
            continue

        # Determine which side the player is primarily on
        # Since we only have avg_y, use that (could be improved with per-frame analysis)
        if stats.avg_y > net_y:
            near_team.add(track_id)
        else:
            far_team.add(track_id)

    logger.debug(
        f"Team identification by court side (net_y={net_y:.2f}): "
        f"near={len(near_team)}, far={len(far_team)}"
    )

    return near_team, far_team


def _find_net_from_bbox_clustering(
    player_positions: list[PlayerPosition],
    players_per_team: int = 2,
) -> float | None:
    """
    Find the net Y position using bbox size to identify near/far teams.

    Near team players have LARGER bboxes (closer to camera).
    Far team players have SMALLER bboxes (further from camera).

    The net is the boundary between the Y ranges of these two groups.

    Args:
        player_positions: All player positions across frames.
        players_per_team: Expected players per team (2 for beach volleyball).

    Returns:
        Y-coordinate of the net (0-1 normalized), or None if unclear.
    """
    # Compute average bbox size and Y per track
    track_sizes: dict[int, list[float]] = {}
    track_y_values: dict[int, list[float]] = {}

    for p in player_positions:
        if p.track_id < 0:
            continue
        if p.track_id not in track_sizes:
            track_sizes[p.track_id] = []
            track_y_values[p.track_id] = []
        track_sizes[p.track_id].append(p.width * p.height)
        track_y_values[p.track_id].append(p.y)

    if len(track_sizes) < players_per_team * 2:
        logger.debug(f"Not enough tracks ({len(track_sizes)}) for bbox clustering")
        return None

    # Compute average size and Y per track
    track_avg_size = {tid: float(np.mean(sizes)) for tid, sizes in track_sizes.items()}
    track_avg_y = {tid: float(np.mean(ys)) for tid, ys in track_y_values.items()}

    # Sort tracks by size (descending)
    sorted_tracks = sorted(track_avg_size.keys(), key=lambda t: track_avg_size[t], reverse=True)

    # Take top tracks (expected court players)
    num_players = players_per_team * 2
    top_tracks = sorted_tracks[:num_players]

    if len(top_tracks) < num_players:
        return None

    # Split into near (larger bbox) and far (smaller bbox) teams
    near_tracks = top_tracks[:players_per_team]  # Largest bboxes
    far_tracks = top_tracks[players_per_team:]   # Smaller bboxes

    # Get Y positions for each team
    near_ys = [track_avg_y[t] for t in near_tracks]
    far_ys = [track_avg_y[t] for t in far_tracks]

    # Near team should have HIGHER Y (closer to camera = bottom of frame)
    # Far team should have LOWER Y (further from camera = top of frame)
    max_far_y = max(far_ys)
    min_near_y = min(near_ys)

    # Log the team positions for debugging
    logger.debug(
        f"Bbox clustering: near_ys={[f'{y:.2f}' for y in near_ys]}, "
        f"far_ys={[f'{y:.2f}' for y in far_ys]}"
    )

    # Validate: far team should be above (lower Y) near team
    if max_far_y >= min_near_y:
        # Teams overlap or are inverted - try using median instead
        all_ys = near_ys + far_ys
        split_y = float(np.median(all_ys))
        logger.debug(
            f"Teams overlap (max_far={max_far_y:.2f} >= min_near={min_near_y:.2f}), "
            f"using median y={split_y:.2f}"
        )
        return split_y

    # Split at the midpoint between teams
    split_y = (max_far_y + min_near_y) / 2

    logger.debug(
        f"Net found between teams: y={split_y:.3f} "
        f"(far_max={max_far_y:.2f}, near_min={min_near_y:.2f})"
    )

    return split_y


def select_two_teams(
    players: list[PlayerPosition],
    split_y: float,
    players_per_team: int,
    primary_tracks: set[int] | None = None,
    strict_primary: bool = True,
    hysteresis: float = 0.0,
    track_team_history: dict[int, int] | None = None,
) -> tuple[list[PlayerPosition], dict[int, int]]:
    """
    Select players using two-team filtering (near/far split) with hysteresis.

    Camera is behind baseline, so teams are split by Y:
    - Near team: y > split_y (closer to camera, larger bboxes)
    - Far team: y <= split_y (further from camera, smaller bboxes)

    Hysteresis prevents flickering when players are near the boundary.
    Players must cross split_y +/- hysteresis to switch teams.

    Args:
        players: List of player detections for a frame.
        split_y: Y-coordinate that splits near/far teams.
        players_per_team: Number of players per team (2 for beach).
        primary_tracks: Optional set of stable track IDs to prioritize.
        strict_primary: If True and primary_tracks is set, ONLY select primary tracks.
            This filters out referees who pass other filters but aren't primary.
        hysteresis: Distance from boundary required to switch teams.
        track_team_history: Dictionary mapping track_id -> last team (0=near, 1=far).
            Updated in-place with new assignments.

    Returns:
        Tuple of (selected players, updated track_team_history).
    """
    # If we have primary tracks and strict mode, only consider primary track players
    if primary_tracks and strict_primary:
        players = [p for p in players if p.track_id in primary_tracks]
        if not players:
            return [], track_team_history or {}

    # Initialize history if not provided
    if track_team_history is None:
        track_team_history = {}

    if len(players) <= players_per_team * 2:
        # Still update history for all players
        for p in players:
            if p.y > split_y:
                track_team_history[p.track_id] = 0  # Near team
            else:
                track_team_history[p.track_id] = 1  # Far team
        return players, track_team_history

    # Split by Y (near/far) with hysteresis
    near_team: list[PlayerPosition] = []
    far_team: list[PlayerPosition] = []

    for p in players:
        prev_team = track_team_history.get(p.track_id)

        if prev_team is not None and hysteresis > 0:
            # Only switch teams if clearly past boundary + hysteresis
            if prev_team == 0:  # Was near team
                if p.y <= split_y - hysteresis:
                    # Crossed to far side
                    far_team.append(p)
                    track_team_history[p.track_id] = 1
                else:
                    # Stay on near side
                    near_team.append(p)
            else:  # Was far team (prev_team == 1)
                if p.y > split_y + hysteresis:
                    # Crossed to near side
                    near_team.append(p)
                    track_team_history[p.track_id] = 0
                else:
                    # Stay on far side
                    far_team.append(p)
        else:
            # No history or no hysteresis, use hard boundary
            if p.y > split_y:
                near_team.append(p)
                track_team_history[p.track_id] = 0
            else:
                far_team.append(p)
                track_team_history[p.track_id] = 1

    def select_from_side(
        side_players: list[PlayerPosition],
        k: int,
    ) -> list[PlayerPosition]:
        """Select top K from one side, prioritizing primary tracks then bbox size."""
        if len(side_players) <= k:
            return side_players

        # If we have primary tracks, prioritize them
        if primary_tracks:
            primary_players = [p for p in side_players if p.track_id in primary_tracks]
            other_players = [p for p in side_players if p.track_id not in primary_tracks]

            # Sort each group by bbox size
            primary_players.sort(key=lambda p: p.width * p.height, reverse=True)
            other_players.sort(key=lambda p: p.width * p.height, reverse=True)

            # Take from primary first, then fill with others if needed
            result = primary_players[:k]
            if len(result) < k:
                result.extend(other_players[: k - len(result)])
            return result

        # No primary tracks, just sort by size
        sorted_players = sorted(
            side_players,
            key=lambda p: p.width * p.height,
            reverse=True,
        )
        return sorted_players[:k]

    # Select from each side
    selected_near = select_from_side(near_team, players_per_team)
    selected_far = select_from_side(far_team, players_per_team)

    result = selected_near + selected_far

    logger.debug(
        f"Two-team selection (y={split_y:.2f}, hysteresis={hysteresis:.3f}): "
        f"{len(players)} -> {len(result)} "
        f"(near: {len(selected_near)}, far: {len(selected_far)})"
    )

    return result, track_team_history


def compute_ball_proximity_scores(
    all_positions: list[PlayerPosition],
    ball_positions: list[BallPosition],
    config: PlayerFilterConfig,
) -> dict[int, float]:
    """
    Compute how often each track appears near ball positions.

    Players who are frequently near the ball are more likely to be real
    court players vs. spectators.

    Args:
        all_positions: All player positions across all frames.
        ball_positions: Ball positions from tracking.
        config: Filter configuration.

    Returns:
        Dictionary mapping track_id to proximity score (0-1).
    """
    if not ball_positions:
        return {}

    # Index ball positions by frame
    ball_by_frame: dict[int, BallPosition] = {}
    for bp in ball_positions:
        if bp.confidence >= config.ball_confidence_threshold:
            ball_by_frame[bp.frame_number] = bp

    # Count proximity for each track
    track_near_count: dict[int, int] = {}
    track_total_count: dict[int, int] = {}

    radius = config.ball_proximity_radius

    for p in all_positions:
        if p.track_id < 0:
            continue

        track_total_count[p.track_id] = track_total_count.get(p.track_id, 0) + 1

        # Check if near ball in this frame
        ball = ball_by_frame.get(p.frame_number)
        if ball:
            dist = ((p.x - ball.x) ** 2 + (p.y - ball.y) ** 2) ** 0.5
            if dist <= radius:
                track_near_count[p.track_id] = track_near_count.get(p.track_id, 0) + 1

    # Compute proximity score (fraction of appearances near ball)
    scores: dict[int, float] = {}
    for track_id, total in track_total_count.items():
        near = track_near_count.get(track_id, 0)
        scores[track_id] = near / total if total > 0 else 0.0

    if scores:
        avg_score = sum(scores.values()) / len(scores)
        logger.debug(
            f"Ball proximity scores: {len(scores)} tracks, avg={avg_score:.2f}"
        )

    return scores


def compute_track_stats(
    all_positions: list[PlayerPosition],
    total_frames: int,
) -> dict[int, TrackStats]:
    """
    Compute statistics for each track across all frames.

    Args:
        all_positions: All player positions from all frames.
        total_frames: Total number of frames in the video.

    Returns:
        Dictionary mapping track_id to TrackStats.
    """
    # Group positions by track ID
    track_positions: dict[int, list[PlayerPosition]] = {}
    for p in all_positions:
        if p.track_id < 0:
            continue  # Skip untracked detections
        if p.track_id not in track_positions:
            track_positions[p.track_id] = []
        track_positions[p.track_id].append(p)

    # Compute stats for each track
    stats: dict[int, TrackStats] = {}
    for track_id, positions in track_positions.items():
        # Count unique frames (track might have multiple detections per frame)
        frame_numbers = [p.frame_number for p in positions]
        unique_frames = len(set(frame_numbers))

        # Average bbox area
        avg_area = sum(p.width * p.height for p in positions) / len(positions)

        # Average confidence
        avg_conf = sum(p.confidence for p in positions) / len(positions)

        # First and last frame
        first_frame = min(frame_numbers)
        last_frame = max(frame_numbers)

        # Position spread (movement) - geometric mean of X/Y std dev
        # Players move around; referees stand in one spot
        x_positions = np.array([p.x for p in positions])
        y_positions = np.array([p.y for p in positions])
        x_std = float(np.std(x_positions)) if len(positions) > 1 else 0.0
        y_std = float(np.std(y_positions)) if len(positions) > 1 else 0.0
        avg_x = float(np.mean(x_positions))
        avg_y = float(np.mean(y_positions))
        # Geometric mean handles case where movement is mostly in one direction
        position_spread = float(np.sqrt(x_std * y_std)) if x_std > 0 and y_std > 0 else max(x_std, y_std)

        stats[track_id] = TrackStats(
            track_id=track_id,
            frame_count=unique_frames,
            total_frames=total_frames,
            avg_bbox_area=avg_area,
            avg_confidence=avg_conf,
            first_frame=first_frame,
            last_frame=last_frame,
            position_spread=position_spread,
            avg_x=avg_x,
            avg_y=avg_y,
            x_std=x_std,
            y_std=y_std,
        )

    return stats


def stabilize_track_ids(
    positions: list[PlayerPosition],
    config: PlayerFilterConfig,
) -> tuple[list[PlayerPosition], dict[int, int]]:
    """
    Stabilize track IDs by merging tracks that represent the same player.

    When ByteTrack loses an association, it assigns a new track ID. This function
    detects when a track ends and a new one starts nearby, and merges them.

    Algorithm:
    1. Build track timelines (first/last frame, last position)
    2. For each track that starts after frame 0:
       - Find tracks that ended within max_gap_frames at similar position
       - If found, remap new track ID to old track ID
    3. Apply remapping to all positions

    Args:
        positions: All player positions (will be modified in place).
        config: Filter configuration with merge thresholds.

    Returns:
        Tuple of (modified positions, id_mapping dict).
        The id_mapping maps old_id -> new_id for merged tracks.
    """
    if not config.stabilize_track_ids or not positions:
        return positions, {}

    # Build track info: first/last frame, first/last position
    track_info: dict[int, dict] = {}
    for p in positions:
        if p.track_id < 0:
            continue

        if p.track_id not in track_info:
            track_info[p.track_id] = {
                "first_frame": p.frame_number,
                "last_frame": p.frame_number,
                "first_pos": (p.x, p.y),
                "last_pos": (p.x, p.y),
            }
        else:
            info = track_info[p.track_id]
            if p.frame_number < info["first_frame"]:
                info["first_frame"] = p.frame_number
                info["first_pos"] = (p.x, p.y)
            if p.frame_number > info["last_frame"]:
                info["last_frame"] = p.frame_number
                info["last_pos"] = (p.x, p.y)

    if not track_info:
        return positions, {}

    # Find tracks to merge: new tracks that start after an old track ends nearby
    id_mapping: dict[int, int] = {}  # new_id -> canonical_id

    # Track effective frame ranges for canonical tracks (updated as we merge)
    # This prevents merging tracks that would overlap with already-merged tracks
    canonical_frame_ranges: dict[int, tuple[int, int]] = {}  # canonical_id -> (first, last)

    # Sort tracks by first_frame
    tracks_by_start = sorted(
        track_info.items(),
        key=lambda x: x[1]["first_frame"],
    )

    for new_id, new_info in tracks_by_start:
        if new_id in id_mapping:
            continue  # Already merged

        new_first_frame = new_info["first_frame"]
        new_last_frame = new_info["last_frame"]
        new_first_pos = new_info["first_pos"]

        # Look for tracks that ended shortly before this one started
        best_match: int | None = None
        best_distance = float("inf")

        for old_id, old_info in track_info.items():
            if old_id == new_id:
                continue
            if old_id in id_mapping.values():
                # This track is already a canonical target
                canonical_id = old_id
            elif old_id in id_mapping:
                # This track was merged into another
                canonical_id = id_mapping[old_id]
            else:
                canonical_id = old_id

            # Use old_info (the track being compared) for position/frame checks
            # This enables chain merging: if 41->1 and 99 starts near where 41 ended,
            # we use 41's last position (in old_info) to match 99, then map 99->1
            old_last_frame = old_info["last_frame"]
            old_last_pos = old_info["last_pos"]

            # Check frame gap (old track must end before or at new track start)
            frame_gap = new_first_frame - old_last_frame
            if frame_gap < 0 or frame_gap > config.max_gap_frames:
                continue

            # Check if merging would create overlapping positions with canonical track
            # This prevents merging two different players into the same track ID
            if canonical_id in canonical_frame_ranges:
                canon_first, canon_last = canonical_frame_ranges[canonical_id]
                # New track overlaps with canonical's effective range
                if new_first_frame <= canon_last and new_last_frame >= canon_first:
                    logger.debug(
                        f"Skipping merge {new_id} -> {canonical_id}: "
                        f"would overlap (new: {new_first_frame}-{new_last_frame}, "
                        f"canonical: {canon_first}-{canon_last})"
                    )
                    continue

            # Check position distance (use squared distance to avoid sqrt)
            dx = new_first_pos[0] - old_last_pos[0]
            dy = new_first_pos[1] - old_last_pos[1]
            distance_sq = dx * dx + dy * dy

            # Allow more distance for larger gaps (players can move during occlusion)
            # Base distance + additional distance per frame of gap
            adaptive_max_distance = (
                config.max_merge_distance
                + frame_gap * config.merge_distance_per_frame
            )
            max_distance_sq = adaptive_max_distance * adaptive_max_distance

            if distance_sq <= max_distance_sq and distance_sq < best_distance:
                best_distance = distance_sq
                best_match = canonical_id

        if best_match is not None and best_match != new_id:
            id_mapping[new_id] = best_match
            # Update canonical's effective frame range to include merged track
            if best_match in canonical_frame_ranges:
                canon_first, canon_last = canonical_frame_ranges[best_match]
                canonical_frame_ranges[best_match] = (
                    min(canon_first, new_first_frame),
                    max(canon_last, new_last_frame),
                )
            else:
                # Initialize canonical range from original track + merged track
                canon_info = track_info[best_match]
                canonical_frame_ranges[best_match] = (
                    min(canon_info["first_frame"], new_first_frame),
                    max(canon_info["last_frame"], new_last_frame),
                )
            logger.debug(
                f"Merging track {new_id} -> {best_match} "
                f"(gap={new_first_frame - track_info[best_match]['last_frame']} frames, "
                f"dist={best_distance ** 0.5:.3f})"
            )

    if not id_mapping:
        return positions, {}

    # Apply remapping to all positions
    remapped_count = 0
    for p in positions:
        if p.track_id in id_mapping:
            p.track_id = id_mapping[p.track_id]
            remapped_count += 1

    logger.info(
        f"Track ID stabilization: merged {len(id_mapping)} tracks, "
        f"remapped {remapped_count} positions"
    )

    return positions, id_mapping


def split_tracks_at_jumps(
    positions: list[PlayerPosition],
    max_displacement: float = 0.25,
    max_frame_gap: int = 3,
) -> tuple[list[PlayerPosition], int]:
    """Split tracks at large inter-frame position jumps.

    When BoT-SORT makes an incorrect association (ID switch), the tracked
    position shows a large displacement in a short time. This function
    detects these jumps and splits the track, creating new track IDs for
    the portion after the jump.

    Must run BEFORE stabilize_track_ids() to prevent merging bad associations.

    Args:
        positions: All player positions (modified in place).
        max_displacement: Max normalized distance for a jump to trigger split.
        max_frame_gap: Only split on jumps within this many frames
            (larger gaps may be legitimate recovery after occlusion).

    Returns:
        Tuple of (modified positions, number of splits).
    """
    if not positions:
        return positions, 0

    # Group by track_id
    tracks: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id < 0:
            continue
        tracks.setdefault(p.track_id, []).append(p)

    if not tracks:
        return positions, 0

    max_id = max(tracks.keys())
    next_id = max_id + 1
    total_splits = 0

    for track_id, track_pos in tracks.items():
        track_pos.sort(key=lambda p: p.frame_number)

        i = 1
        while i < len(track_pos):
            prev = track_pos[i - 1]
            curr = track_pos[i]

            frame_gap = curr.frame_number - prev.frame_number
            if frame_gap <= 0 or frame_gap > max_frame_gap:
                i += 1
                continue

            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > max_displacement:
                new_id = next_id
                next_id += 1
                total_splits += 1

                # Reassign all positions from this point onward in this segment
                for j in range(i, len(track_pos)):
                    track_pos[j].track_id = new_id

                logger.info(
                    f"Split track {track_id} at frame {curr.frame_number}: "
                    f"jump={dist:.3f} in {frame_gap} frames -> new track {new_id}"
                )
                # Don't continue scanning this segment; the remaining positions
                # now belong to new_id and will be checked if needed
                break

            i += 1

    if total_splits:
        logger.info(f"Track jump splitting: {total_splits} tracks split")

    return positions, total_splits


def detect_referee_tracks(
    track_stats: dict[int, TrackStats],
    ball_positions: list[BallPosition] | None,
    config: PlayerFilterConfig,
) -> set[int]:
    """
    Detect referee tracks by movement patterns.

    Referees have distinctive characteristics:
    1. Position near sidelines (x < 0.08 or x > 0.92)
    2. Movement parallel to net (high X variance, low Y variance)
    3. Outside main ball trajectory Y-range
    4. Low ball proximity (they don't interact with the ball)

    Args:
        track_stats: Statistics for each track.
        ball_positions: Ball positions for trajectory analysis (optional).
        config: Filter configuration with referee thresholds.

    Returns:
        Set of track IDs that are likely referees.
    """
    referee_tracks: set[int] = set()

    # Compute ball trajectory Y-range if available
    # Use large margin (0.25) because ball may stay on one side during parts of rally
    # but players are on both sides of the court
    ball_y_min = 0.0
    ball_y_max = 1.0
    if ball_positions:
        confident_ys = [bp.y for bp in ball_positions if bp.confidence >= 0.4]
        if len(confident_ys) >= 10:
            ball_y_min = min(confident_ys)
            ball_y_max = max(confident_ys)
            # Add large margin - court players can be far from ball trajectory
            # Volleyball courts have players on both near and far sides
            ball_y_min = max(0.0, ball_y_min - 0.30)
            ball_y_max = min(1.0, ball_y_max + 0.30)

    for track_id, stats in track_stats.items():
        reasons: list[str] = []

        # Check 1: Near sidelines (use <= for inclusive threshold)
        is_near_sideline = (
            stats.avg_x <= config.referee_sideline_threshold
            or stats.avg_x >= (1.0 - config.referee_sideline_threshold)
        )
        if is_near_sideline:
            reasons.append(f"sideline (x={stats.avg_x:.2f})")

        # Check 2: Movement parallel to net (high X/Y ratio)
        # Referees pace along the net line but don't move toward/away from it
        is_parallel_mover = (
            stats.movement_ratio >= config.referee_movement_ratio_min
            and stats.x_std > 0.01  # Must have some movement (not stationary)
        )
        if is_parallel_mover:
            reasons.append(f"parallel movement (ratio={stats.movement_ratio:.1f})")

        # Check 3: Outside ball trajectory Y-range
        # Referees stand to the side, not in the active play area
        is_outside_ball_range = (
            stats.avg_y < ball_y_min or stats.avg_y > ball_y_max
        )
        if is_outside_ball_range:
            reasons.append(f"outside ball Y (y={stats.avg_y:.2f})")

        # Check 4: Low ball proximity
        has_low_ball_proximity = stats.ball_proximity_score <= config.referee_ball_proximity_max

        # Check 5: Low Y variance - referees don't move up/down court like players
        # Players move 5-15% in Y, referees < 3%
        has_low_y_variance = stats.y_std < config.referee_y_std_max
        if has_low_y_variance:
            reasons.append(f"low Y variance (y_std={stats.y_std:.3f})")

        # A track is likely a referee if:
        # - Near sideline AND (parallel mover OR low ball proximity OR low Y variance)
        # - OR outside ball range AND low ball proximity AND low Y variance
        # - OR low Y variance AND positioned on sideline area (x <= 0.25) - aggressive filter
        # - OR near sideline AND low Y variance (simple case for line referees)
        is_likely_referee = (
            (is_near_sideline and (is_parallel_mover or has_low_ball_proximity or has_low_y_variance))
            or (is_outside_ball_range and has_low_ball_proximity and has_low_y_variance)
            or (has_low_y_variance and stats.avg_x <= 0.25)  # Removed ball proximity requirement
            or (is_near_sideline and has_low_y_variance)  # Simple case: sideline + no vertical movement
        )

        if is_likely_referee:
            stats.is_likely_referee = True
            referee_tracks.add(track_id)
            logger.info(
                f"Track {track_id} identified as likely referee: {', '.join(reasons)}, "
                f"ball_prox={stats.ball_proximity_score:.3f}"
            )

    if referee_tracks:
        logger.info(f"Detected {len(referee_tracks)} likely referee tracks: {sorted(referee_tracks)}")

    return referee_tracks


def identify_primary_tracks(
    track_stats: dict[int, TrackStats],
    config: PlayerFilterConfig,
    court_config: CourtFilterConfig | None = None,
    referee_tracks: set[int] | None = None,
) -> set[int]:
    """
    Identify primary tracks that are likely real court players.

    Hard filters (must pass all):
    1. Not on sidelines (x within primary_sideline_threshold, default 0.10-0.90)
    2. Not identified as referee
    3. Minimum presence rate (default 20%)
    4. Court presence >= threshold (if calibration available)

    Soft filters (used for ranking, relaxed when needed):
    - Stationary check: tracks with low spread AND no ball engagement are
      deprioritized but included as fallbacks when fewer than max_players active

    Selection prioritizes:
    1. Active tracks (moving or ball-engaged) with high stability
    2. Active tracks with lower stability (if needed)
    3. Stationary tracks as fallback (if still need more players)

    Args:
        track_stats: Statistics for each track.
        config: Filter configuration.
        court_config: Court filter configuration (for court presence threshold).
        referee_tracks: Set of track IDs identified as referees (to exclude).

    Returns:
        Set of track IDs that are primary tracks (up to max_players).
    """
    primary: set[tuple[int, float]] = set()  # (track_id, stability_score)
    court_cfg = court_config or CourtFilterConfig()
    referees = referee_tracks or set()

    for track_id, stats in track_stats.items():
        # HARD FILTER 0: Exclude tracks on sidelines (image coordinates)
        # Players are in the middle of the frame; referees stand on sides
        # Beach volleyball: players typically at x=0.25-0.75
        # Use configurable threshold for cameras angled toward court edges
        sideline_min = config.primary_sideline_threshold
        sideline_max = 1.0 - config.primary_sideline_threshold
        if stats.avg_x < sideline_min or stats.avg_x > sideline_max:
            logger.info(
                f"Track {track_id} excluded: avg_x={stats.avg_x:.2f} (sideline position, "
                f"threshold={sideline_min:.2f}-{sideline_max:.2f})"
            )
            continue

        # HARD FILTER 1: Exclude tracks identified as referees
        if track_id in referees or stats.is_likely_referee:
            logger.debug(f"Track {track_id} excluded: identified as referee")
            continue

        # HARD FILTER 2: Court presence (if calibration available)
        # Tracks with less than 50% court presence are definitively not players
        if stats.has_court_stats:
            if stats.court_presence_ratio < court_cfg.min_court_presence_ratio:
                logger.info(
                    f"Track {track_id} excluded: court_presence={stats.court_presence_ratio:.2f} "
                    f"< {court_cfg.min_court_presence_ratio:.2f} (not on court)"
                )
                continue

        # Compute stability score using configurable weights
        stability = compute_stability_score(stats, config)

        # Must meet presence threshold (hard filter)
        if stats.presence_rate < config.min_presence_rate:
            continue

        # Track passes hard filters - add with stability score for later ranking
        primary.add((track_id, stability))

    # Convert to list and sort by stability (descending)
    candidates = sorted(primary, key=lambda x: x[1], reverse=True)

    # Separate candidates into "active" (moving or ball-engaged) and "stationary"
    # Stationary tracks are fallback candidates when we need more players
    active_candidates: list[tuple[int, float]] = []
    stationary_candidates: list[tuple[int, float]] = []

    for tid, stability in candidates:
        stats = track_stats[tid]
        is_stationary = stats.position_spread < config.min_position_spread_for_primary
        has_ball_engagement = stats.ball_proximity_score >= config.min_ball_proximity_for_stationary

        if is_stationary and not has_ball_engagement:
            stationary_candidates.append((tid, stability))
        else:
            active_candidates.append((tid, stability))

    # Split active candidates into stable (pass threshold) and unstable
    stable_idx = 0
    for i, (_, stab) in enumerate(active_candidates):
        if stab < config.min_stability_score:
            stable_idx = i
            break
    else:
        stable_idx = len(active_candidates)

    # Select tracks: prioritize stable active tracks
    selected: list[tuple[int, float]] = list(active_candidates[:stable_idx])

    # If we have fewer than max_players, include lower-stability active tracks
    if len(selected) < config.max_players:
        needed = config.max_players - len(selected)
        for tid, stab in active_candidates[stable_idx : stable_idx + needed]:
            logger.info(
                f"Track {tid} included despite low stability={stab:.3f} "
                f"(need {config.max_players} players, only {len(selected)} stable)"
            )
            selected.append((tid, stab))

    # If we still need more players, include stationary candidates as fallback
    if len(selected) < config.max_players and stationary_candidates:
        needed = config.max_players - len(selected)
        for tid, stab in stationary_candidates[:needed]:
            stats = track_stats[tid]
            logger.info(
                f"Track {tid} included despite low movement: spread={stats.position_spread:.4f}, "
                f"ball_prox={stats.ball_proximity_score:.3f} "
                f"(need {config.max_players} players, only {len(selected)} active)"
            )
            selected.append((tid, stab))

    # Apply max_players limit if we have too many
    if len(selected) > config.max_players:
        # Score by player behavior (ball proximity, movement) rather than just stability
        def player_score(tid: int) -> float:
            s = track_stats[tid]
            spread_normalized = min(s.position_spread / 0.05, 1.0)
            return (
                0.5 * s.ball_proximity_score
                + 0.3 * spread_normalized
                + 0.2 * s.presence_rate
            )

        scored = [(tid, player_score(tid)) for tid, _ in selected]
        scored.sort(key=lambda x: x[1], reverse=True)

        kept_ids = {tid for tid, _ in scored[:config.max_players]}
        excluded_ids = {tid for tid, _ in scored[config.max_players:]}

        logger.info(
            f"Limiting primary tracks: {len(selected)} candidates -> {config.max_players} "
            f"(excluded tracks {sorted(excluded_ids)} with lower player scores)"
        )
        for tid, score in scored:
            s = track_stats[tid]
            status = "KEPT" if tid in kept_ids else "EXCLUDED"
            logger.info(
                f"  Track {tid}: score={score:.3f} (ball={s.ball_proximity_score:.2f}, "
                f"spread={s.position_spread:.4f}, presence={s.presence_rate:.2f}) [{status}]"
            )
        selected = [(tid, stab) for tid, stab in selected if tid in kept_ids]

    # Extract just the track IDs
    result: set[int] = {tid for tid, _ in selected}

    if result:
        court_info = ""
        if any(track_stats[tid].has_court_stats for tid in result):
            court_info = f", court >= {court_cfg.min_court_presence_ratio:.0%}"
        # Count how many are stationary fallbacks
        stationary_count = sum(
            1 for tid in result
            if track_stats[tid].position_spread < config.min_position_spread_for_primary
            and track_stats[tid].ball_proximity_score < config.min_ball_proximity_for_stationary
        )
        fallback_info = f", {stationary_count} stationary fallback(s)" if stationary_count else ""
        logger.info(
            f"Identified {len(result)} primary tracks: {sorted(result)} "
            f"(presence >= {config.min_presence_rate:.0%}{court_info}{fallback_info})"
        )
        # Log stats for primary tracks
        for tid in sorted(result):
            stats = track_stats[tid]
            court_str = ""
            if stats.has_court_stats:
                court_str = f", court={stats.court_presence_ratio:.2f}, interior={stats.interior_ratio:.2f}"
            logger.info(
                f"  Track {tid}: spread={stats.position_spread:.4f}, "
                f"presence={stats.presence_rate:.2f}, bbox={stats.avg_bbox_area:.4f}, "
                f"stability={compute_stability_score(stats, config):.3f}{court_str}"
            )

    return result


def select_with_track_priority(
    players: list[PlayerPosition],
    k: int,
    primary_tracks: set[int],
    strict_primary: bool = True,
) -> list[PlayerPosition]:
    """
    Select top K players, prioritizing primary (stable) tracks.

    If a player has a primary track ID, they are kept before considering
    non-primary tracks. This prevents flickering between real players
    and momentarily large spectators/objects.

    Args:
        players: List of player detections for a frame.
        k: Maximum number of players to keep.
        primary_tracks: Set of track IDs that are primary.
        strict_primary: If True, ONLY return primary tracks (no fillers).
            This prevents flickering when a player is temporarily undetected.

    Returns:
        Up to K players, preferring primary tracks.
    """
    if not primary_tracks:
        # No primary tracks identified, fall back to size-based selection
        return select_top_k_by_size(players, k)

    # In strict mode, only return primary tracks
    if strict_primary:
        primary_players = [p for p in players if p.track_id in primary_tracks]
        return primary_players[:k]

    if len(players) <= k:
        return players

    # Separate primary and non-primary players
    primary_players = [p for p in players if p.track_id in primary_tracks]
    other_players = [p for p in players if p.track_id not in primary_tracks]

    # Sort both by size
    primary_players.sort(key=lambda p: p.width * p.height, reverse=True)
    other_players.sort(key=lambda p: p.width * p.height, reverse=True)

    # Take all primary players (up to k), then fill with others
    result = primary_players[:k]
    remaining_slots = k - len(result)

    if remaining_slots > 0:
        result.extend(other_players[:remaining_slots])

    if len(result) < len(players):
        primary_kept = len([p for p in result if p.track_id in primary_tracks])
        logger.debug(
            f"Track priority selection: {len(players)} -> {len(result)} "
            f"({primary_kept} primary, {len(result) - primary_kept} other)"
        )

    return result


def compute_play_area(
    ball_positions: list[BallPosition],
    config: PlayerFilterConfig,
) -> np.ndarray | None:
    """
    Compute convex hull of ball trajectory with padding.

    Args:
        ball_positions: List of ball detections from BallTracker.
        config: Filter configuration.

    Returns:
        Hull vertices as Nx2 numpy array of normalized coordinates,
        or None if not enough confident ball positions.
    """
    # Filter to confident detections
    confident_positions = [
        (p.x, p.y)
        for p in ball_positions
        if p.confidence >= config.ball_confidence_threshold
    ]

    if len(confident_positions) < config.min_ball_points:
        logger.debug(
            f"Not enough confident ball positions ({len(confident_positions)}) "
            f"for convex hull (need {config.min_ball_points})"
        )
        return None

    points = np.array(confident_positions)

    # Handle edge case: all points collinear or insufficient
    try:
        hull = ConvexHull(points)
    except QhullError as e:
        logger.debug(f"ConvexHull failed (points may be collinear): {e}")
        return None

    hull_points = points[hull.vertices]

    # Expand hull by padding (move each vertex outward from centroid)
    centroid = hull_points.mean(axis=0)
    expanded = centroid + (hull_points - centroid) * (1 + config.hull_padding)

    # Clip to valid range [0, 1]
    expanded = np.clip(expanded, 0.0, 1.0)

    logger.debug(
        f"Computed play area from {len(confident_positions)} ball positions, "
        f"{len(hull_points)} hull vertices"
    )

    return np.asarray(expanded)


def filter_by_play_area(
    players: list[PlayerPosition],
    play_area: np.ndarray | None,
) -> list[PlayerPosition]:
    """
    Filter players to those inside the ball trajectory play area.

    Args:
        players: List of player detections.
        play_area: Hull vertices from compute_play_area(), or None to skip.

    Returns:
        Players inside the play area.
    """
    if play_area is None or len(players) == 0:
        return players

    # Use Delaunay triangulation for efficient point-in-hull tests
    try:
        delaunay = Delaunay(play_area)
    except QhullError as e:
        logger.debug(f"Delaunay triangulation failed: {e}, skipping play area filter")
        return players

    filtered = []
    for player in players:
        point = np.array([player.x, player.y])
        if delaunay.find_simplex(point) >= 0:
            filtered.append(player)

    if len(filtered) < len(players):
        logger.debug(
            f"Play area filter: {len(players)} -> {len(filtered)} "
            f"(removed {len(players) - len(filtered)} outside play area)"
        )

    return filtered


def should_use_ball_filtering(
    ball_positions: list[BallPosition],
    total_frames: int,
    config: PlayerFilterConfig,
) -> bool:
    """
    Determine if ball-based filtering should be used based on detection quality.

    Args:
        ball_positions: List of ball detections.
        total_frames: Total number of frames in the video segment.
        config: Filter configuration.

    Returns:
        True if ball detection quality is sufficient for filtering.
    """
    if total_frames == 0:
        return False

    # Count confident detections
    confident_count = sum(
        1 for p in ball_positions if p.confidence >= config.ball_confidence_threshold
    )

    detection_rate = confident_count / total_frames
    use_ball = detection_rate >= config.min_ball_detection_rate

    logger.debug(
        f"Ball detection rate: {detection_rate:.1%} "
        f"(threshold: {config.min_ball_detection_rate:.1%}), "
        f"use_ball_filtering: {use_ball}"
    )

    return use_ball


def detect_distractor_tracks(
    all_positions: list[PlayerPosition],
    primary_tracks: set[int],
    track_stats: dict[int, TrackStats],
    max_players: int = 4,
    min_coexistence_ratio: float = 0.80,
) -> set[int]:
    """Detect distractor tracks using the hard 4-player constraint.

    A distractor is a non-primary track that mostly exists during frames where
    all expected players are already accounted for. These are spectators, refs,
    or objects that passed earlier filters but co-exist with the full player set.

    Args:
        all_positions: All player positions from all frames.
        primary_tracks: Set of primary (player) track IDs.
        track_stats: Statistics for each track.
        max_players: Expected number of players (4 for beach volleyball).
        min_coexistence_ratio: Fraction of frames where the track coexists with
            all primary tracks to be classified as distractor.

    Returns:
        Set of track IDs classified as distractors.
    """
    if len(primary_tracks) < max_players:
        # Not enough primary tracks identified — can't reliably detect distractors
        return set()

    # Build per-frame track presence and per-track frame sets in one pass
    frame_tracks: dict[int, set[int]] = {}
    track_frames_lookup: dict[int, set[int]] = {}
    for p in all_positions:
        if p.track_id < 0:
            continue
        if p.frame_number not in frame_tracks:
            frame_tracks[p.frame_number] = set()
        frame_tracks[p.frame_number].add(p.track_id)
        if p.track_id not in track_frames_lookup:
            track_frames_lookup[p.track_id] = set()
        track_frames_lookup[p.track_id].add(p.frame_number)

    # Find frames where all primary tracks are present
    all_primary_frames: set[int] = set()
    for frame_num, tracks in frame_tracks.items():
        if primary_tracks <= tracks:  # All primary tracks present
            all_primary_frames.add(frame_num)

    if not all_primary_frames:
        return set()

    # Check each non-primary track
    distractors: set[int] = set()
    non_primary_ids = set(track_stats.keys()) - primary_tracks

    for track_id in non_primary_ids:
        stats = track_stats.get(track_id)
        if stats is None:
            continue

        track_frames = track_frames_lookup.get(track_id, set())
        overlap = track_frames & all_primary_frames
        coexistence_ratio = len(overlap) / len(track_frames) if track_frames else 0.0

        if coexistence_ratio >= min_coexistence_ratio:
            distractors.add(track_id)
            logger.info(
                f"Track {track_id} classified as distractor: coexists with all "
                f"{max_players} primary tracks in {coexistence_ratio:.0%} of its frames"
            )

    if distractors:
        logger.info(
            f"Detected {len(distractors)} distractor tracks: {sorted(distractors)}"
        )

    return distractors


class PlayerFilter:
    """
    Multi-signal player filter to identify court players.

    Uses multiple filtering strategies in order:
    1. Bbox size filtering - removes small background detections
    2. Play area filtering - removes spectators outside court (using ball trajectory)
    3. Two-team selection - selects players per court side (prevents near-side bias)

    Additionally supports track stability to prefer tracks that appear consistently
    across frames. This helps prevent flickering between real players and transient
    detections.

    This combination reliably filters out spectators, referees, and objects
    while ensuring both near-side and far-side players are detected.

    Usage:
        # Create filter with ball positions
        player_filter = PlayerFilter(ball_positions, total_frames, config)

        # IMPORTANT: Call analyze_tracks() before filter() for track stability to work.
        # This identifies primary (stable) tracks and computes the court split.
        player_filter.analyze_tracks(all_positions)

        # Filter each frame (uses track stability if analyze_tracks was called)
        for frame_positions in frames:
            filtered = player_filter.filter(frame_positions)

    Note:
        If analyze_tracks() is not called, filter() falls back to size-based selection
        without track stability prioritization.
    """

    def __init__(
        self,
        ball_positions: list[BallPosition] | None = None,
        total_frames: int = 0,
        config: PlayerFilterConfig | None = None,
        court_calibrator: CourtCalibrator | None = None,
        court_config: CourtFilterConfig | None = None,
    ):
        """
        Initialize player filter.

        Args:
            ball_positions: Ball tracking results for play area filtering.
            total_frames: Total frames in video segment.
            config: Filter configuration.
            court_calibrator: Optional calibrated court calibrator for court-based filtering.
            court_config: Configuration for court-based filtering.
        """
        self.config = config or PlayerFilterConfig()
        self.court_config = court_config or CourtFilterConfig()
        self.ball_positions = ball_positions or []
        self.total_frames = total_frames
        self.court_calibrator = court_calibrator

        # Track stability (computed by analyze_tracks)
        self.track_stats: dict[int, TrackStats] = {}
        self.primary_tracks: set[int] = set()
        self.distractor_tracks: set[int] = set()  # Tracks suppressed by 4-player constraint
        self._tracks_analyzed = False

        # Court position stats (computed by analyze_tracks if calibrator available)
        self.court_position_stats: dict[int, CourtPositionStats] = {}

        # Court split for two-team filtering (Y coordinate, horizontal line)
        # Camera is always behind baseline: teams split by near/far (Y axis)
        self.court_split_y: float | None = None
        self._use_two_team = False

        # Team history for hysteresis (prevents flickering at boundary)
        # Maps track_id -> last team assignment (0=near, 1=far)
        self._track_team_history: dict[int, int] = {}

        # Compute play area from ball trajectory (optional)
        self.play_area: np.ndarray | None = None
        self.use_ball_filtering = False

        if self.ball_positions:
            self.use_ball_filtering = should_use_ball_filtering(
                self.ball_positions, self.total_frames, self.config
            )

            if self.use_ball_filtering:
                self.play_area = compute_play_area(self.ball_positions, self.config)

                if self.play_area is None:
                    self.use_ball_filtering = False
                    logger.debug(
                        "Ball positions insufficient for convex hull, "
                        "skipping play area filter"
                    )

                # Compute court split for two-team filtering
                if self.config.use_two_team_filter:
                    self.court_split_y = compute_court_split(
                        self.ball_positions, self.config
                    )
                    if self.court_split_y is not None:
                        self._use_two_team = True
                        logger.info(
                            f"Two-team filter enabled: y={self.court_split_y:.3f}"
                        )

    def analyze_tracks(self, all_positions: list[PlayerPosition]) -> None:
        """
        Analyze all positions to identify stable (primary) tracks.

        Must be called before filter() for track stability to work.
        If not called, filter() falls back to size-based selection.

        Also computes:
        - Court position stats (if calibrator available)
        - Court split using player positions (more reliable than ball trajectory)

        Args:
            all_positions: All player positions from all frames.
        """
        if not all_positions:
            return

        # Compute ball proximity scores if ball data available
        ball_proximity: dict[int, float] = {}
        if self.ball_positions:
            ball_proximity = compute_ball_proximity_scores(
                all_positions, self.ball_positions, self.config
            )

        # Compute track statistics with ball proximity
        self.track_stats = compute_track_stats(all_positions, self.total_frames)

        # Add ball proximity to track stats (used for scoring, not filtering)
        for track_id, stats in self.track_stats.items():
            stats.ball_proximity_score = ball_proximity.get(track_id, 0.0)

        # Compute court position stats if calibrator available
        if self.court_calibrator is not None and self.court_calibrator.is_calibrated:
            self.court_position_stats = compute_court_position_stats(
                all_positions, self.court_calibrator, self.court_config
            )

            # Add court stats to track stats
            for track_id, court_stats in self.court_position_stats.items():
                if track_id in self.track_stats:
                    self.track_stats[track_id].court_presence_ratio = court_stats.court_presence_ratio
                    self.track_stats[track_id].interior_ratio = court_stats.interior_ratio
                    self.track_stats[track_id].has_court_stats = True

            logger.info(
                f"Court position stats computed for {len(self.court_position_stats)} tracks"
            )

        # Detect referee tracks before identifying primary tracks
        referee_tracks = detect_referee_tracks(
            self.track_stats, self.ball_positions, self.config
        )

        # Identify primary tracks (excluding referees and with court-based hard filter)
        self.primary_tracks = identify_primary_tracks(
            self.track_stats, self.config, self.court_config, referee_tracks
        )

        # Detect distractor tracks (non-players that coexist with all primary tracks)
        self.distractor_tracks = detect_distractor_tracks(
            all_positions,
            self.primary_tracks,
            self.track_stats,
            max_players=self.config.max_players,
        )

        self._tracks_analyzed = True

        # Recompute court split using player positions (more reliable than ball alone)
        # Player positions show clear clustering on opposite sides of the net
        if self.config.use_two_team_filter:
            new_split_y = compute_court_split(
                self.ball_positions, self.config, player_positions=all_positions
            )
            if new_split_y is not None:
                old_split = self.court_split_y
                self.court_split_y = new_split_y
                self._use_two_team = True
                if old_split is not None and abs(old_split - new_split_y) > 0.05:
                    logger.info(
                        f"Court split refined: y={old_split:.3f} -> y={new_split_y:.3f} "
                        f"(using player density)"
                    )
                elif old_split is None:
                    logger.info(f"Two-team filter enabled: y={new_split_y:.3f}")

        # Log with ball proximity info
        court_info = ""
        if self.court_position_stats:
            avg_court = sum(
                s.court_presence_ratio for s in self.court_position_stats.values()
            ) / len(self.court_position_stats)
            court_info = f", avg court presence: {avg_court:.2f}"

        if ball_proximity:
            avg_proximity = sum(
                s.ball_proximity_score for s in self.track_stats.values()
            ) / max(1, len(self.track_stats))
            logger.info(
                f"Track analysis: {len(self.track_stats)} tracks, "
                f"{len(self.primary_tracks)} primary, "
                f"avg ball proximity: {avg_proximity:.2f}{court_info}"
            )
        else:
            logger.info(
                f"Track analysis: {len(self.track_stats)} tracks, "
                f"{len(self.primary_tracks)} primary{court_info}"
            )

    def filter(self, players: list[PlayerPosition]) -> list[PlayerPosition]:
        """
        Filter players to court players only using multiple signals.

        Applies filters in order:
        1. Bbox size filter (removes small background detections)
        2. Play area filter (removes spectators outside court, if ball tracking available)
        3. Stationary filter (removes tracks with low position spread - referees)
        4. Two-team selection OR track priority + top-K

        Args:
            players: List of player detections for a frame.

        Returns:
            Filtered list containing only court players.
        """
        if not players:
            return players

        original_count = len(players)

        # Step 1: Filter by bbox size (removes small background detections)
        # Primary tracks are always kept - far-court players appear smaller
        filtered = filter_by_bbox_size(players, self.config)
        if self._tracks_analyzed and self.primary_tracks:
            filtered_ids = {p.track_id for p in filtered}
            # Add back primary tracks that were filtered out
            for p in players:
                if p.track_id in self.primary_tracks and p.track_id not in filtered_ids:
                    filtered.append(p)
                    filtered_ids.add(p.track_id)

        # Step 2: Filter by play area (removes spectators outside court)
        # This runs BEFORE two-team selection to ensure only court players are considered
        # Primary tracks are always kept - they are identified players, even if momentarily
        # outside the ball trajectory area (e.g., waiting for serve)
        if self.use_ball_filtering and self.play_area is not None:
            if self._tracks_analyzed and self.primary_tracks:
                in_play_area = filter_by_play_area(filtered, self.play_area)
                in_play_area_ids = {p.track_id for p in in_play_area}
                # Add primary tracks that were filtered out
                filtered = in_play_area + [
                    p for p in filtered
                    if p.track_id in self.primary_tracks
                    and p.track_id not in in_play_area_ids
                ]
            else:
                filtered = filter_by_play_area(filtered, self.play_area)

        # Step 3: Filter out distractors, stationary tracks, and likely referees
        # - Distractors: non-primary tracks that coexist with all 4 primary tracks
        # - Stationary objects (low spread AND low ball proximity) are filtered
        # - Tracks marked as likely referees are filtered (unless they're primary)
        # - Players who don't move much still engage with the ball
        if self._tracks_analyzed and self.track_stats:
            before_stationary = len(filtered)

            def is_valid_track(p: PlayerPosition) -> bool:
                # Always keep primary tracks
                if self.primary_tracks and p.track_id in self.primary_tracks:
                    return True

                # Filter out distractor tracks (hard 4-player constraint)
                if p.track_id in self.distractor_tracks:
                    return False

                if p.track_id not in self.track_stats:
                    return True  # Unknown track, keep it
                stats = self.track_stats[p.track_id]

                # Filter out likely referees (sideline/stationary observers)
                if stats.is_likely_referee:
                    return False

                is_stationary = stats.position_spread < self.config.min_position_spread_for_primary
                has_ball_engagement = stats.ball_proximity_score >= self.config.min_ball_proximity_for_stationary
                # Keep if: not stationary OR has ball engagement
                return not is_stationary or has_ball_engagement

            filtered = [p for p in filtered if is_valid_track(p)]
            if len(filtered) < before_stationary:
                logger.debug(
                    f"Stationary/distractor filter: {before_stationary} -> {len(filtered)} "
                    f"(removed {before_stationary - len(filtered)} tracks)"
                )

        # Step 4: Player selection (two-team or top-K)
        if self._use_two_team and self.court_split_y is not None:
            # Two-team filtering: select top-K per court side (near/far)
            # This ensures players from both teams are selected
            # Use strict_primary=False - allow filling with other detections when
            # primary tracks are momentarily lost (better than showing nothing)
            # The play_area filter above already removed off-court detections
            filtered, self._track_team_history = select_two_teams(
                filtered,
                self.court_split_y,
                self.config.players_per_team,
                self.primary_tracks if self._tracks_analyzed else None,
                strict_primary=False,  # Allow filling when primary tracks lost
                hysteresis=self.config.two_team_hysteresis,
                track_team_history=self._track_team_history,
            )
        elif self._tracks_analyzed and self.primary_tracks:
            # Fall back to track priority + top-K
            # Allow filling with other detections when primary tracks lost
            filtered = select_with_track_priority(
                filtered, self.config.max_players, self.primary_tracks, strict_primary=False
            )
        else:
            # Fall back to simple top-K by size
            filtered = select_top_k_by_size(filtered, self.config.max_players)

        if len(filtered) < original_count:
            logger.debug(
                f"Total filter: {original_count} -> {len(filtered)} players"
            )

        return filtered

    @property
    def filter_method(self) -> str:
        """Return description of filtering method being used."""
        methods = ["bbox_size"]
        if self.court_position_stats:
            methods.append("court_presence")  # Hard filter by court presence
        if self._tracks_analyzed and self.track_stats:
            methods.append("stationary")  # Filters low-spread tracks (referees)
        if self.distractor_tracks:
            methods.append("distractor")  # Hard 4-player constraint
        if self._tracks_analyzed and self.primary_tracks:
            methods.append("track_stability")
        if self._use_two_team:
            methods.append("two_team")
        else:
            methods.append("top_k")
        if self.use_ball_filtering and self.play_area is not None:
            methods.append("play_area")
        return "+".join(methods)
