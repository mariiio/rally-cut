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

import dataclasses
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

    # Stationary background track pre-filter
    # Removes tracks from raw positions that are clearly fixed objects (signs, equipment,
    # distant spectators, bags on the floor). These have near-zero position variance.
    # Key insight: real background objects have spread < 0.008 (truly zero motion),
    # while real players even in ready position have spread > 0.012 (body sway, reactions).
    # Threshold 0.010 sits in the gap. yolo11s produces tighter bboxes than yolov8n,
    # so players have lower spread than before — threshold must be conservative.
    # No presence requirement: an object that doesn't move across 50+ detections is
    # inanimate regardless of how intermittently YOLO detects it. A safety net
    # (see remove_stationary_background_tracks) prevents removal when < max_players
    # tracks would survive.
    # Runs before split/merge/link to prevent background tracks from interfering.
    enable_stationary_background_filter: bool = True
    stationary_bg_max_spread: float = 0.010  # Max position_spread (geometric mean of x/y std)
    stationary_bg_min_detections: int = 50  # Minimum detections to be considered

    # Gap interpolation for primary tracks
    # Fills short detection gaps with linearly interpolated positions.
    # Far-court players are missed ~20-25% of frames by YOLO; interpolation recovers
    # these gaps when the same track reappears within max_interpolation_gap frames.
    enable_interpolation: bool = True
    max_interpolation_gap: int = 30  # Max gap in frames to interpolate (~1s at 30fps)
    interpolated_confidence: float = 0.5  # Confidence assigned to interpolated positions

    def scaled_for_fps(self, fps: float) -> PlayerFilterConfig:
        """Return a copy with frame-count thresholds scaled for the video FPS.

        All frame-count parameters are tuned for 30fps. At 60fps the same
        real-world durations correspond to 2x more frames. This method
        scales those parameters so behaviour is consistent across frame
        rates.

        Non-frame-count settings (distances, weights, ratios) are unchanged.
        """
        if fps <= 0 or abs(fps - 30.0) < 1.0:
            return self  # No scaling needed

        ratio = fps / 30.0
        return dataclasses.replace(
            self,
            max_gap_frames=int(round(self.max_gap_frames * ratio)),
            max_interpolation_gap=int(round(self.max_interpolation_gap * ratio)),
            stationary_bg_min_detections=int(round(self.stationary_bg_min_detections * ratio)),
        )


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


def _zone_min_bbox_area(y: float, config_min: float) -> float:
    """Compute zone-dependent minimum bbox area.

    Far-court players (low Y) appear smaller due to perspective.
    Uses the same linear ramp as ``_filter_detections`` in
    ``player_tracker.py`` so that detections accepted pre-tracking
    are not discarded post-tracking.

    For far-court positions where the zone-dependent threshold is below
    ``config_min``, the zone value is used (relaxation). For near-court
    positions the zone threshold equals or exceeds ``config_min``, so
    ``config_min`` applies (no tightening beyond config).
    """
    zone_min = 0.0005 + 0.0025 * y
    # Relax for far-court only; never go above config_min
    return min(zone_min, config_min)


def filter_by_bbox_size(
    players: list[PlayerPosition],
    config: PlayerFilterConfig,
) -> list[PlayerPosition]:
    """
    Filter players by bounding box size.

    Uses zone-dependent minimum area: far-court players (low Y) have a
    lower threshold matching the pre-tracking filter in ``_filter_detections``.
    Near-court players still use ``config.min_bbox_area``.

    Args:
        players: List of player detections.
        config: Filter configuration.

    Returns:
        Players with bboxes above minimum size thresholds.
    """
    filtered = []
    for p in players:
        bbox_area = p.width * p.height
        min_area = _zone_min_bbox_area(p.y, config.min_bbox_area)
        if bbox_area >= min_area and p.height >= config.min_bbox_height:
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


def _get_court_split_y_from_calibration(
    calibrator: CourtCalibrator,
) -> float | None:
    """Get the net line Y position in normalized image coordinates.

    Projects the net center point (court midpoint) to image space.
    Falls back to midpoint of corner Y coordinates when homography
    projection fails (e.g., extreme off-screen near corners).

    Args:
        calibrator: CourtCalibrator instance (must be calibrated).

    Returns:
        Normalized Y coordinate of the net line (0-1), or None on failure.
    """
    from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

    net_court_x = COURT_WIDTH / 2  # Center of net
    net_court_y = COURT_LENGTH / 2  # Net is at midpoint (8m)

    try:
        # court_to_image returns normalized coords when img dims are (1, 1)
        _, net_y = calibrator.court_to_image(
            (net_court_x, net_court_y), 1, 1
        )
        if 0.1 < net_y < 0.9:
            return float(net_y)
        logger.warning(
            "Net Y projection out of reasonable range: %.3f", net_y
        )
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        logger.warning("Failed to project net to image: %s", e)

    # Fallback: compute net_y directly from corner coordinates.
    # Corner order: near-left, near-right, far-right, far-left.
    # This handles extreme off-screen corners where homography is degenerate.
    if calibrator.homography is not None and len(calibrator.homography.image_corners) == 4:
        corners = calibrator.homography.image_corners
        near_y = (corners[0][1] + corners[1][1]) / 2  # near baseline
        far_y = (corners[2][1] + corners[3][1]) / 2   # far baseline
        net_y_fallback = (near_y + far_y) / 2
        if 0.1 < net_y_fallback < 0.9:
            logger.info(
                "Court split from corner midpoint fallback: %.3f", net_y_fallback
            )
            return float(net_y_fallback)
        logger.warning(
            "Corner midpoint fallback also out of range: %.3f", net_y_fallback
        )

    return None


def compute_court_split(
    ball_positions: list[BallPosition],
    config: PlayerFilterConfig,
    player_positions: list[PlayerPosition] | None = None,
    court_calibrator: CourtCalibrator | None = None,
) -> tuple[float, str, dict[int, int]] | None:
    """
    Compute the Y-coordinate that splits the court into near/far teams.

    Camera is always behind baseline, so teams are split by a horizontal line.

    Priority order:
    1. Calibration-derived (precise geometry from court homography)
    2. Bbox clustering (player Y positions + bbox sizes)
    3. Ball trajectory direction changes
    4. Ball trajectory center (fallback)

    Args:
        ball_positions: Ball positions from tracking.
        config: Filter configuration.
        player_positions: All player positions (preferred for split calculation).
        court_calibrator: Optional calibrated court calibrator for precise net_y.

    Returns:
        Tuple of (split_y, confidence, team_assignments) where confidence is
        "high" or "low", and team_assignments maps track_id -> team (0=near,
        1=far) from bbox size ranking. Empty dict when bbox clustering is not
        available. Returns None if insufficient data.
    """
    # Priority 1: Calibration-derived (precise geometry)
    if court_calibrator is not None and court_calibrator.is_calibrated:
        cal_split = _get_court_split_y_from_calibration(court_calibrator)
        if cal_split is not None:
            logger.debug(f"Court split from calibration: y={cal_split:.3f}")
            # Still run bbox clustering to get team assignments from size
            team_assignments: dict[int, int] = {}
            if player_positions and len(player_positions) >= 20:
                bbox_result = _find_net_from_bbox_clustering(
                    player_positions, config.players_per_team,
                    ball_positions=ball_positions or None,
                )
                if bbox_result is not None:
                    team_assignments = bbox_result[2]
            return (cal_split, "high", team_assignments)

    # Priority 2: Bbox clustering (player Y positions + bbox sizes)
    if player_positions and len(player_positions) >= 20:
        result = _find_net_from_bbox_clustering(
            player_positions, config.players_per_team,
            ball_positions=ball_positions or None,
        )
        if result is not None:
            split_y, confidence, team_assignments = result
            logger.debug(
                f"Court split from bbox clustering: y={split_y:.3f} "
                f"(confidence={confidence})"
            )
            return (split_y, confidence, team_assignments)

    # Priority 3: Ball trajectory direction changes
    if ball_positions:
        crossing_y = _find_net_from_ball_crossings(ball_positions)
        if crossing_y is not None:
            logger.debug(f"Court split from ball crossings: y={crossing_y:.3f}")
            return (crossing_y, "low", {})

    # Priority 4: Ball trajectory center (fallback)
    confident = [
        p.y for p in ball_positions
        if p.confidence >= config.ball_confidence_threshold
    ]

    if len(confident) < config.min_ball_points:
        return None

    y_center = (max(confident) + min(confident)) / 2

    logger.debug(f"Court split from ball trajectory center: y={y_center:.3f}")

    return (y_center, "low", {})


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


def classify_teams(
    positions: list[PlayerPosition],
    court_split_y: float,
    window_frames: int = 60,
    precomputed_assignments: dict[int, int] | None = None,
) -> dict[int, int]:
    """Classify each track into team 0 (near) or team 1 (far).

    When precomputed_assignments is provided (from bbox size ranking),
    those assignments are used directly for known tracks, ensuring a
    correct 2+2 split. Remaining tracks (from track splitting or new
    fragments) fall back to median-Y classification.

    Without precomputed assignments, uses median Y position of the first
    window_frames per track. At rally start, teams are physically separated
    so early frames provide clean team anchoring.

    Args:
        positions: All player positions across frames.
        court_split_y: Y-coordinate that splits near/far court (0-1).
        window_frames: Number of initial frames to use per track.
        precomputed_assignments: Optional pre-computed team assignments
            from bbox size ranking (track_id -> team). Tracks in this
            dict use the precomputed team; remaining tracks use Y fallback.

    Returns:
        Dict mapping track_id -> team (0=near, 1=far).
    """
    # Group positions by track_id, sorted by frame
    track_positions: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id < 0:
            continue
        if p.track_id not in track_positions:
            track_positions[p.track_id] = []
        track_positions[p.track_id].append(p)

    team_assignments: dict[int, int] = {}
    precomputed_used = 0

    for track_id, track_pos in track_positions.items():
        # Use precomputed assignment if available for this track
        if precomputed_assignments and track_id in precomputed_assignments:
            team_assignments[track_id] = precomputed_assignments[track_id]
            precomputed_used += 1
            continue

        # Fallback: classify by median Y position
        track_pos.sort(key=lambda p: p.frame_number)
        # Use first window_frames positions
        early = track_pos[:window_frames]
        median_y = float(np.median([p.y for p in early]))
        # Near team (closer to camera) has higher Y values
        team_assignments[track_id] = 0 if median_y > court_split_y else 1

    near_count = sum(1 for t in team_assignments.values() if t == 0)
    far_count = sum(1 for t in team_assignments.values() if t == 1)

    # Fallback: if all players ended up on the same team (court_split_y
    # is too extreme, e.g. above/below all players), split by median Y
    # index.  This guarantees a valid 2-team split.
    if len(team_assignments) >= 4 and (near_count == 0 or far_count == 0):
        sorted_tids = sorted(
            team_assignments.keys(),
            key=lambda t: np.median([p.y for p in track_positions[t][:window_frames]]),
        )
        mid = len(sorted_tids) // 2
        for t in sorted_tids[:mid]:
            team_assignments[t] = 1  # far (lower Y)
        for t in sorted_tids[mid:]:
            team_assignments[t] = 0  # near (higher Y)
        near_count = len(sorted_tids) - mid
        far_count = mid
        logger.info(
            "All tracks on one side with split_y=%.3f, "
            "using median-index fallback: near=%d, far=%d",
            court_split_y, near_count, far_count,
        )

    logger.debug(
        f"Team classification (split_y={court_split_y:.3f}): "
        f"near={near_count}, far={far_count}"
        f"{f', {precomputed_used} from bbox size' if precomputed_used else ''}"
    )

    return team_assignments


def _validate_split_with_ball(
    split_y: float,
    ball_positions: list[BallPosition],
    tolerance: float = 0.10,
) -> bool:
    """Check if ball direction reversals cluster near split_y.

    Cross-validates bbox-derived split_y against ball trajectory. If ball
    Y-direction reversals (which indicate net crossings) cluster far from
    split_y, the bbox grouping might be wrong (e.g., a spectator's bbox
    ranked above a real player).

    Args:
        split_y: Bbox-derived court split Y coordinate.
        ball_positions: Ball positions from tracking.
        tolerance: Maximum allowed disagreement between split_y and ball
            crossing Y (default 0.10 = 10% of screen).

    Returns:
        True if ball data agrees or is insufficient, False if disagreement.
    """
    crossing_y = _find_net_from_ball_crossings(ball_positions)
    if crossing_y is None:
        return True  # No data → assume valid
    return abs(crossing_y - split_y) < tolerance


def _find_net_from_bbox_clustering(
    player_positions: list[PlayerPosition],
    players_per_team: int = 2,
    ball_positions: list[BallPosition] | None = None,
) -> tuple[float, str, dict[int, int]] | None:
    """
    Find the net Y position using bbox size to identify near/far teams.

    Near team players have LARGER bboxes (closer to camera).
    Far team players have SMALLER bboxes (further from camera).

    The net is the boundary between the Y ranges of these two groups.

    Also returns direct team assignments from bbox size ranking, avoiding
    the lossy intermediate step of re-classifying via a Y threshold.

    Args:
        player_positions: All player positions across frames.
        players_per_team: Expected players per team (2 for beach volleyball).
        ball_positions: Optional ball positions for cross-validation.

    Returns:
        Tuple of (split_y, confidence, team_assignments) or None.
        Confidence is "high" when teams are cleanly separated, "low" when
        falling back to median. team_assignments maps track_id -> team
        (0=near, 1=far) for the top tracks identified by bbox size.
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

    # Build team assignments directly from bbox size ranking
    # This is the key fix: assignments come from size, not from Y threshold
    team_assignments: dict[int, int] = {}
    for t in near_tracks:
        team_assignments[t] = 0  # Near team
    for t in far_tracks:
        team_assignments[t] = 1  # Far team

    # Get Y positions for each team
    near_ys = [track_avg_y[t] for t in near_tracks]
    far_ys = [track_avg_y[t] for t in far_tracks]

    # Near team should have HIGHER Y (closer to camera = bottom of frame)
    # Far team should have LOWER Y (further from camera = top of frame)
    max_far_y = max(far_ys)
    min_near_y = min(near_ys)

    # Log the team positions for debugging
    logger.debug(
        f"Bbox clustering: near={near_tracks} ys={[f'{y:.2f}' for y in near_ys]}, "
        f"far={far_tracks} ys={[f'{y:.2f}' for y in far_ys]}"
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
        return (split_y, "low", team_assignments)

    # Split at the midpoint between teams
    split_y = (max_far_y + min_near_y) / 2

    # Cross-validate with ball trajectory if available
    confidence = "high"
    if ball_positions and not _validate_split_with_ball(split_y, ball_positions):
        confidence = "low"
        logger.debug(
            f"Ball crossings disagree with bbox split y={split_y:.3f}, "
            f"downgrading confidence to low"
        )

    logger.debug(
        f"Net found between teams: y={split_y:.3f} "
        f"(far_max={max_far_y:.2f}, near_min={min_near_y:.2f}, "
        f"confidence={confidence})"
    )

    return (split_y, confidence, team_assignments)


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


def _position_spread(x_std: float, y_std: float) -> float:
    """Geometric mean of X/Y standard deviations (handles one-axis-only movement)."""
    if x_std > 0 and y_std > 0:
        return float(np.sqrt(x_std * y_std))
    return max(x_std, y_std)


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
        position_spread = _position_spread(x_std, y_std)

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


def remove_stationary_background_tracks(
    positions: list[PlayerPosition],
    config: PlayerFilterConfig,
    total_frames: int | None = None,
) -> tuple[list[PlayerPosition], set[int]]:
    """Remove tracks that are clearly stationary background objects.

    Background objects (signs, equipment, distant spectators, bags on the floor)
    have very low position variance. Real players always move more than the
    threshold even when standing still due to body sway, game reactions, and
    court movement. An object that doesn't move across 50+ detections is
    inanimate regardless of how intermittently YOLO detects it.

    Runs before split/merge/link to prevent background tracks from interfering
    with post-processing (tracklet linking, court identity, etc.).

    Args:
        positions: All raw player positions from tracker.
        config: Filter configuration with stationary background thresholds.
        total_frames: Total frames in the rally. If None, estimated from positions.

    Returns:
        Tuple of (filtered positions, set of removed track IDs).
    """
    if not positions or not config.enable_stationary_background_filter:
        return positions, set()

    # Estimate total frames if not provided
    if total_frames is None:
        frame_nums = {p.frame_number for p in positions}
        total_frames = max(frame_nums) - min(frame_nums) + 1 if frame_nums else 1

    # Group by track
    tracks: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id not in tracks:
            tracks[p.track_id] = []
        tracks[p.track_id].append(p)

    removed_ids: set[int] = set()

    for track_id, track_positions in tracks.items():
        n = len(track_positions)
        if n < config.stationary_bg_min_detections:
            continue

        # Compute position spread
        xs = np.array([p.x for p in track_positions])
        ys = np.array([p.y for p in track_positions])
        x_std = float(np.std(xs))
        y_std = float(np.std(ys))
        spread = _position_spread(x_std, y_std)

        if spread < config.stationary_bg_max_spread:
            removed_ids.add(track_id)
            presence = n / total_frames
            logger.info(
                f"Stationary background filter: removing track {track_id} "
                f"({n} det, presence={presence:.0%}, spread={spread:.4f}, "
                f"pos=({float(np.mean(xs)):.3f}, {float(np.mean(ys)):.3f}))"
            )

    if removed_ids:
        # Safety: never remove tracks if it would leave fewer than expected players.
        # Background objects don't cause harm when co-existing with real players
        # (primary track selection already deprioritizes stationary tracks), but
        # removing real players in short rallies (100% presence, low spread) is
        # catastrophic for tracking accuracy.
        surviving_ids = set(tracks.keys()) - removed_ids
        if len(surviving_ids) < config.max_players:
            logger.warning(
                f"Stationary background filter: would leave {len(surviving_ids)} "
                f"tracks (< {config.max_players} expected) — skipping removal"
            )
            return positions, set()

        original_count = len(positions)
        positions = [p for p in positions if p.track_id not in removed_ids]
        logger.info(
            f"Stationary background filter: removed {len(removed_ids)} tracks "
            f"({sorted(removed_ids)}), {original_count} -> {len(positions)} positions"
        )

    return positions, removed_ids


def stabilize_track_ids(
    positions: list[PlayerPosition],
    config: PlayerFilterConfig,
    team_assignments: dict[int, int] | None = None,
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
        team_assignments: Optional team classification (track_id -> team).
            Blocks cross-team merges (a fragment should only merge into
            a track from the same team).

    Returns:
        Tuple of (modified positions, id_mapping dict).
        The id_mapping maps merged_id -> canonical_id for merged tracks.
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

            # Block cross-team merges (structurally impossible in volleyball)
            if team_assignments:
                new_team = team_assignments.get(new_id)
                canon_team = team_assignments.get(canonical_id)
                if new_team is not None and canon_team is not None:
                    if new_team != canon_team:
                        continue

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
        # - OR low Y variance AND stationary (low spread) AND positioned near sidelines
        #   The spread check distinguishes true background objects (spread < 0.010)
        #   from real players who have spread > 0.01 even in ready position.
        is_likely_referee = (
            (is_near_sideline and (is_parallel_mover or has_low_ball_proximity or has_low_y_variance))
            or (is_outside_ball_range and has_low_ball_proximity and has_low_y_variance)
            or (has_low_y_variance
                and stats.position_spread < config.stationary_bg_max_spread
                and (stats.avg_x <= config.referee_sideline_threshold
                     or stats.avg_x >= (1.0 - config.referee_sideline_threshold)))
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
    1. Not on sidelines (x within primary_sideline_threshold, default 0.05-0.95)
    2. Minimum presence rate (default 20%)
    3. Not identified as referee

    Soft filters (used for ranking, relaxed when needed):
    - Court presence: tracks with low court presence get 0.5x stability penalty
      but are not excluded, so they can still fill slots when needed
    - Stationary check: tracks with low spread AND no ball engagement are
      deprioritized but included as fallbacks when fewer than max_players active

    Safety net: if fewer than max_players pass hard filters, re-admit the
    best hard-rejected tracks (referee/low_presence). Sideline rejects are
    never re-admitted.

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
    hard_rejected: list[tuple[int, float, str]] = []  # (track_id, stability, reason)
    court_cfg = court_config or CourtFilterConfig()
    referees = referee_tracks or set()

    for track_id, stats in track_stats.items():
        # HARD FILTER 0: Exclude tracks on sidelines (image coordinates)
        # Players are in the middle of the frame; referees stand on sides
        # Beach volleyball: players typically at x=0.25-0.75
        # Use configurable threshold for cameras angled toward court edges
        # Sideline rejects are NEVER re-admitted (truly off-frame)
        sideline_min = config.primary_sideline_threshold
        sideline_max = 1.0 - config.primary_sideline_threshold
        if stats.avg_x < sideline_min or stats.avg_x > sideline_max:
            logger.info(
                f"Track {track_id} excluded: avg_x={stats.avg_x:.2f} (sideline position, "
                f"threshold={sideline_min:.2f}-{sideline_max:.2f})"
            )
            continue

        # Compute stability score using configurable weights
        stability = compute_stability_score(stats, config)

        # SOFT PENALTY: Court presence (if calibration available)
        # Low court presence deprioritizes the track but doesn't exclude it,
        # so it can still fill the 4th slot when no better candidate exists.
        if stats.has_court_stats:
            if stats.court_presence_ratio < court_cfg.min_court_presence_ratio:
                logger.info(
                    f"Track {track_id} deprioritized: court_presence={stats.court_presence_ratio:.2f} "
                    f"< {court_cfg.min_court_presence_ratio:.2f} (low court presence)"
                )
                stability *= 0.5

        # Must meet presence threshold (hard filter)
        if stats.presence_rate < config.min_presence_rate:
            hard_rejected.append((track_id, stability, "low_presence"))
            continue

        # HARD FILTER 2: Referee detection
        if track_id in referees or stats.is_likely_referee:
            logger.debug(f"Track {track_id} excluded: identified as referee")
            hard_rejected.append((track_id, stability, "referee"))
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

    # Score by player behavior (ball proximity, movement, presence, coverage)
    # for ranking when >max_players candidates pass the stability threshold.
    # Coverage = temporal span / total frames. Prevents short-lived tracks
    # (e.g., a near-court player detected for only the first third of a rally)
    # from beating long-lived tracks that cover the full rally.
    def _player_behavior_score(tid: int) -> float:
        s = track_stats[tid]
        spread_normalized = min(s.position_spread / 0.05, 1.0)
        temporal_coverage = min(
            (s.last_frame - s.first_frame) / s.total_frames
            if s.total_frames > 0
            else 0.0,
            1.0,
        )
        return (
            0.4 * s.ball_proximity_score
            + 0.2 * spread_normalized
            + 0.2 * s.presence_rate
            + 0.2 * temporal_coverage
        )

    # Apply max_players limit if we have too many
    if len(selected) > config.max_players:
        scored = [(tid, _player_behavior_score(tid)) for tid, _ in selected]
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
            coverage = (
                (s.last_frame - s.first_frame) / s.total_frames
                if s.total_frames > 0
                else 0.0
            )
            logger.info(
                f"  Track {tid}: score={score:.3f} (ball={s.ball_proximity_score:.2f}, "
                f"spread={s.position_spread:.4f}, presence={s.presence_rate:.2f}, "
                f"coverage={coverage:.2f}) [{status}]"
            )
        selected = [(tid, stab) for tid, stab in selected if tid in kept_ids]

    # Safety net: re-admit hard-rejected tracks when we still have <max_players
    # This handles edge cases where filters are too aggressive for unusual videos.
    # Only referee/low_presence rejects are eligible; sideline rejects are never re-admitted.
    if len(selected) < config.max_players and hard_rejected:
        needed = config.max_players - len(selected)
        hard_rejected.sort(
            key=lambda entry: _player_behavior_score(entry[0]), reverse=True
        )
        for tid, stability, reason in hard_rejected[:needed]:
            logger.warning(
                f"Track {tid} RE-ADMITTED: rejected for '{reason}', "
                f"only {len(selected)} passed hard filters"
            )
            selected.append((tid, stability))

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
        ball_positions: List of ball detections from ball tracker.
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
                    split_result = compute_court_split(
                        self.ball_positions, self.config,
                        court_calibrator=self.court_calibrator,
                    )
                    self.court_split_y = split_result[0] if split_result else None
                    if self.court_split_y is not None:
                        self._use_two_team = True
                        logger.info(
                            f"Two-team filter enabled: y={self.court_split_y:.3f}"
                        )

    def analyze_tracks(self, all_positions: list[PlayerPosition]) -> None:
        """
        Analyze all positions to identify the 4 active player tracks.

        Must be called before filter() for track stability to work.
        If not called, filter() falls back to size-based selection.

        Performs the following steps:
        1. Compute ball proximity scores (if ball data available)
        2. Compute per-track statistics (bbox size, coverage, movement)
        3. Compute court position stats (if calibrator available)
        4. Detect referee tracks (excluded from primary selection)
        5. Identify primary tracks via hard/soft filters with safety net
        6. Detect distractor tracks (non-players coexisting with primaries)
        7. Recompute court split from player position density

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
            split_result = compute_court_split(
                self.ball_positions, self.config,
                player_positions=all_positions,
                court_calibrator=self.court_calibrator,
            )
            new_split_y = split_result[0] if split_result else None
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


def interpolate_player_gaps(
    positions: list[PlayerPosition],
    primary_track_ids: set[int] | list[int],
    config: PlayerFilterConfig | None = None,
) -> tuple[list[PlayerPosition], int]:
    """Interpolate detection gaps for primary player tracks.

    When YOLO misses a player for a few frames (common for far-court players),
    linearly interpolates position and bbox between the last known and next known
    detection. Only interpolates for identified primary tracks.

    Args:
        positions: Filtered player positions (may contain non-primary tracks).
        primary_track_ids: Set of track IDs identified as primary players.
        config: Filter config (uses defaults if None).

    Returns:
        Tuple of (positions with interpolated entries added, count of interpolated).
    """
    cfg = config or PlayerFilterConfig()
    if not cfg.enable_interpolation or not positions or not primary_track_ids:
        return positions, 0

    max_gap = cfg.max_interpolation_gap
    interp_conf = cfg.interpolated_confidence

    # Build per-track frame map for primary tracks only
    primary_set = set(primary_track_ids)
    track_positions: dict[int, dict[int, PlayerPosition]] = {}
    for p in positions:
        if p.track_id in primary_set:
            track_positions.setdefault(p.track_id, {})[p.frame_number] = p

    interpolated: list[PlayerPosition] = []

    for track_id, frame_map in track_positions.items():
        frames = sorted(frame_map.keys())
        if len(frames) < 2:
            continue

        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            gap = f2 - f1

            if gap <= 1 or gap > max_gap:
                continue

            p1, p2 = frame_map[f1], frame_map[f2]

            for f in range(f1 + 1, f2):
                t = (f - f1) / gap
                interpolated.append(PlayerPosition(
                    frame_number=f,
                    track_id=track_id,
                    x=p1.x + t * (p2.x - p1.x),
                    y=p1.y + t * (p2.y - p1.y),
                    width=p1.width + t * (p2.width - p1.width),
                    height=p1.height + t * (p2.height - p1.height),
                    confidence=interp_conf,
                ))

    if interpolated:
        logger.info(
            f"Interpolated {len(interpolated)} positions across "
            f"{len(track_positions)} primary tracks "
            f"(max_gap={max_gap} frames)"
        )
        result = positions + interpolated
        result.sort(key=lambda p: (p.frame_number, p.track_id))
        return result, len(interpolated)

    return positions, 0


def recover_missing_players(
    pipeline_positions: list[PlayerPosition],
    raw_positions: list[PlayerPosition],
    primary_track_ids: set[int] | list[int],
    total_frames: int,
    ball_positions: list[BallPosition] | None = None,
    config: PlayerFilterConfig | None = None,
) -> tuple[list[PlayerPosition], set[int], int]:
    """Recover players lost during intermediate pipeline steps.

    When the pipeline produces <max_players primary tracks but the raw YOLO
    detections show >=max_players concurrent people, a valid player was lost
    during track splitting/merging/stabilization. This function identifies
    the best raw track to recover.

    Recovery criteria for a raw track:
    - Not already represented in pipeline output (spatially distinct)
    - Present in >=20% of frames (min_presence_rate)
    - Reasonable bbox size (>= min_bbox_area)
    - Not on sidelines

    Args:
        pipeline_positions: Positions after pipeline filtering (primary tracks only).
        raw_positions: Original YOLO+BoT-SORT detections (pre-pipeline).
        primary_track_ids: Currently selected primary track IDs.
        total_frames: Total frames in the rally.
        ball_positions: Ball positions for proximity scoring.
        config: Filter configuration.

    Returns:
        Tuple of (updated positions, updated primary_ids, count of recovered tracks).
    """
    cfg = config or PlayerFilterConfig()
    primary_set = set(primary_track_ids)

    if len(primary_set) >= cfg.max_players or not raw_positions:
        return pipeline_positions, primary_set, 0

    needed = cfg.max_players - len(primary_set)

    # Build per-frame position map for existing primary tracks
    primary_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in pipeline_positions:
        if p.track_id in primary_set:
            primary_by_frame.setdefault(p.frame_number, []).append(p)

    # Get raw track stats
    raw_stats = compute_track_stats(raw_positions, total_frames)

    if ball_positions:
        prox = compute_ball_proximity_scores(raw_positions, ball_positions, cfg)
        for tid, score in prox.items():
            if tid in raw_stats:
                raw_stats[tid].ball_proximity_score = score

    # Build per-track positions from raw
    raw_track_positions: dict[int, dict[int, PlayerPosition]] = {}
    for p in raw_positions:
        raw_track_positions.setdefault(p.track_id, {})[p.frame_number] = p

    # Find candidate tracks: not already in primary set, pass basic filters
    candidates: list[tuple[int, float]] = []  # (track_id, score)

    for tid, stats in raw_stats.items():
        # Skip tracks already in primary set
        if tid in primary_set:
            continue

        # Hard filters (same as identify_primary_tracks)
        sideline_min = cfg.primary_sideline_threshold
        if stats.avg_x < sideline_min or stats.avg_x > (1.0 - sideline_min):
            continue
        if stats.presence_rate < cfg.min_presence_rate:
            continue
        if stats.avg_bbox_area < _zone_min_bbox_area(stats.avg_y, cfg.min_bbox_area):
            continue

        # Check spatial distinctness: must not overlap with existing primaries
        # For each frame, compute distance to the nearest primary track.
        # If the median nearest-distance is <0.03, this is the same person.
        frame_map = raw_track_positions.get(tid, {})
        nearest_distances: list[float] = []
        for frame_num, raw_pos in frame_map.items():
            if frame_num in primary_by_frame:
                frame_min = min(
                    (
                        (raw_pos.x - pp.x) ** 2
                        + (raw_pos.y - pp.y) ** 2
                    ) ** 0.5
                    for pp in primary_by_frame[frame_num]
                )
                nearest_distances.append(frame_min)

        if nearest_distances:
            median_nearest = float(np.median(nearest_distances))
            if median_nearest < 0.03:
                # Too close to an existing primary — likely same player
                continue
        else:
            # No overlapping frames — can't verify distinctness
            continue

        # Score: same behavior score used by identify_primary_tracks
        spread_normalized = min(stats.position_spread / 0.05, 1.0)
        temporal_coverage = min(
            (stats.last_frame - stats.first_frame) / stats.total_frames
            if stats.total_frames > 0
            else 0.0,
            1.0,
        )
        score = (
            0.4 * stats.ball_proximity_score
            + 0.2 * spread_normalized
            + 0.2 * stats.presence_rate
            + 0.2 * temporal_coverage
        )
        candidates.append((tid, score))

    if not candidates:
        return pipeline_positions, primary_set, 0

    # Select best candidates
    candidates.sort(key=lambda x: x[1], reverse=True)
    recovered_positions: list[PlayerPosition] = []
    recovered_count = 0

    for tid, score in candidates[:needed]:
        frame_map = raw_track_positions[tid]
        for p in frame_map.values():
            recovered_positions.append(p)
        primary_set.add(tid)
        recovered_count += 1
        logger.info(
            f"Recovered track {tid} from raw positions: "
            f"presence={raw_stats[tid].presence_rate:.2f}, "
            f"area={raw_stats[tid].avg_bbox_area:.4f}, "
            f"score={score:.3f}"
        )

    if recovered_positions:
        result = pipeline_positions + recovered_positions
        result.sort(key=lambda p: (p.frame_number, p.track_id))
        return result, primary_set, recovered_count

    return pipeline_positions, primary_set, 0
