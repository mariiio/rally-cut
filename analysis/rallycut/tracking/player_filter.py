"""
Player filtering to detect only court players using multiple signals.

Filters out spectators, referees, and other non-playing persons using:
1. Bounding box size (court players are closer = larger boxes)
2. Track stability (court players appear consistently across frames)
3. Expected player count (4 for beach volleyball 2v2)
4. Ball trajectory play area (optional refinement)
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

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
        # Weighted combination of presence and size
        # Presence is most important (court players are always visible)
        # Ball proximity boosts score (players are near the ball)
        base_score = self.presence_rate * 0.5 + min(self.avg_bbox_area * 10, 1.0) * 0.3
        # Add ball proximity boost (up to 0.2)
        proximity_boost = self.ball_proximity_score * 0.2
        return base_score + proximity_boost


@dataclass
class PlayerFilterConfig:
    """Configuration for player filtering (beach volleyball 2v2)."""

    # Bbox size filtering - minimum size to be considered a court player
    # These thresholds filter out distant spectators while keeping court players
    min_bbox_area: float = 0.005  # Min 0.5% of frame area
    min_bbox_height: float = 0.10  # Min 10% of frame height

    # Top-K selection - keep only the 4 largest detections per frame
    # Beach volleyball is 2v2 = 4 players on court
    max_players: int = 4
    players_per_team: int = 2  # 2v2 beach volleyball

    # Two-team filtering - split court and select per team
    # This ensures players from both teams are selected
    # Camera is always behind baseline: teams split by Y (near/far, horizontal line)
    use_two_team_filter: bool = True

    # Track stability filtering - prefer tracks that appear consistently
    min_presence_rate: float = 0.3  # Track must appear in 30%+ of frames to be "primary"
    min_stability_score: float = 0.25  # Minimum combined stability score

    # Stability score weights (for compute_stability_score)
    presence_weight: float = 0.5  # Weight for presence rate in stability score
    bbox_area_weight: float = 0.3  # Weight for bbox area in stability score
    ball_proximity_weight: float = 0.2  # Weight for ball proximity in stability score
    bbox_area_scalar: float = 10.0  # Scalar to normalize bbox area (area * scalar, capped at 1.0)

    # Ball proximity boost - tracks near ball get stability boost
    ball_proximity_radius: float = 0.15  # 15% of frame size
    ball_proximity_boost: float = 0.2  # Boost to stability score

    # Position spread requirement for primary tracks (filters out stationary objects)
    # Players move around the court; stationary things (referees, net posts) don't
    # Spread is geometric mean of X/Y std dev (normalized 0-1)
    # Typical player spread: 0.02-0.10, stationary object spread: <0.015 (detection noise)
    min_position_spread_for_primary: float = 0.018  # Base movement threshold

    # Combined filter: tracks with BOTH low spread AND low ball proximity are filtered
    # This catches stationary objects while keeping players who don't move much but engage with ball
    min_ball_proximity_for_stationary: float = 0.03  # If spread < threshold, need 3%+ ball proximity

    # Ball trajectory thresholds (secondary filter)
    ball_confidence_threshold: float = 0.35
    min_ball_points: int = 10  # Need more points for reliable hull
    hull_padding: float = 0.15  # 15% padding around ball trajectory hull (stricter)
    min_ball_detection_rate: float = 0.30  # Higher threshold

    # Track ID stabilization - merge tracks when one ends and another starts nearby
    stabilize_track_ids: bool = True
    max_gap_frames: int = 90  # Max frames between track end and new track start (~3s at 30fps)
    max_merge_distance: float = 0.25  # Max position distance for merging (25% of frame)


def compute_stability_score(stats: TrackStats, config: PlayerFilterConfig) -> float:
    """Compute stability score for a track using configurable weights.

    Args:
        stats: Track statistics.
        config: Filter configuration with weight parameters.

    Returns:
        Stability score (0-1, higher = more stable).
    """
    # Weighted combination of presence and size
    base_score = (
        stats.presence_rate * config.presence_weight
        + min(stats.avg_bbox_area * config.bbox_area_scalar, 1.0) * config.bbox_area_weight
    )
    # Add ball proximity boost
    proximity_boost = stats.ball_proximity_score * config.ball_proximity_weight
    return base_score + proximity_boost


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

    # Fallback: use ball trajectory center
    confident = [
        p.y for p in ball_positions
        if p.confidence >= config.ball_confidence_threshold
    ]

    if len(confident) < config.min_ball_points:
        return None

    # Use ball trajectory Y-range center as court center (net position)
    y_center = (max(confident) + min(confident)) / 2

    logger.debug(f"Court split from ball trajectory: y={y_center:.3f}")

    return y_center


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
) -> list[PlayerPosition]:
    """
    Select players using two-team filtering (near/far split).

    Camera is behind baseline, so teams are split by Y:
    - Near team: y > split_y (closer to camera, larger bboxes)
    - Far team: y <= split_y (further from camera, smaller bboxes)

    Args:
        players: List of player detections for a frame.
        split_y: Y-coordinate that splits near/far teams.
        players_per_team: Number of players per team (2 for beach).
        primary_tracks: Optional set of stable track IDs to prioritize.
        strict_primary: If True and primary_tracks is set, ONLY select primary tracks.
            This filters out referees who pass other filters but aren't primary.

    Returns:
        Selected players (up to 2 * players_per_team).
    """
    # If we have primary tracks and strict mode, only consider primary track players
    if primary_tracks and strict_primary:
        players = [p for p in players if p.track_id in primary_tracks]
        if not players:
            return []

    if len(players) <= players_per_team * 2:
        return players

    # Split by Y (near/far)
    near_team = [p for p in players if p.y > split_y]
    far_team = [p for p in players if p.y <= split_y]

    def select_from_side(
        side_players: list[PlayerPosition],
        k: int,
    ) -> list[PlayerPosition]:
        """Select top K from one side by bbox size."""
        if len(side_players) <= k:
            return side_players
        # Sort by size
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
        f"Two-team selection (y={split_y:.2f}): "
        f"{len(players)} -> {len(result)} "
        f"(near: {len(selected_near)}, far: {len(selected_far)})"
    )

    return result


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

    # Sort tracks by first_frame
    tracks_by_start = sorted(
        track_info.items(),
        key=lambda x: x[1]["first_frame"],
    )

    for new_id, new_info in tracks_by_start:
        if new_id in id_mapping:
            continue  # Already merged

        new_first_frame = new_info["first_frame"]
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

            # Check position distance (use squared distance to avoid sqrt)
            dx = new_first_pos[0] - old_last_pos[0]
            dy = new_first_pos[1] - old_last_pos[1]
            distance_sq = dx * dx + dy * dy
            max_distance_sq = config.max_merge_distance * config.max_merge_distance

            if distance_sq <= max_distance_sq and distance_sq < best_distance:
                best_distance = distance_sq
                best_match = canonical_id

        if best_match is not None and best_match != new_id:
            id_mapping[new_id] = best_match
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


def identify_primary_tracks(
    track_stats: dict[int, TrackStats],
    config: PlayerFilterConfig,
) -> set[int]:
    """
    Identify primary tracks that are likely real court players.

    Primary tracks have high presence rate, stability score, AND position spread.
    The position spread requirement filters out referees who stand in one spot
    while players move around the court.

    Args:
        track_stats: Statistics for each track.
        config: Filter configuration.

    Returns:
        Set of track IDs that are primary (stable) tracks.
    """
    primary: set[int] = set()

    for track_id, stats in track_stats.items():
        # Compute stability score using configurable weights
        stability = compute_stability_score(stats, config)

        # Must meet presence and stability thresholds
        if stats.presence_rate < config.min_presence_rate:
            continue
        if stability < config.min_stability_score:
            continue

        # Filter stationary objects using combined spread + ball proximity check
        # Stationary objects (posts, referees) have low spread AND low ball proximity
        # Players who don't move much still engage with the ball
        is_stationary = stats.position_spread < config.min_position_spread_for_primary
        has_ball_engagement = stats.ball_proximity_score >= config.min_ball_proximity_for_stationary

        if is_stationary and not has_ball_engagement:
            logger.info(
                f"Track {track_id} excluded from primary: spread={stats.position_spread:.4f}, "
                f"ball_prox={stats.ball_proximity_score:.3f} (stationary + no ball engagement)"
            )
            continue

        primary.add(track_id)

    if primary:
        logger.info(
            f"Identified {len(primary)} primary tracks: {sorted(primary)} "
            f"(presence >= {config.min_presence_rate:.0%}, "
            f"stability >= {config.min_stability_score:.2f}, "
            f"spread >= {config.min_position_spread_for_primary:.3f})"
        )
        # Log stats for primary tracks
        for tid in sorted(primary):
            stats = track_stats[tid]
            logger.info(
                f"  Track {tid}: spread={stats.position_spread:.4f}, "
                f"presence={stats.presence_rate:.2f}, bbox={stats.avg_bbox_area:.4f}"
            )

    return primary


def select_with_track_priority(
    players: list[PlayerPosition],
    k: int,
    primary_tracks: set[int],
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

    Returns:
        Up to K players, preferring primary tracks.
    """
    if len(players) <= k:
        return players

    if not primary_tracks:
        # No primary tracks identified, fall back to size-based selection
        return select_top_k_by_size(players, k)

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
    ):
        """
        Initialize player filter.

        Args:
            ball_positions: Ball tracking results for play area filtering.
            total_frames: Total frames in video segment.
            config: Filter configuration.
        """
        self.config = config or PlayerFilterConfig()
        self.ball_positions = ball_positions or []
        self.total_frames = total_frames

        # Track stability (computed by analyze_tracks)
        self.track_stats: dict[int, TrackStats] = {}
        self.primary_tracks: set[int] = set()
        self._tracks_analyzed = False

        # Court split for two-team filtering (Y coordinate, horizontal line)
        # Camera is always behind baseline: teams split by near/far (Y axis)
        self.court_split_y: float | None = None
        self._use_two_team = False

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

        Also recomputes court split using player positions (more reliable than
        ball trajectory alone).

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

        # Identify primary tracks using position spread (filters out stationary referees)
        self.primary_tracks = identify_primary_tracks(self.track_stats, self.config)

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
        if ball_proximity:
            avg_proximity = sum(
                s.ball_proximity_score for s in self.track_stats.values()
            ) / max(1, len(self.track_stats))
            logger.info(
                f"Track analysis: {len(self.track_stats)} tracks, "
                f"{len(self.primary_tracks)} primary, "
                f"avg ball proximity: {avg_proximity:.2f}"
            )
        else:
            logger.info(
                f"Track analysis: {len(self.track_stats)} tracks, "
                f"{len(self.primary_tracks)} primary"
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
        filtered = filter_by_bbox_size(players, self.config)

        # Step 2: Filter by play area (removes spectators outside court)
        # This runs BEFORE two-team selection to ensure only court players are considered
        if self.use_ball_filtering and self.play_area is not None:
            filtered = filter_by_play_area(filtered, self.play_area)

        # Step 3: Filter out stationary tracks (posts, referees)
        # Stationary objects have BOTH low spread AND low ball proximity
        # Players who don't move much still engage with the ball
        if self._tracks_analyzed and self.track_stats:
            before_stationary = len(filtered)

            def is_valid_track(p: PlayerPosition) -> bool:
                if p.track_id not in self.track_stats:
                    return True  # Unknown track, keep it
                stats = self.track_stats[p.track_id]
                is_stationary = stats.position_spread < self.config.min_position_spread_for_primary
                has_ball_engagement = stats.ball_proximity_score >= self.config.min_ball_proximity_for_stationary
                # Keep if: not stationary OR has ball engagement
                return not is_stationary or has_ball_engagement

            filtered = [p for p in filtered if is_valid_track(p)]
            if len(filtered) < before_stationary:
                logger.debug(
                    f"Stationary filter: {before_stationary} -> {len(filtered)} "
                    f"(removed {before_stationary - len(filtered)} stationary tracks)"
                )

        # Step 4: Player selection (two-team or top-K)
        if self._use_two_team and self.court_split_y is not None:
            # Two-team filtering: select top-K per court side (near/far)
            # This ensures players from both teams are selected
            # Use strict_primary=False since we already filtered out referees above
            filtered = select_two_teams(
                filtered,
                self.court_split_y,
                self.config.players_per_team,
                self.primary_tracks if self._tracks_analyzed else None,
                strict_primary=False,  # Don't be strict, we already filtered referees
            )
        elif self._tracks_analyzed and self.primary_tracks:
            # Fall back to track priority + top-K
            filtered = select_with_track_priority(
                filtered, self.config.max_players, self.primary_tracks
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
        if self._tracks_analyzed and self.track_stats:
            methods.append("stationary")  # Filters low-spread tracks (referees)
        if self._tracks_analyzed and self.primary_tracks:
            methods.append("track_stability")
        if self._use_two_team:
            methods.append("two_team")
        else:
            methods.append("top_k")
        if self.use_ball_filtering and self.play_area is not None:
            methods.append("play_area")
        return "+".join(methods)
