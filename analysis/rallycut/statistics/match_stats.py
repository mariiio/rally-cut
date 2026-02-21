"""Per-player and per-match statistics for beach volleyball.

Aggregates tracking and action classification data into meaningful
match statistics: per-player action counts, movement distance,
court zones, and overall match metrics.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.action_classifier import RallyActions
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class PlayerStats:
    """Statistics for a single player across a match or set."""

    track_id: int
    # Action counts
    serves: int = 0
    receives: int = 0
    sets: int = 0
    attacks: int = 0
    blocks: int = 0
    digs: int = 0
    # Movement
    total_distance_px: float = 0.0  # Total movement in pixels
    total_distance_m: float = 0.0  # Total movement in meters (if calibrated)
    avg_speed_px_per_frame: float = 0.0
    # Court zones (normalized 0-1, percentage of time in each zone)
    time_near_court_pct: float = 0.0  # % of time on near side
    time_far_court_pct: float = 0.0  # % of time on far side
    # Position heatmap (10x10 grid, normalized 0-1)
    position_heatmap: list[list[float]] = field(default_factory=list)
    # Frames active
    num_frames: int = 0
    court_side: str = "unknown"  # Primary court side ("near" or "far")

    @property
    def total_actions(self) -> int:
        return self.serves + self.receives + self.sets + self.attacks + self.blocks + self.digs

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "trackId": self.track_id,
            "serves": self.serves,
            "receives": self.receives,
            "sets": self.sets,
            "attacks": self.attacks,
            "blocks": self.blocks,
            "digs": self.digs,
            "totalActions": self.total_actions,
            "totalDistancePx": round(self.total_distance_px, 1),
            "avgSpeedPxPerFrame": round(self.avg_speed_px_per_frame, 4),
            "numFrames": self.num_frames,
            "courtSide": self.court_side,
        }
        if self.total_distance_m > 0:
            result["totalDistanceM"] = round(self.total_distance_m, 1)
        if self.position_heatmap:
            result["positionHeatmap"] = self.position_heatmap
        return result


@dataclass
class RallyStats:
    """Statistics for a single rally."""

    rally_id: str
    duration_frames: int
    duration_seconds: float
    num_contacts: int
    action_sequence: list[str]
    has_block: bool
    has_extended_exchange: bool  # 3+ contacts on a side
    max_rally_velocity: float  # Peak ball velocity during rally
    serving_side: str  # "near" or "far"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "durationFrames": self.duration_frames,
            "durationSeconds": round(self.duration_seconds, 2),
            "numContacts": self.num_contacts,
            "actionSequence": self.action_sequence,
            "hasBlock": self.has_block,
            "hasExtendedExchange": self.has_extended_exchange,
            "maxRallyVelocity": round(self.max_rally_velocity, 4),
            "servingSide": self.serving_side,
        }


@dataclass
class MatchStats:
    """Aggregated statistics for a full match."""

    # Per-player stats
    player_stats: list[PlayerStats] = field(default_factory=list)
    # Per-rally stats
    rally_stats: list[RallyStats] = field(default_factory=list)
    # Match-level aggregates
    total_rallies: int = 0
    total_contacts: int = 0
    avg_rally_duration_s: float = 0.0
    longest_rally_duration_s: float = 0.0
    avg_contacts_per_rally: float = 0.0
    side_out_rate: float = 0.0  # % of rallies won by receiving team
    # Video metadata
    video_fps: float = 30.0
    video_width: int = 1920
    video_height: int = 1080

    def to_dict(self) -> dict[str, Any]:
        return {
            "totalRallies": self.total_rallies,
            "totalContacts": self.total_contacts,
            "avgRallyDurationS": round(self.avg_rally_duration_s, 2),
            "longestRallyDurationS": round(self.longest_rally_duration_s, 2),
            "avgContactsPerRally": round(self.avg_contacts_per_rally, 1),
            "sideOutRate": round(self.side_out_rate, 3),
            "playerStats": [p.to_dict() for p in self.player_stats],
            "rallyStats": [r.to_dict() for r in self.rally_stats],
        }


def compute_player_movement(
    positions: list[PlayerPosition],
    track_id: int,
    video_width: int = 1920,
    video_height: int = 1080,
    calibrator: CourtCalibrator | None = None,
) -> tuple[float, float, float]:
    """Compute total movement distance for a player track.

    Args:
        positions: All player positions.
        track_id: Track ID to compute movement for.
        video_width: Video width for pixel conversion.
        video_height: Video height for pixel conversion.
        calibrator: Optional court calibrator for real-world distance.

    Returns:
        (distance_px, distance_m, avg_speed_px_per_frame)
    """
    track_pos = sorted(
        [p for p in positions if p.track_id == track_id],
        key=lambda p: p.frame_number,
    )

    if len(track_pos) < 2:
        return 0.0, 0.0, 0.0

    total_px = 0.0
    total_m = 0.0
    num_moves = 0

    for i in range(1, len(track_pos)):
        prev = track_pos[i - 1]
        curr = track_pos[i]

        frame_gap = curr.frame_number - prev.frame_number
        if frame_gap <= 0 or frame_gap > 5:
            continue

        dx_px = (curr.x - prev.x) * video_width
        dy_px = (curr.y - prev.y) * video_height
        dist_px = math.sqrt(dx_px * dx_px + dy_px * dy_px)
        total_px += dist_px
        num_moves += 1

        if calibrator is not None and calibrator.is_calibrated:
            try:
                court_prev = calibrator.image_to_court(
                    (prev.x, prev.y + prev.height / 2), video_width, video_height
                )
                court_curr = calibrator.image_to_court(
                    (curr.x, curr.y + curr.height / 2), video_width, video_height
                )
                dist_m = math.sqrt(
                    (court_curr[0] - court_prev[0]) ** 2
                    + (court_curr[1] - court_prev[1]) ** 2
                )
                total_m += dist_m
            except (RuntimeError, ValueError):
                pass

    avg_speed = total_px / num_moves if num_moves > 0 else 0.0

    return total_px, total_m, avg_speed


def compute_position_heatmap(
    positions: list[PlayerPosition],
    track_id: int,
    grid_size: int = 10,
) -> list[list[float]]:
    """Compute position heatmap for a player on a grid.

    Args:
        positions: All player positions.
        track_id: Track ID.
        grid_size: Grid resolution (default 10x10).

    Returns:
        2D list of normalized occupancy values (0-1).
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.float64)

    track_pos = [p for p in positions if p.track_id == track_id]
    if not track_pos:
        return grid.tolist()

    for p in track_pos:
        gx = min(int(p.x * grid_size), grid_size - 1)
        gy = min(int(p.y * grid_size), grid_size - 1)
        grid[gy, gx] += 1

    # Normalize
    total = grid.sum()
    if total > 0:
        grid /= total

    return grid.tolist()


def compute_match_stats(
    rally_actions_list: list[RallyActions],
    player_positions: list[PlayerPosition],
    video_fps: float = 30.0,
    video_width: int = 1920,
    video_height: int = 1080,
    calibrator: CourtCalibrator | None = None,
) -> MatchStats:
    """Compute comprehensive match statistics.

    Args:
        rally_actions_list: Classified actions for each rally.
        player_positions: All player tracking positions across the match.
        video_fps: Video frame rate.
        video_width: Video width.
        video_height: Video height.
        calibrator: Optional court calibrator for real-world distances.

    Returns:
        MatchStats with per-player and per-rally statistics.
    """
    from rallycut.tracking.action_classifier import ActionType

    stats = MatchStats(
        video_fps=video_fps,
        video_width=video_width,
        video_height=video_height,
    )

    # Collect all unique track IDs from actions
    all_track_ids: set[int] = set()
    for ra in rally_actions_list:
        for action in ra.actions:
            if action.player_track_id >= 0:
                all_track_ids.add(action.player_track_id)

    # Also add track IDs from player positions (may have players without actions)
    primary_tracks = set()
    track_frame_counts: dict[int, int] = {}
    for p in player_positions:
        if p.track_id >= 0:
            track_frame_counts[p.track_id] = track_frame_counts.get(p.track_id, 0) + 1

    # Keep top 4 tracks by frame count (beach volleyball = 4 players)
    sorted_tracks = sorted(track_frame_counts.items(), key=lambda x: -x[1])
    for track_id, _ in sorted_tracks[:4]:
        primary_tracks.add(track_id)
        all_track_ids.add(track_id)

    # Compute per-player stats
    for track_id in sorted(all_track_ids):
        player = PlayerStats(track_id=track_id)

        # Count actions
        for ra in rally_actions_list:
            for action in ra.actions:
                if action.player_track_id != track_id:
                    continue
                if action.action_type == ActionType.SERVE:
                    player.serves += 1
                elif action.action_type == ActionType.RECEIVE:
                    player.receives += 1
                elif action.action_type == ActionType.SET:
                    player.sets += 1
                elif action.action_type == ActionType.ATTACK:
                    player.attacks += 1
                elif action.action_type == ActionType.BLOCK:
                    player.blocks += 1
                elif action.action_type == ActionType.DIG:
                    player.digs += 1

        # Compute movement
        dist_px, dist_m, avg_speed = compute_player_movement(
            player_positions, track_id, video_width, video_height, calibrator
        )
        player.total_distance_px = dist_px
        player.total_distance_m = dist_m
        player.avg_speed_px_per_frame = avg_speed

        # Compute court side and position heatmap
        track_pos = [p for p in player_positions if p.track_id == track_id]
        player.num_frames = len(track_pos)

        if track_pos:
            avg_y = np.mean([p.y for p in track_pos])
            player.court_side = "near" if avg_y >= 0.5 else "far"
            near_count = sum(1 for p in track_pos if p.y >= 0.5)
            player.time_near_court_pct = near_count / len(track_pos)
            player.time_far_court_pct = 1.0 - player.time_near_court_pct

        player.position_heatmap = compute_position_heatmap(
            player_positions, track_id
        )

        stats.player_stats.append(player)

    # Compute per-rally stats
    for ra in rally_actions_list:
        if not ra.actions:
            continue

        frames = [a.frame for a in ra.actions]
        duration_frames = max(frames) - min(frames) if len(frames) > 1 else 0
        duration_s = duration_frames / video_fps

        action_seq = [a.action_type.value for a in ra.actions]
        has_block = any(a.action_type == ActionType.BLOCK for a in ra.actions)

        # Extended exchange: any side has 3+ consecutive contacts
        has_extended = False
        consecutive = 1
        for i in range(1, len(ra.actions)):
            if ra.actions[i].court_side == ra.actions[i - 1].court_side:
                consecutive += 1
                if consecutive >= 3:
                    has_extended = True
                    break
            else:
                consecutive = 1

        max_velocity = max((a.velocity for a in ra.actions), default=0.0)

        serve = ra.serve
        serving_side = serve.court_side if serve else "unknown"

        stats.rally_stats.append(RallyStats(
            rally_id=ra.rally_id,
            duration_frames=duration_frames,
            duration_seconds=duration_s,
            num_contacts=ra.num_contacts,
            action_sequence=action_seq,
            has_block=has_block,
            has_extended_exchange=has_extended,
            max_rally_velocity=max_velocity,
            serving_side=serving_side,
        ))

    # Compute match-level aggregates
    stats.total_rallies = len(stats.rally_stats)
    stats.total_contacts = sum(r.num_contacts for r in stats.rally_stats)

    if stats.rally_stats:
        durations = [r.duration_seconds for r in stats.rally_stats]
        stats.avg_rally_duration_s = float(np.mean(durations))
        stats.longest_rally_duration_s = max(durations)
        stats.avg_contacts_per_rally = stats.total_contacts / stats.total_rallies

    logger.info(
        f"Match stats: {stats.total_rallies} rallies, "
        f"{stats.total_contacts} contacts, "
        f"{len(stats.player_stats)} players tracked"
    )

    return stats
