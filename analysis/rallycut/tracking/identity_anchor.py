"""Serve-based identity anchoring for player tracking.

Uses player positions at rally start to identify the server and anchor
all 4 player identities. Combined with service order tracking across
rallies, this provides repeated identity anchors at every rally boundary.

Beach volleyball serving rules:
- Within each team, servers alternate (A, B, A, B...)
- Server stands near the baseline at rally start
- The serving team is the team with a player near the baseline
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Court constants
NET_Y = 8.0  # meters (court center)
NEAR_BASELINE_Y = 0.0  # meters
FAR_BASELINE_Y = 16.0  # meters


@dataclass
class ServeAnchor:
    """Identity anchor from serve detection at rally start."""

    server_track_id: int
    server_team: int  # 0=near, 1=far
    serve_frame: int
    confidence: float

    # Other players identified by position
    teammate_track_id: int = -1  # Same team, not serving
    receiver_track_ids: list[int] = field(default_factory=list)  # Opposite team


@dataclass
class ServiceOrderState:
    """Tracks service order within each team across rallies."""

    # team -> list of (rally_index, player_track_id) for detected serves
    team_serve_history: dict[int, list[tuple[int, int]]] = field(
        default_factory=lambda: {0: [], 1: []}
    )

    def record_serve(
        self, rally_index: int, team: int, track_id: int
    ) -> None:
        """Record a detected serve."""
        self.team_serve_history[team].append((rally_index, track_id))

    def predict_server_track(
        self,
        team: int,
        current_tracks: list[int],
    ) -> tuple[int, float]:
        """Predict which track should serve next based on alternation.

        Returns (predicted_track_id, confidence).
        """
        history = self.team_serve_history.get(team, [])
        if len(history) < 2:
            return -1, 0.0

        # Get the last two servers for this team
        last_server = history[-1][1]
        second_last_server = history[-2][1]

        if last_server == second_last_server:
            # Same player served twice — alternation says the OTHER should serve
            candidates = [t for t in current_tracks if t != last_server]
            if candidates:
                return candidates[0], 0.6
            return -1, 0.0

        # Alternation pattern: next should be same as second_last
        if second_last_server in current_tracks:
            return second_last_server, 0.7

        return -1, 0.0


def detect_serve_anchor(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    ball_positions: list[BallPosition] | None = None,
    calibrator: CourtCalibrator | None = None,
    video_width: int = 1920,
    video_height: int = 1080,
    serve_window_frames: int = 30,
) -> ServeAnchor | None:
    """Detect serve anchor using player positions at rally start.

    Primary signal: player position near baseline in first 30 frames.
    Secondary signal: ball trajectory direction (if available).

    Args:
        positions: Player positions for the rally.
        team_assignments: track_id -> team (0=near, 1=far).
        ball_positions: Optional ball positions for trajectory validation.
        calibrator: Optional court calibrator for metric distances.
        video_width: Video width in pixels.
        video_height: Video height in pixels.
        serve_window_frames: Frames to look at for serve detection.

    Returns:
        ServeAnchor if detected, None otherwise.
    """
    if not positions or not team_assignments:
        return None

    # Group early-frame positions by track
    early_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    min_frame = min(p.frame_number for p in positions)

    for p in positions:
        if (
            p.track_id >= 0
            and p.track_id in team_assignments
            and p.frame_number <= min_frame + serve_window_frames
        ):
            early_positions[p.track_id].append(p)

    if not early_positions:
        return None

    # Compute mean position for each track in early frames
    track_means: dict[int, tuple[float, float]] = {}
    for tid, pos_list in early_positions.items():
        mean_x = float(np.mean([p.x for p in pos_list]))
        mean_y = float(np.mean([p.y for p in pos_list]))
        track_means[tid] = (mean_x, mean_y)

    # Strategy: use court coordinates if calibrated, else use image Y
    server_tid = -1
    server_team = -1
    confidence = 0.0

    if calibrator is not None and calibrator.is_calibrated:
        server_tid, server_team, confidence = _detect_server_court_space(
            track_means, team_assignments, calibrator, video_width, video_height
        )
    else:
        server_tid, server_team, confidence = _detect_server_image_space(
            track_means, team_assignments
        )

    if server_tid < 0 or confidence < 0.3:
        return None

    # Cross-validate with ball trajectory if available
    if ball_positions:
        ball_conf = _validate_with_ball(
            server_tid, server_team, ball_positions, positions,
            min_frame, serve_window_frames
        )
        confidence = 0.7 * confidence + 0.3 * ball_conf

    # Identify other players
    anchor = ServeAnchor(
        server_track_id=server_tid,
        server_team=server_team,
        serve_frame=min_frame,
        confidence=confidence,
    )

    # Find teammate (same team, not server)
    teammates = [
        tid for tid, team in team_assignments.items()
        if team == server_team and tid != server_tid and tid in track_means
    ]
    if teammates:
        anchor.teammate_track_id = teammates[0]

    # Find receivers (opposite team)
    anchor.receiver_track_ids = [
        tid for tid, team in team_assignments.items()
        if team != server_team and tid in track_means
    ]

    logger.info(
        f"Serve anchor: track {server_tid} (team {server_team}), "
        f"confidence={confidence:.2f}"
    )

    return anchor


def _detect_server_court_space(
    track_means: dict[int, tuple[float, float]],
    team_assignments: dict[int, int],
    calibrator: CourtCalibrator,
    video_width: int,
    video_height: int,
) -> tuple[int, int, float]:
    """Detect server using court-space baseline proximity."""
    best_tid = -1
    best_team = -1
    best_baseline_dist = float("inf")

    for tid, (img_x, img_y) in track_means.items():
        team = team_assignments.get(tid, -1)
        if team < 0:
            continue

        try:
            court_x, court_y = calibrator.image_to_court(
                (img_x, img_y), video_width, video_height
            )
        except (RuntimeError, ValueError):
            continue

        # Compute distance to team's baseline
        if team == 0:  # Near team → baseline at Y=0
            baseline_dist = abs(court_y - NEAR_BASELINE_Y)
        else:  # Far team → baseline at Y=16
            baseline_dist = abs(court_y - FAR_BASELINE_Y)

        if baseline_dist < best_baseline_dist:
            best_baseline_dist = baseline_dist
            best_tid = tid
            best_team = team

    if best_tid < 0:
        return -1, -1, 0.0

    # Confidence based on baseline distance (within 3m is high confidence)
    confidence = max(0.0, 1.0 - best_baseline_dist / 6.0)

    # Check if the other player on the same team is further from baseline
    # (server is always closer to baseline than teammate)
    same_team_tracks = [
        tid for tid, team in team_assignments.items()
        if team == best_team and tid != best_tid and tid in track_means
    ]
    if same_team_tracks:
        teammate_tid = same_team_tracks[0]
        teammate_img = track_means[teammate_tid]
        try:
            _, teammate_cy = calibrator.image_to_court(
                teammate_img, video_width, video_height
            )
            if best_team == 0:
                teammate_dist = abs(teammate_cy - NEAR_BASELINE_Y)
            else:
                teammate_dist = abs(teammate_cy - FAR_BASELINE_Y)

            if teammate_dist > best_baseline_dist:
                confidence = min(1.0, confidence + 0.15)
            else:
                confidence = max(0.0, confidence - 0.2)
        except (RuntimeError, ValueError):
            pass

    return best_tid, best_team, confidence


def _detect_server_image_space(
    track_means: dict[int, tuple[float, float]],
    team_assignments: dict[int, int],
) -> tuple[int, int, float]:
    """Detect server using image-space Y position (fallback)."""
    # Near team: higher Y = closer to camera = closer to near baseline
    # Far team: lower Y = further from camera = closer to far baseline

    best_tid = -1
    best_team = -1
    best_score = -1.0

    for tid, (_, img_y) in track_means.items():
        team = team_assignments.get(tid, -1)
        if team < 0:
            continue

        # Score: how extreme is the Y position for this team
        if team == 0:  # Near team — baseline at high Y
            score = img_y  # Higher Y = more likely server
        else:  # Far team — baseline at low Y
            score = 1.0 - img_y  # Lower Y = more likely server

        if score > best_score:
            best_score = score
            best_tid = tid
            best_team = team

    if best_tid < 0:
        return -1, -1, 0.0

    # Confidence based on how extreme the position is
    confidence = min(1.0, best_score * 1.5)  # Scale: 0.67 → 1.0
    return best_tid, best_team, confidence


def _validate_with_ball(
    server_tid: int,
    server_team: int,
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    min_frame: int,
    window: int,
) -> float:
    """Validate serve detection using ball trajectory."""
    # Find ball positions in serve window
    ball_in_window = [
        bp for bp in ball_positions
        if min_frame <= bp.frame_number <= min_frame + window
        and bp.confidence >= 0.3
        and not (bp.x == 0.0 and bp.y == 0.0)
    ]

    if len(ball_in_window) < 5:
        return 0.5  # No data → neutral

    # Check if ball starts near the server's position
    server_positions = [
        p for p in player_positions
        if p.track_id == server_tid
        and min_frame <= p.frame_number <= min_frame + window
    ]

    if not server_positions:
        return 0.5

    # Check proximity of first ball detections to server
    first_balls = ball_in_window[:5]
    server_xy = np.mean([(p.x, p.y) for p in server_positions[:5]], axis=0)

    distances = [
        ((bp.x - server_xy[0]) ** 2 + (bp.y - server_xy[1]) ** 2) ** 0.5
        for bp in first_balls
    ]
    mean_dist = float(np.mean(distances))

    # Close to server = higher confidence
    return max(0.0, 1.0 - mean_dist / 0.3)
