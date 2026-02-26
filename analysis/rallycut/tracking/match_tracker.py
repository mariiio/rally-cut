"""
Match-level player tracking for cross-rally consistency.

Orchestrates player tracking across an entire match to maintain consistent
player IDs (1-4) across rallies using appearance-based matching.

Architecture:
    MatchPlayerTracker (orchestrates entire match)
        │
        ├── Rally 1 → PlayerTracker → RawTracks → FeatureExtractor → AppearanceFeatures
        │                                              ↓
        │                                    CrossRallyAssigner ←── MatchPlayerState
        │                                              ↓
        ├── Rally 2 → PlayerTracker → RawTracks → FeatureExtractor → AppearanceFeatures
        │                                              ↓
        │                                    CrossRallyAssigner (uses accumulated profiles)
        └── ...
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.tracking.ball_features import ServerDetectionResult, detect_server
from rallycut.tracking.identity_anchor import (
    ServeAnchor,
    ServiceOrderState,
    detect_serve_anchor,
)
from rallycut.tracking.player_features import (
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
    extract_appearance_features,
)

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class MatchPlayerState:
    """State of player assignments across a match."""

    # Player profiles (player_id 1-4 -> profile)
    players: dict[int, PlayerAppearanceProfile] = field(default_factory=dict)

    # Current side assignment (player_id -> team: 0=near, 1=far)
    current_side_assignment: dict[int, int] = field(default_factory=dict)

    # Rally indices where side switches were detected
    side_switches: list[int] = field(default_factory=list)

    # History of which player served each rally (rally_index -> player_id)
    serve_player_history: list[int] = field(default_factory=list)

    # Track to player ID assignments for current rally
    # track_id -> player_id
    current_assignments: dict[int, int] = field(default_factory=dict)

    # Service order tracking for alternation prediction
    service_order: ServiceOrderState = field(default_factory=ServiceOrderState)

    def initialize_players(self) -> None:
        """Initialize 4 player profiles for beach volleyball."""
        for player_id in range(1, 5):
            if player_id not in self.players:
                self.players[player_id] = PlayerAppearanceProfile(
                    player_id=player_id,
                    team=0 if player_id <= 2 else 1,  # Players 1-2 near, 3-4 far
                )
        # Initialize side assignments
        for player_id in range(1, 5):
            self.current_side_assignment[player_id] = 0 if player_id <= 2 else 1

    def get_player_id_for_track(self, track_id: int) -> int | None:
        """Get assigned player ID for a track, or None if not assigned."""
        return self.current_assignments.get(track_id)


@dataclass
class RallyTrackingResult:
    """Result of tracking a single rally with consistent player IDs."""

    rally_index: int
    track_to_player: dict[int, int]  # track_id -> player_id (1-4)
    server_player_id: int | None  # Player ID who served, if detected
    side_switch_detected: bool
    assignment_confidence: float  # Overall confidence in assignments


@dataclass
class RallyTrackData:
    """Data for a single rally loaded from the database."""

    rally_id: str
    video_id: str
    start_ms: int
    end_ms: int
    positions: list[PlayerPosition]
    primary_track_ids: list[int]
    court_split_y: float | None
    ball_positions: list[BallPosition]


class MatchPlayerTracker:
    """
    Orchestrates player tracking across an entire match.

    Maintains consistent player IDs (1-4) across rallies by:
    1. Extracting appearance features from each rally
    2. Matching tracks to player profiles using appearance similarity
    3. Detecting side switches based on appearance mismatches
    4. Updating profiles with new appearance data
    """

    def __init__(
        self,
        calibrator: CourtCalibrator | None = None,
    ):
        """
        Initialize match tracker.

        Args:
            calibrator: Optional court calibrator for baseline detection.
        """
        self.calibrator = calibrator
        self.state = MatchPlayerState()
        self.state.initialize_players()
        self.rally_count = 0

    def process_rally(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        player_positions: list[PlayerPosition],
        ball_positions: list[BallPosition] | None = None,
        court_split_y: float | None = None,
    ) -> RallyTrackingResult:
        """
        Process a single rally and assign consistent player IDs.

        Args:
            track_stats: Appearance stats for each track in this rally.
            player_positions: All player positions from tracking.
            ball_positions: Ball positions for server detection.
            court_split_y: Y coordinate splitting near/far teams.

        Returns:
            RallyTrackingResult with track-to-player assignments.
        """
        rally_index = self.rally_count
        self.rally_count += 1

        # Step 1: Identify server using both methods
        server_result: ServerDetectionResult | None = None
        if ball_positions and player_positions:
            server_result = detect_server(
                player_positions, ball_positions,
                rally_start_frame=0,
                calibrator=self.calibrator,
            )

        # Step 1b: Position-based serve anchor (higher recall)
        serve_anchor: ServeAnchor | None = None
        if player_positions:
            # Build team assignments from court_split_y
            team_for_anchor: dict[int, int] = {}
            if court_split_y is not None:
                track_ys: dict[int, list[float]] = defaultdict(list)
                for p in player_positions:
                    if p.track_id >= 0:
                        track_ys[p.track_id].append(p.y)
                for tid, ys in track_ys.items():
                    avg_y = float(np.mean(ys))
                    team_for_anchor[tid] = 0 if avg_y > court_split_y else 1

            if team_for_anchor:
                serve_anchor = detect_serve_anchor(
                    player_positions,
                    team_for_anchor,
                    ball_positions=ball_positions,
                    calibrator=self.calibrator,
                    serve_window_frames=30,
                )

        # Step 2: Split tracks by team (near/far court), keep top 2 per side
        near_tracks, far_tracks = self._split_tracks_by_court(
            track_stats, player_positions, court_split_y
        )
        near_tracks = self._top_tracks_by_frames(near_tracks, track_stats, 2)
        far_tracks = self._top_tracks_by_frames(far_tracks, track_stats, 2)

        # Step 3: Check for side switch (appearance-based)
        side_switch_detected = self._detect_side_switch(
            near_tracks, far_tracks, track_stats
        )

        if side_switch_detected:
            self._apply_side_switch()
            self.state.side_switches.append(rally_index)
            logger.info(f"Side switch detected at rally {rally_index}")

        # Step 4: Match tracks to player profiles using Hungarian algorithm
        track_to_player = self._assign_tracks_to_players(
            near_tracks, far_tracks, track_stats
        )

        # Step 5: Compute confidence BEFORE updating profiles (avoids inflated scores)
        confidence = self._compute_assignment_confidence(track_stats, track_to_player)

        # Step 6: Update player profiles with new appearance data
        self._update_profiles(track_stats, track_to_player)

        # Step 7: Record server if detected (use serve_anchor as fallback)
        server_player_id = None
        if server_result and server_result.track_id >= 0:
            server_player_id = track_to_player.get(server_result.track_id)
        elif serve_anchor and serve_anchor.server_track_id >= 0:
            server_player_id = track_to_player.get(serve_anchor.server_track_id)

        if server_player_id:
            self.state.serve_player_history.append(server_player_id)
            # Track service order for alternation prediction
            server_team = self.state.current_side_assignment.get(
                server_player_id, -1
            )
            if server_team >= 0 and serve_anchor:
                self.state.service_order.record_serve(
                    rally_index, server_team, serve_anchor.server_track_id
                )

        # Store current assignments
        self.state.current_assignments = track_to_player

        return RallyTrackingResult(
            rally_index=rally_index,
            track_to_player=track_to_player,
            server_player_id=server_player_id,
            side_switch_detected=side_switch_detected,
            assignment_confidence=confidence,
        )

    def _split_tracks_by_court(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        player_positions: list[PlayerPosition],
        court_split_y: float | None,
    ) -> tuple[list[int], list[int]]:
        """Split track IDs into near and far court teams."""
        if court_split_y is None:
            # Use bbox size as fallback (larger = near court)
            sizes: dict[int, float] = {}
            for p in player_positions:
                if p.track_id not in sizes:
                    sizes[p.track_id] = 0.0
                sizes[p.track_id] = max(sizes[p.track_id], p.width * p.height)

            sorted_tracks = sorted(sizes.keys(), key=lambda t: sizes.get(t, 0), reverse=True)
            mid = len(sorted_tracks) // 2
            return sorted_tracks[:mid], sorted_tracks[mid:]

        # Compute average Y position for each track
        track_avg_y: dict[int, float] = {}
        track_y_values: dict[int, list[float]] = {}

        for p in player_positions:
            if p.track_id < 0:
                continue
            if p.track_id not in track_y_values:
                track_y_values[p.track_id] = []
            track_y_values[p.track_id].append(p.y)

        for track_id, y_vals in track_y_values.items():
            track_avg_y[track_id] = float(np.mean(y_vals))

        # Split by court_split_y
        near_tracks = [t for t, y in track_avg_y.items() if y > court_split_y]
        far_tracks = [t for t, y in track_avg_y.items() if y <= court_split_y]

        return near_tracks, far_tracks

    def _detect_side_switch(
        self,
        near_tracks: list[int],
        far_tracks: list[int],
        track_stats: dict[int, TrackAppearanceStats],
    ) -> bool:
        """
        Detect if teams have switched sides based on appearance.

        Compares current assignment cost vs swapped assignment cost.
        If swapped is significantly better, switch detected.
        """
        if self.rally_count <= 2:
            # Need stable profiles (at least 2 rallies processed)
            return False

        if not near_tracks or not far_tracks:
            return False

        # Get player IDs for each team based on current assignment
        near_players = [
            pid for pid, team in self.state.current_side_assignment.items()
            if team == 0
        ]
        far_players = [
            pid for pid, team in self.state.current_side_assignment.items()
            if team == 1
        ]

        # Compute normal cost: near_tracks→near_players + far_tracks→far_players
        cost_normal = self._compute_team_assignment_cost(
            near_tracks, near_players, track_stats
        ) + self._compute_team_assignment_cost(
            far_tracks, far_players, track_stats
        )

        # Compute swapped cost: near_tracks→far_players + far_tracks→near_players
        cost_swapped = self._compute_team_assignment_cost(
            near_tracks, far_players, track_stats
        ) + self._compute_team_assignment_cost(
            far_tracks, near_players, track_stats
        )

        # Switch if swapped is significantly cheaper (30% margin)
        if cost_swapped < cost_normal * 0.7:
            logger.info(
                f"Side switch: cost_normal={cost_normal:.3f}, "
                f"cost_swapped={cost_swapped:.3f}"
            )
            return True

        return False

    def _compute_team_assignment_cost(
        self,
        track_ids: list[int],
        player_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
    ) -> float:
        """Compute optimal assignment cost for tracks to players using Hungarian."""
        if not track_ids or not player_ids:
            return 0.0

        n_tracks = min(len(track_ids), 2)
        n_players = len(player_ids)

        # Build cost matrix
        cost_matrix = np.full((n_tracks, n_players), 1.0)
        for i, tid in enumerate(track_ids[:n_tracks]):
            if tid not in track_stats:
                continue
            for j, pid in enumerate(player_ids):
                if pid not in self.state.players:
                    continue
                cost_matrix[i, j] = compute_appearance_similarity(
                    self.state.players[pid], track_stats[tid]
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return float(cost_matrix[row_ind, col_ind].sum())

    def _apply_side_switch(self) -> None:
        """Apply side switch by swapping team assignments."""
        # Swap team assignments
        for player_id in self.state.current_side_assignment:
            current = self.state.current_side_assignment[player_id]
            self.state.current_side_assignment[player_id] = 1 - current

        # Update player profiles
        for player_id, profile in self.state.players.items():
            profile.team = 1 - profile.team

    def _assign_tracks_to_players(
        self,
        near_tracks: list[int],
        far_tracks: list[int],
        track_stats: dict[int, TrackAppearanceStats],
    ) -> dict[int, int]:
        """
        Assign tracks to player IDs using Hungarian algorithm.

        Args:
            near_tracks: Track IDs on near side of court.
            far_tracks: Track IDs on far side of court.
            track_stats: Appearance stats per track for cost computation.

        Returns:
            Dictionary mapping track_id to player_id (1-4).
        """
        assignments: dict[int, int] = {}

        # Get player IDs for each team based on current assignment
        near_players = [
            pid for pid, team in self.state.current_side_assignment.items()
            if team == 0
        ]
        far_players = [
            pid for pid, team in self.state.current_side_assignment.items()
            if team == 1
        ]

        # First rally: no profiles yet, assign arbitrarily
        if self.rally_count <= 1:
            for i, tid in enumerate(near_tracks[:2]):
                if i < len(near_players):
                    assignments[tid] = near_players[i]
            for i, tid in enumerate(far_tracks[:2]):
                if i < len(far_players):
                    assignments[tid] = far_players[i]
            return assignments

        # Subsequent rallies: use Hungarian algorithm per team
        assignments.update(
            self._hungarian_assign(near_tracks, near_players, track_stats)
        )
        assignments.update(
            self._hungarian_assign(far_tracks, far_players, track_stats)
        )

        return assignments

    def _top_tracks_by_frames(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        n: int,
    ) -> list[int]:
        """Return top N tracks by feature count (most observations)."""
        if len(track_ids) <= n:
            return track_ids
        return sorted(
            track_ids,
            key=lambda t: len(track_stats[t].features) if t in track_stats else 0,
            reverse=True,
        )[:n]

    def _hungarian_assign(
        self,
        track_ids: list[int],
        player_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
    ) -> dict[int, int]:
        """Assign tracks to players using Hungarian algorithm on appearance cost."""
        if not track_ids or not player_ids:
            return {}

        n_tracks = len(track_ids)
        n_players = len(player_ids)
        size = max(n_tracks, n_players)

        # Build cost matrix padded with 1.0 for missing entries
        cost_matrix = np.full((size, size), 1.0)
        for i, tid in enumerate(track_ids):
            if tid not in track_stats:
                continue
            for j, pid in enumerate(player_ids):
                if pid not in self.state.players:
                    continue
                cost_matrix[i, j] = compute_appearance_similarity(
                    self.state.players[pid], track_stats[tid]
                )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        result: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_tracks and c < n_players:
                result[track_ids[r]] = player_ids[c]
        return result

    def _update_profiles(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        track_to_player: dict[int, int],
    ) -> None:
        """Update player profiles with new appearance data from this rally."""
        for track_id, player_id in track_to_player.items():
            if track_id not in track_stats:
                continue
            if player_id not in self.state.players:
                continue

            stats = track_stats[track_id]
            profile = self.state.players[player_id]

            # Update profile with each feature sample
            for features in stats.features:
                profile.update_from_features(features)

            profile.rally_count += 1

    def _compute_assignment_confidence(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        track_to_player: dict[int, int],
    ) -> float:
        """Compute overall confidence in track-to-player assignments."""
        if not track_to_player:
            return 0.0

        # If this is the first rally, confidence is low (no baseline)
        if self.rally_count <= 1:
            return 0.5

        # Compute average similarity cost
        costs: list[float] = []
        for track_id, player_id in track_to_player.items():
            if track_id not in track_stats:
                continue
            if player_id not in self.state.players:
                continue

            stats = track_stats[track_id]
            profile = self.state.players[player_id]

            cost = compute_appearance_similarity(profile, stats)
            costs.append(cost)

        if not costs:
            return 0.5

        # Convert cost to confidence (lower cost = higher confidence)
        avg_cost = sum(costs) / len(costs)
        return 1.0 - avg_cost

    def get_consistent_player_id(self, track_id: int) -> int | None:
        """
        Get consistent player ID for a track.

        Args:
            track_id: Track ID from player tracker.

        Returns:
            Player ID (1-4) or None if not assigned.
        """
        return self.state.current_assignments.get(track_id)

    def remap_positions(
        self,
        positions: list[PlayerPosition],
    ) -> list[tuple[PlayerPosition, int | None]]:
        """
        Remap player positions to consistent player IDs.

        Args:
            positions: List of PlayerPosition from tracking.

        Returns:
            List of (position, player_id) tuples.
        """
        return [
            (p, self.get_consistent_player_id(p.track_id))
            for p in positions
        ]


def extract_rally_appearances(
    video_path: Path,
    positions: list[PlayerPosition],
    primary_track_ids: list[int],
    start_ms: int,
    end_ms: int,
    num_samples: int = 12,
) -> dict[int, TrackAppearanceStats]:
    """
    Extract appearance features from video frames for primary tracks.

    Samples ~num_samples evenly-spaced frames per track, reads frames
    in chronological order (single seek pass), and extracts skin/jersey/height.

    Args:
        video_path: Path to the video file.
        positions: All player positions for this rally.
        primary_track_ids: Track IDs to extract features for.
        start_ms: Rally start time in milliseconds.
        end_ms: Rally end time in milliseconds.
        num_samples: Target number of frames to sample per track.

    Returns:
        Dict mapping track_id to TrackAppearanceStats with computed averages.
    """
    if not primary_track_ids or not positions:
        return {}

    primary_set = set(primary_track_ids)

    # Group positions by track_id, only for primary tracks
    track_positions: dict[int, list[PlayerPosition]] = {}
    for p in positions:
        if p.track_id in primary_set:
            if p.track_id not in track_positions:
                track_positions[p.track_id] = []
            track_positions[p.track_id].append(p)

    if not track_positions:
        return {}

    # For each track, pick evenly-spaced sample frames
    # Collect all (frame_number, track_id, position) tuples to read
    frame_requests: dict[int, list[tuple[int, PlayerPosition]]] = {}
    for tid, pos_list in track_positions.items():
        pos_list.sort(key=lambda p: p.frame_number)
        n = len(pos_list)
        if n <= num_samples:
            sample_indices = list(range(n))
        else:
            sample_indices = [
                int(i * (n - 1) / (num_samples - 1)) for i in range(num_samples)
            ]

        for idx in sample_indices:
            p = pos_list[idx]
            fn = p.frame_number
            if fn not in frame_requests:
                frame_requests[fn] = []
            frame_requests[fn].append((tid, p))

    # Sort frame numbers for sequential reading
    sorted_frames = sorted(frame_requests.keys())
    if not sorted_frames:
        return {}

    # Open video and seek to start
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Convert rally-relative frame numbers to absolute video frames
    start_frame = int(start_ms / 1000 * fps)

    # Initialize stats per track
    stats: dict[int, TrackAppearanceStats] = {
        tid: TrackAppearanceStats(track_id=tid)
        for tid in track_positions
    }

    try:
        for fn in sorted_frames:
            abs_frame = start_frame + fn
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret:
                continue

            for tid, p in frame_requests[fn]:
                bbox = (p.x, p.y, p.width, p.height)
                features = extract_appearance_features(
                    np.asarray(frame, dtype=np.uint8),
                    tid, fn, bbox, frame_width, frame_height,
                )
                stats[tid].features.append(features)
    finally:
        cap.release()

    # Compute averages
    for s in stats.values():
        s.compute_averages()

    return stats


@dataclass
class MatchPlayersResult:
    """Result of cross-rally player matching."""

    rally_results: list[RallyTrackingResult]
    player_profiles: dict[int, PlayerAppearanceProfile]  # player_id -> profile


def match_players_across_rallies(
    video_path: Path,
    rallies: list[RallyTrackData],
    num_samples: int = 12,
) -> MatchPlayersResult:
    """
    Match players across all rallies in a video for consistent IDs.

    Creates a MatchPlayerTracker and processes rallies chronologically,
    extracting appearances from video and assigning consistent player IDs 1-4.

    Args:
        video_path: Path to the video file.
        rallies: Rally data sorted chronologically.
        num_samples: Frames to sample per track for appearance.

    Returns:
        MatchPlayersResult with track→player mappings and accumulated profiles.
    """
    tracker = MatchPlayerTracker()
    results: list[RallyTrackingResult] = []

    for rally in rallies:
        # Extract appearance features from video
        track_stats = extract_rally_appearances(
            video_path=video_path,
            positions=rally.positions,
            primary_track_ids=rally.primary_track_ids,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
            num_samples=num_samples,
        )

        # Process rally
        result = tracker.process_rally(
            track_stats=track_stats,
            player_positions=rally.positions,
            ball_positions=rally.ball_positions,
            court_split_y=rally.court_split_y,
        )

        results.append(result)
        logger.info(
            f"Rally {rally.rally_id[:8]}: "
            f"confidence={result.assignment_confidence:.2f}, "
            f"switch={result.side_switch_detected}, "
            f"assignments={result.track_to_player}"
        )

    return MatchPlayersResult(
        rally_results=results,
        player_profiles=dict(tracker.state.players),
    )
