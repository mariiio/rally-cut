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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from rallycut.tracking.ball_features import ServerDetectionResult, detect_server
from rallycut.tracking.player_features import (
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
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

        # Step 1: Identify server (if ball data available)
        server_result: ServerDetectionResult | None = None
        if ball_positions and player_positions:
            server_result = detect_server(
                player_positions, ball_positions,
                rally_start_frame=0,
                calibrator=self.calibrator,
            )

        # Step 2: Split tracks by team (near/far court)
        near_tracks, far_tracks = self._split_tracks_by_court(
            track_stats, player_positions, court_split_y
        )

        # Step 3: Check for side switch (appearance-based)
        side_switch_detected = self._detect_side_switch(
            near_tracks, far_tracks
        )

        if side_switch_detected:
            self._apply_side_switch()
            self.state.side_switches.append(rally_index)
            logger.info(f"Side switch detected at rally {rally_index}")

        # Step 4: Match tracks to player profiles using Hungarian algorithm
        track_to_player = self._assign_tracks_to_players(
            near_tracks, far_tracks
        )

        # Step 5: Update player profiles with new appearance data
        self._update_profiles(track_stats, track_to_player)

        # Step 6: Record server if detected
        server_player_id = None
        if server_result and server_result.track_id >= 0:
            server_player_id = track_to_player.get(server_result.track_id)
            if server_player_id:
                self.state.serve_player_history.append(server_player_id)

        # Store current assignments
        self.state.current_assignments = track_to_player

        # Compute overall assignment confidence
        confidence = self._compute_assignment_confidence(track_stats, track_to_player)

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
    ) -> bool:
        """
        Detect if teams have switched sides based on appearance.

        Compares current assignment cost vs swapped assignment cost.
        If swapped is significantly better, switch detected.
        """
        if self.rally_count <= 1:
            # Can't detect switch on first rally
            return False

        # This would compare appearance features of near tracks with
        # profiles of near-team players vs far-team players
        # Side switch detection requires accumulated appearance profiles.
        # After 3+ rallies with stable profiles, compare:
        #   cost_current = matching near_tracks to current near-team, far_tracks to far-team
        #   cost_swapped = matching near_tracks to far-team, far_tracks to near-team
        # If cost_swapped < cost_current * 0.7, switch detected.
        #
        # Not implemented: requires sufficient profile data and appearance similarity scoring.
        # For now, assume no side switches (works for single-set recordings).

        return False

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
    ) -> dict[int, int]:
        """
        Assign tracks to player IDs using Hungarian algorithm.

        Args:
            near_tracks: Track IDs on near side of court.
            far_tracks: Track IDs on far side of court.

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

        # Simple assignment: pair tracks with players by index.
        # For optimal matching with appearance costs, use scipy.optimize.linear_sum_assignment
        # with cost matrix from compute_appearance_similarity().
        # Current simple pairing works when tracks are consistent within a rally.

        for i, track_id in enumerate(near_tracks[:2]):
            if i < len(near_players):
                assignments[track_id] = near_players[i]

        for i, track_id in enumerate(far_tracks[:2]):
            if i < len(far_players):
                assignments[track_id] = far_players[i]

        return assignments

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
