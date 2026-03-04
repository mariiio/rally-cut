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
from typing import TYPE_CHECKING, Any

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

# Side penalty for global Hungarian assignment. Biases toward expected court side
# but doesn't prevent cross-side matching when appearance is a stronger signal.
# Appearance costs range 0.0-1.0, so 0.15 is meaningful but not dominant.
SIDE_PENALTY = 0.15


def build_match_team_assignments(
    match_analysis: dict[str, Any],
    min_confidence: float = 0.0,
) -> dict[str, dict[int, int]]:
    """Build per-rally team assignments from match analysis JSON.

    Derives team labels (0=near, 1=far) for each track in each rally,
    accounting for cumulative side switches across rallies.

    Convention: player IDs 1-2 = team 0 (near), 3-4 = team 1 (far)
    at baseline. Each side switch flips the mapping.

    Args:
        match_analysis: The match_analysis_json from the videos table.
        min_confidence: Skip rallies below this assignment confidence.

    Returns:
        Dict of rally_id -> {track_id: team (0 or 1)}.
    """
    rallies = match_analysis.get("rallies", [])
    if not isinstance(rallies, list):
        return {}

    result: dict[str, dict[int, int]] = {}
    side_switch_count = 0

    for rally_entry in rallies:
        if rally_entry.get("sideSwitchDetected") or rally_entry.get(
            "side_switch_detected"
        ):
            side_switch_count += 1

        track_to_player = rally_entry.get("trackToPlayer") or rally_entry.get(
            "track_to_player", {}
        )
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        if not rid or not track_to_player:
            continue

        if min_confidence > 0:
            conf = rally_entry.get("assignmentConfidence") or rally_entry.get(
                "assignment_confidence", 0
            )
            if conf < min_confidence:
                continue

        teams: dict[int, int] = {}
        for tid_str, player_id in track_to_player.items():
            pid = int(player_id)
            base_team = 0 if pid <= 2 else 1
            team = base_team if side_switch_count % 2 == 0 else 1 - base_team
            teams[int(tid_str)] = team

        if teams:
            result[rid] = teams

    return result


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

    # Last known position per player (avg of last N frames of previous rally)
    player_last_positions: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

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
    team_assignments: dict[int, int] | None = None  # track_id -> team (0=near, 1=far)


def _compute_track_positions(
    positions: list[PlayerPosition],
    track_ids: list[int],
    window: int = 30,
    *,
    from_start: bool = True,
) -> dict[int, tuple[float, float]]:
    """Compute avg (x, y) for each track from the first or last N frames."""
    track_id_set = set(track_ids)
    by_track: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id in track_id_set:
            by_track[p.track_id].append(p)

    result: dict[int, tuple[float, float]] = {}
    for tid in track_ids:
        pts = by_track.get(tid)
        if not pts:
            continue
        pts.sort(key=lambda p: p.frame_number)
        subset = pts[:window] if from_start else pts[-window:]
        avg_x = sum(p.x for p in subset) / len(subset)
        avg_y = sum(p.y for p in subset) / len(subset)
        result[tid] = (avg_x, avg_y)
    return result


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


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
        team_assignments: dict[int, int] | None = None,
    ) -> RallyTrackingResult:
        """
        Process a single rally and assign consistent player IDs.

        Args:
            track_stats: Appearance stats for each track in this rally.
            player_positions: All player positions from tracking.
            ball_positions: Ball positions for server detection.
            court_split_y: Y coordinate splitting near/far teams.
            team_assignments: Pre-computed track_id -> team (0=near, 1=far)
                from the tracking pipeline's bbox-size clustering.

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

        # Step 2: Classify track sides (soft near/far labels)
        track_avg_y, track_court_sides = self._classify_track_sides(
            track_stats, player_positions, court_split_y, team_assignments
        )

        # Step 3: Select top 4 tracks globally by feature count
        all_track_ids = list(track_court_sides.keys())
        top_tracks = self._top_tracks_by_frames(all_track_ids, track_stats, 4)

        # Step 4: Assign tracks to players
        if self.rally_count <= 1:
            # First rally: deterministic assignment by Y position
            track_to_player = self._initialize_first_rally(
                top_tracks, track_avg_y, track_court_sides
            )
            side_switch_detected = False
        else:
            # Step 4a: Run penalty-free assignment for side switch detection.
            # The side penalty would bias against cross-side matches, preventing
            # switch detection when appearance isn't strong enough to overcome it.
            unpenalized = self._assign_tracks_to_players_global(
                top_tracks, track_stats, track_court_sides, use_side_penalty=False
            )

            # Step 4b: Check for side switch from unpenalized assignment
            side_switch_detected = self._detect_side_switch_from_assignment(
                unpenalized, track_court_sides
            )

            if side_switch_detected:
                self._apply_side_switch()
                self.state.side_switches.append(rally_index)
                logger.info(f"Side switch detected at rally {rally_index}")

            # Step 4c: Final assignment with side penalty (uses updated sides if switched)
            track_to_player = self._assign_tracks_to_players_global(
                top_tracks, track_stats, track_court_sides
            )

        # Step 5: Within-team refinement using position continuity
        if self.rally_count > 1:
            track_to_player = self._refine_within_team(
                track_to_player, player_positions, track_court_sides
            )

        # Store late-rally positions for next rally's continuity check
        self._store_last_positions(track_to_player, player_positions)

        # Step 6: Compute confidence BEFORE updating profiles (avoids inflated scores)
        confidence = self._compute_assignment_confidence(track_stats, track_to_player)

        # Step 7: Update player profiles with new appearance data
        self._update_profiles(track_stats, track_to_player)

        # Step 8: Record server if detected (use serve_anchor as fallback)
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

    def _classify_track_sides(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        player_positions: list[PlayerPosition],
        court_split_y: float | None,
        team_assignments: dict[int, int] | None = None,
    ) -> tuple[dict[int, float], dict[int, int]]:
        """Classify tracks into near/far court with soft labels.

        Priority: team_assignments (from tracking pipeline) > court_split_y > median.
        team_assignments are reliable because they use bbox size clustering
        (near players have larger bboxes), not Y position alone.

        Args:
            track_stats: Appearance stats per track.
            player_positions: All player positions for this rally.
            court_split_y: Y coordinate splitting near/far teams.
            team_assignments: Pre-computed track_id -> team (0=near, 1=far)
                from the tracking pipeline's actions_json.teamAssignments.

        Returns:
            Tuple of (track_avg_y, track_court_sides) where:
                track_avg_y: track_id -> average Y position
                track_court_sides: track_id -> 0 (near) or 1 (far)
        """
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

        track_court_sides: dict[int, int] = {}

        # Priority 1: Use pre-computed team_assignments if they cover most tracks
        if team_assignments:
            covered = [t for t in track_avg_y if t in team_assignments]
            if len(covered) >= len(track_avg_y) * 0.75:
                for t in track_avg_y:
                    if t in team_assignments:
                        track_court_sides[t] = team_assignments[t]
                    else:
                        # Uncovered track: fallback to court_split_y or median
                        if court_split_y is not None:
                            track_court_sides[t] = (
                                0 if track_avg_y[t] > court_split_y else 1
                            )
                        else:
                            track_court_sides[t] = 0  # default near
                logger.info(
                    "Using team_assignments for %d/%d tracks",
                    len(covered),
                    len(track_avg_y),
                )
                return track_avg_y, track_court_sides

        # Priority 2: court_split_y
        if court_split_y is not None:
            # Try splitting by court_split_y
            near = [t for t in track_avg_y if track_avg_y[t] > court_split_y]
            far = [t for t in track_avg_y if track_avg_y[t] <= court_split_y]

            if near and far:
                # Good split — use it
                for t in near:
                    track_court_sides[t] = 0  # near
                for t in far:
                    track_court_sides[t] = 1  # far
                return track_avg_y, track_court_sides

            # All tracks on one side — fall through to median split
            if len(track_avg_y) >= 4:
                logger.info(
                    "court_split_y=%.3f put all %d tracks on one side, "
                    "using median-index split",
                    court_split_y,
                    len(track_avg_y),
                )

        if not track_avg_y:
            return track_avg_y, track_court_sides

        # Priority 3: sort by Y, split at median index
        # Higher Y = near court (closer to camera)
        sorted_tracks = sorted(track_avg_y.keys(), key=lambda t: track_avg_y[t])
        mid = len(sorted_tracks) // 2
        for t in sorted_tracks[:mid]:
            track_court_sides[t] = 1  # far (lower Y)
        for t in sorted_tracks[mid:]:
            track_court_sides[t] = 0  # near (higher Y)

        return track_avg_y, track_court_sides

    def _initialize_first_rally(
        self,
        track_ids: list[int],
        track_avg_y: dict[int, float],
        track_court_sides: dict[int, int],
    ) -> dict[int, int]:
        """Deterministic first-rally assignment sorted by Y within each team.

        Lowest Y in near team -> P1, next -> P2.
        Lowest Y in far team -> P3, next -> P4.
        Eliminates dependency on dict/list ordering.

        Args:
            track_ids: Top tracks to assign (up to 4).
            track_avg_y: Average Y position per track.
            track_court_sides: Track -> 0 (near) or 1 (far).

        Returns:
            track_id -> player_id mapping.
        """
        near_players = sorted(
            pid for pid, team in self.state.current_side_assignment.items()
            if team == 0
        )
        far_players = sorted(
            pid for pid, team in self.state.current_side_assignment.items()
            if team == 1
        )

        # Split tracks by assigned side, sort by Y within each side
        near_tracks = sorted(
            [t for t in track_ids if track_court_sides.get(t) == 0],
            key=lambda t: track_avg_y.get(t, 0.5),
        )
        far_tracks = sorted(
            [t for t in track_ids if track_court_sides.get(t) == 1],
            key=lambda t: track_avg_y.get(t, 0.5),
        )

        assignments: dict[int, int] = {}
        for i, tid in enumerate(near_tracks[:2]):
            if i < len(near_players):
                assignments[tid] = near_players[i]
        for i, tid in enumerate(far_tracks[:2]):
            if i < len(far_players):
                assignments[tid] = far_players[i]

        return assignments

    def _assign_tracks_to_players_global(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        track_court_sides: dict[int, int],
        *,
        use_side_penalty: bool = True,
    ) -> dict[int, int]:
        """Global 4x4 Hungarian assignment with optional side penalty.

        Builds a single cost matrix across all players instead of
        per-team split-then-match. Side penalty biases toward expected
        court side but doesn't prevent cross-side matching.

        Args:
            track_ids: Track IDs to assign (up to 4).
            track_stats: Appearance stats per track.
            track_court_sides: Track -> 0 (near) or 1 (far).
            use_side_penalty: Whether to add side penalty to cost matrix.

        Returns:
            track_id -> player_id mapping.
        """
        if not track_ids:
            return {}

        all_player_ids = sorted(self.state.players.keys())  # [1, 2, 3, 4]
        n_tracks = len(track_ids)
        n_players = len(all_player_ids)
        size = max(n_tracks, n_players)

        # Build cost matrix: appearance + optional side penalty
        default_cost = 1.0 + (SIDE_PENALTY if use_side_penalty else 0.0)
        cost_matrix = np.full((size, size), default_cost)
        for i, tid in enumerate(track_ids):
            if tid not in track_stats:
                continue
            track_side = track_court_sides.get(tid)
            for j, pid in enumerate(all_player_ids):
                if pid not in self.state.players:
                    continue
                appearance_cost = compute_appearance_similarity(
                    self.state.players[pid], track_stats[tid]
                )
                # Add side penalty if track and player are on different sides
                if use_side_penalty:
                    player_side = self.state.current_side_assignment.get(pid)
                    side_pen = SIDE_PENALTY if track_side != player_side else 0.0
                else:
                    side_pen = 0.0
                cost_matrix[i, j] = appearance_cost + side_pen

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        result: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_tracks and c < n_players:
                result[track_ids[r]] = all_player_ids[c]
        return result

    def _detect_side_switch_from_assignment(
        self,
        track_to_player: dict[int, int],
        track_court_sides: dict[int, int],
    ) -> bool:
        """Detect side switch from global assignment result.

        After global assignment, count how many players ended up on the
        opposite side from their profile. If >=3 of 4 flipped -> switch.

        Args:
            track_to_player: Assignment from global Hungarian.
            track_court_sides: Track -> 0 (near) or 1 (far).

        Returns:
            True if side switch detected.
        """
        if self.rally_count <= 2:
            return False

        if len(track_to_player) < 3:
            return False

        flipped = 0
        total = 0
        for tid, pid in track_to_player.items():
            track_side = track_court_sides.get(tid)
            player_expected_side = self.state.current_side_assignment.get(pid)
            if track_side is None or player_expected_side is None:
                continue
            total += 1
            if track_side != player_expected_side:
                flipped += 1

        if total >= 3 and flipped >= 3:
            logger.info(
                "Side switch detected: %d/%d players on opposite side",
                flipped,
                total,
            )
            return True

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

        # Clear position continuity — invalid when teams swap court sides
        self.state.player_last_positions.clear()

    def _refine_within_team(
        self,
        track_to_player: dict[int, int],
        player_positions: list[PlayerPosition],
        track_court_sides: dict[int, int],
    ) -> dict[int, int]:
        """Refine within-team player assignments using position continuity.

        For each team of 2 tracks, compare early-rally positions to previous
        rally's late positions. Swap within-team assignment if it reduces
        total distance by >20% (prevents noise-driven flips).
        """
        if not self.state.player_last_positions:
            return track_to_player

        for team in [0, 1]:
            team_tracks = [
                tid
                for tid, pid in track_to_player.items()
                if track_court_sides.get(tid) == team
            ]
            if len(team_tracks) != 2:
                continue

            t1, t2 = team_tracks
            p1, p2 = track_to_player[t1], track_to_player[t2]

            # Need last positions for both players
            if p1 not in self.state.player_last_positions:
                continue
            if p2 not in self.state.player_last_positions:
                continue

            # Compute early-rally positions
            early = _compute_track_positions(
                player_positions, [t1, t2], window=30, from_start=True
            )
            if t1 not in early or t2 not in early:
                continue

            last_p1 = self.state.player_last_positions[p1]
            last_p2 = self.state.player_last_positions[p2]

            cost_keep = _dist(early[t1], last_p1) + _dist(early[t2], last_p2)
            cost_swap = _dist(early[t1], last_p2) + _dist(early[t2], last_p1)

            # Only swap if clearly better (>20% improvement prevents noise flips)
            if cost_swap < cost_keep * 0.80:
                track_to_player[t1] = p2
                track_to_player[t2] = p1
                logger.info(
                    "Within-team swap: team %d, tracks %d↔%d "
                    "(keep=%.3f, swap=%.3f, improvement=%.0f%%)",
                    team, t1, t2, cost_keep, cost_swap,
                    (1 - cost_swap / cost_keep) * 100 if cost_keep > 0 else 0,
                )

        return track_to_player

    def _store_last_positions(
        self,
        track_to_player: dict[int, int],
        player_positions: list[PlayerPosition],
    ) -> None:
        """Store each player's late-rally position for next rally's continuity check."""
        late = _compute_track_positions(
            player_positions,
            list(track_to_player.keys()),
            window=30,
            from_start=False,
        )
        for tid, pid in track_to_player.items():
            if tid in late:
                self.state.player_last_positions[pid] = late[tid]

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
            team_assignments=rally.team_assignments,
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
