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
    detect_serve_anchor,
)
from rallycut.tracking.player_features import (
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
    compute_track_similarity,
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

# Position continuity weight in the cost matrix. When last positions are known,
# the distance between early-rally track position and previous rally's late position
# is blended into the cost. This is critical for within-team discrimination where
# appearance discriminability is poor (typical cost gap 0.02-0.05 between teammates).
POSITION_WEIGHT = 0.30

# Minimum assignment confidence to update profiles. Below this threshold,
# profile updates are skipped to prevent error propagation (drift).
MIN_PROFILE_UPDATE_CONFIDENCE = 0.55


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
class RallyAssignmentDiagnostics:
    """Diagnostics for a single rally's assignment decision."""

    rally_index: int
    cost_matrix: np.ndarray  # n_tracks x n_players appearance-only costs
    track_ids: list[int]
    player_ids: list[int]
    track_court_sides: dict[int, int]
    assignment: dict[int, int]  # track_id -> player_id
    assignment_margins: dict[int, float]  # player_id -> margin (2nd best - best)


@dataclass
class MatchPlayerState:
    """State of player assignments across a match."""

    # Player profiles (player_id 1-4 -> profile)
    players: dict[int, PlayerAppearanceProfile] = field(default_factory=dict)

    # Current side assignment (player_id -> team: 0=near, 1=far)
    current_side_assignment: dict[int, int] = field(default_factory=dict)

    # Track to player ID assignments for current rally
    # track_id -> player_id
    current_assignments: dict[int, int] = field(default_factory=dict)

    # Last known position per player (avg of last N frames of previous rally)
    player_last_positions: dict[int, tuple[float, float]] = field(
        default_factory=dict
    )

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


@dataclass
class StoredRallyData:
    """Per-rally data stored during Pass 1 for Pass 2 refinement."""

    track_stats: dict[int, TrackAppearanceStats]
    track_court_sides: dict[int, int]
    early_positions: dict[int, tuple[float, float]]
    top_tracks: list[int]
    # Snapshot of player→side mapping at this rally (before any switch applied)
    player_side_assignment: dict[int, int] = field(default_factory=dict)
    # Ball trajectory serve direction: "near", "far", or "?" (unknown)
    serve_direction: str = "?"


def _team_match_cost(
    tids_a: list[int],
    stats_a: dict[int, TrackAppearanceStats],
    tids_b: list[int],
    stats_b: dict[int, TrackAppearanceStats],
) -> float:
    """Best-matching cost between two sets of tracks."""
    if not tids_a or not tids_b:
        return 1.0
    if len(tids_a) == 1 and len(tids_b) == 1:
        return compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]])
    if len(tids_a) >= 2 and len(tids_b) >= 2:
        cost_ab = (
            compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]])
            + compute_track_similarity(stats_a[tids_a[1]], stats_b[tids_b[1]])
        )
        cost_ba = (
            compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[1]])
            + compute_track_similarity(stats_a[tids_a[1]], stats_b[tids_b[0]])
        )
        return min(cost_ab, cost_ba) / 2.0
    return compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]])


def _detect_serve_direction(
    ball_positions: list[BallPosition] | None,
) -> str:
    """Detect serve direction from ball trajectory in first ~45 frames.

    Returns "near" if ball moves upward (near team served),
    "far" if ball moves downward (far team served), "?" if unknown.
    """
    if not ball_positions:
        return "?"

    valid = sorted(
        [b for b in ball_positions
         if b.confidence >= 0.3 and not (b.x == 0.0 and b.y == 0.0)],
        key=lambda b: b.frame_number,
    )
    if len(valid) < 5:
        return "?"

    # Use first 45 frames of ball data
    first_frame = valid[0].frame_number
    early = [b for b in valid if b.frame_number <= first_frame + 45]
    if len(early) < 5:
        return "?"

    # Compare first half vs second half mean Y
    mid = len(early) // 2
    ys = [b.y for b in early]
    y_start = float(np.mean(ys[:mid]))
    y_end = float(np.mean(ys[mid:]))
    dy = y_end - y_start

    # Positive dy = ball moving down = far team served
    # Negative dy = ball moving up = near team served
    if abs(dy) < 0.01:
        return "?"
    return "near" if dy < 0 else "far"


class MatchPlayerTracker:
    """
    Orchestrates player tracking across an entire match.

    Maintains consistent player IDs (1-4) across rallies by:
    1. Extracting appearance features from each rally
    2. Matching tracks to player profiles using appearance similarity
    3. Updating profiles with new appearance data

    Side switch detection uses combinatorial search over ball trajectory
    direction candidates with normalized pairwise appearance scoring.
    """

    def __init__(
        self,
        calibrator: CourtCalibrator | None = None,
        collect_diagnostics: bool = False,
    ):
        """
        Initialize match tracker.

        Args:
            calibrator: Optional court calibrator for baseline detection.
            collect_diagnostics: If True, collect per-rally cost matrices
                and assignment margins for diagnostic analysis.
        """
        self.calibrator = calibrator
        self.state = MatchPlayerState()
        self.state.initialize_players()
        self.rally_count = 0
        self.collect_diagnostics = collect_diagnostics
        self.diagnostics: list[RallyAssignmentDiagnostics] = []
        self.stored_rally_data: list[StoredRallyData] = []

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

        # Compute early-rally positions for position continuity
        early_positions = _compute_track_positions(
            player_positions, top_tracks, window=30, from_start=True
        )

        # Step 4: Assign tracks to players
        # Side switch detection runs in Pass 2 (combinatorial search)
        side_switch_detected = False

        if self.rally_count <= 1:
            track_to_player = self._initialize_first_rally(
                top_tracks, track_avg_y, track_court_sides
            )
        else:
            track_to_player = self._assign_tracks_to_players_global(
                top_tracks, track_stats, track_court_sides,
                early_positions=early_positions,
            )

        # Step 5: Within-team refinement
        if self.rally_count > 1:
            track_to_player = self._refine_within_team(
                track_to_player, player_positions, track_court_sides
            )

        # Store late-rally positions for next rally's continuity check
        self._store_last_positions(track_to_player, player_positions)

        # Step 6: Compute confidence BEFORE updating profiles
        confidence = self._compute_assignment_confidence(track_stats, track_to_player)

        # Step 7: Update player profiles (gated on confidence)
        if self.rally_count <= 1:
            self._update_profiles(track_stats, track_to_player)
        elif confidence >= MIN_PROFILE_UPDATE_CONFIDENCE:
            self._update_profiles(track_stats, track_to_player)
        else:
            logger.info(
                f"Skipping profile update: confidence {confidence:.2f}"
            )

        # Step 8: Record server if detected (use serve_anchor as fallback)
        server_player_id = None
        if server_result and server_result.track_id >= 0:
            server_player_id = track_to_player.get(server_result.track_id)
        elif serve_anchor and serve_anchor.server_track_id >= 0:
            server_player_id = track_to_player.get(serve_anchor.server_track_id)

        # Store current assignments
        self.state.current_assignments = track_to_player

        # Store rally data for Pass 2 refinement.
        # Snapshot current_side_assignment so Pass 2 uses the correct
        # player→side mapping for this rally (not the final post-all-switches state).
        serve_dir = _detect_serve_direction(ball_positions)
        self.stored_rally_data.append(StoredRallyData(
            track_stats=track_stats,
            track_court_sides=track_court_sides,
            early_positions=early_positions,
            top_tracks=top_tracks,
            player_side_assignment=dict(self.state.current_side_assignment),
            serve_direction=serve_dir,
        ))

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

            # All tracks on one side — try team_assignments with any coverage
            # before falling back to median split
            if team_assignments:
                covered = [t for t in track_avg_y if t in team_assignments]
                if covered:
                    for t in track_avg_y:
                        if t in team_assignments:
                            track_court_sides[t] = team_assignments[t]
                        else:
                            track_court_sides[t] = 0  # default near
                    # Verify we got a real split (not all same team)
                    teams_seen = set(track_court_sides.values())
                    if len(teams_seen) >= 2:
                        logger.info(
                            "court_split_y failed, using team_assignments "
                            "for %d/%d tracks",
                            len(covered),
                            len(track_avg_y),
                        )
                        return track_avg_y, track_court_sides
                    # All same team — clear and fall through
                    track_court_sides.clear()

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
        """First-rally assignment sorted by Y within each team.

        Args:
            track_ids: Top tracks to assign (up to 4).
            track_avg_y: Average Y position per track.
            track_court_sides: Track -> 0 (near) or 1 (far).

        Returns:
            track_id -> player_id mapping (Y-sorted default).
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

        # Default Y-sorted assignment
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
        early_positions: dict[int, tuple[float, float]] | None = None,
    ) -> dict[int, int]:
        """Global 4x4 Hungarian assignment with side + position costs.

        Builds a single cost matrix across all players instead of
        per-team split-then-match. Side penalty biases toward expected
        court side. Position continuity biases toward spatial consistency
        with previous rally (critical for within-team discrimination).

        Args:
            track_ids: Track IDs to assign (up to 4).
            track_stats: Appearance stats per track.
            track_court_sides: Track -> 0 (near) or 1 (far).
            use_side_penalty: Whether to add side penalty to cost matrix.
            early_positions: Early-rally positions per track for continuity.

        Returns:
            track_id -> player_id mapping.
        """
        if not track_ids:
            return {}

        all_player_ids = sorted(self.state.players.keys())  # [1, 2, 3, 4]
        n_tracks = len(track_ids)
        n_players = len(all_player_ids)
        size = max(n_tracks, n_players)

        # Check if position continuity is available
        has_positions = (
            early_positions
            and self.state.player_last_positions
            and use_side_penalty  # Only for final assignment, not side-switch detection
        )

        # Build cost matrix: appearance + optional side penalty + position
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

                # Position continuity cost (normalized distance)
                # Only apply within same team — position shouldn't pull
                # tracks cross-team, only disambiguate within-team
                pos_cost = 0.0
                player_side = self.state.current_side_assignment.get(pid)
                if (
                    has_positions
                    and track_side == player_side  # same team only
                    and tid in early_positions  # type: ignore[operator]
                    and pid in self.state.player_last_positions
                ):
                    d = _dist(
                        early_positions[tid],  # type: ignore[index]
                        self.state.player_last_positions[pid],
                    )
                    # Normalize: 0.3 distance ≈ half the court, cap at 1.0
                    pos_cost = min(d / 0.3, 1.0)

                # Side penalty (player_side already computed above)
                if use_side_penalty:
                    side_pen = SIDE_PENALTY if track_side != player_side else 0.0
                else:
                    side_pen = 0.0

                # Blend costs: appearance (1 - POSITION_WEIGHT) + position
                if has_positions and pos_cost > 0:
                    blended = (
                        appearance_cost * (1.0 - POSITION_WEIGHT)
                        + pos_cost * POSITION_WEIGHT
                        + side_pen
                    )
                else:
                    blended = appearance_cost + side_pen

                cost_matrix[i, j] = blended

        # Store appearance-only cost matrix for diagnostics (before side penalty)
        if self.collect_diagnostics and use_side_penalty:
            appearance_only = np.full((n_tracks, n_players), 1.0)
            for i, tid in enumerate(track_ids):
                if tid not in track_stats:
                    continue
                for j, pid in enumerate(all_player_ids):
                    if pid not in self.state.players:
                        continue
                    appearance_only[i, j] = compute_appearance_similarity(
                        self.state.players[pid], track_stats[tid]
                    )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        result: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < n_tracks and c < n_players:
                result[track_ids[r]] = all_player_ids[c]

        # Collect diagnostics after final (penalized) assignment
        if self.collect_diagnostics and use_side_penalty:
            margins: dict[int, float] = {}
            for j, pid in enumerate(all_player_ids):
                col_costs = sorted(appearance_only[:, j]) if n_tracks > 0 else []
                if len(col_costs) >= 2:
                    margins[pid] = float(col_costs[1] - col_costs[0])
                elif len(col_costs) == 1:
                    margins[pid] = float(1.0 - col_costs[0])
            self.diagnostics.append(RallyAssignmentDiagnostics(
                rally_index=self.rally_count - 1,
                cost_matrix=appearance_only,
                track_ids=list(track_ids),
                player_ids=list(all_player_ids),
                track_court_sides=dict(track_court_sides),
                assignment=dict(result),
                assignment_margins=margins,
            ))

        return result

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

    def _detect_side_switches_combinatorial(self) -> list[int]:
        """Detect side switches via combinatorial search.

        Generates candidate switch points from two sources:
        1. Ball trajectory direction (consecutive same-direction serves)
        2. Appearance preference sign change (adjacent rallies prefer
           opposite orientation relative to their neighbors)

        Tries all 2^K combinations of candidates, scoring each partition
        using normalized pairwise team appearance preferences.

        Returns:
            List of rally indices where side switches should be applied.
            Empty if no switches detected or no improvement found.
        """
        n = len(self.stored_rally_data)
        if n < 3:
            return []

        # Step 1: Build pairwise team preference matrix (needed for both
        # candidate generation and scoring).
        # For each pair of rallies, compute preference for "same orientation"
        # vs "opposite orientation" using raw track-to-track comparison.
        # Preference > 0 means same orientation preferred.
        preference = np.zeros((n, n))

        for i in range(n):
            di = self.stored_rally_data[i]
            t0_i = [
                tid for tid in di.top_tracks
                if di.track_court_sides.get(tid) == 0
                and tid in di.track_stats
            ]
            t1_i = [
                tid for tid in di.top_tracks
                if di.track_court_sides.get(tid) == 1
                and tid in di.track_stats
            ]
            if not t0_i or not t1_i:
                continue

            for j in range(i + 1, n):
                dj = self.stored_rally_data[j]
                t0_j = [
                    tid for tid in dj.top_tracks
                    if dj.track_court_sides.get(tid) == 0
                    and tid in dj.track_stats
                ]
                t1_j = [
                    tid for tid in dj.top_tracks
                    if dj.track_court_sides.get(tid) == 1
                    and tid in dj.track_stats
                ]
                if not t0_j or not t1_j:
                    continue

                # Same orientation: near↔near + far↔far
                same_cost = (
                    _team_match_cost(t0_i, di.track_stats, t0_j, dj.track_stats)
                    + _team_match_cost(t1_i, di.track_stats, t1_j, dj.track_stats)
                )
                # Cross orientation: near↔far + far↔near
                cross_cost = (
                    _team_match_cost(t0_i, di.track_stats, t1_j, dj.track_stats)
                    + _team_match_cost(t1_i, di.track_stats, t0_j, dj.track_stats)
                )
                pref = cross_cost - same_cost
                preference[i, j] = pref
                preference[j, i] = pref

        # Step 2: Normalize preferences to remove perspective baseline.
        # Raw preferences are ALL positive because perspective dominates
        # (near-side tracks always match better with near-side).
        # Subtracting row/column means exposes the real signal:
        # same-orientation pairs become positive, cross-orientation negative.
        row_means = np.zeros(n)
        row_counts = np.zeros(n)
        for a in range(n):
            for b in range(n):
                if a != b and preference[a, b] != 0.0:
                    row_means[a] += preference[a, b]
                    row_counts[a] += 1
        for a in range(n):
            if row_counts[a] > 0:
                row_means[a] /= row_counts[a]

        norm_pref = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                if preference[a, b] == 0.0:
                    continue
                val = preference[a, b] - (row_means[a] + row_means[b]) / 2.0
                norm_pref[a, b] = val
                norm_pref[b, a] = val

        # Step 3: Generate candidate switch points from two sources.
        # Source A: Ball trajectory — consecutive same-direction serves.
        serve_dirs = [d.serve_direction for d in self.stored_rally_data]
        ball_candidates: list[int] = []
        prev_dir = None
        for i, d in enumerate(serve_dirs):
            if d != "?" and prev_dir is not None and prev_dir != "?":
                if d == prev_dir:
                    ball_candidates.append(i)
            if d != "?":
                prev_dir = d

        # Source B: Appearance — adjacent rallies where the average
        # normalized preference with earlier rallies flips sign compared
        # to the previous rally. This catches switches invisible to ball
        # direction (e.g., alternating serves across a switch point).
        appearance_candidates: list[int] = []
        if n >= 4:
            # For each rally, compute average norm_pref with all prior rallies
            avg_pref_with_prior = np.zeros(n)
            for k in range(1, n):
                vals = [norm_pref[k, j] for j in range(k) if norm_pref[k, j] != 0.0]
                if vals:
                    avg_pref_with_prior[k] = float(np.mean(vals))

            # Rally 1 special case: if strongly negative with rally 0,
            # it's a candidate for early switch (loop below starts at k=2).
            if avg_pref_with_prior[1] < -0.01:
                appearance_candidates.append(1)

            for k in range(2, n):
                prev_val = avg_pref_with_prior[k - 1]
                curr_val = avg_pref_with_prior[k]
                # Sign flip with sufficient magnitude on both sides
                if (prev_val > 0.01 and curr_val < -0.01) or (
                    prev_val < -0.01 and curr_val > 0.01
                ):
                    appearance_candidates.append(k)

        # Merge and deduplicate
        candidates = sorted(set(ball_candidates) | set(appearance_candidates))

        if not candidates:
            logger.info("Side switch search: no candidates")
            return []

        # Cap at 6 candidates (2^6 = 64 combinations max)
        if len(candidates) > 6:
            logger.info(
                "Side switch search: %d candidates, capping at 6",
                len(candidates),
            )
            candidates = candidates[:6]

        logger.info(
            "Side switch search: %d candidates at %s (ball=%s, appearance=%s)",
            len(candidates), candidates, ball_candidates, appearance_candidates,
        )

        # Step 4: Score partition using normalized preferences.
        # Each switch incurs a penalty (parsimony: prefer fewer switches).
        # Sweep (23 videos): 1.0 = best accuracy (82.9%), 1.4-1.5 = best
        # switch F1 (68.6%). 1.0 catches short-match switches (vuvu, vivi)
        # with acceptable FP rate (8 FPs vs 4 at 1.5).
        switch_penalty = globals().get("_SWITCH_PENALTY_OVERRIDE") or 1.0

        def score_partition(switch_set: set[int]) -> float:
            """Score a partition defined by switch points."""
            orientation = np.zeros(n, dtype=int)
            flipped = False
            for k in range(n):
                if k in switch_set:
                    flipped = not flipped
                orientation[k] = 1 if flipped else 0

            total = 0.0
            for a in range(n):
                for b in range(a + 1, n):
                    if norm_pref[a, b] == 0.0:
                        continue
                    if orientation[a] == orientation[b]:
                        total += norm_pref[a, b]
                    else:
                        total -= norm_pref[a, b]

            total -= len(switch_set) * switch_penalty
            return total

        baseline_score = score_partition(set())

        best_score = baseline_score
        best_switches: list[int] = []
        n_combos = 1 << len(candidates)

        for mask in range(1, n_combos):
            switch_set = {
                candidates[j]
                for j in range(len(candidates))
                if mask & (1 << j)
            }
            score = score_partition(switch_set)
            if score > best_score:
                best_score = score
                best_switches = sorted(switch_set)

        if best_switches:
            improvement = best_score - baseline_score
            logger.info(
                "Side switch search: switches at %s "
                "(score %.3f → %.3f, +%.3f)",
                best_switches, baseline_score, best_score, improvement,
            )
        else:
            logger.info("Side switch search: no switches (baseline is best)")

        return best_switches

    def refine_assignments(
        self,
        initial_results: list[RallyTrackingResult],
    ) -> list[RallyTrackingResult]:
        """Re-score all rallies using final profiles + global within-team voting.

        Three-stage Pass 2:
        0. Combinatorial side switch detection using ball trajectory direction
        1. Re-run cross-team assignment with final profiles
        2. Global within-team pairwise voting

        Args:
            initial_results: Results from Pass 1 forward pass.

        Returns:
            Refined results with potentially corrected assignments.
        """
        if len(self.stored_rally_data) != len(initial_results):
            logger.warning(
                "stored_rally_data length mismatch: %d vs %d results",
                len(self.stored_rally_data),
                len(initial_results),
            )
            return initial_results

        if len(initial_results) <= 1:
            return initial_results

        # Stage 0: Detect side switches and update stored side assignments
        switches = self._detect_side_switches_combinatorial()
        switch_set = set(switches)
        if switches:
            flipped = False
            for i, data in enumerate(self.stored_rally_data):
                if i in switch_set:
                    flipped = not flipped
                if flipped:
                    data.player_side_assignment = {
                        pid: (1 - team)
                        for pid, team in data.player_side_assignment.items()
                    }
            # Mark switch results
            for i in switches:
                if i < len(initial_results):
                    r = initial_results[i]
                    initial_results[i] = RallyTrackingResult(
                        rally_index=r.rally_index,
                        track_to_player=r.track_to_player,
                        server_player_id=r.server_player_id,
                        side_switch_detected=True,
                        assignment_confidence=r.assignment_confidence,
                    )

        # Stage 1: Re-score ALL rallies (including rally 0) with final profiles.
        # Rally 0 was initialized by Y-sort only; re-scoring with accumulated
        # profiles can fix cascade errors where the first rally was wrong.
        refined: list[RallyTrackingResult] = []
        changes = 0

        for i, (data, initial) in enumerate(
            zip(self.stored_rally_data, initial_results)
        ):
            # Restore the player→side mapping from Pass 1 so side penalties
            # are correct for pre-switch rallies.
            saved_side = self.state.current_side_assignment
            self.state.current_side_assignment = data.player_side_assignment

            # No position continuity in Pass 2 — rebuilding the position
            # chain from scratch can propagate errors.
            track_to_player = self._assign_tracks_to_players_global(
                data.top_tracks,
                data.track_stats,
                data.track_court_sides,
            )

            self.state.current_side_assignment = saved_side

            confidence = self._compute_assignment_confidence(
                data.track_stats, track_to_player
            )

            if track_to_player != initial.track_to_player:
                changes += 1

            refined.append(RallyTrackingResult(
                rally_index=initial.rally_index,
                track_to_player=track_to_player,
                server_player_id=initial.server_player_id,
                side_switch_detected=initial.side_switch_detected,
                assignment_confidence=confidence,
            ))

        if changes:
            logger.info("Pass 2 stage 1 changed %d/%d rallies", changes, len(refined))

        # Stage 2: Global within-team voting using raw track comparisons
        refined = self._global_within_team_voting(refined)

        return refined

    def _global_within_team_voting(
        self,
        results: list[RallyTrackingResult],
    ) -> list[RallyTrackingResult]:
        """Fix within-team assignments using global pairwise voting.

        For each team, collects all rally track pairs and computes pairwise
        "same vs swap" preferences using direct track-to-track comparison
        (no accumulated profiles). Finds the globally consistent labeling
        that maximizes agreement across all rally pairs.

        This avoids the profile corruption cascade: even if Pass 1 got
        rally 3 wrong, the raw track features are clean and can vote
        correctly for the global ordering.
        """
        if len(results) < 3:
            return results

        swaps = 0
        for team in [0, 1]:
            team_player_ids = sorted(
                pid for pid, t in self.state.current_side_assignment.items()
                if t == team
            )
            if len(team_player_ids) != 2:
                continue

            p_lo, p_hi = team_player_ids  # e.g., (1, 2) or (3, 4)

            # Collect per-rally track pairs for this team
            # Each entry: (rally_index, track_for_p_lo, track_for_p_hi)
            rally_pairs: list[tuple[int, int, int]] = []
            for i, (data, result) in enumerate(
                zip(self.stored_rally_data, results)
            ):
                # Find the two tracks assigned to this team's players
                t_lo = None
                t_hi = None
                for tid, pid in result.track_to_player.items():
                    if pid == p_lo:
                        t_lo = tid
                    elif pid == p_hi:
                        t_hi = tid

                if t_lo is not None and t_hi is not None:
                    # Verify both have stats
                    if t_lo in data.track_stats and t_hi in data.track_stats:
                        rally_pairs.append((i, t_lo, t_hi))

            if len(rally_pairs) < 3:
                continue

            # Build pairwise preference matrix
            # preference[i][j] > 0 means rallies i and j prefer same ordering
            n = len(rally_pairs)
            preference = np.zeros((n, n))

            for a in range(n):
                ri_a, t_lo_a, t_hi_a = rally_pairs[a]
                stats_lo_a = self.stored_rally_data[ri_a].track_stats[t_lo_a]
                stats_hi_a = self.stored_rally_data[ri_a].track_stats[t_hi_a]

                for b in range(a + 1, n):
                    ri_b, t_lo_b, t_hi_b = rally_pairs[b]
                    stats_lo_b = self.stored_rally_data[ri_b].track_stats[t_lo_b]
                    stats_hi_b = self.stored_rally_data[ri_b].track_stats[t_hi_b]

                    # Cost of "same ordering" (lo↔lo, hi↔hi)
                    same_cost = (
                        compute_track_similarity(stats_lo_a, stats_lo_b)
                        + compute_track_similarity(stats_hi_a, stats_hi_b)
                    )
                    # Cost of "swapped ordering" (lo↔hi, hi↔lo)
                    swap_cost = (
                        compute_track_similarity(stats_lo_a, stats_hi_b)
                        + compute_track_similarity(stats_hi_a, stats_lo_b)
                    )

                    # Positive = same ordering preferred
                    pref = swap_cost - same_cost
                    preference[a, b] = pref
                    preference[b, a] = pref

            # Iterative labeling: all rallies vote against all others.
            # Each rally's label (0=keep, 1=swap) converges to a globally
            # consistent binary partition. The orientation check below
            # resolves the global flip ambiguity using profiles.
            labels = np.zeros(n, dtype=int)  # 0 = same as ref, 1 = swapped
            for _iteration in range(10):
                changed = False
                for k in range(n):
                    # Sum weighted preferences: positive = vote for "same
                    # label as j", negative = vote for "different label".
                    # Flip sign when j is swapped (label=1) since preference
                    # was computed relative to the original ordering.
                    score = 0.0
                    for j in range(n):
                        if j == k:
                            continue
                        p = preference[k, j]
                        if labels[j] == 1:
                            p = -p
                        score += p

                    new_label = 0 if score >= 0 else 1
                    if new_label != labels[k]:
                        labels[k] = new_label
                        changed = True

                if not changed:
                    break

            # Check both orientations against accumulated profiles.
            # Voting finds internally consistent labeling but can't
            # determine which global orientation is correct. Use profiles
            # (from stage 1) to pick the better orientation.
            cost_current = 0.0
            cost_flipped = 0.0
            for idx in range(n):
                ri, t_lo, t_hi = rally_pairs[idx]
                data = self.stored_rally_data[ri]
                # "current" orientation: label=0 → t_lo→p_lo, t_hi→p_hi
                #                        label=1 → t_lo→p_hi, t_hi→p_lo
                if labels[idx] == 0:
                    c_lo, c_hi = t_lo, t_hi
                else:
                    c_lo, c_hi = t_hi, t_lo  # swapped

                if p_lo in self.state.players and p_hi in self.state.players:
                    cost_current += (
                        compute_appearance_similarity(
                            self.state.players[p_lo], data.track_stats[c_lo]
                        )
                        + compute_appearance_similarity(
                            self.state.players[p_hi], data.track_stats[c_hi]
                        )
                    )
                    cost_flipped += (
                        compute_appearance_similarity(
                            self.state.players[p_hi], data.track_stats[c_lo]
                        )
                        + compute_appearance_similarity(
                            self.state.players[p_lo], data.track_stats[c_hi]
                        )
                    )

            # If flipped orientation is better, flip all labels
            if cost_flipped < cost_current:
                labels = 1 - labels
                logger.info(
                    "Within-team vote: team %d flipped orientation "
                    "(cost %.3f → %.3f)",
                    team, cost_current, cost_flipped,
                )

            # Apply swaps where label=1
            for idx in range(n):
                if labels[idx] == 1:
                    ri, t_lo, t_hi = rally_pairs[idx]
                    result = results[ri]
                    new_t2p = dict(result.track_to_player)
                    new_t2p[t_lo] = p_hi
                    new_t2p[t_hi] = p_lo
                    results[ri] = RallyTrackingResult(
                        rally_index=result.rally_index,
                        track_to_player=new_t2p,
                        server_player_id=result.server_player_id,
                        side_switch_detected=result.side_switch_detected,
                        assignment_confidence=result.assignment_confidence,
                    )
                    swaps += 1
                    logger.info(
                        "Within-team vote: rally %d team %d swapped "
                        "(tracks %d↔%d for players %d↔%d)",
                        ri, team, t_lo, t_hi, p_lo, p_hi,
                    )

        if swaps:
            logger.info(
                "Global within-team voting: %d swaps across %d rallies",
                swaps, len(results),
            )
        else:
            logger.info("Global within-team voting: no swaps")

        return results

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

            frame_arr = np.asarray(frame, dtype=np.uint8)

            for tid, p in frame_requests[fn]:
                bbox = (p.x, p.y, p.width, p.height)
                features = extract_appearance_features(
                    frame_arr, tid, fn, bbox, frame_width, frame_height,
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
    diagnostics: list[RallyAssignmentDiagnostics] = field(default_factory=list)


def match_players_across_rallies(
    video_path: Path,
    rallies: list[RallyTrackData],
    num_samples: int = 12,
    collect_diagnostics: bool = False,
) -> MatchPlayersResult:
    """
    Match players across all rallies in a video for consistent IDs.

    Creates a MatchPlayerTracker and processes rallies chronologically,
    extracting appearances from video and assigning consistent player IDs 1-4.

    Args:
        video_path: Path to the video file.
        rallies: Rally data sorted chronologically.
        num_samples: Frames to sample per track for appearance.
        collect_diagnostics: If True, collect per-rally cost matrices
            and assignment margins for diagnostic analysis.

    Returns:
        MatchPlayersResult with track→player mappings and accumulated profiles.
    """
    tracker = MatchPlayerTracker(collect_diagnostics=collect_diagnostics)
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

    # Pass 2: Re-score all rallies with final profiles
    results = tracker.refine_assignments(results)

    return MatchPlayersResult(
        rally_results=results,
        player_profiles=dict(tracker.state.players),
        diagnostics=tracker.diagnostics,
    )
