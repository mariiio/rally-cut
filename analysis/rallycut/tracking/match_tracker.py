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
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

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
    extract_bbox_crop,
)
from rallycut.tracking.team_identity import (
    TeamTemplate,
    build_team_templates,
)

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition
    from rallycut.tracking.reid_general import GeneralReIDModel

logger = logging.getLogger(__name__)

# Side penalty for global Hungarian assignment. Biases toward expected court side
# but doesn't prevent cross-side matching when appearance is a stronger signal.
# Appearance costs range 0.0-1.0, so 0.15 is meaningful but not dominant.
SIDE_PENALTY = 0.15

# Hard side penalty when court calibration provides authoritative side labels.
# Effectively prevents cross-side assignment — players physically cannot be on
# the wrong side of the net.
SIDE_PENALTY_CALIBRATED = 1.0

# Position continuity weight in the cost matrix. When last positions are known,
# the distance between early-rally track position and previous rally's late position
# is blended into the cost. This is critical for within-team discrimination where
# appearance discriminability is poor (typical cost gap 0.02-0.05 between teammates).
POSITION_WEIGHT = 0.30

# Minimum assignment confidence to update profiles. Below this threshold,
# profile updates are skipped to prevent error propagation (drift).
# Tuned via grid search on 28 GT videos: 0.80 → 85.8% vs 0.55 → 84.4%.
MIN_PROFILE_UPDATE_CONFIDENCE = 0.80

# Weight for blending ReID cosine distance with HSV appearance cost.
# When ReID embeddings are available on both profile and track:
#   blended = reid_cost * REID_BLEND + hsv_cost * (1 - REID_BLEND)
# When unavailable or below confidence gate, falls back to HSV-only.
REID_BLEND = 0.50

# Minimum margin (second_best - best ReID cost) to trust the ReID signal
# for a given track. When the margin is small, all players look similar
# to the track → ReID isn't discriminative → fall back to HSV only.
# Protects against regressions from unrepresentative reference crops.
REID_MIN_MARGIN = 0.08

# Frame cutoff used to restrict first-rally side classification to the
# serve formation window.  During the serve the 2v2 arrangement is
# formal (each team on their own side); mid-rally players may cross into
# the opponent's half and contaminate whole-rally Y averages.  Because
# the first rally's assignment seeds the persistent team templates for
# the entire match, a single init error propagates to every subsequent
# rally.  Matches the window used by ``_find_serving_side_by_formation``
# (120 frames) and is strictly wider than ``detect_serve_anchor``'s
# 30-frame serve-detection window — ensures we still have enough
# positions to compute robust averages.
FIRST_RALLY_INIT_WINDOW_FRAMES = 120

# Phase 3 — match_tracker global-seed patch.
# When enabled, before per-rally Hungarian we pool per-(rally, tid) mean HSV
# features across ALL rallies, run k-means k=4 globally, assign clusters to
# player ids by (majority court side, median Y), and seed every player profile
# by feeding ALL tracks in its cluster through update_from_features. This
# replaces the first-rally Y-sort seed (which has zero robustness when the
# first rally is short/noisy/occluded) with a match-global prototype.
# Pre-registered gates on 8-fixture click-GT: see docs/superpowers/plans.
GLOBAL_SEED_ENABLED = os.environ.get("MATCH_TRACKER_GLOBAL_SEED", "0") == "1"

# Cluster centroid pairwise cosine threshold above which the seed is rejected
# — two clusters have collapsed and seeding would produce duplicate pids.
GLOBAL_SEED_MAX_CENTROID_COS = 0.85


def _cluster_feature(stats: TrackAppearanceStats) -> np.ndarray | None:
    """Build a fixed-length clustering feature from track stats.

    Concatenates upper-body HS + lower-body HS histograms (both L1-normalized
    by construction). The 256-dim vector is then L2-normalized so Euclidean
    k-means behaves like cosine clustering — matches probe 2's approach.

    Returns None when either histogram is missing (skip track from seeding).
    """
    if stats.avg_upper_hist is None or stats.avg_lower_hist is None:
        return None
    upper = stats.avg_upper_hist.astype(np.float32).flatten()
    lower = stats.avg_lower_hist.astype(np.float32).flatten()
    feat = np.concatenate([upper, lower])
    norm = float(np.linalg.norm(feat))
    if norm <= 1e-9:
        return None
    return feat / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _teams_from_positions(
    track_to_player: dict[str, Any],
    positions: list[PlayerPosition],
) -> dict[int, int]:
    """Assign team 0 (near) vs team 1 (far) based on rally-start foot-y
    positions. The 2 tids with highest median foot-y at rally start are "near";
    the other 2 are "far".

    Rally-start positions (first 15 frames) are reliable because teams are
    strictly on their own sides by volleyball rule before the serve; later
    frames see defenders/setters crossing the midline, making per-frame
    grouping noisy.

    Returns {track_id: team (0|1)} only when all 4 primary tids have enough
    rally-start samples to cluster confidently. Returns {} otherwise (caller
    falls back to legacy pairing).
    """
    if not track_to_player or not positions:
        return {}

    primary_tids = {int(t) for t in track_to_player.keys()}
    per_tid_ys: dict[int, list[float]] = {tid: [] for tid in primary_tids}
    for p in positions:
        if p.track_id in primary_tids and p.frame_number < 15:
            foot_y = p.y + (p.height or 0) / 2
            per_tid_ys[p.track_id].append(foot_y)
    if any(len(ys) < 3 for ys in per_tid_ys.values()):
        # Fall back to all-frame median if rally-start coverage is thin
        per_tid_ys = {tid: [] for tid in primary_tids}
        for p in positions:
            if p.track_id in primary_tids:
                foot_y = p.y + (p.height or 0) / 2
                per_tid_ys[p.track_id].append(foot_y)
        if any(not ys for ys in per_tid_ys.values()):
            return {}

    import statistics
    medians = {tid: statistics.median(ys) for tid, ys in per_tid_ys.items()}
    if len(medians) < 4:
        return {}
    ranked = sorted(medians.keys(), key=lambda t: -medians[t])
    # Require a minimum gap between the 2nd and 3rd median-y to trust the
    # positional split; below this, both groups straddle the midline and
    # grouping is noise.
    if medians[ranked[1]] - medians[ranked[2]] < 0.03:
        return {}
    near_tids = set(ranked[:2])
    return {tid: (0 if tid in near_tids else 1) for tid in primary_tids}


def verify_team_assignments(
    team_assignments: dict[int, int],
    player_positions: list[PlayerPosition],
    min_gap: float = 0.02,
) -> dict[int, int]:
    """Verify team assignments match actual player positions; flip if inverted.

    Checks whether team 0 tracks have higher avg Y (near court) than team 1.
    If inverted (team 0 is on far side), flips all labels.

    Args:
        team_assignments: track_id -> team (0=near, 1=far).
        player_positions: Player positions for this rally.
        min_gap: Minimum avg Y gap between teams to trigger a flip.
            Below this threshold, the assignment is ambiguous.

    Returns:
        Corrected team assignments (flipped if inverted, unchanged otherwise).
    """
    if not team_assignments or not player_positions:
        return team_assignments

    team_ys: dict[int, list[float]] = {0: [], 1: []}
    for p in player_positions:
        team = team_assignments.get(p.track_id)
        if team is not None:
            team_ys[team].append(p.y)

    if not team_ys[0] or not team_ys[1]:
        return team_assignments

    avg_0 = sum(team_ys[0]) / len(team_ys[0])
    avg_1 = sum(team_ys[1]) / len(team_ys[1])

    # Team 0 should have higher Y (near court, closer to camera).
    # If team 1 has clearly higher Y, the assignment is inverted.
    if avg_1 - avg_0 > min_gap:
        return {tid: 1 - team for tid, team in team_assignments.items()}

    return team_assignments


def build_match_team_assignments(
    match_analysis: dict[str, Any],
    min_confidence: float = 0.0,
    rally_positions: dict[str, list[PlayerPosition]] | None = None,
) -> dict[str, dict[int, int]]:
    """Build per-rally team assignments from match analysis JSON.

    Derives team labels (0=near, 1=far) for each track in each rally,
    accounting for cumulative side switches across rallies.

    Convention: player IDs 1-2 = team 0 (near), 3-4 = team 1 (far)
    at baseline. Each side switch flips the mapping.

    When rally_positions is provided, each rally's team labels are verified
    against actual player Y positions and flipped if inverted (team 0 on
    far side). This corrects errors from wrong initial assignment or missed
    side switches.

    Accepts both camelCase (from API JSON) and snake_case (from Python)
    field names (e.g. sideSwitchDetected / side_switch_detected).

    Args:
        match_analysis: The match_analysis_json from the videos table.
        min_confidence: Skip rallies below this assignment confidence.
        rally_positions: Optional rally_id -> player positions for verification.

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

        # Primary: derive team from physical positions (which 2 tids are
        # physically on near side vs far side during this rally). Falls back
        # to the legacy pid-ordering assumption only when positions are
        # unavailable. The legacy "pids 1,2 = team 0, 3,4 = team 1" pairing
        # is a systemic bug when match-players doesn't cluster by team
        # (observed on 74% of rallies in the 2026-04-24 primitive audit).
        teams: dict[int, int] = {}
        positions_for_rally = (
            rally_positions.get(rid) if rally_positions else None
        )
        if positions_for_rally:
            teams = _teams_from_positions(track_to_player, positions_for_rally)

        if not teams:
            # Legacy fallback: pid-ordering pairing. Broken for rallies where
            # pids aren't assigned to physical teammates — carried only for
            # backward compat when positions aren't available.
            for tid_str, player_id in track_to_player.items():
                pid = int(player_id)
                base_team = 0 if pid <= 2 else 1
                team = base_team if side_switch_count % 2 == 0 else 1 - base_team
                teams[int(tid_str)] = team

        if teams:
            result[rid] = teams

    # Verify against actual positions when available
    if rally_positions:
        for rid, teams in result.items():
            positions = rally_positions.get(rid)
            if positions:
                result[rid] = verify_team_assignments(teams, positions)

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
    # Rally timing (for inter-rally gap calculation in switch detection)
    start_ms: int = 0
    end_ms: int = 0
    # Whether side classification used court calibration (authoritative)
    sides_from_calibration: bool = False


def _team_match_cost(
    tids_a: list[int],
    stats_a: dict[int, TrackAppearanceStats],
    tids_b: list[int],
    stats_b: dict[int, TrackAppearanceStats],
) -> float:
    """Best-matching cost between two sets of tracks."""
    if not tids_a or not tids_b:
        return 1.0
    # Pass ReID blend weight through to track similarity. The margin gate
    # is applied inside compute_track_similarity via the shape check — if
    # embeddings are incompatible it falls back to HSV. For side-switch
    # detection we use a conservative blend: only if BOTH tracks in a pair
    # have embeddings (not just any).
    def _both_have_emb(
        sa: dict[int, TrackAppearanceStats], ta: int,
        sb: dict[int, TrackAppearanceStats], tb: int,
    ) -> bool:
        tsa = sa.get(ta)
        tsb = sb.get(tb)
        return (
            tsa is not None and tsa.reid_embedding is not None
            and tsb is not None and tsb.reid_embedding is not None
        )

    # Only blend if at least one pair has both embeddings
    rb = REID_BLEND if any(
        _both_have_emb(stats_a, tids_a[i], stats_b, tids_b[j])
        for i in range(min(2, len(tids_a)))
        for j in range(min(2, len(tids_b)))
    ) else 0.0
    if len(tids_a) == 1 and len(tids_b) == 1:
        return compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]], rb)
    if len(tids_a) >= 2 and len(tids_b) >= 2:
        cost_ab = (
            compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]], rb)
            + compute_track_similarity(stats_a[tids_a[1]], stats_b[tids_b[1]], rb)
        )
        cost_ba = (
            compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[1]], rb)
            + compute_track_similarity(stats_a[tids_a[1]], stats_b[tids_b[0]], rb)
        )
        return min(cost_ab, cost_ba) / 2.0
    return compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]], rb)


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
        reference_profiles: dict[int, PlayerAppearanceProfile] | None = None,
    ):
        """
        Initialize match tracker.

        Args:
            calibrator: Optional court calibrator for baseline detection.
            collect_diagnostics: If True, collect per-rally cost matrices
                and assignment margins for diagnostic analysis.
            reference_profiles: Optional user-provided frozen profiles (player_id -> profile).
                When provided, profiles are never updated — they anchor all assignments.
        """
        self.calibrator = calibrator
        self.state = MatchPlayerState()
        self.state.initialize_players()
        self.frozen_player_ids: set[int] = set()
        if reference_profiles:
            for pid, profile in reference_profiles.items():
                self.state.players[pid] = profile
                self.frozen_player_ids.add(pid)
        self.rally_count = 0
        self.collect_diagnostics = collect_diagnostics
        self.diagnostics: list[RallyAssignmentDiagnostics] = []
        self.stored_rally_data: list[StoredRallyData] = []
        self._sides_from_calibration = False
        # Phase 3 — set True after global_seed_from_rallies succeeds.
        # Makes the first rally use the global Hungarian path (same as rallies
        # 2+) rather than the Y-sort _initialize_first_rally heuristic.
        self._global_seeded = False

    def process_rally(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        player_positions: list[PlayerPosition],
        ball_positions: list[BallPosition] | None = None,
        court_split_y: float | None = None,
        team_assignments: dict[int, int] | None = None,
        start_ms: int = 0,
        end_ms: int = 0,
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
        #
        # For the first rally, restrict classification to the serve
        # formation window.  The serve moment has formal 2v2 arrangement
        # by volleyball rules; mid-rally, players cross sides and
        # contaminate whole-rally Y averages.  Because the first rally's
        # assignment seeds every subsequent rally's templates, an init
        # error here propagates to the whole match — use the cleanest
        # signal available.
        classification_positions = player_positions
        if self.rally_count == 1 and not self.frozen_player_ids:
            windowed = [
                p for p in player_positions
                if p.frame_number < FIRST_RALLY_INIT_WINDOW_FRAMES
            ]
            # Fall back to full rally if the serve window is sparse
            # (short rally, delayed detection, etc.) — we'd rather use
            # noisy full-rally data than no data.
            if len(windowed) >= 4 * 10:  # ≥10 frames per player
                classification_positions = windowed

        track_avg_y, track_court_sides = self._classify_track_sides(
            track_stats, classification_positions, court_split_y, team_assignments
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

        if self.rally_count <= 1 and not self.frozen_player_ids:
            # Phase 3: seeded profiles enrich the per-match prior, but the
            # first-rally track->pid mapping still goes through the baseline
            # Y-sort so convention remains identical. The seed's benefit
            # is stabilizing rallies 2+ via richer HSV/ReID profiles.
            track_to_player = self._initialize_first_rally(
                top_tracks, track_avg_y, track_court_sides
            )
        else:
            track_to_player = self._assign_tracks_to_players_global(
                top_tracks, track_stats, track_court_sides,
                use_side_penalty=not self.frozen_player_ids,
                early_positions=early_positions,
            )

        # Step 5: Within-team refinement
        if self.rally_count > 1 or self.frozen_player_ids:
            track_to_player = self._refine_within_team(
                track_to_player, player_positions, track_court_sides
            )

        # Store late-rally positions for next rally's continuity check
        self._store_last_positions(track_to_player, player_positions)

        # Step 6: Compute confidence BEFORE updating profiles
        confidence = self._compute_assignment_confidence(track_stats, track_to_player)

        # Step 7: Update player profiles (gated on confidence)
        # Skip frozen (user-provided) profiles, update the rest normally.
        # When globally seeded, the first rally is treated like any other —
        # use the confidence gate instead of the unconditional first-rally
        # update so seed profiles aren't clobbered by a bad first rally.
        if self.rally_count <= 1 and not self._global_seeded:
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
            start_ms=start_ms,
            end_ms=end_ms,
            sides_from_calibration=self._sides_from_calibration,
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

        Priority: court calibration (homography) > team_assignments
        (from tracking pipeline's bbox-size clustering) > court_split_y
        > median Y split.

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
        self._sides_from_calibration = False

        # Priority 0: Court calibration — project foot positions through
        # homography to get court_y in meters.  Net is at 8.0 m.
        # This is authoritative when available.
        if self.calibrator is not None and self.calibrator.is_calibrated:
            track_court_y: dict[int, list[float]] = {}
            for p in player_positions:
                if p.track_id < 0 or p.track_id not in track_avg_y:
                    continue
                foot_x = p.x
                # p.y is bbox center; + height/2 = bottom edge (feet)
                foot_y = p.y + p.height * 0.5
                try:
                    _, cy = self.calibrator.image_to_court(
                        (foot_x, foot_y), 1, 1
                    )
                    if p.track_id not in track_court_y:
                        track_court_y[p.track_id] = []
                    track_court_y[p.track_id].append(cy)
                except Exception:
                    pass

            if track_court_y:
                classified = 0
                for tid in track_avg_y:
                    if tid in track_court_y and track_court_y[tid]:
                        med_cy = float(np.median(track_court_y[tid]))
                        # Net at 8.0 m: near side < 8.0, far side > 8.0
                        track_court_sides[tid] = 0 if med_cy < 8.0 else 1
                        classified += 1

                if classified >= len(track_avg_y) * 0.75:
                    # Fill unclassified tracks using image-space Y position:
                    # higher Y = near court (closer to camera)
                    sorted_unclassified = sorted(
                        (tid for tid in track_avg_y if tid not in track_court_sides),
                        key=lambda t: track_avg_y[t],
                    )
                    mid = len(sorted_unclassified) // 2
                    for idx, tid in enumerate(sorted_unclassified):
                        track_court_sides[tid] = 1 if idx < mid else 0
                    self._sides_from_calibration = True
                    logger.info(
                        "Court calibration classified %d/%d tracks",
                        classified,
                        len(track_avg_y),
                    )
                    return track_avg_y, track_court_sides
                # Not enough tracks classified — fall through
                track_court_sides.clear()

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

    def global_seed_from_rallies(
        self,
        rally_inputs: list[tuple[
            dict[int, TrackAppearanceStats],  # track_stats
            list[PlayerPosition],             # positions
            float | None,                     # court_split_y
            dict[int, int] | None,            # team_assignments
        ]],
        k: int = 4,
    ) -> dict[str, Any]:
        """Phase 3 — seed player profiles from a global k-means over all rallies.

        Pools per-(rally, tid) HSV features across the entire match, runs
        k-means k=4, and seeds ``state.players[pid]`` by feeding every track
        assigned to its cluster through ``update_from_features``. Replaces the
        first-rally Y-sort heuristic — the historical failure mode is that a
        noisy first rally produces duplicate / swapped seed profiles that
        propagate for the whole match.

        The cluster -> player_id mapping uses (majority court side, median Y)
        so that player ids 1-2 land on the near-side clusters and 3-4 on the
        far-side clusters (near = higher image Y, closer to camera).

        Returns a diagnostics dict with per-cluster sizes, sides, and the
        cluster-centroid pairwise cosine similarity matrix. Callers should
        check ``diagnostics["max_centroid_cos"] <= GLOBAL_SEED_MAX_CENTROID_COS``
        for the pre-registered duplicate-pid gate.
        """
        if self.frozen_player_ids:
            logger.info("global_seed_from_rallies: skipped (frozen profiles)")
            return {"seeded": False, "reason": "frozen_profiles"}

        # Collect (rally_idx, tid, feature, side, avg_y) across all rallies.
        entries: list[dict[str, Any]] = []
        for rally_idx, (stats, positions, csy, team_assign) in enumerate(
            rally_inputs
        ):
            # Compute per-track avg Y for tiebreak within a cluster's majority side.
            track_ys: dict[int, list[float]] = defaultdict(list)
            for p in positions:
                if p.track_id >= 0:
                    track_ys[p.track_id].append(p.y)
            for tid, ts in stats.items():
                feat = _cluster_feature(ts)
                if feat is None:
                    continue
                ys = track_ys.get(tid, [])
                if not ys:
                    continue
                avg_y = float(np.mean(ys))
                # Decide side per track: calibrator > team_assign > court_split_y > Y-median.
                side: int | None = None
                if team_assign and tid in team_assign:
                    side = int(team_assign[tid])
                elif csy is not None:
                    side = 0 if avg_y > csy else 1
                else:
                    side = None  # defer to per-cluster majority
                entries.append({
                    "rally_idx": rally_idx,
                    "tid": tid,
                    "feat": feat,
                    "side": side,
                    "avg_y": avg_y,
                })

        if len(entries) < k:
            logger.warning(
                "global_seed_from_rallies: only %d tracks, need ≥%d — skipping",
                len(entries), k,
            )
            return {"seeded": False, "reason": "insufficient_tracks",
                    "n_tracks": len(entries)}

        feats = np.stack([e["feat"] for e in entries], axis=0)
        km = KMeans(n_clusters=k, n_init=30, random_state=42)
        labels = km.fit_predict(feats).astype(np.int32)
        centers = km.cluster_centers_.astype(np.float32)

        # Duplicate-pid gate: pairwise cosine similarity between cluster
        # centroids. When two clusters are nearly identical, k-means has not
        # separated the 4 players and seeding would propagate duplicates.
        max_cos = 0.0
        pair_cos: dict[tuple[int, int], float] = {}
        for i in range(k):
            for j in range(i + 1, k):
                c = _cosine(centers[i], centers[j])
                pair_cos[(i, j)] = c
                if c > max_cos:
                    max_cos = c

        # Per-cluster aggregates for diagnostics + fallback.
        cluster_entries: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for i, e in enumerate(entries):
            cluster_entries[int(labels[i])].append(e)
        cluster_med_y: dict[int, float] = {}
        cluster_side: dict[int, int] = {}
        for c, es in cluster_entries.items():
            sides = [e["side"] for e in es if e["side"] is not None]
            if sides:
                cluster_side[c] = 0 if sides.count(0) >= sides.count(1) else 1
            else:
                cluster_side[c] = -1
            cluster_med_y[c] = float(np.median([e["avg_y"] for e in es]))

        # Cluster -> pid mapping. Must match baseline first-rally Y-sort
        # convention exactly on fixtures where the baseline already works
        # (otherwise we regress correctly-assigned videos). Strategy:
        #
        #   1. Run `_initialize_first_rally`-equivalent on rally 0's TOP 4
        #      tracks: split near/far, sort Y ascending within each side,
        #      assign pids [1,2] near / [3,4] far.
        #   2. Look up each of those 4 first-rally tids' cluster; that
        #      cluster inherits the pid.
        #   3. For clusters not touched by the first rally (≤4 players in
        #      rally 0), fall back to majority-side + Y ordering.
        #
        # This guarantees Phase 3's permutation == baseline's permutation
        # when rally 0 has all 4 primary tracks; the seed's *profile* gain
        # comes from EMA'ing all 4 clusters across ALL rallies.
        # Reproduce exactly what `process_rally` does for the first rally
        # so the seed's permutation cannot drift from baseline. That means:
        #   1. Use the serve-formation-window positions for classification
        #      (same FIRST_RALLY_INIT_WINDOW_FRAMES guard).
        #   2. Route through `_classify_track_sides`, which honors the
        #      calibrator -> team_assign -> csy -> Y-median priority
        #      cascade and therefore matches production exactly.
        #   3. Use `_top_tracks_by_frames` for the top-4 selection.
        r0_stats, r0_positions, r0_csy, r0_team = rally_inputs[0]
        r0_classification_positions = r0_positions
        windowed = [
            p for p in r0_positions
            if p.frame_number < FIRST_RALLY_INIT_WINDOW_FRAMES
        ]
        if len(windowed) >= 4 * 10:
            r0_classification_positions = windowed

        r0_avg_y, r0_sides = self._classify_track_sides(
            r0_stats, r0_classification_positions, r0_csy, r0_team,
        )
        top4_r0 = self._top_tracks_by_frames(list(r0_sides.keys()), r0_stats, 4)

        near_r0 = sorted(
            [t for t in top4_r0 if r0_sides.get(t, 0) == 0],
            key=lambda t: r0_avg_y.get(t, 0.5),
        )
        far_r0 = sorted(
            [t for t in top4_r0 if r0_sides.get(t, 0) == 1],
            key=lambda t: r0_avg_y.get(t, 0.5),
        )

        # Build cluster -> pid via rally-0 Y-sort assignment.
        tid_to_cluster_r0: dict[int, int] = {
            e["tid"]: int(labels[i])
            for i, e in enumerate(entries)
            if e["rally_idx"] == 0
        }
        cluster_to_pid: dict[int, int] = {}
        used_clusters: set[int] = set()
        r0_pid_assignments: list[tuple[int, int]] = []  # (tid, pid) log
        for pid, tid in zip([1, 2], near_r0[:2]):
            cid_lookup = tid_to_cluster_r0.get(tid)
            if cid_lookup is not None and cid_lookup not in used_clusters:
                cluster_to_pid[cid_lookup] = pid
                used_clusters.add(cid_lookup)
                r0_pid_assignments.append((tid, pid))
        for pid, tid in zip([3, 4], far_r0[:2]):
            cid_lookup = tid_to_cluster_r0.get(tid)
            if cid_lookup is not None and cid_lookup not in used_clusters:
                cluster_to_pid[cid_lookup] = pid
                used_clusters.add(cid_lookup)
                r0_pid_assignments.append((tid, pid))

        # Fallback for any cluster not yet assigned (rally 0 missing a
        # player, or cluster collision). Take the remaining clusters by
        # majority-side + ascending median Y, filling remaining pids in
        # side-grouped order [1,2 near, 3,4 far].
        missing_pids = [pid for pid in (1, 2, 3, 4)
                        if pid not in cluster_to_pid.values()]
        unassigned_clusters = [c for c in cluster_entries.keys()
                               if c not in used_clusters]
        if missing_pids and unassigned_clusters:
            near_unassigned = sorted(
                [c for c in unassigned_clusters if cluster_side.get(c, -1) == 0],
                key=lambda c: cluster_med_y[c],
            )
            far_unassigned = sorted(
                [c for c in unassigned_clusters if cluster_side.get(c, -1) == 1],
                key=lambda c: cluster_med_y[c],
            )
            ambig = [c for c in unassigned_clusters
                     if cluster_side.get(c, -1) == -1]
            # Ambiguous clusters → assign to whichever side is short.
            # Sort ambiguous by Y: highest Y goes near first.
            ambig_hi = sorted(ambig, key=lambda c: -cluster_med_y[c])
            for pid in missing_pids:
                if pid in (1, 2) and near_unassigned:
                    cluster_to_pid[near_unassigned.pop(0)] = pid
                elif pid in (3, 4) and far_unassigned:
                    cluster_to_pid[far_unassigned.pop(0)] = pid
                elif ambig_hi:
                    cluster_to_pid[ambig_hi.pop(0 if pid in (1, 2) else -1)] = pid

        # Check gate before mutating state.
        gate_ok = max_cos <= GLOBAL_SEED_MAX_CENTROID_COS

        diagnostics: dict[str, Any] = {
            "seeded": False,
            "k": k,
            "n_tracks": len(entries),
            "cluster_sizes": {c: len(es) for c, es in cluster_entries.items()},
            "cluster_sides": dict(cluster_side),
            "cluster_med_y": dict(cluster_med_y),
            "cluster_to_pid": dict(cluster_to_pid),
            "r0_top4": list(top4_r0),
            "r0_near": list(near_r0),
            "r0_far": list(far_r0),
            "r0_avg_y": dict(r0_avg_y),
            "r0_sides": dict(r0_sides),
            "r0_pid_assignments": r0_pid_assignments,
            "max_centroid_cos": max_cos,
            "pair_centroid_cos": pair_cos,
            "gate_passed": gate_ok,
        }

        if not gate_ok:
            logger.warning(
                "global_seed_from_rallies: gate FAILED — max centroid cosine "
                "%.3f > %.2f; NOT seeding (duplicate-pid risk)",
                max_cos, GLOBAL_SEED_MAX_CENTROID_COS,
            )
            return diagnostics

        # Seed profiles: aggregate all (rally, tid) features per cluster and
        # SET the profile directly (not via EMA). Feeding features through
        # update_from_features with PROFILE_EMA_ALPHA=0.10 turns each profile
        # into a rolling window of the last ~25 updates, so order matters and
        # the profile ends up biased toward whichever track we fed last.
        # Setting averages directly from a TrackAppearanceStats-computed mean
        # gives a stable match-global prototype per cluster.
        cluster_features: dict[int, list[Any]] = defaultdict(list)
        cluster_reid_embs: dict[int, list[np.ndarray]] = defaultdict(list)
        cluster_reid_shape: dict[int, tuple[int, ...]] = {}
        for e_idx, e in enumerate(entries):
            c = int(labels[e_idx])
            rally_idx = e["rally_idx"]
            tid = e["tid"]
            stats, _positions, _csy, _team_assign = rally_inputs[rally_idx]
            ts = stats.get(tid)
            if ts is None:
                continue
            cluster_features[c].extend(ts.features)
            if ts.reid_embedding is not None:
                if c not in cluster_reid_shape:
                    cluster_reid_shape[c] = ts.reid_embedding.shape
                if ts.reid_embedding.shape == cluster_reid_shape[c]:
                    cluster_reid_embs[c].append(ts.reid_embedding)

        for c, feats_list in cluster_features.items():
            pid_for_cluster = cluster_to_pid.get(c)
            if pid_for_cluster is None or not feats_list:
                continue
            pid = pid_for_cluster
            aggregated = TrackAppearanceStats(track_id=-1)
            aggregated.features = list(feats_list)
            aggregated.compute_averages()

            profile = self.state.players[pid]
            if aggregated.avg_upper_hist is not None:
                profile.avg_upper_hist = aggregated.avg_upper_hist
                profile.upper_hist_count = len(feats_list)
            if aggregated.avg_lower_hist is not None:
                profile.avg_lower_hist = aggregated.avg_lower_hist
                profile.lower_hist_count = len(feats_list)
            if aggregated.avg_upper_v_hist is not None:
                profile.avg_upper_v_hist = aggregated.avg_upper_v_hist
                profile.upper_v_hist_count = len(feats_list)
            if aggregated.avg_lower_v_hist is not None:
                profile.avg_lower_v_hist = aggregated.avg_lower_v_hist
                profile.lower_v_hist_count = len(feats_list)
            if aggregated.avg_skin_tone_hsv is not None:
                profile.avg_skin_tone_hsv = aggregated.avg_skin_tone_hsv
                profile.skin_sample_count = len(feats_list)
            if aggregated.avg_dominant_color_hsv is not None:
                profile.avg_dominant_color_hsv = aggregated.avg_dominant_color_hsv
                profile.dominant_color_count = len(feats_list)
            if aggregated.avg_head_hist is not None:
                profile.avg_head_hist = aggregated.avg_head_hist
                profile.head_hist_count = len(feats_list)

            # ReID: average the per-track mean embeddings and re-normalize.
            embs = cluster_reid_embs.get(c, [])
            if embs:
                stacked = np.stack(embs, axis=0)
                mean_emb = stacked.mean(axis=0).astype(np.float32)
                nrm = float(np.linalg.norm(mean_emb))
                if nrm > 1e-9:
                    profile.reid_embedding = mean_emb / nrm
                    profile.reid_embedding_count = len(embs)

        # Set side assignments to match the seed (near pids 1-2 -> 0, far 3-4 -> 1).
        for pid in (1, 2):
            self.state.current_side_assignment[pid] = 0
            self.state.players[pid].team = 0
        for pid in (3, 4):
            self.state.current_side_assignment[pid] = 1
            self.state.players[pid].team = 1

        self._global_seeded = True
        diagnostics["seeded"] = True
        logger.info(
            "global_seed_from_rallies: seeded %d profiles from %d (rally, tid) "
            "tracks; max_centroid_cos=%.3f",
            k, len(entries), max_cos,
        )
        return diagnostics

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

        # Pre-compute per-track ReID margins for confidence gating.
        # Only blend ReID when the margin (2nd best - best cost) is large
        # enough that the model is discriminating, not guessing.
        reid_use_for_track: dict[int, bool] = {}
        reid_costs_cache: dict[tuple[int, int], float] = {}
        for i, tid in enumerate(track_ids):
            if tid not in track_stats:
                continue
            track_emb = track_stats[tid].reid_embedding
            if track_emb is None:
                continue
            costs_for_track: list[float] = []
            for j, pid in enumerate(all_player_ids):
                if pid not in self.state.players:
                    continue
                profile_emb = self.state.players[pid].reid_embedding
                if profile_emb is None:
                    continue
                if profile_emb.shape != track_emb.shape:
                    # Log once per rally — profile is 384-dim (DINOv2 ref crops)
                    # but track is 128-dim (general model) or vice versa
                    logger.debug(
                        "ReID dimension mismatch: profile %s vs track %s, skipping",
                        profile_emb.shape, track_emb.shape,
                    )
                    continue
                rc = 1.0 - float(np.dot(profile_emb, track_emb))
                reid_costs_cache[(tid, pid)] = rc
                costs_for_track.append(rc)
            if len(costs_for_track) >= 2:
                sorted_costs = sorted(costs_for_track)
                margin = sorted_costs[1] - sorted_costs[0]
                reid_use_for_track[tid] = margin >= REID_MIN_MARGIN
            else:
                reid_use_for_track[tid] = False

        # Build cost matrix: appearance + optional side penalty + position
        active_side_penalty = (
            SIDE_PENALTY_CALIBRATED if self._sides_from_calibration
            else SIDE_PENALTY
        )
        default_cost = 1.0 + (active_side_penalty if use_side_penalty else 0.0)
        cost_matrix = np.full((size, size), default_cost)
        for i, tid in enumerate(track_ids):
            if tid not in track_stats:
                continue
            track_side = track_court_sides.get(tid)
            use_reid = reid_use_for_track.get(tid, False)
            for j, pid in enumerate(all_player_ids):
                if pid not in self.state.players:
                    continue
                hsv_cost = compute_appearance_similarity(
                    self.state.players[pid], track_stats[tid]
                )

                # Blend ReID only when margin gate passes for this track
                if use_reid and (tid, pid) in reid_costs_cache:
                    reid_cost = reid_costs_cache[(tid, pid)]
                    appearance_cost = (
                        reid_cost * REID_BLEND + hsv_cost * (1 - REID_BLEND)
                    )
                else:
                    appearance_cost = hsv_cost

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
                    side_pen = active_side_penalty if track_side != player_side else 0.0
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
            # Skip frozen profiles (user-provided reference crops)
            if player_id in self.frozen_player_ids:
                continue

            stats = track_stats[track_id]
            profile = self.state.players[player_id]

            # Update profile with each feature sample
            for features in stats.features:
                profile.update_from_features(features)

            # Update ReID embedding (per-track average, not per-sample)
            if stats.reid_embedding is not None:
                if profile.reid_embedding is None:
                    profile.reid_embedding = stats.reid_embedding.copy()
                else:
                    alpha = profile._ema_weight(profile.reid_embedding_count)
                    profile.reid_embedding = (
                        profile.reid_embedding * (1 - alpha)
                        + stats.reid_embedding * alpha
                    )
                    # Re-normalize to unit sphere
                    norm = np.linalg.norm(profile.reid_embedding)
                    if norm > 0:
                        profile.reid_embedding /= norm
                profile.reid_embedding_count += 1

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

        Two-phase approach with physical constraints:
          Phase A: Dense candidate search — every rally position is a
            candidate, excluding positions with inter-rally gap < 8s
            (physical constraint: walking across the court takes ≥11s).
            Appearance sign changes are prioritized when >8 candidates.
            Try all 2^K valid combinations (K≤8), requiring minimum
            spacing of 4 rallies between switches (beach volleyball
            switches every 5-7 points). Score by normalized pairwise
            team appearance preferences with parsimony penalty.
          Phase B: Refine each detected switch ±1 rally, keeping shifts
            that improve the score or align with serve direction changes.
            Both gap and spacing constraints enforced during refinement.

        Also validates multi-switch results by checking each switch has
        positive marginal contribution (prevents FP piggybacking on TP).

        Returns:
            List of rally indices where side switches should be applied.
            Empty if no switches detected or no improvement found.
        """
        n = len(self.stored_rally_data)
        if n < 3:
            return []

        # Step 1: Build pairwise team preference matrix.
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

        # Step 3: Dense candidate generation — every rally is a candidate,
        # excluding positions where the inter-rally gap is too short for a
        # physical side switch. Players must walk ~16m across the court (≥11s).
        # Threshold set conservatively at 8s (3.3s below the minimum observed
        # GT switch gap of 11.3s across 57 labeled switches in 46 videos).
        min_switch_gap_ms = 8_000
        excluded_by_gap: set[int] = set()
        for k in range(1, n):
            gap_ms = (
                self.stored_rally_data[k].start_ms
                - self.stored_rally_data[k - 1].end_ms
            )
            if 0 < gap_ms < min_switch_gap_ms:
                excluded_by_gap.add(k)
        if excluded_by_gap:
            logger.info(
                "Side switch: excluded %d positions with gap < %.0fs: %s",
                len(excluded_by_gap),
                min_switch_gap_ms / 1000,
                sorted(excluded_by_gap),
            )

        candidates = [k for k in range(1, n) if k not in excluded_by_gap]

        if len(candidates) > 8:
            # Prioritize appearance-based candidates (sign changes in
            # normalized preference with prior rallies).
            priority: list[int] = []
            if n >= 4:
                avg_pref_with_prior = np.zeros(n)
                for k in range(1, n):
                    vals = [
                        norm_pref[k, j]
                        for j in range(k) if norm_pref[k, j] != 0.0
                    ]
                    if vals:
                        avg_pref_with_prior[k] = float(np.mean(vals))

                if avg_pref_with_prior[1] < -0.01:
                    priority.append(1)

                for k in range(2, n):
                    prev_val = avg_pref_with_prior[k - 1]
                    curr_val = avg_pref_with_prior[k]
                    if (prev_val > 0.01 and curr_val < -0.01) or (
                        prev_val < -0.01 and curr_val > 0.01
                    ):
                        priority.append(k)

            # Fill remaining slots with evenly-spaced positions
            remaining = [c for c in candidates if c not in set(priority)]
            if len(priority) >= 8:
                candidates = priority[:8]
            else:
                slots = 8 - len(priority)
                if remaining:
                    step = max(1, (len(remaining) - 1) / max(1, slots - 1))
                    indices = sorted({min(round(i * step), len(remaining) - 1)
                                      for i in range(slots)})
                    candidates = sorted(
                        set(priority) | {remaining[i] for i in indices}
                    )
                else:
                    candidates = priority[:8]

        logger.info(
            "Side switch search: %d candidates at %s",
            len(candidates), candidates,
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

        # Minimum spacing between consecutive switches. Beach volleyball
        # switches sides every 7 points (sets 1-2) or 5 (set 3). Minimum
        # observed GT spacing is 6 rallies across 46 videos. Threshold of
        # 4 is conservative (2 below min observed).
        min_switch_spacing = 4

        def has_valid_spacing(switches: set[int]) -> bool:
            """Check that all switches are >= min_switch_spacing apart."""
            if len(switches) < 2:
                return True
            s = sorted(switches)
            return all(
                s[i + 1] - s[i] >= min_switch_spacing
                for i in range(len(s) - 1)
            )

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
            if not has_valid_spacing(switch_set):
                continue
            score = score_partition(switch_set)
            if score > best_score:
                best_score = score
                best_switches = sorted(switch_set)

        if best_switches:
            # Post-validation: drop switches whose marginal contribution
            # is too small. A FP switch can piggyback on a real TP switch
            # that dominates the total score.
            if len(best_switches) > 1:
                validated: list[int] = []
                for sw in best_switches:
                    without = set(best_switches) - {sw}
                    marginal = best_score - score_partition(without)
                    if marginal > 0:
                        validated.append(sw)
                        logger.info(
                            "  Switch %d: marginal=+%.3f (kept)", sw, marginal,
                        )
                    else:
                        logger.info(
                            "  Switch %d: marginal=%.3f (dropped)", sw, marginal,
                        )
                if len(validated) < len(best_switches):
                    logger.info(
                        "Side switch validation: %d→%d switches",
                        len(best_switches), len(validated),
                    )
                    best_switches = validated
                    if not best_switches:
                        logger.info("Side switch search: all switches dropped")
                        return []
                    best_score = score_partition(set(best_switches))

            # Build set of rally indices where serve direction changes
            # (used for Phase B tie-breaking only).
            serve_dirs = [d.serve_direction for d in self.stored_rally_data]
            direction_changes: set[int] = set()
            for i in range(1, n):
                if (
                    serve_dirs[i] != "?"
                    and serve_dirs[i - 1] != "?"
                    and serve_dirs[i] != serve_dirs[i - 1]
                ):
                    direction_changes.add(i)

            # Phase B: Refine each switch boundary ±1.
            # Ball direction can trigger at rally k when the actual switch
            # is at k±1. Try shifting each switch individually and keep
            # the shift if it improves the score (or aligns with a serve
            # direction change on tie).
            refined = list(best_switches)
            for idx in range(len(refined)):
                sw = refined[idx]
                others = {refined[j] for j in range(len(refined)) if j != idx}
                best_local = sw
                best_local_score = best_score
                for delta in [-1, 1]:
                    alt = sw + delta
                    if alt < 1 or alt >= n or alt in others:
                        continue
                    if alt in excluded_by_gap:
                        continue
                    trial = others | {alt}
                    if not has_valid_spacing(trial):
                        continue
                    trial_score = score_partition(trial)
                    if trial_score > best_local_score or (
                        trial_score == best_local_score
                        and alt in direction_changes
                        and sw not in direction_changes
                    ):
                        best_local = alt
                        best_local_score = trial_score
                if best_local != sw:
                    logger.info(
                        "  Refined switch %d → %d (+%.3f)",
                        sw, best_local, best_local_score - best_score,
                    )
                    refined[idx] = best_local
                    best_score = best_local_score
            best_switches = sorted(refined)

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
            # Restore the player→side mapping and calibration flag from Pass 1
            # so side penalties are correct for pre-switch rallies.
            saved_side = self.state.current_side_assignment
            self.state.current_side_assignment = data.player_side_assignment
            self._sides_from_calibration = data.sides_from_calibration

            # No position continuity in Pass 2 — rebuilding the position
            # chain from scratch can propagate errors.
            track_to_player = self._assign_tracks_to_players_global(
                data.top_tracks,
                data.track_stats,
                data.track_court_sides,
                use_side_penalty=not self.frozen_player_ids,
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

                    # Blend ReID when any track has embeddings
                    _rb = REID_BLEND if (
                        stats_lo_a.reid_embedding is not None
                        or stats_lo_b.reid_embedding is not None
                    ) else 0.0

                    # Cost of "same ordering" (lo↔lo, hi↔hi)
                    same_cost = (
                        compute_track_similarity(stats_lo_a, stats_lo_b, _rb)
                        + compute_track_similarity(stats_hi_a, stats_hi_b, _rb)
                    )
                    # Cost of "swapped ordering" (lo↔hi, hi↔lo)
                    swap_cost = (
                        compute_track_similarity(stats_lo_a, stats_hi_b, _rb)
                        + compute_track_similarity(stats_hi_a, stats_lo_b, _rb)
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
    extract_reid: bool = False,
    reid_model: GeneralReIDModel | None = None,
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
        extract_reid: If True, also extract DINOv2 embeddings per track.
        reid_model: Optional GeneralReIDModel for embedding extraction.
            When provided, uses its projection head instead of raw backbone.

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

    # Collect BGR crops per track for ReID embedding extraction
    reid_crops: dict[int, list[np.ndarray]] = {} if extract_reid else {}

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

                # Extract BGR crop for ReID
                if extract_reid:
                    crop = extract_bbox_crop(
                        frame_arr, bbox, frame_width, frame_height,
                    )
                    if crop is not None:
                        reid_crops.setdefault(tid, []).append(crop)
    finally:
        cap.release()

    # Compute averages
    for s in stats.values():
        s.compute_averages()

    # Compute per-track ReID embeddings from collected crops.
    # Wrapped in try-catch: DINOv2 failure falls back to HSV-only silently.
    if extract_reid and reid_crops:
        try:
            if reid_model is not None:
                # General model: use projection head embeddings
                for tid, crops in reid_crops.items():
                    if not crops or tid not in stats:
                        continue
                    embeddings = reid_model.extract_embeddings(crops)
                    mean_emb = embeddings.mean(axis=0)
                    norm = np.linalg.norm(mean_emb)
                    if norm > 0:
                        mean_emb /= norm
                    stats[tid].reid_embedding = mean_emb
            else:
                # Per-video: raw DINOv2 backbone features (384-dim)
                from rallycut.tracking.reid_embeddings import extract_backbone_features

                for tid, crops in reid_crops.items():
                    if not crops or tid not in stats:
                        continue
                    embeddings = extract_backbone_features(crops)
                    mean_emb = embeddings.mean(axis=0)
                    norm = np.linalg.norm(mean_emb)
                    if norm > 0:
                        mean_emb /= norm
                    stats[tid].reid_embedding = mean_emb
        except Exception:
            logger.warning(
                "ReID embedding extraction failed, falling back to HSV-only",
                exc_info=True,
            )
            # Clear any partial embeddings
            for s in stats.values():
                s.reid_embedding = None

    return stats


@dataclass
class MatchPlayersResult:
    """Result of cross-rally player matching."""

    rally_results: list[RallyTrackingResult]
    player_profiles: dict[int, PlayerAppearanceProfile]  # player_id -> profile
    team_templates: tuple[TeamTemplate, TeamTemplate] | None = None
    diagnostics: list[RallyAssignmentDiagnostics] = field(default_factory=list)


def match_players_across_rallies(
    video_path: Path,
    rallies: list[RallyTrackData],
    num_samples: int = 12,
    collect_diagnostics: bool = False,
    reference_profiles: dict[int, PlayerAppearanceProfile] | None = None,
    extract_reid: bool = False,
    reid_model: GeneralReIDModel | None = None,
    calibrator: CourtCalibrator | None = None,
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
        reference_profiles: Optional user-provided frozen profiles (player_id -> profile).
            When provided, profiles are never updated — they anchor all assignments.
        extract_reid: If True, extract DINOv2 embeddings per track for ReID-based
            cost blending in the Hungarian assignment. Auto-enabled when reference
            profiles contain ReID embeddings or a general ReID model is provided.
        reid_model: Optional GeneralReIDModel for embedding extraction.
            When provided, uses its projection head. Falls back to raw backbone
            when not provided but reference profiles have embeddings.
        calibrator: Optional court calibrator for authoritative near/far side
            classification. When provided, track foot positions are projected
            through the homography to determine court side (net at 8.0m),
            and a hard penalty (SIDE_PENALTY_CALIBRATED) is applied.

    Returns:
        MatchPlayersResult with track→player mappings and accumulated profiles.
    """
    # Auto-enable ReID extraction when general model available
    if not extract_reid and reid_model is not None:
        extract_reid = True
        logger.info("Auto-enabling ReID extraction (general model provided)")

    # Auto-enable ReID extraction when reference profiles have embeddings
    if not extract_reid and reference_profiles:
        if any(p.reid_embedding is not None for p in reference_profiles.values()):
            extract_reid = True
            logger.info("Auto-enabling ReID extraction (reference profiles have embeddings)")

    if reference_profiles:
        logger.info(
            f"Using reference profiles for players: "
            f"{sorted(reference_profiles.keys())}"
        )
    tracker = MatchPlayerTracker(
        calibrator=calibrator,
        collect_diagnostics=collect_diagnostics,
        reference_profiles=reference_profiles,
    )
    results: list[RallyTrackingResult] = []

    # Phase 3 — when MATCH_TRACKER_GLOBAL_SEED=1, pre-extract features for
    # every rally, pool them, and seed profiles via global k-means before
    # the per-rally Hungarian loop. Features are cached in-memory so each
    # rally is only decoded once across seed + assignment.
    prefetched_stats: list[dict[int, TrackAppearanceStats]] = []
    do_global_seed = GLOBAL_SEED_ENABLED and not reference_profiles
    if do_global_seed:
        logger.info(
            "Phase 3: prefetching track_stats for %d rallies to seed profiles "
            "via global k-means (MATCH_TRACKER_GLOBAL_SEED=1)",
            len(rallies),
        )
        for rally in rallies:
            ts = extract_rally_appearances(
                video_path=video_path,
                positions=rally.positions,
                primary_track_ids=rally.primary_track_ids,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                num_samples=num_samples,
                extract_reid=extract_reid,
                reid_model=reid_model,
            )
            prefetched_stats.append(ts)

        seed_inputs = [
            (prefetched_stats[i], r.positions, r.court_split_y, r.team_assignments)
            for i, r in enumerate(rallies)
        ]
        seed_diag = tracker.global_seed_from_rallies(seed_inputs)
        logger.info("global seed diagnostics: %s", seed_diag)

    for rally_idx, rally in enumerate(rallies):
        # Reuse prefetched track_stats when available (Phase 3 seed path);
        # otherwise decode now.
        if do_global_seed:
            track_stats = prefetched_stats[rally_idx]
        else:
            track_stats = extract_rally_appearances(
                video_path=video_path,
                positions=rally.positions,
                primary_track_ids=rally.primary_track_ids,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                num_samples=num_samples,
                extract_reid=extract_reid,
                reid_model=reid_model,
            )

        # Process rally
        result = tracker.process_rally(
            track_stats=track_stats,
            player_positions=rally.positions,
            ball_positions=rally.ball_positions,
            court_split_y=rally.court_split_y,
            team_assignments=rally.team_assignments,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
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

    # Build team templates from final profiles
    team_templates = build_team_templates(tracker.state.players)

    return MatchPlayersResult(
        rally_results=results,
        player_profiles=dict(tracker.state.players),
        team_templates=team_templates,
        diagnostics=tracker.diagnostics,
    )
