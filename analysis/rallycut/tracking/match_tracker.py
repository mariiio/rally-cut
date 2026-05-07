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

import hashlib
import json
import logging
import os
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.tracking import _profile_drift_probe as _probe
from rallycut.tracking._subtrack import SubTrackCandidate
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

    ReIDModel = GeneralReIDModel

logger = logging.getLogger(__name__)

# Side penalty for global Hungarian assignment. Biases toward expected court side
# but doesn't prevent cross-side matching when appearance is a stronger signal.
# Appearance costs range 0.0-1.0, so 0.15 is meaningful but not dominant.
SIDE_PENALTY = 0.15

# Hard side penalty when court calibration provides authoritative side labels.
# Effectively prevents cross-side assignment — players physically cannot be on
# the wrong side of the net.
SIDE_PENALTY_CALIBRATED = 1.0

# Hard team-pair constraint cost (Phase 1 step 3). Applied in Pass 2 Stage 1
# only — after combinatorial side-switch detection has run, so the player→
# side mapping reflects the rally's post-switch state. A track is considered
# "high-confidence side" when the y-coord and bbox-height classifiers AGREE,
# AND the high-confidence agreement set forms a clean 2v2 split. In those
# cases, cross-team Hungarian assignments are made structurally infeasible
# by setting the cell cost to this value (>> max soft cost ~1.5). When the
# partition is degenerate (3v1, 1v3, or fewer than 2 confident per side),
# the constraint is skipped and the soft side penalty governs.
HARD_TEAM_PAIR_COST = 100.0

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
# Override via `MATCH_SOLVER_REID_BLEND` env var to favor ReID more
# heavily on fixtures where HSV is noisy (e.g., uniform-similar players,
# heavy sand background).
REID_BLEND = float(os.environ.get("MATCH_SOLVER_REID_BLEND", "0.50"))

# Identity-first matching for partial-cardinality rallies (Bug D, 2026-05-04).
# When a rally has fewer than 4 primary tracks (e.g., near-side server
# physically occludes the partner the entire rally), the existing path's
# formation-seed Y-sort + side-penalty can misassign the visible tracks'
# PIDs because the partition is degenerate (1v2 instead of 2v2). The
# identity-first path skips formation/side bias and identifies each
# visible track by appearance similarity to the cross-rally per-PID
# gallery profiles (`self.state.players`). Output is a sparse
# track→PID mapping; the missing PID slots are explicitly empty.
#
# Default OFF. Validation precedent: Layer 1 LORO on 4 GT fixtures was
# 97.8% (44/45), with 7/7 side-switch rallies correct — confirms the
# gallery primitive is sound and side-switch invariant. Flag stays OFF
# until visual validation on dd042609 r13/r18 + GT-fixture PERMUTED
# baselines hold.
ENABLE_IDENTITY_FIRST_PARTIAL = (
    os.environ.get("ENABLE_IDENTITY_FIRST_MATCHING", "0") == "1"
)
# Minimum anchor rallies needed in the video before we trust the gallery
# enough to use it for partial-rally identity. Below this we fall back to
# the existing path (which may be wrong for partial cardinality, but at
# least doesn't depend on a sparse gallery).
IDENTITY_FIRST_MIN_ANCHORS = 3

# Minimum margin (second_best - best ReID cost) to trust the ReID signal
# for a given track. When the margin is small, all players look similar
# to the track → ReID isn't discriminative → fall back to HSV only.
# Protects against regressions from unrepresentative reference crops.
REID_MIN_MARGIN = 0.08

# Phase 3 global k-means seeding is a closed NO-GO workstream (see
# memory/attribution_primitive_first_phase0_2026_04_24.md §dormant flags).
# The flag was env-gated and the env-read line was removed during the
# dormant-flag cleanup, but the call-site reference at ~line 2650 was
# left dangling and broke `match-players` at runtime with NameError.
# Defining the constants as `False` / sentinel keeps the dormant path
# compiled while leaving the workstream closed; remove constants +
# call-site together when the path is excised.
GLOBAL_SEED_ENABLED = False
GLOBAL_SEED_MAX_CENTROID_COS = 0.45

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

# ---------------------------------------------------------------------------
# Within-track appearance segmentation (Task 3-5, 2026-04-26)
# ---------------------------------------------------------------------------

# How many appearance windows to score per track. K=6 gives enough resolution
# to localize a flip near either rally end while keeping classifier inference
# cheap.
SEGMENT_NUM_WINDOWS = 6

# Minimum frames per window. Below this, the window is dropped (too noisy).
SEGMENT_MIN_WINDOW_FRAMES = 12

# Minimum per-segment aggregate margin (best-pid prob minus 2nd-best) required
# on BOTH the pre and post segments before a split fires. The 2026-04-26
# few-shot probe measured PRE/POST aggregate margins of +0.188/+0.442
# (cuco r5), +0.228/+0.317 (cuco r3), +0.350/+0.305 (wawa r10) on real
# splits, and +0.086/+0.004 (cuco r5 tid=2) / +0.352/+0.140 (wawa r10 tid=3)
# on borderline cases that should abstain. 0.15 is the tightest gate that
# fires all real splits and abstains on both borderline cases.
SEGMENT_MIN_PER_SEGMENT_MARGIN = 0.15

# Minimum number of consecutive confirming windows on each side of the flip
# point. Prevents single-window noise from triggering a split.
SEGMENT_MIN_CONFIRMING_WINDOWS = 2

# Per-frame inference is expensive — sample at most this many frames per
# window for classifier scoring.
SEGMENT_FRAMES_PER_WINDOW = 4


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
        """Initialize 4 player profiles for beach volleyball.

        Note: the ``team`` and ``current_side_assignment`` defaults seeded
        here using ``pid <= 2`` are PLACEHOLDERS — they get overwritten by
        per-rally positional team detection during ``process_rally`` and by
        ``build_team_templates``' positional mode-vote at the end of the
        match. Don't rely on these initial values for any team-membership
        logic; read from `verify_team_assignments` / `team_templates`.
        """
        for player_id in range(1, 5):
            if player_id not in self.players:
                self.players[player_id] = PlayerAppearanceProfile(
                    player_id=player_id,
                    team=0 if player_id <= 2 else 1,
                )
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
    # Sub-tracks emitted by within-track appearance segmentation (Task 4).
    # Empty when the track-split flag is off, no classifier is available, or
    # no segmentation occurred. Used by the CLI to persist `subTracks` in
    # match_analysis_json so remap-track-ids can apply frame-conditional pid
    # resolution. Default-empty for backward compatibility — callers that
    # don't construct sub-tracks (Pass 2 refinement, scratchpad replay) leave
    # it unset and the field carries forward as `[]`.
    sub_tracks: list[SubTrackCandidate] = field(default_factory=list)


@dataclass
class RallyTrackData:
    """Data for a single rally loaded from the database."""

    rally_id: str
    video_id: str
    start_ms: int
    end_ms: int
    positions: list[PlayerPosition]
    # Invariant: 0..PlayerFilterConfig.max_players DISTINCT NON-NEGATIVE ints.
    # Validated at write by `validate_primary_track_ids` in player_filter;
    # auto-cleaned at read by `load_rallies_for_video` for legacy DB rows.
    # See scripts/repair_primary_track_ids.py for the migration tool.
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
    # Independent bbox-height-based side classification (Phase 1 step 1).
    # Empty dict when fewer than 2 distinct tracks were available. Used by
    # the team-pair inference (Phase 1 step 3) as a cross-check; tracks
    # where this disagrees with `track_court_sides` are treated as
    # low-confidence votes for team membership.
    sides_by_bbox: dict[int, int] = field(default_factory=dict)
    # Late-rally positions per top track (avg over last 30 frames). Used
    # by ``MatchSolver`` for cross-rally position continuity. Not persisted
    # by ``to_dict`` — solver runs in-memory only and the field is not
    # needed for downstream scratchpad inspection.
    late_positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Upstream tracking-pipeline team labels: track_id -> 0 (near) / 1 (far),
    # computed by `compute_court_split` / `classify_teams` from bbox-size
    # clustering. Independent of the y-position and bbox-height side
    # signals computed inside this module — used as a 3rd vote by the
    # multi-signal team-pair partition determination in MatchSolver
    # (`_propose_team_partitions`). Empty dict when upstream couldn't
    # produce assignments.
    team_assignments: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the replay-relevant subset to a JSON-compatible dict.

        Excludes early_positions (unused in Pass 2 — refine_assignments hard-
        codes early_positions=None) and side-switch inputs (serve_direction,
        start_ms, end_ms) since the side-switch partition is captured at the
        match level, not re-detected on replay.
        """
        return {
            "track_stats": {
                str(tid): stats.to_dict() for tid, stats in self.track_stats.items()
            },
            "track_court_sides": {
                str(tid): int(side) for tid, side in self.track_court_sides.items()
            },
            "top_tracks": [int(t) for t in self.top_tracks],
            "player_side_assignment": {
                str(pid): int(team) for pid, team in self.player_side_assignment.items()
            },
            "sides_from_calibration": bool(self.sides_from_calibration),
            "sides_by_bbox": {
                str(tid): int(side) for tid, side in self.sides_by_bbox.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoredRallyData:
        """Deserialize from a dict produced by `to_dict`."""
        return cls(
            track_stats={
                int(tid): TrackAppearanceStats.from_dict(stats_d)
                for tid, stats_d in d.get("track_stats", {}).items()
            },
            track_court_sides={
                int(tid): int(side)
                for tid, side in d.get("track_court_sides", {}).items()
            },
            early_positions={},
            top_tracks=[int(t) for t in d.get("top_tracks", [])],
            player_side_assignment={
                int(pid): int(team)
                for pid, team in d.get("player_side_assignment", {}).items()
            },
            sides_from_calibration=bool(d.get("sides_from_calibration", False)),
            sides_by_bbox={
                int(tid): int(side)
                for tid, side in d.get("sides_by_bbox", {}).items()
            },
        )


def _stored_rally_data_hash(data: StoredRallyData) -> str:
    """SHA256 of the rally's pre-MatchSolver structural state.

    Cache key for the per-rally `assignmentAnchor` mechanism
    (`ENABLE_ASSIGNMENT_ANCHORS=1`). Captures the structural fingerprint of
    the rally: which tracks are top tracks, their court-side classifications,
    and their early/late positions (rounded to 4 decimals to absorb minor
    float-arithmetic noise from per-frame feature extraction).

    NOT hashed: appearance histograms / ReID embeddings. These are functions
    of (positions × video frames × extractor) and have small numerical drift
    on CPU floats. If positions and track IDs match between runs, appearance
    features are reproducible — so the structural fingerprint is the correct
    cache key for "did anything that would change MatchSolver's input
    actually change?". Re-tracking with BoT-SORT changes track IDs (which IS
    captured here); pure feature-extraction non-determinism is not.

    Other rallies' state changes do NOT invalidate this hash — that
    decoupling is exactly the cascade fix (Phase 2 of the post-7307c1d-revert
    refactor).
    """
    def _round_pos(v: tuple[float, float]) -> list[float]:
        return [round(float(v[0]), 4), round(float(v[1]), 4)]

    payload = {
        "top_tracks": sorted(int(t) for t in data.top_tracks),
        "track_court_sides": {
            str(int(k)): int(v) for k, v in sorted(data.track_court_sides.items())
        },
        "sides_by_bbox": {
            str(int(k)): int(v) for k, v in sorted(data.sides_by_bbox.items())
        },
        "early_positions": {
            str(int(k)): _round_pos(v)
            for k, v in sorted(data.early_positions.items())
        },
        "late_positions": {
            str(int(k)): _round_pos(v)
            for k, v in sorted(data.late_positions.items())
        },
        "sides_from_calibration": bool(data.sides_from_calibration),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _aggregate_reid_embeddings(embeddings: np.ndarray) -> np.ndarray | None:
    """Aggregate per-frame ReID embeddings into one per-track embedding.

    Default: MEDOID — the embedding most similar (highest mean cosine
    similarity) to all other frames. Robust to outlier frames (back-
    facing, occluded, motion-blurred) that would pull a mean away from
    the player's true identity centroid.

    The discrimination probe on 5c756c41 (2026-05-02) showed fine-tuned
    OSNet has near-zero off-diagonal similarity (0.05 mean) on
    single-frame embeddings — the model is working. The bug was at this
    aggregation site: `mean(axis=0)` over 12 frames collapsed
    discrimination because some sampled frames were noisy poses. Medoid
    naturally rejects those.

    Override via `RALLYCUT_REID_AGGREGATION` env var:
      - "medoid" (default): outlier-robust single-frame representative.
      - "mean": legacy behavior (backward compat).
    """
    if embeddings is None or embeddings.size == 0:
        return None
    if embeddings.shape[0] == 1:
        return np.asarray(embeddings[0], dtype=np.float32)

    mode = os.environ.get("RALLYCUT_REID_AGGREGATION", "medoid").lower()
    if mode == "mean":
        agg = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(agg))
        if norm > 0:
            agg = agg / norm
        return np.asarray(agg, dtype=np.float32)

    # Medoid: pick the embedding with highest mean cosine similarity to
    # the rest. Embeddings come in already L2-normalized, so cosine
    # similarity is just the dot product.
    sim = embeddings @ embeddings.T  # (N, N)
    # Exclude self-similarity by zeroing the diagonal before the mean.
    n = sim.shape[0]
    mask = 1.0 - np.eye(n, dtype=sim.dtype)
    mean_sim = (sim * mask).sum(axis=1) / max(1, n - 1)
    medoid_idx = int(np.argmax(mean_sim))
    return np.asarray(embeddings[medoid_idx], dtype=np.float32)


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
        self._sides_from_calibration = False
        # Side-switch partition from the most recent refine_assignments() call.
        # Populated by refine_assignments stage 0 so callers (and the relabel
        # scratchpad) can replay Pass-2 without re-running detection.
        self.last_side_switches: list[int] = []

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
        # On the first rally, the calibration projection prefers the
        # clean serve-formation window: the serve moment has a formal 2v2
        # arrangement by volleyball rules, and mid-rally players cross
        # sides and contaminate whole-rally Y averages. The first rally's
        # assignment seeds every subsequent rally's templates, so an init
        # error here propagates to the whole match.
        #
        # We pass full-rally positions to ``_classify_track_sides`` and
        # let it window internally for the calibration projection only —
        # ``track_avg_y`` is built from full positions so late-arriver
        # primary tracks (those starting after the serve window) are
        # classified by the image-Y fallback rather than silently dropped.
        # The ``_classify_sides_by_bbox_height`` cross-check still uses
        # the windowed positions on rally 1 for consistency with the
        # serve-formation intent.
        #
        # Restrict to primary-track samples first — the upstream
        # ``_filter_player_positions`` pipeline is supposed to drop
        # spectator / referee tracks but sometimes leaks them into
        # ``positions_json`` (90266c1d rallies 5+6 have T9/T15/T101/T28/T6
        # with 1-100+ samples each despite primary_track_ids = [1,2,3,4]).
        # ``track_stats`` is already correctly restricted to primary
        # tracks by ``extract_rally_appearances``, so its keys are the
        # trusted primary set. Without this filter, the bbox-height
        # cross-check counts leaked tracks toward its 2v2 partition.
        if track_stats:
            primary_tids = set(track_stats.keys())
            full_primary_positions = [
                p for p in player_positions
                if p.track_id < 0 or p.track_id in primary_tids
            ]
        else:
            full_primary_positions = list(player_positions)

        # Bbox-height cross-check uses windowed positions on rally 1
        # (clean serve-formation snapshot) and full positions otherwise.
        if self.rally_count == 1:
            windowed_primary = [
                p for p in full_primary_positions
                if p.frame_number < FIRST_RALLY_INIT_WINDOW_FRAMES
            ]
            if len(windowed_primary) >= 4 * 10:
                bbox_height_positions = windowed_primary
            else:
                bbox_height_positions = full_primary_positions
            serve_window = FIRST_RALLY_INIT_WINDOW_FRAMES
        else:
            bbox_height_positions = full_primary_positions
            serve_window = None

        track_avg_y, track_court_sides = self._classify_track_sides(
            track_stats, full_primary_positions, court_split_y, team_assignments,
            serve_window_frames=serve_window,
        )

        # Bbox-height side signal (Phase 1 step 1). Independent of the
        # y-coordinate path above. Stored on the rally for downstream
        # team-pair inference (Phase 1 step 3) to cross-check, and logged
        # at WARNING when it disagrees with the y-based classification on
        # any track — a disagreement is diagnostic of an unreliable
        # side signal for that track and the team-pair voter should treat
        # it as low-confidence.
        sides_by_bbox = self._classify_sides_by_bbox_height(bbox_height_positions)
        if sides_by_bbox and track_court_sides:
            disagreements = [
                tid for tid, side in sides_by_bbox.items()
                if tid in track_court_sides and track_court_sides[tid] != side
            ]
            if disagreements:
                logger.warning(
                    "Side signal disagreement (rally %d): tracks %s — y-side "
                    "vs bbox-height-side disagree. Team-pair inference will "
                    "treat these as low-confidence.",
                    self.rally_count, disagreements,
                )

        # Step 3: Select top 4 tracks globally by feature count
        all_track_ids = list(track_court_sides.keys())
        top_tracks = self._top_tracks_by_frames(all_track_ids, track_stats, 4)

        # Compute early-rally positions for position continuity
        early_positions = _compute_track_positions(
            player_positions, top_tracks, window=30, from_start=True
        )
        # Late-rally positions feed MatchSolver's cross-rally position
        # continuity term (Day 2 task 8). In-memory only.
        late_positions = _compute_track_positions(
            player_positions, top_tracks, window=30, from_start=False
        )

        # Step 4: Assign tracks to players
        # Side switch detection runs in Pass 2 (combinatorial search)
        side_switch_detected = False

        sub_tracks: list[SubTrackCandidate] = []

        # W4 (`ENABLE_BBOX_SWAP_DETECTION`) removed 2026-05-03 per
        # dormant_flag_audit_2026_05_03.md. The bbox-overlap within-rally
        # swap detector was re-validated NO-GO twice (2026-04-30 +
        # 2026-05-01) — fixed 0/3 within-rally drift rallies and
        # regressed b5fb0594/r10 to UNLABELED 37.5%. See
        # `w4_revalidated_NOGO_2026_05_01.md`.

        if self.rally_count <= 1:
            # Phase 3: seeded profiles enrich the per-match prior, but the
            # first-rally track->pid mapping still goes through the baseline
            # Y-sort so convention remains identical. The seed's benefit
            # is stabilizing rallies 2+ via richer HSV/ReID profiles.
            track_to_player = self._initialize_first_rally(
                top_tracks, track_avg_y, track_court_sides
            )
        else:
            # Optional pre-Hungarian within-track split (Task 4, 2026-04-26).
            # Currently a stub — the classifier path was ref-crop-driven and
            # is unreachable post-cleanup; method body to be deleted in a
            # later phase.
            sub_tracks = self._maybe_segment_tracks_by_appearance(
                track_ids=top_tracks,
                track_stats=track_stats,
                positions=player_positions,
                classifier=getattr(self, "_few_shot_classifier", None),
                crop_extractor=getattr(self, "_crop_extractor", None),
            )
            if sub_tracks:
                all_pids = sorted(self.state.players.keys())
                direct_subtrack_assignments, remaining_top_tracks, remaining_pids = (
                    self._apply_subtrack_assignments(sub_tracks, top_tracks, all_pids)
                )
                # Run Hungarian on remaining real tracks against remaining pids.
                hungarian_result = self._assign_tracks_to_players_global(
                    remaining_top_tracks, track_stats, track_court_sides,
                    use_side_penalty=True,
                    early_positions=early_positions,
                    restrict_to_pids=remaining_pids,
                )
                # Combine direct sub-track assignments + Hungarian result.
                track_to_player = {**direct_subtrack_assignments, **hungarian_result}
            else:
                track_to_player = self._assign_tracks_to_players_global(
                    top_tracks, track_stats, track_court_sides,
                    use_side_penalty=True,
                    early_positions=early_positions,
                )

        # Stash sub-tracks for downstream per-frame writer (Task 5).
        # Previously this also unioned in W4 sub-tracks; W4 removed
        # 2026-05-03 (dormant_flag_audit_2026_05_03.md).
        self._last_rally_sub_tracks = sub_tracks

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
        # Skip frozen (user-provided) profiles, update the rest normally.
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
            start_ms=start_ms,
            end_ms=end_ms,
            sides_from_calibration=self._sides_from_calibration,
            sides_by_bbox=sides_by_bbox,
            late_positions=late_positions,
            team_assignments=dict(team_assignments) if team_assignments else {},
        ))

        return RallyTrackingResult(
            rally_index=rally_index,
            track_to_player=track_to_player,
            server_player_id=server_player_id,
            side_switch_detected=side_switch_detected,
            assignment_confidence=confidence,
            sub_tracks=list(sub_tracks),
        )

    def _classify_sides_by_bbox_height(
        self,
        player_positions: list[PlayerPosition],
    ) -> dict[int, int]:
        """Classify tracks by median bbox height (a perspective signal).

        In a fixed-camera beach-volleyball setup the camera sits behind one
        baseline. Players closer to the camera have larger bboxes (near
        side, side=0); players on the far side appear smaller (side=1).
        The 2v2 structure means 2 tracks should fall in each cluster.

        Returns:
            ``{track_id: side}`` where ``side ∈ {0 (near), 1 (far)}``,
            or empty when fewer than 2 distinct tracks are available.

        This signal is computed alongside the existing y-coordinate +
        calibrator path so the team-pair inference (Phase 1 step 3) can
        cross-check with it. Within a single rally, near-side and
        far-side bbox heights typically differ by 2-3× — a much stronger
        per-track signal than the y-coordinate gradient when court_split_y
        is noisy or absent.
        """
        per_track_heights: dict[int, list[float]] = {}
        for p in player_positions:
            if p.track_id < 0:
                continue
            per_track_heights.setdefault(p.track_id, []).append(p.height)

        if len(per_track_heights) < 2:
            return {}

        median_h: dict[int, float] = {
            tid: float(np.median(hs))
            for tid, hs in per_track_heights.items()
            if hs
        }
        if len(median_h) < 2:
            return {}

        # 2v2 invariant: rank tracks by height descending; top half = near.
        # Median split is robust to detection noise on individual frames.
        sorted_tracks = sorted(median_h.keys(), key=lambda t: -median_h[t])
        mid = len(sorted_tracks) // 2
        sides: dict[int, int] = {}
        for idx, tid in enumerate(sorted_tracks):
            sides[tid] = 0 if idx < mid else 1
        return sides

    def _classify_track_sides(
        self,
        track_stats: dict[int, TrackAppearanceStats],
        player_positions: list[PlayerPosition],
        court_split_y: float | None,
        team_assignments: dict[int, int] | None = None,
        *,
        serve_window_frames: int | None = None,
    ) -> tuple[dict[int, float], dict[int, int]]:
        """Classify tracks into near/far court with soft labels.

        Priority: court calibration (homography) > team_assignments
        (from tracking pipeline's bbox-size clustering) > court_split_y
        > median Y split.

        Args:
            track_stats: Appearance stats per track.
            player_positions: All player positions for this rally
                (full-rally; the function does its own optional windowing
                for the calibration projection — see ``serve_window_frames``).
            court_split_y: Y coordinate splitting near/far teams.
            team_assignments: Pre-computed track_id -> team (0=near, 1=far)
                from the tracking pipeline's actions_json.teamAssignments.
            serve_window_frames: When set (typically rally 1), restrict
                the calibration-projection input to frames < this value
                to use the clean serve-formation snapshot for the
                authoritative side signal. ``track_avg_y`` is still built
                from full ``player_positions`` so late-arriver tracks
                (those with all positions ≥ the window) are not silently
                dropped — they are classified by the image-Y fallback.

        Returns:
            Tuple of (track_avg_y, track_court_sides) where:
                track_avg_y: track_id -> average Y position
                track_court_sides: track_id -> 0 (near) or 1 (far)

        Contract: every track in ``track_stats`` is guaranteed to appear
        in the returned ``track_court_sides``. An invariant guard at the
        end fills any track that the priority chain missed via image-Y
        fallback and emits a WARNING — repeated firings indicate a hole
        in the priority chain.
        """
        # Compute average Y position for each track from the FULL
        # positions (always — independent of any serve-window restriction).
        # This guarantees every primary track gets an avg_y so late-arrivers
        # can be classified by the image-Y fallback even when their
        # positions all fall outside the serve-formation window.
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

        # Calibration-projection input. When ``serve_window_frames`` is
        # set, restrict to frames < that value (the clean serve-formation
        # snapshot — preferred for the authoritative side signal on
        # rally 1). Otherwise use all positions.
        if serve_window_frames is not None:
            calibration_positions = [
                p for p in player_positions
                if p.frame_number < serve_window_frames
            ]
        else:
            calibration_positions = player_positions

        # Priority 0: Court calibration — project foot positions through
        # homography to get court_y in meters.  Net is at 8.0 m.
        # This is authoritative when available.
        if self.calibrator is not None and self.calibrator.is_calibrated:
            track_court_y: dict[int, list[float]] = {}
            for p in calibration_positions:
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

        # Invariant guard: every track in ``track_stats`` must be
        # classified. The priority chain has historically had holes
        # (e.g. late-arriver tracks under serve-window restriction —
        # b5fb0594 be3134ba 2026-05-07). This guard catches any future
        # gap and falls back to image-Y from track_avg_y, with a WARNING
        # to flag the hole at its source.
        unclassified = set(track_stats.keys()) - set(track_court_sides.keys())
        if unclassified:
            logger.warning(
                "Primary tracks not classified by priority chain: %s. "
                "Falling back to image-Y. This indicates a missing path "
                "in side-classification — investigate if it fires repeatedly.",
                sorted(unclassified),
            )
            for tid in unclassified:
                if tid in track_avg_y:
                    track_court_sides[tid] = (
                        0 if track_avg_y[tid] > 0.5 else 1
                    )
                else:
                    # No avg_y either (track has zero positions in input);
                    # default to near so downstream sees the track.
                    track_court_sides[tid] = 0

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

    def _high_confidence_sides_for_team_pair(
        self,
        track_court_sides: dict[int, int],
        sides_by_bbox: dict[int, int],
    ) -> dict[int, int]:
        """Tracks where y-side and bbox-side AGREE → high-confidence side
        label, suitable for hard team-pair constraint (Phase 1 step 3).

        Only returns the agreement set when it forms a clean 2v2 split.
        Degenerate partitions (3v1, 1v3, 0v2, 2v0) are too risky to
        constrain — Hungarian would either be infeasible or forced into
        nonsense — so we return ``{}`` and let the caller fall through to
        the soft side penalty.
        """
        if not sides_by_bbox or not track_court_sides:
            return {}
        agreed: dict[int, int] = {}
        for tid, y_side in track_court_sides.items():
            bb_side = sides_by_bbox.get(tid)
            if bb_side is not None and bb_side == y_side:
                agreed[tid] = y_side
        near = sum(1 for s in agreed.values() if s == 0)
        far = sum(1 for s in agreed.values() if s == 1)
        if near != 2 or far != 2:
            return {}
        return agreed

    def _assign_tracks_to_players_global(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        track_court_sides: dict[int, int],
        *,
        use_side_penalty: bool = True,
        early_positions: dict[int, tuple[float, float]] | None = None,
        restrict_to_pids: list[int] | None = None,
        track_team_constraint: dict[int, int] | None = None,
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
            restrict_to_pids: Optional list of pids the cost matrix and
                assignment should be restricted to. Default `None` means
                "use all players from `self.state.players`" (preserves
                pre-Task-4 behaviour). When provided (Task 4 dispatch
                after sub-track direct assignment), only these pids
                participate in Hungarian.
            track_team_constraint: Optional ``{track_id: side}`` for
                hard team-pair constraint (Phase 1 step 3). For each
                track in this dict, cross-team cells (track_side !=
                player_side) get cost ``HARD_TEAM_PAIR_COST``,
                structurally forbidding cross-team assignment. Tracks
                NOT in the dict use the soft ``SIDE_PENALTY`` instead.

        Returns:
            track_id -> player_id mapping.
        """
        if not track_ids:
            return {}

        if restrict_to_pids is not None:
            all_player_ids = sorted(restrict_to_pids)
        else:
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

                # Side penalty (player_side already computed above).
                # Hard team-pair constraint (Phase 1 step 3) overrides the
                # soft penalty when the track has a high-confidence side
                # label — cross-team cells become structurally infeasible.
                if (
                    track_team_constraint is not None
                    and tid in track_team_constraint
                    and track_team_constraint[tid] != player_side
                ):
                    side_pen = HARD_TEAM_PAIR_COST
                elif use_side_penalty:
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

    def _maybe_segment_tracks_by_appearance(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        positions: list[PlayerPosition],
        classifier: Any,
        crop_extractor: Callable[[int, int], np.ndarray | None] | None = None,
    ) -> list[SubTrackCandidate]:
        """Stub: pre-Hungarian within-track segmentation.

        Was a flag-gated wrapper around `_segment_tracks_by_appearance` that
        only fired when the video had ref-crop-derived frozen profiles. With
        ref-crops removed in the 2026-05-07 cleanup, the classifier path is
        permanently unreachable and the method returns `[]` unconditionally.
        Kept temporarily so the call site at `_assign_tracks_to_players`
        compiles; the method body and call site are deleted in a later
        phase.
        """
        if os.environ.get("ENABLE_REF_CROP_TRACK_SPLIT", "0") != "1":
            return []
        if classifier is None or not getattr(classifier, "is_trained", False):
            return []
        return []

    @staticmethod
    def _apply_subtrack_assignments(
        sub_tracks: list[SubTrackCandidate],
        top_tracks: list[int],
        all_pids: list[int],
    ) -> tuple[dict[int, int], list[int], list[int]]:
        """Convert sub-track per-segment argmax pids into direct
        track_id -> pid assignments, bypassing Hungarian.

        For each pid claimed by multiple sub-tracks, only the highest-margin
        sub-track wins direct assignment; losers are dropped from `direct`
        (their frames will fall into the per-frame conflict resolver in
        Task 5 and end up unlabeled).

        Args:
            sub_tracks: SubTrackCandidates from `_segment_tracks_by_appearance`.
            top_tracks: All track_ids selected for the rally (real BoT-SORT ids).
            all_pids: All canonical pids (typically [1, 2, 3, 4]).

        Returns:
            direct: dict[synthetic_track_id, pid] for sub-tracks that won.
            remaining_track_ids: real top_tracks with split parents removed.
            remaining_pids: pids NOT claimed by any sub-track in `direct`.
        """
        # Group sub-tracks by claimed pid; pick highest-margin per pid.
        by_pid: dict[int, list[SubTrackCandidate]] = {}
        for s in sub_tracks:
            if s.aggregated_argmax_pid is None:
                continue
            by_pid.setdefault(s.aggregated_argmax_pid, []).append(s)

        direct: dict[int, int] = {}
        for pid, candidates in by_pid.items():
            winner = max(candidates, key=lambda s: s.aggregated_margin or 0.0)
            direct[winner.synthetic_track_id] = pid

        # Parents that contributed any sub-track are removed from real-tracks
        # (the segments now own their frames; the parent is being replaced).
        split_parents = {s.parent_track_id for s in sub_tracks}
        remaining_track_ids = [t for t in top_tracks if t not in split_parents]

        claimed_pids = set(direct.values())
        remaining_pids = [p for p in all_pids if p not in claimed_pids]
        return direct, remaining_track_ids, remaining_pids

    def _resolve_subtrack_pid_conflicts(
        self,
        track_to_player: dict[int, int],
        sub_tracks: list[SubTrackCandidate],
    ) -> dict[int, int]:
        """Stub completed in Task 5 — returns input unchanged for now."""
        return track_to_player

    def _segment_tracks_by_appearance(
        self,
        track_ids: list[int],
        track_stats: dict[int, TrackAppearanceStats],
        positions: list[PlayerPosition],
        classifier: Any,  # PlayerReIDClassifier (Any avoids circular import)
        crop_extractor: Callable[[int, int], np.ndarray | None],
    ) -> list[SubTrackCandidate]:
        """Detect within-track appearance flips using the few-shot classifier.

        For each track:
        1. Sample SEGMENT_NUM_WINDOWS windows across the track's frame range.
        2. Score each window via classifier on SEGMENT_FRAMES_PER_WINDOW crops.
        3. Window argmax_pid + per-window margin.
        4. Flip detected when:
             - argmax_pid changes between adjacent windows AND
             - both pre-flip and post-flip have >= SEGMENT_MIN_CONFIRMING_WINDOWS
               windows agreeing on the same argmax pid AND
             - aggregate per-segment margins >= SEGMENT_MIN_PER_SEGMENT_MARGIN.

        Returns a flat list of SubTrackCandidates. Tracks that don't split
        are absent from the returned list — caller falls back to using the
        original track. Each emitted SubTrackCandidate has its
        `aggregated_argmax_pid` populated, which Task 4 will use for direct
        assignment (bypassing Hungarian).
        """
        if not classifier or not getattr(classifier, "is_trained", False):
            return []

        by_track: dict[int, list[PlayerPosition]] = {}
        for p in positions:
            if p.track_id in track_ids:
                by_track.setdefault(p.track_id, []).append(p)
        for tid in by_track:
            by_track[tid].sort(key=lambda x: x.frame_number)

        sub_tracks: list[SubTrackCandidate] = []

        for tid in track_ids:
            track_positions = by_track.get(tid, [])
            if len(track_positions) < SEGMENT_NUM_WINDOWS * SEGMENT_MIN_WINDOW_FRAMES:
                continue
            f_first = track_positions[0].frame_number
            f_last = track_positions[-1].frame_number
            if f_last - f_first < SEGMENT_NUM_WINDOWS * SEGMENT_MIN_WINDOW_FRAMES:
                continue

            window_bounds = np.linspace(
                f_first, f_last + 1, SEGMENT_NUM_WINDOWS + 1
            ).astype(int)
            window_argmax: list[int | None] = []
            window_margin: list[float] = []
            window_frame_ranges: list[tuple[int, int]] = []

            for w in range(SEGMENT_NUM_WINDOWS):
                w_start = int(window_bounds[w])
                w_end = int(window_bounds[w + 1] - 1)
                window_frame_ranges.append((w_start, w_end))
                window_positions = [
                    p for p in track_positions if w_start <= p.frame_number <= w_end
                ]
                if len(window_positions) < SEGMENT_MIN_WINDOW_FRAMES:
                    window_argmax.append(None)
                    window_margin.append(0.0)
                    continue
                idxs = np.linspace(
                    0, len(window_positions) - 1, SEGMENT_FRAMES_PER_WINDOW
                ).astype(int)
                sample_positions = [window_positions[i] for i in idxs]
                crops: list[np.ndarray] = []
                for p in sample_positions:
                    crop = crop_extractor(tid, p.frame_number)
                    if crop is not None:
                        crops.append(crop)
                if not crops:
                    window_argmax.append(None)
                    window_margin.append(0.0)
                    continue
                probs_list = classifier.predict(crops)
                avg = {
                    pid: float(np.mean([p[pid] for p in probs_list]))
                    for pid in probs_list[0]
                }
                sorted_pids = sorted(avg.items(), key=lambda x: x[1], reverse=True)
                window_argmax.append(sorted_pids[0][0])
                window_margin.append(sorted_pids[0][1] - sorted_pids[1][1])

            # Per-track diagnostic — surface what the splitter sees so users
            # can debug why a track did/didn't split. WARNING level so the
            # default rallycut CLI logging config (which is WARNING) prints
            # these without needing --verbose.
            logger.warning(
                "Track-split window scan: tid=%d  windows=%s",
                tid,
                [
                    (window_frame_ranges[i], window_argmax[i],
                     round(window_margin[i], 3))
                    for i in range(SEGMENT_NUM_WINDOWS)
                ],
            )

            # Try the strict per-window walk first (cheap, exact).
            split_at_window = self._find_segment_flip(window_argmax, window_margin)

            # Fallback: if the strict walk found no flip, try every window
            # boundary as a candidate split and accept the one where BOTH
            # per-segment aggregates clear the margin gate. This handles the
            # common case where the transition window itself has weak/mixed
            # signal (which the strict walk rejects) but pre and post
            # aggregates are clean.
            f_split: int | None = None
            pre_argmax: int | None = None
            post_argmax: int | None = None
            pre_margin: float = 0.0
            post_margin: float = 0.0

            min_seg_frames = SEGMENT_MIN_WINDOW_FRAMES * SEGMENT_MIN_CONFIRMING_WINDOWS

            if split_at_window is not None:
                f_split_candidate = window_frame_ranges[split_at_window][0]
                pre_positions_c = [
                    p for p in track_positions if p.frame_number < f_split_candidate
                ]
                post_positions_c = [
                    p for p in track_positions if p.frame_number >= f_split_candidate
                ]
                if (
                    len(pre_positions_c) >= min_seg_frames
                    and len(post_positions_c) >= min_seg_frames
                ):
                    pa, pm = self._aggregate_segment_classifier(
                        tid, pre_positions_c, classifier, crop_extractor,
                    )
                    qa, qm = self._aggregate_segment_classifier(
                        tid, post_positions_c, classifier, crop_extractor,
                    )
                    if (
                        pa is not None and qa is not None
                        and pa != qa
                        and pm >= SEGMENT_MIN_PER_SEGMENT_MARGIN
                        and qm >= SEGMENT_MIN_PER_SEGMENT_MARGIN
                    ):
                        f_split = f_split_candidate
                        pre_argmax, pre_margin = pa, pm
                        post_argmax, post_margin = qa, qm

            if f_split is None:
                # Fallback: scan every window boundary, pick the boundary
                # where pre+post aggregates are confidently different.
                best: tuple[int, int, int, float, float] | None = None
                for w in range(1, SEGMENT_NUM_WINDOWS):
                    f_split_candidate = window_frame_ranges[w][0]
                    pre_positions_c = [
                        p for p in track_positions if p.frame_number < f_split_candidate
                    ]
                    post_positions_c = [
                        p for p in track_positions if p.frame_number >= f_split_candidate
                    ]
                    if (
                        len(pre_positions_c) < min_seg_frames
                        or len(post_positions_c) < min_seg_frames
                    ):
                        continue
                    pa, pm = self._aggregate_segment_classifier(
                        tid, pre_positions_c, classifier, crop_extractor,
                    )
                    qa, qm = self._aggregate_segment_classifier(
                        tid, post_positions_c, classifier, crop_extractor,
                    )
                    if (
                        pa is None or qa is None
                        or pa == qa
                        or pm < SEGMENT_MIN_PER_SEGMENT_MARGIN
                        or qm < SEGMENT_MIN_PER_SEGMENT_MARGIN
                    ):
                        continue
                    score = pm + qm
                    if best is None or score > best[3] + best[4]:
                        best = (f_split_candidate, pa, qa, pm, qm)
                if best is not None:
                    f_split, pre_argmax, post_argmax, pre_margin, post_margin = best
                    logger.warning(
                        "Track-split fallback boundary scan: tid=%d split@frame=%d "
                        "pre->pid%d (margin %+.3f)  post->pid%d (margin %+.3f)",
                        tid, f_split, pre_argmax, pre_margin,
                        post_argmax, post_margin,
                    )

            if f_split is None:
                logger.warning(
                    "Track-split: tid=%d no split detected "
                    "(strict walk + boundary scan both abstained)",
                    tid,
                )
                continue

            pre_positions = [
                p for p in track_positions if p.frame_number < f_split
            ]
            post_positions = [
                p for p in track_positions if p.frame_number >= f_split
            ]

            parent_stats = track_stats.get(tid)
            if parent_stats is None:
                continue

            pre_stats = self._stats_for_positions(parent_stats, pre_positions, tid)
            post_stats = self._stats_for_positions(parent_stats, post_positions, tid)

            sub_tracks.append(SubTrackCandidate(
                parent_track_id=tid,
                segment_index=0,
                f_start=pre_positions[0].frame_number,
                f_end=pre_positions[-1].frame_number,
                appearance_stats=pre_stats,
                aggregated_argmax_pid=pre_argmax,
                aggregated_margin=pre_margin,
            ))
            sub_tracks.append(SubTrackCandidate(
                parent_track_id=tid,
                segment_index=1,
                f_start=post_positions[0].frame_number,
                f_end=post_positions[-1].frame_number,
                appearance_stats=post_stats,
                aggregated_argmax_pid=post_argmax,
                aggregated_margin=post_margin,
            ))

            logger.warning(
                "Within-track split: tid=%d at frame %d  pre->pid%d "
                "(margin %+.3f, %d frames)  post->pid%d "
                "(margin %+.3f, %d frames)",
                tid, f_split, pre_argmax, pre_margin, len(pre_positions),
                post_argmax, post_margin, len(post_positions),
            )

        return sub_tracks

    @staticmethod
    def _find_segment_flip(
        window_argmax: list[int | None],
        window_margin: list[float],
    ) -> int | None:
        """Find first window index W where:
           - argmax pid at W differs from argmax pid at W-1
           - SEGMENT_MIN_CONFIRMING_WINDOWS windows BEFORE W agree on same pid
           - SEGMENT_MIN_CONFIRMING_WINDOWS windows AT/AFTER W agree on same pid
           - All confirming windows have margin >= SEGMENT_MIN_PER_SEGMENT_MARGIN

        Returns the window index of the flip start, or None.
        """
        n = len(window_argmax)
        if n < 2 * SEGMENT_MIN_CONFIRMING_WINDOWS:
            return None
        for w in range(
            SEGMENT_MIN_CONFIRMING_WINDOWS, n - SEGMENT_MIN_CONFIRMING_WINDOWS + 1
        ):
            pre_pids = window_argmax[max(0, w - SEGMENT_MIN_CONFIRMING_WINDOWS):w]
            post_pids = window_argmax[w:w + SEGMENT_MIN_CONFIRMING_WINDOWS]
            pre_margins = window_margin[max(0, w - SEGMENT_MIN_CONFIRMING_WINDOWS):w]
            post_margins = window_margin[w:w + SEGMENT_MIN_CONFIRMING_WINDOWS]
            if any(p is None for p in pre_pids) or any(p is None for p in post_pids):
                continue
            if len(set(pre_pids)) != 1 or len(set(post_pids)) != 1:
                continue
            if pre_pids[0] == post_pids[0]:
                continue
            if min(pre_margins) < SEGMENT_MIN_PER_SEGMENT_MARGIN:
                continue
            if min(post_margins) < SEGMENT_MIN_PER_SEGMENT_MARGIN:
                continue
            return w
        return None

    @staticmethod
    def _aggregate_segment_classifier(
        track_id: int,
        positions: list[PlayerPosition],
        classifier: Any,
        crop_extractor: Callable[[int, int], np.ndarray | None],
    ) -> tuple[int | None, float]:
        if not positions:
            return None, 0.0
        n_samples = min(8, len(positions))
        idxs = np.linspace(0, len(positions) - 1, n_samples).astype(int)
        crops: list[np.ndarray] = []
        for i in idxs:
            crop = crop_extractor(track_id, positions[i].frame_number)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None, 0.0
        probs_list = classifier.predict(crops)
        avg = {
            pid: float(np.mean([p[pid] for p in probs_list]))
            for pid in probs_list[0]
        }
        sorted_pids = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return sorted_pids[0][0], sorted_pids[0][1] - sorted_pids[1][1]

    @staticmethod
    def _stats_for_positions(
        parent_stats: TrackAppearanceStats,
        positions: list[PlayerPosition],
        track_id: int,
    ) -> TrackAppearanceStats:
        """Build a per-segment TrackAppearanceStats. Inherits the parent's
        avg_* fields and reid_embedding verbatim. Filters the `features`
        list to frames in this segment's range.

        Verbatim inheritance is acceptable because Task 4 bypasses Hungarian
        for sub-tracks (assigning them directly to their aggregated_argmax_pid),
        so per-segment HSV/ReID re-extraction would be wasted work in v1.
        """
        seg_frames = {p.frame_number for p in positions}
        seg_features = [
            f for f in parent_stats.features if f.frame_number in seg_frames
        ]
        return TrackAppearanceStats(
            track_id=track_id,
            features=seg_features,
            avg_skin_tone_hsv=parent_stats.avg_skin_tone_hsv,
            avg_upper_hist=parent_stats.avg_upper_hist,
            avg_lower_hist=parent_stats.avg_lower_hist,
            avg_upper_v_hist=parent_stats.avg_upper_v_hist,
            avg_lower_v_hist=parent_stats.avg_lower_v_hist,
            avg_dominant_color_hsv=parent_stats.avg_dominant_color_hsv,
            avg_head_hist=parent_stats.avg_head_hist,
            reid_embedding=parent_stats.reid_embedding,
        )

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

            # Update ReID embedding (per-track average, not per-sample).
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

    def _select_seed_rally(self) -> int:
        """Pick the rally to anchor the canonical within-team pid layout.

        Rally 0's Y-sort seed is a single point of failure: a poorly-classified
        rally 0 (compressed Y, missing calibration, near-tie within-team Y)
        propagates the wrong pid layout to every subsequent rally. This picks
        the highest-quality rally — one with strong, mutually-consistent side
        classification signals — and uses its Y-sort as the canonical anchor.

        Quality score per rally:
          +1 if 4 distinct top tracks present
          +2 if 2 near + 2 far (well-balanced 2v2 from track_court_sides)
          +1 if sides_from_calibration is True
          +1 if sides_by_bbox agrees with track_court_sides on every top track
          +1 if early_positions populated (required for Y-sort recompute)
          +(-1 if early_positions absent — we cannot recompute Y-sort there)

        Returns:
            Index of the seed rally. Falls back to 0 when:
              - Fewer than 2 stored rallies.
              - early_positions empty on every rally (scratchpad replay path).
              - No non-zero rally clearly outscores rally 0 (margin < 1.0).
        """
        if len(self.stored_rally_data) < 2:
            return 0
        scores: list[float] = []
        for d in self.stored_rally_data:
            s = 0.0
            top4 = list(d.top_tracks[:4])
            if len(set(top4)) == 4:
                s += 1.0
            sides_top = [d.track_court_sides.get(t) for t in top4]
            if sides_top.count(0) == 2 and sides_top.count(1) == 2:
                s += 2.0
            if d.sides_from_calibration:
                s += 1.0
            if d.sides_by_bbox and top4:
                agreements = sum(
                    1 for t in top4
                    if t in d.sides_by_bbox
                    and t in d.track_court_sides
                    and d.sides_by_bbox[t] == d.track_court_sides[t]
                )
                if agreements == 4:
                    s += 1.0
            if d.early_positions:
                s += 1.0
            else:
                # Recomputing Y-sort needs early_positions. Strongly disprefer.
                s -= 1.0
            scores.append(s)
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        if best_idx == 0:
            return 0
        # Require a meaningful margin over rally 0 to switch anchors.
        if scores[best_idx] - scores[0] < 1.0:
            return 0
        return best_idx

    def _within_team_permutation_from_seed(
        self,
        seed_idx: int,
        seed_assignment: dict[int, int],
    ) -> dict[int, int]:
        """Compute the within-team pid permutation implied by re-anchoring.

        Given the seed rally's current Pass-1 ``track_to_player`` mapping,
        compute what the Y-sorted-within-side mapping WOULD be at that rally
        if we re-ran ``_initialize_first_rally`` there. The permutation is
        the relabeling that turns the current pids into the canonical (seed
        Y-sort) pids. Only within-team swaps are permitted — team membership
        is determined by side classification, not Y-sort.

        Returns:
            ``{old_pid: new_pid}`` over {1,2,3,4}. Identity when no relabel
            is needed (seed's current mapping already matches Y-sort).
        """
        identity = {pid: pid for pid in range(1, 5)}
        if not (0 <= seed_idx < len(self.stored_rally_data)):
            return identity
        data = self.stored_rally_data[seed_idx]
        if not data.early_positions:
            return identity
        # Build pid → (side, y) at the seed using the CURRENT track→pid layout.
        pid_to_side_y: dict[int, tuple[int, float]] = {}
        for tid, pid in seed_assignment.items():
            side = data.track_court_sides.get(tid)
            xy = data.early_positions.get(tid)
            if side is None or xy is None:
                continue
            pid_to_side_y[pid] = (side, float(xy[1]))
        if len(pid_to_side_y) != 4:
            return identity
        perm: dict[int, int] = dict(identity)
        for team in (0, 1):
            team_pids = sorted(
                pid for pid, (s, _) in pid_to_side_y.items() if s == team
            )
            if len(team_pids) != 2:
                # Degenerate side classification at the seed — abandon.
                return identity
            # Y-sort: smaller y = upper in frame = first pid in sorted team.
            y_sorted = sorted(team_pids, key=lambda p: pid_to_side_y[p][1])
            # Map y_sorted[0] → team_pids[0]; y_sorted[1] → team_pids[1].
            perm[y_sorted[0]] = team_pids[0]
            perm[y_sorted[1]] = team_pids[1]
        return perm

    def _apply_within_team_permutation(
        self,
        perm: dict[int, int],
        results: list[RallyTrackingResult],
    ) -> list[RallyTrackingResult]:
        """Apply ``{old_pid: new_pid}`` globally across state + results.

        Permutes player profiles, current_side_assignment, current_assignments,
        each rally's player_side_assignment snapshot, and each result's
        track_to_player. Identity perm short-circuits.
        """
        if all(perm.get(pid, pid) == pid for pid in range(1, 5)):
            return results
        # Player profiles
        new_players: dict[int, PlayerAppearanceProfile] = {}
        for old_pid, profile in self.state.players.items():
            new_pid = perm.get(old_pid, old_pid)
            profile.player_id = new_pid
            new_players[new_pid] = profile
        self.state.players = new_players
        # Live assignments
        self.state.current_side_assignment = {
            perm.get(pid, pid): team
            for pid, team in self.state.current_side_assignment.items()
        }
        self.state.current_assignments = {
            tid: perm.get(pid, pid)
            for tid, pid in self.state.current_assignments.items()
        }
        # Per-rally side snapshots
        for d in self.stored_rally_data:
            d.player_side_assignment = {
                perm.get(pid, pid): team
                for pid, team in d.player_side_assignment.items()
            }
        # Pass-1 results
        new_results = [
            RallyTrackingResult(
                rally_index=r.rally_index,
                track_to_player={
                    tid: perm.get(pid, pid)
                    for tid, pid in r.track_to_player.items()
                },
                server_player_id=(
                    perm.get(r.server_player_id, r.server_player_id)
                    if r.server_player_id is not None else None
                ),
                side_switch_detected=r.side_switch_detected,
                assignment_confidence=r.assignment_confidence,
                sub_tracks=r.sub_tracks,
            )
            for r in results
        ]
        return new_results

    def _identity_first_partial_pass(
        self,
        results: list[RallyTrackingResult],
    ) -> list[RallyTrackingResult]:
        """Bug D fix (2026-05-04): gallery-anchored re-assignment for
        rallies with fewer than 4 primary tracks (partial cardinality).

        Triggers per-rally only when N<4 primary tracks. For each visible
        track, computes appearance distance against every per-PID profile
        in `self.state.players` (the cross-rally gallery), then runs a
        rectangular Hungarian over [N tracks × 4 PIDs]. Output is a
        sparse track→PID mapping; the 4-N PIDs the matcher cannot
        identify in this rally are explicitly absent from the result.

        Why this is needed: when a near-side server physically occludes
        their partner for the entire rally, only 3 tracks reach the
        primary-tracks filter. The default 4×4 path's formation-seed
        Y-sort produces a degenerate 1v2 partition; the side-penalty
        and position-continuity costs then bias the visible 3 tracks'
        assignments away from gallery-best. Identity-first removes both
        biases — every visible track is identified by its cross-rally
        appearance signature alone.

        Conservative gates:
        - Flag enabled (`ENABLE_IDENTITY_FIRST_MATCHING=1`).
        - Rally has 1 ≤ N < 4 primary tracks with track_stats.
        - At least IDENTITY_FIRST_MIN_ANCHORS (3) per-PID profiles in
          state.players, each with valid appearance features.

        When any gate fails the rally falls through unchanged.
        """
        if not ENABLE_IDENTITY_FIRST_PARTIAL:
            return results
        if len(self.state.players) < IDENTITY_FIRST_MIN_ANCHORS:
            return results
        # All 4 PIDs must be present in the gallery for a valid
        # rectangular Hungarian. Sparse galleries (e.g., only PIDs 1+2
        # learned so far in a 2-rally bootstrap) can't safely identify
        # which of the 4 the visible tracks are.
        gallery_pids = sorted(self.state.players.keys())
        if gallery_pids != [1, 2, 3, 4]:
            return results

        if len(self.stored_rally_data) != len(results):
            return results

        new_results: list[RallyTrackingResult] = []
        changes = 0
        for i, (data, result) in enumerate(
            zip(self.stored_rally_data, results)
        ):
            top = [t for t in data.top_tracks if t in data.track_stats]
            if not (1 <= len(top) < 4):
                new_results.append(result)
                continue

            # Build cost matrix [N tracks × 4 PIDs]. HSV+ReID blend mirrors
            # `_assign_tracks_to_players_global` (production primitive),
            # but without side-penalty or position-continuity — pure
            # appearance-against-gallery.
            n_tracks = len(top)
            n_pids = 4
            cost_matrix = np.full((n_tracks, n_pids), 1.0, dtype=np.float32)
            for ti, tid in enumerate(top):
                stats = data.track_stats[tid]
                for pj, pid in enumerate(gallery_pids):
                    profile = self.state.players[pid]
                    hsv_cost = compute_appearance_similarity(profile, stats)
                    track_emb = stats.reid_embedding
                    profile_emb = profile.reid_embedding
                    if (
                        track_emb is not None
                        and profile_emb is not None
                        and profile_emb.shape == track_emb.shape
                    ):
                        reid_cost = 1.0 - float(np.dot(profile_emb, track_emb))
                        appearance_cost = (
                            reid_cost * REID_BLEND
                            + hsv_cost * (1 - REID_BLEND)
                        )
                    else:
                        appearance_cost = hsv_cost
                    cost_matrix[ti, pj] = appearance_cost

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            new_track_to_player: dict[int, int] = {}
            for r, c in zip(row_ind, col_ind):
                if r < n_tracks and c < n_pids:
                    new_track_to_player[top[r]] = gallery_pids[c]

            if new_track_to_player == result.track_to_player:
                new_results.append(result)
                continue

            # Recompute confidence under the new assignment.
            confidence = self._compute_assignment_confidence(
                data.track_stats, new_track_to_player,
            )
            logger.info(
                "Identity-first partial pass: rally %d N=%d top=%s "
                "old=%s new=%s conf=%.2f",
                i, n_tracks, top, result.track_to_player,
                new_track_to_player, confidence,
            )
            new_results.append(RallyTrackingResult(
                rally_index=result.rally_index,
                track_to_player=new_track_to_player,
                server_player_id=result.server_player_id,
                side_switch_detected=result.side_switch_detected,
                assignment_confidence=confidence,
                sub_tracks=result.sub_tracks,
            ))
            changes += 1

        if changes:
            logger.info(
                "Identity-first partial pass changed %d/%d partial rallies",
                changes, sum(
                    1 for d in self.stored_rally_data
                    if 1 <= len([t for t in d.top_tracks if t in d.track_stats]) < 4
                ),
            )
        return new_results

    def _post_switch_consensus_pass(
        self,
        results: list[RallyTrackingResult],
        switches: list[int],
    ) -> list[RallyTrackingResult]:
        """Bug C fix (2026-05-04): snap outlier rallies near side-switch
        boundaries to the cross-rally team-partition consensus.

        At side-switch boundaries the per-rally Hungarian sometimes produces
        an outlier permutation due to viewpoint-dependent appearance
        (uniforms look different from front vs back). Surrounding rallies
        converge on a stable team partition, but the boundary rally has a
        deviation that self-corrects within 1-2 rallies. This pass detects
        such outliers and applies the cross-team permutation that snaps
        them to the consensus.

        Conservative gates (all must pass to fire on rally i):
        1. Rally i has exactly 4 primary tracks classified into clean
           2-near + 2-far covering all of {1,2,3,4}.
        2. The "expected" partition is unambiguous, sourced as either:
           (a) ``sideSwitchDetected[i]`` is True AND rally i+1 has a valid
               partition (post-switch reference); OR
           (b) Both i-1 and i+1 have valid partitions that agree with
               each other AND disagree with rally i (consensus
               disagreement).
        3. The implied permutation is a valid bijection over {1,2,3,4}.
        4. ``DISABLE_POST_SWITCH_CONSENSUS=1`` env var is not set
           (emergency rollback).

        The cross-team partition is determined by ``(actual → expected)``;
        within each team there are 2 valid pairings (sorted vs reversed),
        yielding 4 total candidate permutations. We score each candidate
        by track→PID alignment with nearby stable rallies (those with the
        same `expected` partition) and pick the highest-scoring perm.
        This recovers within-team rank correctly when BoT-SORT track IDs
        persist across rallies (the common case). When they don't, the
        scoring falls back to lower hit counts and may pick the same
        perm as sorted-pairing — safe default.

        Known limitations:
        - Detection requires the per-rally team partition (near_pids vs
          far_pids) to disagree with neighbors. When the side classifier
          itself returns inverted/wrong sides at a switch boundary
          (e.g. cross-team grouping), the partition computation produces
          a "matching" partition by coincidence and the outlier isn't
          detected. Catching those cases would require cross-rally
          appearance comparison or AFM-equality across rallies, which
          have their own false-positive risks (BoT-SORT track ID
          renumbering across rallies).
        - When `track_court_sides` is degenerate (3:1 or 1:3 split,
          common at switch boundaries due to transition geometry),
          falls back to Y-sort over `early_positions` to derive a clean
          2v2 partition. Falls through to no-op if even that's
          ambiguous.

        Args:
            results: Per-rally results post Stages 1+2 (or post-MatchSolver
                in the blind path).
            switches: Rally indices where side-switch was detected (only
                used to decide the expected-partition source).

        Returns:
            Results with outlier rallies snapped to consensus. Other
            rallies returned unchanged. Refuses to fire on any ambiguous
            case.
        """
        if os.environ.get("DISABLE_POST_SWITCH_CONSENSUS") == "1":
            return results
        if len(results) < 3 or len(self.stored_rally_data) != len(results):
            return results

        # Step 1: per-rally near-team partition (frozenset of near-side PIDs).
        # Returns None for rallies that don't have a clean 4-player 2-near +
        # 2-far layout — those are skipped from both source and target.
        partitions: list[frozenset[int] | None] = []
        for i, data in enumerate(self.stored_rally_data):
            afm = results[i].track_to_player
            sides = data.track_court_sides or {}
            near = frozenset(
                int(pid) for tid, pid in afm.items()
                if int(tid) > 0 and sides.get(int(tid)) == 0 and int(pid) > 0
            )
            far = frozenset(
                int(pid) for tid, pid in afm.items()
                if int(tid) > 0 and sides.get(int(tid)) == 1 and int(pid) > 0
            )
            if not (
                len(near) == 2 and len(far) == 2
                and (near | far) == {1, 2, 3, 4}
            ):
                # Fallback: when track_court_sides is degenerate (3:1 or 1:3
                # split, common at side-switch boundaries when players are
                # transitioning across the net), use Y-sort over
                # early_positions to derive a clean 2v2 partition. Top 2 in
                # Y (lower-on-screen = closer to camera) = near team.
                primary_pids = [
                    int(pid) for tid, pid in afm.items()
                    if int(tid) > 0 and 1 <= int(pid) <= 4
                ]
                if len(set(primary_pids)) == 4:
                    by_y: list[tuple[int, int, float]] = []  # (tid, pid, avg_y)
                    for tid, pid in afm.items():
                        tid_int, pid_int = int(tid), int(pid)
                        if tid_int <= 0 or not (1 <= pid_int <= 4):
                            continue
                        pos = data.early_positions.get(tid_int)
                        if pos is None:
                            by_y = []
                            break
                        by_y.append((tid_int, pid_int, float(pos[1])))
                    if len(by_y) == 4:
                        # Sort ascending Y (smaller Y = top of image = far court).
                        by_y.sort(key=lambda x: x[2])
                        far = frozenset(p[1] for p in by_y[:2])
                        near = frozenset(p[1] for p in by_y[2:])
            if (
                len(near) == 2 and len(far) == 2
                and (near | far) == {1, 2, 3, 4}
            ):
                partitions.append(near)
            else:
                partitions.append(None)

        switch_set = set(switches)
        n_changes = 0

        for i in range(len(partitions)):
            actual = partitions[i]
            if actual is None:
                continue

            # Step 2: determine the unambiguous expected partition.
            expected: frozenset[int] | None = None
            reason = ""

            if i in switch_set:
                # Switch boundary: post-switch reference is the next stable
                # rally. Look ahead up to 2 rallies for one with a valid
                # partition (in case rally i+1 itself is degenerate).
                for j in range(i + 1, min(i + 3, len(partitions))):
                    if partitions[j] is not None:
                        expected = partitions[j]
                        reason = f"sideSwitchDetected[{i}]; expected from rally {j}"
                        break

            if expected is None:
                # Non-switch rally: require both immediate neighbors to have
                # valid partitions that AGREE with each other.
                prev_idx = i - 1
                next_idx = i + 1
                if prev_idx < 0 or next_idx >= len(partitions):
                    continue
                prev = partitions[prev_idx]
                nxt = partitions[next_idx]
                if prev is None or nxt is None or prev != nxt:
                    continue
                expected = prev
                reason = (
                    f"neighbors {prev_idx} and {next_idx} agree on partition"
                )

            if expected == actual:
                continue  # no disagreement → no change needed

            # Step 3: build the cross-team permutation. The team partition
            # is fixed by `(actual → expected)`; within each team there are
            # 2 valid pairings (sorted vs reversed), so 4 candidate perms.
            # Pick the one that maximizes track→PID alignment with nearby
            # stable rallies — this gets the within-team rank right when
            # BoT-SORT track IDs persist across rallies (the common case).
            actual_near_l = sorted(actual)
            expected_near_l = sorted(expected)
            actual_far_l = sorted({1, 2, 3, 4} - actual)
            expected_far_l = sorted({1, 2, 3, 4} - expected)
            if len(actual_near_l) != 2 or len(expected_near_l) != 2:
                continue

            # Reference AFMs: rallies within ±2 whose partition matches
            # `expected` (i.e., agrees with the consensus). Excludes the
            # outlier itself by partition mismatch.
            reference_afms: list[dict[int, int]] = []
            for j in range(max(0, i - 2), min(len(partitions), i + 3)):
                if j == i or partitions[j] != expected:
                    continue
                reference_afms.append({
                    int(tid): int(pid)
                    for tid, pid in results[j].track_to_player.items()
                    if int(tid) > 0 and 1 <= int(pid) <= 4
                })

            def _score_perm(p: dict[int, int]) -> int:
                """Count post-perm AFM track→PID assignments matching any
                reference rally's AFM at the SAME track ID."""
                if not reference_afms:
                    return 0
                hits = 0
                for tid, old_pid in results[i].track_to_player.items():
                    tid_int = int(tid)
                    if tid_int <= 0 or not (1 <= int(old_pid) <= 4):
                        continue
                    new_pid = p.get(int(old_pid), int(old_pid))
                    for ref in reference_afms:
                        if ref.get(tid_int) == new_pid:
                            hits += 1
                            break
                return hits

            best_perm: dict[int, int] | None = None
            best_score = -1
            for near_pairing in (
                list(zip(actual_near_l, expected_near_l)),
                list(zip(actual_near_l, list(reversed(expected_near_l)))),
            ):
                for far_pairing in (
                    list(zip(actual_far_l, expected_far_l)),
                    list(zip(actual_far_l, list(reversed(expected_far_l)))),
                ):
                    cand: dict[int, int] = {}
                    for a, e in near_pairing:
                        cand[a] = e
                    for a, e in far_pairing:
                        cand[a] = e
                    if (
                        set(cand.keys()) != {1, 2, 3, 4}
                        or set(cand.values()) != {1, 2, 3, 4}
                    ):
                        continue
                    s = _score_perm(cand)
                    if s > best_score:
                        best_score = s
                        best_perm = cand
            if best_perm is None:
                continue
            perm = best_perm

            # Step 4: apply the permutation to rally i's result.
            old = results[i]
            new_track_to_player = {
                tid: perm.get(int(pid), int(pid))
                for tid, pid in old.track_to_player.items()
            }
            new_server = old.server_player_id
            if new_server is not None:
                new_server = perm.get(int(new_server), int(new_server))
            # SubTrackCandidate carries no pid field; the pid is stored in
            # track_to_player keyed on synthetic_track_id, which we already
            # permuted above. Sub-tracks pass through unchanged.
            results[i] = RallyTrackingResult(
                rally_index=old.rally_index,
                track_to_player=new_track_to_player,
                server_player_id=new_server,
                side_switch_detected=old.side_switch_detected,
                assignment_confidence=old.assignment_confidence,
                sub_tracks=old.sub_tracks,
            )
            n_changes += 1
            logger.info(
                "post-switch consensus snap: rally %d near_pids %s → %s "
                "(perm=%s; %s)",
                i, sorted(actual), sorted(expected), perm, reason,
            )

        if n_changes > 0:
            logger.info(
                "post-switch consensus pass corrected %d rally/rallies",
                n_changes,
            )
        return results

    def refine_assignments(
        self,
        initial_results: list[RallyTrackingResult],
        skip_stages_1_and_2: bool = False,
    ) -> list[RallyTrackingResult]:
        """Re-score all rallies using final profiles + global within-team voting.

        Three-stage Pass 2:
        0. Combinatorial side switch detection using ball trajectory direction
        1. Re-run cross-team assignment with final profiles
        2. Global within-team pairwise voting

        Args:
            initial_results: Results from Pass 1 forward pass.
            skip_stages_1_and_2: When True (blind path under
                ``MatchSolver``), run only Stage 0 (side-switch detection)
                and the canonical re-anchor; skip the profile-based
                re-Hungarian (Stage 1) and the global within-team voting
                (Stage 2) since the solver already produced them.

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
        # Expose the partition so the relabel scratchpad can carry it forward.
        self.last_side_switches = sorted(switch_set)
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
                        sub_tracks=r.sub_tracks,
                    )

        # Re-anchor canonical pid layout from the highest-quality rally
        # (Phase 1 step 2).
        seed_idx = self._select_seed_rally()
        if seed_idx > 0 and seed_idx < len(initial_results):
            seed_perm = self._within_team_permutation_from_seed(
                seed_idx, initial_results[seed_idx].track_to_player,
            )
            if any(seed_perm.get(p, p) != p for p in range(1, 5)):
                logger.info(
                    "Pass 2 re-anchoring canonical pid layout from "
                    "rally %d (Y-sort seed): perm=%s",
                    seed_idx, seed_perm,
                )
                initial_results = self._apply_within_team_permutation(
                    seed_perm, initial_results,
                )

        # Blind-path short-circuit: solver replaces Stages 1 and 2.
        if skip_stages_1_and_2:
            # Stage 3 (Bug C fix) still runs in the blind path — it acts on
            # whichever per-rally AFMs reached this point.
            after_consensus = self._post_switch_consensus_pass(
                initial_results, sorted(switch_set),
            )
            # Stage 4 (Bug D fix, default OFF): identity-first re-assign for
            # partial-cardinality rallies. No-op when flag disabled.
            return self._identity_first_partial_pass(after_consensus)

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
            #
            # Hard team-pair constraint (Phase 1 step 3): forbid cross-team
            # assignments for tracks whose y-side and bbox-side agree AND
            # the agreement set forms a clean 2v2. Empty when partition is
            # degenerate; soft side penalty governs in that case.
            team_constraint = self._high_confidence_sides_for_team_pair(
                data.track_court_sides, data.sides_by_bbox,
            )

            # Sub-track-aware path: when Pass 1 detected within-track splits
            # (`initial.sub_tracks` non-empty), apply their direct pid claims
            # before running Hungarian on the remaining real tracks. Without
            # this, Stage 1 silently drops sub-tracks (they're not in the
            # rebuilt track_to_player) and downstream `remap-track-ids` loses
            # the frame-conditional remap.
            if initial.sub_tracks:
                all_pids = sorted(self.state.players.keys())
                direct, remaining_tracks, remaining_pids = (
                    self._apply_subtrack_assignments(
                        initial.sub_tracks, data.top_tracks, all_pids,
                    )
                )
                if remaining_tracks and remaining_pids:
                    hungarian_result = self._assign_tracks_to_players_global(
                        remaining_tracks,
                        data.track_stats,
                        data.track_court_sides,
                        use_side_penalty=True,
                        restrict_to_pids=remaining_pids,
                        track_team_constraint=team_constraint or None,
                    )
                else:
                    hungarian_result = {}
                track_to_player = {**direct, **hungarian_result}
            else:
                track_to_player = self._assign_tracks_to_players_global(
                    data.top_tracks,
                    data.track_stats,
                    data.track_court_sides,
                    use_side_penalty=True,
                    track_team_constraint=team_constraint or None,
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
                sub_tracks=initial.sub_tracks,
            ))

        if changes:
            logger.info("Pass 2 stage 1 changed %d/%d rallies", changes, len(refined))

        # Stage 2: Global within-team voting using raw track comparisons
        refined = self._global_within_team_voting(refined)

        # Stage 3 (Bug C fix): post-switch consensus pass. Snaps outlier
        # rallies near side-switch boundaries to the surrounding consensus
        # team partition. See `_post_switch_consensus_pass` docstring.
        refined = self._post_switch_consensus_pass(refined, sorted(switch_set))

        # Stage 4 (Bug D fix, default OFF): identity-first re-assign for
        # partial-cardinality rallies. See `_identity_first_partial_pass`.
        refined = self._identity_first_partial_pass(refined)

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
                        sub_tracks=result.sub_tracks,
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

    # Quality filter: drop bad-frame samples before they pollute the
    # per-track aggregated histograms. Without this, the average HSV
    # signal is contaminated by occluded / cropped / partial-player
    # frames (visual feedback 2026-05-02 on 5c756c41), making per-track
    # features non-discriminative across rallies even when players are
    # visually distinct. We filter on:
    #   1. Aspect ratio: full-body player bboxes are tall (h/w >= 1.4).
    #      Squashed bboxes (zoomed arm, mid-jump w/ legs out) are noise.
    #   2. Detection confidence (>= 0.5).
    #   3. Occlusion: max IoU with ANY OTHER primary track's bbox at the
    #      same frame (>= 0.3 = significant overlap → player obscured).
    # If fewer than 4 clean positions remain after filtering, fall back
    # to ANY positions (better some signal than none).
    bbox_min_aspect_ratio = 1.4
    bbox_min_confidence = 0.5
    bbox_max_occlusion_iou = 0.3

    # Build a per-frame index of all primary-track bboxes for IoU lookup.
    primary_by_frame: dict[int, list[tuple[int, PlayerPosition]]] = {}
    for p in positions:
        if p.track_id in primary_set:
            primary_by_frame.setdefault(p.frame_number, []).append((p.track_id, p))

    def _bbox_iou(a: PlayerPosition, b: PlayerPosition) -> float:
        ax1, ay1 = a.x - a.width / 2, a.y - a.height / 2
        ax2, ay2 = a.x + a.width / 2, a.y + a.height / 2
        bx1, by1 = b.x - b.width / 2, b.y - b.height / 2
        bx2, by2 = b.x + b.width / 2, b.y + b.height / 2
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return float(inter / union) if union > 0 else 0.0

    def _is_clean(tid: int, p: PlayerPosition) -> bool:
        # Aspect ratio
        if p.width <= 0 or p.height <= 0:
            return False
        if (p.height / p.width) < bbox_min_aspect_ratio:
            return False
        # Confidence
        if (p.confidence or 0.0) < bbox_min_confidence:
            return False
        # Occlusion: IoU with any OTHER primary track at this frame
        other_in_frame = primary_by_frame.get(p.frame_number, [])
        for other_tid, other_p in other_in_frame:
            if other_tid == tid:
                continue
            if _bbox_iou(p, other_p) >= bbox_max_occlusion_iou:
                return False
        return True

    # For each track, pick evenly-spaced sample frames from the CLEAN subset.
    frame_requests: dict[int, list[tuple[int, PlayerPosition]]] = {}
    for tid, pos_list in track_positions.items():
        pos_list.sort(key=lambda p: p.frame_number)
        clean_pos = [p for p in pos_list if _is_clean(tid, p)]
        if len(clean_pos) >= 4:
            source = clean_pos
        else:
            # Too few clean frames — fall back to all positions so the
            # track still contributes some signal.
            logger.debug(
                "extract_rally_appearances: track %d has only %d/%d clean "
                "frames; falling back to all positions",
                tid, len(clean_pos), len(pos_list),
            )
            source = pos_list

        n = len(source)
        if n <= num_samples:
            sample_indices = list(range(n))
        else:
            sample_indices = [
                int(i * (n - 1) / (num_samples - 1)) for i in range(num_samples)
            ]

        for idx in sample_indices:
            p = source[idx]
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

    # Pose-anchored mode: feature flag dispatches to pose-keypoint polygon masks
    # + body-proportion ratios (Workstream 1+2, 2026-04-29). Default: off.
    from rallycut.tracking.pose_anchored_features import (
        extract_pose_anchored_features,
        is_pose_anchored_enabled,
        populate_track_body_proportions,
        run_pose_on_frame,
    )
    pose_anchored = is_pose_anchored_enabled()
    pose_model = None
    if pose_anchored:
        try:
            from rallycut.tracking.pose_anchored_features import get_pose_model
            pose_model = get_pose_model()
            logger.info("extract_rally_appearances: pose-anchored mode ON")
        except Exception:  # noqa: BLE001
            logger.warning(
                "extract_rally_appearances: pose model load failed, "
                "falling back to legacy extraction", exc_info=True,
            )
            pose_anchored = False

    # Per-track per-frame body-proportion samples (only populated under pose mode).
    body_props_per_track: dict[int, list[dict[str, float] | None]] = {
        tid: [] for tid in track_positions
    }

    try:
        for fn in sorted_frames:
            abs_frame = start_frame + fn
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_arr = np.asarray(frame, dtype=np.uint8)

            # Pose-anchored: run YOLO-pose ONCE per frame for all tracks at that frame.
            pose_for_frame = None
            if pose_anchored:
                bboxes_norm_by_tid: dict[
                    int, tuple[float, float, float, float],
                ] = {}
                for tid, p in frame_requests[fn]:
                    cx, cy = float(p.x), float(p.y)
                    w, h = float(p.width), float(p.height)
                    bboxes_norm_by_tid[tid] = (
                        cx - w / 2, cy - h / 2,
                        cx + w / 2, cy + h / 2,
                    )
                pose_for_frame = run_pose_on_frame(
                    frame_arr, bboxes_norm_by_tid, pose_model,
                )

            for tid, p in frame_requests[fn]:
                bbox = (p.x, p.y, p.width, p.height)
                if pose_anchored and pose_for_frame is not None:
                    pose_data = pose_for_frame.by_track.get(tid)
                    features, body_props = extract_pose_anchored_features(
                        frame_arr, tid, fn, bbox, frame_width, frame_height,
                        pose_data,
                    )
                    body_props_per_track[tid].append(body_props)
                else:
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

    # Aggregate per-track body-proportion medians (pose-anchored mode only).
    if pose_anchored:
        for tid, samples in body_props_per_track.items():
            if tid in stats:
                populate_track_body_proportions(stats[tid], samples)

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
                    stats[tid].reid_embedding = _aggregate_reid_embeddings(embeddings)
            else:
                # Per-video: raw DINOv2 backbone features (384-dim)
                from rallycut.tracking.reid_embeddings import extract_backbone_features

                for tid, crops in reid_crops.items():
                    if not crops or tid not in stats:
                        continue
                    embeddings = extract_backbone_features(crops)
                    stats[tid].reid_embedding = _aggregate_reid_embeddings(embeddings)
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
    # Per-rally + match-level state needed to replay Pass 2 stages 1+2
    # under new frozen profiles. Populated by match_players_across_rallies
    # after refine_assignments. See scratchpad_to_dict() for the shape.
    scratchpad: dict[str, Any] = field(default_factory=dict)
    # Per-rally hash of MatchSolver pre-solve state. Populated only when
    # the blind path runs (MatchSolver path). Consumed by the CLI to
    # write `assignmentAnchor` into match_analysis_json so subsequent
    # blind runs can pin per-rally assignments and avoid cross-rally
    # cascade through `_build_appearance_cost`. See
    # `ENABLE_ASSIGNMENT_ANCHORS` and `_stored_rally_data_hash`.
    track_stats_hashes: dict[str, str] = field(default_factory=dict)
    # Cache-hit summary: number of rallies whose assignment was pinned
    # from a prior anchor (vs re-solved by MatchSolver). Useful for
    # surfacing in the CLI output and for monitoring cache health.
    anchor_cache_hits: int = 0
    anchor_cache_total: int = 0


# Confidence threshold for persisting an `assignmentAnchor`. Below this,
# the rally is allowed to re-solve every run (fresh attempt with new
# cross-rally context) instead of locking in a low-confidence decision.
# Lower than `MIN_PROFILE_UPDATE_CONFIDENCE = 0.80` because anchors only
# affect THIS rally's decision (not downstream profiles), but high enough
# to skip clearly-uncertain assignments. Tunable.
ANCHOR_MIN_CONFIDENCE = 0.50

# Matcher version stamp written into every persisted `assignmentAnchor`.
# The anchor cache pins a prior rally's assignment when its
# `trackStatsHash` matches today's input fingerprint — but trackStatsHash
# captures only the INPUT to the matcher, not the matcher's logic itself.
# Without a version stamp, any matcher fix (e.g. seed-init bug, cost-blend
# tuning, side-classifier change) is silently bypassed for already-anchored
# rallies, requiring `--reset-anchors` to manifest.
#
# Bump this any time MatchSolver, _init_from_seed, _build_appearance_cost,
# _classify_track_sides, or any other code that affects the per-rally
# assignment changes meaningfully. The next run will then invalidate stale
# anchors automatically. The string is opaque — anything unique works.
#
# History:
#   "v1" — initial (post commit `20042b7`).
#   "v2" — 2026-05-02: bumped after `70ba038` (team-partitioned seed
#          Y-sort + medoid ReID aggregation) and the matcher_version
#          mechanism itself shipped. Old "v1" anchors invalidate on read.
#   "v3" — 2026-05-03: multi-signal team-pair partition determination.
#          Replaces unanimity gate with candidate-partition enumeration
#          (each side signal proposes its own 2v2; cost-decided
#          Hungarian picks the lowest). Fixes silent partition violations
#          on rallies where y-side and bbox-side disagreed even on a
#          single track. Old "v2" anchors invalidate on read so the
#          fix actually takes effect on previously-anchored rallies.
#   "v4" — 2026-05-04: post-switch consensus pass (Bug C).
#          Adds a final stage to refine_assignments that snaps outlier
#          rallies near side-switch boundaries to the cross-rally team-
#          partition consensus. Fires conservatively (only when
#          surrounding rallies form clear consensus AND the rally has
#          exactly 4 clean primary tracks). Old "v3" anchors invalidate
#          on read so the fix takes effect immediately.
#   "v5" — 2026-05-04: identity-first partial-cardinality assignment (Bug D).
#          Adds a stage to refine_assignments that re-assigns rallies with
#          fewer than 4 primary tracks (e.g., near-server occlusion) using
#          gallery-only appearance scoring + rectangular Hungarian. Default
#          OFF (`ENABLE_IDENTITY_FIRST_MATCHING=1` to enable). Bumps the
#          version even when default-OFF so anchors auto-invalidate on
#          read after enable, ensuring the new path takes effect cleanly.
#  - v6: 2026-05-04 — `remap-track-ids` enforces the
#          primary_track_ids ⊆ trackToPlayer contract: identity
#          passthroughs for unmapped non-colliding tracks were leaking
#          junk PID labels (e.g. "PID 7") into the editor. Tracks
#          without a PID assignment now resolve to UNLABELED_TRACK_ID
#          and their positions get dropped. Bump invalidates cached
#          anchors so primary_track_ids gets cleaned on the next run.
#  - v7: 2026-05-05 — `relink_primary_fragments` adds a bbox-quality-
#          aware merge gate. When BOTH endpoint bboxes are below
#          PRIMARY_RELINK_SMALL_BBOX_AREA (default 0.012), the
#          appearance gate (HSV Bhattacharyya) is bypassed and a
#          motion-only velocity check is applied instead. Rationale:
#          far-side occluded detections produce mostly-sand crops that
#          defeat both HSV histograms and the learned ReID head; the
#          appearance signal is uninformative noise on these inputs.
#          The motion-only path correctly approves the user-confirmed
#          merge in 84e66e74 r13 (P2 PID flicker case) that the
#          previous gate blocked. Non-low-quality bboxes continue
#          through the existing appearance-aware path — bit-exact for
#          clean-bbox cases. Bump invalidates anchors so freshly-merged
#          primary tracks propagate cleanly to the next run.
#  - v8: 2026-05-07 — ref-crop matcher path removed. The frozen-profile
#          dual-path (reference_profiles / frozen_player_ids) collapsed
#          to the always-blind path; reference_profiles parameter +
#          frozen_player_ids attribute + replay_refine_from_scratchpad
#          deleted across phases 1-4 of the cleanup. Bump invalidates
#          anchors so the first run after merge re-solves under the
#          collapsed code path; eval gate confirmed byte-identical
#          PERMUTED panel. DB column Video.canonicalPidMapJson and
#          table player_reference_crops kept dormant for a future
#          post-hoc cluster-pick UX.
MATCHER_VERSION = "v8"


def scratchpad_to_dict(tracker: MatchPlayerTracker) -> dict[str, Any]:
    """Serialize the per-rally + match-level state of a Pass-2 run.

    Persisted on every blind-path solve as `match_analysis_json.rallyScratchpad`
    so future diagnostics can inspect intermediate state without re-running the
    pipeline. Final player profiles are NOT included here — they live at
    `match_analysis_json.playerProfiles` and are loaded by the caller.

    Bundle shape (matches tests in TestMatchScratchpadSerialization):
      - rallies: list of StoredRallyData.to_dict() per rally, in order
      - sideSwitches: sorted rally indices where Pass 2 stage 0 flipped sides
    """
    return {
        "rallies": [data.to_dict() for data in tracker.stored_rally_data],
        "sideSwitches": sorted(tracker.last_side_switches),
    }


def _build_rally_crop_extractor(
    video_path: Path,
    rally_start_ms: int,
    positions: list[PlayerPosition],
    fps: float,
) -> Callable[[int, int], np.ndarray | None]:
    """Build a ``(track_id, rally_relative_frame) -> BGR crop`` closure.

    Used by the within-track appearance splitter (Task 6, 2026-04-26).
    The splitter passes rally-relative frame numbers — the closure adds
    ``rally_start_frame_in_video`` to derive the absolute video frame, looks
    up the bbox via a per-rally cache, seeks the shared ``cv2.VideoCapture``,
    and returns the cropped BGR frame.

    Returns ``None`` on every error path (capture broken, frame not found,
    bbox missing or out-of-range, crop too small) — callers (the splitter's
    crop-collection loop) skip ``None`` and degrade gracefully.

    The capture is opened lazily on first call. Each rally builds its own
    extractor → its own capture; the cap is held until the closure is
    garbage-collected. Acceptable for our scale (~9 fixtures × ~10 rallies);
    revisit if used in a long-running service.
    """
    rally_start_frame_in_video = int(round(rally_start_ms / 1000.0 * fps))
    bbox_by_tid_frame: dict[tuple[int, int], dict[str, float]] = {
        (int(p.track_id), int(p.frame_number)): {
            "x": float(p.x), "y": float(p.y),
            "width": float(p.width), "height": float(p.height),
        }
        for p in positions
    }
    state: dict[str, Any] = {"cap": None}

    def crop_extractor(track_id: int, rally_rel_frame: int) -> np.ndarray | None:
        bbox = bbox_by_tid_frame.get((int(track_id), int(rally_rel_frame)))
        if bbox is None:
            return None
        cap: Any = state["cap"]
        if cap is None:
            cap = cv2.VideoCapture(str(video_path))
            state["cap"] = cap
            if not cap.isOpened():
                return None
        abs_frame = rally_start_frame_in_video + int(rally_rel_frame)
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, img = cap.read()
        if not ok or img is None:
            return None
        h, w = img.shape[:2]
        x1 = max(0, int((bbox["x"] - bbox["width"] / 2) * w))
        y1 = max(0, int((bbox["y"] - bbox["height"] / 2) * h))
        x2 = min(w, int((bbox["x"] + bbox["width"] / 2) * w))
        y2 = min(h, int((bbox["y"] + bbox["height"] / 2) * h))
        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return None
        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
            return None
        return np.asarray(crop, dtype=np.uint8)

    return crop_extractor


def match_players_across_rallies(
    video_path: Path,
    rallies: list[RallyTrackData],
    num_samples: int = 12,
    collect_diagnostics: bool = False,
    extract_reid: bool = False,
    reid_model: GeneralReIDModel | None = None,
    calibrator: CourtCalibrator | None = None,
    *,
    prior_match_analysis: dict[str, Any] | None = None,
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
        extract_reid: If True, extract DINOv2 embeddings per track for ReID-based
            cost blending in the Hungarian assignment. Auto-enabled when a general
            ReID model is provided.
        reid_model: Optional GeneralReIDModel for embedding extraction.
            When provided, uses its projection head.
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

    # Profile-drift probe (Phase 1, Task 3 of post-7307c1d-revert refactor).
    # No-op when MATCH_PLAYERS_PROBE is unset; otherwise writes a sidecar
    # JSON to analysis/reports/profile_drift_probe/ at finalize time.
    _probe.begin_probe(
        video_id=rallies[0].video_id if rallies else "",
        rally_ids=[r.rally_id for r in rallies],
        extra={
            "num_rallies": len(rallies),
            "extract_reid": bool(extract_reid),
        },
    )

    tracker = MatchPlayerTracker(
        calibrator=calibrator,
        collect_diagnostics=collect_diagnostics,
    )

    results: list[RallyTrackingResult] = []

    for rally_idx, rally in enumerate(rallies):
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

    # Pass 2: Re-score all rallies with final profiles.
    #
    # Two paths:
    #   - Frozen profiles (ref-crop bypass, plan Q6=3): keep today's
    #     forward-Hungarian + Pass-2 stages 1+2 path. Day-4's 95.22%
    #     direct-accuracy result on click-GT lives on this path.
    #   - Blind path: a global ``MatchSolver`` replaces Pass-1 forward
    #     Hungarian + Pass-2 stages 1+2. ``refine_assignments`` runs
    #     Stage 0 (side-switch) + the canonical re-anchor only.
    track_stats_hashes: dict[str, str] = {}
    cache_hit_count = 0
    cache_total = 0
    from rallycut.tracking.match_solver import MatchSolver

    # Per-rally pre-MatchSolver state hashes — fed back to the CLI to
    # write `assignmentAnchor` into match_analysis_json. Always
    # populated on the blind path so the next run can use them.
    for i, data in enumerate(tracker.stored_rally_data):
        if i < len(rallies):
            track_stats_hashes[rallies[i].rally_id] = _stored_rally_data_hash(data)

    # Per-rally assignment-anchor cache (ENABLE_ASSIGNMENT_ANCHORS=1).
    # When a rally's pre-MatchSolver hash matches its prior anchor's
    # hash, we pin the prior assignment instead of re-solving — this
    # decouples its decision from cross-rally input drift (the actual
    # cascade source confirmed by the Phase 1 falsification + DB-state
    # determinism probe).
    pinned_assignments: dict[int, dict[int, int]] = {}
    # Default ON — anchor cache decouples each rally from cross-rally
    # input drift. Set ENABLE_ASSIGNMENT_ANCHORS=0 to disable.
    anchors_enabled = os.environ.get("ENABLE_ASSIGNMENT_ANCHORS", "1") != "0"
    if anchors_enabled and prior_match_analysis:
        prior_anchors_by_rid: dict[str, dict[str, Any]] = {}
        for entry in prior_match_analysis.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id")
            anchor = entry.get("assignmentAnchor")
            if rid and isinstance(anchor, dict):
                prior_anchors_by_rid[rid] = anchor

        stale_version_count = 0
        for i, rally in enumerate(rallies):
            if i >= len(tracker.stored_rally_data):
                break
            anchor = prior_anchors_by_rid.get(rally.rally_id)
            if not anchor:
                continue
            # Version-key check: anchors written by an older matcher
            # are silently misleading after any matcher logic change.
            # Drop them so the rally re-solves cleanly.
            if anchor.get("matcherVersion") != MATCHER_VERSION:
                stale_version_count += 1
                continue
            if anchor.get("trackStatsHash") != track_stats_hashes.get(rally.rally_id):
                continue
            raw_assignment = anchor.get("assignment") or {}
            try:
                assignment = {
                    int(k): int(v) for k, v in raw_assignment.items()
                }
            except (TypeError, ValueError):
                continue
            expected_tids = {
                int(t) for t in tracker.stored_rally_data[i].top_tracks
            }
            if set(assignment.keys()) != expected_tids:
                # Anchor's track ids don't fit current state — skip.
                continue
            pinned_assignments[i] = assignment

        if pinned_assignments:
            logger.info(
                "AssignmentAnchor cache: %d/%d rallies pinned",
                len(pinned_assignments), len(tracker.stored_rally_data),
            )
        if stale_version_count:
            logger.info(
                "AssignmentAnchor cache: invalidated %d rally/rallies "
                "with stale matcherVersion (current=%s); they will "
                "re-solve from scratch.",
                stale_version_count, MATCHER_VERSION,
            )

    cache_hit_count = len(pinned_assignments)
    cache_total = len(tracker.stored_rally_data) if anchors_enabled else 0

    _probe.record_track_stats_input(tracker.stored_rally_data)
    solver = MatchSolver(reid_blend=REID_BLEND)
    solved = solver.solve(
        tracker.stored_rally_data,
        pinned_assignments=pinned_assignments or None,
    )

    # Splice solver assignments into the per-rally results. server_player_id
    # is re-derived from the solver's track_to_player using the original
    # server-track identity recovered from the Pass-1 mapping (track_id
    # itself is stable across passes; only its pid label changes).
    spliced: list[RallyTrackingResult] = []
    for i, r in enumerate(results):
        new_t2p = solved[i] if i < len(solved) else r.track_to_player

        server_track_id: int | None = None
        if r.server_player_id is not None:
            for tid, pid in r.track_to_player.items():
                if pid == r.server_player_id:
                    server_track_id = tid
                    break
        new_server_pid = (
            new_t2p.get(server_track_id)
            if server_track_id is not None
            else r.server_player_id
        )

        spliced.append(RallyTrackingResult(
            rally_index=r.rally_index,
            track_to_player=new_t2p,
            server_player_id=new_server_pid,
            side_switch_detected=False,  # Stage 0 sets this below.
            assignment_confidence=r.assignment_confidence,
            sub_tracks=r.sub_tracks,
        ))
    results = spliced

    # Rebuild profiles from solver assignments so downstream consumers
    # (team_templates, scratchpad replay) see solver truth, not the
    # discarded Pass-1 forward Hungarian. Reset first since Pass 1 may
    # have populated some pids and skipped others under the 0.80 gate.
    tracker.state.players.clear()
    tracker.state.initialize_players()
    for i, data in enumerate(tracker.stored_rally_data):
        if i < len(solved) and solved[i]:
            if _probe.is_enabled():
                before = _probe.checksum_profiles(tracker.state.players)
                tracker._update_profiles(data.track_stats, solved[i])
                after = _probe.checksum_profiles(tracker.state.players)
                _probe.record_update_profiles(
                    rally_idx=i,
                    track_to_player=solved[i],
                    before=before,
                    after=after,
                    context="post_solve",
                )
            else:
                tracker._update_profiles(data.track_stats, solved[i])

    results = tracker.refine_assignments(results, skip_stages_1_and_2=True)

    # `_slow_drift_split` (ENABLE_SLOW_DRIFT_SPLIT=1) removed
    # 2026-05-03 per dormant_flag_audit_2026_05_03.md. The
    # position-based bisect detector was parked half-finished
    # with two known unresolved issues (within_rally_swap
    # artifact at bisect frame; anchor-cache interaction); its
    # pattern is generalized by `_within_rally_id_switch.py`
    # (Phase 1+2) which uses appearance signal instead of
    # position drift and works on cases the position detector
    # misses (sub-threshold position shift but clear appearance
    # discontinuity, e.g. 7d77980f / 09553ef1).

    # Within-rally appearance-based ID-switch detector (Phase 1, 2026-05-03).
    # Detects within-rally identity drift via per-track appearance
    # consistency: split each track into 3 windows, flag when intra-
    # window cost exceeds RELATIVE_GATE_K × median(inter-track cost).
    # Robust on cases where position shift is borderline but appearance
    # changes dramatically (BoT-SORT continuing a track on a different
    # physical player after occlusion). Default OFF; enable via
    # ENABLE_WITHIN_RALLY_REPAIR=1.
    #
    # Runs AFTER both the frozen-profiles and blind paths so it applies
    # uniformly — slow-drift's blind-only placement is a known limitation.
    from rallycut.tracking import _within_rally_id_switch as _wris
    if _wris.is_enabled():
        logger.info(
            "within_rally_repair: scanning %d rallies "
            "(ENABLE_WITHIN_RALLY_REPAIR=1)",
            len(rallies),
        )
        n_wris_emitted = 0
        for i, (rally, r) in enumerate(zip(rallies, results)):
            overrides = _wris.maybe_emit_within_rally_split(
                rally_id=rally.rally_id,
                video_path=video_path,
                rally_start_ms=rally.start_ms,
                rally_end_ms=rally.end_ms,
                positions=rally.positions,
                track_to_player=r.track_to_player,
                reid_model=reid_model,
            )
            if not overrides:
                continue
            for ov in overrides:
                r.sub_tracks.append(ov)
                if ov.aggregated_argmax_pid is not None:
                    r.track_to_player[ov.synthetic_track_id] = (
                        ov.aggregated_argmax_pid
                    )
            n_wris_emitted += 1
        if n_wris_emitted:
            logger.info(
                "within_rally_repair: emitted split sub-tracks "
                "for %d rallies",
                n_wris_emitted,
            )

    # Build team templates from canonical-aware positional team membership.
    # Each diagnostic carries the per-rally `track_court_sides` produced by
    # `_classify_track_sides`; pairing with each rally's `track_to_player`
    # yields per-pid mode-vote of "near vs far team" — replaces the legacy
    # `pid <= 2 → team 0` partition that broke under canonical (ref-crop
    # sourced) pids.
    track_to_player_per_rally = [r.track_to_player for r in results]
    track_court_sides_per_rally = [
        d.track_court_sides for d in tracker.diagnostics
    ]
    if len(track_court_sides_per_rally) != len(track_to_player_per_rally):
        # Mismatch shouldn't happen in production (diagnostics are emitted
        # 1:1 with rallies) but if it does, fall through to the legacy
        # partition rather than building a wrong template silently.
        logger.warning(
            "team-template input length mismatch: results=%d diagnostics=%d; "
            "falling back to legacy partition",
            len(track_to_player_per_rally), len(track_court_sides_per_rally),
        )
        track_to_player_per_rally = None  # type: ignore[assignment]
        track_court_sides_per_rally = None  # type: ignore[assignment]
    team_templates = build_team_templates(
        tracker.state.players,
        track_to_player_per_rally=track_to_player_per_rally,
        track_court_sides_per_rally=track_court_sides_per_rally,
    )

    _probe.finalize_probe()

    return MatchPlayersResult(
        rally_results=results,
        player_profiles=dict(tracker.state.players),
        team_templates=team_templates,
        diagnostics=tracker.diagnostics,
        scratchpad=scratchpad_to_dict(tracker),
        track_stats_hashes=track_stats_hashes,
        anchor_cache_hits=cache_hit_count,
        anchor_cache_total=cache_total,
    )
