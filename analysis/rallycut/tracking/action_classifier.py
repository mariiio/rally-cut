"""Action classification for beach volleyball.

Classifies detected ball contacts into volleyball actions
(serve/receive/set/attack/block/dig) using either:

1. **Learned classifier** (default): A trained GBM model predicts dig/set/attack
   from trajectory features + sequence context. Serve, receive, and block stay
   heuristic. Achieves ~92% action accuracy (up from ~48% rule-based).
   Auto-loaded from weights/action_classifier/action_classifier.pkl.

2. **Rule-based state machine** (fallback): Uses contact count per side
   (dig=1st, set=2nd, attack=3rd) with net-crossing detection for side changes.
   No labeled data required.

Beach volleyball rules that constrain both modes:
- 2v2, max 3 contacts per side
- Strict sequence: serve → receive → set → attack
- Blocks count as a contact (unlike indoor volleyball)
- Each rally starts with a serve from behind the baseline
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, overload

from rallycut.tracking.contact_detector import Contact, ContactSequence, ball_crossed_net

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.action_type_classifier import ActionTypeClassifier
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Cached default action type classifier (loaded once from disk on first use)
_default_action_classifier_cache: dict[str, ActionTypeClassifier | None] = {}


def _get_default_action_classifier() -> ActionTypeClassifier | None:
    """Load and cache the default action type classifier from disk.

    Returns None if no trained model exists at the default path.
    """
    if "default" not in _default_action_classifier_cache:
        from rallycut.tracking.action_type_classifier import load_action_type_classifier

        clf = load_action_type_classifier()
        _default_action_classifier_cache["default"] = clf
        if clf is not None:
            logger.info("Auto-loaded action type classifier from default path")
    return _default_action_classifier_cache["default"]


class ActionType(str, Enum):
    """Volleyball action types."""

    SERVE = "serve"
    RECEIVE = "receive"
    SET = "set"
    ATTACK = "attack"
    BLOCK = "block"
    DIG = "dig"  # Defensive save after attack (similar to receive)
    UNKNOWN = "unknown"


@dataclass
class ClassifiedAction:
    """A classified volleyball action."""

    action_type: ActionType
    frame: int
    ball_x: float
    ball_y: float
    velocity: float
    player_track_id: int  # -1 if unknown
    court_side: str  # "near" or "far"
    confidence: float  # Classification confidence (0-1)
    is_synthetic: bool = False  # True for inferred actions (e.g. missed serve)
    team: str = "unknown"  # "A" (near court), "B" (far court), or "unknown"

    # Play-level annotations populated by
    # ``rallycut.statistics.play_annotations.annotate_rally_actions`` when a
    # court calibration is available. Left as ``None`` on every path that
    # doesn't opt in, and omitted from ``to_dict()`` when ``None`` so
    # existing stored JSON is bit-identical on the default pipeline.
    action_zone: int | None = None       # player's feet court-x zone 1-5, team-relative
    attack_direction: str | None = None  # "line" | "cross" | "cut" | None
    set_origin_zone: int | None = None   # setter feet court-x zone 1-5 at set contact
    set_dest_zone: int | None = None     # attacker feet court-x zone 1-5 at next attack

    def to_dict(self) -> dict[str, Any]:
        d = {
            "action": self.action_type.value,
            "frame": self.frame,
            "ballX": self.ball_x,
            "ballY": self.ball_y,
            "velocity": self.velocity,
            "playerTrackId": self.player_track_id,
            "courtSide": self.court_side,
            "confidence": self.confidence,
            "team": self.team,
        }
        # Omitted when False for backward compatibility with existing stored data
        if self.is_synthetic:
            d["isSynthetic"] = True
        if self.action_zone is not None:
            d["actionZone"] = self.action_zone
        if self.attack_direction is not None:
            d["attackDirection"] = self.attack_direction
        if self.set_origin_zone is not None:
            d["setOriginZone"] = self.set_origin_zone
        if self.set_dest_zone is not None:
            d["setDestZone"] = self.set_dest_zone
        return d


def _team_label(
    player_track_id: int,
    team_assignments: dict[int, int] | None,
) -> str:
    """Map a player track ID to a team label using team assignments.

    Convention: team 0 (near court) = "A", team 1 (far court) = "B".
    Matches existing Rally.servingTeam A/B enum in the database.
    """
    if team_assignments and player_track_id >= 0:
        team_int = team_assignments.get(player_track_id)
        if team_int is not None:
            return "A" if team_int == 0 else "B"
    return "unknown"


@dataclass
class RallyActions:
    """All classified actions within a single rally."""

    actions: list[ClassifiedAction] = field(default_factory=list)
    rally_id: str = ""
    team_assignments: dict[int, int] = field(default_factory=dict)
    # Formation-based serving team prediction from player positions at rally
    # start. Set by classify_rally_actions when use_formation_serving_team is
    # enabled. Used as primary signal for `serving_team` (the contact-based
    # signal is only 52% vs 72-76% for formation — see
    # score_tracking_architecture_2026_04.md).
    formation_serving_team: str | None = None

    @property
    def serve(self) -> ClassifiedAction | None:
        """Get the serve action."""
        for a in self.actions:
            if a.action_type == ActionType.SERVE:
                return a
        return None

    @property
    def serving_team(self) -> str | None:
        """Get the team that served, or None if unknown.

        Prefers formation-based prediction (72-76% accuracy) over the
        contact-based serve action team (52% accuracy) when available.
        """
        if self.formation_serving_team is not None:
            return self.formation_serving_team
        serve = self.serve
        return serve.team if serve and serve.team != "unknown" else None

    @property
    def action_sequence(self) -> list[ActionType]:
        """Get ordered action type sequence."""
        return [a.action_type for a in self.actions]

    def actions_by_player(self, track_id: int) -> list[ClassifiedAction]:
        """Get all actions for a specific player."""
        return [a for a in self.actions if a.player_track_id == track_id]

    def actions_by_team(self, team: str) -> list[ClassifiedAction]:
        """Get all actions for a specific team ("A" or "B")."""
        return [a for a in self.actions if a.team == team]

    def actions_by_type(self, action_type: ActionType) -> list[ClassifiedAction]:
        """Get all actions of a specific type."""
        return [a for a in self.actions if a.action_type == action_type]

    @property
    def num_contacts(self) -> int:
        """Total contacts (blocks count as contacts in beach volleyball)."""
        return sum(1 for a in self.actions if a.action_type != ActionType.UNKNOWN)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rallyId": self.rally_id,
            "numContacts": self.num_contacts,
            "actionSequence": [a.value for a in self.action_sequence],
            "actions": [a.to_dict() for a in self.actions],
        }
        if self.team_assignments:
            d["teamAssignments"] = {
                str(tid): ("A" if team == 0 else "B")
                for tid, team in self.team_assignments.items()
            }
        serving = self.serving_team
        if serving:
            d["servingTeam"] = serving
        return d


@dataclass
class ActionClassifierConfig:
    """Configuration for rule-based action classification."""

    # Serve detection
    serve_window_frames: int = 60  # Serve must occur in first N frames (~2s @ 30fps)
    serve_min_velocity: float = 0.025  # Min velocity for serve
    serve_fallback: bool = True  # Treat first contact as serve if none found in window

    # Block detection
    block_max_frame_gap: int = 8  # Max frames between attack and block

    # Confidence thresholds
    high_confidence: float = 0.9  # Confidence when rules match perfectly
    medium_confidence: float = 0.7  # Confidence with some ambiguity
    low_confidence: float = 0.5  # Confidence when classification is uncertain

    # Formation-based serving team detection
    # In beach volleyball, the serving team has one player behind the baseline
    # (server) and one at the net (partner). The receiving team has both
    # players in ready position mid-court. This creates a vertical separation
    # asymmetry in tracked player positions. Measured: 72-76% LOO-video CV on
    # 304 rallies vs 46.2% baseline (see score_tracking_architecture_2026_04.md).
    use_formation_serving_team: bool = True
    formation_window_frames: int = 120  # Frames at rally start to analyze
    # Separation ratio margin: larger = stricter but lower coverage. Sweep
    # on canonical 92-rally action_GT subset:
    #   margin=1.00 → 56.5% (more track_to_player fallback = more noise)
    #   margin=1.15 → 57.6% (best on canonical metric)
    #   margin=1.30 → ~57%  (too strict, abstain loss > fallback gain)
    # Broader 304-rally harness prefers 1.00 (68.8%) because it has fewer
    # low-confidence rallies. 1.15 chosen for canonical metric parity.
    formation_margin: float = 1.15


def _ball_moving_toward_net(
    ball_positions: list[BallPosition],
    contact_frame: int,
    ball_y: float,
    net_y: float,
    look_ahead_frames: int = 15,
    min_toward_ratio: float = 0.5,
) -> bool | None:
    """Check whether ball moves toward net in the frames after a contact.

    Serves go toward the net; receives move laterally/up to a teammate.

    Args:
        ball_positions: Sorted list of ball positions.
        contact_frame: Frame of the contact.
        ball_y: Ball Y at the contact.
        net_y: Net Y position.
        look_ahead_frames: Number of frames to look ahead.
        min_toward_ratio: Minimum fraction of toward-net transitions.

    Returns:
        True if ball moves toward net, False if confirmed not toward net,
        None if insufficient data.
    """
    positions = [
        bp for bp in ball_positions
        if contact_frame < bp.frame_number <= contact_frame + look_ahead_frames
    ]
    if len(positions) < 3:
        return None

    near_side = ball_y > net_y
    toward_count = 0
    total = 0
    for j in range(1, len(positions)):
        dy = positions[j].y - positions[j - 1].y
        if abs(dy) < 0.001:
            continue
        total += 1
        if near_side and dy < 0:  # Decreasing Y = toward net from near side
            toward_count += 1
        elif not near_side and dy > 0:  # Increasing Y = toward net from far side
            toward_count += 1

    if total == 0:
        return None
    return toward_count / total >= min_toward_ratio


def _serve_baselines(net_y: float) -> tuple[float, float]:
    """Compute dynamic serve baseline Y positions.

    Baselines scale proportionally to net_y so they adapt to camera angles.
    Coefficients calibrated to match 0.82/0.18 at net_y=0.5.

    Returns (baseline_near, baseline_far).
    """
    return net_y + (1.0 - net_y) * 0.64, net_y * 0.36


def _find_server_by_position(
    player_positions: list[PlayerPosition],
    start_frame: int,
    net_y: float,
    window_frames: int = 45,
    separation_min: float = 0.04,
    calibrator: CourtCalibrator | None = None,
) -> tuple[int, str, float]:
    """Identify the server from player positions at rally start.

    In beach volleyball (2v2), the server is the only player standing near a
    baseline at rally start.  All others are mid-court or at the net.

    Strategy: split players into near-side and far-side.  On each side, the
    player furthest from the net is the server candidate.  The candidate must
    be separated from their same-side teammate by *separation_min*.  The side
    with the larger separation wins.

    When *calibrator* is provided and calibrated, player foot positions are
    projected to court-space meters (8m × 16m, net at y=8).  This eliminates
    perspective compression that makes far-side detection unreliable in image
    space.  Falls back to image-space when calibrator is unavailable or
    projection produces out-of-bounds results.

    Returns:
        (server_track_id, court_side, confidence).
        (-1, "", 0.0) when no server can be identified.
    """
    end_frame = start_frame + window_frames

    # Gather foot positions per track (foot ≈ bottom of bbox)
    track_foot_ys: dict[int, list[float]] = defaultdict(list)
    track_foot_xs: dict[int, list[float]] = defaultdict(list)
    for p in player_positions:
        if start_frame <= p.frame_number < end_frame:
            track_foot_ys[p.track_id].append(p.y + p.height / 2.0)
            track_foot_xs[p.track_id].append(p.x)

    if not track_foot_ys:
        return -1, "", 0.0

    # Mean foot position per track (image-space, normalized 0-1)
    track_mean_y: dict[int, float] = {
        tid: sum(ys) / len(ys) for tid, ys in track_foot_ys.items()
    }
    track_mean_x: dict[int, float] = {
        tid: sum(xs) / len(xs) for tid, xs in track_foot_xs.items()
    }

    # --- Court-space path (when calibrator available) ---
    if calibrator is not None and calibrator.is_calibrated:
        result = _find_server_court_space(
            track_mean_x, track_mean_y, calibrator,
        )
        if result is not None:
            return result
        # Fall through to image-space on projection failure

    # --- Image-space path (fallback) ---
    return _find_server_image_space(track_mean_y, net_y, separation_min)


# Court-space constants (meters)
_COURT_NET_Y = 8.0  # Net position in court coords
_COURT_LENGTH = 16.0  # Full court length
_COURT_SEP_MIN = 1.0  # Minimum teammate separation to qualify as server candidate
_COURT_BASELINE_FULL_CONF = 3.0  # Baseline gap for confidence=1.0
_COURT_Y_MIN = -3.0  # Sanity bound (baseline - 3m for serving position)
_COURT_Y_MAX = 19.0  # Sanity bound (far baseline + 3m)


def _find_server_court_space(
    track_mean_x: dict[int, float],
    track_mean_y: dict[int, float],
    calibrator: CourtCalibrator,
) -> tuple[int, str, float] | None:
    """Court-space server detection: separation qualifies, baseline decides.

    Combines two signals available in court-space:

    1. **Teammate separation** (relative): the server candidate on each side
       must be separated from their teammate by at least 1m. This filters out
       compact formations where no player is clearly in a serving position.

    2. **Baseline proximity** (absolute): among qualified candidates, the one
       closer to their respective baseline is the server. This is the unique
       advantage of court-space — in image-space, perspective makes baseline
       distance unreliable.

    The key insight: separation qualifies candidates (removes noise), then
    baseline proximity picks the winner (leverages absolute position).  This
    avoids the image-space failure mode where the far side always has more
    separation due to perspective, while also avoiding the pure-baseline
    failure mode where both sides have a player near the baseline.

    Returns None to signal fallback to image-space.
    """
    track_court_y: dict[int, float] = {}
    for tid, foot_y in track_mean_y.items():
        foot_x = track_mean_x.get(tid, 0.5)
        try:
            _, cy = calibrator.image_to_court((foot_x, foot_y), 1, 1)
        except Exception:
            return None
        if cy < _COURT_Y_MIN or cy > _COURT_Y_MAX:
            return None  # Bad projection — fall back
        track_court_y[tid] = cy

    if not track_court_y:
        return None

    # Split into near (court_y < net) and far (court_y >= net)
    near_tracks: list[tuple[int, float]] = []  # (tid, dist_from_net)
    far_tracks: list[tuple[int, float]] = []
    for tid, cy in track_court_y.items():
        dist = abs(cy - _COURT_NET_Y)
        if cy < _COURT_NET_Y:
            near_tracks.append((tid, dist))
        else:
            far_tracks.append((tid, dist))

    # For each side: find server candidate (furthest from net), compute
    # separation from teammate, and baseline distance.
    #   (side, tid, separation, baseline_dist)
    candidates: list[tuple[str, int, float, float]] = []
    for side, tracks in [("near", near_tracks), ("far", far_tracks)]:
        if not tracks:
            continue
        tracks.sort(key=lambda t: t[1], reverse=True)  # Furthest from net first
        best_tid, best_dist = tracks[0]
        # Separation from teammate; for solo player, use distance from net
        # (a solo player far from the net is a strong server signal).
        sep = best_dist - tracks[1][1] if len(tracks) >= 2 else best_dist

        # Baseline distance: near baseline at cy=0, far baseline at cy=16
        cy = track_court_y[best_tid]
        bl_dist = cy if side == "near" else _COURT_LENGTH - cy

        candidates.append((side, best_tid, sep, bl_dist))

    if not candidates:
        return None

    # Filter to candidates with meaningful teammate separation
    qualified = [(s, t, sep, bl) for s, t, sep, bl in candidates
                 if sep >= _COURT_SEP_MIN]

    if not qualified:
        return -1, "", 0.0

    if len(qualified) == 1:
        side, tid, sep, bl_dist = qualified[0]
        confidence = min(1.0, sep / 5.0)
        return tid, side, confidence

    # Both sides qualified: pick by baseline proximity (closer = more likely server)
    qualified.sort(key=lambda c: c[3])  # Sort by baseline_dist ascending
    winner_side, winner_tid, winner_sep, winner_bl = qualified[0]
    other_side, other_tid, other_sep, other_bl = qualified[1]

    bl_gap = other_bl - winner_bl  # How much closer winner is to baseline

    confidence = min(1.0, bl_gap / _COURT_BASELINE_FULL_CONF)
    confidence = max(confidence, 0.15)  # Floor for close calls
    return winner_tid, winner_side, confidence


def _find_server_image_space(
    track_mean_y: dict[int, float],
    net_y: float,
    separation_min: float,
) -> tuple[int, str, float]:
    """Original image-space server detection logic."""
    near_tracks: list[tuple[int, float]] = []
    far_tracks: list[tuple[int, float]] = []
    for tid, foot_y in track_mean_y.items():
        if foot_y > net_y:
            near_tracks.append((tid, foot_y - net_y))
        else:
            far_tracks.append((tid, net_y - foot_y))

    candidates: list[tuple[int, str, float, float]] = []
    for side, tracks in [("near", near_tracks), ("far", far_tracks)]:
        if not tracks:
            continue
        tracks.sort(key=lambda t: t[1], reverse=True)
        best_tid, best_dist = tracks[0]
        if len(tracks) >= 2:
            teammate_dist = tracks[1][1]
            sep = best_dist - teammate_dist
        else:
            sep = best_dist
        candidates.append((best_tid, side, sep, best_dist))

    if not candidates:
        return -1, "", 0.0

    candidates.sort(key=lambda t: t[2], reverse=True)
    best_tid, best_side, best_sep, best_dist = candidates[0]

    if best_sep < separation_min:
        return -1, "", 0.0

    confidence = min(1.0, best_sep / 0.10)
    return best_tid, best_side, confidence


def _compute_auto_split_y(player_positions: list[PlayerPosition]) -> float | None:
    """Recompute court split Y from player position clustering.

    Falls back when `net_y` from `ContactSequence` misclassifies all players
    to one side (observed in video 0a383519 where stored split_y=0.43 put all
    4 tracked players on "near").

    Splits at the biggest gap between consecutive player median Y positions.
    When ≥4 tracks exist, prefers a split that produces a balanced 2v2
    distribution — if the biggest gap gives 3v1 or 4v0, tries the
    second-biggest gap instead.

    Requires at least 3 tracked players.

    Returns:
        Split Y (0-1 normalized), or None when too few tracks.
    """
    by_track: dict[int, list[float]] = defaultdict(list)
    for p in player_positions:
        if p.track_id < 0:
            continue
        by_track[p.track_id].append(p.y + p.height / 2.0)  # foot Y

    if len(by_track) < 3:
        return None

    medians = sorted(
        sum(ys) / len(ys) for ys in by_track.values()
    )

    # Rank all gaps by size.
    gaps: list[tuple[float, float]] = []  # (gap_size, split_y)
    for i in range(len(medians) - 1):
        gap = medians[i + 1] - medians[i]
        split_y = (medians[i] + medians[i + 1]) / 2.0
        gaps.append((gap, split_y))
    gaps.sort(key=lambda g: -g[0])  # largest gap first

    if not gaps:
        return None

    # With ≥4 tracks, prefer a 2v2 split.  The biggest gap often produces
    # 3v1 when one player is near the net line.
    n_tracks = len(medians)
    if n_tracks >= 4:
        for _, split_y in gaps:
            below = sum(1 for m in medians if m <= split_y)
            above = n_tracks - below
            if below >= 2 and above >= 2:
                return split_y

    # Fall back to biggest gap (original behaviour for 3-track cases,
    # or if no 2v2 split exists with ≥4 tracks).
    return gaps[0][1]


def _find_serving_team_by_formation(
    player_positions: list[PlayerPosition],
    start_frame: int,
    net_y: float,
    team_assignments: dict[int, int] | None,
    track_to_player: dict[int, int] | None = None,
    semantic_flip: bool = False,
    window_frames: int = 120,
    margin: float = 1.15,
    ball_positions: list[BallPosition] | None = None,
    calibrator: CourtCalibrator | None = None,
    first_contact_frame: int | None = None,
    adaptive_window: bool = False,
    first_contact: Contact | None = None,
) -> tuple[str | None, float]:
    """Predict serving team from player formation at rally start.

    Delegates to `_find_serving_side_by_formation` for the physical side
    prediction (multi-feature logistic model), then maps "near"/"far" →
    "A"/"B" via `team_assignments` + `semantic_flip`.

    Args:
        player_positions: Tracked player positions for the rally.
        start_frame: Rally start frame (rally-relative, usually 0).
        net_y: Court split Y (from `ContactSequence.net_y`).
        team_assignments: Mapping `track_id → 0/1` (0=near, 1=far after
            `verify_team_assignments`). Primary mapping for side → team.
        track_to_player: Mapping `track_id → player_id (1-4)` from
            cross-rally identification. Fallback when team_assignments
            is unavailable.
        semantic_flip: Invert the final A/B output for flipped rallies.
        window_frames: Frames after `start_frame` to analyze.
        margin: Unused (kept for API compatibility).
        ball_positions: Optional ball detections for multi-feature model.
        calibrator: Optional court calibrator for court-space features.

    Returns:
        (team, confidence) where team is "A", "B", or None.
    """
    serving_side, confidence = _find_serving_side_by_formation(
        player_positions, net_y=net_y, start_frame=start_frame,
        window_frames=window_frames, ball_positions=ball_positions,
        calibrator=calibrator, first_contact_frame=first_contact_frame,
        adaptive_window=adaptive_window,
        first_contact=first_contact,
    )
    if serving_side is None:
        return None, 0.0

    # Determine which track IDs are on the serving side for team mapping.
    # Use original fixed window (0 to start_frame + window_frames) for track
    # discovery — team assignment is match-level and benefits from seeing all
    # tracks in the rally start area, not just the adaptive window.
    end_frame = start_frame + window_frames
    by_track_y: dict[int, list[float]] = defaultdict(list)
    for p in player_positions:
        if p.track_id < 0:
            continue
        if start_frame <= p.frame_number < end_frame:
            by_track_y[p.track_id].append(p.y + p.height / 2.0)

    track_medians = {
        tid: sum(ys) / len(ys) for tid, ys in by_track_y.items()
    }

    effective_split = net_y
    near_count = sum(1 for y in track_medians.values() if y > effective_split)
    if near_count == 0 or near_count == len(track_medians):
        windowed_positions = [
            p for p in player_positions
            if p.track_id >= 0 and start_frame <= p.frame_number < end_frame
        ]
        auto_split = _compute_auto_split_y(windowed_positions)
        if auto_split is not None:
            effective_split = auto_split

    if serving_side == "near":
        side_tids = [t for t, y in track_medians.items() if y > effective_split]
    else:
        side_tids = [t for t, y in track_medians.items() if y <= effective_split]

    if not side_tids:
        return None, confidence

    def _apply_flip(team: str) -> str:
        if not semantic_flip:
            return team
        return "B" if team == "A" else "A"

    if team_assignments is not None:
        for tid in side_tids:
            team_int = team_assignments.get(tid)
            if team_int is not None:
                return _apply_flip("A" if team_int == 0 else "B"), confidence

    if track_to_player:
        for tid in side_tids:
            player_id = track_to_player.get(tid)
            if player_id is None:
                continue
            return _apply_flip("A" if player_id <= 2 else "B"), confidence

    return None, confidence


def _compute_formation_features(
    near_tids: list[int],
    far_tids: list[int],
    track_pos: dict[int, tuple[float, float]],
    effective_split: float,
    net_y: float,
    ball_positions: list[BallPosition] | None = None,
    calibrator: CourtCalibrator | None = None,
    start_frame: int = 0,
    window_frames: int = 120,
) -> dict[str, float | None]:
    """Compute formation features for the multi-feature serving side model.

    All features use positive = near more likely serving convention.
    """
    track_medians_y = {tid: pos[1] for tid, pos in track_pos.items()}
    features: dict[str, float | None] = {}

    # F1: Vertical separation (near_sep - far_sep)
    def _sep(tids: list[int]) -> float:
        if len(tids) >= 2:
            ys = [track_medians_y[t] for t in tids]
            return max(ys) - min(ys)
        return abs(track_medians_y[tids[0]] - effective_split) * 0.5

    features["separation"] = _sep(near_tids) - _sep(far_tids)

    # F2: Server isolation (most isolated player per side)
    all_tids = list(track_pos.keys())
    isolation: dict[int, float] = {}
    for tid in all_tids:
        px, py = track_pos[tid]
        min_dist = float("inf")
        for other in all_tids:
            if other == tid:
                continue
            ox, oy = track_pos[other]
            d = math.sqrt((px - ox) ** 2 + (py - oy) ** 2)
            min_dist = min(min_dist, d)
        isolation[tid] = min_dist
    features["isolation"] = (
        max(isolation[t] for t in near_tids)
        - max(isolation[t] for t in far_tids)
    )

    # F3: Baseline proximity (image-space)
    near_max_y = max(track_medians_y[t] for t in near_tids)
    far_min_y = min(track_medians_y[t] for t in far_tids)
    features["baseline_img"] = near_max_y - (1.0 - far_min_y)

    # F4: Baseline proximity (court-space)
    if calibrator is not None and calibrator.is_calibrated:
        try:
            court_ys: dict[int, float] = {}
            for tid, (fx, fy) in track_pos.items():
                _, cy = calibrator.image_to_court((fx, fy), 1, 1)
                court_ys[tid] = cy
            near_bl = [court_ys[t] for t in near_tids if t in court_ys]
            far_bl = [16.0 - court_ys[t] for t in far_tids if t in court_ys]
            if near_bl and far_bl:
                features["baseline_court"] = min(far_bl) - min(near_bl)
            else:
                features["baseline_court"] = None
        except Exception:
            features["baseline_court"] = None
    else:
        features["baseline_court"] = None

    # F5: Max net distance
    features["net_dist"] = (
        max(abs(track_medians_y[t] - net_y) for t in near_tids)
        - max(abs(track_medians_y[t] - net_y) for t in far_tids)
    )

    # F6: Ball position in early rally frames.
    # Uses absolute frame bound (not windowed) to match training calibration.
    # Ball detection starts during the toss (~38 frames before serve).
    if ball_positions:
        ball_ys = [
            bp.y for bp in ball_positions
            if bp.frame_number < start_frame + window_frames
        ]
        if ball_ys:
            med_ball_y = sum(ball_ys) / len(ball_ys)
            features["ball_pos"] = med_ball_y - net_y
        else:
            features["ball_pos"] = None
    else:
        features["ball_pos"] = None

    # F7: Player count asymmetry
    features["count_asym"] = float(len(near_tids) - len(far_tids))

    return features


# Logistic regression weights for serving side prediction.
# Trained on 448 GT rallies (30 videos), LOO-video CV = 93.0%.
# Features: separation, isolation, baseline_img, baseline_court,
#           net_dist, count_asym. Positive score → near serving.
_FORMATION_WEIGHTS_6 = {
    "intercept": -2.23758492,
    "separation": 33.15155590,
    "isolation": 0.08680416,
    "baseline_img": 4.68423319,
    "baseline_court": 0.16260830,
    "net_dist": -6.97028831,
    "count_asym": -3.28034322,
}
# Extended model with ball position (LOO-video CV = 95.6%, 86% coverage).
_FORMATION_WEIGHTS_7 = {
    "intercept": -4.90035394,
    "separation": 36.27364257,
    "isolation": 2.75031487,
    "baseline_img": 1.17173663,
    "baseline_court": 0.09151576,
    "net_dist": -0.98203087,
    "ball_pos": -10.21039365,
    "count_asym": -3.33282825,
}
_FORMATION_FEATURE_ORDER_6 = [
    "separation", "isolation", "baseline_img",
    "baseline_court", "net_dist", "count_asym",
]
_FORMATION_FEATURE_ORDER_7 = [
    "separation", "isolation", "baseline_img",
    "baseline_court", "net_dist", "ball_pos", "count_asym",
]


def _compute_adaptive_window(
    ball_positions: list[BallPosition] | None,
    first_contact_frame: int | None = None,
    max_anchor_frame: int = 180,
) -> tuple[int, int]:
    """Compute the best formation analysis window from ball/contact timing.

    The serve formation is most informative around the serve moment.
    Ball detection reliably starts during the toss (~38 frames before
    serve contact, median). Uses ball detection as primary anchor and
    first contact as secondary refinement.

    Strategy (validated on 448 rallies, +1.8pp vs fixed window):
      1. If ball + contact available:
         - gap ≤ 15 frames → contact ≈ serve → [ball-30, contact+15]
         - gap > 15 → contact is receive → [ball-15, ball+45]
      2. Ball only → [ball-15, ball+45]
      3. Contact only → [contact-60, contact]
      4. Neither → [0, 120] fallback

    Returns:
        (start_frame, window_frames)
    """
    first_ball: int | None = None
    if ball_positions:
        for bp in ball_positions:
            if bp.frame_number >= 0:
                first_ball = bp.frame_number
                break

    # Guard against very late anchors (mid-rally, not serve)
    if first_ball is not None and first_ball > max_anchor_frame:
        first_ball = None
    if first_contact_frame is not None and first_contact_frame > max_anchor_frame:
        first_contact_frame = None

    if first_ball is not None and first_contact_frame is not None:
        gap = first_contact_frame - first_ball
        if gap <= 15:
            # Contact ≈ serve: window covers formation through serve
            start = max(0, first_ball - 30)
            end = first_contact_frame + 15
            # Guard: ensure positive window (handles edge case where
            # contact is at frame 0 due to noisy detection)
            if end <= start:
                return start, 60
            return start, end - start
        # Contact is receive: focus on ball detection time (serve window)
        start = max(0, first_ball - 15)
        return start, 60

    if first_ball is not None:
        start = max(0, first_ball - 15)
        return start, 60

    if first_contact_frame is not None:
        start = max(0, first_contact_frame - 60)
        return start, 60

    return 0, 120


def _classify_serve_contact(
    contact: Contact,
    net_y: float = 0.5,
    player_distance_min: float = 0.03,
) -> bool | None:
    """Classify whether a detected contact is a serve or receive.

    Uses ball position, net proximity, and player distance — attributes
    already computed by the contact detector. Validated on 322 GT-labeled
    contacts: ballY < net_y is 99% indicative of serve, ballY >= net_y
    is 95% indicative of receive.

    Args:
        contact: The detected contact with ball/player attributes.
        net_y: Court split Y (net position in image space).
        player_distance_min: Minimum player-to-ball distance to qualify
            as a serve (server reaches for toss, so distance is larger
            than a close-range receive).

    Returns:
        True if contact is a serve, False if receive, None if uncertain.
    """
    # Ball below net_y (near side, high Y) → receive
    if contact.ball_y >= net_y:
        return False

    # Ball above net_y (far side, low Y from toss) + not at net + player
    # reaching → serve
    if not contact.is_at_net and contact.player_distance >= player_distance_min:
        return True

    # Uncertain
    return None


def _serving_side_from_contact(
    contact: Contact,
    player_positions: list[PlayerPosition],
    net_y: float = 0.5,
) -> tuple[str | None, float]:
    """Determine serving side from a contact's classification and player position.

    Classifies the contact as serve/receive via ``_classify_serve_contact``,
    then uses the contact player's court position to determine which side
    served:
    - Serve contact → player's side is serving side
    - Receive contact → opposite side is serving side

    Args:
        contact: First detected contact in the rally.
        player_positions: Player positions for locating the contact player.
        net_y: Court split Y.

    Returns:
        (side, confidence) where side is "near", "far", or None.
    """
    is_serve = _classify_serve_contact(contact, net_y)
    if is_serve is None:
        return None, 0.0

    tid = contact.player_track_id
    if tid < 0:
        return None, 0.0

    # Find contact player's foot Y near the contact frame
    player_ys = [
        p.y + p.height / 2.0
        for p in player_positions
        if p.track_id == tid and abs(p.frame_number - contact.frame) < 30
    ]
    if not player_ys:
        return None, 0.0

    player_y = sum(player_ys) / len(player_ys)
    player_side = "near" if player_y > net_y else "far"

    # Confidence from how far ballY is from net_y
    ball_dist_from_net = abs(contact.ball_y - net_y)
    confidence = min(1.0, ball_dist_from_net / 0.15)

    if is_serve:
        return player_side, confidence
    # Receive → opposite side served
    return ("far" if player_side == "near" else "near"), confidence


def _formation_logistic_score(
    player_positions: list[PlayerPosition],
    net_y: float,
    start_frame: int,
    window_frames: int,
    ball_positions: list[BallPosition] | None = None,
    calibrator: CourtCalibrator | None = None,
) -> tuple[str | None, float]:
    """Core formation logistic model on a single time window.

    Clusters players into near/far sides, computes position features, and
    runs the logistic regression.  Returns (side, confidence) without any
    secondary signal fusion.
    """
    end_frame = start_frame + window_frames
    by_track: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for p in player_positions:
        if p.track_id < 0:
            continue
        if start_frame <= p.frame_number < end_frame:
            by_track[p.track_id].append((p.x, p.y + p.height / 2.0))

    if len(by_track) < 2:
        return None, 0.0

    track_pos = {
        tid: (
            sum(xy[0] for xy in xys) / len(xys),
            sum(xy[1] for xy in xys) / len(xys),
        )
        for tid, xys in by_track.items()
    }
    track_medians_y = {tid: pos[1] for tid, pos in track_pos.items()}

    effective_split = net_y
    near_count = sum(1 for y in track_medians_y.values() if y > effective_split)
    far_count = len(track_medians_y) - near_count
    if near_count == 0 or far_count == 0:
        windowed_positions = [
            p for p in player_positions
            if p.track_id >= 0 and start_frame <= p.frame_number < end_frame
        ]
        auto_split = _compute_auto_split_y(windowed_positions)
        if auto_split is None:
            return None, 0.0
        effective_split = auto_split

    near_tids = [t for t, y in track_medians_y.items() if y > effective_split]
    far_tids = [t for t, y in track_medians_y.items() if y <= effective_split]
    if not near_tids or not far_tids:
        return None, 0.0

    features = _compute_formation_features(
        near_tids, far_tids, track_pos, effective_split, net_y,
        ball_positions=ball_positions, calibrator=calibrator,
        window_frames=start_frame + window_frames,
    )

    has_ball = features.get("ball_pos") is not None
    if has_ball:
        weights = _FORMATION_WEIGHTS_7
        feat_order = _FORMATION_FEATURE_ORDER_7
    else:
        weights = _FORMATION_WEIGHTS_6
        feat_order = _FORMATION_FEATURE_ORDER_6

    score = weights["intercept"]
    for fname in feat_order:
        val = features.get(fname)
        if val is None:
            val = 0.0
        score += weights[fname] * val

    if abs(score) < 1e-6:
        return None, 0.0
    prob = 1.0 / (1.0 + math.exp(-score))
    conf = abs(prob - 0.5) * 2.0
    side = "near" if score > 0 else "far"
    return side, conf


def _find_serving_side_by_formation(
    player_positions: list[PlayerPosition],
    net_y: float,
    start_frame: int = 0,
    window_frames: int = 120,
    ball_positions: list[BallPosition] | None = None,
    calibrator: CourtCalibrator | None = None,
    first_contact_frame: int | None = None,
    adaptive_window: bool = False,
    first_contact: Contact | None = None,
) -> tuple[str | None, float]:
    """Return the raw physical serving SIDE without team-label mapping.

    Uses a multi-feature logistic regression trained on 304 GT rallies:
    vertical separation, server isolation, baseline proximity (image +
    court), max net distance, ball position, and player count asymmetry.

    When ``adaptive_window=True`` and ball/contact data is available, runs
    the logistic model on **both** a fixed window (frame 0–120) and an
    adaptive window (anchored on ball detection / first contact), then
    picks the prediction with higher confidence.  The two windows capture
    different serve timing patterns: the fixed window works when the serve
    happens early, the adaptive window works when it happens later.
    Validated at +2.8pp over either window alone on 401 GT rallies.

    Secondary signals (contact classifier, near-side late entry) are fused
    after the window selection, only overriding low-confidence predictions.

    Args:
        player_positions: Tracked player positions for the rally.
        net_y: Court split Y (from ContactSequence.net_y).
        start_frame: Rally start frame (rally-relative, usually 0).
        window_frames: Frames after start_frame to analyze.
        ball_positions: Optional ball detections for adaptive window.
        calibrator: Optional court calibrator for court-space features.
        first_contact_frame: Frame of first detected contact.
        adaptive_window: When True, run dual-window and pick best.
        first_contact: First contact for serve contact classifier.

    Returns:
        (side, confidence) where side is "near", "far", or None.
    """
    if not player_positions:
        return None, 0.0

    # ── Dual-window: run both fixed and adaptive, pick higher confidence ──
    if adaptive_window and ball_positions:
        adaptive_start, adaptive_window_frames = _compute_adaptive_window(
            ball_positions, first_contact_frame,
        )

        # Fixed window (rally start)
        side_fixed, conf_fixed = _formation_logistic_score(
            player_positions, net_y, start_frame, window_frames,
            ball_positions=ball_positions, calibrator=calibrator,
        )
        # Adaptive window (ball/contact anchored)
        side_adaptive, conf_adaptive = _formation_logistic_score(
            player_positions, net_y, adaptive_start, adaptive_window_frames,
            ball_positions=ball_positions, calibrator=calibrator,
        )

        # Pick the more confident prediction
        if side_fixed == side_adaptive:
            # Both agree — use the prediction with combined confidence
            formation_side = side_fixed
            formation_conf = max(conf_fixed, conf_adaptive)
        elif conf_adaptive > conf_fixed:
            formation_side = side_adaptive
            formation_conf = conf_adaptive
        else:
            formation_side = side_fixed
            formation_conf = conf_fixed

        # Late-entry detection scans the union of both windows so it
        # catches servers who enter after the fixed window ends.
        effective_start = min(start_frame, adaptive_start)
        effective_window = (
            max(start_frame + window_frames, adaptive_start + adaptive_window_frames)
            - effective_start
        )
    else:
        # Single window: either adaptive not requested, or no ball data
        # (adaptive falls back to fixed window internally when ball is absent).
        if adaptive_window:
            start_frame, window_frames = _compute_adaptive_window(
                ball_positions, first_contact_frame,
            )
        formation_side, formation_conf = _formation_logistic_score(
            player_positions, net_y, start_frame, window_frames,
            ball_positions=ball_positions, calibrator=calibrator,
        )
        effective_start = start_frame
        effective_window = window_frames

    # ── Secondary signal: serve contact classifier ──
    # Only used when formation is weak (abstained or very low confidence).
    # The contact classifier has a high false-positive rate for serves
    # (64% of receives also have ballY < net_y) so it should NOT override
    # confident formation.
    contact_fusion_conf_gate = 0.15
    if first_contact is not None and (
        formation_side is None or formation_conf < contact_fusion_conf_gate
    ):
        contact_side, contact_conf = _serving_side_from_contact(
            first_contact, player_positions, net_y,
        )
        if contact_side is not None:
            if formation_side is None:
                return contact_side, contact_conf
            if contact_side != formation_side:
                return contact_side, contact_conf
            return formation_side, max(formation_conf, contact_conf)

    # ── Secondary signal: near-side late entry ──
    # Server walks into frame from behind the camera. Only overrides when
    # formation disagrees (pred != "near") — one-directional correction.
    # Scans a wide window because the server may enter well after the
    # formation analysis window ends.  200 frames ≈ 6.7s at 30fps,
    # covering the typical serve-toss timing range.
    late_entry_scan_frames = max(effective_window, 200)
    if formation_side != "near":
        late_side, late_conf = _detect_near_side_late_entry(
            player_positions, net_y, 0, late_entry_scan_frames,
        )
        if late_side is not None:
            return late_side, max(late_conf, formation_conf)

    if formation_side is not None:
        return formation_side, formation_conf
    return None, 0.0


def _detect_near_side_late_entry(
    player_positions: list[PlayerPosition],
    net_y: float,
    start_frame: int,
    window_frames: int,
) -> tuple[str | None, float]:
    """Detect a near-side player entering the frame late at an edge.

    In beach volleyball with a fixed camera, the near-side server often
    starts off-screen (behind/below the camera) and walks into frame
    during the serve toss.  This creates a characteristic pattern: a
    track that first appears after frame 15 at the bottom or side edge
    of the frame, on the near side of the court.

    Only fires for **near-side** late entries — far-side late entries
    are noise (occlusion recovery, background tracking, etc.).  Validated
    on 401 rallies: +2 fixes, 0 regressions when gated to near-side only
    with first_frame in [16, 80].

    Returns ("near", confidence) when detected, (None, 0.0) otherwise.
    """
    # Tuned on 401 GT rallies (28 target videos, 2026-04-13).
    # min_first_frame=15: tracks starting earlier are present from rally start.
    # edge thresholds: near-side server enters at bottom (y>0.88) or
    #   side edges (x<0.08 or x>0.92) of the normalized frame.
    min_first_frame = 15
    max_first_frame = window_frames  # no artificial cap
    edge_x_threshold = 0.08
    edge_y_threshold = 0.88

    end_frame = start_frame + window_frames

    # Build per-track first-appearance info within the analysis window.
    track_first: dict[int, tuple[int, float, float]] = {}  # tid → (frame, x, foot_y)
    track_median_y: dict[int, float] = {}
    by_track_y: dict[int, list[float]] = defaultdict(list)
    for p in player_positions:
        if p.track_id < 0:
            continue
        if start_frame <= p.frame_number < end_frame:
            foot_y = p.y + p.height / 2.0
            by_track_y[p.track_id].append(foot_y)
            if p.track_id not in track_first or p.frame_number < track_first[p.track_id][0]:
                track_first[p.track_id] = (p.frame_number, p.x, foot_y)

    if len(by_track_y) < 2:
        return None, 0.0

    for tid, ys in by_track_y.items():
        ys_sorted = sorted(ys)
        track_median_y[tid] = ys_sorted[len(ys_sorted) // 2]

    # Determine effective court split (same logic as formation predictor).
    effective_split = net_y
    near_count = sum(1 for y in track_median_y.values() if y > effective_split)
    far_count = len(track_median_y) - near_count
    if near_count == 0 or far_count == 0:
        auto_split = _compute_auto_split_y([
            p for p in player_positions
            if p.track_id >= 0 and start_frame <= p.frame_number < end_frame
        ])
        if auto_split is not None:
            effective_split = auto_split

    # Find near-side tracks that enter late at a frame edge.
    near_late_entries: list[tuple[int, int]] = []  # (tid, first_frame)
    has_far_late = False
    for tid, (ff, fx, fy) in track_first.items():
        if ff <= min_first_frame or ff > max_first_frame:
            continue
        at_edge = (
            fx < edge_x_threshold
            or fx > 1.0 - edge_x_threshold
            or fy > edge_y_threshold
        )
        if not at_edge:
            continue
        is_near = track_median_y.get(tid, 0.5) > effective_split
        if is_near:
            near_late_entries.append((tid, ff))
        else:
            has_far_late = True

    # Only fire when near-side has late entries and far-side does not,
    # to avoid ambiguity when both sides have late-arriving tracks.
    if not near_late_entries or has_far_late:
        return None, 0.0

    max_ff = max(ff for _, ff in near_late_entries)
    confidence = min(1.0, max_ff / 60.0)
    return "near", confidence


def _is_ball_on_serve_side(
    ball_y: float,
    court_side: str,
    net_y: float,
    margin: float = 0.05,
) -> bool | None:
    """Check if ball position is on the expected side for a serve.

    A serve from the near side should have ball_y > net_y (near court);
    a serve from the far side should have ball_y < net_y (far court).

    Args:
        ball_y: Ball Y position at the contact.
        court_side: Court side of the serve ("near" or "far").
        net_y: Net Y position.
        margin: Dead zone around net_y where result is indeterminate.

    Returns:
        True if ball is on the correct side for a serve from court_side,
        False if ball is clearly on the wrong side,
        None if ball is near the net (indeterminate).
    """
    if court_side not in ("near", "far"):
        return None

    if court_side == "near":
        if ball_y > net_y + margin:
            return True
        elif ball_y < net_y - margin:
            return False
    else:  # far
        if ball_y < net_y - margin:
            return True
        elif ball_y > net_y + margin:
            return False

    return None  # Ball near the net — indeterminate


def _ball_starts_on_contact_side(
    ball_positions: list[BallPosition],
    frame: int,
    court_side: str,
    net_y: float,
    lookahead: int = 5,
) -> bool | None:
    """Check whether the ball starts on the contact's court side after a hit.

    Averages ball Y over the first *lookahead* frames after *frame*. A serve
    should show the ball on the server's side; a receive will show it on the
    opposite side because the ball has already crossed.

    Returns True/False, or None if there are no ball positions in the window.
    """
    positions = [
        bp for bp in ball_positions
        if frame < bp.frame_number <= frame + lookahead
    ]
    if not positions:
        return None
    avg_y = sum(bp.y for bp in positions) / len(positions)
    if court_side == "near":
        return avg_y >= net_y
    elif court_side == "far":
        return avg_y < net_y
    return None


def _is_ball_one_sided(
    ball_positions: list[BallPosition],
    up_to_frame: int,
    net_y: float,
    min_per_side: int = 3,
) -> bool:
    """Check if ball positions up to a frame are on one side of the net.

    When WASB misses the fast-moving serve ball, all detections start only
    after the ball crosses the net.  Every pre-contact position is on the
    receiver's side, making trajectory-based serve/receive discrimination
    unreliable.

    Args:
        ball_positions: Sorted list of ball positions (by frame_number).
        up_to_frame: Only consider positions at or before this frame.
        net_y: Net Y position.
        min_per_side: Minimum positions required on each side to consider
            the trajectory two-sided (default 3).

    Returns:
        True when fewer than *min_per_side* positions exist on either
        side (includes zero positions on both sides — no trajectory data).
        False when both sides have >= min_per_side positions.
    """
    n_far = 0
    n_near = 0
    for bp in ball_positions:
        if bp.frame_number > up_to_frame:
            break  # ball_positions sorted by frame_number
        if bp.y < net_y:
            n_far += 1
        else:
            n_near += 1
    return n_far < min_per_side or n_near < min_per_side


def _infer_serve_side(
    first_contact: Contact,
    ball_positions: list[BallPosition] | None = None,
    net_y: float = 0.5,
) -> str | None:
    """Infer which side served when no serve contact was detected.

    Two signals in priority order:
    1. First contact's court_side → serve is from the opposite side
       (if we see a receive, serve came from the other court).
    2. Early ball trajectory → if ball moves near→far (decreasing Y toward
       far baseline), near side served; far→near means far side served.

    Returns "near", "far", or None if undecidable.
    """
    # Signal 1: opposite of first contact's court side (strongest)
    if first_contact.court_side in ("near", "far"):
        return "far" if first_contact.court_side == "near" else "near"

    # Signal 2: early ball trajectory direction
    if ball_positions:
        early = [
            bp for bp in ball_positions
            if bp.frame_number <= first_contact.frame
        ]
        if len(early) >= 3:
            first_y = early[0].y
            last_y = early[-1].y
            if first_y > net_y and last_y < first_y:
                return "near"  # Ball started near, moved toward far
            elif first_y < net_y and last_y > first_y:
                return "far"  # Ball started far, moved toward near

    return None


def _make_synthetic_serve(
    serve_side: str,
    first_contact_frame: int,
    net_y: float,
    rally_start_frame: int | None = None,
    server_track_id: int = -1,
) -> ClassifiedAction:
    """Create a synthetic serve action for a missed serve.

    Places the serve at the rally start frame when available and
    reasonably close to the first detected contact, otherwise ~1s
    (30 frames) before the first contact.

    When server_track_id is provided (from position-based server
    detection), the synthetic serve becomes an attributed contact
    that anchors the action chain — enabling team-seeded
    reattribution and correct downstream action classification.

    Args:
        serve_side: Court side of the serve ("near" or "far").
        first_contact_frame: Frame of the first detected contact.
        net_y: Net Y position.
        rally_start_frame: Frame when the rally segment starts (from
            detection). Used for more accurate serve placement.
        server_track_id: Track ID of the server from position
            detection. -1 if server could not be identified (e.g.
            off-screen near-side serve).

    Returns:
        A synthetic ClassifiedAction for the serve.
    """
    baseline_near, baseline_far = _serve_baselines(net_y)

    # Use rally_start_frame if available and within ~3s (90 frames) of
    # first contact. Beyond that, the rally start may be unreliable.
    if (
        rally_start_frame is not None
        and rally_start_frame < first_contact_frame
        and (first_contact_frame - rally_start_frame) <= 90
    ):
        serve_frame = rally_start_frame
    else:
        serve_frame = max(0, first_contact_frame - 30)

    # Higher confidence when we have a real server identity from
    # position detection (game-structure inference + attribution).
    confidence = 0.55 if server_track_id >= 0 else 0.4

    return ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=serve_frame,
        ball_x=0.5,
        ball_y=baseline_near if serve_side == "near" else baseline_far,
        velocity=0.0,
        player_track_id=server_track_id,
        court_side=serve_side,
        confidence=confidence,
        is_synthetic=True,
    )


class ActionClassifier:
    """Volleyball action classifier for beach volleyball.

    Classifies detected ball contacts into volleyball actions using:
    - Heuristic rules for serve, receive, and block (structural actions)
    - Learned GBM classifier for dig/set/attack (when provided)
    - Rule-based touch counting as fallback (when no classifier)

    Rally flow:
    1. SERVE — first contact, from behind baseline
    2. RECEIVE — first non-block contact after serve
    3. SET — second contact on same side
    4. ATTACK — third contact on same side (or ball directed to other court)
    5. After ball crosses net, count resets:
       - DIG (1st contact) → SET (2nd) → ATTACK (3rd)
    6. BLOCK — contact at net immediately after opponent's attack
       (counts as 1st touch on blocker's side in beach volleyball)
    """

    def __init__(self, config: ActionClassifierConfig | None = None):
        self.config = config or ActionClassifierConfig()

    def classify_rally(
        self,
        contact_sequence: ContactSequence,
        rally_id: str = "",
        team_assignments: dict[int, int] | None = None,
        classifier: ActionTypeClassifier | None = None,
        match_team_assignments: dict[int, int] | None = None,
        calibrator: CourtCalibrator | None = None,
        camera_height: float = 0.0,
    ) -> RallyActions:
        """Classify all contacts in a rally into action types.

        Serve, receive, and block use heuristic rules. Remaining contacts
        (dig/set/attack) use the learned classifier when provided, or
        fall back to touch-count rules.

        Args:
            contact_sequence: Detected contacts from ContactDetector.
            rally_id: Optional rally identifier.
            team_assignments: Optional mapping of track_id → team (0=near/A, 1=far/B).
            classifier: Optional trained action type classifier for dig/set/attack.
            match_team_assignments: Optional high-confidence match-level team mapping
                (track_id → team). Used for team-aware touch counting.
            calibrator: Optional court calibrator for court-space server detection.

        Returns:
            RallyActions with classified actions.
        """
        if classifier is not None:
            from rallycut.tracking.action_type_classifier import (
                extract_action_features,
            )

        contacts = contact_sequence.contacts
        start_frame = contact_sequence.rally_start_frame

        if not contacts:
            return RallyActions(rally_id=rally_id)

        actions: list[ClassifiedAction] = []
        serve_detected = False
        serve_side: str | None = None  # Court side of the serve
        serve_track_id: int = -1  # Track ID of the server
        receive_detected = False  # Whether receive has been classified
        current_side: str | None = None  # Side with possession
        contact_count_on_side = 0
        last_action_type: ActionType | None = None

        ball_positions = contact_sequence.ball_positions or None

        # Position-based server detection: identify who the server is from
        # player positions at rally start (before ball tracking kicks in).
        server_pos_tid = -1
        server_pos_side = ""
        pos_conf = 0.0
        if contact_sequence.player_positions:
            server_pos_tid, server_pos_side, pos_conf = _find_server_by_position(
                contact_sequence.player_positions, start_frame,
                contact_sequence.net_y,
                calibrator=calibrator,
            )

        # Determine which contact is the serve (may be outside window if
        # ball tracking starts late — common with detector warmup).
        serve_index, serve_pass = self._find_serve_index(
            contacts, start_frame, contact_sequence.net_y,
            ball_positions=ball_positions,
            server_pos_tid=server_pos_tid,
        )

        # Classifier-assisted serve detection: when heuristic falls through to
        # Pass 3 (first-contact fallback), check if the learned classifier can
        # identify which early contact is actually the serve.
        if serve_pass == 3 and classifier is not None and classifier.is_trained:
            window = self.config.serve_window_frames
            for ci, c in enumerate(contacts[:min(3, len(contacts))]):
                if (c.frame - start_frame) >= window:
                    break
                feat = extract_action_features(
                    contact=c, index=ci, all_contacts=contacts,
                    ball_positions=ball_positions,
                    net_y=contact_sequence.net_y,
                    rally_start_frame=start_frame,
                    team_assignments=match_team_assignments,
                    player_positions=contact_sequence.player_positions or None,
                    calibrator=calibrator,
                    camera_height=camera_height,
                )
                pred_action, _conf = classifier.predict([feat])[0]
                if pred_action == "serve":
                    serve_index = ci
                    serve_pass = 4  # Classifier-assisted
                    logger.debug(
                        "Classifier identified serve at contact %d (frame %d)",
                        ci, c.frame,
                    )
                    break

        for i, contact in enumerate(contacts):
            action_type = ActionType.UNKNOWN
            confidence = self.config.low_confidence
            player_tid = contact.player_track_id
            action_court_side = contact.court_side

            # Check for block (must be at net, immediately after opponent's attack)
            if (
                contact.is_at_net
                and last_action_type == ActionType.ATTACK
                and i > 0
                and (contact.frame - contacts[i - 1].frame) <= self.config.block_max_frame_gap
                and contact.court_side != contacts[i - 1].court_side
            ):
                action_type = ActionType.BLOCK
                confidence = self.config.high_confidence
                # Block counts as 1st touch on blocker's side (beach volleyball)
                current_side = contact.court_side
                contact_count_on_side = 1
                actions.append(ClassifiedAction(
                    action_type=action_type,
                    frame=contact.frame,
                    ball_x=contact.ball_x,
                    ball_y=contact.ball_y,
                    velocity=contact.velocity,
                    player_track_id=contact.player_track_id,
                    court_side=contact.court_side,
                    confidence=confidence,
                ))
                last_action_type = action_type
                continue

            # Skip pre-serve contacts — don't let them corrupt possession state
            if not serve_detected and i != serve_index:
                actions.append(ClassifiedAction(
                    action_type=ActionType.UNKNOWN,
                    frame=contact.frame,
                    ball_x=contact.ball_x,
                    ball_y=contact.ball_y,
                    velocity=contact.velocity,
                    player_track_id=contact.player_track_id,
                    court_side=contact.court_side,
                    confidence=self.config.low_confidence,
                ))
                last_action_type = ActionType.UNKNOWN
                continue

            # Handle possession changes.  Detection priority:
            # 1. Team membership from match_team_assignments (most reliable)
            # 2. ball_crossed_net (endpoint displacement)
            # 3. court_side comparison (fallback)
            team_resolved = False
            if match_team_assignments and i > 0:
                cur_team = match_team_assignments.get(contact.player_track_id)
                prev_team = match_team_assignments.get(
                    contacts[i - 1].player_track_id,
                )
                if cur_team is not None and prev_team is not None:
                    team_resolved = True
                    if cur_team != prev_team:
                        current_side = contact.court_side
                        contact_count_on_side = 0

            if not team_resolved:
                crossed_net: bool | None = None
                if ball_positions and i > 0 and current_side is not None:
                    crossed_net = ball_crossed_net(
                        ball_positions,
                        from_frame=contacts[i - 1].frame,
                        to_frame=contact.frame,
                        net_y=contact_sequence.net_y,
                    )
                if crossed_net is True:
                    current_side = contact.court_side
                    contact_count_on_side = 0
                elif contact.court_side != current_side:
                    current_side = contact.court_side
                    contact_count_on_side = 0

            contact_count_on_side += 1

            # Safety valve: beach volleyball allows max 3 touches per side.
            # If counter exceeds this, a net crossing was missed (e.g., ball
            # trajectory stays visually below net_y due to camera angle).
            # Reset to 1 to resume the dig→set→attack cycle.
            if contact_count_on_side > 3:
                current_side = contact.court_side
                contact_count_on_side = 1

            # Rule-based classification
            if not serve_detected:
                if i == serve_index:
                    # For Pass 3 fallback serves (first-contact guess), verify
                    # with trajectory: ball should move toward net after a serve.
                    # If not, the real serve was likely missed and this contact
                    # is actually the receive. Pass 1/2 serves are more reliable
                    # (arc crossing or baseline/velocity) so skip the check.
                    is_phantom = False
                    # When all pre-contact ball positions are on one side
                    # of the net AND server position is confidently known,
                    # WASB missed the serve trajectory (fast ball near the
                    # server). Trajectory-based phantom checks are
                    # unreliable — skip them and trust server position.
                    suppress_phantom = (
                        serve_pass == 3
                        and ball_positions is not None
                        and _is_ball_one_sided(
                            ball_positions, contact.frame,
                            contact_sequence.net_y,
                        )
                        and server_pos_tid >= 0
                        and pos_conf >= 0.7
                    )
                    # Also require the server to have a contact in the
                    # serve window — if not, the "serve" is likely a
                    # phantom (the real server wasn't detected).
                    if suppress_phantom:
                        suppress_phantom = any(
                            c.player_track_id == server_pos_tid
                            for c in contacts[:min(3, len(contacts))]
                            if (c.frame - start_frame)
                            < self.config.serve_window_frames
                        )
                    if suppress_phantom:
                        logger.debug(
                            "One-sided ball trajectory + server "
                            "position (tid=%d, conf=%.2f) — "
                            "suppressing phantom check at frame %d",
                            server_pos_tid, pos_conf, contact.frame,
                        )
                    elif serve_pass == 3 and ball_positions:
                        toward_net = _ball_moving_toward_net(
                            ball_positions, contact.frame,
                            contact.ball_y, contact_sequence.net_y,
                        )
                        if toward_net is False:
                            is_phantom = True
                        elif (
                            toward_net is None
                            and classifier is not None
                            and classifier.is_trained
                        ):
                            # Insufficient trajectory data for Pass 3
                            # fallback — treat as phantom. The classifier
                            # override below rescues true serves.
                            is_phantom = True
                            logger.debug(
                                "Insufficient trajectory data for Pass 3 "
                                "serve at frame %d — treating as phantom "
                                "(pending classifier check)",
                                contact.frame,
                            )

                    # Pre-contact ball trajectory: check where the ball
                    # was in the ~1.5s before the first contact.  This
                    # window captures the serve flight (ball crossing
                    # from server to receiver) while excluding stale
                    # detections from before the serve (ball at rest,
                    # previous rally, etc.).
                    # Post-contact trajectory can't distinguish serve
                    # from receive (both send ball toward net), but
                    # pre-contact origin is unambiguous.
                    if (
                        not is_phantom
                        and serve_pass == 3
                        and not suppress_phantom
                        and ball_positions
                    ):
                        flight_window = 45  # ~1.5s at 30fps
                        pre = [
                            bp for bp in ball_positions
                            if (contact.frame - flight_window
                                <= bp.frame_number < contact.frame)
                        ]
                        if len(pre) >= 3:
                            margin = 0.05
                            net = contact_sequence.net_y
                            # Use the EARLIEST positions in the window
                            # (where the ball is still on the server's
                            # side, before crossing).  The median of all
                            # positions is dominated by the later half
                            # (ball already on receiver's side) and
                            # washes out the crossing signal.
                            early_ys = [bp.y for bp in pre[:5]]
                            med_y = sorted(early_ys)[
                                len(early_ys) // 2
                            ]
                            started_near = med_y > net + margin
                            started_far = med_y < net - margin
                            if (
                                (started_far
                                 and contact.court_side == "near")
                                or (started_near
                                    and contact.court_side == "far")
                            ):
                                is_phantom = True
                                logger.debug(
                                    "Ball flight from opposite side "
                                    "(median_y=%.3f, net=%.3f, %d pts"
                                    " in %d-frame window) vs contact "
                                    "side (%s) at frame %d — phantom",
                                    med_y, net, len(pre),
                                    flight_window,
                                    contact.court_side,
                                    contact.frame,
                                )

                    # Court-side check for Pass 3 fallback: if ball is
                    # clearly on the wrong side of the net for a serve
                    # from this court side, it's likely a receive.
                    # Skipped when suppress_phantom — ball position data
                    # is unreliable (all on one side of net).
                    if (
                        not is_phantom
                        and serve_pass == 3
                        and not suppress_phantom
                    ):
                        on_serve_side = _is_ball_on_serve_side(
                            contact.ball_y, contact.court_side,
                            contact_sequence.net_y,
                        )
                        if on_serve_side is False:
                            is_phantom = True

                    # Classifier arbitration for Pass 3 serves: both
                    # rescue (phantom → real) and reject (real → phantom).
                    if (
                        serve_pass == 3
                        and classifier is not None
                        and classifier.is_trained
                    ):
                        feat = extract_action_features(
                            contact=contact, index=i, all_contacts=contacts,
                            ball_positions=ball_positions,
                            net_y=contact_sequence.net_y,
                            rally_start_frame=start_frame,
                            team_assignments=match_team_assignments,
                            player_positions=contact_sequence.player_positions or None,
                            calibrator=calibrator,
                            camera_height=camera_height,
                        )
                        pred_action, pred_conf = classifier.predict([feat])[0]
                        if is_phantom and pred_action == "serve":
                            # Rescue: classifier says serve, override phantom
                            is_phantom = False
                            logger.debug(
                                "Classifier overrides phantom serve at "
                                "frame %d (conf=%.2f)",
                                contact.frame, pred_conf,
                            )
                        elif (
                            not is_phantom
                            and pred_action != "serve"
                            and pred_conf > 0.7
                        ):
                            # Reject: classifier confidently says non-serve
                            is_phantom = True
                            logger.debug(
                                "Classifier rejects Pass 3 serve at "
                                "frame %d (pred=%s, conf=%.2f)",
                                contact.frame, pred_action, pred_conf,
                            )

                    if not is_phantom:
                        # Normal serve classification
                        is_in_window = (
                            (contact.frame - start_frame)
                            < self.config.serve_window_frames
                        )
                        action_type = ActionType.SERVE
                        confidence = (
                            self.config.high_confidence if is_in_window
                            else self.config.medium_confidence
                        )
                        serve_detected = True
                        if suppress_phantom:
                            # suppress_phantom is True only when all
                            # phantom checks were skipped, so
                            # is_phantom is always False here.
                            # Ball tracking missed the serve trajectory.
                            # Override court_side and attribution with
                            # server position (ball-derived values are
                            # unreliable — ball was only visible after
                            # crossing the net).
                            serve_side = server_pos_side
                            current_side = server_pos_side
                            action_court_side = server_pos_side
                            serve_track_id = server_pos_tid
                            player_tid = server_pos_tid
                        else:
                            serve_side = contact.court_side
                            current_side = contact.court_side
                            serve_track_id = contact.player_track_id
                        contact_count_on_side = 1
                    else:
                        # Phantom serve: real serve was missed, this is the
                        # receive. Infer serve came from the opposite side.
                        # Position-based server_side is most reliable when
                        # available; fall back to trajectory / court_side.
                        if server_pos_side:
                            serve_side = server_pos_side
                        else:
                            serve_side = _infer_serve_side(
                                contact, ball_positions,
                                contact_sequence.net_y,
                            ) or (
                                "far" if contact.court_side == "near"
                                else "near"
                            )
                        # Prepend synthetic serve action
                        synth = _make_synthetic_serve(
                            serve_side, contact.frame,
                            contact_sequence.net_y,
                            rally_start_frame=start_frame,
                            server_track_id=server_pos_tid,
                        )
                        actions.append(synth)
                        action_type = ActionType.RECEIVE
                        confidence = self.config.medium_confidence
                        serve_detected = True
                        receive_detected = True
                        current_side = contact.court_side
                        contact_count_on_side = 1
                else:
                    action_type = ActionType.UNKNOWN
                    confidence = self.config.low_confidence

            elif not receive_detected and serve_side is not None:
                # First non-block contact after serve is always the receive.
                # No court_side guard needed: if a rare FP contact between
                # serve and real receive gets this label, Rule 4 (duplicate
                # receives → set) in repair_action_sequence handles it.
                action_type = ActionType.RECEIVE
                confidence = self.config.high_confidence
                receive_detected = True
                current_side = contact.court_side
                contact_count_on_side = 1

                # Server can't receive their own serve — re-attribute to
                # next-nearest candidate if the nearest player is the server.
                # Use local variable to avoid mutating the Contact object.
                player_tid = contact.player_track_id
                if (
                    player_tid == serve_track_id
                    and serve_track_id >= 0
                    and contact.player_candidates
                ):
                    for cand_tid, _cand_dist in contact.player_candidates:
                        if cand_tid != serve_track_id:
                            player_tid = cand_tid
                            break

            elif classifier is not None and classifier.is_trained:
                # Learned classifier for dig/set/attack
                feat = extract_action_features(
                    contact=contact, index=i, all_contacts=contacts,
                    ball_positions=ball_positions,
                    net_y=contact_sequence.net_y,
                    rally_start_frame=start_frame,
                    team_assignments=match_team_assignments,
                    player_positions=contact_sequence.player_positions or None,
                    calibrator=calibrator,
                    camera_height=camera_height,
                )
                pred_action, pred_conf = classifier.predict([feat])[0]
                try:
                    action_type = ActionType(pred_action)
                except ValueError:
                    action_type = ActionType.UNKNOWN
                confidence = pred_conf

            else:
                # Touch-count fallback (no classifier)
                if contact_count_on_side == 1:
                    action_type = ActionType.DIG
                    confidence = self.config.medium_confidence
                elif contact_count_on_side == 2:
                    # 2nd touch is usually a set, but allow attack when
                    # ball crosses net (pipe attack / overpass hit).
                    crosses = None
                    if ball_positions:
                        crosses = ball_crossed_net(
                            ball_positions, contact.frame,
                            contact.frame + 15, contact_sequence.net_y,
                        )
                    if crosses is True:
                        action_type = ActionType.ATTACK
                    else:
                        action_type = ActionType.SET
                    confidence = self.config.high_confidence
                elif contact_count_on_side >= 3:
                    action_type = ActionType.ATTACK
                    confidence = self.config.high_confidence

            # Modulate confidence with contact classifier confidence.
            if contact.confidence > 0:
                confidence = min(confidence, contact.confidence)

            actions.append(ClassifiedAction(
                action_type=action_type,
                frame=contact.frame,
                ball_x=contact.ball_x,
                ball_y=contact.ball_y,
                velocity=contact.velocity,
                player_track_id=player_tid,
                court_side=action_court_side,
                confidence=confidence,
            ))
            last_action_type = action_type

        # --- Second pass: re-predict dig/set/attack with prev-action context ---
        _relabel_types = {ActionType.DIG, ActionType.SET, ActionType.ATTACK}
        if classifier is not None and classifier.is_trained:
            from rallycut.tracking.action_type_classifier import set_prev_action_context

            # Build contact index lookup: action index → contact index
            action_to_contact: dict[int, int] = {}
            ci = 0
            for ai, action in enumerate(actions):
                if action.is_synthetic:
                    continue
                if ci < len(contacts) and contacts[ci].frame == action.frame:
                    action_to_contact[ai] = ci
                    ci += 1

            n_relabeled = 0
            for ai, action in enumerate(actions):
                if action.action_type not in _relabel_types:
                    continue
                ci_idx = action_to_contact.get(ai)
                if ci_idx is None:
                    continue

                # Find previous non-unknown/non-block action for context
                prev_action_str = "unknown"
                prev_conf = 0.0
                prev_side: str | None = None
                for j in range(ai - 1, -1, -1):
                    prev_a = actions[j]
                    if prev_a.action_type not in (ActionType.UNKNOWN, ActionType.BLOCK):
                        prev_action_str = prev_a.action_type.value
                        prev_conf = prev_a.confidence
                        prev_side = prev_a.court_side
                        break

                same_side: bool | None = None
                if prev_side and action.court_side in ("near", "far"):
                    same_side = prev_side == action.court_side

                feat = extract_action_features(
                    contact=contacts[ci_idx], index=ci_idx,
                    all_contacts=contacts,
                    ball_positions=ball_positions,
                    net_y=contact_sequence.net_y,
                    rally_start_frame=start_frame,
                    team_assignments=match_team_assignments,
                    player_positions=contact_sequence.player_positions or None,
                    calibrator=calibrator,
                    camera_height=camera_height,
                )
                set_prev_action_context(feat, prev_action_str, prev_conf, same_side)

                pred_action, pred_conf = classifier.predict([feat])[0]
                try:
                    new_type = ActionType(pred_action)
                except ValueError:
                    continue

                if new_type != action.action_type:
                    actions[ai] = ClassifiedAction(
                        action_type=new_type,
                        frame=action.frame,
                        ball_x=action.ball_x,
                        ball_y=action.ball_y,
                        velocity=action.velocity,
                        player_track_id=action.player_track_id,
                        court_side=action.court_side,
                        confidence=pred_conf,
                    )
                    n_relabeled += 1

            if n_relabeled > 0:
                logger.debug(
                    "Second pass: re-labeled %d/%d actions with prev-action context",
                    n_relabeled, len(actions),
                )

        # Stamp team labels on all actions
        if team_assignments:
            for action in actions:
                if action.player_track_id >= 0:
                    action.team = _team_label(action.player_track_id, team_assignments)
                elif action.court_side in ("near", "far"):
                    # Synthetic serves have player_track_id=-1; derive from court_side
                    action.team = "A" if action.court_side == "near" else "B"

        result = RallyActions(
            actions=actions, rally_id=rally_id,
            team_assignments=team_assignments or {},
        )

        if actions:
            seq = [a.action_type.value for a in actions]
            mode = "learned-2pass" if classifier is not None else "rule-based"
            logger.info(
                f"Rally {rally_id}: classified {len(actions)} actions "
                f"({mode}): {seq}"
            )

        return result


    def _find_serve_index(
        self,
        contacts: list[Contact],
        start_frame: int,
        net_y: float = 0.5,
        ball_positions: list[BallPosition] | None = None,
        server_pos_tid: int = -1,
    ) -> tuple[int, int]:
        """Find which contact is the serve.

        Uses four passes with decreasing strictness:
        0. Position-based: matches a pre-computed server track_id (from
           ``_find_server_by_position``) to a contact in the serve window.
        1. Arc-based: first contact in window whose subsequent trajectory crosses
           the net (distinctive serve arc pattern).
        2. Position/velocity: baseline position or high velocity (original heuristic).
        3. Fallback: first contact is the serve.

        Args:
            server_pos_tid: Server track_id from position-based detection.
                -1 when no server was identified from positions.

        Returns:
            Tuple of (index into contacts list, pass number that found it).
            Pass number: 0=position, 1=arc, 2=position/velocity, 3=fallback.
            Index is -1 if no serve found.
        """
        window = self.config.serve_window_frames

        baseline_near, baseline_far = _serve_baselines(net_y)

        # Pass 0: Position-based — match pre-computed server track_id to a
        # contact. Only matches by track_id (not court_side) to avoid FP
        # where the first detected contact is actually the receive.
        if server_pos_tid >= 0 and contacts:
            for i, c in enumerate(contacts):
                if (c.frame - start_frame) >= window:
                    break
                if c.player_track_id == server_pos_tid:
                    logger.debug(
                        "Serve detected via player position at frame %d "
                        "(track %d)",
                        c.frame, server_pos_tid,
                    )
                    return i, 0

        # Pass 1: Arc-based serve detection — check only the first 2 contacts
        # (serve is always at the start of the rally). A serve initiates a
        # trajectory that crosses the net. Limit to first 2 to avoid false
        # positives from mid-rally attacks that also cross the net.
        if ball_positions and len(contacts) >= 2:
            max_arc_candidates = min(2, len(contacts))
            for i in range(max_arc_candidates):
                c = contacts[i]
                if (c.frame - start_frame) >= window:
                    break
                # Check if ball crosses net between this and next contact
                next_frame = (
                    contacts[i + 1].frame if i + 1 < len(contacts)
                    else c.frame + window
                )
                if ball_crossed_net(ball_positions, c.frame, next_frame, net_y) is True:
                    # Validate crossing direction: after a serve, the ball
                    # should start on the server's court side. If the ball
                    # immediately ends up on the opposite side, this contact
                    # is likely a receive, not a serve.
                    on_side = _ball_starts_on_contact_side(
                        ball_positions, c.frame, c.court_side, net_y,
                    )
                    if on_side is False:
                        logger.debug(
                            "Skipping arc serve candidate at frame %d: "
                            "ball starts on opposite side (side=%s, net_y=%.3f)",
                            c.frame, c.court_side, net_y,
                        )
                        continue
                    logger.debug(
                        "Serve detected via arc crossing at frame %d (index %d)",
                        c.frame, i,
                    )
                    return i, 1

        # Pass 2: Position/velocity heuristic within window
        for i, c in enumerate(contacts):
            if (c.frame - start_frame) >= window:
                break
            is_at_baseline = c.ball_y >= baseline_near or c.ball_y <= baseline_far
            if is_at_baseline:
                # Verify ball moves toward net (a serve should).
                # Only reject on confirmed False (away from net = receive).
                # None (insufficient data) and True both accept as serve.
                if ball_positions:
                    toward = _ball_moving_toward_net(
                        ball_positions, c.frame, c.ball_y, net_y,
                    )
                    if toward is False:
                        continue  # Ball moving away from net — likely a receive
                return i, 2
            if c.velocity >= self.config.serve_min_velocity:
                # With ball trajectory, also require ball moving toward net
                # AND ball starting on the contact's court side (rejects
                # receives where the ball immediately crosses to the other
                # side after a high-velocity contact).
                if ball_positions:
                    if _ball_moving_toward_net(
                        ball_positions, c.frame, c.ball_y, net_y,
                    ) is False:
                        continue
                    on_side = _ball_starts_on_contact_side(
                        ball_positions, c.frame, c.court_side, net_y,
                    )
                    if on_side is False:
                        continue
                    return i, 2
                else:
                    return i, 2

        # Pass 3: fallback — first contact is the serve
        if self.config.serve_fallback and contacts:
            logger.info(
                "No serve in %d-frame window, using first contact (frame %d) "
                "as serve fallback",
                window, contacts[0].frame,
            )
            return 0, 3

        return -1, 0


# Max repairs per rally before the circuit breaker stops fixing.
# Heavily broken sequences get worse with cascading local fixes.
_MAX_SEQUENCE_REPAIRS = 3

# Rules disabled by default (ablation showed they hurt accuracy).
# Rule 2: recv/dig→attack → set (-2.6pp action accuracy)
_RULES_DISABLED_BY_DEFAULT: set[int] = {2}


def _reclassify(action: ClassifiedAction, new_type: ActionType) -> ClassifiedAction:
    """Create a copy of *action* with a different action_type.

    Confidence is capped at 0.6 to signal that this label was inferred by
    the repair pass, not the original classifier.
    """
    return ClassifiedAction(
        action_type=new_type,
        frame=action.frame,
        ball_x=action.ball_x,
        ball_y=action.ball_y,
        velocity=action.velocity,
        player_track_id=action.player_track_id,
        court_side=action.court_side,
        confidence=min(action.confidence, 0.6),
        is_synthetic=action.is_synthetic,
        team=action.team,
    )


@overload
def repair_action_sequence(
    actions: list[ClassifiedAction],
    net_y: float = ...,
    ball_positions: list[BallPosition] | None = ...,
    rally_start_frame: int | None = ...,
    server_track_id: int = ...,
    disabled_rules: None = ...,
) -> list[ClassifiedAction]: ...


@overload
def repair_action_sequence(
    actions: list[ClassifiedAction],
    net_y: float = ...,
    ball_positions: list[BallPosition] | None = ...,
    rally_start_frame: int | None = ...,
    server_track_id: int = ...,
    disabled_rules: set[int] = ...,
) -> tuple[list[ClassifiedAction], dict[int, int]]: ...


def repair_action_sequence(
    actions: list[ClassifiedAction],
    net_y: float = 0.5,
    ball_positions: list[BallPosition] | None = None,
    rally_start_frame: int | None = None,
    server_track_id: int = -1,
    disabled_rules: set[int] | None = None,
) -> list[ClassifiedAction] | tuple[list[ClassifiedAction], dict[int, int]]:
    """Repair volleyball-illegal action sequences.

    Missed contacts cause cascade failures in the state machine: missing one
    contact shifts ALL subsequent labels. This function detects and fixes
    common illegal patterns using volleyball rules as constraints.

    Same-side rules only — cross-side sequences (e.g., dig on near → dig on
    far) are legal because each side's touch counter resets independently.

    Repairs applied (in order):
    0. (serve fix) Non-synthetic serve with ball on wrong court side →
       reclassify as receive and prepend a synthetic serve.
       Skipped when ball trajectory is one-sided (WASB missed serve).
    3. (pre-pass) Duplicate serves → extras become dig.
    4. (pre-pass) Duplicate receives → extras become set.
    1. (main pass) Consecutive receives/digs on same side → second becomes set.
    2. (main pass) Receive/dig → attack on same side → attack becomes set
       (only when a subsequent action exists as the actual attack).
    5. (main pass) Set → set on same side → second becomes attack.
    6. (main pass) Attack → attack on same side → first becomes set.

    Circuit breaker: stops after 3 repairs to avoid cascading bad rewrites
    on heavily broken sequences. Rule 0 (wrong-side serve) does not count
    toward the circuit breaker since it is structurally necessary.

    Only repairs non-block/non-unknown actions.

    Args:
        actions: Classified actions from classify_rally.
        net_y: Net Y position.
        ball_positions: Ball positions for one-sided trajectory checks.
        rally_start_frame: Rally start frame for synthetic serve placement.
        server_track_id: Server track ID for synthetic serve attribution.
        disabled_rules: Set of rule numbers (0-6) to skip. None = all enabled.
            When provided, the return type changes to include trigger counts.

    Returns:
        When disabled_rules is None: repaired list of ClassifiedAction.
        When disabled_rules is provided: tuple of (repaired actions,
        dict mapping rule number → trigger count).
    """
    _track_triggers = disabled_rules is not None
    _disabled = (disabled_rules if disabled_rules is not None
                 else _RULES_DISABLED_BY_DEFAULT)
    _triggers: dict[int, int] = {r: 0 for r in range(9)}

    if len(actions) < 2:
        if _track_triggers:
            return actions, _triggers
        return actions

    # Work on a mutable copy
    repaired = list(actions)
    repair_count = 0

    # Find the serve index
    serve_idx: int | None = None
    for i, a in enumerate(repaired):
        if a.action_type == ActionType.SERVE:
            serve_idx = i
            break

    if serve_idx is None:
        if _track_triggers:
            return repaired, _triggers
        return repaired  # Can't repair without a serve anchor

    # ------------------------------------------------------------------
    # Rule 0: Non-synthetic serve with ball on wrong court side.
    # Does NOT count toward circuit breaker (structurally necessary).
    # ------------------------------------------------------------------
    serve = repaired[serve_idx]
    if not serve.is_synthetic:
        on_serve_side = _is_ball_on_serve_side(
            serve.ball_y, serve.court_side, net_y,
        )
        if on_serve_side is False:
            skip_rule0 = (
                ball_positions is not None
                and _is_ball_one_sided(
                    ball_positions, serve.frame, net_y,
                    min_per_side=5,
                )
            )
            if skip_rule0:
                logger.debug(
                    "Rule 0 skipped: one-sided ball trajectory "
                    "at serve frame %d",
                    serve.frame,
                )
            elif 0 not in _disabled:
                _triggers[0] += 1
                opposite = "far" if serve.court_side == "near" else "near"
                synthetic = _make_synthetic_serve(
                    opposite, serve.frame, net_y,
                    rally_start_frame=rally_start_frame,
                    server_track_id=server_track_id,
                )
                repaired[serve_idx] = _reclassify(serve, ActionType.RECEIVE)
                repaired.insert(serve_idx, synthetic)
                logger.debug(
                    "Repair rule 0: serve at f%d on wrong court side "
                    "(%s, ball_y=%.2f, net_y=%.2f) → reclassified as "
                    "receive, synthetic serve prepended",
                    serve.frame, serve.court_side, serve.ball_y, net_y,
                )

    # Re-find serve_idx after possible insertion
    serve_idx = None
    for i, a in enumerate(repaired):
        if a.action_type == ActionType.SERVE:
            serve_idx = i
            break
    if serve_idx is None:
        if _track_triggers:
            return repaired, _triggers
        return repaired

    # ------------------------------------------------------------------
    # Rule 3 (pre-pass): Duplicate non-synthetic serves → extras become receive.
    # The second detected "serve" is almost certainly the serve return.
    # Only non-synthetic serves count; the first real serve is the anchor.
    # ------------------------------------------------------------------
    first_real_serve_found = False
    if 3 not in _disabled:
        for i, a in enumerate(repaired):
            if a.action_type == ActionType.SERVE and not a.is_synthetic:
                if not first_real_serve_found:
                    first_real_serve_found = True
                else:
                    if repair_count >= _MAX_SEQUENCE_REPAIRS:
                        break
                    _triggers[3] += 1
                    repaired[i] = _reclassify(a, ActionType.RECEIVE)
                    repair_count += 1
                    logger.debug(
                        "Repair rule 3: duplicate serve at f%d → receive",
                        a.frame,
                    )

    # ------------------------------------------------------------------
    # Rule 4 (pre-pass): Duplicate receives → extras become set.
    # Always use set (2nd touch) — court_side labels are too unreliable
    # to distinguish same-side (set) from cross-side (dig).
    # ------------------------------------------------------------------
    first_receive_found = False
    if 4 not in _disabled:
        for i, a in enumerate(repaired):
            if a.action_type == ActionType.RECEIVE:
                if not first_receive_found:
                    first_receive_found = True
                else:
                    if repair_count >= _MAX_SEQUENCE_REPAIRS:
                        break
                    _triggers[4] += 1
                    repaired[i] = _reclassify(a, ActionType.SET)
                    repair_count += 1
                    logger.debug(
                        "Repair rule 4: duplicate receive at f%d → set",
                        a.frame,
                    )

    # ------------------------------------------------------------------
    # Rule 8 (pre-pass): Dig immediately after serve → receive.
    # Dig is only valid after an opponent's attack, not after serve.
    # ------------------------------------------------------------------
    if 8 not in _disabled:
        for i in range(1, len(repaired)):
            if repair_count >= _MAX_SEQUENCE_REPAIRS:
                break
            if (
                repaired[i].action_type == ActionType.DIG
                and repaired[i - 1].action_type == ActionType.SERVE
            ):
                _triggers[8] += 1
                repaired[i] = _reclassify(repaired[i], ActionType.RECEIVE)
                repair_count += 1
                logger.debug(
                    "Repair rule 8: dig after serve at f%d → receive",
                    repaired[i].frame,
                )

    # ------------------------------------------------------------------
    # Main pass: Rules 1, 2, 5, 6
    # ------------------------------------------------------------------
    i = serve_idx + 1
    while i < len(repaired):
        if repair_count >= _MAX_SEQUENCE_REPAIRS:
            break

        a = repaired[i]

        # Skip blocks and unknowns — don't touch them
        if a.action_type in (ActionType.BLOCK, ActionType.UNKNOWN):
            i += 1
            continue

        # Look at previous non-block/non-unknown action
        prev_idx: int | None = None
        for j in range(i - 1, -1, -1):
            if repaired[j].action_type not in (ActionType.BLOCK, ActionType.UNKNOWN):
                prev_idx = j
                break

        if prev_idx is None:
            i += 1
            continue

        prev = repaired[prev_idx]

        # Time-gap guard: if >90 frames (3s at 30fps) between contacts,
        # a contact was likely missed and same-side assumptions break down.
        if a.frame - prev.frame > 90:
            i += 1
            continue

        same_side = (
            prev.court_side == a.court_side
            and a.court_side in ("near", "far")
        )

        # Rule 1: Two consecutive receives or digs on same side → second is set
        if (
            same_side
            and prev.action_type in (ActionType.RECEIVE, ActionType.DIG)
            and a.action_type == prev.action_type
            and 1 not in _disabled
        ):
            _triggers[1] += 1
            repaired[i] = _reclassify(a, ActionType.SET)
            repair_count += 1
            logger.debug(
                "Repair rule 1: %s→%s at f%d→f%d, changed second to set",
                prev.action_type.value, a.action_type.value,
                prev.frame, a.frame,
            )

        # Rule 2: receive/dig directly followed by attack on same side
        # with no set → reclassify the attack as set (if there's another
        # action after).
        # DISABLED BY DEFAULT: dig→attack is legal in beach volleyball
        # (pipe attacks, overpass hits). Ablation showed -2.6pp action acc.
        elif (
            same_side
            and prev.action_type in (ActionType.RECEIVE, ActionType.DIG)
            and a.action_type == ActionType.ATTACK
            and 2 not in _disabled
        ):
            next_idx: int | None = None
            for k in range(i + 1, len(repaired)):
                if repaired[k].action_type not in (
                    ActionType.BLOCK, ActionType.UNKNOWN,
                ):
                    next_idx = k
                    break
            if next_idx is not None:
                _triggers[2] += 1
                repaired[i] = _reclassify(a, ActionType.SET)
                repair_count += 1
                logger.debug(
                    "Repair rule 2: %s→attack at f%d→f%d → set "
                    "(next action at f%d)",
                    prev.action_type.value, prev.frame, a.frame,
                    repaired[next_idx].frame,
                )

        # Rule 5: set → set on same side → second becomes attack
        elif (
            same_side
            and prev.action_type == ActionType.SET
            and a.action_type == ActionType.SET
            and 5 not in _disabled
        ):
            _triggers[5] += 1
            repaired[i] = _reclassify(a, ActionType.ATTACK)
            repair_count += 1
            logger.debug(
                "Repair rule 5: set→set at f%d→f%d, second → attack",
                prev.frame, a.frame,
            )

        # Rule 6: attack → attack on same side → first becomes set
        elif (
            same_side
            and prev.action_type == ActionType.ATTACK
            and a.action_type == ActionType.ATTACK
            and 6 not in _disabled
        ):
            _triggers[6] += 1
            repaired[prev_idx] = _reclassify(prev, ActionType.SET)
            repair_count += 1
            logger.debug(
                "Repair rule 6: attack→attack at f%d→f%d, first → set",
                prev.frame, a.frame,
            )

        i += 1

    if _track_triggers:
        return repaired, _triggers
    return repaired


# Net-crossing actions: ball goes to opposite side
_NET_CROSSING_ACTIONS = {ActionType.SERVE, ActionType.ATTACK}
# Same-side actions: ball stays on same side
_SAME_SIDE_ACTIONS = {ActionType.RECEIVE, ActionType.SET, ActionType.DIG}

_HIGH_CONFIDENCE_GATE = 0.6
_MEDIUM_CONFIDENCE_GATE = 0.5
_SIDE_TO_TEAM: dict[str, int] = {"near": 0, "far": 1}


def assign_court_side_from_teams(
    actions: list[ClassifiedAction],
    team_assignments: dict[int, int],
) -> None:
    """Assign court_side directly from team membership (mutates in place).

    For each action with player_track_id in team_assignments:
    team 0 → "near", team 1 → "far".
    """
    n_assigned = 0
    for action in actions:
        if action.player_track_id < 0:
            continue
        team = team_assignments.get(action.player_track_id)
        if team is None:
            continue
        action.court_side = "near" if team == 0 else "far"
        n_assigned += 1
    if n_assigned > 0:
        logger.info(
            "Team-based court_side: assigned %d/%d actions", n_assigned, len(actions),
        )


def propagate_court_side(actions: list[ClassifiedAction]) -> list[ClassifiedAction]:
    """Propagate court_side using volleyball transition rules.

    Uses domain knowledge: serve/attack cross the net (next contact on
    opposite side), receive/set/dig stay on the same side.

    Runs after action classification, before repair_action_sequence().
    """
    if len(actions) < 2:
        return actions

    opposite = {"near": "far", "far": "near"}

    # Forward pass
    for i in range(len(actions) - 1):
        src = actions[i]
        tgt = actions[i + 1]

        if src.court_side not in ("near", "far"):
            continue
        if tgt.action_type in (ActionType.BLOCK, ActionType.UNKNOWN):
            continue
        if src.action_type == ActionType.BLOCK:
            continue

        if src.action_type in _NET_CROSSING_ACTIONS:
            expected = opposite[src.court_side]
            if src.confidence >= _HIGH_CONFIDENCE_GATE:
                # Fill unknown OR override disagreement
                if tgt.court_side != expected:
                    tgt.court_side = expected
        elif src.action_type in _SAME_SIDE_ACTIONS:
            if src.confidence >= _HIGH_CONFIDENCE_GATE:
                # High confidence: fill unknown or override disagreement —
                # consecutive same-side actions are unambiguous (same possession)
                if tgt.court_side != src.court_side:
                    tgt.court_side = src.court_side
            elif src.confidence >= _MEDIUM_CONFIDENCE_GATE:
                # Medium confidence: only fill unknown
                if tgt.court_side == "unknown":
                    tgt.court_side = src.court_side

    # Backward pass: fill unknown predecessors from known successors.
    # Use the PREDECESSOR's action type to determine the transition:
    # if predecessor was net-crossing, successor is on opposite side,
    # so predecessor = opposite(successor). If same-side, predecessor = successor.
    for i in range(len(actions) - 1, 0, -1):
        successor = actions[i]
        predecessor = actions[i - 1]

        if successor.court_side not in ("near", "far"):
            continue
        if predecessor.court_side != "unknown":
            continue
        if predecessor.action_type in (ActionType.BLOCK, ActionType.UNKNOWN):
            continue

        if predecessor.action_type in _NET_CROSSING_ACTIONS:
            # Predecessor crossed net → successor on opposite side → predecessor = opposite
            predecessor.court_side = opposite[successor.court_side]
        elif predecessor.action_type in _SAME_SIDE_ACTIONS:
            # Predecessor stayed same side → successor on same side → predecessor = same
            predecessor.court_side = successor.court_side

    return actions


def _reattribute_server_exclusion(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
) -> int:
    """Exclude server from receive attribution (no team data needed).

    Finds the serve action, then checks all RECEIVE actions. If a receive
    is attributed to the server, re-attributes to next-best candidate.
    This catches receives created by repair_action_sequence after the
    inline server exclusion already ran.

    Returns number of re-attributed actions.
    """
    serve_tid = -1
    for a in actions:
        if a.action_type == ActionType.SERVE and a.player_track_id >= 0:
            serve_tid = a.player_track_id
            break

    if serve_tid < 0:
        return 0

    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}
    n_fixed = 0

    for action in actions:
        if action.action_type != ActionType.RECEIVE:
            continue
        if action.player_track_id != serve_tid:
            continue

        contact = contact_by_frame.get(action.frame)
        if contact is None or not contact.player_candidates:
            continue

        for cand_tid, _cand_dist in contact.player_candidates:
            if cand_tid != serve_tid:
                action.player_track_id = cand_tid
                n_fixed += 1
                break

    return n_fixed


def _reattribute_reid(
    actions: list[ClassifiedAction],
    contact_by_frame: dict[int, Contact],
    reid_predictions: dict[int, dict[str, Any]],
    reid_min_margin: float,
) -> list[ClassifiedAction]:
    """Pass 3: Re-attribute using fine-tuned ReID classifier predictions.

    For each contact where the classifier confidently identifies a different
    player than proximity, re-attribute the action.
    """
    n_reid = 0
    for action in actions:
        if action.player_track_id < 0:
            continue

        reid_pred = reid_predictions.get(action.frame)
        if not reid_pred:
            continue

        best_tid = reid_pred.get("best_tid", -1)
        margin = reid_pred.get("margin", 0.0)

        if best_tid < 0 or best_tid == action.player_track_id:
            continue

        if margin < reid_min_margin:
            continue

        # Verify the candidate is in the contact's player list
        contact = contact_by_frame.get(action.frame)
        if contact is None:
            continue
        cand_tids = {tid for tid, _ in contact.player_candidates}
        if best_tid not in cand_tids:
            continue

        logger.debug(
            "ReID re-attribute frame %d: track %d -> track %d (margin %.3f)",
            action.frame, action.player_track_id, best_tid, margin,
        )
        action.player_track_id = best_tid
        n_reid += 1

    if n_reid > 0:
        logger.info("ReID re-attributed %d/%d actions", n_reid, len(actions))

    return actions


def _compute_expected_teams(
    actions: list[ClassifiedAction],
    team_assignments: dict[int, int],
) -> list[int | None]:
    """Derive expected team for each action from serve identity + action sequence.

    Uses volleyball rules: serve/attack cross the net (next contact on
    opposite team), receive/set/dig stay on same team. Block does not
    cross (defender reacts on same side). The server's team (from
    match-level team_assignments) seeds the chain.

    Synthetic serves (inserted by repair_action_sequence) set expected to
    serve_team and flip current_team to receiving team (serve crosses net).

    Returns list parallel to actions: expected team (0 or 1) per action,
    or None if not determinable (no serve found or unknown action type).
    """
    expected: list[int | None] = [None] * len(actions)

    # Find serve and its team
    serve_team: int | None = None
    for a in actions:
        if a.action_type == ActionType.SERVE and a.player_track_id >= 0:
            serve_team = team_assignments.get(a.player_track_id)
            break

    if serve_team is None:
        return expected

    current_team = serve_team
    for i, action in enumerate(actions):
        if action.action_type == ActionType.UNKNOWN:
            continue
        if action.is_synthetic:
            # Synthetic serves are inferred — trust the team chain.
            # Serve crosses the net, so flip to receiving team.
            if action.action_type == ActionType.SERVE:
                expected[i] = serve_team
                current_team = 1 - serve_team
            continue

        expected[i] = current_team

        # After net-crossing actions, flip to opposite team
        if action.action_type in _NET_CROSSING_ACTIONS:
            current_team = 1 - current_team

    return expected


def reattribute_players(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int] | None,
    max_distance_ratio: float = 1.5,
    reid_predictions: dict[int, dict[str, Any]] | None = None,
    reid_min_margin: float = 0.15,
) -> list[ClassifiedAction]:
    """Re-assign player attribution using type-aware rules and team signal.

    Three passes:
    1. Server exclusion (no team data needed): ensures RECEIVE actions are
       not attributed to the server. Catches cases missed by the inline
       check (e.g. receives created by repair_action_sequence).
    2. Team-based re-attribution (requires team data): derives expected
       team from server identity + action sequence (volleyball rules),
       then swaps players on the wrong team — or unmapped non-player
       tracks (spectator/ref IDs) — to best candidate on the correct
       team within distance cap. For unmapped tracks, falls back to the
       nearest mapped candidate on any team.
    3. ReID re-attribution (requires reid_predictions): for each contact,
       if the fine-tuned classifier is confident that a different candidate
       is the correct player, re-attribute.

    Args:
        actions: Classified actions to potentially re-attribute.
        contacts: Contact objects with player_candidates.
        team_assignments: Map of track_id → team (0=near, 1=far).
        max_distance_ratio: Maximum distance ratio for candidate (1.5 = candidate
            can be up to 50% farther than current player).
        reid_predictions: Map of contact frame → {track_id: player_id} predicted
            by the fine-tuned ReID classifier. When provided, enables Pass 3.
        reid_min_margin: Minimum probability margin for ReID re-attribution.
    """
    # Pass 1: server exclusion (always runs, no team data needed)
    n_server_fixes = _reattribute_server_exclusion(actions, contacts)
    if n_server_fixes > 0:
        logger.info(
            "Server exclusion: re-attributed %d receive(s)", n_server_fixes,
        )

    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    # Pass 2: team-based re-attribution (requires team data)
    if not team_assignments:
        # Skip Pass 2 but still run Pass 3 (ReID) if available
        if reid_predictions:
            return _reattribute_reid(actions, contact_by_frame, reid_predictions, reid_min_margin)
        return actions

    # Derive expected team per action from serve identity + action sequence.
    # This breaks the circular dependency: instead of deriving expected_team
    # from court_side (which depends on the player being evaluated), we use
    # the server's known team + volleyball transition rules.
    expected_teams = _compute_expected_teams(actions, team_assignments)

    # Fallback when serve chain unavailable: map court_side to team
    n_reattributed = 0

    for i, action in enumerate(actions):
        if action.confidence < 0.6:
            continue
        if action.player_track_id < 0:
            continue

        expected_team = expected_teams[i]
        if expected_team is None:
            # No serve-seeded chain available. The court_side fallback is
            # circular (court_side depends on the player being evaluated)
            # and empirically 0% accurate. Skip reattribution entirely.
            continue

        current_team = team_assignments.get(action.player_track_id)

        # Skip if current player is already on the correct team
        if current_team is not None and current_team == expected_team:
            continue

        # current_team is None → unmapped track (may be spectator/ref OR a
        #   real player without a cross-rally team assignment)
        # current_team != expected_team → wrong team
        # In both cases, try to find a better candidate on the expected team.
        is_unmapped = current_team is None

        contact = contact_by_frame.get(action.frame)
        if contact is None or not contact.player_candidates:
            continue

        # Guard: don't override the nearest candidate. Proximity is hard
        # physical evidence; the expected_team chain drifts when contacts
        # are missed or action types are wrong. Only override non-nearest
        # attributions (unmapped tracks or clear team mismatches).
        if (
            not is_unmapped
            and contact.player_candidates
            and contact.player_candidates[0][0] == action.player_track_id
        ):
            continue

        current_dist = contact.player_distance
        if not math.isfinite(current_dist):
            if is_unmapped:
                # Unmapped tracks with no distance info — allow any mapped
                # candidate to win, but only via the expected-team search
                # below (not the any-team fallback).
                current_dist = float("inf")
            else:
                continue

        # Find best candidate on the correct team within distance cap.
        # Unmapped tracks with a finite distance were found near the ball
        # and are likely real players without a cross-rally team assignment,
        # not spectators — apply the same distance cap as mapped tracks.
        # Only use infinite cap when the unmapped track has no distance info.
        if is_unmapped and not math.isfinite(current_dist):
            dist_cap = float("inf")
        else:
            dist_cap = max_distance_ratio * current_dist
        best_tid = -1
        best_dist = float("inf")
        for tid, dist in contact.player_candidates:
            if tid == action.player_track_id:
                continue
            cand_team = team_assignments.get(tid)
            if cand_team != expected_team:
                continue
            if dist <= dist_cap and dist < best_dist:
                best_tid = tid
                best_dist = dist

        # For unmapped tracks with no distance info: fall back to the
        # nearest mapped candidate on any team.  Only when we have no
        # proximity signal at all (inf distance) — when the unmapped track
        # had a finite distance it was likely a real player, not a spectator.
        if best_tid < 0 and is_unmapped and not math.isfinite(contact.player_distance):
            for tid, dist in contact.player_candidates:
                if tid == action.player_track_id:
                    continue
                if team_assignments.get(tid) is None:
                    continue  # also unmapped
                if dist < best_dist:
                    best_tid = tid
                    best_dist = dist

        if best_tid >= 0:
            logger.debug(
                "Re-attribute frame %d: track %d (team %s, dist %.3f) → "
                "track %d (dist %.3f) expected team %d%s",
                action.frame, action.player_track_id, current_team,
                current_dist, best_tid, best_dist, expected_team,
                " (was unmapped)" if is_unmapped else "",
            )
            action.player_track_id = best_tid
            n_reattributed += 1

    if n_reattributed > 0:
        logger.info("Re-attributed %d/%d actions using team signal",
                     n_reattributed, len(actions))

    # Pass 3: ReID re-attribution (requires reid_predictions)
    if reid_predictions:
        _reattribute_reid(actions, contact_by_frame, reid_predictions, reid_min_margin)

    return actions


def correct_team_from_propagation(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    propagated_sides: list[str],
    team_assignments: dict[int, int],
    pre_reattrib_tids: list[int],
    max_distance_ratio: float = 1.5,
) -> list[ClassifiedAction]:
    """Post-hoc team correction using propagated court_side signal.

    Targets actions where Pass 2 (reattribute_players) changed the player
    AND the original player's team matched the propagated court_side. These
    are cascade-error candidates: _compute_expected_teams() incorrectly
    flipped the expected team, causing Pass 2 to swap a correct player to
    the wrong team. Reverts to the original player.

    Falls back to candidate search when original player also doesn't match.

    Args:
        actions: Actions after reattribute_players() has run.
        contacts: Contact objects with player_candidates.
        propagated_sides: Court side per action from propagate_court_side(),
            captured before assign_court_side_from_teams() overwrites.
        team_assignments: Match-level track_id -> team mapping.
        pre_reattrib_tids: Player track IDs before reattribute_players(),
            used to identify which actions Pass 2 changed.
        max_distance_ratio: Candidate must be within this ratio of the
            current player's distance (1.5 = up to 50% farther).
    """
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    n_corrected = 0

    for i, action in enumerate(actions):
        if action.player_track_id < 0:
            continue
        if i >= len(propagated_sides) or i >= len(pre_reattrib_tids):
            continue

        # Only target actions where Pass 2 changed the player
        if pre_reattrib_tids[i] == action.player_track_id:
            continue

        prop_side = propagated_sides[i]
        if prop_side not in ("near", "far"):
            continue

        expected_team = _SIDE_TO_TEAM[prop_side]
        current_team = team_assignments.get(action.player_track_id)

        if current_team is None or current_team == expected_team:
            continue

        # Pass 2 swapped to wrong team per propagation. Try reverting first.
        orig_tid = pre_reattrib_tids[i]
        orig_team = team_assignments.get(orig_tid)
        if orig_tid >= 0 and orig_team == expected_team:
            logger.debug(
                "Post-hoc revert frame %d: track %d (team %d) → "
                "track %d (team %d, original) from propagated side '%s'",
                action.frame, action.player_track_id, current_team,
                orig_tid, expected_team, prop_side,
            )
            action.player_track_id = orig_tid
            n_corrected += 1
            continue

        # Original also wrong team — find best candidate on correct team
        contact = contact_by_frame.get(action.frame)
        if contact is None or not contact.player_candidates:
            continue

        current_dist = contact.player_distance
        if not math.isfinite(current_dist) or current_dist <= 0.0:
            continue

        best_tid = -1
        best_dist = float("inf")
        for tid, dist in contact.player_candidates:
            if tid == action.player_track_id:
                continue
            cand_team = team_assignments.get(tid)
            if cand_team != expected_team:
                continue
            if dist <= max_distance_ratio * current_dist and dist < best_dist:
                best_tid = tid
                best_dist = dist

        if best_tid >= 0:
            logger.debug(
                "Post-hoc team correction frame %d: track %d (team %d) → "
                "track %d (team %d) from propagated side '%s'",
                action.frame, action.player_track_id, current_team,
                best_tid, expected_team, prop_side,
            )
            action.player_track_id = best_tid
            n_corrected += 1

    if n_corrected > 0:
        logger.info(
            "Post-hoc team correction: fixed %d/%d actions", n_corrected, len(actions),
        )

    return actions


def validate_action_sequence(
    actions: list[ClassifiedAction],
    rally_id: str = "",
) -> list[ClassifiedAction]:
    """Validate physical constraints on the action sequence.

    Checks that the classified sequence is physically possible under
    beach volleyball rules and logs warnings for violations. Violations
    indicate upstream errors (missed contacts, wrong court_side, etc.).

    Constraints checked:
    1. Max 3 contacts per side before ball crosses net.
    2. Serve must be first non-unknown action.
    3. No consecutive net crossings without a same-side action between them
       (would mean a team touched the ball 0 times — impossible).

    Does NOT modify the sequence — only logs warnings. The repair pass
    handles fixable violations; this catches unfixable ones for debugging.

    Returns the actions unchanged.
    """
    real_actions = [a for a in actions if a.action_type != ActionType.UNKNOWN]
    if len(real_actions) < 2:
        return actions

    # Check 1: Max 3 contacts per side
    side_count = 0
    current_side: str | None = None
    for a in real_actions:
        if a.court_side not in ("near", "far"):
            continue
        if a.court_side != current_side:
            current_side = a.court_side
            side_count = 1
        else:
            side_count += 1
        if side_count > 3 and a.action_type != ActionType.BLOCK:
            logger.warning(
                "Rally %s: >3 contacts on %s side (contact #%d at frame %d). "
                "Possible missed net crossing or wrong court_side.",
                rally_id, current_side, side_count, a.frame,
            )

    # Check 2: Serve must be first non-unknown action
    if real_actions and real_actions[0].action_type != ActionType.SERVE:
        logger.warning(
            "Rally %s: first action is %s, not serve (frame %d).",
            rally_id, real_actions[0].action_type.value, real_actions[0].frame,
        )

    # Check 3: Consecutive net-crossing actions with no same-side action between
    net_crossing_types = {ActionType.SERVE, ActionType.ATTACK}
    prev_was_crossing = False
    for a in real_actions:
        is_crossing = a.action_type in net_crossing_types
        if is_crossing and prev_was_crossing:
            logger.warning(
                "Rally %s: consecutive net-crossing actions at frame %d "
                "(%s). Missing same-side contact between them.",
                rally_id, a.frame, a.action_type.value,
            )
        prev_was_crossing = is_crossing

    return actions


# --- Viterbi sequence decoding ---

# Transition matrix for beach volleyball action sequences.
# Hand-tuned probabilities — learned transitions from GT data were tested
# (scripts/learn_viterbi_params.py) but performed -0.1pp worse due to
# overfitting to training distribution. Hand-tuned values provide better
# regularization for the Viterbi decoder.
_VITERBI_TRANSITIONS: dict[tuple[ActionType, ActionType], float] = {
    # After serve: receive or block on opposite side
    (ActionType.SERVE, ActionType.RECEIVE): 0.85,
    (ActionType.SERVE, ActionType.DIG): 0.10,
    (ActionType.SERVE, ActionType.BLOCK): 0.05,
    # After receive: set or attack (same side)
    (ActionType.RECEIVE, ActionType.SET): 0.80,
    (ActionType.RECEIVE, ActionType.ATTACK): 0.15,
    (ActionType.RECEIVE, ActionType.DIG): 0.05,
    # After set: attack (same side)
    (ActionType.SET, ActionType.ATTACK): 0.90,
    (ActionType.SET, ActionType.SET): 0.05,
    (ActionType.SET, ActionType.DIG): 0.05,
    # After attack: dig, block, or receive on opposite side
    (ActionType.ATTACK, ActionType.DIG): 0.50,
    (ActionType.ATTACK, ActionType.BLOCK): 0.20,
    (ActionType.ATTACK, ActionType.RECEIVE): 0.05,
    (ActionType.ATTACK, ActionType.SET): 0.10,
    (ActionType.ATTACK, ActionType.ATTACK): 0.15,
    # After block: dig/set/attack on blocker's side, or dig on opponent's
    (ActionType.BLOCK, ActionType.DIG): 0.40,
    (ActionType.BLOCK, ActionType.SET): 0.25,
    (ActionType.BLOCK, ActionType.ATTACK): 0.20,
    (ActionType.BLOCK, ActionType.BLOCK): 0.05,
    (ActionType.BLOCK, ActionType.RECEIVE): 0.10,
    # After dig: set or attack (same side)
    (ActionType.DIG, ActionType.SET): 0.65,
    (ActionType.DIG, ActionType.ATTACK): 0.25,
    (ActionType.DIG, ActionType.DIG): 0.10,
}

# Minimum transition probability for unlisted pairs (prevents log(0))
_VITERBI_MIN_PROB = 0.001

# Actions eligible for Viterbi re-labeling (serve/receive stay heuristic)
_VITERBI_RELABEL_TYPES = {ActionType.DIG, ActionType.SET, ActionType.ATTACK}

# Candidate labels for Viterbi decoding at each position
_VITERBI_CANDIDATES = [ActionType.DIG, ActionType.SET, ActionType.ATTACK]

# Default confidence cap for Viterbi re-labeling.  Set to 1.0 to always
# apply Viterbi (no gating); the legacy value 0.65 only relabels
# low-confidence predictions.
_VITERBI_RELABEL_CONFIDENCE_CAP_DEFAULT = 1.0


def viterbi_decode_actions(
    actions: list[ClassifiedAction],
    *,
    confidence_cap: float = _VITERBI_RELABEL_CONFIDENCE_CAP_DEFAULT,
) -> list[ClassifiedAction]:
    """Apply Viterbi decoding to enforce sequence constraints.

    Uses dynamic programming to find the most probable action sequence
    given the classifier's per-contact predictions and volleyball
    transition probabilities. Only re-labels dig/set/attack contacts;
    serve, receive, and block labels are preserved from heuristic rules.

    The emission probability is derived from the classifier's confidence:
    - For the originally predicted label: confidence
    - For alternative labels: (1 - confidence) / (n_candidates - 1)

    Args:
        actions: Classified actions (after propagate_court_side and repair).

    Returns:
        Actions with potentially re-labeled dig/set/attack contacts.
    """
    # Find indices of actions eligible for Viterbi re-labeling
    relabel_indices = [
        i for i, a in enumerate(actions)
        if a.action_type in _VITERBI_RELABEL_TYPES
    ]

    if len(relabel_indices) < 2:
        return actions  # Nothing to decode — need at least 2 for transitions

    # Build the full sequence (including fixed labels) for transition scoring
    # but only relabel the eligible positions.

    # For each relabel position, compute emission log-probabilities
    n_candidates = len(_VITERBI_CANDIDATES)

    def emission_log_probs(action: ClassifiedAction) -> dict[ActionType, float]:
        """Log-probability of observing this action under each candidate label."""
        probs: dict[ActionType, float] = {}
        conf = max(0.1, min(0.99, action.confidence))
        other_prob = (1.0 - conf) / max(1, n_candidates - 1)
        for cand in _VITERBI_CANDIDATES:
            if cand == action.action_type:
                probs[cand] = math.log(conf)
            else:
                probs[cand] = math.log(other_prob)
        return probs

    def transition_log_prob(prev: ActionType, curr: ActionType) -> float:
        """Log-probability of transitioning from prev to curr."""
        p = _VITERBI_TRANSITIONS.get((prev, curr), _VITERBI_MIN_PROB)
        return math.log(p)

    # Get the action type immediately before the first relabel position
    # (could be a fixed serve/receive/block)
    def prev_fixed_type(relabel_pos: int) -> ActionType | None:
        """Find the action type of the previous non-relabel action."""
        idx = relabel_indices[relabel_pos]
        for j in range(idx - 1, -1, -1):
            if actions[j].action_type != ActionType.UNKNOWN:
                return actions[j].action_type
        return None

    # Viterbi forward pass
    n_positions = len(relabel_indices)
    # viterbi[t][state] = (log_prob, backpointer_state)
    viterbi: list[dict[ActionType, tuple[float, ActionType | None]]] = []

    # Initialize first position
    first_action = actions[relabel_indices[0]]
    emissions_0 = emission_log_probs(first_action)
    prev_type = prev_fixed_type(0)

    init: dict[ActionType, tuple[float, ActionType | None]] = {}
    for cand in _VITERBI_CANDIDATES:
        score = emissions_0[cand]
        if prev_type is not None:
            score += transition_log_prob(prev_type, cand)
        init[cand] = (score, None)
    viterbi.append(init)

    # Forward pass
    for t in range(1, n_positions):
        curr_action = actions[relabel_indices[t]]
        emissions = emission_log_probs(curr_action)

        # Check if there are fixed (non-relabel) actions between
        # relabel_indices[t-1] and relabel_indices[t]
        gap_start = relabel_indices[t - 1] + 1
        gap_end = relabel_indices[t]
        # Find the last fixed action type in the gap
        gap_type: ActionType | None = None
        for j in range(gap_end - 1, gap_start - 1, -1):
            if actions[j].action_type != ActionType.UNKNOWN:
                gap_type = actions[j].action_type
                break

        step: dict[ActionType, tuple[float, ActionType | None]] = {}
        for curr_cand in _VITERBI_CANDIDATES:
            best_score = float("-inf")
            best_prev: ActionType | None = None
            for prev_cand in _VITERBI_CANDIDATES:
                prev_score = viterbi[t - 1][prev_cand][0]
                if gap_type is not None:
                    # Transition from previous relabel → gap fixed → current
                    trans = (
                        transition_log_prob(prev_cand, gap_type)
                        + transition_log_prob(gap_type, curr_cand)
                    )
                else:
                    trans = transition_log_prob(prev_cand, curr_cand)
                score = prev_score + trans + emissions[curr_cand]
                if score > best_score:
                    best_score = score
                    best_prev = prev_cand
            step[curr_cand] = (best_score, best_prev)
        viterbi.append(step)

    # Backtrack to find best sequence
    best_final = max(_VITERBI_CANDIDATES, key=lambda c: viterbi[-1][c][0])
    decoded: list[ActionType] = [best_final]
    for t in range(n_positions - 1, 0, -1):
        _, prev_state = viterbi[t][decoded[-1]]
        decoded.append(prev_state if prev_state is not None else decoded[-1])
    decoded.reverse()

    # Apply re-labeling only for low-confidence actions.
    n_changed = 0
    result = list(actions)
    for t, idx in enumerate(relabel_indices):
        if decoded[t] != result[idx].action_type:
            if result[idx].confidence > confidence_cap:
                continue  # Trust high-confidence predictions
            logger.debug(
                "Viterbi: frame %d %s → %s (conf=%.2f)",
                result[idx].frame, result[idx].action_type.value,
                decoded[t].value, result[idx].confidence,
            )
            result[idx] = _reclassify(result[idx], decoded[t])
            n_changed += 1

    if n_changed > 0:
        logger.info("Viterbi decoding: re-labeled %d/%d actions", n_changed, len(actions))

    return result


def classify_rally_actions(
    contact_sequence: ContactSequence,
    rally_id: str = "",
    config: ActionClassifierConfig | None = None,
    use_classifier: bool = True,
    team_assignments: dict[int, int] | None = None,
    match_team_assignments: dict[int, int] | None = None,
    reid_predictions: dict[int, dict[str, Any]] | None = None,
    visual_classifier: Any = None,
    visual_video_cap: Any = None,
    visual_positions_json: list[dict[str, Any]] | None = None,
    visual_rally_start_frame: int = 0,
    visual_frame_w: int = 0,
    visual_frame_h: int = 0,
    calibrator: CourtCalibrator | None = None,
    track_to_player: dict[int, int] | None = None,
    formation_semantic_flip: bool = False,
    camera_height: float = 0.0,
) -> RallyActions:
    """Convenience function to classify actions in a rally.

    When use_classifier=True and a trained action type model exists on disk,
    uses the learned classifier for dig/set/attack classification. Otherwise
    falls back to the rule-based state machine.

    Pipeline:
    1. classify_rally() — initial action types + serve detection
       (uses match_team_assignments for team-aware touch counting when available)
    2. repair_action_sequence(Rule 1 only) — fix consecutive recv/dig → set
    3. viterbi_decode_actions() — sequence-level smoothing
    4. validate_action_sequence() — log constraint violations
    5. assign_court_side_from_teams() — overwrite court_side from match teams
    6. reattribute_players() — server exclusion + server-seeded team chain
       for player re-attribution

    Args:
        contact_sequence: Contacts detected by ContactDetector.
        rally_id: Optional rally identifier.
        config: Optional classifier configuration.
        use_classifier: Whether to auto-load and use the learned classifier.
        team_assignments: Optional mapping of track_id → team (0=near/A, 1=far/B).
            Used for team labeling and action classification.
        match_team_assignments: Optional high-confidence match-level team mapping
            (track_id → team). Used for post-classification player
            re-attribution and court_side labeling.
            Should only be provided when assignment confidence >= 0.70.
        reid_predictions: Optional ReID predictions per contact frame
            ({frame: {"best_tid": int, "margin": float}}). Enables Pass 3
            in reattribute_players(). Off by default — currently net negative.
        visual_classifier: Optional trained VisualAttributionClassifier.
            When provided with video context, runs visual re-attribution
            after proximity-based reattribution.
        visual_video_cap: Open cv2.VideoCapture (required if visual_classifier).
        visual_positions_json: Player positions for clip extraction.
        visual_rally_start_frame: Absolute frame of rally start.
        visual_frame_w: Video frame width in pixels.
        visual_frame_h: Video frame height in pixels.
        calibrator: Optional court calibrator for court-space server detection.
        track_to_player: Optional mapping `track_id → player_id (1-4)` from
            match-level cross-rally identification. Used as a fallback by
            `_find_serving_team_by_formation` when `team_assignments` is
            unavailable.
        formation_semantic_flip: Passed to `_find_serving_team_by_formation`
            to correct the physical-near team convention on flipped rallies.
            True when the cumulative side-switch count before this rally is
            odd. Caller computes from `match_analysis_json.sideSwitchDetected`.

    Returns:
        RallyActions with all classified actions. `serving_team` property
        prefers the formation-based prediction (set via
        `result.formation_serving_team`) over contact-based serve detection
        when `config.use_formation_serving_team` is True (default).
    """
    # Only re-attribute with match-level teams (high-confidence cross-rally data).
    # Per-rally team_assignments are too unreliable and cause net regressions.
    reattrib_teams = match_team_assignments
    action_classifier = ActionClassifier(config)

    learned = None
    if use_classifier:
        learned = _get_default_action_classifier()
        if learned is not None and not learned.is_trained:
            learned = None

    result = action_classifier.classify_rally(
        contact_sequence, rally_id,
        team_assignments=team_assignments,
        classifier=learned,
        match_team_assignments=match_team_assignments,
        calibrator=calibrator,
        camera_height=camera_height,
    )

    # Repair with only Rule 1 (consecutive recv/dig → set, +0.8pp LOO-CV).
    # All other rules hurt accuracy — see scripts/ablate_repair_rules.py.
    result.actions, _ = repair_action_sequence(
        result.actions,
        net_y=contact_sequence.net_y,
        ball_positions=contact_sequence.ball_positions,
        rally_start_frame=contact_sequence.rally_start_frame,
        disabled_rules={0, 2, 3, 5, 6},
    )

    result.actions = viterbi_decode_actions(result.actions)
    result.actions = validate_action_sequence(result.actions, rally_id)

    if match_team_assignments:
        assign_court_side_from_teams(result.actions, match_team_assignments)

    result.actions = reattribute_players(
        result.actions, contact_sequence.contacts, reattrib_teams,
        max_distance_ratio=1.5,
        reid_predictions=reid_predictions,
    )

    # Visual attribution pass (overrides proximity-based attribution)
    if (visual_classifier is not None and visual_video_cap is not None
            and visual_positions_json is not None):
        from rallycut.tracking.visual_attribution import visual_reattribute
        visual_reattribute(
            result.actions, contact_sequence.contacts,
            visual_positions_json, visual_video_cap,
            visual_rally_start_frame, visual_frame_w, visual_frame_h,
            visual_classifier, team_assignments=reattrib_teams,
        )

    # Formation-based serving team prediction. Uses player positions at
    # rally start (not contact-based). Set as primary signal on
    # RallyActions.serving_team. See _find_serving_team_by_formation
    # docstring for rationale and measured accuracy.
    #
    # NOTE: `start_frame=0` (rally-relative), not `rally_start_frame` which
    # is the first *ball* detection frame (~60-90 in). We want the formation
    # BEFORE the ball is in play — when the server is still behind the
    # baseline and the partner is at the net.
    cfg = config if config is not None else ActionClassifierConfig()
    if cfg.use_formation_serving_team:
        # Determine first contact frame for adaptive window
        first_contact_frame_val: int | None = None
        first_contact_obj: Contact | None = None
        if contact_sequence.contacts:
            first_contact_frame_val = contact_sequence.contacts[0].frame
            first_contact_obj = contact_sequence.contacts[0]

        formation_team, _ = _find_serving_team_by_formation(
            contact_sequence.player_positions,
            start_frame=0,
            net_y=contact_sequence.net_y,
            team_assignments=reattrib_teams or team_assignments,
            track_to_player=track_to_player,
            semantic_flip=formation_semantic_flip,
            window_frames=cfg.formation_window_frames,
            margin=cfg.formation_margin,
            ball_positions=contact_sequence.ball_positions,
            calibrator=calibrator,
            first_contact_frame=first_contact_frame_val,
            adaptive_window=True,
            first_contact=first_contact_obj,
        )
        if formation_team is not None:
            result.formation_serving_team = formation_team

    return result
