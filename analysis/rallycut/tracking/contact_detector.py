"""Ball contact detection from trajectory inflection points.

Detects moments when a player contacts the ball by analyzing the ball
trajectory for sharp direction changes (inflection points). Each detected
contact is attributed to the nearest player.

This module provides a clean, standalone API for contact detection,
building on the lower-level contact detection in ball_features.py.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from rallycut.tracking.pose_attribution.features import KPT_LEFT_WRIST, KPT_RIGHT_WRIST

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_classifier import ContactClassifier
    from rallycut.tracking.player_tracker import PlayerPosition
    from rallycut.tracking.pose_attribution.inference import (
        PoseAttributionInference,
    )
    from rallycut.tracking.temporal_attribution.inference import (
        TemporalAttributionInference,
    )

logger = logging.getLogger(__name__)

# Minimum confidence to treat a ball position as a real detection.
# Ball detector confidence is bimodal: either 0.0 (no detection) or >=0.3 (confident).
_CONFIDENCE_THRESHOLD = 0.3

# Cached temporal attributor (loaded once from disk on first use)
_temporal_attributor_cache: dict[str, TemporalAttributionInference | None] = {}


def _get_temporal_attributor() -> TemporalAttributionInference | None:
    """Load and cache the temporal attribution model from disk.

    Returns None if no trained model exists at the default path.
    """
    if "default" not in _temporal_attributor_cache:
        from pathlib import Path

        from rallycut.tracking.temporal_attribution.inference import (
            TemporalAttributionInference,
        )

        model_path = (
            Path(__file__).parent.parent.parent
            / "weights"
            / "temporal_attribution"
            / "best_temporal_attribution.joblib"
        )
        if model_path.exists():
            _temporal_attributor_cache["default"] = TemporalAttributionInference(
                model_path
            )
            logger.info("Auto-loaded temporal attribution model from default path")
        else:
            _temporal_attributor_cache["default"] = None
    return _temporal_attributor_cache["default"]


# Cached pose attributor (loaded once from disk on first use)
_pose_attributor_cache: dict[str, PoseAttributionInference | None] = {}


def _get_pose_attributor() -> PoseAttributionInference | None:
    """Load and cache the pose attribution model from disk."""
    if "default" not in _pose_attributor_cache:
        from pathlib import Path

        from rallycut.tracking.pose_attribution.inference import (
            PoseAttributionInference,
        )

        model_path = (
            Path(__file__).parent.parent.parent
            / "weights"
            / "pose_attribution"
            / "pose_attribution.joblib"
        )
        if model_path.exists():
            _pose_attributor_cache["default"] = PoseAttributionInference(model_path)
            logger.info("Auto-loaded pose attribution model from default path")
        else:
            _pose_attributor_cache["default"] = None
    return _pose_attributor_cache["default"]


# Cached default classifier (loaded once from disk on first use)
_default_classifier_cache: dict[str, ContactClassifier | None] = {}


def _get_default_classifier() -> ContactClassifier | None:
    """Load and cache the default contact classifier from disk.

    Returns None if no trained model exists at the default path.
    """
    if "default" not in _default_classifier_cache:
        from rallycut.tracking.contact_classifier import load_contact_classifier

        clf = load_contact_classifier()
        _default_classifier_cache["default"] = clf
        if clf is not None:
            logger.info("Auto-loaded contact classifier from default path")
    return _default_classifier_cache["default"]


@dataclass
class ContactDetectionConfig:
    """Configuration for trajectory-based contact detection."""

    # Velocity peak detection (velocities are in normalized image-space units/frame)
    min_peak_velocity: float = 0.008  # Min velocity for a contact peak
    min_peak_prominence: float = 0.003  # Min prominence relative to neighboring velocity values
    smoothing_window: int = 5  # Frames for velocity smoothing
    min_peak_distance_frames: int = 12  # Min frames between contacts (~0.4s @ 30fps)

    # Direction change thresholds
    min_direction_change_deg: float = 20.0  # Min angle change to confirm contact
    direction_check_frames: int = 8  # Frames before/after to check direction

    # Inflection detection
    enable_inflection_detection: bool = True
    min_inflection_angle_deg: float = 15.0  # Min angle for inflection candidate
    inflection_check_frames: int = 5  # Frames before/after for inflection check

    # Noise spike filter
    enable_noise_filter: bool = True
    noise_spike_max_jump: float = 0.20  # Max distance to predecessor/successor

    # Player proximity (distances in fraction of frame width/height, image-space)
    player_contact_radius: float = 0.15  # Max Euclidean distance for player attribution
    player_search_frames: int = 5  # Search ±N frames for nearest player (classifier feature)
    player_candidate_search_frames: int = 15  # Wider search for ranked candidates (~500ms)

    # High-velocity contacts (lenient validation)
    high_velocity_threshold: float = 0.025  # Auto-accept above this velocity

    # Warmup filter: skip candidates in the first N frames (ball tracking warmup)
    warmup_skip_frames: int = 5  # Skip first ~0.17s of ball tracking (avoid warmup noise)

    # Minimum velocity for any candidate (floor for inflection/reversal candidates)
    min_candidate_velocity: float = 0.003  # Below this, direction change is likely noise

    # Parabolic arc breakpoint detection
    enable_parabolic_detection: bool = True
    parabolic_window_frames: int = 12  # Sliding window size for parabolic fit
    parabolic_stride: int = 3  # Window slide step
    parabolic_min_residual: float = 0.015  # Min residual peak to flag breakpoint
    parabolic_min_prominence: float = 0.008  # Min prominence for residual peaks

    # Deceleration candidate detection (catches receives/digs)
    enable_deceleration_detection: bool = True
    deceleration_min_speed_before: float = 0.008  # Min incoming speed to qualify
    deceleration_min_drop_ratio: float = 0.3  # Min speed drop ratio (30%)
    deceleration_window: int = 5  # Frames before/after to check

    # Post-serve receive candidate search
    enable_post_serve_receive: bool = True
    post_serve_search_window: int = 15  # ±N frames around net crossing for receive

    # Player-proximity candidate refinement
    enable_proximity_candidates: bool = True
    proximity_search_window: int = 8  # Search ±N frames around each candidate

    # Trajectory-peak refinement: after candidate generation, shift each candidate
    # to the frame with maximum direction change within a small window. This
    # corrects the common misalignment where generators detect the velocity peak
    # (effect of contact) several frames before the actual contact (cause).
    enable_trajectory_refinement: bool = True
    trajectory_refinement_window: int = 5  # Search ±N frames for peak direction change

    # Direction-change peak candidates: scan direction_change across all frames
    # and fire at local maxima above a threshold. Existing generators (velocity
    # peak, inflection, reversal) detect the EFFECT of contact (velocity spike)
    # which lags the actual contact by several frames. This generator fires at
    # the cause (peak direction change), filling gaps where other generators
    # place candidates >5 frames from the true contact.
    enable_direction_change_candidates: bool = True
    direction_change_candidate_min_deg: float = 25.0  # Min direction change to qualify
    direction_change_candidate_prominence: float = 10.0  # Min prominence for peaks

    # Court position baselines (fixed defaults assuming net_y≈0.5).
    # action_classifier.py computes dynamic baselines from actual net_y instead.
    baseline_y_near: float = 0.82  # Near baseline Y threshold
    baseline_y_far: float = 0.18  # Far baseline Y threshold
    serve_window_frames: int = 60  # Serve must occur in first N frames (~2s)

    # Player-motion candidate generation: detect contacts from player body motion
    # (arm swings, jumps) even when ball trajectory is flat (blocks, soft touches).
    # Disabled by default: adds 265 candidates but only 9 TPs (3.4% hit rate),
    # hurting classifier LOO CV. The bbox motion features on existing candidates
    # capture the same signal more effectively.
    enable_player_motion_candidates: bool = False
    player_motion_min_d_y: float = 0.015  # Min peak Y shift to qualify as contact motion
    player_motion_min_d_height: float = 0.015  # Min peak height change to qualify
    player_motion_max_ball_distance: float = 0.20  # Max ball-player distance for motion candidate

    # Temporal attribution: trajectory-based model that overrides proximity
    # attribution. Used as a fallback when pose attribution is unavailable
    # (no pose model on disk or use_pose_attribution=False).
    use_temporal_attribution: bool = True
    temporal_attribution_min_confidence: float = 0.6  # Min softmax confidence to accept

    # Pose attribution: per-candidate binary classifier using YOLO-Pose keypoints.
    # Takes precedence over temporal_attribution when both are enabled and the
    # pose model loads successfully. Enabled 2026-04-07 with combined md=6 model:
    # +3.4pp player attribution (68.7% -> 72.1%), +4.1pp court-side (82.0% -> 86.1%),
    # dig +8.4pp, set +5.0pp, attack +3.8pp. Requires PlayerPosition.keypoints
    # to be populated (via inline tracking enrichment or inject_keypoints.py).
    use_pose_attribution: bool = True
    pose_attribution_min_confidence: float = 0.5  # Min P(touching) to accept

    # Adaptive deduplication: use shorter min distance (_CROSS_SIDE_MIN_DISTANCE=4)
    # for cross-side contacts, preserving the full 12-frame gap same-side.
    # Enabled 2026-04-07 to rescue block contacts — attack→block transitions
    # are 3-5 frames apart and were being merged with the attack by the
    # final dedup pass. Evaluated on the 339-rally action set:
    #   Action Acc 87.1% → 86.9% (-0.2pp)
    #   Attribution 72.1% → 72.1% (unchanged)
    #   Court-Side 79.6% → 86.1% (+6.5pp) ← the real win
    #   Block F1 0.0% → 7.1% (1/27 TP, 0 FP)
    # The large court-side gain comes from more cross-side candidates
    # surviving dedup, letting the resolver disambiguate sides more often.
    adaptive_dedup: bool = True

    # Sequence-based contact recovery (shipped 2026-04-07).
    # Two-signal agreement gate: a trajectory candidate the GBM rejects is
    # rescued iff (1) the MS-TCN++ sequence model has a non-background peak
    # >= SEQ_RECOVERY_TAU within +-5 frames AND (2) the GBM still gave the
    # candidate a non-trivial score >= SEQ_RECOVERY_CLF_FLOOR. Both
    # thresholds live as module-level constants in
    # `sequence_action_runtime.py` (read at call time so sweep harnesses can
    # monkey-patch them). Empirical basis: memory/fn_sequence_signal_2026_04.md.
    enable_sequence_recovery: bool = True


@dataclass
class Contact:
    """A detected ball contact event."""

    frame: int  # Frame number of contact
    ball_x: float  # Ball X position (normalized 0-1)
    ball_y: float  # Ball Y position (normalized 0-1)
    velocity: float  # Ball velocity at contact (normalized units/frame)
    direction_change_deg: float  # Trajectory direction change in degrees

    # Player attribution
    player_track_id: int = -1  # Track ID of contacting player (-1 = unknown)
    player_distance: float = float("inf")  # Distance to nearest player
    player_candidates: list[tuple[int, float]] = field(default_factory=list)
    # Ranked (track_id, distance) pairs, sorted by distance. First = nearest.
    candidate_bbox_motion: dict[int, tuple[float, float]] = field(default_factory=dict)
    # {track_id: (max_d_y, max_d_height)} — peak frame-to-frame bbox deltas in ±5 frames.

    # Court context
    court_side: str = "unknown"  # "near", "far", or "unknown"
    is_at_net: bool = False  # True if contact near net area

    # Validation
    is_validated: bool = False  # True if contact passed validation checks
    confidence: float = 0.0  # Classifier confidence (0-1), set by Phase 3 classifier
    arc_fit_residual: float = 0.0  # Parabolic arc fit residual at this frame

    def to_dict(self) -> dict:
        return {
            "frame": self.frame,
            "ballX": self.ball_x,
            "ballY": self.ball_y,
            "velocity": self.velocity,
            "directionChangeDeg": self.direction_change_deg,
            "playerTrackId": self.player_track_id,
            "playerDistance": self.player_distance if math.isfinite(self.player_distance) else None,
            "courtSide": self.court_side,
            "isAtNet": self.is_at_net,
            "isValidated": self.is_validated,
            "confidence": self.confidence,
            "arcFitResidual": self.arc_fit_residual,
            "playerCandidates": [
                [tid, d if math.isfinite(d) else None]
                for tid, d in self.player_candidates
            ],
            "candidateBboxMotion": {
                str(tid): [dy, dh]
                for tid, (dy, dh) in self.candidate_bbox_motion.items()
            } if self.candidate_bbox_motion else None,
        }


@dataclass
class ContactSequence:
    """A sequence of contacts detected within a rally."""

    contacts: list[Contact] = field(default_factory=list)
    net_y: float = 0.50  # Estimated net Y position
    rally_start_frame: int = 0
    ball_positions: list[BallPosition] = field(default_factory=list)
    player_positions: list[PlayerPosition] = field(default_factory=list)

    @property
    def num_contacts(self) -> int:
        return len(self.contacts)

    @property
    def serve_contact(self) -> Contact | None:
        """Get the first contact (likely serve)."""
        return self.contacts[0] if self.contacts else None

    def contacts_on_side(self, side: str) -> list[Contact]:
        """Get contacts on a specific court side."""
        return [c for c in self.contacts if c.court_side == side]

    def to_dict(self) -> dict:
        return {
            "numContacts": self.num_contacts,
            "netY": self.net_y,
            "rallyStartFrame": self.rally_start_frame,
            "contacts": [c.to_dict() for c in self.contacts],
        }


def _compute_velocities(
    ball_positions: list[BallPosition],
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> dict[int, tuple[float, float, float]]:
    """Compute ball velocity at each frame using central differences.

    Uses central difference (F+1 − F−1) for interior frames to avoid the
    +1-frame lag inherent in backward differences.  Falls back to forward
    or backward difference at sequence boundaries / detection gaps.

    Returns:
        Dict mapping frame_number to (velocity, vx, vy) in normalized units/frame.
    """
    confident = [
        bp for bp in ball_positions
        if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 2:
        return {}

    max_gap = 5  # max frame gap to consider consecutive

    velocities: dict[int, tuple[float, float, float]] = {}

    for i in range(len(confident)):
        has_prev = (
            i > 0
            and 0 < confident[i].frame_number - confident[i - 1].frame_number <= max_gap
        )
        has_next = (
            i < len(confident) - 1
            and 0 < confident[i + 1].frame_number - confident[i].frame_number <= max_gap
        )

        if has_prev and has_next:
            # Central difference — no directional lag
            prev_bp = confident[i - 1]
            nxt_bp = confident[i + 1]
            total_gap = nxt_bp.frame_number - prev_bp.frame_number
            dx = (nxt_bp.x - prev_bp.x) / total_gap
            dy = (nxt_bp.y - prev_bp.y) / total_gap
        elif has_next:
            # Forward difference (first frame or gap before)
            curr_bp = confident[i]
            nxt_bp = confident[i + 1]
            gap = nxt_bp.frame_number - curr_bp.frame_number
            dx = (nxt_bp.x - curr_bp.x) / gap
            dy = (nxt_bp.y - curr_bp.y) / gap
        elif has_prev:
            # Backward difference (last frame or gap after)
            prev_bp = confident[i - 1]
            curr_bp = confident[i]
            gap = curr_bp.frame_number - prev_bp.frame_number
            dx = (curr_bp.x - prev_bp.x) / gap
            dy = (curr_bp.y - prev_bp.y) / gap
        else:
            continue

        speed = math.sqrt(dx * dx + dy * dy)
        velocities[confident[i].frame_number] = (speed, dx, dy)

    return velocities


def _smooth_signal(values: list[float], window: int) -> list[float]:
    """Apply moving average smoothing."""
    if len(values) < window:
        return values

    half_w = window // 2
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - half_w)
        end = min(len(values), i + half_w + 1)
        smoothed.append(sum(values[start:end]) / (end - start))
    return smoothed


def compute_direction_change(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
    check_frames: int = 5,
) -> float:
    """Compute trajectory direction change at a frame in degrees (0-180).

    Finds the nearest ball positions before and after the given frame
    (within check_frames), computes incoming and outgoing direction
    vectors, and returns the angle between them. Interpolates if no
    exact position exists at the target frame.
    """
    before_frame = None
    after_frame = None

    for offset in range(1, check_frames + 1):
        if before_frame is None and (frame - offset) in ball_by_frame:
            before_frame = frame - offset
        if after_frame is None and (frame + offset) in ball_by_frame:
            after_frame = frame + offset
        if before_frame is not None and after_frame is not None:
            break

    if before_frame is None or after_frame is None:
        return 0.0

    bp_before = ball_by_frame[before_frame]
    bp_at = ball_by_frame.get(frame)
    bp_after = ball_by_frame[after_frame]

    if bp_at is None:
        t = (frame - before_frame) / (after_frame - before_frame)
        at_x = bp_before.x + t * (bp_after.x - bp_before.x)
        at_y = bp_before.y + t * (bp_after.y - bp_before.y)
    else:
        at_x, at_y = bp_at.x, bp_at.y

    vec_in = (at_x - bp_before.x, at_y - bp_before.y)
    vec_out = (bp_after.x - at_x, bp_after.y - at_y)

    mag_in = math.sqrt(vec_in[0] ** 2 + vec_in[1] ** 2)
    mag_out = math.sqrt(vec_out[0] ** 2 + vec_out[1] ** 2)

    if mag_in < 1e-6 or mag_out < 1e-6:
        return 0.0

    dot = vec_in[0] * vec_out[0] + vec_in[1] * vec_out[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag_in * mag_out)))

    return float(np.degrees(np.arccos(cos_angle)))


_MIN_WRIST_CONF = 0.3  # Minimum keypoint confidence to use wrist position


def _player_to_ball_dist(
    player: PlayerPosition,
    ball_x: float,
    ball_y: float,
) -> float:
    """Image-space distance from player to ball.

    Uses the closer wrist keypoint when pose data is available with
    sufficient confidence. Falls back to bbox upper-quarter (torso/arms)
    when keypoints are absent or low-confidence.

    Wrist distance is a +2.5pp improvement over bbox centroid for
    attribution (diagnostic: scripts/diagnose_keypoint_attribution.py).
    Volleyball contacts happen with hands/arms, so wrist position is a
    better proxy for who is touching the ball.

    IMPORTANT: this MUST remain in image-space (normalized coords).
    Switching to court-space distance shifts feature distributions and
    breaks the contact classifier (tested: F1 89.4% -> 76.0%).
    """
    # Try wrist keypoints first (COCO 17-keypoint format)
    if player.keypoints is not None and len(player.keypoints) >= 17:
        best_wrist_dist = math.inf
        for kpt_idx in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
            kx, ky, kc = player.keypoints[kpt_idx]
            if kc >= _MIN_WRIST_CONF:
                d = math.sqrt((ball_x - kx) ** 2 + (ball_y - ky) ** 2)
                if d < best_wrist_dist:
                    best_wrist_dist = d
        if best_wrist_dist < math.inf:
            return best_wrist_dist

    # Fallback: bbox upper-quarter
    player_x = player.x
    player_y = player.y - player.height * 0.25
    return math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)


def _find_nearest_player(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 5,
) -> tuple[int, float, float]:
    """Find nearest player to ball at given frame.

    Uses wrist keypoint distance when pose data is available, falling
    back to bbox upper-quarter distance. See _player_to_ball_dist().

    Returns:
        (track_id, distance, player_center_y). track_id=-1 if no player found.
        player_center_y is the bbox center Y (for court side determination).
    """
    best_track_id = -1
    best_dist = float("inf")
    best_player_y = 0.5

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        dist = _player_to_ball_dist(p, ball_x, ball_y)

        if dist < best_dist:
            best_dist = dist
            best_track_id = p.track_id
            best_player_y = p.y  # bbox center Y for court side

    return best_track_id, best_dist, best_player_y


def _depth_scale_at_y(
    y: float,
    court_calibrator: CourtCalibrator | None,
) -> float:
    """Compute perspective depth scaling factor at a given image Y position.

    Uses court corner geometry to estimate how much image-space distances
    are compressed at this Y position. Far-court positions (low Y) have
    higher compression (scale > 1), near-court (high Y) have scale ≈ 1.

    Returns 1.0 if no calibration is available.
    """
    if court_calibrator is None or not court_calibrator.is_calibrated:
        return 1.0
    homography = court_calibrator.homography
    if homography is None or len(homography.image_corners) != 4:
        return 1.0

    corners = homography.image_corners
    near_y = (corners[0][1] + corners[1][1]) / 2
    far_y = (corners[2][1] + corners[3][1]) / 2
    near_w = abs(corners[1][0] - corners[0][0])
    far_w = abs(corners[2][0] - corners[3][0])

    if near_y <= far_y or near_w <= 0 or far_w <= 0:
        return 1.0

    # Interpolate court width at this Y position
    t = max(0.0, min(1.0, (y - far_y) / (near_y - far_y)))
    width_at_y = far_w + t * (near_w - far_w)
    return near_w / width_at_y if width_at_y > 0 else 1.0


def _find_nearest_players(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 15,
    max_candidates: int = 4,
    court_calibrator: CourtCalibrator | None = None,
) -> list[tuple[int, float, float]]:
    """Find nearest players to ball, ranked by perspective-corrected distance.

    Uses wrist keypoint distance when pose data is available, falling
    back to bbox upper-quarter distance. See _player_to_ball_dist().

    Ranks candidates by depth-scaled distance: wrist/bbox distance
    multiplied by a perspective correction factor derived from the court
    corners. Far-court distances are scaled up (they appear artificially
    small due to perspective compression).

    Returns:
        List of (track_id, distance, player_center_y), sorted by
        depth-corrected distance. Up to max_candidates entries.
    """
    # Best distances per track (a track may appear in multiple frames)
    # track_id → (rank_dist, img_dist, center_y)
    best_per_track: dict[int, tuple[float, float, float]] = {}

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        img_dist = _player_to_ball_dist(p, ball_x, ball_y)

        # Rank by depth-corrected distance: scale by perspective at player Y.
        # Use bbox upper-quarter Y for perspective scaling (stable regardless
        # of whether wrist or bbox was used for distance).
        scale_y = p.y - p.height * 0.25
        scale = _depth_scale_at_y(scale_y, court_calibrator)
        rank_dist = img_dist * scale

        if p.track_id not in best_per_track or rank_dist < best_per_track[p.track_id][0]:
            best_per_track[p.track_id] = (rank_dist, img_dist, p.y)

    ranked = sorted(best_per_track.items(), key=lambda x: x[1][0])
    return [
        (tid, img_dist, center_y)
        for tid, (_rank_dist, img_dist, center_y) in ranked[:max_candidates]
    ]


def _compute_candidate_bbox_motion(
    player_positions: list[PlayerPosition],
    contact_frame: int,
    candidate_track_ids: list[int],
    window: int = 5,
) -> dict[int, tuple[float, float]]:
    """Compute peak frame-to-frame bbox motion for each candidate track.

    For each candidate, collects positions in ±window frames and computes the
    maximum absolute frame-to-frame change in Y (vertical shift from jumps)
    and height (elongation from arm swings / crouching).

    Returns:
        {track_id: (max_delta_y, max_delta_height)}.
        Tracks with < 2 observations in the window are omitted.
    """
    candidate_set = set(candidate_track_ids)
    # Collect per-track positions in the window, keyed by frame
    track_frames: dict[int, dict[int, PlayerPosition]] = {}
    for p in player_positions:
        if p.track_id not in candidate_set:
            continue
        if abs(p.frame_number - contact_frame) > window:
            continue
        track_frames.setdefault(p.track_id, {})[p.frame_number] = p

    result: dict[int, tuple[float, float]] = {}
    for tid in candidate_track_ids:
        frames = track_frames.get(tid)
        if not frames or len(frames) < 2:
            continue
        sorted_pos = [frames[f] for f in sorted(frames)]
        max_dy = max(
            abs(sorted_pos[i + 1].y - sorted_pos[i].y)
            for i in range(len(sorted_pos) - 1)
        )
        max_dh = max(
            abs(sorted_pos[i + 1].height - sorted_pos[i].height)
            for i in range(len(sorted_pos) - 1)
        )
        result[tid] = (max_dy, max_dh)
    return result


def _find_player_motion_candidates(
    player_positions: list[PlayerPosition],
    ball_by_frame: dict[int, BallPosition],
    existing_candidates: list[int],
    min_distance_frames: int = 12,
    motion_window: int = 5,
    min_d_y: float = 0.015,
    min_d_height: float = 0.015,
    max_ball_distance: float = 0.20,
) -> list[int]:
    """Generate contact candidates from player body motion near the ball.

    Detects frames where a player shows significant motion (jump, arm swing)
    AND is close to the ball, even when ball trajectory doesn't change.
    Addresses blocks and soft touches where existing generators fail.

    Returns:
        List of candidate frames not already covered by existing candidates.
    """
    if not player_positions or not ball_by_frame:
        return []

    existing_set = set(existing_candidates)
    ball_frames = sorted(ball_by_frame.keys())
    if not ball_frames:
        return []

    # Group player positions by frame and by track_id for efficient lookup
    players_by_frame: dict[int, list[PlayerPosition]] = {}
    players_by_track: dict[int, list[PlayerPosition]] = {}
    for p in player_positions:
        players_by_frame.setdefault(p.frame_number, []).append(p)
        players_by_track.setdefault(p.track_id, []).append(p)

    # Sort per-track positions by frame for windowed lookups
    for positions in players_by_track.values():
        positions.sort(key=lambda p: p.frame_number)

    # Pre-sort existing candidates for efficient proximity check
    sorted_existing = sorted(existing_set)

    candidates: list[int] = []

    # Scan ball frames for player motion peaks near the ball
    for bf in ball_frames:
        # Skip if near an existing candidate (binary search on sorted list)
        if any(abs(bf - ec) < min_distance_frames for ec in sorted_existing):
            continue

        ball = ball_by_frame[bf]

        # Check each player visible at this frame
        players_at_frame = players_by_frame.get(bf, [])
        for player in players_at_frame:
            # Coarse proximity gate — intentionally uses bbox, not wrist keypoints.
            # This is a disabled-by-default candidate generator, not attribution;
            # we just need "is a player near the ball?" not "who touched it?"
            player_x = player.x
            player_y = player.y - player.height * 0.25
            dist = math.sqrt((ball.x - player_x) ** 2 + (ball.y - player_y) ** 2)
            if dist > max_ball_distance:
                continue

            # Compute peak motion for this player in ±window (O(1) lookup via pre-grouped dict)
            positions_in_window: list[PlayerPosition] = [
                p for p in players_by_track.get(player.track_id, [])
                if abs(p.frame_number - bf) <= motion_window
            ]

            if len(positions_in_window) < 2:
                continue

            positions_in_window.sort(key=lambda p: p.frame_number)
            max_dy = max(
                abs(positions_in_window[i + 1].y - positions_in_window[i].y)
                for i in range(len(positions_in_window) - 1)
            )
            max_dh = max(
                abs(positions_in_window[i + 1].height - positions_in_window[i].height)
                for i in range(len(positions_in_window) - 1)
            )

            if max_dy >= min_d_y or max_dh >= min_d_height:
                candidates.append(bf)
                existing_set.add(bf)
                break  # One player qualifying is enough for this frame

    return candidates


def _filter_noise_spikes(
    ball_positions: list[BallPosition],
    max_jump: float,
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> list[BallPosition]:
    """Zero out noise spikes where ball jumps far from both predecessor and successor.

    The ball detector produces single-frame false positives that jump to player positions.
    If a position is far from BOTH its predecessor and successor, it's a spike.
    """
    confident = [
        bp for bp in ball_positions if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 3:
        return ball_positions

    # Build set of spike frame numbers
    spike_frames: set[int] = set()

    for i in range(1, len(confident) - 1):
        prev_bp = confident[i - 1]
        curr_bp = confident[i]
        next_bp = confident[i + 1]

        dist_prev = math.sqrt(
            (curr_bp.x - prev_bp.x) ** 2 + (curr_bp.y - prev_bp.y) ** 2
        )
        dist_next = math.sqrt(
            (curr_bp.x - next_bp.x) ** 2 + (curr_bp.y - next_bp.y) ** 2
        )

        if dist_prev > max_jump and dist_next > max_jump:
            spike_frames.add(curr_bp.frame_number)

    if not spike_frames:
        return ball_positions

    logger.debug(f"Noise filter: zeroing {len(spike_frames)} spike frames")

    # Return new list with spike frames zeroed (confidence=0.0)
    return [
        replace(bp, confidence=0.0) if bp.frame_number in spike_frames else bp
        for bp in ball_positions
    ]


def _find_inflection_candidates(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    min_angle_deg: float,
    check_frames: int,
    min_distance_frames: int,
) -> list[int]:
    """Find trajectory inflection points (direction changes) as contact candidates.

    Returns sorted list of candidate frame numbers.
    """
    if len(confident_frames) < 3:
        return []

    # Compute angle at each confident frame
    frame_angles: list[tuple[int, float]] = []
    for frame in confident_frames:
        angle = compute_direction_change(ball_by_frame, frame, check_frames)
        if angle >= min_angle_deg:
            frame_angles.append((frame, angle))

    if not frame_angles:
        return []

    # Enforce min distance: when two candidates are close, keep largest angle
    frame_angles.sort(key=lambda x: x[0])
    angle_by_frame = dict(frame_angles)
    selected: list[int] = []

    for frame, angle in frame_angles:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif angle > angle_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _find_velocity_reversal_candidates(
    velocities: dict[int, tuple[float, float, float]],
    frames: list[int],
    min_distance_frames: int,
) -> list[int]:
    """Find frames where ball velocity reverses direction.

    A real contact reverses the ball's velocity direction. This catches contacts
    that velocity peaks miss (e.g., a dig that barely changes speed but reverses
    direction).

    Returns:
        Sorted list of candidate frame numbers where velocity reverses.
    """
    if len(frames) < 3:
        return []

    reversal_frames: list[tuple[int, float]] = []

    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]

        if curr_frame - prev_frame > 5:
            continue

        prev_vel = velocities.get(prev_frame)
        curr_vel = velocities.get(curr_frame)
        if prev_vel is None or curr_vel is None:
            continue

        _, prev_vx, prev_vy = prev_vel
        _, curr_vx, curr_vy = curr_vel

        # Dot product of consecutive velocity vectors
        dot = prev_vx * curr_vx + prev_vy * curr_vy
        if dot < 0:
            # Magnitude of the reversal (more negative = sharper reversal)
            reversal_frames.append((curr_frame, -dot))

    if not reversal_frames:
        return []

    # Enforce min distance: keep strongest reversal when close
    reversal_frames.sort(key=lambda x: x[0])
    selected: list[int] = []
    strength_by_frame = dict(reversal_frames)

    for frame, strength in reversal_frames:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif strength > strength_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _find_deceleration_candidates(
    velocities: dict[int, tuple[float, float, float]],
    frames: list[int],
    smoothed_speeds: list[float],
    min_distance_frames: int,
    min_speed_before: float = 0.008,
    min_speed_drop_ratio: float = 0.3,
    window: int = 5,
) -> list[int]:
    """Find frames where ball speed drops sharply (deceleration contacts).

    Receives and digs decelerate the ball — the player absorbs momentum.
    Velocity peak detection misses these because the contact is a velocity
    MINIMUM, not maximum. This function finds local speed minima preceded
    by high speed (the incoming ball) as contact candidates.

    Args:
        velocities: Dict mapping frame_number to (speed, vx, vy).
        frames: Sorted frame numbers with velocity data.
        smoothed_speeds: Smoothed speed values aligned with frames.
        min_distance_frames: Minimum frames between candidates.
        min_speed_before: Minimum speed in the window before the candidate
            to qualify as "incoming ball" (filters slow-ball noise).
        min_speed_drop_ratio: Minimum ratio of speed drop (1 - min/max)
            within the analysis window. 0.3 = speed must drop by ≥30%.
        window: Frames before/after to check for speed context.

    Returns:
        Sorted list of candidate frame numbers where deceleration occurs.
    """
    if len(frames) < window * 2 + 1:
        return []

    decel_frames: list[tuple[int, float]] = []

    for i in range(window, len(frames) - window):
        curr_frame = frames[i]
        curr_speed = smoothed_speeds[i]

        # Look at speed in the window BEFORE this frame (i >= window guaranteed)
        before_speeds = smoothed_speeds[i - window:i]
        if not before_speeds:
            continue

        max_before = max(before_speeds)
        if max_before < min_speed_before:
            continue  # No fast incoming ball

        drop_ratio = 1.0 - (curr_speed / max_before)
        if drop_ratio < min_speed_drop_ratio:
            continue  # Not enough deceleration

        # Verify it's a local minimum: speed at this frame should be ≤ neighbors
        after_speeds = smoothed_speeds[i + 1:min(len(smoothed_speeds), i + window + 1)]
        if not after_speeds:
            continue

        # Must be lower than at least half the before and after neighbors
        lower_than_before = sum(1 for s in before_speeds if curr_speed <= s)
        lower_than_after = sum(1 for s in after_speeds if curr_speed <= s)
        if lower_than_before < len(before_speeds) // 2:
            continue
        if lower_than_after < len(after_speeds) // 2:
            continue

        # Strength = magnitude of the speed drop
        decel_strength = max_before - curr_speed
        decel_frames.append((curr_frame, decel_strength))

    if not decel_frames:
        return []

    # Enforce min distance: keep strongest deceleration when close
    decel_frames.sort(key=lambda x: x[0])
    selected: list[int] = []
    strength_by_frame = dict(decel_frames)

    for frame, strength in decel_frames:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif strength > strength_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _find_parabolic_breakpoints(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    window_frames: int = 12,
    stride: int = 3,
    min_residual: float = 0.015,
    min_prominence: float = 0.008,
    min_distance_frames: int = 12,
) -> tuple[list[int], dict[int, float]]:
    """Find contact candidates by detecting transitions between parabolic arcs.

    A volleyball in free flight follows a parabolic arc (gravity). Each contact
    creates a transition between arcs. Fit parabolas to overlapping windows and
    compute per-frame fitting residuals. Residual peaks indicate contacts.

    Key insight: arc apexes (current FP source) lie ON the parabola and have
    LOW residuals. Contacts BREAK the parabola and have HIGH residuals.

    Args:
        ball_by_frame: Ball positions indexed by frame number.
        confident_frames: Sorted frame numbers with confident detections.
        window_frames: Size of sliding window for parabolic fit.
        stride: Step size for sliding window.
        min_residual: Minimum residual to consider as breakpoint.
        min_prominence: Minimum peak prominence.
        min_distance_frames: Minimum distance between breakpoints.

    Returns:
        Tuple of (breakpoint_frames, residual_by_frame) where residual_by_frame
        maps frame numbers to their arc fit residuals.
    """
    from scipy.signal import find_peaks

    if len(confident_frames) < window_frames:
        return [], {}

    # Build arrays of (frame, x, y) for confident positions
    frames_arr = np.array(confident_frames, dtype=np.float64)
    x_arr = np.array([ball_by_frame[f].x for f in confident_frames])
    y_arr = np.array([ball_by_frame[f].y for f in confident_frames])

    # Per-frame residual accumulator (sum of residuals, count of windows)
    residual_sum: dict[int, float] = {}
    residual_count: dict[int, int] = {}

    # Slide window across the trajectory
    for start_idx in range(0, len(confident_frames) - window_frames + 1, stride):
        end_idx = start_idx + window_frames
        win_frames = frames_arr[start_idx:end_idx]
        win_x = x_arr[start_idx:end_idx]
        win_y = y_arr[start_idx:end_idx]

        # Normalize frame numbers to [0, 1] for numerical stability
        t = win_frames - win_frames[0]
        t_range = t[-1]
        if t_range < 1.0:
            continue
        t = t / t_range

        # Fit degree-2 polynomial (parabola) to x(t) and y(t)
        try:
            px = np.polyfit(t, win_x, 2)
            py = np.polyfit(t, win_y, 2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # Compute residuals per frame
        x_pred = np.polyval(px, t)
        y_pred = np.polyval(py, t)
        residuals = np.sqrt((win_x - x_pred) ** 2 + (win_y - y_pred) ** 2)

        for i in range(window_frames):
            frame = confident_frames[start_idx + i]
            residual_sum[frame] = residual_sum.get(frame, 0.0) + float(residuals[i])
            residual_count[frame] = residual_count.get(frame, 0) + 1

    # Average residuals across all windows that included each frame
    residual_by_frame: dict[int, float] = {}
    for frame in confident_frames:
        if frame in residual_sum and residual_count.get(frame, 0) > 0:
            residual_by_frame[frame] = residual_sum[frame] / residual_count[frame]

    if not residual_by_frame:
        return [], {}

    # Extract peaks from the residual signal
    ordered_frames = sorted(residual_by_frame.keys())
    residual_signal = [residual_by_frame[f] for f in ordered_frames]

    if len(residual_signal) < 3:
        return [], residual_by_frame

    peak_indices, _ = find_peaks(
        residual_signal,
        height=min_residual,
        prominence=min_prominence,
        distance=max(1, min_distance_frames // stride),
    )

    breakpoint_frames = [ordered_frames[idx] for idx in peak_indices]

    # Enforce min_distance_frames between breakpoints
    if len(breakpoint_frames) > 1:
        filtered = [breakpoint_frames[0]]
        for frame in breakpoint_frames[1:]:
            if frame - filtered[-1] >= min_distance_frames:
                filtered.append(frame)
            elif residual_by_frame.get(frame, 0) > residual_by_frame.get(filtered[-1], 0):
                filtered[-1] = frame
        breakpoint_frames = filtered

    logger.debug(
        f"Parabolic breakpoints: {len(breakpoint_frames)} from {len(confident_frames)} frames"
    )

    return breakpoint_frames, residual_by_frame


def _compute_acceleration(
    velocities: dict[int, tuple[float, float, float]],
    frame: int,
    window: int = 3,
) -> float:
    """Compute acceleration (velocity change magnitude) near a frame.

    Returns the maximum absolute speed change between consecutive frames
    within ±window of the target frame. Contacts cause sudden speed changes.
    """
    nearby_frames = sorted(f for f in velocities if abs(f - frame) <= window)
    if len(nearby_frames) < 2:
        return 0.0

    max_accel = 0.0
    for i in range(1, len(nearby_frames)):
        prev_f = nearby_frames[i - 1]
        curr_f = nearby_frames[i]
        gap = curr_f - prev_f
        if gap <= 0 or gap > 5:
            continue
        speed_prev = velocities[prev_f][0]
        speed_curr = velocities[curr_f][0]
        accel = abs(speed_curr - speed_prev) / gap
        if accel > max_accel:
            max_accel = accel

    return max_accel


def _compute_velocity_ratio(
    velocities: dict[int, tuple[float, float, float]],
    frame: int,
    window: int = 5,
) -> float:
    """Compute speed ratio after/before a candidate frame.

    Returns after_speed / before_speed. Values > 1 mean ball accelerated
    (e.g. attack), < 1 mean ball decelerated (e.g. receive/dig).
    Returns 1.0 if insufficient data.
    """
    before_speeds = [
        velocities[f][0] for f in range(frame - window, frame)
        if f in velocities
    ]
    after_speeds = [
        velocities[f][0] for f in range(frame + 1, frame + window + 1)
        if f in velocities
    ]
    if not before_speeds or not after_speeds:
        return 1.0
    before_med = sorted(before_speeds)[len(before_speeds) // 2]
    after_med = sorted(after_speeds)[len(after_speeds) // 2]
    return (after_med + 0.001) / (before_med + 0.001)


def _count_consecutive_detections(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
) -> int:
    """Count consecutive ball detections around a candidate frame.

    Counts the length of the contiguous run of ball detections containing
    the candidate frame. Real contacts tend to occur within long continuous
    trajectory segments; isolated detections are more likely noise.
    """
    count = 1  # the frame itself
    # Count backward
    f = frame - 1
    while f in ball_by_frame:
        count += 1
        f -= 1
    # Count forward
    f = frame + 1
    while f in ball_by_frame:
        count += 1
        f += 1
    return count


def _compute_trajectory_curvature(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
    window: int = 5,
) -> float:
    """Compute trajectory curvature (inverse radius) near a frame.

    Uses three points: before, at, and after the candidate frame.
    High curvature = sharp bend (contact). Low curvature = smooth arc (free flight).
    Returns curvature as 1/radius, or 0.0 if not enough data.
    """
    before_frame = None
    after_frame = None

    for offset in range(1, window + 1):
        if before_frame is None and (frame - offset) in ball_by_frame:
            before_frame = frame - offset
        if after_frame is None and (frame + offset) in ball_by_frame:
            after_frame = frame + offset
        if before_frame is not None and after_frame is not None:
            break

    if before_frame is None or after_frame is None:
        return 0.0

    bp_at = ball_by_frame.get(frame)
    if bp_at is None:
        return 0.0

    bp_before = ball_by_frame[before_frame]
    bp_after = ball_by_frame[after_frame]

    # Menger curvature: 4*area / (|AB|*|BC|*|CA|)
    ax, ay = bp_before.x, bp_before.y
    bx, by = bp_at.x, bp_at.y
    cx, cy = bp_after.x, bp_after.y

    # Twice the signed area of triangle
    twice_area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))

    ab = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
    bc = math.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
    ca = math.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)

    denom = ab * bc * ca
    if denom < 1e-12:
        return 0.0

    return twice_area / denom


# Calibrated on 328 rallies (1466 transitions): 75.0% accuracy vs 62.5%
# with the previous side-of-net comparison.
_NET_CROSSING_Y_THRESHOLD = 0.15


def ball_crossed_net(
    ball_positions: list[BallPosition],
    from_frame: int,
    to_frame: int,
    net_y: float,
    min_frames_per_side: int = 2,
) -> bool | None:
    """Check if ball crossed net between two contacts using Y displacement.

    Compares the absolute Y displacement between the start and end of the
    inter-contact window against a calibrated threshold.  Large displacement
    indicates the ball traveled across the court (net crossing); small
    displacement indicates same-side play.

    This is more robust than comparing which side of net_y each endpoint
    falls on, because the threshold approach is insensitive to net_y
    estimation errors.

    Args:
        ball_positions: Sorted list of confident ball positions.
        from_frame: Start of frame range (exclusive).
        to_frame: End of frame range (exclusive).
        net_y: Net Y position (retained for API compatibility).
        min_frames_per_side: Min frames on each side to confirm crossing.

    Returns:
        True if ball crossed net, False if confirmed no crossing,
        None if insufficient data to determine.
    """
    positions_in_range = [
        bp for bp in ball_positions
        if from_frame < bp.frame_number < to_frame
    ]
    min_required = min_frames_per_side * 2
    if len(positions_in_range) < min_required:
        return None

    # Median Y near the start and end of the window
    n = min_frames_per_side
    start_median_y = statistics.median(
        bp.y for bp in positions_in_range[:n]
    )
    end_median_y = statistics.median(
        bp.y for bp in positions_in_range[-n:]
    )

    y_delta = abs(end_median_y - start_median_y)
    if y_delta > _NET_CROSSING_Y_THRESHOLD:
        return True
    return False


def _check_net_crossing(
    ball_by_frame: dict[int, BallPosition],
    frame: int,
    net_y: float,
    window: int = 5,
) -> bool:
    """Check if the ball crosses net_y within ±window frames of the given frame."""
    nearby = sorted(f for f in ball_by_frame if abs(f - frame) <= window)
    if len(nearby) < 2:
        return False

    for i in range(1, len(nearby)):
        prev_y = ball_by_frame[nearby[i - 1]].y
        curr_y = ball_by_frame[nearby[i]].y
        if (prev_y < net_y <= curr_y) or (curr_y < net_y <= prev_y):
            return True

    return False


def _find_net_crossing_candidates(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    net_y: float,
    min_distance_frames: int,
) -> list[int]:
    """Find candidates where ball crosses net Y position.

    When the ball Y crosses net_y between consecutive confident frames,
    a contact likely occurred (attack/serve crossing the net). The frame
    just before the crossing is marked as a candidate.

    Returns:
        Sorted list of candidate frame numbers.
    """
    if len(confident_frames) < 2:
        return []

    crossing_frames: list[tuple[int, float]] = []

    for i in range(1, len(confident_frames)):
        prev_frame = confident_frames[i - 1]
        curr_frame = confident_frames[i]

        # Only consider consecutive-ish frames (skip large gaps)
        if curr_frame - prev_frame > 5:
            continue

        prev_y = ball_by_frame[prev_frame].y
        curr_y = ball_by_frame[curr_frame].y

        # Check if net_y is crossed (ball goes from one side to the other)
        if (prev_y < net_y <= curr_y) or (curr_y < net_y <= prev_y):
            # Magnitude of crossing = distance traveled across net
            crossing_mag = abs(curr_y - prev_y)
            # Use the frame just before crossing
            crossing_frames.append((prev_frame, crossing_mag))

    if not crossing_frames:
        return []

    # Enforce min distance: keep strongest crossing when close
    crossing_frames.sort(key=lambda x: x[0])
    selected: list[int] = []
    mag_by_frame = dict(crossing_frames)

    for frame, mag in crossing_frames:
        if not selected:
            selected.append(frame)
            continue

        if frame - selected[-1] >= min_distance_frames:
            selected.append(frame)
        elif mag > mag_by_frame[selected[-1]]:
            selected[-1] = frame

    return selected


def _find_post_serve_receive_candidate(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    net_y: float,
    serve_frame: int,
    player_positions: list[PlayerPosition] | None,
    search_window: int = 15,
    player_search_frames: int = 5,
    max_player_distance: float = 0.15,
    max_frames_after_serve: int = 60,
) -> int | None:
    """Find a receive candidate by locating the net crossing after a serve.

    After a serve, the ball crosses the net and the receiver contacts it.
    This function finds where the ball Y crosses net_y after the serve,
    then searches around that crossing point for the frame of minimum
    player-ball distance — the actual receive moment.

    Args:
        ball_by_frame: Frame-indexed ball positions.
        confident_frames: Sorted confident frame numbers.
        net_y: Net Y position.
        serve_frame: Frame of the detected serve.
        player_positions: Player positions for proximity search.
        search_window: ±frames around net crossing to search for player.
        player_search_frames: Frames for player lookup.
        max_player_distance: Max ball-player distance for a valid receive.
        max_frames_after_serve: Max frames after serve to search for crossing.

    Returns:
        Frame number of the receive candidate, or None if not found.
    """
    # Find the first net crossing after the serve
    crossing_frame: int | None = None
    frames_after_serve = [
        f for f in confident_frames
        if serve_frame < f <= serve_frame + max_frames_after_serve
    ]
    if len(frames_after_serve) < 2:
        return None

    for i in range(1, len(frames_after_serve)):
        prev_f = frames_after_serve[i - 1]
        curr_f = frames_after_serve[i]
        if curr_f - prev_f > 5:
            continue
        prev_y = ball_by_frame[prev_f].y
        curr_y = ball_by_frame[curr_f].y
        if (prev_y < net_y <= curr_y) or (curr_y < net_y <= prev_y):
            crossing_frame = curr_f
            break

    if crossing_frame is None:
        return None

    # Require player positions for proximity validation
    if not player_positions:
        return None

    best_frame: int | None = None
    best_distance = max_player_distance

    for offset in range(-search_window, search_window + 1):
        f = crossing_frame + offset
        ball = ball_by_frame.get(f)
        if ball is None:
            continue

        _, dist, _ = _find_nearest_player(
            f, ball.x, ball.y, player_positions,
            search_frames=player_search_frames,
        )
        if dist < best_distance:
            best_distance = dist
            best_frame = f

    return best_frame


def _merge_candidates(
    velocity_peak_frames: list[int],
    inflection_frames: list[int],
    min_distance_frames: int,
) -> list[int]:
    """Merge velocity peak and inflection candidates, preferring velocity peaks.

    Velocity peaks are kept as-is. Inflection candidates are added only if
    no existing candidate is within min_distance_frames.
    """
    merged = set(velocity_peak_frames)

    for frame in inflection_frames:
        too_close = any(
            abs(frame - existing) < min_distance_frames
            for existing in merged
        )
        if not too_close:
            merged.add(frame)

    return sorted(merged)


def estimate_net_position(
    ball_positions: list[BallPosition],
    confidence_threshold: float = _CONFIDENCE_THRESHOLD,
) -> float:
    """Estimate net Y position from ball trajectory Y extrema midpoint.

    Uses the midpoint between local Y minima (far-side arc peaks) and local
    Y maxima (near-side arc peaks) for more robust estimation than simple
    direction-reversal median.

    Returns:
        Estimated net Y position (normalized 0-1). Returns 0.5 as fallback.
    """
    confident = [
        bp for bp in ball_positions if bp.confidence >= confidence_threshold
    ]
    confident.sort(key=lambda bp: bp.frame_number)

    if len(confident) < 10:
        return 0.5

    # Find local Y minima and maxima using a sliding window
    window = 5
    local_minima_ys: list[float] = []
    local_maxima_ys: list[float] = []

    for i in range(window, len(confident) - window):
        curr_y = confident[i].y
        neighbors_before = [confident[j].y for j in range(i - window, i)]
        neighbors_after = [confident[j].y for j in range(i + 1, i + window + 1)]

        if all(curr_y <= y for y in neighbors_before) and all(
            curr_y <= y for y in neighbors_after
        ):
            local_minima_ys.append(curr_y)
        elif all(curr_y >= y for y in neighbors_before) and all(
            curr_y >= y for y in neighbors_after
        ):
            local_maxima_ys.append(curr_y)

    if local_minima_ys and local_maxima_ys:
        return (float(np.median(local_minima_ys)) + float(np.median(local_maxima_ys))) / 2

    # Fallback: Y range midpoint
    y_values = [bp.y for bp in confident]
    return (min(y_values) + max(y_values)) / 2


def _find_direction_change_candidates(
    ball_by_frame: dict[int, BallPosition],
    confident_frames: list[int],
    min_angle_deg: float = 25.0,
    check_frames: int = 8,
    min_distance_frames: int = 12,
    prominence: float = 10.0,
) -> list[int]:
    """Find candidate frames at local maxima of direction change.

    Existing generators (velocity peaks, inflections) detect the EFFECT of
    a contact (the velocity spike after the hit). This generator detects the
    CAUSE — the frame where the ball trajectory changes direction most sharply.

    Uses scipy find_peaks on the direction_change signal to find local maxima
    above min_angle_deg with sufficient prominence.
    """
    from scipy.signal import find_peaks as _find_peaks

    if len(confident_frames) < 3:
        return []

    # Compute direction change at every confident frame
    dc_values = []
    dc_frames = []
    for frame in confident_frames:
        dc = compute_direction_change(ball_by_frame, frame, check_frames)
        dc_values.append(dc)
        dc_frames.append(frame)

    if not dc_values:
        return []

    dc_array = np.array(dc_values)

    # Find peaks in the direction-change signal
    peak_indices, _ = _find_peaks(
        dc_array,
        height=min_angle_deg,
        prominence=prominence,
        distance=min_distance_frames,
    )

    return [dc_frames[i] for i in peak_indices]


def compute_seq_max_nonbg(
    sequence_probs: np.ndarray | None,
    frame: int,
    window: int = 5,
) -> float:
    """Max non-background sequence model probability within ±window frames.

    Returns 0.0 when sequence_probs is None or the frame is out of range.
    Used as a contact classifier feature providing temporal context from the
    MS-TCN++ model that single-frame trajectory features lack.
    """
    if sequence_probs is None or sequence_probs.ndim != 2 or sequence_probs.shape[0] < 2:
        return 0.0
    t_seq = sequence_probs.shape[1]
    lo = max(0, frame - window)
    hi = min(t_seq - 1, frame + window)
    if hi < lo:
        return 0.0
    return float(sequence_probs[1:, lo:hi + 1].max())


def _refine_candidates_to_trajectory_peak(
    candidate_frames: list[int],
    ball_by_frame: dict[int, BallPosition],
    direction_check_frames: int = 8,
    search_window: int = 5,
    first_frame: int = 0,
    serve_window_frames: int = 60,
) -> list[int]:
    """Shift each candidate to the frame with maximum direction change.

    Candidate generators (velocity peaks, inflections, reversals) detect the
    EFFECT of a contact — the velocity spike or trajectory change that follows
    the actual contact moment. This corrects the misalignment by searching
    ±search_window frames for the peak direction change.

    Constraints (informed by 364-rally diagnostic):
    - Search window capped at ±5 frames (not ±8). Wider windows cause serves
      to jump 8-16 frames to the ball-toss peak instead of the contact frame.
    - Candidates in the serve window (first 60 frames) are NOT refined — serve
      trajectories have multiple direction-change peaks (toss, contact, arc)
      and refinement picks the wrong one 62% of the time.
    - No internal dedup — the post-classifier _deduplicate_contacts() handles
      dedup with court-side awareness (cross-side distance=4 for attack→block).
    """
    if not candidate_frames:
        return candidate_frames

    refined: list[int] = []
    for frame in candidate_frames:
        # Skip refinement in serve window — multiple trajectory peaks
        if frame - first_frame < serve_window_frames:
            refined.append(frame)
            continue

        best_frame = frame
        best_dir_change = compute_direction_change(
            ball_by_frame, frame, direction_check_frames
        )

        for offset in range(-search_window, search_window + 1):
            if offset == 0:
                continue
            f = frame + offset
            if f not in ball_by_frame:
                continue
            dc = compute_direction_change(
                ball_by_frame, f, direction_check_frames
            )
            if dc > best_dir_change:
                best_dir_change = dc
                best_frame = f

        refined.append(best_frame)

    return refined


def _find_proximity_frame(
    frame: int,
    ball_by_frame: dict[int, BallPosition],
    player_positions: list[PlayerPosition],
    search_window: int = 8,
    player_search_frames: int = 3,
    max_distance: float = 0.15,
) -> int | None:
    """Find frame of minimum player-ball distance within search window.

    Trajectory-based candidates (velocity peaks, inflections) detect the EFFECT
    of a contact, which lags by several frames. The actual contact occurs when
    ball touches the player — the frame of minimum player-ball distance.

    Returns:
        Frame number with minimum distance, or None if no player within
        max_distance in the search window.
    """
    best_frame = None
    best_distance = max_distance

    for offset in range(-search_window, search_window + 1):
        f = frame + offset
        ball = ball_by_frame.get(f)
        if ball is None:
            continue
        _, dist, _ = _find_nearest_player(
            f, ball.x, ball.y, player_positions,
            search_frames=player_search_frames,
        )
        if dist < best_distance:
            best_distance = dist
            best_frame = f

    return best_frame


# Minimum frames between contacts on different court sides.
# Attack → block/dig across the net happens in 2-4 frames, so we allow
# much closer spacing than same-side contacts (which are physically
# impossible within ~0.4s = 12 frames).
_CROSS_SIDE_MIN_DISTANCE = 4


def _deduplicate_contacts(
    contacts: list[Contact],
    min_distance: int,
    adaptive: bool = False,
) -> list[Contact]:
    """Remove duplicate contacts within min_distance frames, keeping higher confidence.

    When adaptive=True, uses shorter distance for cross-side contacts:
    same-side must be min_distance apart, but cross-side (different
    court_side) only need _CROSS_SIDE_MIN_DISTANCE frames apart, since
    attack→block/dig across the net is physically valid in 2-4 frames.
    """
    if not contacts:
        return contacts

    sorted_contacts = sorted(contacts, key=lambda c: c.confidence, reverse=True)
    result: list[Contact] = []

    for contact in sorted_contacts:
        too_close = False
        for existing in result:
            frame_gap = abs(contact.frame - existing.frame)
            if adaptive:
                sides_known = (
                    contact.court_side in ("near", "far")
                    and existing.court_side in ("near", "far")
                )
                if sides_known and contact.court_side != existing.court_side:
                    effective_min = _CROSS_SIDE_MIN_DISTANCE
                else:
                    effective_min = min_distance
            else:
                effective_min = min_distance

            if frame_gap < effective_min:
                too_close = True
                break
        if not too_close:
            result.append(contact)

    return sorted(result, key=lambda c: c.frame)


def _resolve_court_side(
    ball_x: float,
    ball_y: float,
    player_track_id: int,
    team_assignments: dict[int, int] | None,
    court_calibrator: CourtCalibrator | None,
    estimated_net_y: float,
    player_y: float | None = None,
) -> str:
    """Determine which court side the ball is on using multiple signals.

    Signal priority:
    1. Per-rally player identity (median Y based)
    2. Calibration projection — perspective-correct via homography
    3. Nearest player Y position (more reliable than ball Y for near-net contacts)
    4. Ball Y-threshold fallback
    """
    # Signal 1: Per-rally player identity
    if team_assignments and player_track_id >= 0 and player_track_id in team_assignments:
        return "near" if team_assignments[player_track_id] == 0 else "far"

    # Signal 2: Calibration projection (perspective-correct)
    if court_calibrator is not None and court_calibrator.is_calibrated:
        try:
            # image_to_court takes normalized coords (0-1), pixel dims unused
            # but required by API — pass 1,1 since coords are already normalized
            court_x, court_y = court_calibrator.image_to_court(
                (ball_x, ball_y), 1, 1,
            )
            # Beach volleyball: 8m x 16m, near side y < 8.0 (origin at near-left)
            if 0.0 <= court_y <= 16.0:
                return "near" if court_y < 8.0 else "far"
        except (RuntimeError, np.linalg.LinAlgError):
            pass  # Calibration failed, fall through

    # Signal 3: Player Y position — player stays on their side even during
    # near-net actions. Uses ball-trajectory net_y as threshold: despite being
    # computed from ball positions, it separates near/far players better than
    # court_split_y because it sits cleanly between the two teams.
    if player_y is not None:
        return "far" if player_y < estimated_net_y else "near"

    # Signal 4: Ball Y-threshold fallback
    return "far" if ball_y < estimated_net_y else "near"


def detect_contacts(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition] | None = None,
    config: ContactDetectionConfig | None = None,
    net_y: float | None = None,  # deprecated: ignored, kept for caller compat
    frame_count: int | None = None,
    classifier: ContactClassifier | None = None,
    use_classifier: bool = True,
    team_assignments: dict[int, int] | None = None,
    court_calibrator: CourtCalibrator | None = None,
    sequence_probs: np.ndarray | None = None,
) -> ContactSequence:
    """Detect ball contacts from trajectory inflection points and velocity peaks.

    Algorithm:
    1. Pre-filter noise spikes (single-frame false positives)
    2. Estimate net position from ball trajectory
    3. Compute smoothed ball velocity signal
    4. Find velocity peak candidates (local maxima)
    5. Find inflection candidates (direction changes)
    5c. Find parabolic arc breakpoint candidates
    6. Merge candidates (velocity peaks preferred)
    7. Validate each candidate (classifier or hand-tuned gates), attribute player

    Args:
        ball_positions: Ball tracking positions.
        player_positions: Player tracking positions (optional but recommended).
        config: Detection configuration.
        net_y: Deprecated — ignored. Net position is always estimated from ball
            trajectory (court_split_y from player tracking is not an accurate
            proxy for the net's image-space position).
        frame_count: Total rally frames. If provided, candidates beyond this frame
            are suppressed (post-rally ball pickup/warmdown).
        classifier: Optional trained ContactClassifier. When provided, replaces the
            hand-tuned 3-tier validation gates with learned predictions.
        use_classifier: When True (default) and no explicit classifier is provided,
            auto-loads the default classifier from disk if available. Set to False
            to force hand-tuned validation gates.
        team_assignments: Map from track_id → team (0=near, 1=far). Per-rally teams
            from median Y position.
        court_calibrator: Calibrated court projector. When provided, ball side uses
            perspective-correct projection via homography.
        sequence_probs: MS-TCN++ per-frame action probabilities, shape
            (NUM_CLASSES, T). When provided and `cfg.enable_sequence_recovery`
            is True, the main classifier loop rescues trajectory candidates
            the GBM rejected if `max(sequence_probs[1:, f+-5]) >=
            SEQ_RECOVERY_TAU` AND the GBM score >= SEQ_RECOVERY_CLF_FLOOR
            (both constants in `sequence_action_runtime.py`, read at call
            time). This is a two-signal agreement gate — no new candidates
            are injected; only pre-existing trajectory candidates are
            rescued. The old 7 `seq_p_*` features on `CandidateFeatures`
            were dropped pre-2026-04-07.

    Returns:
        ContactSequence with all detected contacts.
    """
    # Auto-load classifier if not explicitly provided
    if classifier is None and use_classifier:
        classifier = _get_default_classifier()
    from scipy.signal import find_peaks

    from rallycut.tracking.temporal_attribution.features import (
        extract_attribution_features,
    )

    cfg = config or ContactDetectionConfig()

    # Auto-load attribution models
    pose_attributor = (
        _get_pose_attributor() if cfg.use_pose_attribution else None
    )
    temporal_attributor = (
        _get_temporal_attributor()
        if cfg.use_temporal_attribution and pose_attributor is None
        else None
    )

    if not ball_positions:
        return ContactSequence()

    # Step 1: Pre-filter noise spikes
    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(
            ball_positions, cfg.noise_spike_max_jump
        )

    # Step 2: Estimate net position from ball trajectory.
    # Always use ball trajectory — the external net_y (often court_split_y from
    # player tracking) reflects where players stand, NOT where the net appears
    # in image space. Ball trajectory extrema bracket the net more accurately.
    estimated_net_y = estimate_net_position(ball_positions)

    # Step 3: Compute velocities from filtered positions
    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return ContactSequence(net_y=estimated_net_y)

    # Sort frames and smooth velocity
    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return ContactSequence(net_y=estimated_net_y)

    speeds = [velocities[f][0] for f in frames]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)

    # Step 4: Find velocity peak candidates
    peak_indices, _ = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )
    velocity_peak_frames = [frames[idx] for idx in peak_indices]

    # Index ball positions by frame (only confident ones)
    ball_by_frame: dict[int, BallPosition] = {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    first_frame = frames[0]

    # Step 5a: Find inflection candidates
    confident_frames = sorted(ball_by_frame.keys())
    inflection_frames: list[int] = []
    if cfg.enable_inflection_detection:
        inflection_frames = _find_inflection_candidates(
            ball_by_frame,
            confident_frames,
            min_angle_deg=cfg.min_inflection_angle_deg,
            check_frames=cfg.inflection_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    # Step 5b: Find velocity reversal candidates
    reversal_frames = _find_velocity_reversal_candidates(
        velocities, frames, cfg.min_peak_distance_frames
    )

    # Step 5c: Find deceleration candidates (catches receives/digs)
    deceleration_frames: list[int] = []
    if cfg.enable_deceleration_detection:
        deceleration_frames = _find_deceleration_candidates(
            velocities,
            frames,
            smoothed,
            cfg.min_peak_distance_frames,
            min_speed_before=cfg.deceleration_min_speed_before,
            min_speed_drop_ratio=cfg.deceleration_min_drop_ratio,
            window=cfg.deceleration_window,
        )

    # Step 5d: Find parabolic arc breakpoint candidates
    parabolic_frames: list[int] = []
    residual_by_frame: dict[int, float] = {}
    if cfg.enable_parabolic_detection:
        parabolic_frames, residual_by_frame = _find_parabolic_breakpoints(
            ball_by_frame,
            confident_frames,
            window_frames=cfg.parabolic_window_frames,
            stride=cfg.parabolic_stride,
            min_residual=cfg.parabolic_min_residual,
            min_prominence=cfg.parabolic_min_prominence,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    # Step 5e: Find net-crossing candidates
    net_crossing_frames = _find_net_crossing_candidates(
        ball_by_frame, confident_frames, estimated_net_y, cfg.min_peak_distance_frames
    )

    # Step 5g: Find direction-change peak candidates
    # Detects the CAUSE of contact (peak trajectory change) rather than the
    # EFFECT (velocity spike). Fills gaps where velocity/inflection generators
    # fire >5 frames from the actual contact.
    direction_change_frames: list[int] = []
    if cfg.enable_direction_change_candidates:
        direction_change_frames = _find_direction_change_candidates(
            ball_by_frame,
            confident_frames,
            min_angle_deg=cfg.direction_change_candidate_min_deg,
            check_frames=cfg.direction_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
            prominence=cfg.direction_change_candidate_prominence,
        )

    # Step 6: Merge all candidates.
    # Direction-change peaks get HIGHEST priority because they fire at the actual
    # contact frame (cause), while velocity/inflection fire at the effect (velocity
    # spike several frames later). When both fire within min_distance, the
    # direction-change frame is kept — it's closer to GT and produces better features.
    # _merge_candidates keeps the first arg's frames, adding second arg's only
    # if no existing candidate is within min_distance_frames.
    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    traditional_candidates = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )
    # Add deceleration candidates (catches receives/digs that decelerate the ball)
    with_deceleration = _merge_candidates(
        traditional_candidates, deceleration_frames, cfg.min_peak_distance_frames
    )
    # Add parabolic breakpoints (catches soft touches missed by velocity/inflection)
    with_parabolic = _merge_candidates(
        with_deceleration, parabolic_frames, cfg.min_peak_distance_frames
    )
    # Add net-crossing candidates (lowest priority — fills gaps from other detectors)
    with_net_crossing = _merge_candidates(
        with_parabolic, net_crossing_frames, cfg.min_peak_distance_frames
    )
    # Merge direction-change peaks LAST but with HIGHER priority: direction-change
    # frames replace nearby velocity/inflection frames because they're closer to
    # the actual contact. _merge_candidates keeps first arg, so we swap the order.
    candidate_frames = _merge_candidates(
        direction_change_frames, with_net_crossing, cfg.min_peak_distance_frames
    ) if direction_change_frames else with_net_crossing

    # Step 5f: Player-motion candidates — detect contacts from player body motion
    # near the ball, even when ball trajectory doesn't change (blocks, soft touches)
    n_player_motion = 0
    if cfg.enable_player_motion_candidates and player_positions:
        player_motion_frames = _find_player_motion_candidates(
            player_positions, ball_by_frame, candidate_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
            min_d_y=cfg.player_motion_min_d_y,
            min_d_height=cfg.player_motion_min_d_height,
            max_ball_distance=cfg.player_motion_max_ball_distance,
        )
        if player_motion_frames:
            n_player_motion = len(player_motion_frames)
            candidate_frames = _merge_candidates(
                candidate_frames, player_motion_frames, cfg.min_peak_distance_frames
            )

    if not candidate_frames:
        return ContactSequence(net_y=estimated_net_y)

    # Step 6a2: Post-serve receive candidate search.
    # After the serve, the ball crosses the net. Search around the crossing
    # point for the nearest player-ball proximity — the receive moment.
    # This structurally generates a receive candidate tied to the serve→receive
    # transition, even when the receive doesn't produce a velocity peak.
    n_post_serve = 0
    if cfg.enable_post_serve_receive and player_positions:
        # Identify likely serve: first candidate in the serve window
        serve_frame: int | None = None
        for f in candidate_frames:
            if f - first_frame < cfg.serve_window_frames:
                serve_frame = f
                break
        if serve_frame is not None:
            receive_frame = _find_post_serve_receive_candidate(
                ball_by_frame, confident_frames, estimated_net_y,
                serve_frame, player_positions,
                search_window=cfg.post_serve_search_window,
                player_search_frames=cfg.player_search_frames,
                max_player_distance=cfg.player_contact_radius,
            )
            if receive_frame is not None:
                candidate_set_check = set(candidate_frames)
                if receive_frame not in candidate_set_check and not any(
                    abs(receive_frame - f) < cfg.min_peak_distance_frames
                    for f in candidate_frames
                ):
                    candidate_frames = sorted(candidate_set_check | {receive_frame})
                    n_post_serve = 1

    # Step 6b: Trajectory-peak refinement (constrained ±5 frames, skip serves).
    # Shifts trajectory candidates to their local direction-change peak. Applied
    # BEFORE proximity candidates so proximity search starts from refined positions.
    if cfg.enable_trajectory_refinement:
        candidate_frames = _refine_candidates_to_trajectory_peak(
            candidate_frames, ball_by_frame,
            direction_check_frames=cfg.direction_check_frames,
            search_window=cfg.trajectory_refinement_window,
            first_frame=first_frame,
            serve_window_frames=cfg.serve_window_frames,
        )

    # Step 6c: Generate player-proximity candidates
    # Proximity candidates are generated at the frame of minimum player-ball
    # distance near each candidate — closer to the actual contact moment.
    n_proximity = 0
    if player_positions and cfg.enable_proximity_candidates:
        candidate_set = set(candidate_frames)
        proximity_frames: list[int] = []
        for frame in candidate_frames:
            prox = _find_proximity_frame(
                frame, ball_by_frame, player_positions,
                search_window=cfg.proximity_search_window,
                player_search_frames=cfg.player_search_frames,
                max_distance=cfg.player_contact_radius,
            )
            if prox is not None and prox != frame and prox not in candidate_set:
                proximity_frames.append(prox)
                candidate_set.add(prox)
        if proximity_frames:
            n_proximity = len(proximity_frames)
            candidate_frames = sorted(candidate_set)

    # Build velocity lookup for any frame
    velocity_lookup = dict(zip(frames, smoothed))

    # Sequence-model support helper. Returns True iff any frame in [f-W, f+W]
    # has `max(sequence_probs[1:, :]) >= SEQ_RECOVERY_TAU`. Used below to
    # rescue trajectory candidates the GBM rejects but the sequence model
    # endorses (two-signal agreement gate). See the big note at step 5e in
    # memory/fn_sequence_signal_2026_04.md for the empirical basis.
    seq_peak_nonbg = None
    seq_recovery_tau = 1.1  # Default: never triggers.
    if (
        cfg.enable_sequence_recovery
        and sequence_probs is not None
        and sequence_probs.ndim == 2
        and sequence_probs.shape[0] >= 2
    ):
        from rallycut.tracking.sequence_action_runtime import (  # noqa: PLC0415
            SEQ_RECOVERY_TAU,
        )
        seq_peak_nonbg = sequence_probs[1:, :].max(axis=0)
        seq_recovery_tau = SEQ_RECOVERY_TAU

    def _has_sequence_support(frame: int, window: int = 5) -> bool:
        if seq_peak_nonbg is None:
            return False
        t_seq = seq_peak_nonbg.shape[0]
        lo = max(0, frame - window)
        hi = min(t_seq - 1, frame + window)
        if hi < lo:
            return False
        return bool(seq_peak_nonbg[lo:hi + 1].max() >= seq_recovery_tau)

    def _get_seq_max_nonbg(frame: int, window: int = 5) -> float:
        return compute_seq_max_nonbg(sequence_probs, frame, window)

    net_zone = 0.08  # ±8% of screen around net
    contacts: list[Contact] = []
    prev_accepted_frame = 0  # Track ACCEPTED contacts for frames_since_last

    for frame in candidate_frames:
        # Skip warmup period (ball tracking produces false detections early)
        if frame - first_frame < cfg.warmup_skip_frames:
            continue

        # Skip post-rally candidates (ball pickup, warmdown)
        if frame_count is not None and frame_count > 0 and frame > frame_count:
            continue

        # Get velocity (may be 0 for inflection-only candidates)
        velocity = velocity_lookup.get(frame, 0.0)

        # Get ball position at candidate frame
        ball = ball_by_frame.get(frame)
        if ball is None:
            for offset in [-1, 1, -2, 2, -3, 3]:
                ball = ball_by_frame.get(frame + offset)
                if ball is not None:
                    break
        if ball is None:
            continue

        # Compute direction change
        direction_change = compute_direction_change(
            ball_by_frame, frame, cfg.direction_check_frames
        )

        # Find nearest player (narrow window — matches classifier training semantics)
        # MUST use image-space distance to preserve classifier feature distribution.
        # Note: velocities now use central differences (reduced lag vs old backward
        # difference).  player_search_frames=5 still covers residual timing jitter.
        nearest_player_y: float | None = None
        if player_positions:
            track_id, player_dist, nearest_player_y = _find_nearest_player(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_search_frames,
            )
        else:
            track_id = -1
            player_dist = float("inf")

        # Find ranked candidates using depth-corrected distance for re-attribution.
        # Perspective compression makes far-court players appear closer than they
        # are; depth scaling corrects this (median 3.9x far-to-near ratio).
        candidates: list[tuple[int, float, float]] = []
        bbox_motion: dict[int, tuple[float, float]] = {}
        if player_positions:
            candidates = _find_nearest_players(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_candidate_search_frames,
                court_calibrator=court_calibrator,
            )
            if candidates:
                bbox_motion = _compute_candidate_bbox_motion(
                    player_positions, frame,
                    [tid for tid, _, _ in candidates],
                )

        # Velocity floor: skip very low velocity candidates (tracking noise)
        if velocity < cfg.min_candidate_velocity:
            continue

        # Determine court side: per-rally > calibration > player Y > ball Y
        court_side = _resolve_court_side(
            ball.x, ball.y, track_id, team_assignments,
            court_calibrator, estimated_net_y,
            player_y=nearest_player_y,
        )

        # Check if at net
        is_at_net = abs(ball.y - estimated_net_y) < net_zone

        has_player = player_dist <= cfg.player_contact_radius
        has_direction_change = direction_change >= cfg.min_direction_change_deg
        arc_residual = residual_by_frame.get(frame, 0.0)

        # Compute new trajectory features
        acceleration = _compute_acceleration(velocities, frame, window=3)
        curvature = _compute_trajectory_curvature(ball_by_frame, frame, window=5)
        is_net_cross = _check_net_crossing(
            ball_by_frame, frame, estimated_net_y, window=5
        )

        # Compute frames since last ACCEPTED contact (not just any candidate).
        # Previous semantics tracked all candidates, but this caused the GBM to
        # penalize rapid real contacts (attack→dig at 3-5 frame spacing) because
        # frames_since_last was always small in dense candidate regions. Measuring
        # from last accepted contact means only validated contacts affect spacing.
        # Must match training script semantics.
        frames_since_last = (
            frame - prev_accepted_frame if prev_accepted_frame > 0 else 0
        )

        if classifier is not None and classifier.is_trained:
            # Phase 3: Use learned classifier
            from rallycut.tracking.contact_classifier import CandidateFeatures

            # Ball detection density: fraction of ±10 frames with ball
            density_window = 10
            n_with_ball = sum(
                1 for f in range(frame - density_window, frame + density_window + 1)
                if f in ball_by_frame
            )
            ball_detection_density = n_with_ball / (2 * density_window + 1)

            # Vertical velocity component
            vel_y = velocities[frame][2] if frame in velocities else 0.0

            # Velocity ratio: speed after / speed before candidate
            vel_ratio = _compute_velocity_ratio(velocities, frame, window=5)

            # Consecutive ball detections around candidate frame
            consec = _count_consecutive_detections(ball_by_frame, frame)

            # Aggregate bbox motion features from candidate players
            best_d_y = max((dy for dy, _ in bbox_motion.values()), default=0.0)
            best_d_h = max((dh for _, dh in bbox_motion.values()), default=0.0)
            nearest_d_y = bbox_motion.get(track_id, (0.0, 0.0))[0] if track_id >= 0 else 0.0
            nearest_d_h = bbox_motion.get(track_id, (0.0, 0.0))[1] if track_id >= 0 else 0.0

            # NOTE 2026-04-07: MS-TCN++ seq_p_* features removed from
            # CandidateFeatures after the contact_classifier_audit found
            # importance == 0.0000 (trainer always passed zero-filled probs).
            # `sequence_probs` is still accepted for backward compat with
            # callers but no longer feeds the contact GBM. The MS-TCN++ signal
            # still reaches actions via apply_sequence_override at stage 14.

            # Pose features for nearest player (0.0 if no keypoints)
            from rallycut.tracking.pose_attribution.features import (
                extract_contact_pose_features_for_nearest,
            )
            (
                pose_wrist_vel_max,
                pose_hand_ball_dist_min,
                pose_arm_ext_change,
                pose_conf_mean,
                pose_both_arms_raised,
            ) = extract_contact_pose_features_for_nearest(
                contact_frame=frame,
                nearest_track_id=track_id,
                player_positions=player_positions or [],
                ball_at_contact=(ball.x, ball.y),
                ball_by_frame=ball_by_frame,
            )

            features = CandidateFeatures(
                frame=frame,
                velocity=velocity,
                direction_change_deg=direction_change,
                arc_fit_residual=arc_residual,
                acceleration=acceleration,
                trajectory_curvature=curvature,
                velocity_y=vel_y,
                velocity_ratio=vel_ratio,
                player_distance=player_dist,
                best_player_max_d_y=best_d_y,
                best_player_max_d_height=best_d_h,
                nearest_player_max_d_y=nearest_d_y,
                nearest_player_max_d_height=nearest_d_h,
                ball_x=ball.x,
                ball_y=ball.y,
                ball_y_relative_net=ball.y - estimated_net_y,
                is_net_crossing=is_net_cross,
                frames_since_last=frames_since_last,
                ball_detection_density=ball_detection_density,
                consecutive_detections=consec,
                frames_since_rally_start=frame - first_frame,
                nearest_active_wrist_velocity_max=pose_wrist_vel_max,
                nearest_hand_ball_dist_min=pose_hand_ball_dist_min,
                nearest_active_arm_extension_change=pose_arm_ext_change,
                nearest_pose_confidence_mean=pose_conf_mean,
                nearest_both_arms_raised=pose_both_arms_raised,
                seq_max_nonbg=_get_seq_max_nonbg(frame),
            )
            results = classifier.predict([features])
            is_validated, confidence = results[0]
            # Two-signal agreement rescue: if the classifier rejected this
            # candidate but the sequence model has a non-background peak
            # >= SEQ_RECOVERY_TAU within +-5 frames AND the classifier gave
            # it a non-trivial score >= SEQ_RECOVERY_CLF_FLOOR, accept it.
            #
            # This rescues the 181 rejected_by_classifier FNs documented in
            # memory/fn_sequence_signal_2026_04.md: trajectory candidates
            # where the GBM scores median 0.22 (under 0.35 gate) but MS-TCN++
            # endorses the frame with peak >= 0.80. Two detectors with
            # asymmetric strengths agreeing is stronger evidence than either
            # alone, so their conjunction warrants a lower single-source
            # confidence requirement.
            if (
                not is_validated
                and seq_peak_nonbg is not None
                and _has_sequence_support(frame)
            ):
                from rallycut.tracking.sequence_action_runtime import (  # noqa: PLC0415
                    SEQ_RECOVERY_CLF_FLOOR,
                )
                if confidence >= SEQ_RECOVERY_CLF_FLOOR:
                    is_validated = True
        else:
            # Fallback: Hand-tuned 3-tier validation gates
            # Tier 1: Strong signal — high velocity + direction change (definitive)
            # Tier 2: High velocity + player nearby (serves/spikes without direction
            #         change, e.g. when ball tracking has gaps around the contact)
            # Tier 3: Player confirmed — player nearby + trajectory signal
            is_high_velocity = velocity >= cfg.high_velocity_threshold
            is_strong = is_high_velocity and has_direction_change
            is_fast_with_player = is_high_velocity and has_player
            is_player_confirmed = has_player and (
                has_direction_change or velocity >= cfg.min_peak_velocity
            )
            is_validated = is_strong or is_fast_with_player or is_player_confirmed
            confidence = 0.0

        if not is_validated:
            continue

        # Sequential attribution fix: when the nearest player is the same as
        # the previous contact AND a close alternative exists, prefer the
        # alternative. In volleyball, the same player rarely touches the ball
        # consecutively. Applied AFTER validation to preserve the contact
        # classifier's feature distribution (player_distance unchanged).
        if (
            track_id >= 0
            and contacts
            and track_id == contacts[-1].player_track_id
            and len(candidates) >= 2
        ):
            _, d1, _ = candidates[0]
            alt_tid, d2, _alt_y = candidates[1]
            margin = d2 - d1
            if margin < 0.05 and alt_tid != track_id:
                track_id = alt_tid
                player_dist = d2

        # Attribution model override: either pose-based or temporal.
        # Pose attribution uses per-candidate binary scoring;
        # temporal attribution uses canonical-slot prediction.
        # Only applied when ≥2 candidates exist.
        if (
            pose_attributor is not None
            and player_positions
            and len(candidates) >= 2
        ):
            from rallycut.tracking.pose_attribution.features import (
                extract_candidate_features,
            )

            pa_contact_index = len(contacts)
            pa_side_count = 1
            for prev_c in reversed(contacts):
                if prev_c.court_side == court_side:
                    pa_side_count += 1
                else:
                    break
            pa_side_count = min(pa_side_count, 3)

            cand_result = extract_candidate_features(
                contact_frame=frame,
                ball_positions=ball_positions,
                player_positions=player_positions,
                contact_index=pa_contact_index,
                side_count=pa_side_count,
            )
            if cand_result is not None:
                cand_feats = [f for _, f in cand_result]
                cand_tids = [tid for tid, _ in cand_result]
                pred_tid, pred_conf = pose_attributor.predict(
                    cand_feats, cand_tids
                )
                if (
                    pred_conf >= cfg.pose_attribution_min_confidence
                    and pred_tid >= 0
                ):
                    track_id = pred_tid
                    for cand_tid, cand_dist, _ in candidates:
                        if cand_tid == pred_tid:
                            player_dist = cand_dist
                            break
        elif (
            temporal_attributor is not None
            and player_positions
            and len(candidates) >= 2
        ):
            # Compute contact sequence context. Note: contact_index counts only
            # validated contacts here, while training uses all stored contacts.
            # The tree model is robust to this small divergence.
            ta_contact_index = len(contacts)
            ta_side_count = 1
            for prev_c in reversed(contacts):
                if prev_c.court_side == court_side:
                    ta_side_count += 1
                else:
                    break
            ta_side_count = min(ta_side_count, 3)

            feat_result = extract_attribution_features(
                contact_frame=frame,
                ball_positions=ball_positions,
                player_positions=player_positions,
                contact_index=ta_contact_index,
                side_count=ta_side_count,
            )
            if feat_result is not None:
                feats, canonical_tids = feat_result
                pred_tid, pred_conf = temporal_attributor.predict(
                    feats, canonical_tids
                )
                if (
                    pred_conf >= cfg.temporal_attribution_min_confidence
                    and pred_tid >= 0
                ):
                    track_id = pred_tid
                    # Update distance to reflect the chosen player
                    for cand_tid, cand_dist, _ in candidates:
                        if cand_tid == pred_tid:
                            player_dist = cand_dist
                            break

        contacts.append(Contact(
            frame=frame,
            ball_x=ball.x,
            ball_y=ball.y,
            velocity=velocity,
            direction_change_deg=direction_change,
            player_track_id=track_id,
            player_distance=player_dist,
            player_candidates=[(tid, d) for tid, d, _y in candidates],
            candidate_bbox_motion=bbox_motion,
            court_side=court_side,
            is_at_net=is_at_net,
            is_validated=is_validated,
            confidence=confidence,
            arc_fit_residual=arc_residual,
        ))
        prev_accepted_frame = frame

    # Deduplicate contacts from proximity + standard candidates at similar frames
    pre_dedup = len(contacts)
    contacts = _deduplicate_contacts(
        contacts, cfg.min_peak_distance_frames, adaptive=cfg.adaptive_dedup,
    )

    logger.info(
        f"Detected {len(contacts)} contacts "
        f"({len(velocity_peak_frames)} vel peaks + "
        f"{len(inflection_frames)} inflections + "
        f"{len(deceleration_frames)} decel + "
        f"{len(parabolic_frames)} parabolic + "
        f"{len(direction_change_frames)} dir-change + "
        f"{len(net_crossing_frames)} net-cross + "
        f"{n_post_serve} post-serve + "
        f"{n_player_motion} player-motion + "
        f"{n_proximity} proximity → "
        f"{len(candidate_frames)} candidates"
        f"{f', {pre_dedup - len(contacts)} deduped' if pre_dedup > len(contacts) else ''}"
        f", net_y={estimated_net_y:.3f})"
    )

    # Pass confident ball positions for net-crossing detection in action classifier
    confident_positions = [
        bp for bp in ball_positions if bp.confidence >= _CONFIDENCE_THRESHOLD
    ]

    return ContactSequence(
        contacts=contacts,
        net_y=estimated_net_y,
        rally_start_frame=first_frame,
        ball_positions=confident_positions,
        player_positions=player_positions or [],
    )
