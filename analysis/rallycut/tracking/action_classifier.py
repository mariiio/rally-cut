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
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any

from rallycut.tracking.contact_detector import Contact, ContactSequence, ball_crossed_net

if TYPE_CHECKING:
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

    @property
    def serve(self) -> ClassifiedAction | None:
        """Get the serve action."""
        for a in self.actions:
            if a.action_type == ActionType.SERVE:
                return a
        return None

    @property
    def serving_team(self) -> str | None:
        """Get the team that served, or None if unknown."""
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
) -> tuple[int, str, float]:
    """Identify the server from player positions at rally start.

    In beach volleyball (2v2), the server is the only player standing near a
    baseline at rally start.  All others are mid-court or at the net.

    Strategy: split players into near-side (foot_y > net_y) and far-side
    (foot_y < net_y).  On each side, the player furthest from the net is the
    server candidate.  The candidate must be separated from their same-side
    teammate by *separation_min*.  The side with the larger separation wins.

    Returns:
        (server_track_id, court_side, confidence).
        (-1, "", 0.0) when no server can be identified.
    """
    end_frame = start_frame + window_frames

    # Gather foot-Y per track (foot ≈ bottom of bbox = y + height/2)
    track_foot_ys: dict[int, list[float]] = defaultdict(list)
    for p in player_positions:
        if start_frame <= p.frame_number < end_frame:
            track_foot_ys[p.track_id].append(p.y + p.height / 2.0)

    if not track_foot_ys:
        return -1, "", 0.0

    # Mean foot-Y per track
    track_means: dict[int, float] = {
        tid: sum(ys) / len(ys) for tid, ys in track_foot_ys.items()
    }

    # Split into near (foot_y > net_y) and far (foot_y <= net_y) sides.
    # Distance from net = how far toward the baseline the player is.
    near_tracks: list[tuple[int, float]] = []  # (tid, dist_from_net)
    far_tracks: list[tuple[int, float]] = []
    for tid, foot_y in track_means.items():
        if foot_y > net_y:
            near_tracks.append((tid, foot_y - net_y))
        else:
            far_tracks.append((tid, net_y - foot_y))

    # On each side, find the server candidate (furthest from net) and the
    # separation from their teammate.
    candidates: list[tuple[int, str, float, float]] = []  # tid, side, sep, dist
    for side, tracks in [("near", near_tracks), ("far", far_tracks)]:
        if not tracks:
            continue
        tracks.sort(key=lambda t: t[1], reverse=True)
        best_tid, best_dist = tracks[0]
        if len(tracks) >= 2:
            teammate_dist = tracks[1][1]
            sep = best_dist - teammate_dist
        else:
            sep = best_dist  # Only player on this side
        candidates.append((best_tid, side, sep, best_dist))

    if not candidates:
        return -1, "", 0.0

    # Pick the side with the largest separation
    candidates.sort(key=lambda t: t[2], reverse=True)
    best_tid, best_side, best_sep, best_dist = candidates[0]

    if best_sep < separation_min:
        return -1, "", 0.0

    # Confidence: based on separation from teammate
    confidence = min(1.0, best_sep / 0.10)

    return best_tid, best_side, confidence


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
) -> ClassifiedAction:
    """Create a synthetic serve action for a missed serve.

    Places the serve at the rally start frame when available and
    reasonably close to the first detected contact, otherwise ~1s
    (30 frames) before the first contact.

    Args:
        serve_side: Court side of the serve ("near" or "far").
        first_contact_frame: Frame of the first detected contact.
        net_y: Net Y position.
        rally_start_frame: Frame when the rally segment starts (from
            detection). Used for more accurate serve placement.

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

    return ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=serve_frame,
        ball_x=0.5,
        ball_y=baseline_near if serve_side == "near" else baseline_far,
        velocity=0.0,
        player_track_id=-1,
        court_side=serve_side,
        confidence=0.4,
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
                        # Only trigger phantom on confirmed False.
                        # None (insufficient data) → keep as serve
                        # (conservative).
                        if toward_net is False:
                            is_phantom = True

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

                    # Classifier override: if the learned classifier says
                    # "serve", trust it over trajectory-based phantom detection.
                    if (
                        is_phantom
                        and classifier is not None
                        and classifier.is_trained
                    ):
                        feat = extract_action_features(
                            contact=contact, index=i, all_contacts=contacts,
                            ball_positions=ball_positions,
                            net_y=contact_sequence.net_y,
                            rally_start_frame=start_frame,
                            team_assignments=match_team_assignments,
                        )
                        pred_action, pred_conf = classifier.predict([feat])[0]
                        if pred_action == "serve":
                            is_phantom = False
                            logger.debug(
                                "Classifier overrides phantom serve at "
                                "frame %d (conf=%.2f)",
                                contact.frame, pred_conf,
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
                        )
                        # Attach server track_id from position detection
                        if server_pos_tid >= 0:
                            synth = replace(synth, player_track_id=server_pos_tid)
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
            mode = "learned" if classifier is not None else "rule-based"
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


def repair_action_sequence(
    actions: list[ClassifiedAction],
    net_y: float = 0.5,
    ball_positions: list[BallPosition] | None = None,
    rally_start_frame: int | None = None,
) -> list[ClassifiedAction]:
    """Repair volleyball-illegal action sequences.

    Missed contacts cause cascade failures in the state machine: missing one
    contact shifts ALL subsequent labels. This function detects and fixes
    common illegal patterns using volleyball rules as constraints.

    Same-side rules only — cross-side sequences (e.g., dig on near → dig on
    far) are legal because each side's touch counter resets independently.

    Repairs applied (in order):
    0. (serve fix) Non-synthetic serve with ball on wrong court side →
       reclassify as receive and prepend a synthetic serve.
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
        ball_positions: Ball positions (unused currently, reserved for future).
        rally_start_frame: Rally start frame for synthetic serve placement.

    Returns:
        Repaired list of ClassifiedAction (possibly longer if synthetic
        serve inserted).
    """
    if len(actions) < 2:
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
            # Skip Rule 0 when ball trajectory is one-sided — ball_y
            # is unreliable (WASB missed the serve trajectory).
            # The confidence gate is applied upstream in classify_rally
            # (suppress_phantom requires pos_conf >= 0.7); Rule 0 only
            # checks the trajectory signal since it runs post-hoc on an
            # already-classified serve.
            skip_rule0 = (
                ball_positions is not None
                and _is_ball_one_sided(
                    ball_positions, serve.frame, net_y,
                )
            )
            if skip_rule0:
                logger.debug(
                    "Rule 0 skipped: one-sided ball trajectory "
                    "at serve frame %d",
                    serve.frame,
                )
            else:
                opposite = "far" if serve.court_side == "near" else "near"
                synthetic = _make_synthetic_serve(
                    opposite, serve.frame, net_y,
                    rally_start_frame=rally_start_frame,
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
        return repaired

    # ------------------------------------------------------------------
    # Rule 3 (pre-pass): Duplicate non-synthetic serves → extras become dig.
    # Only non-synthetic serves count; the first real serve is the anchor.
    # ------------------------------------------------------------------
    first_real_serve_found = False
    for i, a in enumerate(repaired):
        if a.action_type == ActionType.SERVE and not a.is_synthetic:
            if not first_real_serve_found:
                first_real_serve_found = True
            else:
                if repair_count >= _MAX_SEQUENCE_REPAIRS:
                    break
                repaired[i] = _reclassify(a, ActionType.DIG)
                repair_count += 1
                logger.debug(
                    "Repair rule 3: duplicate serve at f%d → dig",
                    a.frame,
                )

    # ------------------------------------------------------------------
    # Rule 4 (pre-pass): Duplicate receives → extras become set.
    # Always use set (2nd touch) — court_side labels are too unreliable
    # to distinguish same-side (set) from cross-side (dig).
    # ------------------------------------------------------------------
    first_receive_found = False
    for i, a in enumerate(repaired):
        if a.action_type == ActionType.RECEIVE:
            if not first_receive_found:
                first_receive_found = True
            else:
                if repair_count >= _MAX_SEQUENCE_REPAIRS:
                    break
                repaired[i] = _reclassify(a, ActionType.SET)
                repair_count += 1
                logger.debug(
                    "Repair rule 4: duplicate receive at f%d → set",
                    a.frame,
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

        same_side = (
            prev.court_side == a.court_side
            and a.court_side in ("near", "far")
        )

        # Rule 1: Two consecutive receives or digs on same side → second is set
        if (
            same_side
            and prev.action_type in (ActionType.RECEIVE, ActionType.DIG)
            and a.action_type == prev.action_type
        ):
            repaired[i] = _reclassify(a, ActionType.SET)
            repair_count += 1
            logger.debug(
                "Repair rule 1: %s→%s at f%d→f%d, changed second to set",
                prev.action_type.value, a.action_type.value,
                prev.frame, a.frame,
            )

        # Rule 2: receive/dig directly followed by attack on same side
        # with no set → reclassify the attack as set (if there's another
        # action after)
        elif (
            same_side
            and prev.action_type in (ActionType.RECEIVE, ActionType.DIG)
            and a.action_type == ActionType.ATTACK
        ):
            next_idx: int | None = None
            for k in range(i + 1, len(repaired)):
                if repaired[k].action_type not in (
                    ActionType.BLOCK, ActionType.UNKNOWN,
                ):
                    next_idx = k
                    break
            if next_idx is not None:
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
        ):
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
        ):
            repaired[prev_idx] = _reclassify(prev, ActionType.SET)
            repair_count += 1
            logger.debug(
                "Repair rule 6: attack→attack at f%d→f%d, first → set",
                prev.frame, a.frame,
            )

        i += 1

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

    Synthetic serves (inserted by repair_action_sequence) reset the chain
    to serve_team — this is correct for rally-start synthetics but may
    drift if repair injects mid-rally synthetics after misclassification.

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
            # Synthetic serves are inferred — trust the team chain
            if action.action_type == ActionType.SERVE:
                expected[i] = serve_team
                current_team = serve_team
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
# Encodes which (prev_action, next_action) pairs are legal and their
# relative probabilities. Transitions not listed have probability 0.
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

# Only relabel actions below this confidence — high-confidence GBM
# predictions are usually correct, Viterbi helps ambiguous cases.
_VITERBI_RELABEL_CONFIDENCE_CAP = 0.65


def viterbi_decode_actions(
    actions: list[ClassifiedAction],
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
            if result[idx].confidence > _VITERBI_RELABEL_CONFIDENCE_CAP:
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
) -> RallyActions:
    """Convenience function to classify actions in a rally.

    When use_classifier=True and a trained action type model exists on disk,
    uses the learned classifier for dig/set/attack classification. Otherwise
    falls back to the rule-based state machine.

    Pipeline:
    1. classify_rally() — initial action types + serve detection
       (uses match_team_assignments for team-aware touch counting when available)
    2. propagate_court_side() — infer court_side from action-type transitions
    3. repair_action_sequence() — fix illegal patterns using court_side
    4. viterbi_decode_actions() — sequence-level smoothing
    5. validate_action_sequence() — log constraint violations
    6. assign_court_side_from_teams() — overwrite court_side from match teams
       (more accurate than propagation for output + downstream consumers)
    7. reattribute_players() — server exclusion + server-seeded team chain
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

    Returns:
        RallyActions with all classified actions.
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
    )

    # Propagation infers court_side from action-type transitions (serve/attack
    # cross net, receive/set/dig stay same side). This drives repair_action_sequence
    # which fixes illegal patterns.
    result.actions = propagate_court_side(result.actions)

    result.actions = repair_action_sequence(
        result.actions,
        net_y=contact_sequence.net_y,
        ball_positions=contact_sequence.ball_positions,
        rally_start_frame=contact_sequence.rally_start_frame,
    )
    result.actions = viterbi_decode_actions(result.actions)
    result.actions = validate_action_sequence(result.actions, rally_id)

    # Snapshot propagated court_side before team-based overwrite.
    # Used by post-hoc team correction to bypass _compute_expected_teams() cascade.
    propagated_sides = [a.court_side for a in result.actions]

    # Overwrite court_side from team membership (73% accurate vs 61%
    # from propagation). This improves court_side labels for output but
    # doesn't affect reattribution — Pass 2 uses server-seeded expected
    # teams from _compute_expected_teams() instead of court_side.
    if match_team_assignments:
        assign_court_side_from_teams(result.actions, match_team_assignments)

    pre_reattrib_tids = [a.player_track_id for a in result.actions]

    result.actions = reattribute_players(
        result.actions, contact_sequence.contacts, reattrib_teams,
        max_distance_ratio=1.5,
        reid_predictions=reid_predictions,
    )

    # Post-hoc team correction: use propagated court_side (bidirectional,
    # confidence-gated) to fix cascade errors from _compute_expected_teams().
    # Only targets actions where Pass 2 changed the player.
    if match_team_assignments:
        result.actions = correct_team_from_propagation(
            result.actions, contact_sequence.contacts,
            propagated_sides, match_team_assignments,
            pre_reattrib_tids=pre_reattrib_tids,
        )
        # Re-apply court_side from (now-corrected) player's team
        assign_court_side_from_teams(result.actions, match_team_assignments)

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

    return result
