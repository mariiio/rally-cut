"""Action classification for beach volleyball.

Classifies detected ball contacts into volleyball actions
(serve/receive/set/attack/block/dig) using either:

1. **Learned classifier** (default): A trained GBM model predicts dig/set/attack
   from trajectory features + sequence context. Serve, receive, and block stay
   heuristic. Achieves ~87% action accuracy (up from ~48% rule-based).
   Auto-loaded from weights/action_classifier/action_classifier.pkl.

2. **Rule-based state machine** (fallback): Uses contact count per side
   (dig=1st, set=2nd, attack=3rd) with net-crossing detection for side changes.
   No labeled data required.

Beach volleyball rules that constrain both modes:
- 2v2, max 3 contacts per side
- Strict sequence: serve → receive → set → attack
- Blocks don't count as a contact (max 3 contacts after block)
- Each rally starts with a serve from behind the baseline
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from rallycut.tracking.contact_detector import Contact, ContactSequence

if TYPE_CHECKING:
    from rallycut.tracking.action_type_classifier import ActionTypeClassifier
    from rallycut.tracking.ball_tracker import BallPosition

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
        """Total contacts (excluding blocks)."""
        return sum(1 for a in self.actions if a.action_type != ActionType.BLOCK)

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


def _ball_crossed_net(
    ball_positions: list[BallPosition],
    from_frame: int,
    to_frame: int,
    net_y: float,
    min_frames_per_side: int = 2,
) -> bool | None:
    """Check if ball Y crossed net_y between two frames.

    Detects any transient crossing: if the ball starts on one side and
    at any point spends ≥min_frames_per_side consecutive frames on the
    other side, a crossing occurred. This handles cases where the ball
    crosses the net and immediately bounces back (e.g., attack → dig).

    Args:
        ball_positions: Sorted list of confident ball positions.
        from_frame: Start of frame range (exclusive).
        to_frame: End of frame range (exclusive).
        net_y: Net Y position.
        min_frames_per_side: Min frames on each side to confirm crossing.

    Returns:
        True if ball crossed net, False if confirmed no crossing,
        None if insufficient data to determine.
    """
    positions_in_range = [
        bp for bp in ball_positions
        if from_frame < bp.frame_number < to_frame
    ]
    if len(positions_in_range) < min_frames_per_side * 2:
        return None

    # Determine starting side from first min_frames consecutive frames
    start_near = 0
    start_far = 0
    for bp in positions_in_range:
        if bp.y >= net_y:
            if start_far == 0:
                start_near += 1
            else:
                break
        else:
            if start_near == 0:
                start_far += 1
            else:
                break

    if start_near < min_frames_per_side and start_far < min_frames_per_side:
        return None  # Can't determine starting side (noisy data)

    starting_is_near = start_near >= min_frames_per_side

    # Scan for any contiguous block on the opposite side
    consecutive_opposite = 0
    for bp in positions_in_range:
        is_near = bp.y >= net_y
        if is_near != starting_is_near:
            consecutive_opposite += 1
            if consecutive_opposite >= min_frames_per_side:
                return True
        else:
            consecutive_opposite = 0

    return False


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
) -> ClassifiedAction:
    """Create a synthetic serve action for a missed serve.

    Places the serve ~1s before the first detected contact, at the
    appropriate baseline position.

    Args:
        serve_side: Court side of the serve ("near" or "far").
        first_contact_frame: Frame of the first detected contact.
        net_y: Net Y position.

    Returns:
        A synthetic ClassifiedAction for the serve.
    """
    baseline_near, baseline_far = _serve_baselines(net_y)

    return ClassifiedAction(
        action_type=ActionType.SERVE,
        frame=max(0, first_contact_frame - 30),
        ball_x=0.5,
        ball_y=baseline_near if serve_side == "near" else baseline_far,
        velocity=0.0,
        player_track_id=-1,
        court_side=serve_side,
        confidence=0.4,
        is_synthetic=True,
    )


class ActionClassifier:
    """Rule-based volleyball action classifier.

    Uses the strict beach volleyball contact sequence to classify actions:

    Rally flow:
    1. SERVE — first contact, from behind baseline
    2. RECEIVE — first contact on receiving side
    3. SET — second contact on same side
    4. ATTACK — third contact on same side (or ball directed to other court)
    5. After ball crosses net, count resets:
       - DIG (1st contact) → SET (2nd) → ATTACK (3rd)
    6. BLOCK — contact at net immediately after opponent's attack

    The classifier uses a state machine that tracks:
    - Which side has possession
    - Contact count on current side
    - Whether serve has occurred
    """

    def __init__(self, config: ActionClassifierConfig | None = None):
        self.config = config or ActionClassifierConfig()

    def classify_rally(
        self,
        contact_sequence: ContactSequence,
        rally_id: str = "",
        team_assignments: dict[int, int] | None = None,
    ) -> RallyActions:
        """Classify all contacts in a rally into action types.

        Args:
            contact_sequence: Detected contacts from ContactDetector.
            rally_id: Optional rally identifier.
            team_assignments: Optional mapping of track_id → team (0=near/A, 1=far/B).

        Returns:
            RallyActions with classified actions.
        """
        contacts = contact_sequence.contacts
        start_frame = contact_sequence.rally_start_frame

        if not contacts:
            return RallyActions(rally_id=rally_id)

        actions: list[ClassifiedAction] = []
        serve_detected = False
        serve_side: str | None = None  # Court side of the serve
        receive_detected = False  # Whether receive has been classified
        current_side: str | None = None  # Side with possession
        contact_count_on_side = 0
        last_action_type: ActionType | None = None

        ball_positions = contact_sequence.ball_positions or None

        # Determine which contact is the serve (may be outside window if
        # ball tracking starts late — common with VballNet warmup).
        serve_index, serve_pass = self._find_serve_index(
            contacts, start_frame, contact_sequence.net_y,
            ball_positions=ball_positions,
        )

        for i, contact in enumerate(contacts):
            action_type = ActionType.UNKNOWN
            confidence = self.config.low_confidence

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
                # Block doesn't count as a contact for the 3-touch limit
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

            # Handle possession changes: check ball trajectory crossing.
            # _ball_crossed_net returns True/False/None (tri-state):
            #   True  = confirmed crossing → reset counter
            #   False = confirmed no crossing → trust trajectory over court_side
            #   None  = insufficient data → fall back to court_side comparison
            crossed_net: bool | None = None
            if ball_positions and i > 0 and current_side is not None:
                crossed_net = _ball_crossed_net(
                    ball_positions,
                    from_frame=contacts[i - 1].frame,
                    to_frame=contact.frame,
                    net_y=contact_sequence.net_y,
                )
            if crossed_net is True:
                current_side = contact.court_side
                contact_count_on_side = 0
            elif contact.court_side != current_side:
                if crossed_net is False:
                    # Confirmed no crossing — trust trajectory, keep counter.
                    # Safety valve: beach volleyball max 3 touches + block.
                    if contact_count_on_side >= 4:
                        current_side = contact.court_side
                        contact_count_on_side = 0
                else:
                    # crossed_net is None (insufficient data or no ball_positions)
                    # Fall back to court_side comparison
                    current_side = contact.court_side
                    contact_count_on_side = 0

            contact_count_on_side += 1

            # Unconditional safety valve: beach volleyball allows max 3 touches
            # per side. If counter exceeds this, a net crossing was missed
            # (e.g., ball trajectory stays visually below net_y due to camera
            # angle). Reset to 1 to resume the dig→set→attack cycle.
            if contact_count_on_side > 3 and receive_detected:
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
                    if serve_pass == 3 and ball_positions:
                        toward_net = _ball_moving_toward_net(
                            ball_positions, contact.frame,
                            contact.ball_y, contact_sequence.net_y,
                        )
                        # Only trigger phantom on confirmed False.
                        # None (insufficient data) → keep as serve (conservative).
                        if toward_net is False:
                            is_phantom = True

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
                        serve_side = contact.court_side
                        current_side = contact.court_side
                        contact_count_on_side = 1
                    else:
                        # Phantom serve: real serve was missed, this is the
                        # receive. Infer serve came from the opposite side.
                        serve_side = _infer_serve_side(
                            contact, ball_positions,
                            contact_sequence.net_y,
                        ) or (
                            "far" if contact.court_side == "near"
                            else "near"
                        )
                        # Prepend synthetic serve action
                        actions.append(_make_synthetic_serve(
                            serve_side, contact.frame,
                            contact_sequence.net_y,
                        ))
                        action_type = ActionType.RECEIVE
                        confidence = self.config.medium_confidence
                        serve_detected = True
                        receive_detected = True
                        current_side = contact.court_side
                        contact_count_on_side = 1
                else:
                    action_type = ActionType.UNKNOWN
                    confidence = self.config.low_confidence

            elif (
                not receive_detected
                and serve_side is not None
                and (contact.court_side != serve_side or crossed_net is True)
            ):
                # First contact on the opposite side from serve = receive.
                # Also triggers if ball crossed net (even if court_side
                # matches serve_side due to ball_y near net threshold).
                # Robust to FP contacts between serve and receive.
                action_type = ActionType.RECEIVE
                confidence = self.config.high_confidence
                receive_detected = True
                current_side = contact.court_side
                contact_count_on_side = 1

            elif contact_count_on_side == 1:
                # First contact on this side (after initial receive)
                action_type = ActionType.DIG
                confidence = self.config.medium_confidence

            elif contact_count_on_side == 2:
                action_type = ActionType.SET
                confidence = self.config.high_confidence

            elif contact_count_on_side >= 3:
                action_type = ActionType.ATTACK
                confidence = self.config.high_confidence

            # Modulate action confidence with contact classifier confidence.
            # If the contact was scored by a classifier (confidence > 0), blend
            # it with the rule-based confidence so low-confidence contacts get
            # lower action confidence.
            if contact.confidence > 0:
                confidence = min(confidence, contact.confidence)

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
            logger.info(f"Rally {rally_id}: classified {len(actions)} actions: {seq}")

        return result

    def classify_rally_learned(
        self,
        contact_sequence: ContactSequence,
        classifier: ActionTypeClassifier,
        rally_id: str = "",
        team_assignments: dict[int, int] | None = None,
    ) -> RallyActions:
        """Classify contacts using the learned action type classifier.

        Serve detection uses existing 3-pass heuristic. Block detection
        stays rule-based. All other contacts are classified by the model.

        Args:
            contact_sequence: Detected contacts from ContactDetector.
            classifier: Trained ActionTypeClassifier.
            rally_id: Optional rally identifier.
            team_assignments: Optional mapping of track_id → team (0=near/A, 1=far/B).

        Returns:
            RallyActions with classified actions.
        """
        from rallycut.tracking.action_type_classifier import extract_action_features

        contacts = contact_sequence.contacts
        start_frame = contact_sequence.rally_start_frame

        if not contacts:
            return RallyActions(rally_id=rally_id)

        ball_positions = contact_sequence.ball_positions or None
        actions: list[ClassifiedAction] = []

        # Determine serve index using existing heuristic
        serve_index, serve_pass = self._find_serve_index(
            contacts, start_frame, contact_sequence.net_y,
            ball_positions=ball_positions,
        )

        serve_detected = False
        serve_side: str | None = None
        receive_detected = False
        last_action_type: ActionType | None = None

        for i, contact in enumerate(contacts):
            # Block detection (rule-based): at net + immediately after attack
            if (
                contact.is_at_net
                and last_action_type == ActionType.ATTACK
                and i > 0
                and (contact.frame - contacts[i - 1].frame)
                    <= self.config.block_max_frame_gap
                and contact.court_side != contacts[i - 1].court_side
            ):
                actions.append(ClassifiedAction(
                    action_type=ActionType.BLOCK,
                    frame=contact.frame,
                    ball_x=contact.ball_x,
                    ball_y=contact.ball_y,
                    velocity=contact.velocity,
                    player_track_id=contact.player_track_id,
                    court_side=contact.court_side,
                    confidence=self.config.high_confidence,
                ))
                last_action_type = ActionType.BLOCK
                continue

            # Pre-serve contacts → UNKNOWN
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

            # Serve detection (heuristic)
            if not serve_detected and i == serve_index:
                # Phantom serve check (same as rule-based)
                is_phantom = False
                if serve_pass == 3 and ball_positions:
                    toward_net = _ball_moving_toward_net(
                        ball_positions, contact.frame,
                        contact.ball_y, contact_sequence.net_y,
                    )
                    # Only trigger phantom on confirmed False.
                    # None (insufficient data) → keep as serve (conservative).
                    if toward_net is False:
                        is_phantom = True

                if not is_phantom:
                    actions.append(ClassifiedAction(
                        action_type=ActionType.SERVE,
                        frame=contact.frame,
                        ball_x=contact.ball_x,
                        ball_y=contact.ball_y,
                        velocity=contact.velocity,
                        player_track_id=contact.player_track_id,
                        court_side=contact.court_side,
                        confidence=self.config.high_confidence,
                    ))
                    serve_detected = True
                    serve_side = contact.court_side
                    last_action_type = ActionType.SERVE
                    continue
                else:
                    # Phantom serve → inject synthetic serve + classify
                    # this contact as receive
                    serve_side = _infer_serve_side(
                        contact, ball_positions,
                        contact_sequence.net_y,
                    ) or (
                        "far" if contact.court_side == "near"
                        else "near"
                    )
                    actions.append(_make_synthetic_serve(
                        serve_side, contact.frame,
                        contact_sequence.net_y,
                    ))
                    actions.append(ClassifiedAction(
                        action_type=ActionType.RECEIVE,
                        frame=contact.frame,
                        ball_x=contact.ball_x,
                        ball_y=contact.ball_y,
                        velocity=contact.velocity,
                        player_track_id=contact.player_track_id,
                        court_side=contact.court_side,
                        confidence=self.config.medium_confidence,
                    ))
                    serve_detected = True
                    receive_detected = True
                    last_action_type = ActionType.RECEIVE
                    continue

            # Receive: first contact on opposite side from serve.
            # Uses receive_detected flag (not last_action_type) so FP contacts
            # between serve and receive don't suppress heuristic receive detection.
            crossed_net: bool | None = None
            if not receive_detected and serve_side is not None and i > 0:
                if ball_positions:
                    crossed_net = _ball_crossed_net(
                        ball_positions,
                        contacts[i - 1].frame, contact.frame,
                        contact_sequence.net_y,
                    )
                if contact.court_side != serve_side or crossed_net is True:
                    actions.append(ClassifiedAction(
                        action_type=ActionType.RECEIVE,
                        frame=contact.frame,
                        ball_x=contact.ball_x,
                        ball_y=contact.ball_y,
                        velocity=contact.velocity,
                        player_track_id=contact.player_track_id,
                        court_side=contact.court_side,
                        confidence=self.config.high_confidence,
                    ))
                    receive_detected = True
                    last_action_type = ActionType.RECEIVE
                    continue

            # All other contacts: use learned classifier
            feat = extract_action_features(
                contact=contact,
                index=i,
                all_contacts=contacts,
                ball_positions=ball_positions,
                net_y=contact_sequence.net_y,
                rally_start_frame=start_frame,
            )

            predictions = classifier.predict([feat])
            pred_action, pred_conf = predictions[0]

            try:
                action_type = ActionType(pred_action)
            except ValueError:
                action_type = ActionType.UNKNOWN

            # Modulate with contact confidence
            confidence = pred_conf
            if contact.confidence > 0:
                confidence = min(confidence, contact.confidence)

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

        # Stamp team labels on all actions
        if team_assignments:
            for action in actions:
                if action.player_track_id >= 0:
                    action.team = _team_label(action.player_track_id, team_assignments)
                elif action.court_side in ("near", "far"):
                    action.team = "A" if action.court_side == "near" else "B"

        result = RallyActions(
            actions=actions, rally_id=rally_id,
            team_assignments=team_assignments or {},
        )

        if actions:
            seq = [a.action_type.value for a in actions]
            logger.info(
                f"Rally {rally_id}: classified {len(actions)} actions "
                f"(learned): {seq}"
            )

        return result

    def _find_serve_index(
        self,
        contacts: list[Contact],
        start_frame: int,
        net_y: float = 0.5,
        ball_positions: list[BallPosition] | None = None,
    ) -> tuple[int, int]:
        """Find which contact is the serve.

        Uses three passes with decreasing strictness:
        1. Arc-based: first contact in window whose subsequent trajectory crosses
           the net (distinctive serve arc pattern).
        2. Position/velocity: baseline position or high velocity (original heuristic).
        3. Fallback: first contact is the serve.

        Returns:
            Tuple of (index into contacts list, pass number that found it).
            Pass number: 1=arc, 2=position/velocity, 3=fallback, 0=not found.
            Index is -1 if no serve found.
        """
        window = self.config.serve_window_frames

        baseline_near, baseline_far = _serve_baselines(net_y)

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
                if _ball_crossed_net(ball_positions, c.frame, next_frame, net_y) is True:
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
                return i, 2
            if c.velocity >= self.config.serve_min_velocity:
                # With ball trajectory, also require ball moving toward net
                if ball_positions:
                    if _ball_moving_toward_net(
                        ball_positions, c.frame, c.ball_y, net_y,
                    ):
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


def classify_rally_actions(
    contact_sequence: ContactSequence,
    rally_id: str = "",
    config: ActionClassifierConfig | None = None,
    use_classifier: bool = True,
    team_assignments: dict[int, int] | None = None,
) -> RallyActions:
    """Convenience function to classify actions in a rally.

    When use_classifier=True and a trained action type model exists on disk,
    uses the learned classifier for dig/set/attack classification. Otherwise
    falls back to the rule-based state machine.

    Args:
        contact_sequence: Contacts detected by ContactDetector.
        rally_id: Optional rally identifier.
        config: Optional classifier configuration.
        use_classifier: Whether to auto-load and use the learned classifier.
        team_assignments: Optional mapping of track_id → team (0=near/A, 1=far/B).

    Returns:
        RallyActions with all classified actions.
    """
    action_classifier = ActionClassifier(config)

    if use_classifier:
        learned = _get_default_action_classifier()
        if learned is not None and learned.is_trained:
            return action_classifier.classify_rally_learned(
                contact_sequence, learned, rally_id,
                team_assignments=team_assignments,
            )

    return action_classifier.classify_rally(
        contact_sequence, rally_id,
        team_assignments=team_assignments,
    )
