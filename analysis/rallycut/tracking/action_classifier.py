"""Rule-based action classification for beach volleyball.

Classifies detected ball contacts into volleyball actions
(serve/receive/set/spike/block) using the contact sequence, court side,
ball direction, and beach volleyball rules.

Beach volleyball rules that constrain classification:
- 2v2, max 3 contacts per side
- Strict sequence: serve → receive → set → attack
- Blocks don't count as a contact (max 3 contacts after block)
- Each rally starts with a serve from behind the baseline

No labeled action data is required — classification is purely rule-based.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from rallycut.tracking.contact_detector import Contact, ContactSequence

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Volleyball action types."""

    SERVE = "serve"
    RECEIVE = "receive"
    SET = "set"
    SPIKE = "spike"
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action_type.value,
            "frame": self.frame,
            "ballX": self.ball_x,
            "ballY": self.ball_y,
            "velocity": self.velocity,
            "playerTrackId": self.player_track_id,
            "courtSide": self.court_side,
            "confidence": self.confidence,
        }


@dataclass
class RallyActions:
    """All classified actions within a single rally."""

    actions: list[ClassifiedAction] = field(default_factory=list)
    rally_id: str = ""

    @property
    def serve(self) -> ClassifiedAction | None:
        """Get the serve action."""
        for a in self.actions:
            if a.action_type == ActionType.SERVE:
                return a
        return None

    @property
    def action_sequence(self) -> list[ActionType]:
        """Get ordered action type sequence."""
        return [a.action_type for a in self.actions]

    def actions_by_player(self, track_id: int) -> list[ClassifiedAction]:
        """Get all actions for a specific player."""
        return [a for a in self.actions if a.player_track_id == track_id]

    def actions_by_type(self, action_type: ActionType) -> list[ClassifiedAction]:
        """Get all actions of a specific type."""
        return [a for a in self.actions if a.action_type == action_type]

    @property
    def num_contacts(self) -> int:
        """Total contacts (excluding blocks)."""
        return sum(1 for a in self.actions if a.action_type != ActionType.BLOCK)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rallyId": self.rally_id,
            "numContacts": self.num_contacts,
            "actionSequence": [a.value for a in self.action_sequence],
            "actions": [a.to_dict() for a in self.actions],
        }


@dataclass
class ActionClassifierConfig:
    """Configuration for rule-based action classification."""

    # Serve detection
    serve_window_frames: int = 60  # Serve must occur in first N frames (~2s @ 30fps)
    serve_min_velocity: float = 0.015  # Min velocity for serve
    serve_fallback: bool = True  # Treat first contact as serve if none found in window

    # Block detection
    block_max_frame_gap: int = 8  # Max frames between attack and block
    block_net_zone: float = 0.08  # ±8% of screen around net

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
) -> bool:
    """Check if ball Y crossed net_y between two frames.

    Requires the ball to be on one side for min_frames_per_side frames,
    then the other side for min_frames_per_side frames, to avoid noise.

    Args:
        ball_positions: Sorted list of confident ball positions.
        from_frame: Start of frame range (exclusive).
        to_frame: End of frame range (exclusive).
        net_y: Net Y position.
        min_frames_per_side: Min frames on each side to confirm crossing.

    Returns:
        True if ball crossed net between from_frame and to_frame.
    """
    positions_in_range = [
        bp for bp in ball_positions
        if from_frame < bp.frame_number < to_frame
    ]
    if len(positions_in_range) < min_frames_per_side * 2:
        return False

    # Count consecutive frames on each side from start and end
    near_from_start = 0
    far_from_start = 0
    for bp in positions_in_range:
        if bp.y >= net_y:
            if far_from_start == 0:
                near_from_start += 1
            else:
                break
        else:
            if near_from_start == 0:
                far_from_start += 1
            else:
                break

    near_from_end = 0
    far_from_end = 0
    for bp in reversed(positions_in_range):
        if bp.y >= net_y:
            if far_from_end == 0:
                near_from_end += 1
            else:
                break
        else:
            if near_from_end == 0:
                far_from_end += 1
            else:
                break

    # Crossing: started on one side, ended on the other
    started_near = near_from_start >= min_frames_per_side
    started_far = far_from_start >= min_frames_per_side
    ended_near = near_from_end >= min_frames_per_side
    ended_far = far_from_end >= min_frames_per_side

    return (started_near and ended_far) or (started_far and ended_near)


class ActionClassifier:
    """Rule-based volleyball action classifier.

    Uses the strict beach volleyball contact sequence to classify actions:

    Rally flow:
    1. SERVE — first contact, from behind baseline
    2. RECEIVE — first contact on receiving side
    3. SET — second contact on same side
    4. SPIKE — third contact on same side (or ball directed to other court)
    5. After ball crosses net, count resets:
       - DIG (1st contact) → SET (2nd) → SPIKE (3rd)
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
    ) -> RallyActions:
        """Classify all contacts in a rally into action types.

        Args:
            contact_sequence: Detected contacts from ContactDetector.
            rally_id: Optional rally identifier.

        Returns:
            RallyActions with classified actions.
        """
        contacts = contact_sequence.contacts
        start_frame = contact_sequence.rally_start_frame

        if not contacts:
            return RallyActions(rally_id=rally_id)

        # Determine which contact is the serve (may be outside window if
        # ball tracking starts late — common with VballNet warmup).
        serve_index = self._find_serve_index(
            contacts, start_frame, contact_sequence.net_y,
        )

        actions: list[ClassifiedAction] = []
        serve_detected = False
        serve_side: str | None = None  # Court side of the serve
        receive_detected = False  # Whether receive has been classified
        current_side: str | None = None  # Side with possession
        contact_count_on_side = 0
        last_action_type: ActionType | None = None

        for i, contact in enumerate(contacts):
            action_type = ActionType.UNKNOWN
            confidence = self.config.low_confidence

            # Check for block (must be at net, immediately after opponent's attack)
            if (
                contact.is_at_net
                and last_action_type == ActionType.SPIKE
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

            # Handle possession changes: check ball trajectory crossing,
            # then fall back to per-contact court_side comparison.
            crossed_net = False
            if (
                contact_sequence.ball_positions
                and i > 0
                and current_side is not None
            ):
                crossed_net = _ball_crossed_net(
                    contact_sequence.ball_positions,
                    from_frame=contacts[i - 1].frame,
                    to_frame=contact.frame,
                    net_y=contact_sequence.net_y,
                )
            if crossed_net or contact.court_side != current_side:
                current_side = contact.court_side
                contact_count_on_side = 0

            contact_count_on_side += 1

            # Rule-based classification
            if not serve_detected:
                if i == serve_index:
                    is_in_window = (
                        (contact.frame - start_frame) < self.config.serve_window_frames
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
                    action_type = ActionType.UNKNOWN
                    confidence = self.config.low_confidence

            elif (
                not receive_detected
                and serve_side is not None
                and contact.court_side != serve_side
            ):
                # First contact on the opposite side from serve = receive.
                # Robust to FP contacts between serve and receive.
                action_type = ActionType.RECEIVE
                confidence = self.config.high_confidence
                receive_detected = True
                contact_count_on_side = 1

            elif contact_count_on_side == 1:
                # First contact on this side (after initial receive)
                action_type = ActionType.DIG
                confidence = self.config.medium_confidence

            elif contact_count_on_side == 2:
                action_type = ActionType.SET
                confidence = self.config.high_confidence

            elif contact_count_on_side >= 3:
                action_type = ActionType.SPIKE
                confidence = self.config.high_confidence

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

        result = RallyActions(actions=actions, rally_id=rally_id)

        if actions:
            seq = [a.action_type.value for a in actions]
            logger.info(f"Rally {rally_id}: classified {len(actions)} actions: {seq}")

        return result

    def _find_serve_index(
        self,
        contacts: list[Contact],
        start_frame: int,
        net_y: float = 0.5,
    ) -> int:
        """Find which contact is the serve.

        First tries to find a serve within the serve window (baseline position
        or high velocity). If none found and serve_fallback is enabled, treats
        the first contact as the serve — ball tracking often starts late due to
        VballNet warmup, so the first detected contact is typically the serve
        or very close to it.

        Returns:
            Index into contacts list, or -1 if no serve found.
        """
        window = self.config.serve_window_frames

        # Dynamic baselines: scale proportionally to net_y so they adapt
        # to camera angles. Coefficients calibrated to match 0.82/0.18 at net_y=0.5.
        baseline_near = net_y + (1.0 - net_y) * 0.64
        baseline_far = net_y * 0.36

        # Pass 1: strict serve detection within window
        for i, c in enumerate(contacts):
            if (c.frame - start_frame) >= window:
                break
            is_at_baseline = c.ball_y >= baseline_near or c.ball_y <= baseline_far
            if is_at_baseline or c.velocity >= self.config.serve_min_velocity:
                return i

        # Pass 2: fallback — first contact is the serve
        if self.config.serve_fallback and contacts:
            logger.info(
                "No serve in %d-frame window, using first contact (frame %d) "
                "as serve fallback",
                window, contacts[0].frame,
            )
            return 0

        return -1


def classify_rally_actions(
    contact_sequence: ContactSequence,
    rally_id: str = "",
    config: ActionClassifierConfig | None = None,
) -> RallyActions:
    """Convenience function to classify actions in a rally.

    Args:
        contact_sequence: Contacts detected by ContactDetector.
        rally_id: Optional rally identifier.
        config: Optional classifier configuration.

    Returns:
        RallyActions with all classified actions.
    """
    classifier = ActionClassifier(config)
    return classifier.classify_rally(contact_sequence, rally_id)
