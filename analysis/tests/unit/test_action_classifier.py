"""Unit tests for rule-based action classification."""

from __future__ import annotations

from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    ClassifiedAction,
    RallyActions,
    classify_rally_actions,
)
from rallycut.tracking.contact_detector import Contact, ContactSequence


def _contact(
    frame: int,
    ball_y: float = 0.5,
    velocity: float = 0.02,
    direction_change: float = 60.0,
    court_side: str = "near",
    is_at_net: bool = False,
    player_track_id: int = -1,
) -> Contact:
    """Helper to create a Contact."""
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=ball_y,
        velocity=velocity,
        direction_change_deg=direction_change,
        player_track_id=player_track_id,
        court_side=court_side,
        is_at_net=is_at_net,
        is_validated=True,
    )


class TestServeDetection:
    """Tests for serve classification."""

    def test_first_contact_at_baseline_is_serve(self) -> None:
        """First contact near baseline in serve window is classified as serve."""
        contacts = [_contact(frame=10, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert len(result.actions) == 1
        assert result.actions[0].action_type == ActionType.SERVE

    def test_first_contact_high_velocity_is_serve(self) -> None:
        """First contact with high velocity in serve window is serve."""
        contacts = [_contact(frame=10, ball_y=0.6, velocity=0.025, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert result.serve is not None
        assert result.serve.action_type == ActionType.SERVE

    def test_late_contact_not_serve_without_fallback(self) -> None:
        """Contact outside serve window is not classified as serve when fallback disabled."""
        config = ActionClassifierConfig(serve_fallback=False)
        contacts = [_contact(frame=100, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, config=config)
        # Outside 60-frame window with no fallback, should be UNKNOWN
        assert result.serve is None

    def test_late_contact_fallback_serve(self) -> None:
        """First contact outside window is treated as serve with fallback."""
        contacts = [_contact(frame=100, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        # Fallback enabled by default: first contact becomes serve
        assert result.serve is not None
        assert result.serve.action_type == ActionType.SERVE
        assert result.serve.confidence == 0.7  # Medium confidence for fallback


class TestContactSequenceClassification:
    """Tests for full contact sequence classification."""

    def test_serve_receive_set_spike_sequence(self) -> None:
        """Standard serve → receive → set → spike sequence."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive (opposite side)
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set (same side)
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Spike (3rd contact)
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)

        assert len(result.actions) == 4
        assert result.action_sequence == [
            ActionType.SERVE,
            ActionType.RECEIVE,
            ActionType.SET,
            ActionType.SPIKE,
        ]

    def test_serve_receive_dig_set_spike(self) -> None:
        """After an attack, the other side digs → sets → spikes."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Spike
            _contact(frame=70, ball_y=0.7, court_side="near"),   # Dig (back to near)
            _contact(frame=80, ball_y=0.75, court_side="near"),  # Set
            _contact(frame=90, ball_y=0.65, court_side="near"),  # Spike
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)

        assert len(result.actions) == 7
        assert result.actions[4].action_type == ActionType.DIG
        assert result.actions[5].action_type == ActionType.SET
        assert result.actions[6].action_type == ActionType.SPIKE

    def test_block_detection(self) -> None:
        """Block: contact at net immediately after opponent's spike."""
        config = ActionClassifierConfig(block_max_frame_gap=10)
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),     # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),      # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),     # Set
            _contact(frame=55, ball_y=0.4, court_side="far"),      # Spike
            _contact(frame=60, ball_y=0.48, court_side="near",     # Block
                     is_at_net=True),
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        classifier = ActionClassifier(config)
        result = classifier.classify_rally(seq)

        assert result.actions[4].action_type == ActionType.BLOCK

    def test_block_not_detected_if_too_far_from_net(self) -> None:
        """Block requires is_at_net to be True."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),     # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),      # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),     # Set
            _contact(frame=55, ball_y=0.4, court_side="far"),      # Spike
            _contact(frame=60, ball_y=0.7, court_side="near",      # NOT at net
                     is_at_net=False),
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)

        # Should be DIG instead of BLOCK
        assert result.actions[4].action_type != ActionType.BLOCK


class TestPossessionTracking:
    """Tests for court side possession tracking."""

    def test_side_change_resets_contact_count(self) -> None:
        """When ball crosses to other side, contact count resets."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve (near, 1st)
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive (far, 1st)
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set (far, 2nd)
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)

        # Contact on far side after serve should reset to 1, so 2nd = set
        assert result.actions[1].action_type == ActionType.RECEIVE  # 1st on far
        assert result.actions[2].action_type == ActionType.SET  # 2nd on far


class TestEmptyInputs:
    """Tests for edge cases with empty inputs."""

    def test_empty_contacts(self) -> None:
        """Empty contact sequence returns empty actions."""
        seq = ContactSequence()
        result = classify_rally_actions(seq)
        assert len(result.actions) == 0
        assert result.serve is None
        assert result.num_contacts == 0

    def test_single_contact(self) -> None:
        """Single contact at baseline is classified as serve."""
        contacts = [_contact(frame=5, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert len(result.actions) == 1


class TestRallyActionsProperties:
    """Tests for RallyActions helper properties."""

    def test_action_sequence(self) -> None:
        """action_sequence returns ordered list of types."""
        actions = [
            ClassifiedAction(ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9),
            ClassifiedAction(ActionType.RECEIVE, 30, 0.5, 0.3, 0.03, 2, "far", 0.8),
        ]
        rally = RallyActions(actions=actions, rally_id="test")
        assert rally.action_sequence == [ActionType.SERVE, ActionType.RECEIVE]

    def test_actions_by_player(self) -> None:
        """actions_by_player filters by track ID."""
        actions = [
            ClassifiedAction(ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9),
            ClassifiedAction(ActionType.RECEIVE, 30, 0.5, 0.3, 0.03, 2, "far", 0.8),
            ClassifiedAction(ActionType.SET, 40, 0.5, 0.25, 0.02, 2, "far", 0.7),
        ]
        rally = RallyActions(actions=actions)
        assert len(rally.actions_by_player(2)) == 2
        assert len(rally.actions_by_player(1)) == 1
        assert len(rally.actions_by_player(99)) == 0

    def test_actions_by_type(self) -> None:
        """actions_by_type filters by action type."""
        actions = [
            ClassifiedAction(ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9),
            ClassifiedAction(ActionType.SPIKE, 55, 0.5, 0.4, 0.04, 2, "far", 0.8),
        ]
        rally = RallyActions(actions=actions)
        assert len(rally.actions_by_type(ActionType.SERVE)) == 1
        assert len(rally.actions_by_type(ActionType.SPIKE)) == 1
        assert len(rally.actions_by_type(ActionType.BLOCK)) == 0

    def test_num_contacts_excludes_blocks(self) -> None:
        """num_contacts counts all actions except blocks."""
        actions = [
            ClassifiedAction(ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9),
            ClassifiedAction(ActionType.RECEIVE, 30, 0.5, 0.3, 0.03, 2, "far", 0.8),
            ClassifiedAction(ActionType.BLOCK, 60, 0.5, 0.48, 0.04, 1, "near", 0.9),
        ]
        rally = RallyActions(actions=actions)
        assert rally.num_contacts == 2  # Excludes block

    def test_to_dict(self) -> None:
        """to_dict produces expected structure."""
        actions = [
            ClassifiedAction(ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9),
        ]
        rally = RallyActions(actions=actions, rally_id="test-123")
        d = rally.to_dict()
        assert d["rallyId"] == "test-123"
        assert d["numContacts"] == 1
        assert d["actionSequence"] == ["serve"]
        assert len(d["actions"]) == 1
