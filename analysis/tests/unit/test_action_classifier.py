"""Unit tests for rule-based action classification."""

from __future__ import annotations

from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    ClassifiedAction,
    RallyActions,
    _ball_crossed_net,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition
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


def _bp(frame: int, y: float) -> BallPosition:
    """Helper to create a BallPosition with given frame and Y."""
    return BallPosition(frame_number=frame, x=0.5, y=y, confidence=0.9)


class TestBallCrossedNet:
    """Tests for net-crossing detection between contacts."""

    def test_clear_crossing_near_to_far(self) -> None:
        """Ball crossing from near (high Y) to far (low Y) is detected."""
        positions = [
            _bp(10, 0.7), _bp(11, 0.65), _bp(12, 0.6),  # Near side
            _bp(13, 0.45), _bp(14, 0.4), _bp(15, 0.35),  # Far side
        ]
        assert _ball_crossed_net(positions, from_frame=9, to_frame=16, net_y=0.5)

    def test_clear_crossing_far_to_near(self) -> None:
        """Ball crossing from far (low Y) to near (high Y) is detected."""
        positions = [
            _bp(10, 0.3), _bp(11, 0.35), _bp(12, 0.4),   # Far side
            _bp(13, 0.55), _bp(14, 0.6), _bp(15, 0.65),   # Near side
        ]
        assert _ball_crossed_net(positions, from_frame=9, to_frame=16, net_y=0.5)

    def test_no_crossing_same_side(self) -> None:
        """Ball staying on one side is not a crossing."""
        positions = [
            _bp(10, 0.7), _bp(11, 0.72), _bp(12, 0.68),
            _bp(13, 0.71), _bp(14, 0.69),
        ]
        assert not _ball_crossed_net(positions, from_frame=9, to_frame=15, net_y=0.5)

    def test_too_few_positions(self) -> None:
        """Returns False with too few positions between contacts."""
        positions = [_bp(10, 0.7), _bp(11, 0.3)]
        assert not _ball_crossed_net(positions, from_frame=9, to_frame=12, net_y=0.5)

    def test_noisy_single_frame_not_crossing(self) -> None:
        """Single-frame noise crossing net is not enough (needs min_frames_per_side)."""
        positions = [
            _bp(10, 0.6), _bp(11, 0.55),  # Near
            _bp(12, 0.45),                  # Single far frame
            _bp(13, 0.55), _bp(14, 0.6),   # Back to near
        ]
        assert not _ball_crossed_net(positions, from_frame=9, to_frame=15, net_y=0.5)


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


class TestNetCrossingPossession:
    """Tests for possession changes using ball trajectory crossing."""

    def test_net_crossing_resets_count(self) -> None:
        """Ball trajectory crossing net resets contact count on new side."""
        # Contacts on far side, then ball crosses net, then near side
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Spike
            _contact(frame=70, ball_y=0.7, court_side="near"),   # Dig
        ]
        # Ball positions showing crossing between frame 55 and 70
        ball_positions = [
            _bp(56, 0.38), _bp(57, 0.40), _bp(58, 0.42),  # Far side
            _bp(59, 0.45), _bp(60, 0.48),
            _bp(61, 0.52), _bp(62, 0.55), _bp(63, 0.58),  # Near side
            _bp(64, 0.60), _bp(65, 0.63),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq)
        assert result.actions[4].action_type == ActionType.DIG  # 1st on near side

    def test_no_crossing_keeps_count(self) -> None:
        """Without net crossing, same-side contacts continue counting."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.35, court_side="far"),   # Set
            # FP contact on far side (no net crossing between)
            _contact(frame=55, ball_y=0.38, court_side="far"),   # Spike (3rd)
            _contact(frame=65, ball_y=0.32, court_side="far"),   # Spike (4th, >= 3)
        ]
        # Ball staying on far side between contacts (no crossing)
        ball_positions = [
            _bp(31, 0.32), _bp(32, 0.34), _bp(33, 0.30),
            _bp(46, 0.33), _bp(47, 0.35), _bp(48, 0.37),
            _bp(56, 0.36), _bp(57, 0.34), _bp(58, 0.32),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq)
        assert result.actions[3].action_type == ActionType.SPIKE
        assert result.actions[4].action_type == ActionType.SPIKE


class TestContactCap:
    """Tests for 3-contact-per-side cap."""

    def test_fourth_contact_is_spike(self) -> None:
        """Fourth contact on same side without crossing is still SPIKE (>= 3)."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Spike (3rd)
            _contact(frame=65, ball_y=0.32, court_side="far"),   # Spike (4th, >= 3)
        ]
        # No ball_positions = fallback to court_side comparison
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert result.actions[3].action_type == ActionType.SPIKE
        assert result.actions[4].action_type == ActionType.SPIKE

    def test_three_contacts_still_works(self) -> None:
        """Normal 3-contact sequence (receive/set/spike) still classified correctly."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),
            _contact(frame=30, ball_y=0.3, court_side="far"),
            _contact(frame=45, ball_y=0.25, court_side="far"),
            _contact(frame=55, ball_y=0.35, court_side="far"),
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert result.actions[1].action_type == ActionType.RECEIVE
        assert result.actions[2].action_type == ActionType.SET
        assert result.actions[3].action_type == ActionType.SPIKE


class TestDynamicServeBaseline:
    """Tests for dynamic serve baseline thresholds."""

    def test_baseline_adapts_to_net_y(self) -> None:
        """Ball near dynamic baseline detected as serve for non-standard net_y."""
        # With net_y=0.4: baseline_near = 0.4 + 0.6*0.64 = 0.784
        #                  baseline_far  = 0.4 * 0.36 = 0.144
        contacts = [_contact(frame=10, ball_y=0.80, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.4, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert result.serve is not None

    def test_far_baseline_adapts(self) -> None:
        """Far baseline adapts to net_y for far-side serves."""
        # With net_y=0.6: baseline_far = 0.6 * 0.36 = 0.216
        contacts = [_contact(frame=10, ball_y=0.20, court_side="far")]
        seq = ContactSequence(contacts=contacts, net_y=0.6, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert result.serve is not None

    def test_velocity_still_triggers_serve(self) -> None:
        """High-velocity contact in serve window is still detected as serve."""
        contacts = [_contact(frame=10, ball_y=0.55, velocity=0.025, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq)
        assert result.serve is not None
