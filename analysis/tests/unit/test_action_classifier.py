"""Unit tests for rule-based action classification."""

from __future__ import annotations

from unittest.mock import MagicMock

from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    ClassifiedAction,
    RallyActions,
    _ball_crossed_net,
    _ball_moving_toward_net,
    _infer_serve_side,
    _make_synthetic_serve,
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
        """Returns None (insufficient data) with too few positions between contacts."""
        positions = [_bp(10, 0.7), _bp(11, 0.3)]
        assert _ball_crossed_net(positions, from_frame=9, to_frame=12, net_y=0.5) is None

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
        result = classify_rally_actions(seq, use_classifier=False)
        assert len(result.actions) == 1
        assert result.actions[0].action_type == ActionType.SERVE

    def test_first_contact_high_velocity_is_serve(self) -> None:
        """First contact with high velocity in serve window is serve."""
        contacts = [_contact(frame=10, ball_y=0.6, velocity=0.025, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.serve is not None
        assert result.serve.action_type == ActionType.SERVE

    def test_late_contact_not_serve_without_fallback(self) -> None:
        """Contact outside serve window is not classified as serve when fallback disabled."""
        config = ActionClassifierConfig(serve_fallback=False)
        contacts = [_contact(frame=100, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, config=config, use_classifier=False)
        # Outside 60-frame window with no fallback, should be UNKNOWN
        assert result.serve is None

    def test_late_contact_fallback_serve(self) -> None:
        """First contact outside window is treated as serve with fallback."""
        contacts = [_contact(frame=100, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        # Fallback enabled by default: first contact becomes serve
        assert result.serve is not None
        assert result.serve.action_type == ActionType.SERVE
        assert result.serve.confidence == 0.7  # Medium confidence for fallback


class TestContactSequenceClassification:
    """Tests for full contact sequence classification."""

    def test_serve_receive_set_attack_sequence(self) -> None:
        """Standard serve → receive → set → attack sequence."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive (opposite side)
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set (same side)
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Attack (3rd contact)
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

        assert len(result.actions) == 4
        assert result.action_sequence == [
            ActionType.SERVE,
            ActionType.RECEIVE,
            ActionType.SET,
            ActionType.ATTACK,
        ]

    def test_serve_receive_dig_set_attack(self) -> None:
        """After an attack, the other side digs → sets → attacks."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Attack
            _contact(frame=70, ball_y=0.7, court_side="near"),   # Dig (back to near)
            _contact(frame=80, ball_y=0.75, court_side="near"),  # Set
            _contact(frame=90, ball_y=0.65, court_side="near"),  # Attack
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

        assert len(result.actions) == 7
        assert result.actions[4].action_type == ActionType.DIG
        assert result.actions[5].action_type == ActionType.SET
        assert result.actions[6].action_type == ActionType.ATTACK

    def test_block_detection(self) -> None:
        """Block: contact at net immediately after opponent's attack."""
        config = ActionClassifierConfig(block_max_frame_gap=10)
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),     # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),      # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),     # Set
            _contact(frame=55, ball_y=0.4, court_side="far"),      # Attack
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
            _contact(frame=55, ball_y=0.4, court_side="far"),      # Attack
            _contact(frame=60, ball_y=0.7, court_side="near",      # NOT at net
                     is_at_net=False),
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

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
        result = classify_rally_actions(seq, use_classifier=False)

        # Contact on far side after serve should reset to 1, so 2nd = set
        assert result.actions[1].action_type == ActionType.RECEIVE  # 1st on far
        assert result.actions[2].action_type == ActionType.SET  # 2nd on far


class TestEmptyInputs:
    """Tests for edge cases with empty inputs."""

    def test_empty_contacts(self) -> None:
        """Empty contact sequence returns empty actions."""
        seq = ContactSequence()
        result = classify_rally_actions(seq, use_classifier=False)
        assert len(result.actions) == 0
        assert result.serve is None
        assert result.num_contacts == 0

    def test_single_contact(self) -> None:
        """Single contact at baseline is classified as serve."""
        contacts = [_contact(frame=5, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
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
            ClassifiedAction(ActionType.ATTACK, 55, 0.5, 0.4, 0.04, 2, "far", 0.8),
        ]
        rally = RallyActions(actions=actions)
        assert len(rally.actions_by_type(ActionType.SERVE)) == 1
        assert len(rally.actions_by_type(ActionType.ATTACK)) == 1
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
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Attack
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
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.actions[4].action_type == ActionType.DIG  # 1st on near side

    def test_no_crossing_keeps_count(self) -> None:
        """Without net crossing, same-side contacts continue counting."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.35, court_side="far"),   # Set
            # FP contact on far side (no net crossing between)
            _contact(frame=55, ball_y=0.38, court_side="far"),   # Attack (3rd)
            _contact(frame=65, ball_y=0.32, court_side="far"),   # DIG (4th, safety valve resets)
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
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.actions[3].action_type == ActionType.ATTACK
        # 4th contact on same side triggers safety valve (max 3 in beach volleyball)
        assert result.actions[4].action_type == ActionType.DIG


class TestContactCap:
    """Tests for 3-contact-per-side cap."""

    def test_fourth_contact_resets_to_dig(self) -> None:
        """Fourth contact on same side triggers safety valve (beach max 3 touches)."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Attack (3rd)
            _contact(frame=65, ball_y=0.32, court_side="far"),   # DIG (4th, reset)
        ]
        # No ball_positions = fallback to court_side comparison
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.actions[3].action_type == ActionType.ATTACK
        # Safety valve: 4th contact resets to 1 → DIG
        assert result.actions[4].action_type == ActionType.DIG

    def test_three_contacts_still_works(self) -> None:
        """Normal 3-contact sequence (receive/set/attack) still classified correctly."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),
            _contact(frame=30, ball_y=0.3, court_side="far"),
            _contact(frame=45, ball_y=0.25, court_side="far"),
            _contact(frame=55, ball_y=0.35, court_side="far"),
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.actions[1].action_type == ActionType.RECEIVE
        assert result.actions[2].action_type == ActionType.SET
        assert result.actions[3].action_type == ActionType.ATTACK


class TestDynamicServeBaseline:
    """Tests for dynamic serve baseline thresholds."""

    def test_baseline_adapts_to_net_y(self) -> None:
        """Ball near dynamic baseline detected as serve for non-standard net_y."""
        # With net_y=0.4: baseline_near = 0.4 + 0.6*0.64 = 0.784
        #                  baseline_far  = 0.4 * 0.36 = 0.144
        contacts = [_contact(frame=10, ball_y=0.80, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.4, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.serve is not None

    def test_far_baseline_adapts(self) -> None:
        """Far baseline adapts to net_y for far-side serves."""
        # With net_y=0.6: baseline_far = 0.6 * 0.36 = 0.216
        contacts = [_contact(frame=10, ball_y=0.20, court_side="far")]
        seq = ContactSequence(contacts=contacts, net_y=0.6, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.serve is not None

    def test_velocity_still_triggers_serve(self) -> None:
        """High-velocity contact with toward-net trajectory is detected as serve."""
        contacts = [_contact(frame=10, ball_y=0.55, velocity=0.025, court_side="near")]
        # Ball moving toward net (near side: Y decreasing toward 0.5)
        ball_positions = [
            _bp(11, 0.54), _bp(12, 0.53), _bp(13, 0.52),
            _bp(14, 0.51), _bp(15, 0.50), _bp(16, 0.49),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.serve is not None

    def test_velocity_without_trajectory_still_triggers_serve(self) -> None:
        """High-velocity contact without ball_positions falls back to velocity-only."""
        contacts = [_contact(frame=10, ball_y=0.55, velocity=0.025, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.serve is not None


class TestBallMovingTowardNet:
    """Tests for the _ball_moving_toward_net helper."""

    def test_near_side_toward_net(self) -> None:
        """Ball on near side (Y > net_y) moving toward net = decreasing Y."""
        positions = [
            _bp(11, 0.70), _bp(12, 0.68), _bp(13, 0.65),
            _bp(14, 0.62), _bp(15, 0.58), _bp(16, 0.55),
        ]
        assert _ball_moving_toward_net(positions, contact_frame=10, ball_y=0.72, net_y=0.5)

    def test_far_side_toward_net(self) -> None:
        """Ball on far side (Y < net_y) moving toward net = increasing Y."""
        positions = [
            _bp(11, 0.30), _bp(12, 0.32), _bp(13, 0.35),
            _bp(14, 0.38), _bp(15, 0.42), _bp(16, 0.45),
        ]
        assert _ball_moving_toward_net(positions, contact_frame=10, ball_y=0.28, net_y=0.5)

    def test_lateral_movement_not_toward_net(self) -> None:
        """Ball moving laterally (Y oscillating, not trending toward net)."""
        positions = [
            _bp(11, 0.70), _bp(12, 0.73), _bp(13, 0.70),
            _bp(14, 0.73), _bp(15, 0.70), _bp(16, 0.73),
        ]
        assert not _ball_moving_toward_net(
            positions, contact_frame=10, ball_y=0.70, net_y=0.5,
        )

    def test_insufficient_data(self) -> None:
        """Returns None with fewer than 3 positions in the look-ahead window."""
        positions = [_bp(11, 0.70), _bp(12, 0.65)]
        assert _ball_moving_toward_net(
            positions, contact_frame=10, ball_y=0.72, net_y=0.5,
        ) is None


class TestPreServeIsolation:
    """Tests for pre-serve contact isolation from state machine."""

    def test_pre_serve_contacts_are_unknown(self) -> None:
        """Contacts before the serve are classified as UNKNOWN."""
        contacts = [
            _contact(frame=3, ball_y=0.5, velocity=0.005, court_side="far"),  # Pre-serve FP
            _contact(frame=10, ball_y=0.85, court_side="near"),  # Actual serve
            _contact(frame=30, ball_y=0.3, court_side="far"),   # Receive
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[0].action_type == ActionType.UNKNOWN
        assert result.actions[1].action_type == ActionType.SERVE
        assert result.actions[2].action_type == ActionType.RECEIVE

    def test_pre_serve_contacts_dont_corrupt_possession(self) -> None:
        """Pre-serve contacts on opposite side don't reset possession counter."""
        contacts = [
            _contact(frame=3, ball_y=0.5, velocity=0.005, court_side="far"),  # Pre-serve FP
            _contact(frame=10, ball_y=0.85, court_side="near"),  # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set (2nd on far)
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[0].action_type == ActionType.UNKNOWN
        assert result.actions[1].action_type == ActionType.SERVE
        assert result.actions[2].action_type == ActionType.RECEIVE
        # Without isolation, the far-side FP would have set current_side="far",
        # and contact_count would be wrong
        assert result.actions[3].action_type == ActionType.SET


class TestServeReceiveDisambiguation:
    """Tests for serve vs receive disambiguation using trajectory."""

    def test_high_velocity_receive_not_classified_as_serve(self) -> None:
        """High-velocity contact with lateral trajectory is NOT serve."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve (baseline)
            # High-velocity contact on far side — could be confused as serve
            # but ball goes laterally, not toward net
            _contact(frame=30, ball_y=0.3, velocity=0.030, court_side="far"),
        ]
        # Ball moving laterally after frame 30 (not toward net)
        ball_positions = [
            _bp(31, 0.31), _bp(32, 0.32), _bp(33, 0.31),
            _bp(34, 0.30), _bp(35, 0.31), _bp(36, 0.32),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # First contact at baseline should be serve
        assert result.actions[0].action_type == ActionType.SERVE
        # Second contact should be receive, not serve
        assert result.actions[1].action_type == ActionType.RECEIVE

    def test_high_velocity_serve_with_toward_net_trajectory(self) -> None:
        """High-velocity contact with toward-net trajectory IS serve."""
        contacts = [
            _contact(frame=10, ball_y=0.55, velocity=0.030, court_side="near"),
        ]
        # Ball moving toward net from near side (decreasing Y)
        ball_positions = [
            _bp(11, 0.54), _bp(12, 0.52), _bp(13, 0.50),
            _bp(14, 0.48), _bp(15, 0.46), _bp(16, 0.44),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)
        assert result.actions[0].action_type == ActionType.SERVE


class TestTrajectoryPossession:
    """Tests for trajectory-authoritative possession tracking."""

    def test_fp_near_net_doesnt_reset_counter_with_trajectory(self) -> None:
        """FP contact near net with different court_side doesn't reset if no crossing."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set (2nd)
            # FP near net tagged as "near" side, but ball didn't cross net
            _contact(frame=52, ball_y=0.48, court_side="near"),
            _contact(frame=60, ball_y=0.35, court_side="far"),   # Should be attack (3rd+)
        ]
        # Ball stays on far side between contacts 45-60 (no crossing)
        ball_positions = [
            _bp(46, 0.28), _bp(47, 0.30), _bp(48, 0.33),
            _bp(49, 0.36), _bp(50, 0.38), _bp(51, 0.40),
            _bp(53, 0.42), _bp(54, 0.40), _bp(55, 0.38),
            _bp(56, 0.36), _bp(57, 0.34), _bp(58, 0.32),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[1].action_type == ActionType.RECEIVE
        assert result.actions[2].action_type == ActionType.SET
        # FP at net: trajectory says no crossing, so counter keeps incrementing
        # Contact 3 (index 3) is still on far side in the state machine's view
        assert result.actions[3].action_type == ActionType.ATTACK
        # Contact 4 (index 4): safety valve resets at >3 contacts (beach max 3)
        assert result.actions[4].action_type == ActionType.DIG

    def test_safety_valve_at_4_contacts(self) -> None:
        """After 4 contacts on same side with trajectory, force possession change."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive (1st)
            _contact(frame=40, ball_y=0.25, court_side="far"),   # Set (2nd)
            _contact(frame=50, ball_y=0.35, court_side="far"),   # Attack (3rd)
            _contact(frame=60, ball_y=0.30, court_side="far"),   # 4th on far
            # 5th contact, court_side changes — safety valve should trigger
            _contact(frame=75, ball_y=0.7, court_side="near"),
        ]
        # Ball stays on far side (no crossing detected by _ball_crossed_net)
        ball_positions = [
            _bp(31, 0.32), _bp(32, 0.30), _bp(33, 0.28),
            _bp(41, 0.27), _bp(42, 0.30), _bp(43, 0.33),
            _bp(51, 0.34), _bp(52, 0.32), _bp(53, 0.30),
            _bp(61, 0.32), _bp(62, 0.34), _bp(63, 0.36),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # After 4 contacts on far, safety valve triggers for near contact
        assert result.actions[5].action_type == ActionType.DIG

    def test_no_trajectory_falls_back_to_court_side(self) -> None:
        """Without ball_positions, court_side comparison still works for possession."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set
            _contact(frame=55, ball_y=0.35, court_side="far"),   # Attack
            _contact(frame=70, ball_y=0.7, court_side="near"),   # Dig
        ]
        # No ball_positions — falls back to court_side comparison
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[4].action_type == ActionType.DIG


class TestNetCrossingReceiveFallback:
    """Tests for receive classification when ball crosses net."""

    def test_receive_detected_via_net_crossing(self) -> None:
        """Receive detected even if court_side matches serve_side, via net crossing."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            # Contact right at net — court_side is "near" (same as serve) but
            # ball actually crossed the net
            _contact(frame=30, ball_y=0.52, court_side="near"),
        ]
        # Ball crosses from near to far between contacts
        ball_positions = [
            _bp(6, 0.80), _bp(7, 0.75), _bp(8, 0.70),   # Near side
            _bp(10, 0.60), _bp(12, 0.55),
            _bp(15, 0.50),
            _bp(18, 0.45), _bp(20, 0.40), _bp(22, 0.35),  # Far side
            _bp(25, 0.38), _bp(27, 0.42), _bp(29, 0.48),  # Comes back
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[1].action_type == ActionType.RECEIVE


class TestPhantomServe:
    """Tests for phantom serve detection (missed serve → first contact is receive)."""

    def test_pass3_fallback_with_no_toward_net_becomes_receive(self) -> None:
        """When Pass 3 fallback is used and ball doesn't move toward net,
        a synthetic serve is prepended and the first contact becomes receive."""
        # Contact NOT at baseline, NOT high velocity → Pass 1 and 2 fail
        # Pass 3 picks it as first-contact fallback
        contacts = [
            _contact(frame=10, ball_y=0.4, velocity=0.010, court_side="far"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
        ]
        # Ball moves AWAY from net (net at 0.5, far side = below)
        # Toward net from far side = increasing Y. This ball decreases.
        ball_positions = [
            _bp(11, 0.39), _bp(12, 0.37), _bp(13, 0.35),
            _bp(14, 0.33), _bp(15, 0.32), _bp(16, 0.31),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # Phantom serve: synthetic serve prepended, first contact → receive
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is True
        assert result.actions[1].action_type == ActionType.RECEIVE

    def test_pass1_arc_serve_not_overridden_by_phantom(self) -> None:
        """When Pass 1 (arc crossing) finds the serve, phantom serve
        does NOT trigger even if trajectory is ambiguous."""
        contacts = [
            _contact(frame=10, ball_y=0.7, velocity=0.010, court_side="near"),
            _contact(frame=40, ball_y=0.3, court_side="far"),
        ]
        # Ball crosses net between contacts (Pass 1 arc detection triggers)
        ball_positions = [
            _bp(11, 0.68), _bp(12, 0.65), _bp(13, 0.60),
            _bp(15, 0.55), _bp(17, 0.50),
            _bp(20, 0.45), _bp(22, 0.40), _bp(25, 0.35),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # Pass 1 found the serve via arc — should stay as serve
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[1].action_type == ActionType.RECEIVE

    def test_pass2_baseline_serve_not_overridden_by_phantom(self) -> None:
        """When Pass 2 (baseline position) finds the serve, phantom serve
        does NOT trigger."""
        contacts = [
            _contact(frame=10, ball_y=0.85, velocity=0.010, court_side="near"),
            _contact(frame=40, ball_y=0.3, court_side="far"),
        ]
        # Ball goes laterally after serve (doesn't move toward net immediately)
        ball_positions = [
            _bp(11, 0.84), _bp(12, 0.84), _bp(13, 0.85),
            _bp(14, 0.84), _bp(15, 0.85), _bp(16, 0.84),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # Pass 2 found serve at baseline — should stay as serve regardless
        assert result.actions[0].action_type == ActionType.SERVE


class TestSyntheticServe:
    """Tests for synthetic serve injection when serve is missed."""

    def test_phantom_serve_injects_synthetic(self) -> None:
        """Phantom serve detected → synthetic SERVE prepended, first contact → RECEIVE."""
        # Pass 3 fallback: not at baseline, low velocity
        contacts = [
            _contact(frame=10, ball_y=0.4, velocity=0.010, court_side="far"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
        ]
        # Ball moves away from net (far side, Y decreasing = away from net)
        ball_positions = [
            _bp(11, 0.39), _bp(12, 0.37), _bp(13, 0.35),
            _bp(14, 0.33), _bp(15, 0.32), _bp(16, 0.31),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # Synthetic serve should be first action
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is True
        assert result.actions[0].confidence == 0.4
        assert result.actions[0].player_track_id == -1
        # Original contact becomes receive
        assert result.actions[1].action_type == ActionType.RECEIVE
        assert result.actions[1].is_synthetic is False

    def test_real_serve_not_synthetic(self) -> None:
        """Pass 1/2 serve → no synthetic, no is_synthetic flag."""
        contacts = [_contact(frame=10, ball_y=0.85, court_side="near")]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is False

    def test_synthetic_serve_has_correct_side(self) -> None:
        """Synthetic serve side = opposite of first contact's court_side."""
        contacts = [
            _contact(frame=10, ball_y=0.4, velocity=0.010, court_side="far"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
        ]
        # Ball away from net
        ball_positions = [
            _bp(11, 0.39), _bp(12, 0.37), _bp(13, 0.35),
            _bp(14, 0.33), _bp(15, 0.32), _bp(16, 0.31),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # First contact is on far side → serve must be from near side
        assert result.actions[0].court_side == "near"

    def test_synthetic_serve_downstream_sequence(self) -> None:
        """Full sequence SERVE(synthetic)→RECEIVE→SET→ATTACK works."""
        contacts = [
            _contact(frame=10, ball_y=0.4, velocity=0.010, court_side="far"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
            _contact(frame=55, ball_y=0.38, court_side="far"),
        ]
        # Ball away from net
        ball_positions = [
            _bp(11, 0.39), _bp(12, 0.37), _bp(13, 0.35),
            _bp(14, 0.33), _bp(15, 0.32), _bp(16, 0.31),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.action_sequence == [
            ActionType.SERVE,    # synthetic
            ActionType.RECEIVE,  # reclassified from first contact
            ActionType.SET,      # 2nd on far
            ActionType.ATTACK,   # 3rd on far
        ]

    def test_toward_net_none_keeps_serve(self) -> None:
        """Insufficient ball data → conservative, keep as serve (no phantom)."""
        # Pass 3 fallback, only 2 ball positions after contact → toward_net=None
        contacts = [
            _contact(frame=10, ball_y=0.4, velocity=0.010, court_side="far"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
        ]
        # Only 2 positions → _ball_moving_toward_net returns None
        ball_positions = [_bp(11, 0.39), _bp(12, 0.37)]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # toward_net=None → conservative, treat as normal serve
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is False

    def test_toward_net_none_same_side_keeps_serve(self) -> None:
        """Insufficient ball data + inferred serve side == contact → normal serve."""
        # Contact on near side, so _infer_serve_side returns "far" (opposite)
        # which ≠ "near" → phantom. But if we force same-side, it should keep serve.
        # This test uses no ball_positions at all (serve_pass=3, no ball data),
        # so toward_net check is skipped entirely.
        contacts = [
            _contact(frame=10, ball_y=0.6, velocity=0.010, court_side="near"),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # No ball_positions → phantom check skipped → normal serve
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is False

    def test_synthetic_serve_to_dict(self) -> None:
        """Synthetic serve includes isSynthetic in to_dict output."""
        serve = _make_synthetic_serve("near", 30, 0.5)
        d = serve.to_dict()
        assert d["isSynthetic"] is True
        assert d["action"] == "serve"
        assert d["playerTrackId"] == -1

    def test_real_serve_to_dict_no_synthetic_key(self) -> None:
        """Non-synthetic action omits isSynthetic from to_dict output."""
        action = ClassifiedAction(
            ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9,
        )
        d = action.to_dict()
        assert "isSynthetic" not in d

    def test_infer_serve_side_from_first_contact(self) -> None:
        """_infer_serve_side returns opposite of first contact's court_side."""
        contact = _contact(frame=10, court_side="far")
        assert _infer_serve_side(contact) == "near"

        contact2 = _contact(frame=10, court_side="near")
        assert _infer_serve_side(contact2) == "far"

    def test_infer_serve_side_court_side_takes_priority(self) -> None:
        """court_side (Signal 1) takes priority over ball trajectory (Signal 2)."""
        contact = _contact(frame=20, court_side="near")
        # Ball trajectory suggests near side served, but court_side="near"
        # → Signal 1 returns "far" (opposite) without consulting trajectory
        ball_positions = [
            _bp(5, 0.8), _bp(8, 0.7), _bp(12, 0.6),
        ]
        assert _infer_serve_side(contact, ball_positions, net_y=0.5) == "far"

    def test_synthetic_serve_frame_placement(self) -> None:
        """Synthetic serve is placed ~1s (30 frames) before first contact."""
        serve = _make_synthetic_serve("near", 50, 0.5)
        assert serve.frame == 20  # 50 - 30

        # Don't go below 0
        serve_early = _make_synthetic_serve("near", 10, 0.5)
        assert serve_early.frame == 0

    def test_phantom_serve_injects_synthetic_learned_path(self) -> None:
        """classify_rally_learned also injects synthetic serve on phantom."""
        contacts = [
            _contact(frame=10, ball_y=0.4, velocity=0.010, court_side="far"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
            _contact(frame=55, ball_y=0.38, court_side="far"),
        ]
        # Ball away from net → phantom serve
        ball_positions = [
            _bp(11, 0.39), _bp(12, 0.37), _bp(13, 0.35),
            _bp(14, 0.33), _bp(15, 0.32), _bp(16, 0.31),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
            ball_positions=ball_positions,
        )

        # Mock classifier — predict returns (action_str, confidence)
        mock_clf = MagicMock()
        mock_clf.predict.return_value = [("set", 0.8)]

        classifier = ActionClassifier()
        result = classifier.classify_rally_learned(seq, mock_clf, rally_id="test")

        # Synthetic serve prepended, first contact → receive
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is True
        assert result.actions[1].action_type == ActionType.RECEIVE
        assert result.actions[1].is_synthetic is False
        # Remaining contacts go through the learned classifier
        assert len(result.actions) >= 3
