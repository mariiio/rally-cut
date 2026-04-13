"""Unit tests for rule-based action classification."""

from __future__ import annotations

from unittest.mock import MagicMock

from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    ClassifiedAction,
    RallyActions,
    _ball_moving_toward_net,
    _ball_starts_on_contact_side,
    _classify_serve_contact,
    _serving_side_from_contact,
    _compute_auto_split_y,
    _compute_expected_teams,
    _find_server_by_position,
    _find_serving_side_by_formation,
    _find_serving_team_by_formation,
    _infer_serve_side,
    _is_ball_on_serve_side,
    _make_synthetic_serve,
    _reattribute_server_exclusion,
    classify_rally_actions,
    reattribute_players,
    repair_action_sequence,
    validate_action_sequence,
    viterbi_decode_actions,
)
from rallycut.court.calibration import CourtCalibrator
from rallycut.tracking.action_type_classifier import _count_contacts_on_side
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import Contact, ContactSequence, ball_crossed_net
from rallycut.tracking.player_tracker import PlayerPosition


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
        assert ball_crossed_net(positions, from_frame=9, to_frame=16, net_y=0.5)

    def test_clear_crossing_far_to_near(self) -> None:
        """Ball crossing from far (low Y) to near (high Y) is detected."""
        positions = [
            _bp(10, 0.3), _bp(11, 0.35), _bp(12, 0.4),   # Far side
            _bp(13, 0.55), _bp(14, 0.6), _bp(15, 0.65),   # Near side
        ]
        assert ball_crossed_net(positions, from_frame=9, to_frame=16, net_y=0.5)

    def test_no_crossing_same_side(self) -> None:
        """Ball staying on one side is not a crossing."""
        positions = [
            _bp(10, 0.7), _bp(11, 0.72), _bp(12, 0.68),
            _bp(13, 0.71), _bp(14, 0.69),
        ]
        assert not ball_crossed_net(positions, from_frame=9, to_frame=15, net_y=0.5)

    def test_too_few_positions(self) -> None:
        """Returns None (insufficient data) with too few positions between contacts."""
        positions = [_bp(10, 0.7), _bp(11, 0.3)]
        assert ball_crossed_net(positions, from_frame=9, to_frame=12, net_y=0.5) is None

    def test_noisy_single_frame_not_crossing(self) -> None:
        """Single-frame noise crossing net is not enough (needs min_frames_per_side)."""
        positions = [
            _bp(10, 0.6), _bp(11, 0.55),  # Near
            _bp(12, 0.45),                  # Single far frame
            _bp(13, 0.55), _bp(14, 0.6),   # Back to near
        ]
        assert not ball_crossed_net(positions, from_frame=9, to_frame=15, net_y=0.5)

    def test_arc_over_net_near_to_far(self) -> None:
        """Ball arcs above net (low Y at apex) but starts near, ends far."""
        positions = [
            _bp(10, 0.65), _bp(12, 0.55),   # Start: near side
            _bp(14, 0.30), _bp(16, 0.20),   # Arc above net
            _bp(18, 0.25), _bp(20, 0.35),   # End: far side
        ]
        assert ball_crossed_net(positions, from_frame=9, to_frame=21, net_y=0.5) is True

    def test_arc_over_net_far_to_near(self) -> None:
        """Ball arcs above net from far side to near side."""
        positions = [
            _bp(10, 0.35), _bp(12, 0.40),   # Start: far side
            _bp(14, 0.25), _bp(16, 0.20),   # Arc above net
            _bp(18, 0.55), _bp(20, 0.65),   # End: near side
        ]
        assert ball_crossed_net(positions, from_frame=9, to_frame=21, net_y=0.5) is True

    def test_small_displacement_near_net_returns_false(self) -> None:
        """Small Y displacement near net is not a crossing (below threshold)."""
        positions = [
            _bp(10, 0.51), _bp(11, 0.52),   # Just barely near side
            _bp(12, 0.48), _bp(13, 0.47),   # Just barely far side
        ]
        # y_delta = |0.475 - 0.515| = 0.04 < 0.14 threshold → False
        assert ball_crossed_net(positions, from_frame=9, to_frame=14, net_y=0.5) is False

    def test_net_y_not_used_in_decision(self) -> None:
        """Crossing detected by Y displacement regardless of net_y value."""
        positions = [
            _bp(10, 0.7), _bp(11, 0.65),    # Start: high Y
            _bp(14, 0.4), _bp(15, 0.35),    # End: low Y
        ]
        # y_delta = |0.375 - 0.675| = 0.3 > threshold → True
        # Even with wildly wrong net_y, displacement still detects crossing
        assert ball_crossed_net(positions, from_frame=9, to_frame=16, net_y=0.1) is True
        assert ball_crossed_net(positions, from_frame=9, to_frame=16, net_y=0.9) is True


class TestCountContactsOnSide:
    """Tests for _count_contacts_on_side fallthrough to court_side."""

    def test_court_side_fallback_with_ball_positions(self) -> None:
        """Court side resets count even when ball_positions exist but show no crossing."""
        contacts = [
            _contact(frame=30, ball_y=0.3, court_side="far"),
            _contact(frame=55, ball_y=0.65, court_side="near"),
        ]
        # Ball positions all on far side — ball_crossed_net returns False
        ball_positions = [
            _bp(35, 0.32), _bp(40, 0.30), _bp(45, 0.28), _bp(50, 0.26),
        ]
        count = _count_contacts_on_side(contacts, 1, ball_positions, net_y=0.5)
        assert count == 1  # Reset due to court_side change

    def test_same_side_continues_count(self) -> None:
        """Same court_side and no crossing increments count."""
        contacts = [
            _contact(frame=30, ball_y=0.3, court_side="far"),
            _contact(frame=45, ball_y=0.25, court_side="far"),
        ]
        ball_positions = [
            _bp(33, 0.28), _bp(35, 0.26), _bp(37, 0.27), _bp(40, 0.25),
        ]
        count = _count_contacts_on_side(contacts, 1, ball_positions, net_y=0.5)
        assert count == 2


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

    def test_block_counts_as_touch(self) -> None:
        """Block counts as 1st touch in beach volleyball — next contact is set (2nd)."""
        config = ActionClassifierConfig(block_max_frame_gap=10)
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),     # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),      # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),     # Set
            _contact(frame=55, ball_y=0.4, court_side="far"),      # Attack
            _contact(frame=60, ball_y=0.48, court_side="near",     # Block (1st)
                     is_at_net=True),
            _contact(frame=70, ball_y=0.65, court_side="near"),    # Set (2nd)
            _contact(frame=80, ball_y=0.6, court_side="near"),     # Attack (3rd)
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5, rally_start_frame=0)
        classifier = ActionClassifier(config)
        result = classifier.classify_rally(seq)

        assert result.actions[4].action_type == ActionType.BLOCK
        assert result.actions[5].action_type == ActionType.SET    # 2nd touch
        assert result.actions[6].action_type == ActionType.ATTACK  # 3rd touch

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

    def test_num_contacts_includes_blocks(self) -> None:
        """num_contacts includes blocks (beach volleyball: block counts as a touch)."""
        actions = [
            ClassifiedAction(ActionType.SERVE, 5, 0.5, 0.8, 0.02, 1, "near", 0.9),
            ClassifiedAction(ActionType.RECEIVE, 30, 0.5, 0.3, 0.03, 2, "far", 0.8),
            ClassifiedAction(ActionType.BLOCK, 60, 0.5, 0.48, 0.04, 1, "near", 0.9),
        ]
        rally = RallyActions(actions=actions)
        assert rally.num_contacts == 3  # Block counts in beach volleyball

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

    def test_court_side_change_resets_counter(self) -> None:
        """Court side change resets touch counter even if ball trajectory is ambiguous."""
        contacts = [
            _contact(frame=5, ball_y=0.85, court_side="near"),   # Serve
            _contact(frame=30, ball_y=0.3, court_side="far"),    # Receive
            _contact(frame=45, ball_y=0.25, court_side="far"),   # Set (2nd)
            # Contact near net tagged as "near" side — court_side change resets counter
            _contact(frame=52, ball_y=0.48, court_side="near"),
            _contact(frame=60, ball_y=0.35, court_side="far"),   # After reset
        ]
        # Ball stays on far side between contacts 45-60 (no crossing detected
        # by ball_crossed_net, but court_side changed — trust court_side)
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
        # Court side changed to "near" → counter reset → 1st touch;
        # Viterbi favours SET→ATTACK (0.90) over SET→DIG (0.05) so both
        # contacts get relabeled to the most likely post-set action.
        assert result.actions[3].action_type == ActionType.ATTACK
        # Back to "far" → counter reset → Viterbi prefers ATTACK→DIG
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
        # Ball stays on far side (no crossing detected by ball_crossed_net)
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

        # Safety valve triggers at 4th far contact (index 4 → DIG),
        # next near contact (index 5): rule-based assigns DIG (1st on side),
        # but Viterbi corrects to SET (DIG→SET is 0.65, DIG→DIG only 0.10).
        assert result.actions[4].action_type == ActionType.DIG
        assert result.actions[5].action_type == ActionType.SET

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
        """Synthetic serve uses rally_start_frame when available."""
        # With rally_start_frame within 90 frames
        serve = _make_synthetic_serve("near", 50, 0.5, rally_start_frame=10)
        assert serve.frame == 10

        # Without rally_start_frame, falls back to first_contact - 30
        serve_no_start = _make_synthetic_serve("near", 50, 0.5)
        assert serve_no_start.frame == 20  # 50 - 30

        # Don't go below 0
        serve_early = _make_synthetic_serve("near", 10, 0.5)
        assert serve_early.frame == 0

    def test_synthetic_serve_ignores_distant_rally_start(self) -> None:
        """Rally start >90 frames before first contact is ignored."""
        serve = _make_synthetic_serve("near", 200, 0.5, rally_start_frame=50)
        assert serve.frame == 170  # 200 - 30, ignores start_frame=50

    def test_synthetic_serve_ignores_start_after_contact(self) -> None:
        """Rally start after first contact is ignored."""
        serve = _make_synthetic_serve("near", 50, 0.5, rally_start_frame=60)
        assert serve.frame == 20  # 50 - 30

    def test_phantom_serve_injects_synthetic_learned_path(self) -> None:
        """classify_rally with classifier also injects synthetic serve on phantom."""
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
        result = classifier.classify_rally(seq, rally_id="test", classifier=mock_clf)

        # Synthetic serve prepended, first contact → receive
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is True
        assert result.actions[1].action_type == ActionType.RECEIVE
        assert result.actions[1].is_synthetic is False
        # Remaining contacts go through the learned classifier
        assert len(result.actions) >= 3


class TestIsBallOnServeSide:
    """Tests for the _is_ball_on_serve_side helper."""

    def test_near_side_serve_ball_on_near(self) -> None:
        """Ball on near side (Y > net_y + margin) → True for near serve."""
        assert _is_ball_on_serve_side(0.7, "near", 0.5) is True

    def test_near_side_serve_ball_on_far(self) -> None:
        """Ball on far side (Y < net_y - margin) → False for near serve."""
        assert _is_ball_on_serve_side(0.3, "near", 0.5) is False

    def test_far_side_serve_ball_on_far(self) -> None:
        """Ball on far side → True for far serve."""
        assert _is_ball_on_serve_side(0.3, "far", 0.5) is True

    def test_far_side_serve_ball_on_near(self) -> None:
        """Ball on near side → False for far serve."""
        assert _is_ball_on_serve_side(0.7, "far", 0.5) is False

    def test_ball_near_net_is_indeterminate(self) -> None:
        """Ball within margin of net → None (indeterminate)."""
        assert _is_ball_on_serve_side(0.52, "near", 0.5) is None
        assert _is_ball_on_serve_side(0.48, "far", 0.5) is None

    def test_unknown_court_side(self) -> None:
        """Unknown court side → None."""
        assert _is_ball_on_serve_side(0.7, "unknown", 0.5) is None


class TestPass2BaselineTrajectoryGate:
    """Tests for trajectory check on Pass 2 baseline serve detection."""

    def test_baseline_contact_away_from_net_skipped(self) -> None:
        """Baseline contact with ball moving away from net is not serve."""
        # Contact at near baseline but ball moves away from net
        contacts = [
            _contact(frame=10, ball_y=0.85, court_side="near"),
            _contact(frame=40, ball_y=0.6, court_side="near"),
        ]
        # Ball moves AWAY from net (near side, Y increasing = away)
        ball_positions = [
            _bp(11, 0.86), _bp(12, 0.87), _bp(13, 0.88),
            _bp(14, 0.89), _bp(15, 0.90), _bp(16, 0.91),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 0, 0.5, ball_positions=ball_positions,
        )
        # Pass 2 should skip the baseline contact; falls to pass 3
        assert serve_pass == 3

    def test_baseline_contact_toward_net_accepted(self) -> None:
        """Baseline contact with ball moving toward net is serve."""
        contacts = [
            _contact(frame=10, ball_y=0.85, court_side="near"),
            _contact(frame=40, ball_y=0.3, court_side="far"),
        ]
        # Ball moves toward net (near side, Y decreasing)
        ball_positions = [
            _bp(11, 0.83), _bp(12, 0.80), _bp(13, 0.76),
            _bp(14, 0.72), _bp(15, 0.68), _bp(16, 0.64),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 0, 0.5, ball_positions=ball_positions,
        )
        # Pass 1 (arc) or Pass 2 (baseline+trajectory) both validly detect
        # this serve — arc may fire first when ball crosses net.
        assert serve_pass in (1, 2)
        assert serve_idx == 0

    def test_baseline_contact_no_trajectory_accepted(self) -> None:
        """Baseline contact without ball positions is accepted (conservative)."""
        contacts = [
            _contact(frame=10, ball_y=0.85, court_side="near"),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 0, 0.5, ball_positions=None,
        )
        assert serve_pass == 2
        assert serve_idx == 0


class TestBallStartsOnContactSide:
    """Tests for _ball_starts_on_contact_side helper."""

    def test_ball_on_near_side(self) -> None:
        positions = [_bp(11, 0.6), _bp(12, 0.62), _bp(13, 0.58)]
        assert _ball_starts_on_contact_side(positions, 10, "near", 0.5) is True

    def test_ball_on_far_side_for_near_contact(self) -> None:
        positions = [_bp(11, 0.40), _bp(12, 0.38), _bp(13, 0.42)]
        assert _ball_starts_on_contact_side(positions, 10, "near", 0.5) is False

    def test_ball_on_far_side_for_far_contact(self) -> None:
        positions = [_bp(11, 0.35), _bp(12, 0.32), _bp(13, 0.38)]
        assert _ball_starts_on_contact_side(positions, 10, "far", 0.5) is True

    def test_no_positions_returns_none(self) -> None:
        assert _ball_starts_on_contact_side([], 10, "near", 0.5) is None

    def test_unknown_side_returns_none(self) -> None:
        positions = [_bp(11, 0.6)]
        assert _ball_starts_on_contact_side(positions, 10, "unknown", 0.5) is None


class TestPass1ArcDirectionValidation:
    """Tests for direction validation in Pass 1 (arc-based serve detection)."""

    def test_arc_serve_rejected_when_ball_on_opposite_side(self) -> None:
        """Arc crossing detected but ball starts on wrong side → skip."""
        # Contact at far side (ball_y=0.3), next contact at near side
        contacts = [
            _contact(frame=97, ball_y=0.467, court_side="near"),
            _contact(frame=146, ball_y=0.35, court_side="far"),
        ]
        # Ball crosses net between 97→146, but at frame 98-102 ball is
        # already on far side (< net_y) — this is a receive, not a serve
        ball_positions = [
            _bp(95, 0.43), _bp(97, 0.467),
            _bp(98, 0.45), _bp(99, 0.44), _bp(100, 0.42),
            _bp(101, 0.40), _bp(102, 0.38),
            # ball continues to far side, then comes back
            _bp(120, 0.30), _bp(130, 0.35), _bp(140, 0.42),
            _bp(145, 0.50), _bp(146, 0.55),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 80, 0.5, ball_positions=ball_positions,
        )
        # Pass 1 should skip contact 0 (ball on wrong side), fall to pass 3
        assert serve_pass != 1

    def test_arc_serve_accepted_when_ball_on_correct_side(self) -> None:
        """Arc crossing detected with ball on correct side → accepted."""
        contacts = [
            _contact(frame=10, ball_y=0.7, court_side="near"),
            _contact(frame=40, ball_y=0.3, court_side="far"),
        ]
        # Ball starts on near side (>0.5) then crosses to far side
        ball_positions = [
            _bp(11, 0.68), _bp(12, 0.65), _bp(13, 0.62),
            _bp(14, 0.58), _bp(15, 0.55),
            _bp(20, 0.48), _bp(25, 0.42), _bp(30, 0.35),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 0, 0.5, ball_positions=ball_positions,
        )
        assert serve_pass == 1
        assert serve_idx == 0


class TestPass2VelocityDirectionValidation:
    """Tests for direction validation in Pass 2 velocity branch."""

    def test_high_velocity_contact_rejected_when_ball_on_wrong_side(self) -> None:
        """High velocity contact with ball on wrong side → skip."""
        contacts = [
            # High velocity, NOT at baseline, near side
            _contact(frame=97, ball_y=0.60, velocity=0.05, court_side="near"),
            _contact(frame=140, ball_y=0.35, court_side="far"),
        ]
        # Ball at f98-102 is on far side (< 0.5)
        ball_positions = [
            _bp(95, 0.55), _bp(97, 0.60),
            _bp(98, 0.45), _bp(99, 0.43), _bp(100, 0.41),
            _bp(101, 0.39), _bp(102, 0.37),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 80, 0.5, ball_positions=ball_positions,
        )
        # Should NOT be picked as serve via velocity (pass 2)
        assert not (serve_pass == 2 and serve_idx == 0)

    def test_high_velocity_contact_accepted_when_ball_on_correct_side(self) -> None:
        """High velocity contact with ball on correct side → accepted."""
        contacts = [
            _contact(frame=10, ball_y=0.60, velocity=0.05, court_side="near"),
            _contact(frame=40, ball_y=0.3, court_side="far"),
        ]
        # Ball starts on near side
        ball_positions = [
            _bp(11, 0.62), _bp(12, 0.60), _bp(13, 0.57),
            _bp(14, 0.54), _bp(15, 0.51),
        ]
        classifier = ActionClassifier()
        serve_idx, serve_pass = classifier._find_serve_index(
            contacts, 0, 0.5, ball_positions=ball_positions,
        )
        assert serve_pass == 2
        assert serve_idx == 0


class TestPass3CourtSidePhantom:
    """Tests for court-side phantom serve detection on Pass 3 fallback."""

    def test_ball_on_wrong_side_triggers_phantom(self) -> None:
        """Pass 3 contact with ball clearly on receiving side → phantom serve."""
        # Contact on "near" side but ball_y=0.3 (far side of court)
        contacts = [
            _contact(frame=10, ball_y=0.3, velocity=0.010, court_side="near"),
            _contact(frame=40, ball_y=0.35, court_side="far"),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        # Should detect phantom and inject synthetic serve
        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is True
        assert result.actions[1].action_type == ActionType.RECEIVE

    def test_ball_on_correct_side_no_phantom(self) -> None:
        """Pass 3 contact with ball on correct side → normal serve."""
        contacts = [
            _contact(frame=10, ball_y=0.7, velocity=0.010, court_side="near"),
        ]
        seq = ContactSequence(
            contacts=contacts, net_y=0.5, rally_start_frame=0,
        )
        result = classify_rally_actions(seq, use_classifier=False)

        assert result.actions[0].action_type == ActionType.SERVE
        assert result.actions[0].is_synthetic is False


class TestRepairWrongSideServe:
    """Tests for repair rule 0: serve on wrong court side."""

    def test_wrong_side_serve_repaired(self) -> None:
        """Non-synthetic serve with ball on wrong side → reclassified as receive."""
        actions = [
            # Serve labeled "near" but ball_y=0.3 (far side)
            ClassifiedAction(
                ActionType.SERVE, 64, 0.5, 0.3, 0.02, 1, "near", 0.9,
            ),
            ClassifiedAction(
                ActionType.SET, 80, 0.5, 0.35, 0.01, 2, "far", 0.8,
            ),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5, rally_start_frame=30)

        # Synthetic serve prepended, original serve → receive
        assert len(repaired) == 3
        assert repaired[0].action_type == ActionType.SERVE
        assert repaired[0].is_synthetic is True
        assert repaired[0].court_side == "far"  # Opposite of original
        assert repaired[0].frame == 30  # rally_start_frame
        assert repaired[1].action_type == ActionType.RECEIVE
        assert repaired[1].frame == 64

    def test_correct_side_serve_not_repaired(self) -> None:
        """Serve with ball on correct side is NOT repaired."""
        actions = [
            ClassifiedAction(
                ActionType.SERVE, 10, 0.5, 0.85, 0.02, 1, "near", 0.9,
            ),
            ClassifiedAction(
                ActionType.RECEIVE, 30, 0.5, 0.3, 0.01, 2, "far", 0.8,
            ),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert len(repaired) == 2
        assert repaired[0].action_type == ActionType.SERVE
        assert repaired[0].is_synthetic is False

    def test_synthetic_serve_not_repaired(self) -> None:
        """Synthetic serve is NOT touched by Rule 0 (already handled)."""
        actions = [
            ClassifiedAction(
                ActionType.SERVE, 10, 0.5, 0.3, 0.0, -1, "near", 0.4,
                is_synthetic=True,
            ),
            ClassifiedAction(
                ActionType.RECEIVE, 40, 0.5, 0.35, 0.01, 2, "far", 0.8,
            ),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert len(repaired) == 2
        assert repaired[0].action_type == ActionType.SERVE
        assert repaired[0].is_synthetic is True


# --- Helpers for compact action construction ---

def _ca(
    action_type: ActionType,
    frame: int,
    court_side: str = "near",
    confidence: float = 0.8,
    is_synthetic: bool = False,
) -> ClassifiedAction:
    """Compact helper for creating ClassifiedAction in repair tests."""
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.6 if court_side == "near" else 0.4,
        velocity=0.02,
        player_track_id=1,
        court_side=court_side,
        confidence=confidence,
        is_synthetic=is_synthetic,
    )


class TestRepairDuplicateServe:
    """Tests for repair rule 3: duplicate serves."""

    def test_two_real_serves_second_becomes_receive(self) -> None:
        """Second non-synthetic serve → receive (then Rule 4 → set)."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.RECEIVE, 30, "far"),
            _ca(ActionType.SERVE, 60, "far"),  # duplicate
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[0].action_type == ActionType.SERVE
        # Rule 3 changes to receive, then Rule 4 changes duplicate receive to set
        assert repaired[2].action_type == ActionType.SET
        assert repaired[2].confidence <= 0.6

    def test_synthetic_serve_not_touched(self) -> None:
        """Synthetic duplicate serve is NOT repaired (only real dupes)."""
        actions = [
            _ca(ActionType.SERVE, 10, "near", is_synthetic=True),
            _ca(ActionType.SERVE, 20, "near"),  # real serve
            _ca(ActionType.RECEIVE, 40, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        # First serve is synthetic (kept), second is the real one (kept)
        assert repaired[0].action_type == ActionType.SERVE
        assert repaired[0].is_synthetic is True
        assert repaired[1].action_type == ActionType.SERVE
        assert repaired[1].is_synthetic is False

    def test_three_serves_all_extras_repaired(self) -> None:
        """Three duplicate serves → extras become receive (Rule 3),
        then duplicate receives → set (Rule 4)."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.SERVE, 30, "far"),   # dup → receive (Rule 3)
            _ca(ActionType.SERVE, 50, "near"),  # dup → receive (Rule 3) → set (Rule 4)
            _ca(ActionType.DIG, 90, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[0].action_type == ActionType.SERVE    # original
        assert repaired[1].action_type == ActionType.RECEIVE  # Rule 3
        assert repaired[2].action_type == ActionType.SET      # Rule 3 + Rule 4


class TestRepairDuplicateReceive:
    """Tests for repair rule 4: duplicate receives."""

    def test_two_receives_same_side_second_becomes_set(self) -> None:
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.RECEIVE, 30, "far"),
            _ca(ActionType.RECEIVE, 50, "far"),  # duplicate same side
            _ca(ActionType.ATTACK, 70, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[2].action_type == ActionType.SET

    def test_two_receives_different_sides_second_becomes_set(self) -> None:
        """Court_side labels are unreliable, so always use set."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.RECEIVE, 30, "far"),
            _ca(ActionType.RECEIVE, 50, "near"),  # duplicate different side
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[2].action_type == ActionType.SET

    def test_unknown_side_still_repaired(self) -> None:
        """Unknown court_side doesn't block repair (no side check needed)."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.RECEIVE, 30, "unknown"),
            _ca(ActionType.RECEIVE, 50, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[2].action_type == ActionType.SET


class TestRepairSetSet:
    """Tests for repair rule 5: set → set same side."""

    def test_set_set_same_side_second_becomes_attack(self) -> None:
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.DIG, 30, "far"),
            _ca(ActionType.SET, 50, "far"),
            _ca(ActionType.SET, 70, "far"),  # duplicate set
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[3].action_type == ActionType.ATTACK

    def test_set_set_different_sides_no_repair(self) -> None:
        """set on near → set on far is legal (new possession)."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.SET, 30, "near"),
            _ca(ActionType.SET, 50, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        # Different sides → no same-side rule applies
        assert repaired[2].action_type == ActionType.SET


class TestRepairAttackAttack:
    """Tests for repair rule 6: attack → attack same side."""

    def test_attack_attack_same_side_first_becomes_set(self) -> None:
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.DIG, 30, "far"),
            _ca(ActionType.ATTACK, 50, "far"),
            _ca(ActionType.ATTACK, 70, "far"),  # duplicate attack
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[2].action_type == ActionType.SET
        assert repaired[3].action_type == ActionType.ATTACK  # unchanged

    def test_attack_attack_different_sides_no_repair(self) -> None:
        """attack on near → attack on far is legal (new possession)."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.ATTACK, 30, "near"),
            _ca(ActionType.ATTACK, 50, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        # Different sides → no same-side rule applies
        assert repaired[2].action_type == ActionType.ATTACK


class TestCircuitBreaker:
    """Tests for the circuit breaker: max 3 repairs per rally."""

    def test_breaker_with_mixed_violations(self) -> None:
        """Mix of violations — only first 3 repaired, rest untouched.

        Rule 3 changes duplicate serve → receive (R=1).
        Rule 4 changes duplicate receives → set (R=2, R=3 → breaker).
        Main pass rules don't fire (breaker already at 3).
        """
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.SERVE, 20, "near"),   # dup serve → receive (R=1)
            _ca(ActionType.RECEIVE, 30, "far"),
            _ca(ActionType.RECEIVE, 50, "far"),  # dup receive → set (R=2)
            _ca(ActionType.SET, 70, "far"),
            _ca(ActionType.SET, 90, "far"),       # would be rule 5 but breaker
            _ca(ActionType.SET, 110, "far"),      # untouched
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        assert repaired[1].action_type == ActionType.RECEIVE  # rule 3 (R=1)
        # receive@20 (from rule 3) is first receive, receive@30 is dup → set (R=2)
        assert repaired[2].action_type == ActionType.SET       # rule 4 (R=2)
        assert repaired[3].action_type == ActionType.SET       # rule 4 (R=3, breaker)
        # Breaker hit — remaining violations untouched
        assert repaired[4].action_type == ActionType.SET       # untouched
        assert repaired[5].action_type == ActionType.SET       # untouched
        assert repaired[6].action_type == ActionType.SET       # untouched

    def test_no_repairs_on_clean_sequence(self) -> None:
        """Clean sequence → 0 repairs, all actions unchanged."""
        actions = [
            _ca(ActionType.SERVE, 10, "near"),
            _ca(ActionType.RECEIVE, 30, "far"),
            _ca(ActionType.SET, 50, "far"),
            _ca(ActionType.ATTACK, 70, "far"),
        ]
        repaired = repair_action_sequence(actions, net_y=0.5)

        types = [a.action_type for a in repaired]
        assert types == [
            ActionType.SERVE, ActionType.RECEIVE,
            ActionType.SET, ActionType.ATTACK,
        ]


class TestServerExclusion:
    """Tests for server can't receive their own serve."""

    def test_receive_reattributed_away_from_server(self) -> None:
        """When nearest player at receive is the server, pick next candidate."""
        serve_contact = _contact(10, ball_y=0.7, velocity=0.03, court_side="near",
                                 player_track_id=1)
        recv_contact = _contact(40, ball_y=0.3, velocity=0.02, court_side="far",
                                player_track_id=1)
        # Server is tid=1, but candidate tid=3 is available
        recv_contact.player_candidates = [(1, 0.05), (3, 0.08)]

        seq = ContactSequence(
            contacts=[serve_contact, recv_contact],
            net_y=0.5,
            rally_start_frame=0,
        )
        classifier = ActionClassifier()
        result = classifier.classify_rally(seq)

        serve_action = next(a for a in result.actions if a.action_type == ActionType.SERVE)
        recv_action = next(a for a in result.actions if a.action_type == ActionType.RECEIVE)

        assert serve_action.player_track_id == 1
        assert recv_action.player_track_id == 3  # Re-attributed away from server

    def test_receive_keeps_attribution_when_different_player(self) -> None:
        """When nearest player at receive is NOT the server, keep it."""
        serve_contact = _contact(10, ball_y=0.7, velocity=0.03, court_side="near",
                                 player_track_id=1)
        recv_contact = _contact(40, ball_y=0.3, velocity=0.02, court_side="far",
                                player_track_id=3)
        recv_contact.player_candidates = [(3, 0.05), (1, 0.10)]

        seq = ContactSequence(
            contacts=[serve_contact, recv_contact],
            net_y=0.5,
            rally_start_frame=0,
        )
        classifier = ActionClassifier()
        result = classifier.classify_rally(seq)

        recv_action = next(a for a in result.actions if a.action_type == ActionType.RECEIVE)
        assert recv_action.player_track_id == 3  # Unchanged

    def test_receive_no_candidates_keeps_server(self) -> None:
        """When no candidates available, keep server attribution (graceful)."""
        serve_contact = _contact(10, ball_y=0.7, velocity=0.03, court_side="near",
                                 player_track_id=1)
        recv_contact = _contact(40, ball_y=0.3, velocity=0.02, court_side="far",
                                player_track_id=1)
        # No candidates — can't re-attribute
        recv_contact.player_candidates = []

        seq = ContactSequence(
            contacts=[serve_contact, recv_contact],
            net_y=0.5,
            rally_start_frame=0,
        )
        classifier = ActionClassifier()
        result = classifier.classify_rally(seq)

        recv_action = next(a for a in result.actions if a.action_type == ActionType.RECEIVE)
        assert recv_action.player_track_id == 1  # Kept (no alternatives)


class TestReattributeServerExclusion:
    """Tests for post-classification server exclusion pass."""

    def _ca(
        self, action_type: ActionType, frame: int,
        player: int = 1, court_side: str = "near",
    ) -> ClassifiedAction:
        return ClassifiedAction(
            action_type=action_type, frame=frame,
            ball_x=0.5, ball_y=0.5, velocity=0.02,
            player_track_id=player, court_side=court_side,
            confidence=0.9,
        )

    def test_fixes_receive_with_server_tid(self) -> None:
        """Post-pass catches receive attributed to server."""
        actions = [
            self._ca(ActionType.SERVE, 10, player=1, court_side="near"),
            self._ca(ActionType.RECEIVE, 40, player=1, court_side="far"),
        ]
        contacts = [
            _contact(10, player_track_id=1),
            _contact(40, player_track_id=1),
        ]
        contacts[1].player_candidates = [(1, 0.05), (3, 0.08)]

        n = _reattribute_server_exclusion(actions, contacts)
        assert n == 1
        assert actions[1].player_track_id == 3

    def test_no_change_when_different_player(self) -> None:
        """No change when receive already has different player."""
        actions = [
            self._ca(ActionType.SERVE, 10, player=1, court_side="near"),
            self._ca(ActionType.RECEIVE, 40, player=3, court_side="far"),
        ]
        contacts = [
            _contact(10, player_track_id=1),
            _contact(40, player_track_id=3),
        ]
        contacts[1].player_candidates = [(3, 0.05), (1, 0.10)]

        n = _reattribute_server_exclusion(actions, contacts)
        assert n == 0
        assert actions[1].player_track_id == 3

    def test_works_without_team_data(self) -> None:
        """Server exclusion works via reattribute_players even without teams."""
        actions = [
            self._ca(ActionType.SERVE, 10, player=1, court_side="near"),
            self._ca(ActionType.RECEIVE, 40, player=1, court_side="far"),
        ]
        contacts = [
            _contact(10, player_track_id=1),
            _contact(40, player_track_id=1),
        ]
        contacts[1].player_candidates = [(1, 0.05), (3, 0.08)]

        # Call reattribute_players with no team data — server exclusion still fires
        reattribute_players(actions, contacts, team_assignments=None)
        assert actions[1].player_track_id == 3


class TestViterbiDecoding:
    """Tests for Viterbi sequence decoding."""

    @staticmethod
    def _ca(
        action_type: ActionType,
        frame: int,
        court_side: str = "near",
        confidence: float = 0.7,
    ) -> ClassifiedAction:
        return ClassifiedAction(
            action_type=action_type,
            frame=frame,
            ball_x=0.5,
            ball_y=0.5,
            velocity=0.02,
            player_track_id=-1,
            court_side=court_side,
            confidence=confidence,
        )

    def test_no_change_on_valid_sequence(self) -> None:
        """Valid sequence should not be modified."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near", 0.9),
            self._ca(ActionType.RECEIVE, 30, "far", 0.9),
            self._ca(ActionType.SET, 40, "far", 0.9),
            self._ca(ActionType.ATTACK, 50, "far", 0.9),
        ]
        result = viterbi_decode_actions(actions)
        assert [a.action_type for a in result] == [
            ActionType.SERVE, ActionType.RECEIVE,
            ActionType.SET, ActionType.ATTACK,
        ]

    def test_fixes_attack_set_confusion(self) -> None:
        """Low-confidence attack after receive should become set."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near", 0.9),
            self._ca(ActionType.RECEIVE, 30, "far", 0.9),
            # Misclassified: should be set, not attack (low confidence)
            self._ca(ActionType.ATTACK, 40, "far", 0.3),
            self._ca(ActionType.ATTACK, 50, "far", 0.8),
        ]
        result = viterbi_decode_actions(actions)
        # Viterbi should prefer set→attack over attack→attack
        assert result[2].action_type == ActionType.SET
        assert result[3].action_type == ActionType.ATTACK

    def test_high_confidence_not_relabeled_with_cap(self) -> None:
        """With an explicit confidence cap, high-confidence actions are preserved."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near", 0.9),
            self._ca(ActionType.RECEIVE, 30, "far", 0.9),
            # High-confidence attack after receive — Viterbi prefers set here
            # but confidence cap (0.65) should prevent relabeling
            self._ca(ActionType.ATTACK, 40, "far", 0.8),
            self._ca(ActionType.ATTACK, 50, "far", 0.8),
        ]
        result = viterbi_decode_actions(actions, confidence_cap=0.65)
        # Both attacks should stay — confidence 0.8 > 0.65 cap
        assert result[2].action_type == ActionType.ATTACK
        assert result[3].action_type == ActionType.ATTACK

    def test_default_cap_allows_relabeling(self) -> None:
        """Default confidence_cap=1.0 allows Viterbi to relabel any contact."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near", 0.9),
            self._ca(ActionType.RECEIVE, 30, "far", 0.9),
            self._ca(ActionType.ATTACK, 40, "far", 0.8),
            self._ca(ActionType.ATTACK, 50, "far", 0.8),
        ]
        result = viterbi_decode_actions(actions)
        # Viterbi prefers RECEIVE→SET→ATTACK over RECEIVE→ATTACK→ATTACK
        assert result[2].action_type == ActionType.SET
        assert result[3].action_type == ActionType.ATTACK

    def test_preserves_serve_receive(self) -> None:
        """Serve and receive should never be relabeled."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near", 0.5),
            self._ca(ActionType.RECEIVE, 30, "far", 0.5),
            self._ca(ActionType.DIG, 50, "near", 0.5),
        ]
        result = viterbi_decode_actions(actions)
        assert result[0].action_type == ActionType.SERVE
        assert result[1].action_type == ActionType.RECEIVE

    def test_single_relabel_position(self) -> None:
        """With only one relabel-eligible action, return unchanged."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near", 0.9),
            self._ca(ActionType.RECEIVE, 30, "far", 0.9),
            self._ca(ActionType.SET, 40, "far", 0.7),
        ]
        result = viterbi_decode_actions(actions)
        # Only one relabel position — Viterbi can't improve
        assert len(result) == 3


class TestValidateActionSequence:
    """Tests for action sequence validation."""

    @staticmethod
    def _ca(
        action_type: ActionType,
        frame: int,
        court_side: str = "near",
    ) -> ClassifiedAction:
        return ClassifiedAction(
            action_type=action_type,
            frame=frame,
            ball_x=0.5,
            ball_y=0.5,
            velocity=0.02,
            player_track_id=-1,
            court_side=court_side,
            confidence=0.8,
        )

    def test_valid_sequence_returns_unchanged(self) -> None:
        """Valid sequence should be returned unchanged."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near"),
            self._ca(ActionType.RECEIVE, 30, "far"),
            self._ca(ActionType.SET, 40, "far"),
            self._ca(ActionType.ATTACK, 50, "far"),
        ]
        result = validate_action_sequence(actions, "test")
        assert result is actions

    def test_returns_actions_unchanged(self) -> None:
        """Validation should never modify the action list."""
        actions = [
            self._ca(ActionType.RECEIVE, 10, "near"),  # Missing serve
            self._ca(ActionType.SET, 20, "near"),
        ]
        result = validate_action_sequence(actions, "test")
        assert result is actions

    def test_consecutive_net_crossings_detected(self) -> None:
        """Consecutive net-crossing actions are detected but actions returned."""
        actions = [
            self._ca(ActionType.SERVE, 10, "near"),
            self._ca(ActionType.ATTACK, 30, "far"),  # Two crossings in a row
        ]
        result = validate_action_sequence(actions, "test")
        assert len(result) == 2  # Still returned unchanged


def _player_pos(
    frame: int, track_id: int, y: float, x: float = 0.5,
    width: float = 0.05, height: float = 0.15,
) -> PlayerPosition:
    """Helper to create a PlayerPosition."""
    return PlayerPosition(
        frame_number=frame, track_id=track_id,
        x=x, y=y, width=width, height=height, confidence=0.9,
    )


class TestFindServingSideByFormation:
    """Tests for _find_serving_side_by_formation."""

    def test_auto_split_uses_windowed_positions(self) -> None:
        """Auto-split should use positions from the formation window, not full rally."""
        positions: list[PlayerPosition] = []
        # Formation window (frames 0-60): clear 2+2 split
        for f in range(60):
            positions.append(_player_pos(f, 1, y=0.70))  # near
            positions.append(_player_pos(f, 2, y=0.60))  # near
            positions.append(_player_pos(f, 3, y=0.30))  # far
            positions.append(_player_pos(f, 4, y=0.20))  # far
        # Later frames (100-200): all players move to near side
        for f in range(100, 200):
            positions.append(_player_pos(f, 1, y=0.80))
            positions.append(_player_pos(f, 2, y=0.70))
            positions.append(_player_pos(f, 3, y=0.65))
            positions.append(_player_pos(f, 4, y=0.60))

        # With net_y=0.2, all windowed players are "above" it.
        # auto_split should use windowed positions and find 2+2 split.
        side, conf = _find_serving_side_by_formation(
            positions, net_y=0.2, start_frame=0, window_frames=60,
        )
        assert side is not None, "Should not abstain — 4 players clearly 2+2 in window"


class TestFindServerByPosition:
    """Tests for position-based server detection."""

    def test_both_sides_ambiguous_no_detection(self) -> None:
        """When teammates on each side have tiny separation, no detection."""
        # Near: both at nearly identical distance from net (sep < 0.01)
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.600))  # near, foot=0.675
            positions.append(_player_pos(f, 2, y=0.605))  # near, foot=0.680
            positions.append(_player_pos(f, 3, y=0.395))  # far, foot=0.470
            positions.append(_player_pos(f, 4, y=0.400))  # far, foot=0.475
        tid, side, conf = _find_server_by_position(positions, 0, 0.5)
        assert tid == -1  # Separation=0.005 < 0.01 on both sides

    def test_server_at_near_baseline(self) -> None:
        """Player right at the near baseline is detected."""
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.60))  # near, foot=0.675
            positions.append(_player_pos(f, 2, y=0.80))  # near, foot=0.875 (far from net)
            positions.append(_player_pos(f, 3, y=0.35))  # far, foot=0.425
            positions.append(_player_pos(f, 4, y=0.40))  # far, foot=0.475
        tid, side, conf = _find_server_by_position(positions, 0, 0.5)
        # Near side: track 2 dist=0.375, track 1 dist=0.175, sep=0.20
        assert tid == 2
        assert side == "near"
        assert conf > 0.0

    def test_server_at_far_baseline(self) -> None:
        """Player right at the far baseline is detected."""
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.60))  # near, foot=0.675
            positions.append(_player_pos(f, 2, y=0.62))  # near, foot=0.695
            positions.append(_player_pos(f, 3, y=0.38))  # far, foot=0.455
            positions.append(_player_pos(f, 4, y=0.15))  # far, foot=0.225 (far from net)
        tid, side, conf = _find_server_by_position(positions, 0, 0.5)
        # Far side: track 4 dist=0.275, track 3 dist=0.045, sep=0.23
        assert tid == 4
        assert side == "far"

    def test_all_players_midcourt_no_detection(self) -> None:
        """When all players are mid-court with tiny separation, no detection."""
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.530))  # near, foot=0.605
            positions.append(_player_pos(f, 2, y=0.535))  # near, foot=0.610
            positions.append(_player_pos(f, 3, y=0.465))  # far, foot=0.540
            positions.append(_player_pos(f, 4, y=0.470))  # far, foot=0.545
        tid, side, conf = _find_server_by_position(positions, 0, 0.5)
        assert tid == -1  # Separation=0.005 < 0.01

    def test_two_players_near_same_baseline_no_detection(self) -> None:
        """When teammates on both sides have tiny separation, no detection."""
        positions = []
        for f in range(45):
            # Near: both similarly far from net (sep=0.005)
            positions.append(_player_pos(f, 1, y=0.620))  # foot=0.695
            positions.append(_player_pos(f, 2, y=0.625))  # foot=0.700
            # Far: both similarly far from net (sep=0.005)
            positions.append(_player_pos(f, 3, y=0.375))  # foot=0.450
            positions.append(_player_pos(f, 4, y=0.380))  # foot=0.455
        tid, side, conf = _find_server_by_position(positions, 0, 0.5)
        assert tid == -1  # Separation=0.005 < 0.01 on both sides

    def test_no_positions_returns_nothing(self) -> None:
        """Empty player positions returns no server."""
        tid, side, conf = _find_server_by_position([], 0, 0.5)
        assert tid == -1

    def test_window_filters_late_frames(self) -> None:
        """Only positions within the window are considered."""
        positions = []
        # First 60 frames (default window): both near side, tiny separation
        for f in range(60):
            positions.append(_player_pos(f, 1, y=0.530))  # foot=0.605
            positions.append(_player_pos(f, 2, y=0.535))  # foot=0.610
        # Frame 70+: track 1 moves far from net (outside default 60-frame window)
        for f in range(70, 90):
            positions.append(_player_pos(f, 1, y=0.80))
            positions.append(_player_pos(f, 2, y=0.535))
        tid, side, conf = _find_server_by_position(positions, 0, 0.5)
        assert tid == -1  # Late positions outside window are ignored


class TestFindServerByPositionCourtSpace:
    """Tests for court-space server detection with calibrator."""

    @staticmethod
    def _make_calibrator() -> CourtCalibrator:
        """Create a calibrator with a simple linear mapping.

        Camera convention: bottom of frame (large y) = near court (small court-y).
        Maps image coords linearly to 8m×16m court.
        """
        calibrator = CourtCalibrator()
        # near-left at bottom-left, far-right at top-right
        image_corners = [
            (0.0, 1.0),   # near-left  → court (0, 0)
            (1.0, 1.0),   # near-right → court (8, 0)
            (1.0, 0.0),   # far-right  → court (8, 16)
            (0.0, 0.0),   # far-left   → court (0, 16)
        ]
        calibrator.calibrate(image_corners)
        return calibrator

    def test_far_side_server_detected_in_court_space(self) -> None:
        """Far-side players close in image-space but separated in court-space.

        In image space, 0.05 separation is barely above separation_min=0.04
        after foot offset.  But in court space the server is at the far
        baseline (~14m) vs teammate near the net (~9m) — 5m separation.
        """
        # Perspective-like mapping: far court compressed into small image region
        # Near corners at image y=0.55..0.95, far corners at image y=0.05..0.15
        calibrator = CourtCalibrator()
        calibrator.calibrate([
            (0.05, 0.95),  # near-left  → court (0, 0)
            (0.95, 0.95),  # near-right → court (8, 0)
            (0.65, 0.10),  # far-right  → court (8, 16)
            (0.35, 0.10),  # far-left   → court (0, 16)
        ])
        positions = []
        for f in range(45):
            # Near side: clear server at baseline
            positions.append(_player_pos(f, 1, y=0.55, x=0.3))  # near, near net
            positions.append(_player_pos(f, 2, y=0.80, x=0.7))  # near, at baseline
            # Far side: both in a small image-Y band (compressed)
            positions.append(_player_pos(f, 3, y=0.22, x=0.45))  # far, near net
            positions.append(_player_pos(f, 4, y=0.08, x=0.50))  # far, at baseline
        tid, side, conf = _find_server_by_position(
            positions, 0, 0.50, calibrator=calibrator,
        )
        # Court-space should detect a server (either near or far side)
        assert tid != -1
        assert side in ("near", "far")
        assert conf > 0.0

    def test_near_side_server_agrees_with_image_space(self) -> None:
        """Near-side server detected in court-space agrees with image-space."""
        calibrator = self._make_calibrator()
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.60, x=0.3))  # near-ish
            positions.append(_player_pos(f, 2, y=0.80, x=0.7))  # near baseline
            positions.append(_player_pos(f, 3, y=0.35, x=0.4))  # far
            positions.append(_player_pos(f, 4, y=0.40, x=0.6))  # far
        # Without calibrator (image-space)
        tid_img, side_img, _ = _find_server_by_position(positions, 0, 0.5)
        # With calibrator (court-space)
        tid_court, side_court, _ = _find_server_by_position(
            positions, 0, 0.5, calibrator=calibrator,
        )
        assert tid_img == tid_court
        assert side_img == side_court

    def test_fallback_when_no_calibrator(self) -> None:
        """Without calibrator, behaves identically to image-space."""
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.60))
            positions.append(_player_pos(f, 2, y=0.80))
            positions.append(_player_pos(f, 3, y=0.35))
            positions.append(_player_pos(f, 4, y=0.40))
        tid_none, side_none, conf_none = _find_server_by_position(
            positions, 0, 0.5, calibrator=None,
        )
        tid_default, side_default, conf_default = _find_server_by_position(
            positions, 0, 0.5,
        )
        assert tid_none == tid_default
        assert side_none == side_default
        assert conf_none == conf_default

    def test_fallback_on_bad_projection(self) -> None:
        """When calibrator produces out-of-bounds projections, falls back."""
        # Create a degenerate calibrator that maps everything to crazy values
        # by using very tight image corners
        calibrator = CourtCalibrator()
        calibrator.calibrate([
            (0.499, 0.499),
            (0.501, 0.499),
            (0.501, 0.501),
            (0.499, 0.501),
        ])
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.60, x=0.3))
            positions.append(_player_pos(f, 2, y=0.80, x=0.7))
            positions.append(_player_pos(f, 3, y=0.35, x=0.4))
            positions.append(_player_pos(f, 4, y=0.15, x=0.5))
        # Court-space projection gives wild values → should fall back to image-space
        tid_bad, side_bad, conf_bad = _find_server_by_position(
            positions, 0, 0.5, calibrator=calibrator,
        )
        # Compare against no-calibrator (pure image-space) result
        tid_none, side_none, conf_none = _find_server_by_position(
            positions, 0, 0.5, calibrator=None,
        )
        assert tid_bad == tid_none
        assert side_bad == side_none
        assert conf_bad == conf_none

    def test_court_space_ambiguous_both_sides(self) -> None:
        """Both sides have small court-space separation → no detection."""
        calibrator = self._make_calibrator()
        # Place all players near the net in court space
        # With linear mapping: y=0.45 → court_y=7.2, y=0.55 → court_y=8.8
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.42, x=0.3))  # court~6.7
            positions.append(_player_pos(f, 2, y=0.48, x=0.7))  # court~7.7
            positions.append(_player_pos(f, 3, y=0.52, x=0.4))  # court~8.3
            positions.append(_player_pos(f, 4, y=0.58, x=0.6))  # court~9.3
        tid, side, conf = _find_server_by_position(
            positions, 0, 0.5, calibrator=calibrator,
        )
        # Court-space separations: near ~0.96m, far ~0.08m — both < 1.0m min
        assert tid == -1


class TestPositionBasedServePass:
    """Integration tests for Pass 0 in _find_serve_index."""

    def test_serve_found_by_track_id(self) -> None:
        """Pass 0 finds serve contact matching server's track_id."""
        positions = []
        for f in range(45):
            positions.append(_player_pos(f, 1, y=0.80))  # server (far from net)
            positions.append(_player_pos(f, 2, y=0.55))
            positions.append(_player_pos(f, 3, y=0.30))
            positions.append(_player_pos(f, 4, y=0.40))

        contacts = [
            _contact(10, ball_y=0.80, player_track_id=1, court_side="near"),
            _contact(40, ball_y=0.50, player_track_id=2, court_side="near"),
        ]
        # Server is track 1 (detected from positions)
        classifier = ActionClassifier()
        idx, pass_num = classifier._find_serve_index(
            contacts, 0, 0.5, server_pos_tid=1,
        )
        assert idx == 0
        assert pass_num == 0

    def test_no_matching_track_falls_through(self) -> None:
        """Pass 0 falls through when no contact has the server's track_id."""
        contacts = [
            _contact(10, ball_y=0.75, player_track_id=2, court_side="near"),
            _contact(40, ball_y=0.40, player_track_id=3, court_side="far"),
        ]
        classifier = ActionClassifier()
        # Server is track 1 but no contact has that track_id
        idx, pass_num = classifier._find_serve_index(
            contacts, 0, 0.5, server_pos_tid=1,
        )
        assert pass_num != 0

    def test_falls_through_without_server(self) -> None:
        """Without server_pos_tid, Pass 0 is skipped."""
        contacts = [
            _contact(10, ball_y=0.85, velocity=0.03, court_side="near"),
        ]
        classifier = ActionClassifier()
        idx, pass_num = classifier._find_serve_index(
            contacts, 0, 0.5,
        )
        # Should use Pass 2 (baseline position)
        assert pass_num != 0


class TestSyntheticServeAttribution:
    """Tests for synthetic serve server_track_id and team chain."""

    def test_synthetic_serve_with_server_track_id(self) -> None:
        """Synthetic serve carries server_track_id and higher confidence."""
        serve = _make_synthetic_serve("near", 50, 0.5, server_track_id=5)
        assert serve.player_track_id == 5
        assert serve.confidence == 0.55
        assert serve.is_synthetic is True

    def test_synthetic_serve_without_server_track_id(self) -> None:
        """Default synthetic serve has player_track_id=-1 and low confidence."""
        serve = _make_synthetic_serve("near", 50, 0.5)
        assert serve.player_track_id == -1
        assert serve.confidence == 0.4
        assert serve.is_synthetic is True

    def test_synthetic_serve_to_dict_with_attribution(self) -> None:
        """Synthetic serve with real player_track_id serializes correctly."""
        serve = _make_synthetic_serve("near", 30, 0.5, server_track_id=3)
        d = serve.to_dict()
        assert d["isSynthetic"] is True
        assert d["playerTrackId"] == 3
        assert d["confidence"] == 0.55

    def test_repair_rule0_carries_server_track_id(self) -> None:
        """Rule 0 synthetic serve inherits server_track_id."""
        # Serve on "near" but ball_y suggests it's on far side (wrong side)
        actions = [
            ClassifiedAction(
                ActionType.SERVE, 10, 0.5, 0.2, 0.03, 5, "near", 0.8,
            ),
            ClassifiedAction(
                ActionType.RECEIVE, 25, 0.5, 0.7, 0.02, 3, "far", 0.7,
            ),
        ]
        repaired = repair_action_sequence(
            actions, net_y=0.5, server_track_id=5,
        )
        # Rule 0 should prepend synthetic serve on "far" side
        assert repaired[0].is_synthetic is True
        assert repaired[0].action_type == ActionType.SERVE
        assert repaired[0].player_track_id == 5
        assert repaired[0].court_side == "far"
        # Original serve reclassified as receive
        assert repaired[1].action_type == ActionType.RECEIVE
        assert repaired[1].is_synthetic is False

    def test_repair_rule0_without_server_track_id(self) -> None:
        """Rule 0 synthetic serve defaults to player_track_id=-1."""
        actions = [
            ClassifiedAction(
                ActionType.SERVE, 10, 0.5, 0.2, 0.03, 5, "near", 0.8,
            ),
            ClassifiedAction(
                ActionType.RECEIVE, 25, 0.5, 0.7, 0.02, 3, "far", 0.7,
            ),
        ]
        # No server_track_id → default -1
        repaired = repair_action_sequence(actions, net_y=0.5)
        assert repaired[0].is_synthetic is True
        assert repaired[0].player_track_id == -1
        assert repaired[0].confidence == 0.4

    def test_compute_expected_teams_with_synthetic_serve(self) -> None:
        """Team chain flips correctly after synthetic serve."""
        actions = [
            _make_synthetic_serve("near", 0, 0.5, server_track_id=1),
            ClassifiedAction(
                ActionType.RECEIVE, 20, 0.5, 0.7, 0.02, 2, "far", 0.7,
            ),
            ClassifiedAction(
                ActionType.SET, 30, 0.5, 0.7, 0.01, 3, "far", 0.7,
            ),
            ClassifiedAction(
                ActionType.ATTACK, 40, 0.5, 0.3, 0.03, 3, "far", 0.8,
            ),
            ClassifiedAction(
                ActionType.DIG, 55, 0.5, 0.7, 0.02, 1, "near", 0.7,
            ),
        ]
        # Team 0 = near (server's team), team 1 = far (receiver's team)
        teams = {1: 0, 2: 1, 3: 1}
        expected = _compute_expected_teams(actions, teams)
        # Serve is team 0 (server's team)
        assert expected[0] == 0
        # After serve crosses net → receive on team 1
        assert expected[1] == 1
        # Set stays on team 1
        assert expected[2] == 1
        # Attack is on team 1 (before flip)
        assert expected[3] == 1
        # After attack crosses net → dig on team 0
        assert expected[4] == 0


class TestFindServingTeamByFormation:
    """Tests for formation-based serving team detection."""

    def _formation_positions(
        self,
        near_ys: tuple[float, float],
        far_ys: tuple[float, float],
        n_frames: int = 120,
    ) -> list[PlayerPosition]:
        """Build a formation with 2 near + 2 far players at given foot Ys.

        `_player_pos` produces foot Y = y + 0.075 (height/2 = 0.075). So
        to get a foot Y of 0.80, pass y=0.725.
        """
        positions = []
        for f in range(n_frames):
            positions.append(_player_pos(f, 1, y=near_ys[0] - 0.075))
            positions.append(_player_pos(f, 2, y=near_ys[1] - 0.075))
            positions.append(_player_pos(f, 3, y=far_ys[0] - 0.075))
            positions.append(_player_pos(f, 4, y=far_ys[1] - 0.075))
        return positions

    def test_near_serves_larger_separation(self) -> None:
        """Near side has big separation (server+net partner) → team A."""
        # Near: 0.90 (baseline) + 0.55 (net partner) → sep=0.35
        # Far:  0.42 + 0.45 (both mid-court)          → sep=0.03
        positions = self._formation_positions(
            near_ys=(0.90, 0.55), far_ys=(0.42, 0.45),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}  # team 0 on near, team 1 on far
        team, conf = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5, team_assignments=teams,
        )
        assert team == "A"
        assert conf > 0.0

    def test_far_serves_larger_separation(self) -> None:
        """Far side has big separation → team B."""
        # Near: mid-court cluster → sep=0.03
        # Far: 0.10 (baseline) + 0.45 (net partner) → sep=0.35
        positions = self._formation_positions(
            near_ys=(0.55, 0.58), far_ys=(0.10, 0.45),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}
        team, conf = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5, team_assignments=teams,
        )
        assert team == "B"
        assert conf > 0.0

    def test_ambiguous_separations_low_confidence(self) -> None:
        """Similar separations on both sides → low confidence prediction.

        The multi-feature model uses isolation, baseline proximity, etc.
        beyond just separation, so it may still predict with low confidence
        even when separations are similar.
        """
        positions = self._formation_positions(
            near_ys=(0.60, 0.65), far_ys=(0.35, 0.40),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}
        team, confidence = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5, team_assignments=teams,
        )
        # Multi-feature model may predict but with low confidence
        if team is not None:
            assert confidence < 0.5

    def test_team_flipped_far_serves_returns_team_0(self) -> None:
        """When team assignment is flipped (team 0 on far), far-serving
        returns team 'A'. This is the sideSwitch case — the heuristic
        picks the serving SIDE, and team_assignments maps side→team."""
        positions = self._formation_positions(
            near_ys=(0.55, 0.58), far_ys=(0.10, 0.45),
        )
        # Flipped: team 0 = far, team 1 = near
        teams = {1: 1, 2: 1, 3: 0, 4: 0}
        team, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5, team_assignments=teams,
        )
        # Far side serves, team 0 is on far → team "A"
        assert team == "A"

    def test_no_team_assignments_returns_none(self) -> None:
        """Without team_assignments, cannot map side → team."""
        positions = self._formation_positions(
            near_ys=(0.90, 0.55), far_ys=(0.42, 0.45),
        )
        team, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5, team_assignments=None,
        )
        assert team is None

    def test_broken_net_y_auto_recomputes_split(self) -> None:
        """Stored net_y that misclassifies all players triggers auto-split.

        This is the video 0a383519 case: stored court_split_y=0.43 put all
        4 players on 'near'. Auto-split should detect the real cluster gap.
        """
        # All 4 players are above net_y=0.4 (all classified "near")
        # True near cluster at foot=0.60 and 0.80; far cluster at 0.45 and 0.50
        positions = self._formation_positions(
            near_ys=(0.80, 0.60), far_ys=(0.45, 0.50),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}
        team, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.40,  # Broken split
            team_assignments=teams,
        )
        # Auto-split finds the real gap between 0.50 and 0.60.
        # Near sep = 0.80 - 0.60 = 0.20 (big); far sep = 0.50 - 0.45 = 0.05
        # Near side serves → team A
        assert team == "A"

    def test_window_excludes_late_frames(self) -> None:
        """Late frames beyond window_frames are ignored.

        Formation at frames 0-120 says near serves, but if we set a tiny
        window we only see initial frames — result should still be valid.
        Here, formation is consistent across all frames so result matches.
        """
        positions = self._formation_positions(
            near_ys=(0.90, 0.55), far_ys=(0.42, 0.45),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}
        team, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5,
            team_assignments=teams, window_frames=30,
        )
        assert team == "A"

    def test_graduated_confidence(self) -> None:
        """Multi-feature model predicts with graduated confidence.

        Near has clear server-at-baseline formation (sep=0.30, player near
        bottom of frame). Far side is compact (sep=0.05). Model predicts
        near serving with moderate confidence.
        """
        # Near: server at 0.85, partner at 0.55 (sep=0.30)
        # Far: both mid-court at 0.40 & 0.35 (sep=0.05)
        positions = self._formation_positions(
            near_ys=(0.85, 0.55), far_ys=(0.40, 0.35),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}
        team, conf = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5,
            team_assignments=teams,
        )
        assert team == "A"
        assert conf > 0

    def test_too_few_players_returns_none(self) -> None:
        """With <2 tracked players, cannot compute separation."""
        positions = [_player_pos(f, 1, y=0.7) for f in range(60)]
        team, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5, team_assignments={1: 0},
        )
        assert team is None

    def test_track_to_player_fallback_when_team_assignments_none(self) -> None:
        """When team_assignments is None, fall back to track_to_player
        (semantic via player IDs: 1-2=A, 3-4=B)."""
        positions = self._formation_positions(
            near_ys=(0.90, 0.55), far_ys=(0.42, 0.45),
        )
        # Tracks 1,2 are on near with player_ids 1,2 (team A)
        track_to_player = {1: 1, 2: 2, 3: 3, 4: 4}
        team, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5,
            team_assignments=None,
            track_to_player=track_to_player,
        )
        # Near serves, tracks on near are players 1-2 → team A
        assert team == "A"

    def test_semantic_flip_inverts_output(self) -> None:
        """`semantic_flip=True` inverts the final A/B output. Used on
        flipped rallies where team_assignments' physical-near convention
        gives the wrong semantic team."""
        positions = self._formation_positions(
            near_ys=(0.90, 0.55), far_ys=(0.42, 0.45),
        )
        teams = {1: 0, 2: 0, 3: 1, 4: 1}
        # Without flip: near serves, near = team 0 → "A"
        team_no_flip, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5,
            team_assignments=teams, semantic_flip=False,
        )
        assert team_no_flip == "A"
        # With flip: near serves but semantic flip inverts → "B"
        team_flipped, _ = _find_serving_team_by_formation(
            positions, start_frame=0, net_y=0.5,
            team_assignments=teams, semantic_flip=True,
        )
        assert team_flipped == "B"


class TestComputeAutoSplitY:
    """Tests for auto-split Y recomputation."""

    def test_gap_between_two_clusters(self) -> None:
        """Two clear player clusters → split at the gap."""
        positions = []
        for f in range(10):
            # Near cluster: foot=0.75, 0.80
            positions.append(_player_pos(f, 1, y=0.675))
            positions.append(_player_pos(f, 2, y=0.725))
            # Far cluster: foot=0.40, 0.45
            positions.append(_player_pos(f, 3, y=0.325))
            positions.append(_player_pos(f, 4, y=0.375))
        split = _compute_auto_split_y(positions)
        assert split is not None
        # Gap is between 0.45 and 0.75 → split ≈ 0.60
        assert 0.55 < split < 0.65

    def test_too_few_tracks_returns_none(self) -> None:
        """Fewer than 3 tracks → None."""
        positions = [
            _player_pos(0, 1, y=0.3),
            _player_pos(0, 2, y=0.7),
        ]
        assert _compute_auto_split_y(positions) is None

    def test_ignores_negative_track_ids(self) -> None:
        """Tracks with track_id < 0 are excluded."""
        positions = [
            _player_pos(0, -1, y=0.1),
            _player_pos(0, -1, y=0.9),
            _player_pos(0, 1, y=0.3),
            _player_pos(0, 2, y=0.7),
        ]
        assert _compute_auto_split_y(positions) is None


class TestClassifyServeContact:
    """Tests for _classify_serve_contact."""

    def test_serve_ball_high_not_at_net(self) -> None:
        """Ball above net (low Y from toss), not at net, player reaching → serve."""
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=1, player_distance=0.06,
            is_at_net=False, court_side="far",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is True

    def test_receive_ball_below_net(self) -> None:
        """Ball below net (high Y, near side) → receive."""
        contact = Contact(
            frame=80, ball_x=0.5, ball_y=0.7,
            velocity=0.02, direction_change_deg=130.0,
            player_track_id=2, player_distance=0.02,
            is_at_net=False, court_side="near",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is False

    def test_at_net_not_serve(self) -> None:
        """Contact at net with ball high → uncertain."""
        contact = Contact(
            frame=60, ball_x=0.5, ball_y=0.4,
            velocity=0.015, direction_change_deg=140.0,
            player_track_id=3, player_distance=0.02,
            is_at_net=True, court_side="near",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is None

    def test_player_too_close_uncertain(self) -> None:
        """Ball high but player very close → uncertain."""
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=1, player_distance=0.01,
            is_at_net=False, court_side="far",
        )
        result = _classify_serve_contact(contact, net_y=0.5)
        assert result is None

    def test_serving_side_from_serve_contact(self) -> None:
        """Serve contact → player's side is serving side."""
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=1, player_distance=0.06,
            is_at_net=False, court_side="far",
        )
        # Player 1 on near side (foot Y = 0.70 + 0.075 = 0.775 > 0.5)
        positions = [_player_pos(f, 1, y=0.70) for f in range(60)]
        positions += [_player_pos(f, 3, y=0.30) for f in range(60)]

        side, conf = _serving_side_from_contact(contact, positions, net_y=0.5)
        assert side == "near"
        assert conf > 0

    def test_serving_side_from_receive_contact(self) -> None:
        """Receive contact → opposite of player's side."""
        contact = Contact(
            frame=80, ball_x=0.5, ball_y=0.7,
            velocity=0.02, direction_change_deg=130.0,
            player_track_id=3, player_distance=0.02,
            is_at_net=False, court_side="near",
        )
        # Player 3 on far side (foot Y = 0.30 + 0.075 = 0.375 < 0.5)
        positions = [_player_pos(f, 1, y=0.70) for f in range(90)]
        positions += [_player_pos(f, 3, y=0.30) for f in range(90)]

        side, conf = _serving_side_from_contact(contact, positions, net_y=0.5)
        assert side == "near"  # player 3 on far received → near served
        assert conf > 0

    def test_serving_side_no_player_positions(self) -> None:
        """No positions for the contact player → None."""
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=99, player_distance=0.06,
            is_at_net=False, court_side="far",
        )
        side, conf = _serving_side_from_contact(contact, [], net_y=0.5)
        assert side is None

    def test_fusion_contact_overrides_weak_formation(self) -> None:
        """Contact classifier overrides formation when formation conf is low."""
        # Ambiguous formation: both sides nearly equal separation
        positions: list[PlayerPosition] = []
        for f in range(120):
            positions.append(_player_pos(f, 1, y=0.545))  # near, foot=0.62
            positions.append(_player_pos(f, 2, y=0.505))  # near, foot=0.58
            positions.append(_player_pos(f, 3, y=0.345))  # far, foot=0.42
            positions.append(_player_pos(f, 4, y=0.305))  # far, foot=0.38
        # Serve contact: ball high, player 3 (far side) is server
        contact = Contact(
            frame=50, ball_x=0.5, ball_y=0.3,
            velocity=0.01, direction_change_deg=150.0,
            player_track_id=3, player_distance=0.06,
            is_at_net=False, court_side="far",
        )
        side, conf = _find_serving_side_by_formation(
            positions, net_y=0.5, first_contact=contact,
        )
        assert side is not None

    def test_fusion_contact_does_not_override_strong_formation(self) -> None:
        """Contact does NOT override high-confidence formation."""
        # Clear formation: near side has server at baseline
        positions: list[PlayerPosition] = []
        for f in range(120):
            positions.append(_player_pos(f, 1, y=0.825))  # near baseline, foot=0.90
            positions.append(_player_pos(f, 2, y=0.475))  # near net, foot=0.55
            positions.append(_player_pos(f, 3, y=0.345))  # far, foot=0.42
            positions.append(_player_pos(f, 4, y=0.305))  # far, foot=0.38
        # Receive contact on near side (ballY > net_y → receive)
        contact = Contact(
            frame=80, ball_x=0.5, ball_y=0.7,
            velocity=0.02, direction_change_deg=130.0,
            player_track_id=1, player_distance=0.02,
            is_at_net=False, court_side="near",
        )
        side, conf = _find_serving_side_by_formation(
            positions, net_y=0.5, first_contact=contact,
        )
        # Formation is confident "near" → contact should NOT override
        assert side == "near"
