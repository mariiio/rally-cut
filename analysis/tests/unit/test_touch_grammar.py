"""Tests for touch grammar constraints."""

from __future__ import annotations

from rallycut.tracking.contact_detector import Contact, ContactSequence
from rallycut.tracking.touch_grammar import (
    GrammarScore,
    score_contact_grammar,
    score_swap_hypothesis,
)


def _make_contact(
    frame: int,
    track_id: int,
    ball_y: float = 0.5,
    side: str = "unknown",
) -> Contact:
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=ball_y,
        velocity=0.02,
        direction_change_deg=30.0,
        player_track_id=track_id,
        court_side=side,
    )


class TestGrammarScore:
    def test_no_violations(self) -> None:
        score = GrammarScore(total_contacts=5)
        assert score.violation_rate == 0.0
        assert score.plausibility == 1.0

    def test_all_violations(self) -> None:
        score = GrammarScore(
            consecutive_violations=2,
            max_touches_violations=2,
            wrong_side_violations=1,
            total_contacts=5,
        )
        assert score.violation_rate == 1.0
        assert score.plausibility == 0.0

    def test_empty_contacts(self) -> None:
        score = GrammarScore(total_contacts=0)
        assert score.violation_rate == 0.0
        assert score.plausibility == 1.0


class TestScoreContactGrammar:
    def test_valid_sequence(self) -> None:
        """Alternating contacts with no violations."""
        contacts = [
            _make_contact(10, track_id=1, ball_y=0.7),  # Near side
            _make_contact(20, track_id=2, ball_y=0.7),  # Near side
            _make_contact(30, track_id=1, ball_y=0.7),  # Near side
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5)
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        score = score_contact_grammar(seq, team_assignments)
        assert score.consecutive_violations == 0
        # 3 touches on one side → 0 max_touches violations (within limit)
        assert score.max_touches_violations == 0

    def test_consecutive_touch_violation(self) -> None:
        """Same player touching twice should be flagged."""
        contacts = [
            _make_contact(10, track_id=1, ball_y=0.7),
            _make_contact(20, track_id=1, ball_y=0.7),  # Same player!
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5)
        team_assignments = {1: 0}

        score = score_contact_grammar(seq, team_assignments)
        assert score.consecutive_violations == 1

    def test_max_touches_violation(self) -> None:
        """More than 3 touches on one side should be flagged."""
        contacts = [
            _make_contact(10, track_id=1, ball_y=0.7),
            _make_contact(20, track_id=2, ball_y=0.7),
            _make_contact(30, track_id=1, ball_y=0.7),
            _make_contact(40, track_id=2, ball_y=0.7),  # 4th touch on near side
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5)
        team_assignments = {1: 0, 2: 0}

        score = score_contact_grammar(seq, team_assignments)
        assert score.max_touches_violations == 1  # 4 - 3 = 1

    def test_wrong_side_violation(self) -> None:
        """Contact by near-team player when ball is on far side."""
        contacts = [
            _make_contact(10, track_id=1, ball_y=0.7),  # First contact normal
            _make_contact(20, track_id=1, ball_y=0.3),  # Ball on far side (wrong)
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5)
        team_assignments = {1: 0}  # Near team

        score = score_contact_grammar(seq, team_assignments)
        assert score.wrong_side_violations == 1

    def test_empty_sequence(self) -> None:
        seq = ContactSequence(contacts=[], net_y=0.5)
        score = score_contact_grammar(seq, {})
        assert score.total_contacts == 0
        assert score.plausibility == 1.0


class TestScoreSwapHypothesis:
    def test_insufficient_contacts(self) -> None:
        """With <2 contacts, should return neutral scores."""
        seq = ContactSequence(
            contacts=[_make_contact(10, 1)],
            net_y=0.5,
        )
        no_swap, swap = score_swap_hypothesis(
            seq, {1: 0, 2: 1}, 1, 2, 20
        )
        assert no_swap == 0.5
        assert swap == 0.5

    def test_swap_reduces_violations(self) -> None:
        """When swapping IDs would reduce violations, swap score should be higher."""
        # Contacts that look wrong: near-team player touching on far side
        contacts = [
            _make_contact(10, track_id=1, ball_y=0.7),  # Near side, correct
            _make_contact(30, track_id=1, ball_y=0.3),  # Far side, wrong for near team
            _make_contact(50, track_id=1, ball_y=0.3),  # Far side, wrong for near team
        ]
        seq = ContactSequence(contacts=contacts, net_y=0.5)
        team_assignments = {1: 0, 2: 1}

        no_swap, swap = score_swap_hypothesis(
            seq, team_assignments, 1, 2, swap_from_frame=25
        )
        # After swap from frame 25: track 1 becomes team 1 (far)
        # So ball_y=0.3 contacts are now on correct side
        # swap should be better
        assert swap >= no_swap
