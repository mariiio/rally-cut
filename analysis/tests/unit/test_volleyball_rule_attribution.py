"""Unit tests for the A1 volleyball-rule attribution pass.

Spec: docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md
Plan: docs/superpowers/plans/2026-05-13-a1-volleyball-rule-attribution.md
"""

from __future__ import annotations

import pytest


def _make_action(
    frame: int,
    action_type: str,
    player_track_id: int,
    confidence: float = 0.9,
    team: str = "unknown",
):
    """Build a ClassifiedAction with sensible defaults for tests.

    Notes vs. the plan's fixture:
    - ClassifiedAction.team is `str` (not Optional); defaults to "unknown".
    - direction_change_deg is NOT a ClassifiedAction field — that lives on
      Contact. We omit it here.
    """
    from rallycut.tracking.action_classifier import (
        ActionType,
        ClassifiedAction,
    )

    return ClassifiedAction(
        frame=frame,
        action_type=ActionType(action_type),
        confidence=confidence,
        player_track_id=player_track_id,
        court_side="far",
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.01,
        is_synthetic=False,
        team=team,
    )


def _make_contact(frame: int, candidates: list[tuple[int, float]]):
    """Build a Contact with a player_candidates list (tid, distance).

    Notes vs. the plan's fixture:
    - Contact.direction_change_deg is required (no default) — we set 0.0.
    """
    from rallycut.tracking.contact_detector import Contact

    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.01,
        direction_change_deg=0.0,
        player_track_id=candidates[0][0] if candidates else -1,
        player_distance=candidates[0][1] if candidates else float("inf"),
        player_candidates=candidates,
        court_side="far",
    )


class TestVolleyballRulePass:
    """A1: anti-self-touch for SET/RECEIVE/DIG with block exception."""

    def test_cascade_dig_flips_to_same_team_alt(self):
        """Cascade frame 225: attack(p2 B) -> dig(p2 B). Alt p1 (B)
        at 0.041 normalized is within abstention bound. Should flip.
        """
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
        ]
        contacts = [
            _make_contact(176, [(2, 0.052), (3, 0.125), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041), (3, 0.101)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}  # 1=team B, 0=team A

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 1, "one flip expected"
        assert actions[0].player_track_id == 2, "prev attack unchanged"
        assert actions[1].player_track_id == 1, "dig flipped to track 1"
        assert actions[1].attribution_uncertain is False

    def test_cascade_set_flips_to_same_team_alt(self):
        """Cascade frame 128: receive(p2 B) -> set(p2 B). Same shape."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(76, "receive", 2),
            _make_action(128, "set", 2),
        ]
        contacts = [
            _make_contact(76, [(2, 0.020), (1, 0.080), (3, 0.150)]),
            _make_contact(128, [(2, 0.009), (1, 0.087), (3, 0.097)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 1
        assert actions[1].player_track_id == 1

    def test_block_exception_no_flip(self):
        """block -> same-player is legal. No flip."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(100, "block", 2),
            _make_action(110, "dig", 2),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (1, 0.10)]),
            _make_contact(110, [(2, 0.01), (1, 0.04)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0
        assert actions[1].player_track_id == 2  # unchanged

    def test_cross_team_no_flip(self):
        """Cross-team consecutive is not a C-4 violation. No flip."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(100, "attack", 2),
            _make_action(130, "dig", 3),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (1, 0.10)]),
            _make_contact(130, [(3, 0.02), (4, 0.10)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0
        assert actions[1].player_track_id == 3  # unchanged

    def test_abstention_when_alt_too_far(self):
        """No same-team alt within 0.3 normalized -> abstain, mark
        attribution_uncertain=True, don't flip.
        """
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(100, "receive", 2),
            _make_action(130, "set", 2),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (1, 0.40)]),
            _make_contact(130, [(2, 0.01), (1, 0.45)]),  # alt at 0.45 > 0.3
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0, "no flip — abstained"
        assert actions[1].player_track_id == 2  # unchanged
        assert actions[1].attribution_uncertain is True

    def test_no_same_team_alt_at_all_abstains(self):
        """Only 2 candidates and both are the same player -> abstain."""
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(100, "receive", 2),
            _make_action(130, "set", 2),
        ]
        contacts = [
            _make_contact(100, [(2, 0.05), (3, 0.20)]),
            _make_contact(130, [(2, 0.01), (3, 0.20)]),  # 3 is team A
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 0
        assert actions[1].attribution_uncertain is True

    def test_multiple_pairs_in_one_rally(self):
        """Cascade with multiple same-player back-to-back pairs.

        Forward iteration semantics: each pair (i-1, i) is evaluated in
        frame order; the rule flips action[i] (current side) when
        triggered. Flipping i resolves pair (i-1, i) and may eliminate
        violations for pair (i, i+1).

        Sequence: set p2 -> attack p2 -> dig p2 -> set p2.
          Pair (128, 176): same p2, prev=set != block -> flip 176 to p1.
            After: 176 is p1.
          Pair (176, 225): now (p1 attack, p2 dig) -> no violation.
          Pair (225, 276): same p2, prev=dig != block -> flip 276 to p1.
        Total: 2 flips. Final ids: [2, 1, 2, 1].
        """
        from rallycut.tracking.action_classifier import (
            _attribution_volleyball_rule_pass,
        )

        actions = [
            _make_action(128, "set", 2),
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
            _make_action(276, "set", 2),
        ]
        contacts = [
            _make_contact(128, [(2, 0.009), (1, 0.087)]),
            _make_contact(176, [(2, 0.052), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041)]),
            _make_contact(276, [(2, 0.003), (1, 0.103)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        n = _attribution_volleyball_rule_pass(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert n == 2
        assert [a.player_track_id for a in actions] == [2, 1, 2, 1]

    def test_default_off_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When USE_VOLLEYBALL_RULE_ATTRIBUTION is unset or 0, the
        pass is a no-op when called via reattribute_players.
        """
        from rallycut.tracking.action_classifier import reattribute_players

        monkeypatch.delenv(
            "USE_VOLLEYBALL_RULE_ATTRIBUTION", raising=False,
        )
        # Disable the unrelated Pass 2c within-team swap so this test
        # isolates the A1 pass behavior. (Pass 2c also swaps p2->p1 on
        # contact frame 225 because p1 is rank-1 same-team.)
        monkeypatch.setenv("WITHIN_TEAM_PROXIMITY_SWAP", "0")

        actions = [
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
        ]
        contacts = [
            _make_contact(176, [(2, 0.052), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        result = reattribute_players(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        # Without the env flag, dig stays as p2 (the cascade error
        # persists pre-A1).
        assert result[1].player_track_id == 2

    def test_enabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When USE_VOLLEYBALL_RULE_ATTRIBUTION=1, the pass fires."""
        from rallycut.tracking.action_classifier import reattribute_players

        monkeypatch.setenv("USE_VOLLEYBALL_RULE_ATTRIBUTION", "1")

        actions = [
            _make_action(176, "attack", 2),
            _make_action(225, "dig", 2),
        ]
        contacts = [
            _make_contact(176, [(2, 0.052), (1, 0.146)]),
            _make_contact(225, [(2, 0.006), (1, 0.041)]),
        ]
        team_assignments = {1: 1, 2: 1, 3: 0, 4: 0}

        result = reattribute_players(
            actions=actions,
            contacts=contacts,
            team_assignments=team_assignments,
        )

        assert result[1].player_track_id == 1
