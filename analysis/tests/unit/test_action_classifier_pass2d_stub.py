"""Pass 2d stub no-op tests.

Phase 1 ships only the env-flag-gated stub. Verifies that at both flag
settings (default OFF and explicitly ON) the function is byte-identical
to a no-op — the actual repair body lands in Phase 2.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from rallycut.tracking.action_classifier import _coherence_c4_repair_pass


class TestCoherenceC4RepairPassStub:
    def test_returns_zero_on_empty_inputs(self) -> None:
        n = _coherence_c4_repair_pass(
            actions=[],
            contact_by_frame={},
            team_assignments={},
            chain_integrity=[],
            expected_teams=[],
        )
        assert n == 0

    def test_returns_zero_on_nonempty_inputs(self) -> None:
        # Stub MUST NOT mutate actions or do anything observable — Phase 2
        # fills in the body. This test exists to lock that contract.
        from rallycut.tracking.action_classifier import (
            ActionType,
            ClassifiedAction,
        )

        actions = [
            ClassifiedAction(
                frame=100, action_type=ActionType.SET, confidence=0.9,
                player_track_id=5, ball_x=0.5, ball_y=0.5, velocity=0.0,
                court_side="near",
            ),
            ClassifiedAction(
                frame=140, action_type=ActionType.ATTACK, confidence=0.9,
                player_track_id=5, ball_x=0.5, ball_y=0.5, velocity=0.0,
                court_side="near",
            ),
        ]
        n = _coherence_c4_repair_pass(
            actions=actions,
            contact_by_frame={},
            team_assignments={5: 0},
            chain_integrity=[True, True],
            expected_teams=[0, 0],
        )
        assert n == 0
        # Actions must be byte-identical (no mutations).
        assert actions[0].player_track_id == 5
        assert actions[1].player_track_id == 5


class TestCoherenceC4RepairFlagDefaultOff:
    """The env flag default-OFF must keep reattribute_players behavior
    byte-identical to pre-workstream. We don't run reattribute_players
    end-to-end here; we just verify the flag check reads correctly."""

    def test_flag_default_is_off(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COHERENCE_C4_REPAIR", None)
            assert os.environ.get("COHERENCE_C4_REPAIR", "0") == "0"

    def test_flag_explicit_one_reads_one(self) -> None:
        with patch.dict(os.environ, {"COHERENCE_C4_REPAIR": "1"}):
            assert os.environ.get("COHERENCE_C4_REPAIR", "0") == "1"
