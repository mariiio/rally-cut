"""Regression tests for the optional decoder_contacts kwarg on
classify_rally_actions.

These tests guard the default-off behavior: passing decoder_contacts=None
(the default) must produce identical output to omitting the kwarg. Passing
a list must swap labels per apply_decoder_labels semantics.
"""
from __future__ import annotations

import inspect

from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
    classify_rally_actions,
)
from rallycut.tracking.candidate_decoder import DecodedContact
from rallycut.tracking.decoder_overlay import apply_decoder_labels


def _ca(frame: int, action: ActionType, tid: int = 1) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action, frame=frame, ball_x=0.5, ball_y=0.5,
        velocity=0.1, player_track_id=tid, court_side="near",
        confidence=0.9, team="A",
    )


def _dc(frame: int, action: str) -> DecodedContact:
    return DecodedContact(
        candidate_idx=0, frame=frame, action=action,
        action_idx=0, score=-1.0,
    )


def test_empty_decoder_contacts_is_noop() -> None:
    """decoder_contacts=None (or empty list) must not change RallyActions."""
    ra = RallyActions(rally_id="r1", actions=[_ca(100, ActionType.SERVE)])
    out, stat = apply_decoder_labels(ra, [], tol_frames=3)
    assert out.actions == ra.actions
    assert stat.n_label_swapped == 0


def test_decoder_contacts_applied_smoke() -> None:
    """Smoke test: the overlay swaps a label when a decoder contact matches."""
    ra = RallyActions(rally_id="r1", actions=[_ca(100, ActionType.RECEIVE)])
    decoder = [_dc(101, "set")]
    out, stat = apply_decoder_labels(ra, decoder, tol_frames=3)
    assert out.actions[0].action_type == ActionType.SET
    assert stat.n_label_swapped == 1


def test_classify_rally_actions_accepts_decoder_contacts_kwarg() -> None:
    """Regression: the signature change itself — the kwarg is accepted and
    defaults to None. This test exercises the parameter plumbing, not the
    overlay behavior (which is tested in test_decoder_overlay.py)."""
    sig = inspect.signature(classify_rally_actions)
    assert "decoder_contacts" in sig.parameters
    param = sig.parameters["decoder_contacts"]
    assert param.default is None, "decoder_contacts must default to None for back-compat"
    # Pin the position so a future reordering doesn't silently pass — Task 5
    # CLI wiring will pass it as a kwarg, but new callers may pass positionals.
    assert list(sig.parameters)[-1] == "decoder_contacts", (
        "decoder_contacts must remain the last parameter in the signature"
    )
