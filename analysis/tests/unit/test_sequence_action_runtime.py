"""Tests for rallycut.tracking.sequence_action_runtime.apply_sequence_override.

The override rewrites non-serve action types with MS-TCN++'s per-frame
argmax. Two invariants matter for the 2026-04-14 serve-hygiene workstream:

1. The override never *produces* SERVE. Serve is heuristic-only — a
   contact classified as dig/set/attack must never be reclassified as
   serve by the override, because that breaks the "exactly one serve
   per rally" invariant enforced by classify_rally.
2. Existing SERVE actions are not rewritten. (Already guarded; keep it
   covered as a regression test.)
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
)
from rallycut.tracking.sequence_action_runtime import apply_sequence_override


def _action(
    frame: int,
    action_type: ActionType,
    court_side: str = "near",
    is_synthetic: bool = False,
) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        player_track_id=1,
        court_side=court_side,
        confidence=0.7,
        is_synthetic=is_synthetic,
    )


def _probs_with_peak(n_frames: int, frame: int, class_idx: int) -> np.ndarray:
    """Return (7, n_frames) probs array with class_idx hot at `frame`.

    Index 0 is background; indices 1..6 correspond to
    `trajectory_features.ACTION_TYPES` (serve, receive, set, attack,
    dig, block). `class_idx` here is the 0-based position inside the
    non-background slice — so 0 → serve.
    """
    probs = np.full((7, n_frames), 0.01, dtype=np.float32)
    probs[1 + class_idx, frame] = 0.95
    return probs


class TestApplySequenceOverride:
    def test_override_does_not_produce_serve(self) -> None:
        """A dig contact must never be rewritten to SERVE by the override.

        Serve is heuristic-selected (exactly one per rally). Letting the
        override emit SERVE creates the 'double serve' symptom from the
        2026-04-14 audit.
        """
        rally = RallyActions(rally_id="test")
        rally.actions = [_action(frame=50, action_type=ActionType.DIG)]
        probs = _probs_with_peak(n_frames=100, frame=50, class_idx=0)  # serve peak

        apply_sequence_override(rally, probs)

        assert rally.actions[0].action_type == ActionType.DIG

    def test_override_preserves_existing_serve(self) -> None:
        """An existing SERVE action is never rewritten."""
        rally = RallyActions(rally_id="test")
        rally.actions = [_action(frame=10, action_type=ActionType.SERVE)]
        # Force argmax to "dig" — override must still leave the serve alone.
        probs = _probs_with_peak(n_frames=100, frame=10, class_idx=4)  # dig peak

        apply_sequence_override(rally, probs)

        assert rally.actions[0].action_type == ActionType.SERVE

    def test_override_still_rewrites_non_serve_classes(self) -> None:
        """Sanity: override functions normally for dig↔attack transitions.

        Regression guard — our serve-guard change must not accidentally
        disable the rest of the override pipeline.
        """
        rally = RallyActions(rally_id="test")
        rally.actions = [_action(frame=40, action_type=ActionType.DIG)]
        probs = _probs_with_peak(n_frames=100, frame=40, class_idx=3)  # attack peak

        apply_sequence_override(rally, probs)

        assert rally.actions[0].action_type == ActionType.ATTACK
