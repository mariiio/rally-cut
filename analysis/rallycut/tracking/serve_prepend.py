"""Serve-Peak Prepend Synthesis (v1.3).

Pure-predicate gate that decides whether to prepend a synthetic serve at
the MS-TCN++ serve-class peak when the first classified action was
mis-labeled as serve.

Spec: docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md
"""
from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES

SERVE_PREPEND_PEAK_FLOOR: float = 0.95
"""Minimum MS-TCN++ serve-class probability at the peak frame."""

SERVE_PREPEND_MIN_GAP: int = 25
"""Minimum gap (in frames) between the peak and the first classified
action's frame. Filters normal serve-buildup peaks on correctly detected
serves (which sit ~5-15 frames before contact)."""

SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL: float = 0.50
"""If the first classified action's frame has serve-class probability at
or above this, the first action is plausibly a real serve — don't
override."""

SERVE_PREPEND_GUARD_FRAMES: int = 15
"""Search window upper bound is `first_action_frame - GUARD_FRAMES`. This
prevents picking peaks that are part of the first action's buildup."""

# Module flag for the clean A/B harness. Default False = production behavior.
_DISABLE_V13_PREPEND: bool = False


def should_prepend_serve(
    *,
    sequence_probs: np.ndarray | None,
    first_action_frame: int,
    first_action_serve_prob: float,
    rally_start_frame: int,
) -> int | None:
    """Return the peak frame to prepend at, or None if the gate doesn't fire.

    Conditions (all must hold):
      1. `sequence_probs` is not None.
      2. `first_action_serve_prob` < SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL.
      3. The search window [rally_start_frame, first_action_frame - GUARD) is
         non-empty.
      4. The serve-class argmax in that window has prob >= SERVE_PREPEND_PEAK_FLOOR.
      5. `first_action_frame - peak_frame` >= SERVE_PREPEND_MIN_GAP.
    """
    if _DISABLE_V13_PREPEND:
        return None
    if sequence_probs is None:
        return None
    if first_action_serve_prob >= SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL:
        return None
    upper_excl = first_action_frame - SERVE_PREPEND_GUARD_FRAMES
    if upper_excl <= rally_start_frame:
        return None
    serve_idx = ACTION_TYPES.index("serve") + 1
    if serve_idx >= sequence_probs.shape[0]:
        return None
    window = sequence_probs[serve_idx, rally_start_frame:upper_excl]
    if window.size == 0:
        return None
    peak_offset = int(np.argmax(window))
    peak_prob = float(window[peak_offset])
    if peak_prob < SERVE_PREPEND_PEAK_FLOOR:
        return None
    peak_frame = rally_start_frame + peak_offset
    if first_action_frame - peak_frame < SERVE_PREPEND_MIN_GAP:
        return None
    return peak_frame
