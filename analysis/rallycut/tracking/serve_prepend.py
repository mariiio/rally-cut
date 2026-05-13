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
"""MS-TCN++ serve-class probability ceiling at the first action's frame.
If MS-TCN says the first action looks like a serve (>= ceiling), don't
override regardless of peaks elsewhere."""

SERVE_PREPEND_CLASSIFIER_CONF_CEIL: float = 0.50
"""action_classifier confidence ceiling on the first action's serve label.
If the action_classifier (rule-based + GBM, using direction change,
velocity, player proximity, court position, ball position) was confident
in its serve label (>= ceiling), don't override even when MS-TCN
disagrees.

Combined with SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL, the gate requires
DUAL-SIGNAL AGREEMENT before prepending: BOTH MS-TCN AND
action_classifier must indicate the first action is mis-labeled.
Single-model proxies for "first action is mis-labeled as serve" are
fragile when the model itself is wrong (false-positive serve peaks);
requiring agreement between two independent classifiers is more robust.

Empirical bug case (06f0b063 rally 8): real serve at f=94 with
action_classifier confidence 0.65 was overridden by an MS-TCN
false-positive serve peak at f=4 (prob >= 0.95), producing a duplicate
synthetic serve. Adding this ceiling vetoes that override."""

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
    first_action_classifier_confidence: float,
    rally_start_frame: int,
) -> int | None:
    """Return the peak frame to prepend at, or None if the gate doesn't fire.

    Conditions (all must hold):
      1. `sequence_probs` is not None.
      2. `first_action_serve_prob` < SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL
         (MS-TCN says the first action's frame doesn't look like a serve).
      3. `first_action_classifier_confidence` < SERVE_PREPEND_CLASSIFIER_CONF_CEIL
         (action_classifier wasn't confident in its serve label either —
         dual-signal agreement that the label is unreliable).
      4. The search window [rally_start_frame, first_action_frame - GUARD) is
         non-empty.
      5. The serve-class argmax in that window has prob >= SERVE_PREPEND_PEAK_FLOOR.
      6. `first_action_frame - peak_frame` >= SERVE_PREPEND_MIN_GAP.
    """
    if _DISABLE_V13_PREPEND:
        return None
    if sequence_probs is None:
        return None
    if first_action_serve_prob >= SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL:
        return None
    if first_action_classifier_confidence >= SERVE_PREPEND_CLASSIFIER_CONF_CEIL:
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
