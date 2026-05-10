"""Frame placement for synthetic serves via MS-TCN++ serve-class peak.

Used by `_make_synthetic_serve` to land synthetic serves at the actual
serve frame instead of the placeholder `first_contact_frame - 30`.

Signal: MS-TCN++ serve-class peak in [rally_start, first_contact - SEARCH_GUARD].
Strong iff peak probability >= SERVE_SEQ_FLOOR. Returns None when no peak
exceeds the floor (caller falls back to the legacy formula).

(An earlier design also considered a ball-trajectory direction-change
"burst" signal, but the 2026-05-10 calibration on 194 correctly-placed
real serves found direction-change is not discriminative — pre-rally
non-serve frames routinely exceed the direction-change of real serves.
Dropped from v1.1; revisit with a different trajectory metric in v1.2.)

Spec: docs/superpowers/specs/2026-05-10-synthetic-serve-placement-design.md
"""

from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES

SERVE_SEQ_FLOOR: float = 0.50
SEARCH_GUARD: int = 5
MAX_PRESERVE_FRAMES: int = 150

# Index of "serve" in the seq_probs array (offset by 1 for the bg row).
_SERVE_SEQ_INDEX: int = ACTION_TYPES.index("serve") + 1


def pick_synthetic_serve_frame(
    *,
    sequence_probs: np.ndarray,
    rally_start_frame: int,
    first_contact_frame: int,
) -> int | None:
    """Pick a frame for a synthetic serve from the MS-TCN++ serve peak.

    Returns a frame in `[rally_start, first_contact_frame - SEARCH_GUARD]`
    when the serve-class peak in that window exceeds SERVE_SEQ_FLOOR.
    Returns None otherwise (caller falls back to the legacy placeholder).
    """
    lo = max(0, rally_start_frame)
    hi = first_contact_frame - SEARCH_GUARD
    if hi < lo:
        return None
    if (
        sequence_probs.ndim != 2
        or sequence_probs.shape[0] <= _SERVE_SEQ_INDEX
    ):
        return None
    t = sequence_probs.shape[1]
    hi_clip = min(t - 1, hi)
    if hi_clip < lo:
        return None
    slice_ = sequence_probs[_SERVE_SEQ_INDEX, lo:hi_clip + 1]
    if slice_.size == 0:
        return None
    rel = int(np.argmax(slice_))
    p_seq = float(slice_[rel])
    if p_seq < SERVE_SEQ_FLOOR:
        return None
    picked = lo + rel
    earliest = first_contact_frame - MAX_PRESERVE_FRAMES
    if picked < earliest:
        picked = earliest
    return picked
