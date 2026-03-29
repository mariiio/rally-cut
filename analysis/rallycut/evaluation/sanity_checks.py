"""Sanity checks for predicted action sequences.

Detects common-sense violations in volleyball action sequences:
- Time gaps: consecutive contacts separated by >3 seconds mid-rally
- Illegal sequences: same-action repeats on the same side (dig→dig, set→set, etc.)
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass
class SanityViolation:
    rally_id: str
    violation_type: str  # "time_gap" | "same_action_repeat"
    contact_index: int
    description: str


# Actions where same-side consecutive repeats are illegal in volleyball
_ILLEGAL_SAME_SIDE_REPEATS = {"dig", "set", "block"}

# Actions that indicate ball crossed the net (transition to other team)
_SIDE_CHANGING_ACTIONS = {"serve", "attack", "block"}


def check_time_gaps(
    frames: Sequence[int],
    *,
    rally_id: str = "",
    fps: float = 30.0,
    max_gap_s: float = 3.0,
) -> list[SanityViolation]:
    """Flag consecutive contacts separated by more than *max_gap_s* seconds."""
    violations: list[SanityViolation] = []
    max_gap_frames = max_gap_s * fps

    for i in range(1, len(frames)):
        gap = frames[i] - frames[i - 1]
        if gap > max_gap_frames:
            gap_s = gap / fps
            violations.append(SanityViolation(
                rally_id=rally_id,
                violation_type="time_gap",
                contact_index=i,
                description=f"contacts {i - 1}→{i}: {gap_s:.1f}s gap ({gap} frames)",
            ))

    return violations


def _track_sides(actions: Sequence[str]) -> list[int]:
    """Assign a side index (0 or 1) to each contact based on net crossings.

    Serve starts on side 0. Each attack or block flips the side.
    """
    sides: list[int] = []
    side = 0
    for i, action in enumerate(actions):
        if i > 0 and action in _SIDE_CHANGING_ACTIONS:
            # Ball crossed the net before this contact
            if action == "serve":
                side = 0  # serve always resets to side 0
            elif action in ("attack", "block"):
                # Receiving side gets the ball
                pass  # side stays same — the *next* transition flips
        sides.append(side)
        # Flip side after attack/block (ball goes to other side)
        if action in ("attack", "block"):
            side = 1 - side
    return sides


def check_illegal_sequences(
    actions: Sequence[str],
    *,
    rally_id: str = "",
) -> list[SanityViolation]:
    """Flag same-action repeats on the same side of the net."""
    violations: list[SanityViolation] = []
    if len(actions) < 2:
        return violations

    sides = _track_sides(actions)

    for i in range(1, len(actions)):
        action = actions[i].lower()
        prev_action = actions[i - 1].lower()

        if action == prev_action and sides[i] == sides[i - 1]:
            if action in _ILLEGAL_SAME_SIDE_REPEATS:
                violations.append(SanityViolation(
                    rally_id=rally_id,
                    violation_type="same_action_repeat",
                    contact_index=i,
                    description=f"contacts {i - 1}→{i}: {prev_action}→{action} on same side",
                ))

    return violations


def check_all(
    actions: Sequence[str],
    frames: Sequence[int],
    *,
    rally_id: str = "",
    fps: float = 30.0,
    max_gap_s: float = 3.0,
) -> list[SanityViolation]:
    """Run all sanity checks and return combined violations."""
    violations: list[SanityViolation] = []
    violations.extend(check_time_gaps(frames, rally_id=rally_id, fps=fps, max_gap_s=max_gap_s))
    violations.extend(check_illegal_sequences(actions, rally_id=rally_id))
    return violations
