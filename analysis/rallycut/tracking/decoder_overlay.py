"""Overlay decoder-chosen action labels onto an existing ClassifiedAction list.

The contract is deliberately narrow: swap action_type only, nothing else.
This preserves all attribution work (playerTrackId, court_side, team,
zones, ...) done by the action_classifier pipeline, while adopting the
decoder's improved action-type labels for frames both layers agree on.

Frames the decoder emits that have NO nearby ClassifiedAction are DROPPED
- we do not add contacts. The current detect_contacts path owns which
candidates become contacts; the decoder only owns their action label.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from rallycut.tracking.action_classifier import ActionType, RallyActions
from rallycut.tracking.candidate_decoder import DecodedContact


@dataclass(frozen=True)
class OverlayStat:
    """Summary of what the overlay did. For observability; callers may ignore."""

    n_decoder_contacts: int
    n_detected_contacts: int
    n_matched: int
    n_label_swapped: int

    def as_dict(self) -> dict[str, int]:
        return {
            "n_decoder_contacts": self.n_decoder_contacts,
            "n_detected_contacts": self.n_detected_contacts,
            "n_matched": self.n_matched,
            "n_label_swapped": self.n_label_swapped,
        }


_DECODER_ACTION_TO_TYPE: dict[str, ActionType] = {
    "serve": ActionType.SERVE,
    "receive": ActionType.RECEIVE,
    "set": ActionType.SET,
    "attack": ActionType.ATTACK,
    "dig": ActionType.DIG,
    "block": ActionType.BLOCK,
}


def apply_decoder_labels(
    rally_actions: RallyActions,
    decoder_contacts: list[DecodedContact],
    tol_frames: int = 3,
) -> tuple[RallyActions, OverlayStat]:
    """Return a new RallyActions where each matched action has its label
    swapped to the decoder's choice.

    Matching is greedy: for each decoder contact in frame order, pick the
    UNMATCHED ClassifiedAction with the smallest |delta frame| <= tol_frames.
    No action is matched twice. Ties (equal |delta frame|) are broken by list
    order - the FIRST matching entry in rally_actions.actions wins.

    Only ``action_type`` is modified on matched entries. ``playerTrackId``,
    ``court_side``, ``team``, ``confidence``, zones, and every other field
    are preserved byte-for-byte via :func:`dataclasses.replace`.

    Decoder contacts with no matching detected action are dropped - we
    never add new entries.
    """
    detected = list(rally_actions.actions)
    matched_idx: set[int] = set()
    swap_count = 0

    # Sort decoder contacts by frame so greedy matching is deterministic.
    decoder_sorted = sorted(decoder_contacts, key=lambda d: d.frame)

    for d in decoder_sorted:
        best_idx = -1
        best_gap = tol_frames + 1
        for i, ca in enumerate(detected):
            if i in matched_idx:
                continue
            gap = abs(ca.frame - d.frame)
            if gap < best_gap:
                best_gap = gap
                best_idx = i
        if best_idx == -1:
            continue
        matched_idx.add(best_idx)
        target_type = _DECODER_ACTION_TO_TYPE.get(d.action)
        if target_type is None:
            continue
        if detected[best_idx].action_type != target_type:
            detected[best_idx] = replace(detected[best_idx], action_type=target_type)
            swap_count += 1

    new_rally_actions = replace(rally_actions, actions=detected)
    stat = OverlayStat(
        n_decoder_contacts=len(decoder_contacts),
        n_detected_contacts=len(rally_actions.actions),
        n_matched=len(matched_idx),
        n_label_swapped=swap_count,
    )
    return new_rally_actions, stat
