"""Builder that converts a `ContactSequence` from the parallel Viterbi
decoder path (`detect_contacts_via_decoder`) into a `RallyActions`
object that downstream consumers (action_classifier.RallyActions,
match-stats aggregator, web/API) accept unchanged.

Phase 4 of the parallel-decoder ship plan
(`docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`). The
legacy GBM path runs `classify_rally_actions(contact_seq, ...)` to
produce action labels — which is the +0% to -1.5pp Action Acc lever
the decoder is supposed to replace. For the parallel-decoder path,
the decoder's Viterbi grammar already emitted action labels in
`Contact.decoder_action`; this builder lifts them into the
`ClassifiedAction` schema downstream consumers expect.

This is deliberately minimal — it does NOT run the heavy
relabel/repair passes from `classify_rally_actions`. The decoder's
grammar prior + transition matrix already enforce the same kind of
action-sequence sanity, so the legacy passes are largely redundant on
this path. Phase 5 cleanup will validate this empirically and remove
the dead overlap.
"""

from __future__ import annotations

from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    RallyActions,
    _team_label,
)
from rallycut.tracking.contact_detector import ContactSequence


def build_rally_actions_from_decoder(
    contact_seq: ContactSequence,
    rally_id: str = "",
    team_assignments: dict[int, int] | None = None,
) -> RallyActions:
    """Convert decoder-path contacts to `RallyActions`.

    Each Contact with a populated `decoder_action` field becomes one
    ClassifiedAction. Contacts without a decoder action (which should
    not happen on the decoder path, but defensively skip) are omitted.

    Args:
        contact_seq: ContactSequence from `detect_contacts_via_decoder`.
        rally_id: Rally identifier for downstream attribution lookup.
        team_assignments: Optional map track_id -> team (0=near, 1=far)
            used to populate `ClassifiedAction.team`.

    Returns:
        A RallyActions with one ClassifiedAction per decoder-emitted
        contact, in frame order.
    """
    actions: list[ClassifiedAction] = []
    team_assignments = team_assignments or {}

    for contact in contact_seq.contacts:
        if contact.decoder_action is None:
            continue
        try:
            action_type = ActionType(contact.decoder_action)
        except ValueError:
            action_type = ActionType.UNKNOWN
        actions.append(ClassifiedAction(
            action_type=action_type,
            frame=contact.frame,
            ball_x=contact.ball_x,
            ball_y=contact.ball_y,
            velocity=contact.velocity,
            player_track_id=contact.player_track_id,
            court_side=contact.court_side,
            confidence=contact.confidence,
            is_synthetic=contact.is_synthetic,
            team=_team_label(contact.player_track_id, team_assignments),
        ))

    return RallyActions(
        actions=actions,
        rally_id=rally_id,
        team_assignments=team_assignments,
    )
