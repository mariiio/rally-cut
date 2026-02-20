"""Touch grammar constraints for volleyball identity resolution.

Uses volleyball rules to score the plausibility of contact sequences
with specific player attributions. Rules encoded:
- No consecutive touches by the same player (except after block)
- Max 3 touches per side before ball crosses net
- Contacts should come from players on the correct side of the ball

These scores feed into the swap/no-swap hypothesis scoring to help
disambiguate player identities during net interactions.
"""

from __future__ import annotations

import logging
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.tracking.contact_detector import Contact, ContactSequence

logger = logging.getLogger(__name__)


@dataclass
class GrammarScore:
    """Plausibility score from touch grammar analysis."""

    consecutive_violations: int = 0  # Same player touching twice in a row
    max_touches_violations: int = 0  # >3 touches on one side
    wrong_side_violations: int = 0  # Contact on wrong side of net
    total_contacts: int = 0

    @property
    def violation_rate(self) -> float:
        """Total violations as fraction of contacts."""
        if self.total_contacts == 0:
            return 0.0
        total = (
            self.consecutive_violations
            + self.max_touches_violations
            + self.wrong_side_violations
        )
        return total / self.total_contacts

    @property
    def plausibility(self) -> float:
        """Plausibility score (1.0 = perfect, 0.0 = many violations)."""
        return max(0.0, 1.0 - self.violation_rate)


def score_contact_grammar(
    contact_sequence: ContactSequence,
    team_assignments: dict[int, int],
) -> GrammarScore:
    """Score the plausibility of a contact sequence under volleyball rules.

    Args:
        contact_sequence: Detected contacts with player attributions.
        team_assignments: track_id -> team (0=near, 1=far).

    Returns:
        GrammarScore with violation counts.
    """
    contacts = contact_sequence.contacts
    score = GrammarScore(total_contacts=len(contacts))

    if len(contacts) < 2:
        return score

    # Track consecutive-touch violations
    prev_player = -1
    for contact in contacts:
        player = contact.player_track_id
        if player < 0:
            prev_player = -1
            continue

        if player == prev_player:
            score.consecutive_violations += 1
        prev_player = player

    # Track max-touches-per-side violations
    # Split contacts into possessions (separated by net crossings)
    possessions = _split_into_possessions(contacts, contact_sequence.net_y)
    for possession in possessions:
        if len(possession) > 3:
            score.max_touches_violations += len(possession) - 3

    # Track wrong-side contacts
    net_y = contact_sequence.net_y
    for contact in contacts:
        player = contact.player_track_id
        if player < 0:
            continue

        team = team_assignments.get(player, -1)
        if team < 0:
            continue

        # Near team (0) contacts should have ball_y > net_y (near side)
        # Far team (1) contacts should have ball_y < net_y (far side)
        if team == 0 and contact.ball_y < net_y - 0.05:
            score.wrong_side_violations += 1
        elif team == 1 and contact.ball_y > net_y + 0.05:
            score.wrong_side_violations += 1

    return score


def score_swap_hypothesis(
    contact_sequence: ContactSequence,
    team_assignments: dict[int, int],
    track_a: int,
    track_b: int,
    swap_from_frame: int,
) -> tuple[float, float]:
    """Score grammar plausibility for no-swap vs swap hypotheses.

    Args:
        contact_sequence: Contact sequence for the rally.
        team_assignments: Current track_id -> team mapping.
        track_a: First track in the swap pair.
        track_b: Second track in the swap pair.
        swap_from_frame: Frame from which swap would apply.

    Returns:
        (no_swap_score, swap_score) each in [0, 1].
    """
    contacts = contact_sequence.contacts
    if len(contacts) < 2:
        return 0.5, 0.5

    # Score with current assignments (no swap)
    no_swap_score = score_contact_grammar(
        contact_sequence, team_assignments
    )

    # Score with swapped assignments
    swapped_assignments = dict(team_assignments)
    team_a = swapped_assignments.get(track_a)
    team_b = swapped_assignments.get(track_b)
    if team_a is not None:
        swapped_assignments[track_b] = team_a
    if team_b is not None:
        swapped_assignments[track_a] = team_b

    # Also swap player_track_id in contacts after swap_from_frame
    swapped_contacts: list[Contact] = []
    for c in contacts:
        if c.frame >= swap_from_frame:
            new_c = _copy_contact_with_swapped_id(c, track_a, track_b)
            swapped_contacts.append(new_c)
        else:
            swapped_contacts.append(c)

    # Build a temporary sequence with swapped contacts
    from rallycut.tracking.contact_detector import ContactSequence

    swapped_seq = ContactSequence(
        contacts=swapped_contacts,
        net_y=contact_sequence.net_y,
        rally_start_frame=contact_sequence.rally_start_frame,
        ball_positions=contact_sequence.ball_positions,
    )

    swap_grammar = score_contact_grammar(
        swapped_seq, swapped_assignments
    )

    return no_swap_score.plausibility, swap_grammar.plausibility


def _split_into_possessions(
    contacts: list[Contact],
    net_y: float,
) -> list[list[Contact]]:
    """Split contacts into possessions based on net crossings."""
    if not contacts:
        return []

    possessions: list[list[Contact]] = [[]]
    prev_side = "near" if contacts[0].ball_y > net_y else "far"

    for contact in contacts:
        current_side = "near" if contact.ball_y > net_y else "far"
        if current_side != prev_side:
            possessions.append([])
            prev_side = current_side
        possessions[-1].append(contact)

    return possessions


def _copy_contact_with_swapped_id(
    contact: Contact,
    track_a: int,
    track_b: int,
) -> Contact:
    """Create a copy of a Contact with swapped track ID."""
    new_contact = copy(contact)
    if new_contact.player_track_id == track_a:
        new_contact.player_track_id = track_b
    elif new_contact.player_track_id == track_b:
        new_contact.player_track_id = track_a
    return new_contact
