"""Identity-aware player attribution for ball contacts.

Replaces proximity-based attribution with identity-first attribution.
Instead of picking the nearest player and trying to fix it post-hoc,
this module uses per-frame identity classification to know each player's
team before attribution, then picks the nearest player on the correct team.

Key insight from oracle analysis: knowing player identity fixes 83.8% of
attribution errors (95.7% ceiling vs 74.4% proximity baseline). The main
gain comes from correct TEAM assignment (87.7% oracle team), which image-
space proximity cannot reliably determine in end-line camera geometry.

Usage:
    # After contact detection and identity classification
    attributed = attribute_contacts_with_identity(
        contacts=contact_sequence.contacts,
        identity_labels=identity_labels,  # from FrameIdentityClassifier
        player_teams=player_teams,        # {player_id: team}
    )
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.tracking.contact_detector import Contact
    from rallycut.tracking.identity_classifier import IdentityLabel

logger = logging.getLogger(__name__)


# Maximum distance multiplier when preferring a same-team candidate over
# the proximity winner. A same-team candidate at 2.0x the nearest distance
# is still preferred over a cross-team candidate.
MAX_TEAM_DISTANCE_RATIO = 2.5

# Minimum identity confidence to trust team assignment from identity.
# Below this, fall back to proximity-only.
MIN_IDENTITY_CONFIDENCE = 0.35


def attribute_contacts_with_identity(
    contacts: list[Contact],
    identity_labels: dict[int, dict[int, IdentityLabel]],
    player_teams: dict[int, int] | None = None,
) -> list[Contact]:
    """Attribute contacts using identity-aware team assignment.

    For each contact, determines the expected team from possession counting
    (consecutive contacts by same team), then picks the nearest player on
    the correct team using identity labels.

    Args:
        contacts: Detected contacts with player_candidates (ranked by distance).
        identity_labels: {track_id: {frame: IdentityLabel}} from
            FrameIdentityClassifier.classify_detections_batch().
        player_teams: {player_id: team (0/1)}. If None, derived from identity
            labels.

    Returns:
        New list of Contact objects with updated player_track_id and
        player_distance.
    """
    if not contacts:
        return contacts

    if not identity_labels:
        logger.debug("No identity labels available, returning contacts unchanged")
        return contacts

    # Build player_teams from identity labels if not provided
    if player_teams is None:
        player_teams = _infer_player_teams(identity_labels)

    # Build flat lookup: {(track_id, frame) -> IdentityLabel}
    # Also build per-track latest label for fallback
    label_lookup: dict[tuple[int, int], IdentityLabel] = {}
    track_latest: dict[int, IdentityLabel] = {}
    for tid, frame_labels in identity_labels.items():
        for frame, label in frame_labels.items():
            label_lookup[(tid, frame)] = label
        if frame_labels:
            latest_frame = max(frame_labels.keys())
            track_latest[tid] = frame_labels[latest_frame]

    # First pass: attribute each contact independently using identity
    attributed: list[Contact] = []
    n_identity_attributed = 0
    n_team_swaps = 0
    n_unchanged = 0

    # Track team possession for expected-team inference
    last_team: int | None = None

    for contact in contacts:
        best_tid, best_dist, swap_reason = _pick_best_candidate(
            contact=contact,
            label_lookup=label_lookup,
            track_latest=track_latest,
            player_teams=player_teams,
            last_team=last_team,
        )

        if best_tid >= 0:
            # Update last_team from the attributed player's team
            best_label = _get_label(best_tid, contact.frame, label_lookup, track_latest)
            if best_label is not None and best_label.confidence >= MIN_IDENTITY_CONFIDENCE:
                last_team = best_label.team

        if best_tid != contact.player_track_id:
            new_contact = replace(
                contact,
                player_track_id=best_tid,
                player_distance=best_dist,
            )
            attributed.append(new_contact)

            if swap_reason == "team":
                n_team_swaps += 1
            n_identity_attributed += 1
        else:
            attributed.append(contact)
            n_unchanged += 1

            # Still track team from proximity-attributed player
            if contact.player_track_id >= 0:
                cur_label = _get_label(
                    contact.player_track_id, contact.frame,
                    label_lookup, track_latest,
                )
                if cur_label is not None and cur_label.confidence >= MIN_IDENTITY_CONFIDENCE:
                    last_team = cur_label.team

    logger.info(
        "Identity attribution: %d/%d changed (%d team swaps), %d unchanged",
        n_identity_attributed, len(contacts), n_team_swaps, n_unchanged,
    )

    return attributed


def derive_team_from_identity(
    contacts: list[Contact],
    identity_labels: dict[int, dict[int, IdentityLabel]],
    player_teams: dict[int, int] | None = None,
) -> list[str]:
    """Derive court_side for each contact from identity labels.

    Returns list of court_side strings ("near", "far", "unknown") parallel
    to contacts.
    """
    if not identity_labels:
        return [c.court_side for c in contacts]

    if player_teams is None:
        player_teams = _infer_player_teams(identity_labels)

    # Build lookup
    label_lookup: dict[tuple[int, int], IdentityLabel] = {}
    track_latest: dict[int, IdentityLabel] = {}
    for tid, frame_labels in identity_labels.items():
        for frame, label in frame_labels.items():
            label_lookup[(tid, frame)] = label
        if frame_labels:
            latest_frame = max(frame_labels.keys())
            track_latest[tid] = frame_labels[latest_frame]

    sides: list[str] = []
    for contact in contacts:
        tid = contact.player_track_id
        if tid < 0:
            sides.append("unknown")
            continue

        id_label = _get_label(tid, contact.frame, label_lookup, track_latest)
        if id_label is not None and id_label.confidence >= MIN_IDENTITY_CONFIDENCE:
            sides.append("near" if id_label.team == 0 else "far")
        else:
            sides.append(contact.court_side)

    return sides


def count_contacts_on_team(
    contacts: list[Contact],
    index: int,
    identity_labels: dict[int, dict[int, IdentityLabel]],
) -> int:
    """Count consecutive same-team contacts ending at index.

    Uses identity to determine team instead of net-crossing detection.
    This replaces the unreliable _ball_crossed_net() approach.

    Returns count (1-3, capped at beach volleyball max).
    """
    if not identity_labels or index < 0 or index >= len(contacts):
        return 1

    # Build lookup
    label_lookup: dict[tuple[int, int], IdentityLabel] = {}
    track_latest: dict[int, IdentityLabel] = {}
    for tid, frame_labels in identity_labels.items():
        for frame, label in frame_labels.items():
            label_lookup[(tid, frame)] = label
        if frame_labels:
            latest_frame = max(frame_labels.keys())
            track_latest[tid] = frame_labels[latest_frame]

    # Get current contact's team
    current_tid = contacts[index].player_track_id
    current_label = _get_label(
        current_tid, contacts[index].frame, label_lookup, track_latest,
    )
    if current_label is None or current_label.confidence < MIN_IDENTITY_CONFIDENCE:
        return 1

    current_team = current_label.team
    count = 1

    # Walk backward counting same-team contacts
    for i in range(index - 1, -1, -1):
        prev_tid = contacts[i].player_track_id
        prev_label = _get_label(
            prev_tid, contacts[i].frame, label_lookup, track_latest,
        )
        if prev_label is None or prev_label.confidence < MIN_IDENTITY_CONFIDENCE:
            break
        if prev_label.team != current_team:
            break
        count += 1
        if count >= 3:
            break

    return min(count, 3)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_label(
    track_id: int,
    frame: int,
    label_lookup: dict[tuple[int, int], IdentityLabel],
    track_latest: dict[int, IdentityLabel],
) -> IdentityLabel | None:
    """Look up identity label for a track at a frame, with fallback."""
    label = label_lookup.get((track_id, frame))
    if label is not None:
        return label

    # Try nearby frames (±3)
    for offset in range(1, 4):
        for f in (frame + offset, frame - offset):
            label = label_lookup.get((track_id, f))
            if label is not None:
                return label

    # Fall back to track's latest overall label
    return track_latest.get(track_id)


def _pick_best_candidate(
    contact: Contact,
    label_lookup: dict[tuple[int, int], IdentityLabel],
    track_latest: dict[int, IdentityLabel],
    player_teams: dict[int, int],
    last_team: int | None,
) -> tuple[int, float, str]:
    """Pick the best player candidate for a contact using identity and motion.

    Strategy: score each candidate by combining identity confidence, distance,
    and bbox motion (peak Y shift and height change in ±5 frames). Identity
    provides the team signal, motion captures arm swings / jumps at contact.
    When identity is confident, prefer the identified player even if farther
    away. When identity is uncertain, fall back to proximity.

    Returns (track_id, distance, reason) where reason is one of:
    - "proximity": same as original (nearest player)
    - "team": swapped to candidate on different team based on identity
    - "identity": swapped based on identity confidence
    """
    candidates = contact.player_candidates
    if not candidates:
        return contact.player_track_id, contact.player_distance, "proximity"

    nearest_tid, nearest_dist = candidates[0]

    # Get identity labels for all candidates
    candidate_scores: list[tuple[int, float, float, IdentityLabel | None]] = []
    for tid, dist in candidates:
        label = _get_label(tid, contact.frame, label_lookup, track_latest)
        candidate_scores.append((tid, dist, 0.0, label))

    # If only one candidate or no identity data, keep proximity
    has_identity = any(
        lbl is not None and lbl.confidence >= MIN_IDENTITY_CONFIDENCE
        for _, _, _, lbl in candidate_scores
    )
    if not has_identity:
        return nearest_tid, nearest_dist, "proximity"

    # Bbox motion: normalize max_d_y and max_d_height across candidates
    # so that the candidate with the largest motion gets motion_factor=1.0.
    bbox_motion = contact.candidate_bbox_motion
    max_motion = 0.0
    for tid, _ in candidates:
        if tid in bbox_motion:
            dy, dh = bbox_motion[tid]
            max_motion = max(max_motion, dy + dh)

    # Score each candidate: identity confidence, penalized by distance,
    # boosted by bbox motion (arm swing / jump at contact time).
    best_tid = nearest_tid
    best_dist = nearest_dist
    best_score = -1.0
    reason = "proximity"

    for tid, dist, _, label in candidate_scores:
        # Bbox motion factor: fraction of max observed motion across candidates
        motion_factor = 0.0
        if max_motion > 1e-6 and tid in bbox_motion:
            dy, dh = bbox_motion[tid]
            motion_factor = (dy + dh) / max_motion

        if label is None or label.confidence < MIN_IDENTITY_CONFIDENCE:
            # Unidentified candidate: proximity + motion
            prox = 1.0 - min(dist / max(nearest_dist * 3, 0.01), 1.0)
            score = 0.25 * prox + 0.05 * motion_factor
        else:
            # Identified candidate: identity dominates, distance + motion secondary
            dist_factor = 1.0 - min(dist / max(nearest_dist * MAX_TEAM_DISTANCE_RATIO, 0.01), 1.0)
            score = 0.65 * label.confidence + 0.25 * dist_factor + 0.10 * motion_factor

        if score > best_score:
            best_score = score
            best_tid = tid
            best_dist = dist

    if best_tid != nearest_tid:
        best_label = _get_label(best_tid, contact.frame, label_lookup, track_latest)
        nearest_label = _get_label(nearest_tid, contact.frame, label_lookup, track_latest)
        if (best_label is not None and nearest_label is not None
                and best_label.team != nearest_label.team):
            reason = "team"
        else:
            reason = "identity"

    return best_tid, best_dist, reason


def _infer_player_teams(
    identity_labels: dict[int, dict[int, IdentityLabel]],
) -> dict[int, int]:
    """Infer player-to-team mapping from identity labels."""
    teams: dict[int, int] = {}
    for _tid, frame_labels in identity_labels.items():
        for _frame, label in frame_labels.items():
            if label.player_id >= 0 and label.confidence >= MIN_IDENTITY_CONFIDENCE:
                teams[label.player_id] = label.team
    return teams
