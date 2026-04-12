"""Convention resolution: map team templates to A/B labels.

The A/B label is inherently arbitrary without external grounding.
This module provides two resolution modes:

- **Automatic** (production): Team that served first = team A.
  Anchored to rally 1's formation + team localization.
- **GT calibration** (eval): Majority-vote from GT serving_team labels
  determines which template = GT's team A.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rallycut.tracking.team_identity import TeamLocalizationResult, TeamTemplate

logger = logging.getLogger(__name__)


@dataclass
class ConventionResult:
    """Result of convention resolution."""

    initial_near_is_a: bool  # Whether near = team A at rally 0
    confidence: float  # 0-1
    method: str  # "automatic" or "gt_calibration"


def resolve_automatic(
    templates: tuple[TeamTemplate, TeamTemplate],
    first_rally_formation_side: str | None,
    first_rally_localization: TeamLocalizationResult | None,
) -> ConventionResult:
    """Resolve A/B convention automatically from first rally.

    Convention: the team that served first = team A.
    - Formation tells us which side served first (near/far)
    - TeamLocalizer tells us which team template is on which side
    - Together: serving-side template = team A

    Args:
        templates: The two team templates (labeled "0" and "1").
        first_rally_formation_side: "near" or "far" (which side served).
        first_rally_localization: Which template is near/far.

    Returns:
        ConventionResult with initial_near_is_a.
    """
    template_0, template_1 = templates

    # If we can't determine formation or team localization, default near=A
    if first_rally_formation_side is None or first_rally_localization is None:
        return ConventionResult(
            initial_near_is_a=True, confidence=0.0, method="automatic",
        )

    if first_rally_localization.team_near is None:
        return ConventionResult(
            initial_near_is_a=True, confidence=0.0, method="automatic",
        )

    # Which template served first?
    serving_template: str | None
    if first_rally_formation_side == "near":
        serving_template = first_rally_localization.team_near
    else:
        serving_template = first_rally_localization.team_far

    if serving_template is None:
        return ConventionResult(
            initial_near_is_a=True, confidence=0.0, method="automatic",
        )

    # Convention: serving template = "A". Near = A iff serving template is near.
    # Relabel templates accordingly.
    if serving_template == first_rally_localization.team_near:
        # Serving team is near → near = A
        initial_near_is_a = True
    else:
        # Serving team is far → near = B
        initial_near_is_a = False

    # Assign A/B labels to templates
    if initial_near_is_a:
        template_0.team_label = "A" if first_rally_localization.team_near == template_0.team_label else "B"
        template_1.team_label = "B" if template_0.team_label == "A" else "A"
    else:
        template_0.team_label = "B" if first_rally_localization.team_near == template_0.team_label else "A"
        template_1.team_label = "A" if template_0.team_label == "B" else "B"

    logger.info(
        "Convention resolved: near=%s, serving_side=%s, "
        "template_0=%s (players %s), template_1=%s (players %s)",
        "A" if initial_near_is_a else "B",
        first_rally_formation_side,
        template_0.team_label, template_0.player_ids,
        template_1.team_label, template_1.player_ids,
    )

    return ConventionResult(
        initial_near_is_a=initial_near_is_a,
        confidence=first_rally_localization.confidence,
        method="automatic",
    )


def resolve_from_gt(
    templates: tuple[TeamTemplate, TeamTemplate],
    gt_serving_teams: list[str | None],
    team_localizations: list[TeamLocalizationResult | None],
    formation_sides: list[str | None],
) -> ConventionResult:
    """Resolve A/B convention from GT labels via majority vote.

    For each GT-labeled rally:
    - Formation says which side served (near/far)
    - TeamLocalizer says which template is near
    - GT says which team label (A/B) served
    - Together: determine which template = GT's team A

    Args:
        templates: The two team templates.
        gt_serving_teams: Per-rally GT serving team ("A"/"B"/None).
        team_localizations: Per-rally team localization results.
        formation_sides: Per-rally formation side ("near"/"far"/None).

    Returns:
        ConventionResult with GT-calibrated initial_near_is_a.
    """
    template_0, template_1 = templates

    votes_near_is_a = 0.0
    votes_near_is_b = 0.0

    for gt, loc, formation_side in zip(
        gt_serving_teams, team_localizations, formation_sides,
    ):
        if gt is None or loc is None or formation_side is None:
            continue
        if loc.team_near is None or loc.confidence < 0.1:
            continue

        # Which template served?
        serving_template: str | None
        if formation_side == "near":
            serving_template = loc.team_near
        else:
            serving_template = loc.team_far

        if serving_template is None:
            continue

        # GT says team X served. Template Y served.
        # If gt="A" and serving_template is near → near = A
        weight = loc.confidence
        if gt == "A":
            if serving_template == loc.team_near:
                votes_near_is_a += weight
            else:
                votes_near_is_b += weight
        else:  # gt == "B"
            if serving_template == loc.team_near:
                votes_near_is_b += weight
            else:
                votes_near_is_a += weight

    total = votes_near_is_a + votes_near_is_b
    if total < 1e-6:
        # No signal — fall back to automatic
        logger.warning("GT calibration: no valid votes, falling back to near=A")
        return ConventionResult(
            initial_near_is_a=True, confidence=0.0, method="gt_calibration",
        )

    initial_near_is_a = votes_near_is_a >= votes_near_is_b
    confidence = abs(votes_near_is_a - votes_near_is_b) / total

    # Assign A/B labels to templates
    # Template whose players are near in rally 0 gets the near label
    if team_localizations and team_localizations[0] is not None:
        loc0 = team_localizations[0]
        if loc0.team_near is not None:
            near_template_label = loc0.team_near
            if initial_near_is_a:
                # Near template = A
                for t in (template_0, template_1):
                    if t.team_label == near_template_label:
                        t.team_label = "A"
                    else:
                        t.team_label = "B"
            else:
                # Near template = B
                for t in (template_0, template_1):
                    if t.team_label == near_template_label:
                        t.team_label = "B"
                    else:
                        t.team_label = "A"

    logger.info(
        "GT calibration: near=%s (confidence=%.2f, votes A=%.1f B=%.1f)",
        "A" if initial_near_is_a else "B", confidence,
        votes_near_is_a, votes_near_is_b,
    )

    return ConventionResult(
        initial_near_is_a=initial_near_is_a,
        confidence=confidence,
        method="gt_calibration",
    )
