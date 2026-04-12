"""First-class team identity from player appearance profiles.

Decouples team identity from court position. A team is defined as a pair
of individually-identified players with stored appearance signatures.
TeamLocalizer determines which team is on which side per rally using
individual identity matching (DINOv2/OSNet at 94.9%), independent of
the player ID scheme.

Used by the Viterbi decoder to break the dual-hypothesis tie (which was
previously a structural no-op because complementary sequences have
identical plausibility statistics).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.tracking.player_features import (
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
)

logger = logging.getLogger(__name__)

# Minimum inter-team discriminability (Bhattacharyya distance between
# team lower-body histograms) to consider team templates reliable.
# Below this, TeamLocalizer confidence is set to 0.
MIN_TEAM_DISCRIMINABILITY = 0.10

# Minimum margin between best and second-best team assignment for a
# track to count as confident. Below this, the localization result
# for that track is ambiguous.
MIN_LOCALIZATION_MARGIN = 0.03


@dataclass
class TeamTemplate:
    """Appearance signature for a team (pair of players).

    Built from match_tracker's PlayerAppearanceProfile after Pass 2.
    Stored in match_analysis_json for downstream use.
    """

    team_label: str  # "A" or "B"
    player_ids: list[int]  # e.g. [1, 2] or [3, 4]
    confidence: float = 0.0  # Inter-team discriminability

    # Per-player appearance data (indexed same as player_ids)
    _profiles: list[PlayerAppearanceProfile] = field(
        default_factory=list, repr=False,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict for match_analysis_json."""
        return {
            "teamLabel": self.team_label,
            "playerIds": self.player_ids,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        profiles: dict[int, PlayerAppearanceProfile] | None = None,
    ) -> TeamTemplate:
        """Deserialize from dict, optionally attaching live profiles."""
        template = cls(
            team_label=d["teamLabel"],
            player_ids=d["playerIds"],
            confidence=d.get("confidence", 0.0),
        )
        if profiles:
            template._profiles = [
                profiles[pid] for pid in template.player_ids
                if pid in profiles
            ]
        return template


@dataclass
class TeamLocalizationResult:
    """Per-rally result of team localization."""

    team_near: str | None  # "A" or "B", or None if ambiguous
    team_far: str | None  # "A" or "B", or None if ambiguous
    confidence: float  # 0-1, based on assignment margin


def _bhattacharyya_distance(
    hist_a: np.ndarray | None, hist_b: np.ndarray | None,
) -> float:
    """Bhattacharyya distance between two L1-normalized histograms.

    Returns 0 (identical) to 1 (no overlap). Returns 0.5 if either is None.
    """
    if hist_a is None or hist_b is None:
        return 0.5
    a = hist_a.flatten().astype(np.float64)
    b = hist_b.flatten().astype(np.float64)
    # Normalize to sum to 1
    sa, sb = a.sum(), b.sum()
    if sa < 1e-10 or sb < 1e-10:
        return 0.5
    a = a / sa
    b = b / sb
    bc = float(np.sum(np.sqrt(a * b)))
    return 1.0 - bc


def build_team_templates(
    profiles: dict[int, PlayerAppearanceProfile],
) -> tuple[TeamTemplate, TeamTemplate]:
    """Build two team templates from match_tracker's player profiles.

    Groups by match_tracker's convention: players 1-2 = team 0, 3-4 = team 1.
    Labels are initially "0" and "1" — ConventionResolver assigns A/B.

    Args:
        profiles: Player ID (1-4) to appearance profile.

    Returns:
        (template_0, template_1) with initial labels "0" and "1".
    """
    team_0_ids = sorted(pid for pid in profiles if pid <= 2)
    team_1_ids = sorted(pid for pid in profiles if pid >= 3)

    team_0_profiles = [profiles[pid] for pid in team_0_ids if pid in profiles]
    team_1_profiles = [profiles[pid] for pid in team_1_ids if pid in profiles]

    # Compute inter-team discriminability from lower-body histograms.
    # Average pairwise Bhattacharyya distance between teams.
    distances: list[float] = []
    for p0 in team_0_profiles:
        for p1 in team_1_profiles:
            d = _bhattacharyya_distance(p0.avg_lower_hist, p1.avg_lower_hist)
            distances.append(d)

    discriminability = float(np.mean(distances)) if distances else 0.0

    template_0 = TeamTemplate(
        team_label="0",
        player_ids=team_0_ids,
        confidence=discriminability,
        _profiles=team_0_profiles,
    )
    template_1 = TeamTemplate(
        team_label="1",
        player_ids=team_1_ids,
        confidence=discriminability,
        _profiles=team_1_profiles,
    )

    logger.info(
        "Built team templates: team_0=%s team_1=%s discriminability=%.3f",
        team_0_ids, team_1_ids, discriminability,
    )

    return template_0, template_1


def localize_teams(
    track_stats: dict[int, TrackAppearanceStats],
    track_positions_y: dict[int, float],
    templates: tuple[TeamTemplate, TeamTemplate],
) -> TeamLocalizationResult:
    """Determine which team is near and which is far for a single rally.

    Uses individual identity matching: each track is matched to the closest
    player in any team template via appearance similarity. Tracks are then
    grouped by team, and the team with higher average Y is near.

    Args:
        track_stats: Per-track appearance stats for this rally.
        track_positions_y: track_id -> average Y position (higher = near).
        templates: The two team templates.

    Returns:
        TeamLocalizationResult with team_near/team_far labels and confidence.
    """
    template_0, template_1 = templates
    all_template_profiles = template_0._profiles + template_1._profiles
    all_template_pids = template_0.player_ids + template_1.player_ids

    if not all_template_profiles or not track_stats:
        return TeamLocalizationResult(team_near=None, team_far=None, confidence=0.0)

    # Build cost matrix: tracks x template players
    track_ids = list(track_stats.keys())
    n_tracks = len(track_ids)
    n_players = len(all_template_profiles)

    if n_tracks == 0 or n_players == 0:
        return TeamLocalizationResult(team_near=None, team_far=None, confidence=0.0)

    cost_matrix = np.ones((n_tracks, n_players), dtype=np.float64)
    for i, tid in enumerate(track_ids):
        stats = track_stats[tid]
        for j, profile in enumerate(all_template_profiles):
            cost_matrix[i, j] = compute_appearance_similarity(profile, stats)

    # Solve assignment (handles n_tracks != n_players)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Map tracks to teams
    team_0_set = set(template_0.player_ids)
    team_y: dict[str, list[float]] = {
        template_0.team_label: [],
        template_1.team_label: [],
    }

    margins: list[float] = []
    for r, c in zip(row_ind, col_ind):
        tid = track_ids[r]
        pid = all_template_pids[c]
        best_cost = cost_matrix[r, c]

        # Compute margin: difference between best and second-best
        row_costs = sorted(cost_matrix[r])
        margin = row_costs[1] - row_costs[0] if len(row_costs) > 1 else 0.0
        margins.append(margin)

        if margin < MIN_LOCALIZATION_MARGIN:
            continue  # Ambiguous — skip this track

        if tid in track_positions_y:
            team_label = template_0.team_label if pid in team_0_set else template_1.team_label
            team_y[team_label].append(track_positions_y[tid])

    # Determine which team is near (higher Y)
    label_0 = template_0.team_label
    label_1 = template_1.team_label

    if not team_y[label_0] or not team_y[label_1]:
        return TeamLocalizationResult(team_near=None, team_far=None, confidence=0.0)

    mean_y_0 = float(np.mean(team_y[label_0]))
    mean_y_1 = float(np.mean(team_y[label_1]))

    if mean_y_0 > mean_y_1:
        team_near, team_far = label_0, label_1
    else:
        team_near, team_far = label_1, label_0

    # Confidence: mean margin across assigned tracks
    avg_margin = float(np.mean(margins)) if margins else 0.0
    confidence = min(1.0, avg_margin / 0.15)  # Normalize: 0.15 margin → 1.0

    return TeamLocalizationResult(
        team_near=team_near,
        team_far=team_far,
        confidence=confidence,
    )
