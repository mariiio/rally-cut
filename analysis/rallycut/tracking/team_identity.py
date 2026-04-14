"""First-class team identity from player appearance profiles.

A team is a pair of individually-identified players. Team templates are
stored in match_analysis_json and used for per-rally team localization:
given track_to_player (which player IDs are where) and Y positions,
determine which team is near and which is far — independently per rally,
with no accumulated side-switch state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from rallycut.tracking.player_features import PlayerAppearanceProfile

if TYPE_CHECKING:
    from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)


@dataclass
class TeamTemplate:
    """Appearance signature for a team (pair of players).

    Built from match_tracker's PlayerAppearanceProfile after Pass 2.
    Stored in match_analysis_json for downstream use.
    """

    team_label: str  # "0" or "1" (convention assigns A/B separately)
    player_ids: list[int]  # e.g. [1, 2] or [3, 4]
    confidence: float = 0.0  # Inter-team discriminability

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
    ) -> TeamTemplate:
        """Deserialize from dict."""
        return cls(
            team_label=d["teamLabel"],
            player_ids=d["playerIds"],
            confidence=d.get("confidence", 0.0),
        )


def _bhattacharyya_distance(
    hist_a: np.ndarray | None, hist_b: np.ndarray | None,
) -> float:
    """Bhattacharyya distance between two L1-normalized histograms."""
    if hist_a is None or hist_b is None:
        return 0.5
    a = hist_a.flatten().astype(np.float64)
    b = hist_b.flatten().astype(np.float64)
    sa, sb = a.sum(), b.sum()
    if sa < 1e-10 or sb < 1e-10:
        return 0.5
    a = a / sa
    b = b / sb
    return 1.0 - float(np.sum(np.sqrt(a * b)))


def build_team_templates(
    profiles: dict[int, PlayerAppearanceProfile],
) -> tuple[TeamTemplate, TeamTemplate]:
    """Build two team templates from match_tracker's player profiles.

    Groups by match_tracker's convention: players 1-2 = team 0, 3-4 = team 1.

    Args:
        profiles: Player ID (1-4) to appearance profile.

    Returns:
        (template_0, template_1) with labels "0" and "1".
    """
    team_0_ids = sorted(pid for pid in profiles if pid <= 2)
    team_1_ids = sorted(pid for pid in profiles if pid >= 3)

    team_0_profiles = [profiles[pid] for pid in team_0_ids if pid in profiles]
    team_1_profiles = [profiles[pid] for pid in team_1_ids if pid in profiles]

    # Inter-team discriminability from lower-body histograms.
    distances: list[float] = []
    for p0 in team_0_profiles:
        for p1 in team_1_profiles:
            d = _bhattacharyya_distance(p0.avg_lower_hist, p1.avg_lower_hist)
            distances.append(d)
    discriminability = float(np.mean(distances)) if distances else 0.0

    template_0 = TeamTemplate(
        team_label="0", player_ids=team_0_ids, confidence=discriminability,
    )
    template_1 = TeamTemplate(
        team_label="1", player_ids=team_1_ids, confidence=discriminability,
    )

    logger.info(
        "Built team templates: team_0=%s team_1=%s discriminability=%.3f",
        team_0_ids, team_1_ids, discriminability,
    )
    return template_0, template_1


def localize_team_near(
    positions: list[PlayerPosition],
    track_to_player: dict[int, int],
    templates: tuple[TeamTemplate, TeamTemplate],
    min_y_gap: float = 0.03,
) -> str | None:
    """Determine which team template is near for a single rally.

    Uses track_to_player to map tracks to player IDs, then groups by
    team template and compares mean Y positions. Higher Y = near (closer
    to camera).

    Returns None when the Y gap between teams is below ``min_y_gap``,
    indicating that localization is unreliable (e.g. narrow-angle cameras
    where near and far players have similar Y positions). Callers should
    fall back to Viterbi-based side switch detection in this case.

    Args:
        positions: Player positions for this rally.
        track_to_player: track_id -> player_id (1-4).
        templates: The two team templates.
        min_y_gap: Minimum mean-Y gap between teams to trust the result.
            Below this threshold, returns None. Default 0.03 (~3% of image
            height) covers narrow-angle cameras where track_to_player
            phantom flips cause wrong team grouping.

            Raising to 0.05 was tested: isolated team_near accuracy
            improves 83.8→87.4% but end-to-end score_accuracy regresses
            (88.6→87.5%) because filtered rallies fall back to Viterbi
            which is less accurate than marginal team_near returns in the
            production context.

    Returns:
        Team label of the near team ("0" or "1"), or None if ambiguous.
    """
    if not positions or not track_to_player:
        return None

    t0, t1 = templates
    t0_pids = set(t0.player_ids)
    t1_pids = set(t1.player_ids)

    # Mean foot-Y per player_id
    pid_ys: dict[int, list[float]] = {}
    for p in positions:
        pid = track_to_player.get(p.track_id)
        if pid is not None:
            pid_ys.setdefault(pid, []).append(p.y + p.height / 2.0)

    if not pid_ys:
        return None

    t0_y = [float(np.mean(pid_ys[pid])) for pid in t0_pids if pid in pid_ys]
    t1_y = [float(np.mean(pid_ys[pid])) for pid in t1_pids if pid in pid_ys]

    if not t0_y or not t1_y:
        return None

    mean_t0 = float(np.mean(t0_y))
    mean_t1 = float(np.mean(t1_y))

    if abs(mean_t0 - mean_t1) < min_y_gap:
        return None

    return t0.team_label if mean_t0 > mean_t1 else t1.team_label


def resolve_serving_team(
    formation_side: str | None,
    team_near_label: str | None,
    templates: tuple[TeamTemplate, TeamTemplate],
    label_a: str,
) -> str | None:
    """Resolve serving team from formation side + team localization.

    Args:
        formation_side: "near" or "far" (which side serves).
        team_near_label: Template label of the near team ("0"/"1").
        templates: The two team templates (for deriving team_far).
        label_a: Which template label corresponds to "A".

    Returns:
        "A" or "B", or None if insufficient data.
    """
    if formation_side is None or team_near_label is None:
        return None

    t0, t1 = templates
    all_labels = {t0.team_label, t1.team_label}

    serving_label: str | None
    if formation_side == "near":
        serving_label = team_near_label
    else:
        remaining = all_labels - {team_near_label}
        serving_label = remaining.pop() if remaining else None

    if serving_label is None:
        return None

    return "A" if serving_label == label_a else "B"


def calibrate_convention_from_gt(
    gt_serving_teams: list[str | None],
    formation_sides: list[str | None],
    team_near_labels: list[str | None],
    templates: tuple[TeamTemplate, TeamTemplate],
) -> str:
    """Determine which template label = GT's "A" via majority vote.

    For each GT-labeled rally where formation + team localization are
    available, determines which template served and what GT calls that
    team. Standard eval methodology (label alignment).

    Returns:
        Template label that maps to "A". Defaults to "0".
    """
    t0, t1 = templates
    all_labels = {t0.team_label, t1.team_label}
    votes: dict[str, int] = {}

    for gt, formation_side, team_near in zip(
        gt_serving_teams, formation_sides, team_near_labels,
    ):
        if gt is None or formation_side is None or team_near is None:
            continue

        serving_label: str | None
        if formation_side == "near":
            serving_label = team_near
        else:
            remaining = all_labels - {team_near}
            serving_label = remaining.pop() if remaining else None

        if serving_label is None:
            continue

        votes.setdefault(serving_label, 0)
        if gt == "A":
            votes[serving_label] += 1
        else:
            votes[serving_label] -= 1

    if not votes:
        return "0"
    return max(votes, key=lambda k: votes[k])
