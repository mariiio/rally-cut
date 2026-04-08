"""Session 5 — score ground truth & score_accuracy metric helpers.

The score metric is the fraction of GT-labeled rallies whose **predicted
serving team matches the GT serving team**.

**Why serving team only**

In volleyball the rally winner serves the next rally, so `point_winner[i] =
serving_team[i+1]` is exact. That means labeling `gt_serving_team` per
rally is sufficient to validate any chain: no separate point-winner label
and no volleyball end-rule heuristic are needed.

This matters in practice because RallyCut's GT set is a **subset of each
match**, not the whole match. The end-rule heuristic (`_resolve_last_rally`
in `match_stats.py`) assumes scores near 15 or 21 — it would mis-score the
"last" GT rally when that rally is actually mid-match. Keeping the metric
to a pure per-rally serving-team comparison avoids that trap entirely.

Predicted serving team comes directly from `RallyActions.serving_team`
(the `team` attribute of the classified serve action). We do NOT call
`compute_match_scores` here — it applies the end-rule and forward-pass
attribution that only make sense on a full match.

No DB access here — GT arrives on ``RallyData`` via the rally loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.tracking.action_classifier import RallyActions


@dataclass
class ScoreMetrics:
    """Scalar outputs of the score metric pass."""

    score_accuracy: float  # fraction of labeled rallies where pred serving == gt serving
    n_rallies_scored: int  # number of rallies with a non-null gt_serving_team
    n_videos_with_any_gt: int  # videos contributing at least one scored rally


def compute_score_metrics(
    pred_by_video: dict[str, list[tuple[int, RallyActions]]],
    gt_lookup: dict[str, tuple[str | None, str | None]],
) -> ScoreMetrics:
    """Compute ``score_accuracy`` across all videos.

    Args:
        pred_by_video: ``video_id -> list of (start_ms, RallyActions)``.
            Only the ``RallyActions`` is consumed (its ``rally_id`` and
            ``serving_team``); order does not matter.
        gt_lookup: ``rally_id -> (gt_serving_team, gt_point_winner)``. Only
            ``gt_serving_team`` is consumed; ``gt_point_winner`` is ignored
            (kept in the shape for backward compatibility with callers).

    Returns:
        ``ScoreMetrics`` with the scalar accuracy plus coverage counts.
    """
    total = 0
    correct = 0
    videos_with_any = 0

    for _video_id, rally_order in pred_by_video.items():
        scored_in_video = 0
        for _start_ms, ra in rally_order:
            rid: str = getattr(ra, "rally_id", "")
            if not rid:
                continue
            gt = gt_lookup.get(rid)
            if gt is None:
                continue
            gt_serving = gt[0]
            if gt_serving is None:
                continue
            total += 1
            scored_in_video += 1
            pred_serving = ra.serving_team  # may be None when no serve detected
            if pred_serving is not None and pred_serving == gt_serving:
                correct += 1
        if scored_in_video > 0:
            videos_with_any += 1

    return ScoreMetrics(
        score_accuracy=(correct / total) if total > 0 else 0.0,
        n_rallies_scored=total,
        n_videos_with_any_gt=videos_with_any,
    )
