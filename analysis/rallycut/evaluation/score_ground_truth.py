"""Session 5 — score ground truth & score_accuracy metric helpers.

The score metric is the fraction of GT rallies whose predicted
(serving_team, point_winner) both match GT, chained at the match level via
``compute_match_scores``.

**GT storage & inference**

The labeler only writes ``gt_serving_team`` for every rally, plus
``gt_point_winner`` for the **last** rally of a match. Per-rally GT point
winners are then derived:

- Rally ``i`` (``i < N-1``): ``gt_point_winner = rally[i+1].gt_serving_team``.
  In volleyball the team that wins a rally serves the next one, so this is
  exact.
- Last rally ``N-1``: the labeler's explicit ``gt_point_winner`` is
  authoritative. A match is only scored when its last rally has an explicit
  label.

This module:

1. Groups per-rally predictions by video into match-ordered sequences.
2. Runs the production scorer (`compute_match_scores`) on each match. The
   predicted last rally is resolved by the volleyball end-rule heuristic
   already baked into that function.
3. Compares the resulting per-rally ``RallyScoreState`` to the derived GT.

Only matches where every rally has ``gt_serving_team`` set and the last
rally additionally has ``gt_point_winner`` set are scored; partial matches
are skipped and their count is reported back to the caller.

No DB access here — GT arrives on ``RallyData`` via the rally loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from rallycut.statistics.match_stats import RallyScoreState, compute_match_scores

if TYPE_CHECKING:
    from rallycut.tracking.action_classifier import RallyActions


@dataclass
class ScoreGtEntry:
    """Per-rally score ground truth extracted from the Rally row."""

    rally_id: str
    order_key: tuple[int, int]  # (start_ms, index) — stable per-match order
    gt_serving_team: str  # 'A' | 'B'
    gt_point_winner: str  # 'A' | 'B'


@dataclass
class ScoreMetrics:
    """Scalar outputs of the score metric pass."""

    score_accuracy: float  # primary
    score_chain_accuracy: float  # running score strict
    final_score_accuracy: float  # per-match final
    n_rallies_scored: int
    n_matches_scored: int
    n_matches_skipped_partial_gt: int


def _collect_gt(
    rallies_by_video: dict[str, list[tuple[int, RallyActions]]],
    gt_lookup: dict[str, tuple[str | None, str | None]],
) -> dict[str, list[ScoreGtEntry]]:
    """Return per-video GT lists for videos where every rally has
    ``gt_serving_team`` set and the last rally additionally has an explicit
    ``gt_point_winner``. Videos missing any of those are dropped.

    Each entry's ``gt_point_winner`` is either the user's explicit label
    (last rally only) or the next rally's ``gt_serving_team`` (all others).
    """
    out: dict[str, list[ScoreGtEntry]] = {}
    for video_id, rally_order in rallies_by_video.items():
        # Sort by start_ms so "next rally" is well-defined.
        ordered = sorted(rally_order, key=lambda t: t[0])
        # First pass: must have gt_serving_team on every rally, and an
        # explicit gt_point_winner on the last.
        raw: list[tuple[int, str, str, str | None]] = []  # (start_ms, rid, serving, explicit_winner)
        complete = True
        for start_ms, ra in ordered:
            rid = getattr(ra, "rally_id", "")
            serving, winner = gt_lookup.get(rid, (None, None))
            if serving is None:
                complete = False
                break
            raw.append((start_ms, rid, serving, winner))
        if not complete or not raw:
            continue
        if raw[-1][3] is None:
            # Last rally lacks an explicit point winner — skip this match.
            continue
        # Second pass: derive point winner for each rally.
        entries: list[ScoreGtEntry] = []
        for i, (start_ms, rid, serving, explicit) in enumerate(raw):
            derived: str
            if i < len(raw) - 1:
                derived = raw[i + 1][2]  # next rally's serving team
            else:
                assert explicit is not None  # verified above
                derived = explicit
            entries.append(
                ScoreGtEntry(
                    rally_id=rid,
                    order_key=(start_ms, i),
                    gt_serving_team=serving,
                    gt_point_winner=derived,
                )
            )
        out[video_id] = entries
    return out


def compute_score_metrics(
    pred_by_video: dict[str, list[tuple[int, RallyActions]]],
    gt_lookup: dict[str, tuple[str | None, str | None]],
) -> ScoreMetrics:
    """Compute score metrics across all videos.

    Args:
        pred_by_video: video_id -> list of (start_ms, RallyActions) in any
            order; this function sorts by start_ms before chaining.
        gt_lookup: rally_id -> (gt_serving_team, gt_point_winner). Either
            value may be None (unlabeled). Entire videos are skipped when any
            of their rallies is missing GT.

    Returns:
        ``ScoreMetrics`` with three accuracies + counts.
    """
    gt_per_video = _collect_gt(pred_by_video, gt_lookup)

    total_rallies = 0
    correct_rallies = 0
    chain_correct = 0
    matches_total = 0
    matches_final_correct = 0
    matches_skipped = sum(
        1 for vid in pred_by_video if vid not in gt_per_video and pred_by_video[vid]
    )

    for video_id, gt_entries in gt_per_video.items():
        rally_order = sorted(pred_by_video[video_id], key=lambda t: t[0])
        rally_actions_list = [ra for _, ra in rally_order]

        # Only score rallies that have GT (intersection on rally_id).
        gt_by_rid = {e.rally_id: e for e in gt_entries}
        if not any(ra.rally_id in gt_by_rid for ra in rally_actions_list):
            continue

        predicted_states: list[RallyScoreState] = compute_match_scores(rally_actions_list)
        pred_by_rid = {s.rally_id: s for s in predicted_states}

        matches_total += 1

        # Per-rally transition correctness (primary) + chain correctness.
        gt_a = 0
        gt_b = 0
        last_gt_a = 0
        last_gt_b = 0
        pred_last = predicted_states[-1] if predicted_states else None

        for e in gt_entries:
            if e.gt_point_winner == "A":
                gt_a += 1
            elif e.gt_point_winner == "B":
                gt_b += 1
            ps = pred_by_rid.get(e.rally_id)
            if ps is None:
                total_rallies += 1
                continue
            total_rallies += 1
            if (
                ps.serving_team == e.gt_serving_team
                and ps.point_winner == e.gt_point_winner
            ):
                correct_rallies += 1
            if ps.score_a == gt_a and ps.score_b == gt_b:
                chain_correct += 1
            last_gt_a, last_gt_b = gt_a, gt_b

        if (
            pred_last is not None
            and pred_last.score_a == last_gt_a
            and pred_last.score_b == last_gt_b
        ):
            matches_final_correct += 1

    def _safe(n: int, d: int) -> float:
        return float(n) / float(d) if d > 0 else 0.0

    return ScoreMetrics(
        score_accuracy=_safe(correct_rallies, total_rallies),
        score_chain_accuracy=_safe(chain_correct, total_rallies),
        final_score_accuracy=_safe(matches_final_correct, matches_total),
        n_rallies_scored=total_rallies,
        n_matches_scored=matches_total,
        n_matches_skipped_partial_gt=matches_skipped,
    )
