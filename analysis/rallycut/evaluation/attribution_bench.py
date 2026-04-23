"""Attribution benchmark primitives.

Single source of truth for matching GT actions to pipeline actions and categorizing
the result. Used by:

- ``scripts/phase0_lock_baseline.py`` — freezes a pipeline snapshot as baseline JSON.
- ``scripts/bench_attribution.py`` — scores any baseline-shaped JSON and supports
  diff vs baseline.

A "rally record" in the expected schema is::

    {
        "rally_id": str,
        "video_id": str,
        "fixture": str,
        "primary_track_ids": list[int] | None,
        "team_assignments": {str: str},            # pid -> team letter
        "serving_team": str | None,
        "gt_actions": [{frame, action, playerTrackId, ballX?, ballY?}, ...],
        "pipeline_actions": [{frame, action, playerTrackId, courtSide?,
                              confidence?, team?}, ...],
        "pipeline_contacts": [...],                # optional, passed through
    }

The matching tolerance is 10 frames by convention (the north-star metric defined
in the primitive-first plan §Phase 0.1 and §2 metric split).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

MATCH_TOLERANCE_FRAMES = 10

CATEGORIES = (
    "correct",
    "wrong_cross_team",
    "wrong_same_team",
    "wrong_unknown_team",
    "abstained",
    "missing",
)
WRONG_CATEGORIES = (
    "wrong_cross_team",
    "wrong_same_team",
    "wrong_unknown_team",
)


def classify_action(
    gt_action: dict[str, Any],
    pipeline_action: dict[str, Any] | None,
    team_assignments: dict[str, str],
) -> tuple[str, str]:
    """Return (category, reason) for a single (gt, pipeline) pair.

    category ∈ CATEGORIES. "abstained" is when pipeline emits an action but
    explicitly sets ``playerTrackId`` to None (first-class output of the Phase-2
    confidence-gated chooser). "missing" is when no pipeline action landed within
    tolerance of the GT frame.
    """
    if pipeline_action is None:
        return "missing", "no pipeline action within tolerance"

    pl_pid = pipeline_action.get("playerTrackId")
    if pl_pid is None:
        return "abstained", "pipeline emitted action with playerTrackId=None"

    gt_pid = gt_action["playerTrackId"]
    if int(gt_pid) == int(pl_pid):
        return "correct", ""

    gt_team = team_assignments.get(str(gt_pid))
    pl_team = team_assignments.get(str(pl_pid))
    if gt_team is None or pl_team is None:
        return "wrong_unknown_team", f"gt_team={gt_team!r} pl_team={pl_team!r}"
    if gt_team == pl_team:
        return "wrong_same_team", f"both on team {gt_team}"
    return "wrong_cross_team", f"gt={gt_team} vs pl={pl_team}"


def match_gt_to_pipeline(
    gt_actions: list[dict[str, Any]],
    pipeline_actions: list[dict[str, Any]],
    tolerance: int = MATCH_TOLERANCE_FRAMES,
) -> list[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Greedy nearest-first match of GT actions to pipeline actions.

    A pipeline action can only be claimed by one GT action. Among candidates in
    tolerance, prefer same action-type then smallest frame-distance.
    """
    used_pl_indices: set[int] = set()
    matches: list[tuple[dict, dict | None]] = []
    for gt in gt_actions:
        gt_frame = gt["frame"]
        gt_type = gt.get("action")
        candidates = [
            (idx, pl)
            for idx, pl in enumerate(pipeline_actions)
            if idx not in used_pl_indices
            and abs(pl["frame"] - gt_frame) <= tolerance
        ]
        if not candidates:
            matches.append((gt, None))
            continue
        candidates.sort(
            key=lambda c: (
                0 if c[1].get("action") == gt_type else 1,
                abs(c[1]["frame"] - gt_frame),
            )
        )
        chosen_idx, chosen_pl = candidates[0]
        used_pl_indices.add(chosen_idx)
        matches.append((gt, chosen_pl))
    return matches


def score_rally(
    rally: dict[str, Any],
    tolerance: int = MATCH_TOLERANCE_FRAMES,
) -> dict[str, Any]:
    """Score a single rally record. Mutates-nothing; returns a new dict with
    ``matches`` (per-GT-action category) and ``rally_totals`` (category counts).
    """
    gt = rally["gt_actions"]
    pipeline = rally.get("pipeline_actions", [])
    team_assignments = rally.get("team_assignments") or {}
    pairs = match_gt_to_pipeline(gt, pipeline, tolerance)

    per_action: list[dict[str, Any]] = []
    totals = dict.fromkeys(CATEGORIES, 0)
    totals["n_gt_actions"] = len(gt)
    for gt_action, pl_action in pairs:
        cat, reason = classify_action(gt_action, pl_action, team_assignments)
        totals[cat] += 1
        per_action.append(
            {
                "gt_frame": gt_action["frame"],
                "gt_action": gt_action.get("action"),
                "gt_pid": gt_action["playerTrackId"],
                "pl_frame": pl_action["frame"] if pl_action else None,
                "pl_action": pl_action.get("action") if pl_action else None,
                "pl_pid": pl_action.get("playerTrackId") if pl_action else None,
                "pl_confidence": pl_action.get("confidence") if pl_action else None,
                "pl_court_side": pl_action.get("courtSide") if pl_action else None,
                "category": cat,
                "reason": reason,
            }
        )
    return {"matches": per_action, "rally_totals": totals}


def _zero_totals() -> dict[str, int]:
    z = dict.fromkeys(CATEGORIES, 0)
    z["n_gt_actions"] = 0
    return z


def aggregate(
    rallies: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-fixture + combined metrics given scored rally records
    (each containing a ``rally_totals`` dict, typically produced by
    ``score_rally``).
    """
    per_fixture: dict[str, dict[str, int]] = {}
    combined = _zero_totals()
    for r in rallies:
        fx = r.get("fixture", "unknown")
        t = r.get("rally_totals") or score_rally(r)["rally_totals"]
        fx_totals = per_fixture.setdefault(fx, _zero_totals())
        for k, v in t.items():
            fx_totals[k] += v
            combined[k] += v

    def _rates(totals: dict[str, int]) -> dict[str, float]:
        n = totals["n_gt_actions"]
        wrong = sum(totals[k] for k in WRONG_CATEGORIES)
        return {
            "correct_rate": totals["correct"] / n if n else 0.0,
            "wrong_rate": wrong / n if n else 0.0,
            "missing_rate": totals["missing"] / n if n else 0.0,
            "abstained_rate": totals["abstained"] / n if n else 0.0,
        }

    return {
        "per_fixture": {
            fx: {"counts": t, "rates": _rates(t)} for fx, t in per_fixture.items()
        },
        "combined": {"counts": combined, "rates": _rates(combined)},
    }


def transition_matrix(
    baseline_rallies: list[dict[str, Any]],
    experiment_rallies: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Count transitions baseline.category → experiment.category on matched GT
    actions. Requires same rally_ids and same GT actions in both.

    Action identity = (rally_id, gt_frame, gt_pid).
    """
    baseline_idx = {
        (r["rally_id"], m["gt_frame"], m["gt_pid"]): m["category"]
        for r in baseline_rallies
        for m in r.get("matches", [])
    }
    matrix: dict[str, dict[str, int]] = {
        c_from: dict.fromkeys(CATEGORIES, 0) for c_from in CATEGORIES
    }
    for r in experiment_rallies:
        for m in r.get("matches", []):
            key = (r["rally_id"], m["gt_frame"], m["gt_pid"])
            c_from = baseline_idx.get(key)
            c_to = m["category"]
            if c_from is None:
                continue
            matrix[c_from][c_to] += 1
    return matrix
