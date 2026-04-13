"""Load side-switch flags from both GT sources.

Merges two sources of switch information, matching the API's
``computeNearSideByRally`` logic:

1. ``match_analysis_json.rallies[].sideSwitchDetected`` — pipeline-detected
2. ``rally.gt_side_switch`` — per-rally manual override from the Score GT UI

Per-rally overrides take precedence (``True`` forces a switch,
``False`` suppresses one, ``None`` defers to the analysis flag).
"""

from __future__ import annotations

from rallycut.evaluation.tracking.db import get_connection


def load_analysis_switch_flags(video_ids: set[str]) -> dict[str, bool]:
    """Load ``sideSwitchDetected`` flags from ``match_analysis_json``.

    Returns a mapping of ``rally_id -> True`` for rallies where the
    pipeline detected a side switch.  Rallies without the flag are
    absent (not mapped to ``False``).
    """
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    out: dict[str, bool] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, match_analysis_json FROM videos
            WHERE id IN ({placeholders})
        """, list(video_ids))
        for _vid, maj in cur.fetchall():
            if not maj or not isinstance(maj, dict):
                continue
            rallies_arr = maj.get("rallies", [])
            if not isinstance(rallies_arr, list):
                continue
            for entry in rallies_arr:
                if not isinstance(entry, dict):
                    continue
                rid = entry.get("rallyId") or entry.get("rally_id")
                if not rid:
                    continue
                flag = (
                    entry.get("sideSwitchDetected") is True
                    or entry.get("side_switch_detected") is True
                )
                if flag:
                    out[rid] = True
    return out


def resolve_side_flipped(
    video_ids: set[str],
    *,
    gt_only: bool = True,
) -> dict[str, bool]:
    """Compute per-rally ``side_flipped`` from merged switch sources.

    Walks rallies in chronological order per video, accumulating the
    XOR of resolved switch flags.

    Args:
        video_ids: Videos to process.
        gt_only: When True, only returns flips for rallies with
            ``gt_serving_team`` set.  When False, returns flips for
            ALL rallies in the video (needed by production_eval where
            the flip chain must span non-GT rallies too).

    Returns:
        ``rally_id -> flipped`` mapping.
    """
    if not video_ids:
        return {}
    analysis_flags = load_analysis_switch_flags(video_ids)
    placeholders = ", ".join(["%s"] * len(video_ids))
    gt_filter = "AND gt_serving_team IS NOT NULL" if gt_only else ""
    rally_info: dict[str, list[tuple[str, bool | None]]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT id, video_id, gt_side_switch FROM rallies
            WHERE video_id IN ({placeholders})
              {gt_filter}
            ORDER BY video_id, start_ms
        """, list(video_ids))
        for row in cur.fetchall():
            rally_info.setdefault(str(row[1]), []).append(
                (str(row[0]), row[2] if isinstance(row[2], bool) else None)
            )

    result: dict[str, bool] = {}
    for _vid, rally_list in rally_info.items():
        flipped = False
        for rid, gt_sw in rally_list:
            if gt_sw is not None:
                switched = bool(gt_sw)
            else:
                switched = analysis_flags.get(rid, False)
            if switched:
                flipped = not flipped
            result[rid] = flipped
    return result
