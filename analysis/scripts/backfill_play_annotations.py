"""Backfill play annotation fields on all stored rally actions.

Reads each rally's already-stored actions_json, ball_positions_json,
and positions_json from the DB. Parses actions into ClassifiedAction
objects, runs ``annotate_rally_actions`` (purely additive — no ML
re-runs), and writes the annotated actions_json back. Court
calibration is loaded per video. Uncalibrated videos are skipped.

Touches ONLY the ``actions_json`` column on ``player_tracks``.
Does NOT re-run detection, tracking, contact detection, or action
classification. All base action labels are bit-identical before and
after — only new optional fields are added.

Usage
-----
    cd analysis
    uv run python scripts/backfill_play_annotations.py              # dry run
    uv run python scripts/backfill_play_annotations.py --apply      # write to DB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from production_eval import _build_calibrators  # noqa: E402

from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.statistics.play_annotations import annotate_rally_actions  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionType,
    ClassifiedAction,
    RallyActions,
)
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402

console = Console()


def _parse_ball(raw: list[dict]) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _parse_action(d: dict) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=ActionType(d["action"]),
        frame=d["frame"],
        ball_x=d.get("ballX", 0.0),
        ball_y=d.get("ballY", 0.0),
        velocity=d.get("velocity", 0.0),
        player_track_id=d.get("playerTrackId", -1),
        court_side=d.get("courtSide", "unknown"),
        confidence=d.get("confidence", 0.0),
        is_synthetic=d.get("isSynthetic", False),
        team=d.get("team", "unknown"),
    )


def _normalize_and_extract(
    stored: dict | list | None,
) -> tuple[list[dict], dict[str, Any]]:
    """Normalize any stored actions_json shape to (action_dicts, meta).

    The DB has been written in several shapes over time:

    1. **Flat ActionsData** (correct API shape)::

           {"rallyId": ..., "actions": [...], "numContacts": N,
            "actionSequence": [...], "teamAssignments": {...}, ...}

    2. **Nested RallyActions wrapper** (corrupted by an earlier buggy
       backfill that set ``out["actions"] = rally_actions.to_dict()``)::

           {"actions": {"rallyId": ..., "actions": [...], ...},
            "teamAssignments": {...}}

    3. **Bare list** (very old format)::

           {"actions": [...]}

    Returns ``(action_dicts, meta)`` where ``action_dicts`` is the list
    of ClassifiedAction.to_dict() dicts and ``meta`` holds everything
    else (teamAssignments, servingTeam). This lets the caller rebuild
    a canonical flat shape regardless of the input.
    """
    if not isinstance(stored, dict):
        return [], {}
    meta: dict[str, Any] = {}
    if "teamAssignments" in stored:
        meta["teamAssignments"] = stored["teamAssignments"]
    if "servingTeam" in stored:
        meta["servingTeam"] = stored["servingTeam"]

    inner = stored.get("actions")
    if isinstance(inner, list):
        # Shape 1 (flat) or Shape 3 (bare list). Pull rallyId/numContacts
        # from the top level if present so the re-serialized form
        # preserves them.
        for k in ("rallyId", "numContacts", "actionSequence"):
            if k in stored:
                meta[k] = stored[k]
        return inner, meta
    if isinstance(inner, dict):
        # Shape 2 (nested wrapper). The inner dict has the RallyActions
        # fields; prefer its teamAssignments/servingTeam if present.
        for k in ("teamAssignments", "servingTeam", "rallyId", "numContacts", "actionSequence"):
            if k in inner:
                meta[k] = inner[k]
        action_list = inner.get("actions")
        if isinstance(action_list, list):
            return action_list, meta
    return [], meta


def _build_flat_actions_json(
    rally_actions: RallyActions, meta: dict[str, Any]
) -> dict:
    """Build the canonical flat ActionsData shape the API expects.

    The DB column ``actions_json`` is typed as ``ActionsData`` in
    ``playerTrackingService.ts`` — the ``actions`` key MUST be a list
    of ``ActionInfo`` dicts, not a nested RallyActions wrapper.
    """
    out = rally_actions.to_dict()  # {rallyId, numContacts, actionSequence, actions[list], teamAssignments?, servingTeam?}
    # Preserve servingTeam from the stored meta if the fresh to_dict()
    # didn't emit one (it only emits when the RallyActions has a valid
    # serving player attached).
    if "servingTeam" in meta and "servingTeam" not in out:
        out["servingTeam"] = meta["servingTeam"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Write changes to DB (default is dry run)")
    args = parser.parse_args()

    console.print(f"[bold]Backfill play annotations ({'APPLY' if args.apply else 'DRY RUN'})[/bold]")

    query = """
        SELECT
            pt.id as pt_id,
            r.id as rally_id,
            r.video_id,
            pt.actions_json,
            pt.ball_positions_json,
            pt.positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.actions_json IS NOT NULL
          AND pt.positions_json IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    console.print(f"  {len(rows)} rallies with stored actions")

    video_ids = {row[2] for row in rows}
    calibrators = _build_calibrators(video_ids)
    n_calibrated = sum(1 for c in calibrators.values() if getattr(c, "is_calibrated", False))
    console.print(f"  {n_calibrated}/{len(video_ids)} videos have court calibration")

    n_annotated = 0
    n_skipped = 0
    n_no_actions = 0
    updates: list[tuple[str, str]] = []  # (new_json_str, pt_id)

    for idx, row in enumerate(rows, start=1):
        pt_id, rally_id, video_id, actions_json, ball_json, positions_json = row

        cal = calibrators.get(video_id)
        if cal is None or not getattr(cal, "is_calibrated", False):
            n_skipped += 1
            continue

        if not actions_json or not isinstance(actions_json, dict):
            n_no_actions += 1
            continue

        # Normalize any historical stored shape → flat (action_dicts, meta).
        action_dicts, meta = _normalize_and_extract(actions_json)
        if not action_dicts:
            n_no_actions += 1
            continue

        classified: list[ClassifiedAction] = []
        for ad in action_dicts:
            try:
                classified.append(_parse_action(ad))
            except (KeyError, ValueError):
                continue

        rally_actions = RallyActions(
            actions=classified,
            rally_id=rally_id,
        )
        # Restore team_assignments from the normalized meta.
        raw_teams = meta.get("teamAssignments", {})
        if raw_teams:
            rally_actions.team_assignments = {
                int(tid): 0 if team == "A" else 1
                for tid, team in raw_teams.items()
            }

        ball_positions = _parse_ball(ball_json or [])
        stats = annotate_rally_actions(
            rally_actions, ball_positions, positions_json or [], cal
        )

        # Always write the flat canonical shape — this repairs any
        # previously-nested rallies even if the annotate step was a
        # no-op (e.g. rally has no attacks/sets).
        new_json = _build_flat_actions_json(rally_actions, meta)
        updates.append((json.dumps(new_json), pt_id))
        if stats.attacks_annotated > 0 or stats.sets_annotated > 0:
            n_annotated += 1
        else:
            n_skipped += 1

        if idx % 50 == 0 or idx == len(rows):
            console.print(f"  [{idx}/{len(rows)}] annotated={n_annotated} skipped={n_skipped}")

    console.print()
    console.print(f"  total rallies:    {len(rows)}")
    console.print(f"  annotated:        {n_annotated}")
    console.print(f"  skipped (no cal): {n_skipped}")
    console.print(f"  no actions:       {n_no_actions}")
    console.print(f"  DB updates ready: {len(updates)}")

    if args.apply and updates:
        console.print("[bold]Writing to DB...[/bold]")
        with get_connection() as conn:
            with conn.cursor() as cur:
                for new_json_str, pt_id in updates:
                    cur.execute(
                        "UPDATE player_tracks SET actions_json = %s::jsonb WHERE id = %s",
                        [new_json_str, pt_id],
                    )
            conn.commit()
        console.print(f"[green]done[/green] updated {len(updates)} player_tracks rows")
    elif not args.apply:
        console.print("[yellow]dry run — pass --apply to write[/yellow]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
