"""Compute match statistics from tracked rally data in the database."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

console = Console()
logger = logging.getLogger(__name__)


def _load_rally_actions_and_positions(
    video_id: str,
) -> tuple[list[Any], list[Any], float, dict[str, list[Any]], Any, Any, int, int]:
    """Load actions, positions, and extended data from DB for all tracked rallies.

    Returns:
        (rally_actions_list, all_positions, video_fps,
         ball_positions_map, calibrator, match_analysis,
         video_width, video_height)
    """
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import get_connection
    from rallycut.tracking.action_classifier import (
        ActionType,
        ClassifiedAction,
        RallyActions,
    )
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.player_tracker import PlayerPosition

    # Load rally-level data (actions, positions, ball positions)
    rally_query = """
        SELECT
            r.id as rally_id,
            pt.positions_json,
            pt.actions_json,
            pt.ball_positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND pt.positions_json IS NOT NULL
          AND pt.actions_json IS NOT NULL
        ORDER BY r.start_ms
    """

    # Load video-level data (calibration, match analysis, dimensions)
    video_query = """
        SELECT
            court_calibration_json,
            match_analysis_json,
            width,
            height
        FROM videos
        WHERE id = %s
    """

    rally_actions_list: list[RallyActions] = []
    all_positions: list[PlayerPosition] = []
    ball_positions_map: dict[str, list[BallPosition]] = {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rally_rows = cur.fetchall()

            cur.execute(video_query, [video_id])
            video_row = cur.fetchone()

    # Parse video-level data
    calibrator = None
    match_analysis = None
    video_width = 1920
    video_height = 1080

    if video_row:
        cal_json, ma_json, v_width, v_height = video_row
        if v_width:
            video_width = cast(int, v_width)
        if v_height:
            video_height = cast(int, v_height)

        # Build calibrator from court corners
        if cal_json and isinstance(cal_json, list) and len(cal_json) == 4:
            try:
                calibrator = CourtCalibrator()
                corners = [(c["x"], c["y"]) for c in cal_json]
                calibrator.calibrate(corners)
            except (KeyError, ValueError, RuntimeError) as e:
                logger.warning("Failed to load court calibration: %s", e)
                calibrator = None

        if ma_json and isinstance(ma_json, dict):
            match_analysis = ma_json

    for row in rally_rows:
        rally_id_val, positions_json, actions_json, ball_json = row

        # Parse actions
        actions_data = cast(dict[str, Any] | None, actions_json)
        if not actions_data:
            continue

        actions: list[ClassifiedAction] = []
        team_assignments: dict[int, int] = {}

        if "teamAssignments" in actions_data:
            for tid_str, team_label in actions_data["teamAssignments"].items():
                team_int = 0 if team_label == "A" else 1 if team_label == "B" else -1
                if team_int >= 0:
                    team_assignments[int(tid_str)] = team_int

        for a in actions_data.get("actions", []):
            try:
                action_type = ActionType(a["action"])
            except (ValueError, KeyError):
                action_type = ActionType.UNKNOWN

            actions.append(ClassifiedAction(
                action_type=action_type,
                frame=a.get("frame", 0),
                ball_x=a.get("ballX", 0.0),
                ball_y=a.get("ballY", 0.0),
                velocity=a.get("velocity", 0.0),
                player_track_id=a.get("playerTrackId", -1),
                court_side=a.get("courtSide", "unknown"),
                confidence=a.get("confidence", 0.0),
                is_synthetic=a.get("isSynthetic", False),
                team=a.get("team", "unknown"),
            ))

        rally_id_str = str(rally_id_val)
        rally_actions_list.append(RallyActions(
            actions=actions,
            rally_id=rally_id_str,
            team_assignments=team_assignments,
        ))

        # Parse positions
        pos_json = cast(list[dict[str, Any]] | None, positions_json)
        if pos_json:
            for p in pos_json:
                all_positions.append(PlayerPosition(
                    frame_number=p.get("frameNumber", 0),
                    track_id=p.get("trackId", -1),
                    x=p.get("x", 0.0),
                    y=p.get("y", 0.0),
                    width=p.get("width", 0.0),
                    height=p.get("height", 0.0),
                    confidence=p.get("confidence", 0.0),
                ))

        # Parse ball positions
        bp_json = cast(list[dict[str, Any]] | None, ball_json)
        if bp_json:
            ball_list: list[BallPosition] = []
            for bp in bp_json:
                ball_list.append(BallPosition(
                    frame_number=bp.get("frameNumber", 0),
                    x=bp.get("x", 0.0),
                    y=bp.get("y", 0.0),
                    confidence=bp.get("confidence", 0.0),
                ))
            ball_positions_map[rally_id_str] = ball_list

    return (
        rally_actions_list, all_positions, 30.0,
        ball_positions_map, calibrator, match_analysis,
        video_width, video_height,
    )


@handle_errors
def compute_match_stats_cmd(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to compute match statistics for",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file for match statistics",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
) -> None:
    """Compute match statistics from tracked rally data.

    Reads action classifications and player positions from the database,
    computes per-player stats (kills, aces, efficiency), per-rally stats
    (terminal action, point winner), team stats, and score progression.

    Example:
        rallycut compute-match-stats abc123
        rallycut compute-match-stats abc123 -o stats.json
    """
    from rallycut.statistics.match_stats import compute_match_stats

    if not quiet:
        console.print(f"[bold]Match Statistics:[/bold] video {video_id[:8]}...")

    (
        rally_actions_list, all_positions, video_fps,
        ball_positions_map, calibrator, match_analysis,
        video_width, video_height,
    ) = _load_rally_actions_and_positions(video_id)

    if not rally_actions_list:
        console.print("[red]Error:[/red] No action data found in tracked rallies.")
        raise typer.Exit(1)

    if not quiet:
        console.print(f"  Loaded {len(rally_actions_list)} rallies with actions")
        if ball_positions_map:
            console.print(
                f"  Ball positions: {len(ball_positions_map)} rallies"
            )
        if calibrator:
            console.print("  Court calibration: loaded")
        if match_analysis:
            console.print("  Match analysis: loaded")

    stats = compute_match_stats(
        rally_actions_list=rally_actions_list,
        player_positions=all_positions,
        video_fps=video_fps,
        video_width=video_width,
        video_height=video_height,
        calibrator=calibrator,
        ball_positions_map=ball_positions_map,
        match_analysis=match_analysis,
    )

    result = stats.to_dict()

    if output:
        with open(output, "w") as f:
            json.dump(result, f, indent=2)
        if not quiet:
            console.print(f"\n[green]Stats saved to {output}[/green]")
    else:
        print(json.dumps(result))

    if not quiet:
        console.print(f"\n  Rallies: {stats.total_rallies}")
        console.print(f"  Total contacts: {stats.total_contacts}")
        console.print(f"  Players: {len(stats.player_stats)}")
        if stats.score_progression:
            console.print(
                f"  Score: {stats.final_score_a}-{stats.final_score_b}"
            )
        for ts in stats.team_stats:
            console.print(
                f"  Team {ts.team}: {ts.kills}K/{ts.attack_errors}E "
                f"({ts.kill_pct:.0%} kill%), {ts.aces} aces"
            )
