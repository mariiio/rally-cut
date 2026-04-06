"""Diagnose expected-team errors in the action classification pipeline.

Replays _compute_expected_teams() step-by-step on all GT rallies and
categorises every court-side / expected-team error:
  - first_action: chain first diverged HERE (a misclassified action)
  - cascade: chain was already wrong from an earlier error
  - serve_seed: serve team identification was incorrect

Also measures whether parabolic pre-contact ball trajectory would have
predicted the correct side at each error contact (spoiler: ~53%, barely
above chance — see attribution_trajectory_experiment.md).

Key finding (2026-04-06): the expected-team chain is only 45% accurate
overall. 65% of errors are cascades from ~160 first-action errors.
The reattribute_players() guards prevent the chain from ever acting,
making Pass 2 effectively dead code.

Usage:
    cd analysis
    uv run python scripts/diagnose_expected_team_errors.py
    uv run python scripts/diagnose_expected_team_errors.py --rally <id>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.split import add_split_argument, apply_split
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    _compute_expected_teams,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

# Re-use data loading from eval_action_detection
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)


@dataclass
class ErrorDetail:
    """Details of a single expected-team error."""

    rally_id: str
    contact_idx: int  # position in matched contact list
    gt_frame: int
    gt_action: str
    gt_team: int  # 0=near, 1=far
    pred_action: str
    pred_team: int | None  # expected_team from chain
    category: str  # first_action | cascade | serve_seed
    chain_first_wrong_idx: int  # index of first wrong action in chain
    traj_prediction: str  # "near", "far", "unknown"
    traj_correct: bool | None  # True if trajectory predicted correct side


def _load_match_team_assignments(
    video_ids: set[str],
    rally_positions: dict[str, list[PlayerPosition]] | None = None,
) -> dict[str, dict[int, int]]:
    """Load match-level team assignments (confidence >= 0.70)."""
    from rallycut.tracking.match_tracker import build_match_team_assignments

    if not video_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """

    result: dict[str, dict[int, int]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            rows = cur.fetchall()

    for _vid, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        result.update(build_match_team_assignments(
            ma_json, min_confidence=0.70, rally_positions=rally_positions,
        ))
    return result


def _parabolic_trajectory_court_side(
    ball_positions: list[BallPosition],
    contact_frame: int,
    net_y: float,
    look_back: int = 15,
    min_points: int = 5,
) -> tuple[str, float]:
    """Predict court side from multi-frame parabolic trajectory fit.

    Fits y(t) = at^2 + bt + c to pre-contact ball positions and
    extrapolates forward. Returns (predicted_side, r_squared).

    Note: this achieves ~53% accuracy on error contacts (~69% overall),
    barely above chance. Kept for diagnostic completeness.
    """
    frames = []
    ys = []
    for bp in ball_positions:
        if (bp.confidence >= 0.3
                and contact_frame - look_back <= bp.frame_number < contact_frame):
            frames.append(bp.frame_number)
            ys.append(bp.y)

    if len(frames) < min_points:
        return "unknown", 0.0

    t = np.array(frames, dtype=np.float64) - contact_frame
    y = np.array(ys, dtype=np.float64)

    try:
        coeffs = np.polyfit(t, y, 2)
    except (np.linalg.LinAlgError, ValueError):
        return "unknown", 0.0

    y_pred = np.polyval(coeffs, t)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - ss_res / max(ss_tot, 1e-12)

    y_at_future = np.polyval(coeffs, 5.0)

    if y_at_future > net_y + 0.02:
        return "near", float(r_squared)
    elif y_at_future < net_y - 0.02:
        return "far", float(r_squared)
    else:
        return "unknown", float(r_squared)


def _replay_expected_teams_with_gt(
    pred_actions: list[ClassifiedAction],
    gt_labels: list[GtLabel],
    team_assignments: dict[int, int],
    ball_positions: list[BallPosition],
    net_y: float,
    tolerance_frames: int,
    rally_id: str,
) -> list[ErrorDetail]:
    """Replay expected-team computation and categorise errors.

    Matches predicted actions to GT, computes expected_teams from the
    forward chain, then categorises each mismatch as first_action
    (chain first diverges), cascade (already wrong), or serve_seed.
    """
    pred_dicts = [a.to_dict() for a in pred_actions]
    real_preds = [p for p in pred_dicts if not p.get("isSynthetic")]
    avail_tids = {a.player_track_id for a in pred_actions if a.player_track_id >= 0}

    matches, _unmatched = match_contacts(
        gt_labels, real_preds, tolerance=tolerance_frames,
        available_track_ids=avail_tids, team_assignments=team_assignments,
    )

    expected_teams = _compute_expected_teams(pred_actions, team_assignments)

    pred_frame_to_idx: dict[int, int] = {}
    for i, a in enumerate(pred_actions):
        pred_frame_to_idx[a.frame] = i

    # Build (match_idx, expected_team, gt_team) for each matched contact
    matched_with_teams: list[tuple[int, int | None, int | None]] = []
    for m_idx, m in enumerate(matches):
        if m.pred_frame is None:
            matched_with_teams.append((m_idx, None, None))
            continue

        pred_idx = pred_frame_to_idx.get(m.pred_frame)
        exp_team = expected_teams[pred_idx] if pred_idx is not None else None

        gt_team = None
        for gt in gt_labels:
            if gt.frame == m.gt_frame and gt.player_track_id >= 0:
                gt_team = team_assignments.get(gt.player_track_id)
                break

        matched_with_teams.append((m_idx, exp_team, gt_team))

    # Find first divergence point
    first_wrong_idx = -1
    for i, (_, exp, gt) in enumerate(matched_with_teams):
        if exp is not None and gt is not None and exp != gt:
            first_wrong_idx = i
            break

    errors: list[ErrorDetail] = []

    for i, (m_idx, exp_team, gt_team) in enumerate(matched_with_teams):
        m = matches[m_idx]

        if exp_team is None or gt_team is None:
            continue
        if exp_team == gt_team:
            continue

        # Categorise the error
        if first_wrong_idx < 0:
            category = "serve_seed"
        elif i == first_wrong_idx:
            # Check if serve team seed was wrong
            serve_team = None
            for a in pred_actions:
                if a.action_type == ActionType.SERVE and a.player_track_id >= 0:
                    serve_team = team_assignments.get(a.player_track_id)
                    break
            gt_serve_team = None
            for gt in gt_labels:
                if gt.action == "serve" and gt.player_track_id >= 0:
                    gt_serve_team = team_assignments.get(gt.player_track_id)
                    break

            if (serve_team is not None and gt_serve_team is not None
                    and serve_team != gt_serve_team):
                category = "serve_seed"
            else:
                category = "first_action"
        else:
            category = "cascade"

        traj_side, _traj_conf = _parabolic_trajectory_court_side(
            ball_positions, m.gt_frame, net_y,
        )
        gt_side = "near" if gt_team == 0 else "far"
        traj_correct = (traj_side == gt_side) if traj_side != "unknown" else None

        errors.append(ErrorDetail(
            rally_id=rally_id,
            contact_idx=i,
            gt_frame=m.gt_frame,
            gt_action=m.gt_action,
            gt_team=gt_team,
            pred_action=m.pred_action or "?",
            pred_team=exp_team,
            category=category,
            chain_first_wrong_idx=first_wrong_idx,
            traj_prediction=traj_side,
            traj_correct=traj_correct,
        ))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose expected-team errors in action classification",
    )
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument("--tolerance-ms", type=int, default=167)
    add_split_argument(parser)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    rallies = apply_split(rallies, args)

    if not rallies:
        console.print("[red]No rallies found with action GT.[/red]")
        return

    # Load team assignments with position-based verification
    video_ids = {r.video_id for r in rallies}
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in r.positions_json
            ]

    match_teams_by_rally = _load_match_team_assignments(
        video_ids, rally_positions=rally_pos_lookup,
    )

    console.print(f"\n[bold]Diagnosing expected-team errors on {len(rallies)} rallies[/bold]")
    n_with_teams = sum(1 for r in rallies if r.rally_id in match_teams_by_rally)
    console.print(f"  Match teams: {n_with_teams}/{len(rallies)} rallies\n")

    all_errors: list[ErrorDetail] = []
    total_matched = 0
    total_with_team_info = 0
    total_expected_correct = 0
    skipped_no_ball = 0
    skipped_no_teams = 0

    for idx, rally in enumerate(rallies):
        print(f"[{idx + 1}/{len(rallies)}] {rally.rally_id[:8]}...", end=" ", flush=True)

        match_teams = match_teams_by_rally.get(rally.rally_id)
        if not match_teams:
            print("skip (no teams)")
            skipped_no_teams += 1
            continue

        if not rally.ball_positions_json:
            print("skip (no ball)")
            skipped_no_ball += 1
            continue

        ball_positions = [
            BallPosition(
                frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions: list[PlayerPosition] = []
        if rally.positions_json:
            player_positions = [
                PlayerPosition(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )

        rally_actions = classify_rally_actions(
            contacts, rally.rally_id,
            match_team_assignments=match_teams,
        )

        tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))

        errors = _replay_expected_teams_with_gt(
            rally_actions.actions,
            rally.gt_labels,
            match_teams,
            ball_positions,
            contacts.net_y,
            tolerance_frames,
            rally.rally_id,
        )

        # Count matched contacts with team info
        pred_dicts = [a.to_dict() for a in rally_actions.actions]
        real_preds = [p for p in pred_dicts if not p.get("isSynthetic")]
        avail_tids = {a.player_track_id for a in rally_actions.actions
                      if a.player_track_id >= 0}
        matches, _ = match_contacts(
            rally.gt_labels, real_preds, tolerance=tolerance_frames,
            available_track_ids=avail_tids, team_assignments=match_teams,
        )

        exp_teams = _compute_expected_teams(rally_actions.actions, match_teams)
        pred_frame_to_idx = {a.frame: i for i, a in enumerate(rally_actions.actions)}

        for m in matches:
            if m.pred_frame is not None:
                total_matched += 1
                for gt in rally.gt_labels:
                    if gt.frame == m.gt_frame and gt.player_track_id >= 0:
                        gt_team = match_teams.get(gt.player_track_id)
                        if gt_team is not None:
                            total_with_team_info += 1
                            pi = pred_frame_to_idx.get(m.pred_frame)
                            if pi is not None and exp_teams[pi] == gt_team:
                                total_expected_correct += 1
                        break

        n_errors = len(errors)
        print(f"{n_errors} errors")
        all_errors.extend(errors)

    # --- Summary ---
    console.print(f"\n{'=' * 60}")
    console.print("[bold]Expected-Team Error Diagnosis[/bold]")
    console.print(f"{'=' * 60}")
    console.print(f"Rallies processed: {len(rallies) - skipped_no_ball - skipped_no_teams}")
    console.print(f"  Skipped (no ball): {skipped_no_ball}")
    console.print(f"  Skipped (no teams): {skipped_no_teams}")
    console.print(f"Total matched contacts: {total_matched}")
    console.print(f"  With team info: {total_with_team_info}")
    console.print(f"  Expected-team correct: {total_expected_correct}/{total_with_team_info} "
                  f"({total_expected_correct / max(1, total_with_team_info):.1%})")
    console.print(f"  Expected-team errors: {len(all_errors)}")

    if not all_errors:
        console.print("\n[green]No expected-team errors found![/green]")
        return

    # Category breakdown
    console.print(f"\n[bold]Error Categories[/bold]")
    cats: dict[str, int] = defaultdict(int)
    for e in all_errors:
        cats[e.category] += 1

    cat_table = Table()
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("%", justify="right")
    for cat in ["first_action", "cascade", "serve_seed"]:
        n = cats.get(cat, 0)
        pct = n / len(all_errors) * 100
        cat_table.add_row(cat, str(n), f"{pct:.1f}%")
    console.print(cat_table)

    # Cascade chain analysis
    errors_by_rally: dict[str, list[ErrorDetail]] = defaultdict(list)
    for e in all_errors:
        errors_by_rally[e.rally_id].append(e)

    chain_lengths: list[int] = []
    for errs in errors_by_rally.values():
        if any(e.category == "first_action" for e in errs):
            chain_lengths.append(len(errs))

    console.print(f"\n[bold]Cascade Chains[/bold]")
    console.print(f"  Rallies with chain errors: {len(chain_lengths)}")
    if chain_lengths:
        console.print(f"  Avg errors per rally: {np.mean(chain_lengths):.1f}")
        console.print(f"  Max errors per rally: {max(chain_lengths)}")
        n_cascade = cats.get("cascade", 0)
        n_first = cats.get("first_action", 0)
        console.print(f"  Fixing {n_first} first-action errors would also fix "
                      f"up to {n_cascade} cascade errors ({n_first + n_cascade} total)")

    # Trajectory analysis at error contacts
    console.print(f"\n[bold]Trajectory Prediction at Error Contacts[/bold]")
    traj_correct = sum(1 for e in all_errors if e.traj_correct is True)
    traj_wrong = sum(1 for e in all_errors if e.traj_correct is False)
    traj_unknown = sum(1 for e in all_errors if e.traj_correct is None)
    traj_evaluable = traj_correct + traj_wrong

    traj_table = Table()
    traj_table.add_column("Trajectory", style="bold")
    traj_table.add_column("Count", justify="right")
    traj_table.add_column("%", justify="right")
    traj_table.add_row("Correct", str(traj_correct),
                       f"{traj_correct / max(1, traj_evaluable):.1%} of evaluable")
    traj_table.add_row("Wrong", str(traj_wrong),
                       f"{traj_wrong / max(1, traj_evaluable):.1%} of evaluable")
    traj_table.add_row("Unknown (insufficient data)", str(traj_unknown),
                       f"{traj_unknown / max(1, len(all_errors)):.1%} of all")
    console.print(traj_table)

    # Per-action breakdown
    console.print(f"\n[bold]Errors by GT Action Type[/bold]")
    act_counts: dict[str, int] = defaultdict(int)
    for e in all_errors:
        act_counts[e.gt_action] += 1
    for act in sorted(act_counts, key=act_counts.get, reverse=True):  # type: ignore[arg-type]
        console.print(f"  {act}: {act_counts[act]}")

    # Per-rally detail table (top 20)
    console.print(f"\n[bold]Per-Rally Error Detail (top 20)[/bold]")
    detail_table = Table()
    detail_table.add_column("Rally", style="dim", max_width=10)
    detail_table.add_column("Errors", justify="right")
    detail_table.add_column("First", justify="right")
    detail_table.add_column("Cascade", justify="right")
    detail_table.add_column("Serve", justify="right")
    detail_table.add_column("Traj OK", justify="right")

    for rid in sorted(errors_by_rally, key=lambda r: len(errors_by_rally[r]), reverse=True)[:20]:
        errs = errors_by_rally[rid]
        n_first = sum(1 for e in errs if e.category == "first_action")
        n_casc = sum(1 for e in errs if e.category == "cascade")
        n_serve = sum(1 for e in errs if e.category == "serve_seed")
        n_traj_ok = sum(1 for e in errs if e.traj_correct is True)
        n_traj_eval = sum(1 for e in errs if e.traj_correct is not None)
        traj_str = f"{n_traj_ok}/{n_traj_eval}" if n_traj_eval > 0 else "-"
        detail_table.add_row(
            rid[:10], str(len(errs)),
            str(n_first), str(n_casc), str(n_serve), traj_str,
        )
    console.print(detail_table)


if __name__ == "__main__":
    main()
