"""Diagnose player attribution errors and their cascade to action classification.

Classifies each attribution error into:
- correct: predicted player == GT player
- wrong_nearest: GT player was in candidates but not ranked #1
- wrong_team: GT player exists in tracking but mapped to wrong team
- missing_player: GT player's track_id not in candidates at contact frame
- no_player: prediction has track_id=-1
- stale_gt: GT track_id doesn't exist in current tracking data

Also measures the court_side cascade: attribution wrong → court_side wrong → action wrong.

Usage:
    cd analysis
    uv run python scripts/diagnose_attribution.py
    uv run python scripts/diagnose_attribution.py --rally <id>
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)

# Reuse from eval_action_detection
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
    _load_match_team_assignments,
)

console = Console()


@dataclass
class AttributionDiag:
    """Diagnostic info for a single matched contact."""

    rally_id: str
    frame: int
    gt_action: str
    pred_action: str | None
    gt_track_id: int
    pred_track_id: int
    error_type: str  # correct, wrong_nearest, wrong_team, missing_player, no_player, stale_gt
    # For wrong_nearest: rank of GT player in candidates
    gt_rank: int = -1
    gt_distance: float = float("inf")
    pred_distance: float = float("inf")
    # Court side info
    pred_court_side: str = "unknown"
    gt_court_side: str = "unknown"  # what court_side would be with correct player
    court_side_wrong: bool = False
    action_wrong: bool = False


def _resolve_hypothetical_court_side(
    gt_track_id: int,
    team_assignments: dict[int, int] | None,
) -> str | None:
    """What court_side would be if the GT player were used."""
    if team_assignments and gt_track_id >= 0 and gt_track_id in team_assignments:
        return "near" if team_assignments[gt_track_id] == 0 else "far"
    return None


def diagnose_rally(
    rally: RallyData,
    team_assignments: dict[int, int] | None,
    calibrator: CourtCalibrator | None,
    tolerance_frames: int,
) -> list[AttributionDiag]:
    """Diagnose attribution errors for a single rally."""
    if not rally.ball_positions_json:
        return []

    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]

    player_positions: list[PlayerPos] = []
    if rally.positions_json:
        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"],
                y=pp["y"],
                width=pp["width"],
                height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        court_calibrator=calibrator,
    )
    contacts = contact_seq.contacts

    rally_actions = classify_rally_actions(
        contact_seq, rally.rally_id,
        match_team_assignments=team_assignments,
    )
    pred_actions = [a.to_dict() for a in rally_actions.actions]

    # Filter out synthetic
    real_pred_actions = [a for a in pred_actions if not a.get("isSynthetic")]

    # Available track IDs
    avail_tids: set[int] | None = None
    if rally.positions_json:
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

    matches, _ = match_contacts(
        rally.gt_labels, real_pred_actions,
        tolerance=tolerance_frames, available_track_ids=avail_tids,
    )

    # Build contact lookup by frame for candidates
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    results: list[AttributionDiag] = []

    for m in matches:
        if m.pred_frame is None:
            continue  # Unmatched GT (FN) — not relevant for attribution

        gt_tid = next(
            (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
            -1,
        )
        pred_tid = next(
            (a.get("playerTrackId", -1) for a in real_pred_actions if a.get("frame") == m.pred_frame),
            -1,
        )
        pred_court_side = next(
            (a.get("courtSide", "unknown") for a in real_pred_actions if a.get("frame") == m.pred_frame),
            "unknown",
        )
        pred_action = m.pred_action

        # Find the contact object for this prediction
        contact = contact_by_frame.get(m.pred_frame)

        # Classify error type
        if avail_tids is not None and gt_tid >= 0 and gt_tid not in avail_tids:
            error_type = "stale_gt"
            gt_rank = -1
            gt_dist = float("inf")
            pred_dist = float("inf")
        elif gt_tid == pred_tid:
            error_type = "correct"
            gt_rank = 0
            gt_dist = contact.player_distance if contact else float("inf")
            pred_dist = gt_dist
        elif pred_tid == -1:
            error_type = "no_player"
            gt_rank = -1
            gt_dist = float("inf")
            pred_dist = float("inf")
        elif contact and contact.player_candidates:
            candidate_tids = [tid for tid, _ in contact.player_candidates]
            if gt_tid in candidate_tids:
                gt_rank = candidate_tids.index(gt_tid)
                gt_dist = next(d for tid, d in contact.player_candidates if tid == gt_tid)
                pred_dist = contact.player_distance
                error_type = "wrong_nearest"
            else:
                error_type = "missing_player"
                gt_rank = -1
                gt_dist = float("inf")
                pred_dist = contact.player_distance if contact else float("inf")
        else:
            error_type = "missing_player"
            gt_rank = -1
            gt_dist = float("inf")
            pred_dist = float("inf")

        # Court side cascade: what would court_side be with the correct player?
        gt_court_side = _resolve_hypothetical_court_side(gt_tid, team_assignments)
        court_side_wrong = False
        if gt_court_side is not None and pred_court_side in ("near", "far"):
            court_side_wrong = gt_court_side != pred_court_side

        diag = AttributionDiag(
            rally_id=rally.rally_id,
            frame=m.gt_frame,
            gt_action=m.gt_action,
            pred_action=pred_action,
            gt_track_id=gt_tid,
            pred_track_id=pred_tid,
            error_type=error_type,
            gt_rank=gt_rank,
            gt_distance=gt_dist,
            pred_distance=pred_dist,
            pred_court_side=pred_court_side,
            gt_court_side=gt_court_side or "unknown",
            court_side_wrong=court_side_wrong,
            action_wrong=(m.gt_action != pred_action) if pred_action else True,
        )
        results.append(diag)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose player attribution errors")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument("--tolerance-ms", type=int, default=167, help="Matching tolerance in ms")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies found with action GT.[/red]")
        return

    # Load calibrations and team assignments (same as eval)
    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    console.print(f"\n[bold]Diagnosing attribution across {len(rallies)} rallies[/bold]\n")

    all_diags: list[AttributionDiag] = []
    per_rally_errors: dict[str, list[AttributionDiag]] = defaultdict(list)

    for i, rally in enumerate(rallies):
        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        teams = match_teams_by_rally.get(rally.rally_id)
        cal = calibrators.get(rally.video_id)

        diags = diagnose_rally(rally, teams, cal, tol)
        all_diags.extend(diags)
        per_rally_errors[rally.rally_id] = diags

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(rallies)}] processed")

    print(f"  [{len(rallies)}/{len(rallies)}] done\n")

    if not all_diags:
        console.print("[red]No matched contacts to diagnose.[/red]")
        return

    # === Error type breakdown ===
    error_counts: dict[str, int] = defaultdict(int)
    for d in all_diags:
        error_counts[d.error_type] += 1

    total = len(all_diags)
    console.print(f"[bold]Attribution Error Breakdown ({total} matched contacts):[/bold]")
    for etype in ["correct", "wrong_nearest", "missing_player", "no_player", "stale_gt"]:
        count = error_counts.get(etype, 0)
        pct = count / total * 100
        extra = ""
        if etype == "wrong_nearest":
            extra = " — GT player was candidate #2-4"
        elif etype == "missing_player":
            extra = " — GT player not in candidate list"
        elif etype == "no_player":
            extra = " — track_id = -1"
        elif etype == "stale_gt":
            extra = " — GT references old tracking"
        console.print(f"  {etype:20s} {pct:5.1f}%  ({count}){extra}")

    # === Per-action breakdown ===
    console.print()
    action_table = Table(title="Per-Action Attribution Error Breakdown")
    action_table.add_column("Action", style="bold")
    action_table.add_column("Total", justify="right")
    action_table.add_column("Correct", justify="right")
    action_table.add_column("Wrong Near", justify="right")
    action_table.add_column("Missing", justify="right")
    action_table.add_column("No Player", justify="right")
    action_table.add_column("Stale GT", justify="right")
    action_table.add_column("Accuracy", justify="right")

    for action in ["serve", "receive", "set", "attack", "dig", "block"]:
        action_diags = [d for d in all_diags if d.gt_action == action]
        if not action_diags:
            continue
        n = len(action_diags)
        correct = sum(1 for d in action_diags if d.error_type == "correct")
        wrong_near = sum(1 for d in action_diags if d.error_type == "wrong_nearest")
        missing = sum(1 for d in action_diags if d.error_type == "missing_player")
        no_player = sum(1 for d in action_diags if d.error_type == "no_player")
        stale = sum(1 for d in action_diags if d.error_type == "stale_gt")
        acc = correct / max(1, n - stale) * 100
        action_table.add_row(
            action.capitalize(), str(n), str(correct),
            str(wrong_near), str(missing), str(no_player), str(stale),
            f"{acc:.1f}%",
        )
    console.print(action_table)

    # === Wrong nearest: distance analysis ===
    wrong_nearest = [d for d in all_diags if d.error_type == "wrong_nearest"]
    if wrong_nearest:
        console.print(f"\n[bold]Wrong Nearest Player Analysis ({len(wrong_nearest)} cases):[/bold]")
        # Rank distribution
        rank_counts: dict[int, int] = defaultdict(int)
        for d in wrong_nearest:
            rank_counts[d.gt_rank] += 1
        for rank in sorted(rank_counts):
            console.print(f"  GT player was candidate #{rank + 1}: {rank_counts[rank]} cases")

        # Distance comparison
        dists = [(d.pred_distance, d.gt_distance) for d in wrong_nearest
                 if d.pred_distance < 1.0 and d.gt_distance < 1.0]
        if dists:
            avg_pred = sum(p for p, _ in dists) / len(dists)
            avg_gt = sum(g for _, g in dists) / len(dists)
            avg_ratio = sum(g / max(p, 1e-6) for p, g in dists) / len(dists)
            console.print(f"  Avg predicted player distance: {avg_pred:.4f}")
            console.print(f"  Avg GT player distance:        {avg_gt:.4f}")
            console.print(f"  Avg GT/pred distance ratio:    {avg_ratio:.2f}x")

    # === Court side cascade ===
    evaluable = [d for d in all_diags if d.error_type not in ("stale_gt",)]
    attr_wrong = [d for d in evaluable if d.error_type != "correct"]
    attr_wrong_cs_wrong = [d for d in attr_wrong if d.court_side_wrong]
    cs_wrong_action_wrong = [d for d in attr_wrong_cs_wrong if d.action_wrong]

    console.print(f"\n[bold]Court Side Cascade Analysis:[/bold]")
    console.print(f"  Attribution wrong:                    {len(attr_wrong)}/{len(evaluable)} "
                  f"({len(attr_wrong)/max(1,len(evaluable))*100:.1f}%)")
    console.print(f"  Attr wrong → court_side wrong:        {len(attr_wrong_cs_wrong)}/{len(attr_wrong)} "
                  f"({len(attr_wrong_cs_wrong)/max(1,len(attr_wrong))*100:.1f}%)")
    console.print(f"  Court_side wrong → action also wrong: {len(cs_wrong_action_wrong)}/{len(attr_wrong_cs_wrong)} "
                  f"({len(cs_wrong_action_wrong)/max(1,len(attr_wrong_cs_wrong))*100:.1f}%)")

    # How many total action errors are caused by court_side cascade?
    total_action_errors = sum(1 for d in evaluable if d.action_wrong)
    console.print(f"\n  Total action errors:                  {total_action_errors}")
    console.print(f"  Action errors from court_side cascade: {len(cs_wrong_action_wrong)} "
                  f"({len(cs_wrong_action_wrong)/max(1,total_action_errors)*100:.1f}% of action errors)")

    # === Worst rallies ===
    console.print()
    rally_table = Table(title="Worst Rallies by Attribution Error Rate (min 3 contacts)")
    rally_table.add_column("Rally ID", style="dim", max_width=12)
    rally_table.add_column("Contacts", justify="right")
    rally_table.add_column("Correct", justify="right")
    rally_table.add_column("Errors", justify="right")
    rally_table.add_column("Error%", justify="right")
    rally_table.add_column("Has Teams", justify="center")
    rally_table.add_column("Error Types", style="dim")

    rally_stats = []
    for rid, diags in per_rally_errors.items():
        evaluable_diags = [d for d in diags if d.error_type != "stale_gt"]
        if len(evaluable_diags) < 3:
            continue
        correct = sum(1 for d in evaluable_diags if d.error_type == "correct")
        errors = len(evaluable_diags) - correct
        error_rate = errors / len(evaluable_diags)
        has_teams = rid in match_teams_by_rally
        error_types = defaultdict(int)
        for d in evaluable_diags:
            if d.error_type != "correct":
                error_types[d.error_type] += 1
        rally_stats.append((rid, len(evaluable_diags), correct, errors, error_rate, has_teams, dict(error_types)))

    rally_stats.sort(key=lambda x: -x[4])
    for rid, n, correct, errors, rate, has_teams, etypes in rally_stats[:15]:
        etype_str = ", ".join(f"{k}={v}" for k, v in sorted(etypes.items(), key=lambda x: -x[1]))
        rally_table.add_row(
            rid[:8], str(n), str(correct), str(errors),
            f"{rate*100:.0f}%",
            "Y" if has_teams else "N",
            etype_str,
        )
    console.print(rally_table)

    # === Per-case detail for wrong_nearest (to understand what's going on) ===
    if wrong_nearest:
        console.print()
        detail_table = Table(title=f"Wrong Nearest Details (top 20 by distance ratio)")
        detail_table.add_column("Rally", style="dim", max_width=8)
        detail_table.add_column("Frame", justify="right")
        detail_table.add_column("GT Act", style="bold")
        detail_table.add_column("Pred Act")
        detail_table.add_column("GT TID", justify="right")
        detail_table.add_column("Pred TID", justify="right")
        detail_table.add_column("GT Rank", justify="right")
        detail_table.add_column("Pred Dist", justify="right")
        detail_table.add_column("GT Dist", justify="right")
        detail_table.add_column("Ratio", justify="right")
        detail_table.add_column("CS Wrong")

        sorted_wn = sorted(wrong_nearest, key=lambda d: d.gt_distance / max(d.pred_distance, 1e-6), reverse=True)
        for d in sorted_wn[:20]:
            ratio = d.gt_distance / max(d.pred_distance, 1e-6)
            detail_table.add_row(
                d.rally_id[:8], str(d.frame),
                d.gt_action, d.pred_action or "—",
                str(d.gt_track_id), str(d.pred_track_id),
                f"#{d.gt_rank + 1}",
                f"{d.pred_distance:.3f}", f"{d.gt_distance:.3f}",
                f"{ratio:.1f}x",
                "YES" if d.court_side_wrong else "",
            )
        console.print(detail_table)


if __name__ == "__main__":
    main()
