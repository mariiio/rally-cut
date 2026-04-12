"""Investigate player attribution errors using oracle (permutation-invariant) matching.

The diagnose_attribution.py script uses literal track IDs, which conflates
genuine attribution errors with track ID drift from retracking. This script
uses Hungarian assignment (same as production_eval's player_attribution_oracle)
to first establish per-rally optimal track ID mapping, then categorizes only
the GENUINE errors (~16% of contacts) that survive oracle relabeling.

Error categories (after oracle relabeling):
- oracle_correct: correct under optimal permutation
- wrong_player_same_team: correct team but wrong individual (within-team error)
- wrong_team: attributed to player on wrong team
- missing_from_candidates: GT player (remapped) not in candidate list
- unmappable: GT track ID has no oracle mapping (no co-occurring predictions)

For each genuine error, extracts:
- Proximity margin (distance between chosen and correct player)
- Candidate rank of correct player
- Per-action breakdown
- Per-rally error clustering

Usage:
    cd analysis
    uv run python scripts/investigate_attribution.py
    uv run python scripts/investigate_attribution.py --rally <id>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


@dataclass
class OracleError:
    """A genuine attribution error surviving oracle relabeling."""

    rally_id: str
    video_id: str
    frame: int
    gt_action: str
    pred_action: str | None
    # Oracle-remapped IDs
    gt_remapped_tid: int
    pred_tid: int
    # Error classification
    error_type: str  # wrong_player_same_team, wrong_team, missing_from_candidates, unmappable
    # Proximity info
    pred_distance: float  # distance of chosen player
    gt_distance: float  # distance of correct (remapped) player
    distance_margin: float  # gt_distance - pred_distance (positive = GT was farther)
    gt_rank: int  # rank of correct player in candidates (-1 if not present)
    n_candidates: int
    # Team info
    pred_team: int | None  # 0=near, 1=far
    gt_team: int | None


def oracle_remap(
    matches: list[tuple[int, int]],
) -> dict[int, int]:
    """Build optimal GT→pred track ID mapping via Hungarian assignment.

    Args:
        matches: List of (gt_tid, pred_tid) pairs from matched contacts.

    Returns:
        Mapping from gt_tid to pred_tid under optimal assignment.
    """
    if not matches:
        return {}

    gt_ids: list[int] = []
    pred_ids: list[int] = []
    for gt_tid, pred_tid in matches:
        if gt_tid not in gt_ids:
            gt_ids.append(gt_tid)
        if pred_tid not in pred_ids:
            pred_ids.append(pred_tid)

    if not gt_ids or not pred_ids:
        return {}

    size = max(len(gt_ids), len(pred_ids))
    cost = np.zeros((size, size), dtype=np.float64)
    for gt_tid, pred_tid in matches:
        g_idx = gt_ids.index(gt_tid)
        p_idx = pred_ids.index(pred_tid)
        cost[g_idx, p_idx] -= 1.0

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping: dict[int, int] = {}
    for r, c in zip(row_ind, col_ind):
        if r < len(gt_ids) and c < len(pred_ids):
            mapping[gt_ids[r]] = pred_ids[c]
    return mapping


def investigate_rally(
    rally: RallyData,
    team_assignments: dict[int, int] | None,
    calibrator: CourtCalibrator | None,
    tolerance_frames: int,
) -> tuple[int, int, list[OracleError]]:
    """Investigate attribution errors for a single rally with oracle relabeling.

    Returns:
        (oracle_correct, oracle_evaluable, errors)
    """
    if not rally.ball_positions_json:
        return 0, 0, []

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

    avail_tids = {pp["trackId"] for pp in rally.positions_json} if rally.positions_json else set()

    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        court_calibrator=calibrator,
    )
    contacts = contact_seq.contacts
    contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

    rally_actions = classify_rally_actions(
        contact_seq, rally.rally_id,
        match_team_assignments=team_assignments,
    )
    pred_actions = [a.to_dict() for a in rally_actions.actions]
    real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

    fps = rally.fps or 30.0
    tol = max(1, round(fps * tolerance_frames / 1000))

    matches, _ = match_contacts(
        rally.gt_labels, real_pred,
        tolerance=tol, available_track_ids=avail_tids,
    )

    # Build per-rally teams
    net_y_val = rally.court_split_y or 0.5
    per_rally_teams = classify_teams(player_positions, net_y_val)
    if team_assignments:
        per_rally_teams.update(team_assignments)

    # Step 1: Collect (gt_tid, pred_tid) pairs for oracle
    tid_pairs: list[tuple[int, int]] = []
    matched_info: list[tuple] = []  # (match, gt_tid, pred_tid, contact)

    for m in matches:
        if m.pred_frame is None:
            continue
        if not m.player_evaluable:
            continue
        # Exclude blocks (same as production_eval)
        if m.gt_action == "block":
            continue

        gt_tid = next(
            (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
            -1,
        )
        pred_tid = next(
            (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
            -1,
        )

        if gt_tid < 0 or pred_tid < 0:
            continue
        if gt_tid not in avail_tids:
            continue

        contact = contact_by_frame.get(m.pred_frame)
        tid_pairs.append((gt_tid, pred_tid))
        matched_info.append((m, gt_tid, pred_tid, contact))

    if not matched_info:
        return 0, 0, []

    # Step 2: Oracle relabeling
    mapping = oracle_remap(tid_pairs)

    # Step 3: Classify each contact
    oracle_correct = 0
    oracle_evaluable = len(matched_info)
    errors: list[OracleError] = []

    for m, gt_tid, pred_tid, contact in matched_info:
        remapped_gt_tid = mapping.get(gt_tid)

        if remapped_gt_tid is not None and remapped_gt_tid == pred_tid:
            oracle_correct += 1
            continue

        # This is a genuine error. Classify it.
        if remapped_gt_tid is None:
            error_type = "unmappable"
            gt_dist = float("inf")
            pred_dist = contact.player_distance if contact else float("inf")
            gt_rank = -1
            n_cands = len(contact.player_candidates) if contact and contact.player_candidates else 0
        elif contact and contact.player_candidates:
            candidate_tids = [tid for tid, _ in contact.player_candidates]
            n_cands = len(candidate_tids)

            if remapped_gt_tid in candidate_tids:
                gt_rank = candidate_tids.index(remapped_gt_tid)
                gt_dist = next(d for tid, d in contact.player_candidates if tid == remapped_gt_tid)
            else:
                gt_rank = -1
                gt_dist = float("inf")

            pred_dist = contact.player_distance

            # Determine if same team or different team
            pred_team = per_rally_teams.get(pred_tid)
            gt_team = per_rally_teams.get(remapped_gt_tid)

            if gt_rank == -1:
                error_type = "missing_from_candidates"
            elif pred_team is not None and gt_team is not None and pred_team != gt_team:
                error_type = "wrong_team"
            else:
                error_type = "wrong_player_same_team"
        else:
            error_type = "missing_from_candidates"
            gt_dist = float("inf")
            pred_dist = float("inf")
            gt_rank = -1
            n_cands = 0

        pred_team_val = per_rally_teams.get(pred_tid)
        gt_team_val = per_rally_teams.get(remapped_gt_tid) if remapped_gt_tid else None

        errors.append(OracleError(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            frame=m.gt_frame,
            gt_action=m.gt_action,
            pred_action=m.pred_action,
            gt_remapped_tid=remapped_gt_tid if remapped_gt_tid is not None else -1,
            pred_tid=pred_tid,
            error_type=error_type,
            pred_distance=pred_dist,
            gt_distance=gt_dist,
            distance_margin=gt_dist - pred_dist,
            gt_rank=gt_rank,
            n_candidates=n_cands,
            pred_team=pred_team_val,
            gt_team=gt_team_val,
        ))

    return oracle_correct, oracle_evaluable, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Investigate player attribution errors (oracle-aware)")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument("--tolerance-ms", type=int, default=167)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies found with action GT.[/red]")
        return

    console.print(f"\n[bold]Attribution Investigation (oracle-aware) — {len(rallies)} rallies[/bold]\n")

    # Load calibrations and team assignments
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

    team_map = _load_match_team_assignments(video_ids)

    total_correct = 0
    total_evaluable = 0
    all_errors: list[OracleError] = []

    for i, rally in enumerate(rallies):
        cal = calibrators.get(rally.video_id)
        match_teams = team_map.get(rally.rally_id)

        correct, evaluable, errors = investigate_rally(
            rally, match_teams, cal, args.tolerance_ms,
        )
        total_correct += correct
        total_evaluable += evaluable
        all_errors.extend(errors)

        if (i + 1) % 50 == 0:
            console.print(
                f"  [{i + 1}/{len(rallies)}] "
                f"{total_correct}/{total_evaluable} correct, "
                f"{len(all_errors)} errors"
            )

    n_errors = len(all_errors)
    console.print(f"\n[bold]Oracle Attribution Results[/bold]")
    console.print(f"  Evaluable contacts: {total_evaluable}")
    console.print(f"  Oracle correct:     {total_correct} ({total_correct / max(1, total_evaluable):.1%})")
    console.print(f"  Genuine errors:     {n_errors} ({n_errors / max(1, total_evaluable):.1%})")

    if not all_errors:
        console.print("[green]No genuine errors found![/green]")
        return

    # === Section 1: Error Type Breakdown ===
    console.print(f"\n[bold cyan]1. Error Type Breakdown[/bold cyan]")
    type_counts: dict[str, int] = defaultdict(int)
    for e in all_errors:
        type_counts[e.error_type] += 1

    type_table = Table()
    type_table.add_column("Error Type")
    type_table.add_column("Count", justify="right")
    type_table.add_column("% of Errors", justify="right")
    type_table.add_column("% of All", justify="right")
    for etype in ["wrong_player_same_team", "wrong_team", "missing_from_candidates", "unmappable"]:
        c = type_counts.get(etype, 0)
        type_table.add_row(
            etype,
            str(c),
            f"{c / max(1, n_errors):.1%}",
            f"{c / max(1, total_evaluable):.1%}",
        )
    console.print(type_table)

    # === Section 2: Per-Action Breakdown ===
    console.print(f"\n[bold cyan]2. Per-Action Breakdown[/bold cyan]")
    action_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Count correct per action from all rallies (we only have errors here, need total)
    # We'll compute from total_evaluable and errors
    action_total: dict[str, int] = defaultdict(int)
    action_errors: dict[str, int] = defaultdict(int)
    for e in all_errors:
        action_errors[e.gt_action] += 1

    # We need per-action totals. Derive from all_errors + correct (approximate).
    # Better: count per-action evaluable from the raw data. Since we don't have that,
    # show error counts per action.
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("Errors", justify="right")
    act_table.add_column("% of Errors", justify="right")
    act_table.add_column("wrong_same_team", justify="right")
    act_table.add_column("wrong_team", justify="right")
    act_table.add_column("missing", justify="right")
    act_table.add_column("unmappable", justify="right")

    for action in ["serve", "receive", "set", "attack", "dig", "block"]:
        action_errs = [e for e in all_errors if e.gt_action == action]
        if not action_errs:
            continue
        n = len(action_errs)
        wst = sum(1 for e in action_errs if e.error_type == "wrong_player_same_team")
        wt = sum(1 for e in action_errs if e.error_type == "wrong_team")
        mis = sum(1 for e in action_errs if e.error_type == "missing_from_candidates")
        unm = sum(1 for e in action_errs if e.error_type == "unmappable")
        act_table.add_row(
            action, str(n), f"{n / max(1, n_errors):.1%}",
            str(wst), str(wt), str(mis), str(unm),
        )
    console.print(act_table)

    # === Section 3: Proximity Margin Analysis ===
    console.print(f"\n[bold cyan]3. Proximity Margin Analysis[/bold cyan]")
    # Only for errors where GT player was in candidates
    in_candidates = [e for e in all_errors if e.gt_rank >= 0]
    if in_candidates:
        margins = [e.distance_margin for e in in_candidates]
        gt_dists = [e.gt_distance for e in in_candidates]
        pred_dists = [e.pred_distance for e in in_candidates]
        ratios = [e.gt_distance / max(0.0001, e.pred_distance) for e in in_candidates]

        console.print(f"  Errors where GT was in candidates: {len(in_candidates)}/{n_errors}")
        console.print(f"  Errors where GT was NOT in candidates: {n_errors - len(in_candidates)}/{n_errors}")
        console.print()
        console.print(f"  Distance margin (GT - pred, positive = GT farther):")
        console.print(f"    Median: {np.median(margins):.4f}")
        console.print(f"    p25:    {np.percentile(margins, 25):.4f}")
        console.print(f"    p75:    {np.percentile(margins, 75):.4f}")
        console.print(f"    Mean:   {np.mean(margins):.4f}")
        console.print()
        console.print(f"  GT/pred distance ratio:")
        console.print(f"    Median: {np.median(ratios):.2f}x")
        console.print(f"    p25:    {np.percentile(ratios, 25):.2f}x")
        console.print(f"    p75:    {np.percentile(ratios, 75):.2f}x")
        console.print()

        # Rank distribution of GT player
        rank_counts: dict[int, int] = defaultdict(int)
        for e in in_candidates:
            rank_counts[e.gt_rank] += 1
        console.print(f"  GT player rank in candidates:")
        for rank in sorted(rank_counts):
            console.print(f"    #{rank + 1}: {rank_counts[rank]} ({rank_counts[rank] / len(in_candidates):.1%})")

        # Margin buckets
        console.print(f"\n  Margin buckets (how much farther was GT player):")
        buckets = [
            ("< 0.01 (very close)", lambda m: m < 0.01),
            ("0.01-0.03 (close)", lambda m: 0.01 <= m < 0.03),
            ("0.03-0.05 (moderate)", lambda m: 0.03 <= m < 0.05),
            ("0.05-0.10 (significant)", lambda m: 0.05 <= m < 0.10),
            ("> 0.10 (large)", lambda m: m >= 0.10),
        ]
        for label, pred in buckets:
            count = sum(1 for m in margins if pred(m))
            console.print(f"    {label}: {count} ({count / len(in_candidates):.1%})")
    else:
        console.print("  [yellow]No errors with GT in candidates[/yellow]")

    # === Section 4: Per-Video Error Clustering ===
    console.print(f"\n[bold cyan]4. Error Clustering by Video[/bold cyan]")
    video_errors: dict[str, int] = defaultdict(int)
    for e in all_errors:
        video_errors[e.video_id] += 1

    vid_table = Table()
    vid_table.add_column("Video ID (prefix)")
    vid_table.add_column("Errors", justify="right")
    vid_table.add_column("% of Errors", justify="right")
    for vid, count in sorted(video_errors.items(), key=lambda x: -x[1])[:10]:
        vid_table.add_row(
            vid[:8], str(count), f"{count / max(1, n_errors):.1%}",
        )
    console.print(vid_table)

    # === Section 5: Worst Rallies ===
    console.print(f"\n[bold cyan]5. Worst Rallies (most genuine errors)[/bold cyan]")
    rally_errors: dict[str, list[OracleError]] = defaultdict(list)
    for e in all_errors:
        rally_errors[e.rally_id].append(e)

    worst_table = Table()
    worst_table.add_column("Rally (prefix)")
    worst_table.add_column("Errors", justify="right")
    worst_table.add_column("Types")
    worst_table.add_column("Actions")
    worst_table.add_column("Margins")

    for rid, errs in sorted(rally_errors.items(), key=lambda x: -len(x[1]))[:15]:
        types = ", ".join(sorted({e.error_type for e in errs}))
        actions = ", ".join(sorted({e.gt_action for e in errs}))
        margins_str = ", ".join(
            f"{e.distance_margin:.3f}" for e in errs if e.gt_rank >= 0
        ) or "n/a"
        worst_table.add_row(
            rid[:8], str(len(errs)), types, actions, margins_str,
        )
    console.print(worst_table)

    # === Section 6: Detailed Error List (first 30) ===
    console.print(f"\n[bold cyan]6. Detailed Errors (first 30)[/bold cyan]")
    detail_table = Table()
    detail_table.add_column("Rally")
    detail_table.add_column("Frame", justify="right")
    detail_table.add_column("GT Act")
    detail_table.add_column("Pred Act")
    detail_table.add_column("Error Type")
    detail_table.add_column("GT TID", justify="right")
    detail_table.add_column("Pred TID", justify="right")
    detail_table.add_column("GT Rank", justify="right")
    detail_table.add_column("Pred Dist", justify="right")
    detail_table.add_column("GT Dist", justify="right")
    detail_table.add_column("Margin", justify="right")

    for e in all_errors[:30]:
        gt_rank_str = f"#{e.gt_rank + 1}" if e.gt_rank >= 0 else "N/A"
        detail_table.add_row(
            e.rally_id[:8],
            str(e.frame),
            e.gt_action,
            e.pred_action or "?",
            e.error_type,
            str(e.gt_remapped_tid),
            str(e.pred_tid),
            gt_rank_str,
            f"{e.pred_distance:.4f}" if e.pred_distance < float("inf") else "inf",
            f"{e.gt_distance:.4f}" if e.gt_distance < float("inf") else "inf",
            f"{e.distance_margin:.4f}" if e.distance_margin < float("inf") else "inf",
        )
    console.print(detail_table)

    # === Summary ===
    console.print(f"\n[bold green]Summary[/bold green]")
    console.print(f"  Oracle accuracy: {total_correct / max(1, total_evaluable):.1%} "
                  f"({total_correct}/{total_evaluable})")
    console.print(f"  Genuine errors: {n_errors} ({n_errors / max(1, total_evaluable):.1%})")

    dominant = max(type_counts, key=type_counts.get) if type_counts else "none"  # type: ignore[arg-type]
    console.print(f"  Dominant error type: {dominant} ({type_counts.get(dominant, 0)} cases)")

    if in_candidates:
        close_errors = sum(1 for m in margins if m < 0.03)
        console.print(f"  Close-margin errors (<0.03): {close_errors}/{len(in_candidates)} "
                      f"({close_errors / len(in_candidates):.1%})")
        console.print(f"  → These are cases where a better ranking model could help.")

        far_errors = sum(1 for m in margins if m >= 0.05)
        console.print(f"  Far-margin errors (≥0.05): {far_errors}/{len(in_candidates)} "
                      f"({far_errors / len(in_candidates):.1%})")
        console.print(f"  → These likely need fundamentally different signals (not just ranking).")


if __name__ == "__main__":
    main()
