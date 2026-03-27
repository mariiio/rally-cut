"""Comprehensive error anatomy of the action detection + attribution pipeline.

For all GT rallies, categorizes every error into fixable vs fundamental buckets:

Attribution errors:
- pipeline_corrupted: detect_contacts got it right, reattribute_players broke it
- temporal_regression: proximity was right, temporal model overrode incorrectly
- both_wrong_gt_in_cands: both wrong but GT player was in candidate list (model limit)
- gt_not_in_candidates: GT player not in candidate list (detection/tracking gap)
- stale_gt: GT track_id not in current tracking data (GT needs update)

Action errors:
- Confusion matrix with per-rally breakdown
- Correlation with attribution errors (causal chain)

Contact detection errors:
- FN by GT action type (what contacts are we missing)
- FP by predicted action type (what false detections look like)

Usage:
    cd analysis
    uv run python scripts/analyze_error_anatomy.py
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    MatchResult,
    RallyData,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

ACTION_TYPES = ["serve", "receive", "set", "attack", "block", "dig"]


@dataclass
class ContactDiag:
    """Full diagnostic record for a single matched GT contact."""

    rally_id: str
    gt_frame: int
    gt_action: str
    gt_track_id: int

    pred_frame: int | None
    pred_action: str | None
    proximity_choice: int  # candidate #1 (nearest player)
    detect_choice: int  # after temporal model in detect_contacts
    final_choice: int  # after reattribute_players

    gt_in_candidates: bool
    gt_rank: int  # rank of GT player in candidates (-1 if not present)
    gt_in_tracking: bool  # is GT track_id in current tracking data?

    # Derived
    attribution_category: str = ""
    action_correct: bool = False
    attribution_correct: bool = False


def _classify_attribution_error(d: ContactDiag) -> str:
    """Classify an attribution error into a category."""
    if not d.gt_in_tracking:
        return "stale_gt"
    if d.final_choice == d.gt_track_id:
        return "correct"
    # Error cases below
    if d.detect_choice == d.gt_track_id:
        return "pipeline_corrupted"
    if d.proximity_choice == d.gt_track_id and d.detect_choice != d.gt_track_id:
        return "temporal_regression"
    if d.gt_in_candidates:
        return "both_wrong_gt_in_cands"
    return "gt_not_in_candidates"


def analyze_rally(
    rally: RallyData,
    team_assignments: dict[int, int] | None,
    calibrator: CourtCalibrator | None,
    tolerance_frames: int,
) -> tuple[list[ContactDiag], list[MatchResult], list[dict]]:
    """Run pipeline and diagnose every contact in a rally."""
    if not rally.ball_positions_json:
        return [], [], []

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

    # Run contact detection (captures proximity + temporal model choices)
    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=ContactDetectionConfig(),
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
        team_assignments=team_assignments,
        court_calibrator=calibrator,
    )
    contacts = contact_seq.contacts

    # Snapshot detect_contacts-level attribution before reattribute_players
    detect_attr: dict[int, int] = {}  # frame -> track_id from detect_contacts
    proximity_attr: dict[int, int] = {}  # frame -> candidate #1
    for c in contacts:
        detect_attr[c.frame] = c.player_track_id
        if c.player_candidates:
            proximity_attr[c.frame] = c.player_candidates[0][0]
        else:
            proximity_attr[c.frame] = -1

    # Run full action classification (includes reattribute_players)
    rally_actions = classify_rally_actions(
        contact_seq, rally.rally_id,
        match_team_assignments=team_assignments,
    )
    pred_actions = [a.to_dict() for a in rally_actions.actions]
    real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

    # Available track IDs
    avail_tids: set[int] | None = None
    if rally.positions_json:
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

    matches, unmatched = match_contacts(
        rally.gt_labels, real_pred,
        tolerance=tolerance_frames,
        available_track_ids=avail_tids,
    )

    # Build diagnostic records for matched contacts
    diags: list[ContactDiag] = []
    for m in matches:
        gt_tid = next(
            (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
            -1,
        )

        if m.pred_frame is not None:
            final_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )
            prox = proximity_attr.get(m.pred_frame, -1)
            det = detect_attr.get(m.pred_frame, -1)
        else:
            final_tid = -1
            prox = -1
            det = -1

        gt_in_tracking = True
        if avail_tids is not None and gt_tid >= 0:
            gt_in_tracking = gt_tid in avail_tids

        # Check if GT player was in candidate list
        gt_in_cands = False
        gt_rank = -1
        contact = next((c for c in contacts if c.frame == m.pred_frame), None) if m.pred_frame else None
        if contact and contact.player_candidates and gt_tid >= 0:
            cand_tids = [tid for tid, _ in contact.player_candidates]
            if gt_tid in cand_tids:
                gt_in_cands = True
                gt_rank = cand_tids.index(gt_tid)

        d = ContactDiag(
            rally_id=rally.rally_id,
            gt_frame=m.gt_frame,
            gt_action=m.gt_action,
            gt_track_id=gt_tid,
            pred_frame=m.pred_frame,
            pred_action=m.pred_action,
            proximity_choice=prox,
            detect_choice=det,
            final_choice=final_tid,
            gt_in_candidates=gt_in_cands,
            gt_rank=gt_rank,
            gt_in_tracking=gt_in_tracking,
        )
        d.attribution_category = _classify_attribution_error(d)
        d.action_correct = (m.gt_action == m.pred_action) if m.pred_action else False
        d.attribution_correct = d.attribution_category == "correct"
        diags.append(d)

    return diags, matches, unmatched


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found with action GT.[/red]")
        return

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

    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    console.print(f"\n[bold]Error Anatomy Analysis: {len(rallies)} rallies[/bold]\n")

    all_diags: list[ContactDiag] = []
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    per_rally: dict[str, list[ContactDiag]] = defaultdict(list)

    t0 = time.monotonic()
    for i, rally in enumerate(rallies):
        fps = rally.fps or 30.0
        tol = max(1, round(fps * 167 / 1000))
        teams = match_teams_by_rally.get(rally.rally_id)
        cal = calibrators.get(rally.video_id)

        diags, matches, unmatched = analyze_rally(rally, teams, cal, tol)
        all_diags.extend(diags)
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)
        per_rally[rally.rally_id] = diags

        elapsed = time.monotonic() - t0
        if (i + 1) % 10 == 0 or i == len(rallies) - 1:
            print(f"  [{i + 1}/{len(rallies)}] {elapsed:.1f}s")

    # ===================================================================
    # Section 1: Attribution Error Breakdown
    # ===================================================================
    matched = [d for d in all_diags if d.pred_frame is not None]
    total_matched = len(matched)

    attr_counts: Counter[str] = Counter()
    for d in matched:
        attr_counts[d.attribution_category] += 1

    console.print(f"\n{'=' * 70}")
    console.print(f"[bold]1. ATTRIBUTION ERROR BREAKDOWN ({total_matched} matched contacts)[/bold]")
    console.print(f"{'=' * 70}")

    categories = [
        ("correct", "Correct attribution", ""),
        ("pipeline_corrupted", "Pipeline corrupted", "detect_contacts right, reattribute broke it — FIXABLE"),
        ("temporal_regression", "Temporal model regression", "proximity was right, temporal overrode — FIXABLE"),
        ("both_wrong_gt_in_cands", "Both wrong, GT in candidates", "model limit, GT was available — PARTIALLY FIXABLE"),
        ("gt_not_in_candidates", "GT not in candidates", "detection/tracking gap — HARD TO FIX"),
        ("stale_gt", "Stale GT", "GT references old tracking — NOT AN ERROR"),
    ]

    for cat, label, note in categories:
        count = attr_counts.get(cat, 0)
        pct = count / max(1, total_matched) * 100
        suffix = f"  ({note})" if note else ""
        console.print(f"  {label:35s} {count:4d}  ({pct:5.1f}%){suffix}")

    evaluable = [d for d in matched if d.attribution_category != "stale_gt"]
    n_eval = len(evaluable)
    n_correct = sum(1 for d in evaluable if d.attribution_category == "correct")
    console.print(f"\n  Evaluable attribution accuracy: {n_correct}/{n_eval} = {n_correct / max(1, n_eval) * 100:.1f}%")

    # ===================================================================
    # Section 2: Attribution Errors by Action Type
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]2. ATTRIBUTION ERRORS BY ACTION TYPE[/bold]")
    console.print(f"{'=' * 70}")

    attr_table = Table()
    attr_table.add_column("Action", style="bold")
    attr_table.add_column("Total", justify="right")
    attr_table.add_column("Correct", justify="right")
    attr_table.add_column("Pipeline", justify="right")
    attr_table.add_column("Temporal", justify="right")
    attr_table.add_column("Both Wrong", justify="right")
    attr_table.add_column("No Cands", justify="right")
    attr_table.add_column("Stale", justify="right")
    attr_table.add_column("Eval Acc", justify="right")

    for action in ACTION_TYPES:
        action_diags = [d for d in matched if d.gt_action == action]
        if not action_diags:
            continue
        n = len(action_diags)
        counts = Counter(d.attribution_category for d in action_diags)
        ev = [d for d in action_diags if d.attribution_category != "stale_gt"]
        ev_correct = sum(1 for d in ev if d.attribution_category == "correct")
        attr_table.add_row(
            action.capitalize(),
            str(n),
            str(counts.get("correct", 0)),
            str(counts.get("pipeline_corrupted", 0)),
            str(counts.get("temporal_regression", 0)),
            str(counts.get("both_wrong_gt_in_cands", 0)),
            str(counts.get("gt_not_in_candidates", 0)),
            str(counts.get("stale_gt", 0)),
            f"{ev_correct / max(1, len(ev)) * 100:.0f}%",
        )
    console.print(attr_table)

    # ===================================================================
    # Section 3: Action Confusion Matrix
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]3. ACTION CONFUSION MATRIX[/bold]")
    console.print(f"{'=' * 70}")

    action_matched = [d for d in matched if d.pred_action is not None]
    action_correct = sum(1 for d in action_matched if d.action_correct)
    action_errors = [d for d in action_matched if not d.action_correct]
    console.print(f"  Action accuracy: {action_correct}/{len(action_matched)} = "
                  f"{action_correct / max(1, len(action_matched)) * 100:.1f}%")
    console.print(f"  Action errors: {len(action_errors)}\n")

    # Build confusion matrix
    confusion: Counter[tuple[str, str]] = Counter()
    for d in action_matched:
        if d.pred_action:
            confusion[(d.gt_action, d.pred_action)] += 1

    # Print confusion matrix
    cm_table = Table(title="GT (rows) vs Predicted (columns)")
    cm_table.add_column("GT \\ Pred", style="bold")
    for a in ACTION_TYPES:
        cm_table.add_column(a[:4].title(), justify="right")
    cm_table.add_column("Total", justify="right", style="bold")

    for gt_a in ACTION_TYPES:
        row_vals = []
        for pred_a in ACTION_TYPES:
            c = confusion.get((gt_a, pred_a), 0)
            if c == 0:
                row_vals.append(".")
            elif gt_a == pred_a:
                row_vals.append(f"[bold green]{c}[/bold green]")
            else:
                row_vals.append(f"[red]{c}[/red]")
        row_total = sum(confusion.get((gt_a, p), 0) for p in ACTION_TYPES)
        cm_table.add_row(gt_a.capitalize(), *row_vals, str(row_total))
    console.print(cm_table)

    # Top confusion pairs
    console.print("\n  [bold]Top Confusion Pairs:[/bold]")
    pair_counts: Counter[tuple[str, str]] = Counter()
    for d in action_errors:
        if d.pred_action:
            pair_counts[(d.gt_action, d.pred_action)] += 1
    for (gt, pred), count in pair_counts.most_common(10):
        console.print(f"    {gt:8s} -> {pred:8s}  {count:3d}")

    # ===================================================================
    # Section 4: Attribution -> Action Error Cascade
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]4. ATTRIBUTION -> ACTION ERROR CASCADE[/bold]")
    console.print(f"{'=' * 70}")

    # Among action errors, how many also had wrong attribution?
    action_err_with_attr_err = sum(
        1 for d in action_errors
        if d.attribution_category not in ("correct", "stale_gt")
    )
    action_err_with_attr_ok = len(action_errors) - action_err_with_attr_err

    console.print(f"  Total action errors:                          {len(action_errors)}")
    console.print(f"  Action errors WITH attribution error:         {action_err_with_attr_err} "
                  f"({action_err_with_attr_err / max(1, len(action_errors)) * 100:.0f}%)")
    console.print(f"  Action errors WITH correct attribution:       {action_err_with_attr_ok} "
                  f"({action_err_with_attr_ok / max(1, len(action_errors)) * 100:.0f}%)")

    # Breakdown: which attribution categories cause action errors?
    console.print("\n  [bold]Action errors by attribution category:[/bold]")
    attr_cat_to_action_err: Counter[str] = Counter()
    for d in action_errors:
        attr_cat_to_action_err[d.attribution_category] += 1
    for cat, _, _ in categories:
        c = attr_cat_to_action_err.get(cat, 0)
        if c > 0:
            console.print(f"    {cat:35s} {c:3d}")

    # Reverse: among attribution errors, how many cause action errors?
    attr_errors = [d for d in evaluable if d.attribution_category != "correct"]
    attr_err_action_wrong = sum(1 for d in attr_errors if not d.action_correct)
    console.print(f"\n  Attribution errors that cascade to action error: "
                  f"{attr_err_action_wrong}/{len(attr_errors)} "
                  f"({attr_err_action_wrong / max(1, len(attr_errors)) * 100:.0f}%)")

    # ===================================================================
    # Section 5: Contact Detection Errors (FN / FP)
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]5. CONTACT DETECTION ERRORS[/bold]")
    console.print(f"{'=' * 70}")

    fn_contacts = [d for d in all_diags if d.pred_frame is None]
    fp_contacts = all_unmatched

    total_gt = len(all_diags)
    tp = total_matched
    fn = len(fn_contacts)
    fp = len(fp_contacts)
    recall = tp / max(1, total_gt)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    console.print(f"  GT contacts: {total_gt}, TP: {tp}, FN: {fn}, FP: {fp}")
    console.print(f"  Recall: {recall:.1%}, Precision: {precision:.1%}, F1: {f1:.1%}\n")

    # FN by GT action type
    console.print("  [bold]False Negatives by GT Action Type (missed contacts):[/bold]")
    fn_by_action: Counter[str] = Counter()
    for d in fn_contacts:
        fn_by_action[d.gt_action] += 1

    fn_table = Table()
    fn_table.add_column("Action", style="bold")
    fn_table.add_column("FN Count", justify="right")
    fn_table.add_column("Total GT", justify="right")
    fn_table.add_column("FN Rate", justify="right")

    for action in ACTION_TYPES:
        fn_count = fn_by_action.get(action, 0)
        total_action_gt = sum(1 for d in all_diags if d.gt_action == action)
        fn_rate = fn_count / max(1, total_action_gt) * 100
        if total_action_gt > 0:
            fn_table.add_row(
                action.capitalize(),
                str(fn_count),
                str(total_action_gt),
                f"{fn_rate:.0f}%",
            )
    console.print(fn_table)

    # FP by predicted action type
    console.print("\n  [bold]False Positives by Predicted Action Type (spurious detections):[/bold]")
    fp_by_action: Counter[str] = Counter()
    for fp_pred in fp_contacts:
        fp_by_action[fp_pred.get("action", "unknown")] += 1

    for action, count in fp_by_action.most_common():
        console.print(f"    {action:12s} {count:3d}")

    # FN concentration by rally
    console.print("\n  [bold]FN Concentration by Rally (top 10):[/bold]")
    fn_by_rally: Counter[str] = Counter()
    for d in fn_contacts:
        fn_by_rally[d.rally_id] += 1
    for rid, count in fn_by_rally.most_common(10):
        rally_gt = sum(1 for d in all_diags if d.rally_id == rid)
        console.print(f"    {rid[:8]}  FN={count}/{rally_gt}")

    # ===================================================================
    # Section 6: Top 10 Worst Rallies
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]6. TOP 10 WORST RALLIES (combined error score)[/bold]")
    console.print(f"{'=' * 70}")

    rally_scores: list[tuple[str, int, int, int, int, int, float]] = []
    for rid, diags in per_rally.items():
        n_gt = len(diags)
        if n_gt < 2:
            continue
        n_fn = sum(1 for d in diags if d.pred_frame is None)
        matched_diags = [d for d in diags if d.pred_frame is not None]
        n_attr_err = sum(
            1 for d in matched_diags
            if d.attribution_category not in ("correct", "stale_gt")
        )
        n_action_err = sum(1 for d in matched_diags if not d.action_correct)
        total_errors = n_fn + n_attr_err + n_action_err
        error_rate = total_errors / max(1, n_gt)
        rally_scores.append((rid, n_gt, n_fn, n_attr_err, n_action_err, total_errors, error_rate))

    rally_scores.sort(key=lambda x: -x[5])

    worst_table = Table()
    worst_table.add_column("Rally ID", style="dim", max_width=12)
    worst_table.add_column("GT", justify="right")
    worst_table.add_column("FN", justify="right")
    worst_table.add_column("Attr Err", justify="right")
    worst_table.add_column("Act Err", justify="right")
    worst_table.add_column("Total Err", justify="right")
    worst_table.add_column("Attr Categories", style="dim")

    for rid, n_gt, n_fn, n_attr, n_act, total, rate in rally_scores[:10]:
        matched_diags = [d for d in per_rally[rid] if d.pred_frame is not None]
        cats = Counter(
            d.attribution_category for d in matched_diags
            if d.attribution_category not in ("correct", "stale_gt")
        )
        cat_str = ", ".join(f"{k}={v}" for k, v in cats.most_common())
        worst_table.add_row(
            rid[:8], str(n_gt), str(n_fn), str(n_attr), str(n_act),
            str(total), cat_str,
        )
    console.print(worst_table)

    # ===================================================================
    # Section 7: Action Error Concentration
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]7. ACTION ERROR CONCENTRATION[/bold]")
    console.print(f"{'=' * 70}")

    rally_action_errors: Counter[str] = Counter()
    for d in action_errors:
        rally_action_errors[d.rally_id] += 1

    # Distribution
    err_counts = sorted(rally_action_errors.values(), reverse=True)
    total_action_errs = sum(err_counts)
    if err_counts:
        top5_errs = sum(err_counts[:5])
        console.print(f"  Total action errors: {total_action_errs} across {len(err_counts)} rallies")
        console.print(f"  Top 5 rallies account for: {top5_errs}/{total_action_errs} "
                      f"({top5_errs / max(1, total_action_errs) * 100:.0f}%) of action errors")
        n_zero_err = len(per_rally) - len(rally_action_errors)
        console.print(f"  Rallies with 0 action errors: {n_zero_err}/{len(per_rally)}")

    # Histogram
    console.print("\n  [bold]Action errors per rally distribution:[/bold]")
    bins = [0, 1, 2, 3, 5, 100]
    rally_err_list = [rally_action_errors.get(rid, 0) for rid in per_rally]
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 100:
            label = f"  {lo}+"
        else:
            label = f"  {lo}" if lo == hi - 1 else f"  {lo}-{hi - 1}"
        count = sum(1 for e in rally_err_list if lo <= e < hi)
        console.print(f"    {label:8s} errors: {count:3d} rallies")

    # ===================================================================
    # Section 8: Fixable vs Fundamental Summary
    # ===================================================================
    console.print(f"\n{'=' * 70}")
    console.print("[bold]8. FIXABLE vs FUNDAMENTAL ERROR BUDGET[/bold]")
    console.print(f"{'=' * 70}")

    n_stale = attr_counts.get("stale_gt", 0)
    n_pipeline = attr_counts.get("pipeline_corrupted", 0)
    n_temporal = attr_counts.get("temporal_regression", 0)
    n_both_wrong = attr_counts.get("both_wrong_gt_in_cands", 0)
    n_no_cands = attr_counts.get("gt_not_in_candidates", 0)
    n_attr_correct = attr_counts.get("correct", 0)

    console.print("\n  [bold]Attribution Error Budget:[/bold]")
    console.print(f"    Correct:                       {n_attr_correct:4d}")
    console.print(f"    Stale GT (ignore):             {n_stale:4d}")
    console.print("    --- Fixable ---")
    console.print(f"    Pipeline corrupted:            {n_pipeline:4d}  (fix reattribute logic)")
    console.print(f"    Temporal model regression:      {n_temporal:4d}  (improve/disable temporal model)")
    console.print("    --- Partially fixable ---")
    console.print(f"    Both wrong, GT in candidates:  {n_both_wrong:4d}  (better model could pick right)")
    console.print("    --- Fundamental limit ---")
    console.print(f"    GT not in candidates:          {n_no_cands:4d}  (tracking/detection gap)")

    fixable = n_pipeline + n_temporal
    partial = n_both_wrong
    fundamental = n_no_cands
    total_err = fixable + partial + fundamental
    console.print(f"\n    Total attribution errors (excl stale): {total_err}")
    console.print(f"    Fixable:      {fixable:4d} ({fixable / max(1, total_err) * 100:.0f}%)")
    console.print(f"    Partial:      {partial:4d} ({partial / max(1, total_err) * 100:.0f}%)")
    console.print(f"    Fundamental:  {fundamental:4d} ({fundamental / max(1, total_err) * 100:.0f}%)")

    # Action error budget
    console.print("\n  [bold]Action Error Budget:[/bold]")
    action_err_pure = sum(
        1 for d in action_errors
        if d.attribution_category in ("correct", "stale_gt")
    )
    action_err_from_attr = len(action_errors) - action_err_pure
    console.print(f"    Total action errors:            {len(action_errors)}")
    console.print(f"    Pure classification errors:     {action_err_pure:4d}  (attribution was correct)")
    console.print(f"    Caused by attribution errors:   {action_err_from_attr:4d}  (wrong player -> wrong features)")

    # Contact detection budget
    console.print("\n  [bold]Contact Detection Error Budget:[/bold]")
    console.print(f"    False negatives:  {fn:4d}  (missed real contacts)")
    console.print(f"    False positives:  {fp:4d}  (spurious detections)")

    # Overall
    console.print("\n  [bold]Overall Pipeline Error Summary:[/bold]")
    console.print(f"    Contact F1:        {f1:.1%}  ({fn} FN + {fp} FP)")
    eval_total = len(evaluable)
    eval_attr_correct = sum(1 for d in evaluable if d.attribution_correct)
    console.print(f"    Attribution:       {eval_attr_correct / max(1, eval_total):.1%}  "
                  f"({eval_total - eval_attr_correct} errors / {eval_total} evaluable)")
    console.print(f"    Action accuracy:   {action_correct / max(1, len(action_matched)):.1%}  "
                  f"({len(action_errors)} errors / {len(action_matched)} matched)")

    # Ceiling if all fixable attribution errors were fixed
    if fixable > 0:
        # How many action errors would be eliminated if fixable attr errors were fixed?
        action_err_from_fixable = sum(
            1 for d in action_errors
            if d.attribution_category in ("pipeline_corrupted", "temporal_regression")
        )
        new_attr_acc = (eval_attr_correct + fixable) / max(1, eval_total)
        new_action_err = len(action_errors) - action_err_from_fixable
        new_action_acc = (len(action_matched) - new_action_err) / max(1, len(action_matched))
        console.print("\n  [bold]Ceiling if fixable attribution errors fixed:[/bold]")
        console.print(f"    Attribution:     {new_attr_acc:.1%}  (+{fixable / max(1, eval_total) * 100:.1f}pp)")
        console.print(f"    Action accuracy: {new_action_acc:.1%}  "
                      f"(+{action_err_from_fixable / max(1, len(action_matched)) * 100:.1f}pp, "
                      f"assumes {action_err_from_fixable} action errors also fixed)")


if __name__ == "__main__":
    main()
