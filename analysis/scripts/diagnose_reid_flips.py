"""Diagnose ReID Pass 3 flip patterns to understand why it's net negative.

For each contact with a ReID prediction, compares:
- Baseline attribution (after Pass 1+2, no ReID)
- ReID pick (what Pass 3 would change it to)
- Ground truth

Classifies each potential flip as:
- correct_fix: ReID fixes a wrong attribution
- harmful_flip: ReID breaks a correct attribution
- neutral: ReID picks same as baseline, or both wrong

Reports margin distributions, team violation rates, and per-video accuracy
to inform optimal margin gate and team constraints.

Usage:
    cd analysis
    uv run python scripts/diagnose_reid_flips.py
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.tracking.action_classifier import (
    _compute_expected_teams,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    RallyData,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    _train_reid_classifiers,
    _compute_reid_predictions_for_rally,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies with action GT[/red]")
        return

    video_ids = {r.video_id for r in rallies}

    # Load team assignments
    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    # Train ReID classifiers
    console.print("[bold]Training ReID classifiers...[/bold]")
    reid_classifiers = _train_reid_classifiers(video_ids)
    if not reid_classifiers:
        console.print("[red]No ReID classifiers trained (no reference crops)[/red]")
        return

    track_to_player_maps = _load_track_to_player_maps(video_ids)

    # Videos with ReID coverage
    reid_video_ids = set(reid_classifiers.keys())
    reid_rallies = [r for r in rallies if r.video_id in reid_video_ids]
    console.print(
        f"\n[bold]Diagnosing ReID flips on {len(reid_rallies)} rallies "
        f"({len(reid_video_ids)} videos with ReID)[/bold]\n"
    )

    # Collect per-contact diagnostics
    flips: list[dict[str, Any]] = []
    total_contacts_with_reid = 0
    total_contacts_evaluated = 0

    for idx, rally in enumerate(reid_rallies):
        t0 = time.monotonic()
        print(f"[{idx + 1}/{len(reid_rallies)}] {rally.rally_id[:8]}...", end=" ", flush=True)

        if not rally.ball_positions_json:
            print("skip (no ball)")
            continue

        # Run pipeline WITHOUT ReID
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions = []
        if rally.positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
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

        match_teams = match_teams_by_rally.get(rally.rally_id)
        baseline_actions = classify_rally_actions(
            contacts, rally.rally_id,
            match_team_assignments=match_teams,
        )

        # Get ReID predictions
        reid_cls = reid_classifiers.get(rally.video_id)
        t2p = track_to_player_maps.get(rally.rally_id, {})
        if reid_cls is None or not t2p or not rally.positions_json:
            print("skip (no ReID data)")
            continue

        reid_preds = _compute_reid_predictions_for_rally(
            reid_cls, rally.video_id, rally.start_ms, rally.fps,
            contacts.contacts, rally.positions_json, t2p,
        )

        if not reid_preds:
            print("skip (no predictions)")
            continue

        # Match GT to baseline predictions
        pred_dicts = [a.to_dict() for a in baseline_actions.actions if not a.is_synthetic]
        tolerance_frames = max(1, round(rally.fps * 167 / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json} if rally.positions_json else None

        matches, _ = match_contacts(
            rally.gt_labels, pred_dicts,
            tolerance=tolerance_frames,
            available_track_ids=avail_tids,
            team_assignments=match_teams,
        )

        # Build action lookup for expected teams
        expected_teams: list[int | None] = []
        if match_teams:
            expected_teams = _compute_expected_teams(
                baseline_actions.actions, match_teams,
            )

        action_by_frame = {a.frame: (i, a) for i, a in enumerate(baseline_actions.actions)}

        n_with_reid = 0
        for m in matches:
            if m.pred_frame is None or not m.player_evaluable:
                continue

            total_contacts_evaluated += 1

            reid_pred = reid_preds.get(m.pred_frame)
            if not reid_pred:
                continue

            n_with_reid += 1
            total_contacts_with_reid += 1

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            baseline_tid = next(
                (p.get("playerTrackId", -1) for p in pred_dicts if p.get("frame") == m.pred_frame),
                -1,
            )
            reid_tid = reid_pred["best_tid"]
            margin = reid_pred["margin"]

            # Team info
            baseline_team = match_teams.get(baseline_tid) if match_teams else None
            reid_team = match_teams.get(reid_tid) if match_teams else None
            gt_team = match_teams.get(gt_tid) if match_teams else None

            # Expected team from server-seeded chain
            action_info = action_by_frame.get(m.pred_frame)
            exp_team = None
            if action_info and expected_teams:
                action_idx, _ = action_info
                if action_idx < len(expected_teams):
                    exp_team = expected_teams[action_idx]

            # Classify flip
            would_switch = reid_tid != baseline_tid and reid_tid >= 0
            baseline_correct = baseline_tid == gt_tid
            reid_correct = reid_tid == gt_tid

            if would_switch:
                if reid_correct and not baseline_correct:
                    flip_type = "correct_fix"
                elif not reid_correct and baseline_correct:
                    flip_type = "harmful_flip"
                elif reid_correct and baseline_correct:
                    flip_type = "neutral_both_correct"
                else:
                    flip_type = "neutral_both_wrong"
            else:
                flip_type = "no_switch"

            # Team violation: does ReID pick violate expected team?
            team_violation = False
            if exp_team is not None and reid_team is not None:
                team_violation = reid_team != exp_team

            flips.append({
                "rally_id": rally.rally_id,
                "video_id": rally.video_id,
                "frame": m.pred_frame,
                "gt_action": m.gt_action,
                "gt_tid": gt_tid,
                "baseline_tid": baseline_tid,
                "reid_tid": reid_tid,
                "margin": margin,
                "baseline_correct": baseline_correct,
                "reid_correct": reid_correct,
                "would_switch": would_switch,
                "flip_type": flip_type,
                "baseline_team": baseline_team,
                "reid_team": reid_team,
                "gt_team": gt_team,
                "expected_team": exp_team,
                "team_violation": team_violation,
            })

        elapsed = time.monotonic() - t0
        print(f"{n_with_reid} ReID contacts ({elapsed:.1f}s)")

    # --- Aggregate Analysis ---
    console.print(f"\n[bold]ReID Flip Diagnosis[/bold]")
    console.print(f"  Rallies analyzed: {len(reid_rallies)}")
    console.print(f"  Contacts evaluated: {total_contacts_evaluated}")
    console.print(f"  Contacts with ReID: {total_contacts_with_reid}")

    # Flip type breakdown
    type_counts: dict[str, int] = defaultdict(int)
    for f in flips:
        type_counts[f["flip_type"]] += 1

    console.print(f"\n[bold]Flip Type Breakdown[/bold]")
    flip_table = Table()
    flip_table.add_column("Type", style="bold")
    flip_table.add_column("Count", justify="right")
    flip_table.add_column("% of ReID contacts", justify="right")
    for ft in ["correct_fix", "harmful_flip", "neutral_both_wrong", "neutral_both_correct", "no_switch"]:
        count = type_counts.get(ft, 0)
        pct = count / max(1, total_contacts_with_reid)
        style = "green" if ft == "correct_fix" else "red" if ft == "harmful_flip" else ""
        flip_table.add_row(ft, str(count), f"{pct:.1%}", style=style)
    console.print(flip_table)

    # Team violation analysis
    switches = [f for f in flips if f["would_switch"]]
    violations = [f for f in switches if f["team_violation"]]
    console.print(f"\n[bold]Team Violations[/bold]")
    console.print(f"  Switches: {len(switches)}")
    console.print(f"  Team violations: {len(violations)} ({len(violations)/max(1,len(switches)):.0%})")
    if violations:
        v_harmful = sum(1 for v in violations if v["flip_type"] == "harmful_flip")
        v_correct = sum(1 for v in violations if v["flip_type"] == "correct_fix")
        console.print(f"    Harmful: {v_harmful}, Correct: {v_correct}")

    # Margin analysis
    console.print(f"\n[bold]Margin Distribution[/bold]")
    correct_margins = [f["margin"] for f in flips if f["flip_type"] == "correct_fix"]
    harmful_margins = [f["margin"] for f in flips if f["flip_type"] == "harmful_flip"]

    if correct_margins:
        console.print(
            f"  Correct flips:  min={min(correct_margins):.3f} "
            f"median={sorted(correct_margins)[len(correct_margins)//2]:.3f} "
            f"max={max(correct_margins):.3f} (n={len(correct_margins)})"
        )
    if harmful_margins:
        console.print(
            f"  Harmful flips:  min={min(harmful_margins):.3f} "
            f"median={sorted(harmful_margins)[len(harmful_margins)//2]:.3f} "
            f"max={max(harmful_margins):.3f} (n={len(harmful_margins)})"
        )

    # Margin threshold sweep
    if correct_margins or harmful_margins:
        console.print(f"\n[bold]Margin Threshold Sweep[/bold]")
        sweep_table = Table()
        sweep_table.add_column("Threshold", justify="right")
        sweep_table.add_column("Correct", justify="right")
        sweep_table.add_column("Harmful", justify="right")
        sweep_table.add_column("Net", justify="right")
        sweep_table.add_column("Team-Constrained Net", justify="right")

        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            n_correct = sum(1 for m in correct_margins if m >= threshold)
            n_harmful = sum(1 for m in harmful_margins if m >= threshold)
            net = n_correct - n_harmful

            # Team-constrained: remove team violations
            tc_correct = sum(
                1 for f in flips
                if f["flip_type"] == "correct_fix" and f["margin"] >= threshold
                and not f["team_violation"]
            )
            tc_harmful = sum(
                1 for f in flips
                if f["flip_type"] == "harmful_flip" and f["margin"] >= threshold
                and not f["team_violation"]
            )
            tc_net = tc_correct - tc_harmful

            style = "bold" if threshold == 0.15 else ""
            sweep_table.add_row(
                f"{threshold:.2f}" + (" *" if threshold == 0.15 else ""),
                str(n_correct), str(n_harmful), str(net),
                str(tc_net),
                style=style,
            )
        console.print(sweep_table)
        console.print("[dim]* = current default threshold[/dim]")

    # Per-video breakdown
    console.print(f"\n[bold]Per-Video ReID Accuracy[/bold]")
    vid_table = Table()
    vid_table.add_column("Video", style="dim", max_width=10)
    vid_table.add_column("Contacts", justify="right")
    vid_table.add_column("Switches", justify="right")
    vid_table.add_column("Correct", justify="right")
    vid_table.add_column("Harmful", justify="right")
    vid_table.add_column("Net", justify="right")
    vid_table.add_column("Team Viol.", justify="right")

    by_video: dict[str, list[dict]] = defaultdict(list)
    for f in flips:
        by_video[f["video_id"]].append(f)

    for vid in sorted(by_video.keys()):
        vf = by_video[vid]
        n_contacts = len(vf)
        n_switches = sum(1 for f in vf if f["would_switch"])
        n_correct = sum(1 for f in vf if f["flip_type"] == "correct_fix")
        n_harmful = sum(1 for f in vf if f["flip_type"] == "harmful_flip")
        n_team_viol = sum(1 for f in vf if f["would_switch"] and f["team_violation"])
        vid_table.add_row(
            vid[:8], str(n_contacts), str(n_switches),
            str(n_correct), str(n_harmful), str(n_correct - n_harmful),
            str(n_team_viol),
        )

    console.print(vid_table)

    # Per-contact details for switches
    if switches:
        console.print(f"\n[bold]All Switches (detail)[/bold]")
        detail_table = Table()
        detail_table.add_column("Rally", style="dim", max_width=10)
        detail_table.add_column("Frame", justify="right")
        detail_table.add_column("Action")
        detail_table.add_column("GT→Base→ReID")
        detail_table.add_column("Margin", justify="right")
        detail_table.add_column("Type")
        detail_table.add_column("Team Viol.")

        for f in sorted(switches, key=lambda x: -x["margin"]):
            style = "green" if f["flip_type"] == "correct_fix" else "red" if f["flip_type"] == "harmful_flip" else ""
            detail_table.add_row(
                f["rally_id"][:8],
                str(f["frame"]),
                f["gt_action"],
                f"{f['gt_tid']}→{f['baseline_tid']}→{f['reid_tid']}",
                f"{f['margin']:.3f}",
                f["flip_type"],
                "YES" if f["team_violation"] else "",
                style=style,
            )

        console.print(detail_table)


if __name__ == "__main__":
    main()
