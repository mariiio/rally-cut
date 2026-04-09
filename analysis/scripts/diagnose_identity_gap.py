"""Diagnose: why doesn't perfect naming translate to attribution improvement?

For each attribution error, checks whether:
1. GT player's track has a match-level identity (player 1-4)
2. GT player's track is in the candidate list at the contact frame
3. If identity IS known, what prevents correct attribution

This reveals whether the bottleneck is track ID instability, candidate list
gaps, distance caps, or contact detection misses.

Usage:
    cd analysis
    uv run python scripts/diagnose_identity_gap.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    _load_match_team_assignments,
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", type=str)
    parser.add_argument("--tolerance-ms", type=int, default=167)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT.[/red]")
        return

    video_ids = {r.video_id for r in rallies}
    rally_ids = {r.rally_id for r in rallies}

    # Load match-level identity mappings
    track_to_player = _load_track_to_player_maps(video_ids)
    team_assignments = _load_match_team_assignments(video_ids, min_confidence=0.0)

    rallies_with_identity = sum(1 for r in rallies if r.rally_id in track_to_player)
    console.print(f"Rallies with match-level identity: {rallies_with_identity}/{len(rallies)}")

    calibrators: dict[str, CourtCalibrator | None] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    console.print(f"\n[bold]Identity gap diagnosis across {len(rallies)} rallies[/bold]\n")

    total = 0
    correct = 0

    # Error categories
    cat_no_identity = 0          # Rally has no match-level identity mapping
    cat_gt_no_mapping = 0        # GT track not in track_to_player
    cat_gt_not_in_cands = 0      # GT track not in candidate list
    cat_gt_in_cands_rank1 = 0    # GT in candidates, rank 1 on their team (identity would fix)
    cat_gt_in_cands_rank2plus = 0  # GT in candidates, not rank 1 on team (within-team error)
    cat_same_identity = 0        # Pred and GT map to same player ID (track ID instability)
    cat_identity_would_fix = 0   # Using identity would correctly re-attribute

    # Detailed: track stability around contact frame
    track_instability_cases: list[dict] = []

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        cal = calibrators.get(rally.video_id)
        positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        net_y_val = rally.court_split_y or 0.5
        per_rally_teams = classify_teams(positions, net_y_val)

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )
        contacts = contact_seq.contacts
        contact_by_frame = {c.frame: c for c in contacts}

        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions_list = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions_list if not a.get("isSynthetic")]

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

        t2p = track_to_player.get(rally.rally_id, {})

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        # Build position lookup for track stability analysis
        pos_by_frame: dict[int, list[PlayerPos]] = defaultdict(list)
        for p in positions:
            pos_by_frame[p.frame_number].append(p)

        for m in matches:
            if m.pred_frame is None:
                continue
            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in avail_tids:
                continue

            contact = contact_by_frame.get(m.pred_frame)
            if contact is None or not contact.player_candidates:
                continue

            pred_tid = contact.player_candidates[0][0]
            total += 1

            if pred_tid == gt_tid:
                correct += 1
                continue

            # Error case — diagnose
            if not t2p:
                cat_no_identity += 1
                continue

            gt_player_id = t2p.get(gt_tid)
            pred_player_id = t2p.get(pred_tid)

            if gt_player_id is None:
                cat_gt_no_mapping += 1
                continue

            # Check if GT track is in candidates
            cand_tids = [tid for tid, _ in contact.player_candidates]
            if gt_tid not in cand_tids:
                cat_gt_not_in_cands += 1
                continue

            # GT is in candidates — check if identity distinguishes
            if gt_player_id == pred_player_id:
                # Both tracks map to same player! Track ID instability
                cat_same_identity += 1

                # Check: are there two different track IDs with same player ID
                # near the contact frame?
                nearby_tids_for_player = set()
                for f_off in range(-10, 11):
                    for p in pos_by_frame.get(m.pred_frame + f_off, []):
                        if t2p.get(p.track_id) == gt_player_id:
                            nearby_tids_for_player.add(p.track_id)

                track_instability_cases.append({
                    "rally_id": rally.rally_id[:8],
                    "frame": m.pred_frame,
                    "gt_tid": gt_tid,
                    "pred_tid": pred_tid,
                    "player_id": gt_player_id,
                    "nearby_tids": sorted(nearby_tids_for_player),
                })
                continue

            # Identity differs — would using identity fix it?
            # Find all candidates with gt_player_id
            identity_cands = [
                (tid, d) for tid, d in contact.player_candidates
                if t2p.get(tid) == gt_player_id
            ]
            if identity_cands and identity_cands[0][0] == gt_tid:
                cat_identity_would_fix += 1
            else:
                # GT player ID in candidates but not the GT track
                # Could be another track with same player ID
                gt_team = per_rally_teams.get(gt_tid)
                same_team_cands = [
                    (tid, d) for tid, d in contact.player_candidates
                    if per_rally_teams.get(tid) == gt_team
                ]
                if same_team_cands and same_team_cands[0][0] == gt_tid:
                    cat_gt_in_cands_rank1 += 1
                else:
                    cat_gt_in_cands_rank2plus += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts, {correct} correct")

    errors = total - correct
    console.print(f"\n[bold]Results: {total} evaluable, {correct} correct ({correct/max(1,total):.1%})[/bold]")
    console.print(f"Errors: {errors}\n")

    table = Table(title="Error Breakdown by Identity Gap Category")
    table.add_column("Category", style="bold")
    table.add_column("Count")
    table.add_column("% of errors")
    table.add_column("Implication")

    rows = [
        ("No identity mapping", cat_no_identity, "Rally missing match analysis"),
        ("GT track unmapped", cat_gt_no_mapping, "Track not in track_to_player"),
        ("GT not in candidates", cat_gt_not_in_cands, "Detection gap at contact frame"),
        ("Same identity (track instability)", cat_same_identity, "Two tracks → same player near contact"),
        ("Identity would fix", cat_identity_would_fix, "Knowing player ID → correct attribution"),
        ("GT rank 1 on team", cat_gt_in_cands_rank1, "Team knowledge alone would fix"),
        ("GT rank 2+ on team", cat_gt_in_cands_rank2plus, "True within-team ambiguity"),
    ]
    for label, count, impl in rows:
        table.add_row(label, str(count), f"{count/max(1,errors):.1%}", impl)
    console.print(table)

    # Ceiling calculations
    fixable_by_identity = cat_identity_would_fix + cat_gt_in_cands_rank1
    console.print(f"\n[bold green]Fixable by perfect identity: {fixable_by_identity} ({fixable_by_identity/max(1,errors):.1%} of errors)[/bold green]")
    console.print(f"[bold green]Ceiling: {correct + fixable_by_identity}/{total} = {(correct + fixable_by_identity)/max(1,total):.1%}[/bold green]")

    unfixable = cat_gt_not_in_cands + cat_gt_in_cands_rank2plus + cat_same_identity + cat_no_identity + cat_gt_no_mapping
    console.print(f"[bold yellow]Unfixable by identity alone: {unfixable}[/bold yellow]")
    console.print(f"  - No identity: {cat_no_identity}")
    console.print(f"  - GT unmapped: {cat_gt_no_mapping}")
    console.print(f"  - Detection gap: {cat_gt_not_in_cands}")
    console.print(f"  - Track instability: {cat_same_identity}")
    console.print(f"  - Within-team ambiguity: {cat_gt_in_cands_rank2plus}")

    # Track instability details
    if track_instability_cases:
        console.print(f"\n[bold cyan]Track instability cases ({len(track_instability_cases)}):[/bold cyan]")
        inst_table = Table()
        inst_table.add_column("Rally")
        inst_table.add_column("Frame")
        inst_table.add_column("GT tid")
        inst_table.add_column("Pred tid")
        inst_table.add_column("Player ID")
        inst_table.add_column("Nearby tids for same player")
        for c in track_instability_cases[:20]:
            inst_table.add_row(
                c["rally_id"],
                str(c["frame"]),
                f"T{c['gt_tid']}",
                f"T{c['pred_tid']}",
                f"P{c['player_id']}",
                str(c["nearby_tids"]),
            )
        console.print(inst_table)


if __name__ == "__main__":
    main()
