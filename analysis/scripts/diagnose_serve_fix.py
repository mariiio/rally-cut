"""Diagnose: what if we fix serve attribution using identity at detection time?

The serve player is known from match_players (server_player_id in match_analysis).
If we override the serve contact's track_id to the correct server track BEFORE
action classification, does _compute_expected_teams() cascade correctly?

Measures three levels:
1. Baseline (proximity)
2. Serve-only fix (override serve track_id from identity)
3. Serve + receive fix (first 2 contacts)

Usage:
    cd analysis
    uv run python scripts/diagnose_serve_fix.py
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    ContactSequence,
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


def _fix_contact_attribution(
    contact_seq: ContactSequence,
    fix_indices: set[int],
    track_to_player: dict[int, int],
    team_assignments: dict[int, int],
    expected_team_per_idx: dict[int, int],
) -> ContactSequence:
    """Override contact track_id for specified indices using identity.

    For each contact at a fix_index, find the nearest candidate whose
    identity (from track_to_player) maps to the expected team.
    """
    contacts = list(contact_seq.contacts)
    for idx in fix_indices:
        if idx >= len(contacts):
            continue
        c = contacts[idx]
        if not c.player_candidates:
            continue

        expected_team = expected_team_per_idx.get(idx)
        if expected_team is None:
            continue

        # Find nearest candidate on expected team
        best_tid = -1
        best_dist = float("inf")
        for tid, dist in c.player_candidates:
            pid = track_to_player.get(tid)
            if pid is None:
                continue
            cand_team = 0 if pid <= 2 else 1
            if cand_team != expected_team:
                continue
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid >= 0 and best_tid != c.player_track_id:
            contacts[idx] = Contact(
                frame=c.frame,
                ball_x=c.ball_x,
                ball_y=c.ball_y,
                velocity=c.velocity,
                direction_change_deg=c.direction_change_deg,
                player_track_id=best_tid,
                player_distance=best_dist,
                player_candidates=c.player_candidates,
                court_side=c.court_side,
                is_at_net=c.is_at_net,
                is_validated=c.is_validated,
                confidence=c.confidence,
                arc_fit_residual=c.arc_fit_residual,
            )

    return ContactSequence(
        contacts=contacts,
        net_y=contact_seq.net_y,
        ball_positions=contact_seq.ball_positions,
        rally_start_frame=contact_seq.rally_start_frame,
    )


def _get_server_team(
    match_teams: dict[int, int],
    track_to_player: dict[int, int],
    rally_gt_labels: list,
) -> int | None:
    """Get the server's team from GT or from match analysis."""
    # Use GT serve label to identify server track, then look up team
    gt_serve = next((gt for gt in rally_gt_labels if gt.action == "serve"), None)
    if gt_serve is None:
        return None
    gt_tid = gt_serve.player_track_id
    pid = track_to_player.get(gt_tid)
    if pid is None:
        return None
    return 0 if pid <= 2 else 1


def _evaluate_attribution(
    rally, contacts, match_teams, args,
) -> tuple[int, int]:
    """Run action classification and evaluate attribution. Returns (correct, total)."""
    ra = classify_rally_actions(contacts, rally.rally_id, match_team_assignments=match_teams)
    pred_actions = [a.to_dict() for a in ra.actions]
    real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

    fps = rally.fps or 30.0
    tol = max(1, round(fps * args.tolerance_ms / 1000))
    avail_tids = {pp["trackId"] for pp in rally.positions_json} if rally.positions_json else set()

    matches, _ = match_contacts(
        rally.gt_labels, real_pred,
        tolerance=tol, available_track_ids=avail_tids,
    )

    correct = 0
    total = 0
    for m in matches:
        if m.pred_frame is None:
            continue
        gt_tid = next(
            (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame), -1,
        )
        if gt_tid < 0 or gt_tid not in avail_tids:
            continue
        pred_tid = next(
            (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame), -1,
        )
        total += 1
        if pred_tid == gt_tid:
            correct += 1

    return correct, total


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
    match_teams_by_rally = _load_match_team_assignments(video_ids, 0.70)
    t2p_maps = _load_track_to_player_maps(video_ids)

    calibrators: dict[str, CourtCalibrator | None] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    console.print(f"\n[bold]Serve-fix diagnosis across {len(rallies)} rallies[/bold]\n")

    # Track results for each config
    results: dict[str, dict[str, int]] = {
        "baseline": {"correct": 0, "total": 0},
        "fix_serve": {"correct": 0, "total": 0},
        "fix_serve+receive": {"correct": 0, "total": 0},
        "fix_first3": {"correct": 0, "total": 0},
    }

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        cal = calibrators.get(rally.video_id)
        positions = [
            PlayerPos(
                frame_number=pp["frameNumber"], track_id=pp["trackId"],
                x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        mt = match_teams_by_rally.get(rally.rally_id)
        t2p = t2p_maps.get(rally.rally_id, {})

        contact_seq = detect_contacts(
            ball_positions=ball_positions, player_positions=positions,
            config=ContactDetectionConfig(), net_y=rally.court_split_y,
            frame_count=rally.frame_count or None, court_calibrator=cal,
        )

        # Baseline
        c, t = _evaluate_attribution(rally, contact_seq, mt, args)
        results["baseline"]["correct"] += c
        results["baseline"]["total"] += t

        if not t2p or not mt:
            # No identity data — same as baseline for all configs
            for key in ["fix_serve", "fix_serve+receive", "fix_first3"]:
                results[key]["correct"] += c
                results[key]["total"] += t
            continue

        # Determine server team from GT
        server_team = _get_server_team(mt, t2p, rally.gt_labels)

        if server_team is not None:
            # Volleyball rules: serve=team0, receive=team1, set=team1, attack=team1...
            # First contact (serve) is server_team, second (receive) is 1-server_team
            expected_per_idx = {
                0: server_team,               # serve
                1: 1 - server_team,           # receive
                2: 1 - server_team,           # set
            }

            # Fix serve only
            fixed = _fix_contact_attribution(
                contact_seq, {0}, t2p, mt, expected_per_idx,
            )
            c2, t2 = _evaluate_attribution(rally, fixed, mt, args)
            results["fix_serve"]["correct"] += c2
            results["fix_serve"]["total"] += t2

            # Fix serve + receive
            fixed2 = _fix_contact_attribution(
                contact_seq, {0, 1}, t2p, mt, expected_per_idx,
            )
            c3, t3 = _evaluate_attribution(rally, fixed2, mt, args)
            results["fix_serve+receive"]["correct"] += c3
            results["fix_serve+receive"]["total"] += t3

            # Fix first 3
            fixed3 = _fix_contact_attribution(
                contact_seq, {0, 1, 2}, t2p, mt, expected_per_idx,
            )
            c4, t4 = _evaluate_attribution(rally, fixed3, mt, args)
            results["fix_first3"]["correct"] += c4
            results["fix_first3"]["total"] += t4
        else:
            # No server info — same as baseline
            for key in ["fix_serve", "fix_serve+receive", "fix_first3"]:
                results[key]["correct"] += c
                results[key]["total"] += t

        if (i + 1) % 20 == 0:
            bl = results["baseline"]
            console.print(
                f"  [{i + 1}/{len(rallies)}] baseline {bl['correct']}/{bl['total']}"
                f" = {bl['correct']/max(1,bl['total']):.1%}"
            )

    # Results table
    table = Table(title="Serve-Fix Attribution Results")
    table.add_column("Configuration")
    table.add_column("Correct")
    table.add_column("Total")
    table.add_column("Accuracy")
    table.add_column("Delta")

    bl_acc = results["baseline"]["correct"] / max(1, results["baseline"]["total"])
    for key, label in [
        ("baseline", "Baseline (proximity)"),
        ("fix_serve", "Fix serve only"),
        ("fix_serve+receive", "Fix serve + receive"),
        ("fix_first3", "Fix first 3 contacts"),
    ]:
        r = results[key]
        acc = r["correct"] / max(1, r["total"])
        delta = acc - bl_acc
        table.add_row(
            label,
            str(r["correct"]),
            str(r["total"]),
            f"{acc:.1%}",
            f"{delta:+.1%}" if key != "baseline" else "—",
        )

    console.print(table)


if __name__ == "__main__":
    main()
