"""Diagnose action classification confusions.

For each misclassified contact, extracts the 15 ActionFeatures and compares
feature distributions of correct vs confused predictions. Flags cases where
tracking errors cause misclassification.

Usage:
    cd analysis
    uv run python scripts/diagnose_action_confusions.py
    uv run python scripts/diagnose_action_confusions.py --rally <id>
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.action_type_classifier import (
    ActionFeatures,
    extract_action_features,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    MatchResult,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose action confusions")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument(
        "--tolerance-ms", type=int, default=167,
        help="Time tolerance in ms for matching (default: 167)",
    )
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    # Load calibrators and match teams
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

    console.print(f"\n[bold]Diagnosing action confusions across {len(rallies)} rallies[/bold]\n")

    # Collect all matches with their features
    all_correct: list[tuple[str, ActionFeatures]] = []  # (action, features)
    all_confused: list[tuple[str, str, ActionFeatures, str]] = []  # (gt, pred, features, rally_id)
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    for i, rally in enumerate(rallies):
        print(f"  [{i + 1}/{len(rallies)}] {rally.rally_id[:8]}...", end="\r")

        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
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
            court_calibrator=calibrators.get(rally.video_id),
        )

        match_teams = match_teams_by_rally.get(rally.rally_id)
        rally_actions = classify_rally_actions(
            contacts, rally.rally_id,
            match_team_assignments=match_teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))
        matches, unmatched = match_contacts(
            rally.gt_labels, real_pred, tolerance=tolerance_frames,
        )
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        # Extract features for matched contacts
        net_y = rally.court_split_y or 0.5
        contact_list = contacts.contacts

        for m in matches:
            if m.pred_frame is None or m.pred_action is None:
                continue

            # Find the contact object matching the predicted frame
            contact_idx = None
            for ci, c in enumerate(contact_list):
                if c.frame == m.pred_frame:
                    contact_idx = ci
                    break

            if contact_idx is None:
                continue

            feat = extract_action_features(
                contact=contact_list[contact_idx],
                index=contact_idx,
                all_contacts=contact_list,
                ball_positions=ball_positions,
                net_y=net_y,
            )

            if m.gt_action == m.pred_action:
                all_correct.append((m.gt_action, feat))
            else:
                all_confused.append((m.gt_action, m.pred_action, feat, rally.rally_id))

    console.print()

    # === 1. Confusion pair counts ===
    pair_counts: Counter[tuple[str, str]] = Counter()
    for gt, pred, _, _ in all_confused:
        pair_counts[(gt, pred)] += 1

    pair_table = Table(title="Top Confusion Pairs (GT → Pred)")
    pair_table.add_column("GT", style="bold")
    pair_table.add_column("Pred", style="bold")
    pair_table.add_column("Count", justify="right")
    for (gt, pred), count in pair_counts.most_common(15):
        pair_table.add_row(gt, pred, str(count))
    console.print(pair_table)

    # === 2. Per confusion pair: feature comparison ===
    feature_names = ActionFeatures.feature_names()

    for (gt_act, pred_act), count in pair_counts.most_common(8):
        if count < 2:
            continue

        console.print(f"\n[bold]{gt_act} → {pred_act} ({count} cases)[/bold]")

        # Get features for this confusion pair
        confused_feats = [
            f for g, p, f, _ in all_confused if g == gt_act and p == pred_act
        ]
        # Get features for correctly classified GT action
        correct_feats = [f for a, f in all_correct if a == gt_act]

        if not correct_feats:
            console.print("  [dim]No correct examples for comparison[/dim]")
            continue

        # Compare distributions
        feat_table = Table(title=f"Feature Comparison: correct {gt_act} vs misclassified as {pred_act}")
        feat_table.add_column("Feature")
        feat_table.add_column("Correct (med)", justify="right")
        feat_table.add_column("Confused (med)", justify="right")
        feat_table.add_column("Diff", justify="right")
        feat_table.add_column("Flag", justify="center")

        correct_arrays = [f.to_array() for f in correct_feats]
        confused_arrays = [f.to_array() for f in confused_feats]

        correct_mat = np.array(correct_arrays)
        confused_mat = np.array(confused_arrays)

        for fi, fname in enumerate(feature_names):
            c_med = float(np.median(correct_mat[:, fi]))
            x_med = float(np.median(confused_mat[:, fi]))
            diff = x_med - c_med

            # Flag notable differences
            flag = ""
            if fname == "player_distance" and x_med > 0.15:
                flag = "TRACKING"
            elif fname == "contact_count_on_current_side" and abs(diff) >= 1:
                flag = "NET_CROSS"
            elif abs(diff) > 0.5 * max(abs(c_med), 0.01):
                flag = "notable"

            feat_table.add_row(
                fname,
                f"{c_med:.3f}",
                f"{x_med:.3f}",
                f"{diff:+.3f}",
                flag,
            )

        console.print(feat_table)

        # Print individual cases with tracking flags
        tracking_caused = 0
        net_cross_caused = 0
        for gt, pred, feat, rally_id in all_confused:
            if gt != gt_act or pred != pred_act:
                continue
            if not math.isfinite(feat.player_distance) or feat.player_distance > 0.15:
                tracking_caused += 1
            if abs(feat.contact_count_on_current_side - np.median(correct_mat[:, 9])) >= 1.5:
                net_cross_caused += 1

        if tracking_caused > 0 or net_cross_caused > 0:
            console.print(f"  Root causes: tracking={tracking_caused}, net_crossing={net_cross_caused} / {count}")

    # === 3. Overall tracking-caused misclassification ===
    total_tracking = sum(
        1 for _, _, f, _ in all_confused
        if not math.isfinite(f.player_distance) or f.player_distance > 0.15
    )
    console.print(f"\n[bold]Overall: {total_tracking}/{len(all_confused)} misclassifications have player_distance > 0.15 (tracking issue)[/bold]")

    # === 4. Per-case detail for top confusions ===
    detail_table = Table(title=f"\nAll Misclassified Contacts ({len(all_confused)})")
    detail_table.add_column("Rally", style="dim", max_width=8)
    detail_table.add_column("GT", max_width=7)
    detail_table.add_column("Pred", max_width=7)
    detail_table.add_column("vel", justify="right", max_width=6)
    detail_table.add_column("dir_chg", justify="right", max_width=6)
    detail_table.add_column("ball_y", justify="right", max_width=6)
    detail_table.add_column("y_rel", justify="right", max_width=6)
    detail_table.add_column("plr_dst", justify="right", max_width=6)
    detail_table.add_column("side_ct", justify="right", max_width=6)
    detail_table.add_column("idx", justify="right", max_width=4)
    detail_table.add_column("post_dy", justify="right", max_width=7)
    detail_table.add_column("post_dx", justify="right", max_width=7)
    detail_table.add_column("pre_dy", justify="right", max_width=7)
    detail_table.add_column("frm_gap", justify="right", max_width=7)
    detail_table.add_column("Flag")

    for gt, pred, feat, rally_id in sorted(all_confused, key=lambda x: (x[0], x[1])):
        plr_str = f"{feat.player_distance:.3f}" if math.isfinite(feat.player_distance) else "inf"
        flag = ""
        if not math.isfinite(feat.player_distance) or feat.player_distance > 0.15:
            flag = "TRACKING"
        detail_table.add_row(
            rally_id[:8],
            gt, pred,
            f"{feat.velocity:.3f}",
            f"{feat.direction_change_deg:.0f}",
            f"{feat.ball_y:.3f}",
            f"{feat.ball_y_relative_net:.3f}",
            plr_str,
            str(feat.contact_count_on_current_side),
            str(feat.contact_index_in_rally),
            f"{feat.post_contact_dy:.3f}",
            f"{feat.post_contact_dx:.3f}",
            f"{feat.pre_contact_dy:.3f}",
            str(feat.frames_since_last_contact),
            flag,
        )

    console.print(detail_table)


if __name__ == "__main__":
    main()
