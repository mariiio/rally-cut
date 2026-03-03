"""Diagnose player attribution errors in contact detection.

For each matched contact, finds ALL players near the ball, ranks by distance,
and shows where the GT player falls in the ranking. Analyzes what signals
could improve attribution.

Usage:
    cd analysis
    uv run python scripts/diagnose_player_attribution.py
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()


@dataclass
class PlayerCandidate:
    track_id: int
    distance: float
    player_x: float
    player_y: float  # upper-quarter y (contact point)
    center_y: float  # bbox center y (for court side)
    rank: int


def find_all_players_ranked(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 5,
) -> list[PlayerCandidate]:
    """Find all players near ball, ranked by distance."""
    candidates: dict[int, tuple[float, float, float, float]] = {}

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue

        player_x = p.x
        player_y = p.y - p.height * 0.25

        dist = math.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

        if p.track_id not in candidates or dist < candidates[p.track_id][0]:
            candidates[p.track_id] = (dist, player_x, player_y, p.y)

    ranked = sorted(candidates.items(), key=lambda x: x[1][0])
    return [
        PlayerCandidate(
            track_id=tid,
            distance=vals[0],
            player_x=vals[1],
            player_y=vals[2],
            center_y=vals[3],
            rank=i + 1,
        )
        for i, (tid, vals) in enumerate(ranked)
    ]


def get_median_y(track_id: int, player_positions: list[PlayerPosition]) -> float | None:
    """Get median Y of a track (for team classification)."""
    ys = [p.y for p in player_positions if p.track_id == track_id]
    if not ys:
        return None
    ys.sort()
    return ys[len(ys) // 2]


def main() -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found with action ground truth.[/red]")
        return

    console.print(f"[bold]Analyzing player attribution across {len(rallies)} rallies[/bold]\n")

    gt_rank_counts = Counter()
    gt_not_found_count = 0
    total_matched = 0
    total_with_gt_track = 0  # contacts where GT has a valid track_id

    # Per-action rank distribution
    action_rank_counts: dict[str, Counter] = defaultdict(Counter)

    # "Not found" analysis
    not_found_details: list[dict] = []

    # Team-based analysis
    team_would_help = 0  # rank-2 cases where team assignment would pick GT
    team_would_hurt = 0  # rank-2 cases where team signal picks wrong
    rank2_total = 0
    rank1_would_regress = 0  # rank-1 cases where team signal would override correct
    rank1_analyzed = 0

    # Distance ratio for errors
    distance_ratios: list[float] = []

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPosition(
                frame_number=bp["frameNumber"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions: list[PlayerPosition] = []
        if rally.positions_json:
            player_positions = [
                PlayerPosition(
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

        # Compute team assignments from median Y
        all_track_ids = set(p.track_id for p in player_positions)
        track_median_y: dict[int, float] = {}
        for tid in all_track_ids:
            my = get_median_y(tid, player_positions)
            if my is not None:
                track_median_y[tid] = my

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )

        rally_actions = classify_rally_actions(contacts, rally.rally_id)
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        tolerance_frames = max(1, round(rally.fps * 150 / 1000))
        matches, _ = match_contacts(rally.gt_labels, real_pred, tolerance=tolerance_frames)

        for m in matches:
            if m.pred_frame is None:
                continue

            total_matched += 1

            gt_label = None
            for gt in rally.gt_labels:
                if gt.frame == m.gt_frame:
                    gt_label = gt
                    break
            if gt_label is None or gt_label.player_track_id < 0:
                continue

            total_with_gt_track += 1
            gt_tid = gt_label.player_track_id

            pred_tid = None
            pred_ball_x = None
            pred_ball_y = None
            for pa in real_pred:
                if pa.get("frame") == m.pred_frame:
                    pred_tid = pa.get("playerTrackId", -1)
                    pred_ball_x = pa.get("ballX")
                    pred_ball_y = pa.get("ballY")
                    break

            if pred_tid is None or pred_ball_x is None or pred_ball_y is None:
                continue

            candidates = find_all_players_ranked(
                m.pred_frame, pred_ball_x, pred_ball_y, player_positions
            )

            gt_rank = None
            gt_cand = None
            pred_cand = None
            for c in candidates:
                if c.track_id == gt_tid:
                    gt_rank = c.rank
                    gt_cand = c
                if c.track_id == pred_tid:
                    pred_cand = c

            if gt_rank is not None:
                gt_rank_counts[gt_rank] += 1
                action_rank_counts[m.gt_action][gt_rank] += 1

                if gt_cand and pred_cand and gt_cand.distance > 0 and pred_cand.distance > 0:
                    distance_ratios.append(gt_cand.distance / pred_cand.distance)

                # Team-based analysis
                net_y = rally.court_split_y or 0.5
                ball_side = 0 if pred_ball_y > net_y else 1  # 0=near, 1=far

                if gt_rank == 2 and gt_cand and pred_cand:
                    rank2_total += 1

                    gt_team = None
                    pred_team = None
                    if gt_tid in track_median_y:
                        gt_team = 0 if track_median_y[gt_tid] > net_y else 1
                    if pred_tid in track_median_y:
                        pred_team = 0 if track_median_y[pred_tid] > net_y else 1

                    if gt_team is not None and pred_team is not None:
                        if gt_team == ball_side and pred_team != ball_side:
                            team_would_help += 1
                        elif pred_team == ball_side and gt_team != ball_side:
                            team_would_hurt += 1

                elif gt_rank == 1 and pred_cand:
                    # Check if team signal would override this correct attribution
                    rank1_analyzed += 1
                    pred_team = None
                    if pred_tid in track_median_y:
                        pred_team = 0 if track_median_y[pred_tid] > net_y else 1

                    if pred_team is not None and pred_team != ball_side:
                        # Team signal disagrees with correct attribution
                        # Would it override? Check if #2 candidate matches ball side
                        if len(candidates) >= 2:
                            c2 = candidates[1]  # 2nd nearest
                            c2_team = None
                            if c2.track_id in track_median_y:
                                c2_team = 0 if track_median_y[c2.track_id] > net_y else 1
                            if c2_team == ball_side:
                                rank1_would_regress += 1
            else:
                gt_not_found_count += 1

                # Analyze why GT track not found
                gt_exists_anywhere = gt_tid in all_track_ids
                gt_frames = [p.frame_number for p in player_positions
                             if p.track_id == gt_tid]
                frame_gap = None
                if gt_frames:
                    closest_frame = min(abs(f - m.pred_frame) for f in gt_frames)
                    frame_gap = closest_frame

                not_found_details.append({
                    "rally": rally.rally_id[:8],
                    "frame": m.pred_frame,
                    "action": m.gt_action,
                    "gt_tid": gt_tid,
                    "exists": gt_exists_anywhere,
                    "frame_gap": frame_gap,
                    "n_gt_frames": len(gt_frames),
                    "n_candidates": len(candidates),
                    "candidate_tids": [c.track_id for c in candidates],
                })

    # === Results ===
    console.print(f"\n[bold]Player Attribution Diagnostic[/bold]")
    console.print(f"  Total matched contacts: {total_matched}")
    console.print(f"  With GT track_id: {total_with_gt_track}")
    console.print(f"  Without GT track_id: {total_matched - total_with_gt_track}\n")

    # Rank distribution
    console.print("[bold]GT Player Distance Ranking[/bold]")
    rank_table = Table()
    rank_table.add_column("Rank", justify="right")
    rank_table.add_column("Count", justify="right")
    rank_table.add_column("Pct", justify="right")
    rank_table.add_column("Cumulative", justify="right")

    cumulative = 0
    for rank in sorted(gt_rank_counts.keys()):
        count = gt_rank_counts[rank]
        pct = count / total_with_gt_track * 100
        cumulative += pct
        rank_table.add_row(f"#{rank}", str(count), f"{pct:.1f}%", f"{cumulative:.1f}%")
    if gt_not_found_count > 0:
        rank_table.add_row(
            "Not found", str(gt_not_found_count),
            f"{gt_not_found_count / total_with_gt_track * 100:.1f}%", ""
        )
    console.print(rank_table)

    # Per-action breakdown
    console.print("\n[bold]Per-Action Rank Distribution[/bold]")
    action_table = Table()
    action_table.add_column("Action", style="bold")
    action_table.add_column("#1 (correct)", justify="right")
    action_table.add_column("#2", justify="right")
    action_table.add_column("#3+", justify="right")
    action_table.add_column("Not found", justify="right")
    action_table.add_column("Total", justify="right")
    action_table.add_column("Accuracy", justify="right")

    for action in ["serve", "receive", "set", "attack", "dig", "block"]:
        counts = action_rank_counts.get(action, Counter())
        total = sum(counts.values())
        # Count not-found per action
        nf = sum(1 for d in not_found_details if d["action"] == action)
        full_total = total + nf
        if full_total == 0:
            continue
        r1 = counts.get(1, 0)
        r2 = counts.get(2, 0)
        r3plus = sum(v for k, v in counts.items() if k >= 3)
        action_table.add_row(
            action,
            f"{r1} ({r1/full_total*100:.0f}%)",
            f"{r2} ({r2/full_total*100:.0f}%)",
            f"{r3plus}" if r3plus > 0 else "0",
            f"{nf}" if nf > 0 else "0",
            str(full_total),
            f"{r1/full_total*100:.1f}%",
        )
    console.print(action_table)

    # "Not found" analysis
    console.print(f"\n[bold]'Not Found' Analysis ({gt_not_found_count} cases)[/bold]")
    exists_count = sum(1 for d in not_found_details if d["exists"])
    not_exists = sum(1 for d in not_found_details if not d["exists"])
    console.print(f"  GT track exists in rally but outside search window: {exists_count}")
    console.print(f"  GT track doesn't exist in rally at all: {not_exists}")

    if exists_count > 0:
        gaps = [d["frame_gap"] for d in not_found_details if d["exists"] and d["frame_gap"] is not None]
        if gaps:
            gaps.sort()
            console.print(f"  Frame gap to nearest GT track frame: median={gaps[len(gaps)//2]}, max={max(gaps)}")

    # Show "not found" samples
    if not_found_details:
        nf_table = Table(title="Not Found Samples (first 15)")
        nf_table.add_column("Rally", max_width=8)
        nf_table.add_column("Frame", justify="right")
        nf_table.add_column("Action")
        nf_table.add_column("GT TID", justify="right")
        nf_table.add_column("Exists?")
        nf_table.add_column("Gap", justify="right")
        nf_table.add_column("#GT frames", justify="right")
        nf_table.add_column("Candidates")

        for d in not_found_details[:15]:
            nf_table.add_row(
                d["rally"],
                str(d["frame"]),
                d["action"],
                str(d["gt_tid"]),
                "YES" if d["exists"] else "NO",
                str(d["frame_gap"]) if d["frame_gap"] is not None else "-",
                str(d["n_gt_frames"]),
                str(d["candidate_tids"][:4]),
            )
        console.print(nf_table)

    # Summary
    correct = gt_rank_counts.get(1, 0)
    rank2 = gt_rank_counts.get(2, 0)
    r3plus = sum(v for k, v in gt_rank_counts.items() if k >= 3)

    # Team-based signal analysis
    console.print(f"\n[bold]Team-Based Signal (ball side vs player team)[/bold]")
    console.print(f"  Rank-2 fixes (GT team matches ball side): +{team_would_help}")
    console.print(f"  Rank-2 anti-fixes (wrong team matches ball side): -{team_would_hurt}")
    console.print(f"  Rank-2 neutral/no data: {rank2_total - team_would_help - team_would_hurt}")
    console.print(f"  Rank-1 regressions (correct player overridden): -{rank1_would_regress}/{rank1_analyzed}")
    net = team_would_help - team_would_hurt - rank1_would_regress
    new_correct = correct + net
    console.print(f"  [bold]Net effect: {'+' if net >= 0 else ''}{net} → {new_correct}/{total_with_gt_track} ({new_correct/max(1,total_with_gt_track)*100:.1f}%)[/bold]")

    # Distance ratio for errors
    error_ratios = [r for r in distance_ratios if r > 1.001]
    if error_ratios:
        error_ratios.sort()
        console.print(f"\n[bold]Distance Ratio (GT_dist / pred_dist for errors)[/bold]")
        console.print(f"  Errors: {len(error_ratios)}")
        console.print(f"  Median: {error_ratios[len(error_ratios)//2]:.2f}x")
        under_1_5x = sum(1 for r in error_ratios if r < 1.5)
        under_2x = sum(1 for r in error_ratios if r < 2.0)
        console.print(f"  Within 1.5x: {under_1_5x}/{len(error_ratios)} ({under_1_5x/len(error_ratios)*100:.0f}%)")
        console.print(f"  Within 2x:   {under_2x}/{len(error_ratios)} ({under_2x/len(error_ratios)*100:.0f}%)")

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Current accuracy: {correct}/{total_with_gt_track} ({correct/max(1,total_with_gt_track)*100:.1f}%)")
    console.print(f"  GT player is #2 (tiebreaker): {rank2} ({rank2/max(1,total_with_gt_track)*100:.1f}%)")
    console.print(f"  GT player is #3+ (hard): {r3plus} ({r3plus/max(1,total_with_gt_track)*100:.1f}%)")
    console.print(f"  GT player not found: {gt_not_found_count} ({gt_not_found_count/max(1,total_with_gt_track)*100:.1f}%)")
    console.print(f"  Ceiling (rank 1+2): {(correct+rank2)/max(1,total_with_gt_track)*100:.1f}%")


if __name__ == "__main__":
    main()
