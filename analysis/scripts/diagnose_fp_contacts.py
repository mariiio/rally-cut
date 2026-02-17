"""Diagnose false positive contacts to find filterable patterns.

Runs contact detection on all rallies with action GT, matches to ground truth,
and compares TP vs FP contact characteristics to identify systematic differences
that could be used to reduce FPs.

Usage:
    cd analysis
    uv run python scripts/diagnose_fp_contacts.py
    uv run python scripts/diagnose_fp_contacts.py --rally <id>
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import load_rallies_with_action_gt

console = Console()


@dataclass
class ContactWithContext:
    """A contact with its TP/FP status and distance to nearest GT."""

    contact: Contact
    pred_action: str
    is_tp: bool
    nearest_gt_dist: int  # Frame distance to nearest GT contact
    nearest_gt_action: str  # Action type of nearest GT contact
    rally_id: str


def analyze_rallies(rally_id: str | None = None) -> list[ContactWithContext]:
    """Run detection, match to GT, and return annotated contacts."""
    rallies = load_rallies_with_action_gt(rally_id=rally_id)
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return []

    all_contacts: list[ContactWithContext] = []

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

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

        player_positions = []
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
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )

        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)

        # Build action map: frame -> action string
        action_map: dict[int, str] = {}
        for a in rally_actions.actions:
            action_map[a.frame] = a.action_type.value

        # Match using same tolerance as eval (167ms -> ~5 frames)
        tolerance = max(1, round(rally.fps * 167 / 1000))
        gt_frames = sorted(gt.frame for gt in rally.gt_labels)
        gt_action_map = {gt.frame: gt.action for gt in rally.gt_labels}

        # Greedy matching (same as eval)
        used_gt: set[int] = set()
        tp_pred_frames: set[int] = set()

        pred_contacts = sorted(contact_seq.contacts, key=lambda c: c.frame)
        gt_sorted = sorted(rally.gt_labels, key=lambda g: g.frame)

        for gt in gt_sorted:
            best_idx: int | None = None
            best_dist = tolerance + 1
            for i, pred in enumerate(pred_contacts):
                if pred.frame in tp_pred_frames:
                    continue
                dist = abs(gt.frame - pred.frame)
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                tp_pred_frames.add(pred_contacts[best_idx].frame)
                used_gt.add(gt.frame)

        # Annotate each predicted contact
        for contact in contact_seq.contacts:
            is_tp = contact.frame in tp_pred_frames

            # Distance to nearest GT
            if gt_frames:
                nearest_gt_dist = min(abs(contact.frame - gf) for gf in gt_frames)
                nearest_gt_frame = min(gt_frames, key=lambda gf: abs(contact.frame - gf))
                nearest_gt_action = gt_action_map.get(nearest_gt_frame, "?")
            else:
                nearest_gt_dist = 9999
                nearest_gt_action = "?"

            all_contacts.append(ContactWithContext(
                contact=contact,
                pred_action=action_map.get(contact.frame, "unknown"),
                is_tp=is_tp,
                nearest_gt_dist=nearest_gt_dist,
                nearest_gt_action=nearest_gt_action,
                rally_id=rally.rally_id,
            ))

    return all_contacts


def print_distribution(
    label: str,
    tp_values: list[float],
    fp_values: list[float],
) -> None:
    """Print side-by-side TP vs FP distribution for a metric."""
    if not tp_values or not fp_values:
        return

    tp_med = statistics.median(tp_values)
    fp_med = statistics.median(fp_values)
    tp_mean = statistics.mean(tp_values)
    fp_mean = statistics.mean(fp_values)
    tp_p25 = sorted(tp_values)[len(tp_values) // 4]
    fp_p25 = sorted(fp_values)[len(fp_values) // 4]
    tp_p75 = sorted(tp_values)[3 * len(tp_values) // 4]
    fp_p75 = sorted(fp_values)[3 * len(fp_values) // 4]

    console.print(f"  [bold]{label}[/bold]")
    console.print(
        f"    TP (n={len(tp_values):3d}): "
        f"mean={tp_mean:.4f}  median={tp_med:.4f}  "
        f"p25={tp_p25:.4f}  p75={tp_p75:.4f}  "
        f"range=[{min(tp_values):.4f}, {max(tp_values):.4f}]"
    )
    console.print(
        f"    FP (n={len(fp_values):3d}): "
        f"mean={fp_mean:.4f}  median={fp_med:.4f}  "
        f"p25={fp_p25:.4f}  p75={fp_p75:.4f}  "
        f"range=[{min(fp_values):.4f}, {max(fp_values):.4f}]"
    )
    sep = ">>>" if fp_med > tp_med * 1.3 else ("<<<" if tp_med > fp_med * 1.3 else " ~ ")
    console.print(f"    Separation: {sep} (median ratio: {fp_med / max(tp_med, 1e-9):.2f}x)")
    console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose FP contact patterns")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    args = parser.parse_args()

    contacts = analyze_rallies(rally_id=args.rally)
    if not contacts:
        return

    tps = [c for c in contacts if c.is_tp]
    fps = [c for c in contacts if not c.is_tp]

    console.print(f"\n[bold]Contact Analysis: {len(tps)} TP, {len(fps)} FP[/bold]\n")

    # === Distribution comparisons ===
    console.print("[bold underline]Feature Distributions (TP vs FP)[/bold underline]\n")

    print_distribution(
        "Velocity",
        [c.contact.velocity for c in tps],
        [c.contact.velocity for c in fps],
    )

    print_distribution(
        "Direction Change (deg)",
        [c.contact.direction_change_deg for c in tps],
        [c.contact.direction_change_deg for c in fps],
    )

    print_distribution(
        "Player Distance",
        [c.contact.player_distance for c in tps if c.contact.player_distance < 1.0],
        [c.contact.player_distance for c in fps if c.contact.player_distance < 1.0],
    )

    print_distribution(
        "Ball Y",
        [c.contact.ball_y for c in tps],
        [c.contact.ball_y for c in fps],
    )

    print_distribution(
        "Ball X",
        [c.contact.ball_x for c in tps],
        [c.contact.ball_x for c in fps],
    )

    # Gap to nearest GT
    print_distribution(
        "Gap to Nearest GT (frames)",
        [float(c.nearest_gt_dist) for c in tps],
        [float(c.nearest_gt_dist) for c in fps],
    )

    # === Gap between consecutive contacts ===
    console.print("[bold underline]Inter-Contact Gap (frames to previous contact)[/bold underline]\n")

    # Sort all contacts by rally + frame to compute gaps
    by_rally: dict[str, list[ContactWithContext]] = {}
    for c in contacts:
        by_rally.setdefault(c.rally_id, []).append(c)

    tp_gaps: list[float] = []
    fp_gaps: list[float] = []
    for rally_contacts in by_rally.values():
        sorted_contacts = sorted(rally_contacts, key=lambda c: c.contact.frame)
        for i in range(1, len(sorted_contacts)):
            gap = sorted_contacts[i].contact.frame - sorted_contacts[i - 1].contact.frame
            if sorted_contacts[i].is_tp:
                tp_gaps.append(float(gap))
            else:
                fp_gaps.append(float(gap))

    if tp_gaps and fp_gaps:
        print_distribution("Inter-Contact Gap", tp_gaps, fp_gaps)

    # === Validation tier breakdown ===
    console.print("[bold underline]Validation Tier Breakdown[/bold underline]\n")

    # Reconstruct validation tier from features
    cfg = ContactDetectionConfig()
    tp_tiers: dict[str, int] = {"tier1_strong": 0, "tier2_fast_player": 0, "tier3_player": 0, "other": 0}
    fp_tiers: dict[str, int] = {"tier1_strong": 0, "tier2_fast_player": 0, "tier3_player": 0, "other": 0}

    for c in contacts:
        high_vel = c.contact.velocity >= cfg.high_velocity_threshold
        has_dir = c.contact.direction_change_deg >= cfg.min_direction_change_deg
        has_player = c.contact.player_distance <= cfg.player_contact_radius

        if high_vel and has_dir:
            tier = "tier1_strong"
        elif high_vel and has_player:
            tier = "tier2_fast_player"
        elif has_player and (has_dir or c.contact.velocity >= cfg.min_peak_velocity):
            tier = "tier3_player"
        else:
            tier = "other"

        if c.is_tp:
            tp_tiers[tier] += 1
        else:
            fp_tiers[tier] += 1

    tier_table = Table(title="Validation Tiers")
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("TP", justify="right")
    tier_table.add_column("FP", justify="right")
    tier_table.add_column("FP Rate", justify="right")
    for tier in ["tier1_strong", "tier2_fast_player", "tier3_player", "other"]:
        tp_n = tp_tiers[tier]
        fp_n = fp_tiers[tier]
        total = tp_n + fp_n
        fp_rate = fp_n / max(1, total)
        tier_table.add_row(tier, str(tp_n), str(fp_n), f"{fp_rate:.0%}")
    console.print(tier_table)

    # === Court side breakdown ===
    console.print("\n[bold underline]Court Side Breakdown[/bold underline]\n")

    side_table = Table(title="Court Side")
    side_table.add_column("Side", style="bold")
    side_table.add_column("TP", justify="right")
    side_table.add_column("FP", justify="right")
    side_table.add_column("FP Rate", justify="right")
    for side in ["near", "far", "unknown"]:
        tp_n = sum(1 for c in tps if c.contact.court_side == side)
        fp_n = sum(1 for c in fps if c.contact.court_side == side)
        total = tp_n + fp_n
        fp_rate = fp_n / max(1, total)
        side_table.add_row(side, str(tp_n), str(fp_n), f"{fp_rate:.0%}")
    console.print(side_table)

    # === Predicted action breakdown for FPs ===
    console.print("\n[bold underline]FP Contacts by Predicted Action[/bold underline]\n")

    action_table = Table(title="FP Actions (what are FPs being classified as?)")
    action_table.add_column("Pred Action", style="bold")
    action_table.add_column("Count", justify="right")
    action_table.add_column("Pct", justify="right")

    fp_actions: dict[str, int] = {}
    for c in fps:
        fp_actions[c.pred_action] = fp_actions.get(c.pred_action, 0) + 1
    for action, count in sorted(fp_actions.items(), key=lambda x: -x[1]):
        action_table.add_row(action, str(count), f"{count / len(fps):.0%}")
    console.print(action_table)

    # === Nearest GT action for FPs ===
    console.print("\n[bold underline]FP Nearest GT Contact[/bold underline]\n")

    near_table = Table(title="Nearest GT action for each FP (what GT is nearby?)")
    near_table.add_column("Nearest GT", style="bold")
    near_table.add_column("Count", justify="right")
    near_table.add_column("Avg Gap (frames)", justify="right")

    fp_near_gt: dict[str, list[int]] = {}
    for c in fps:
        fp_near_gt.setdefault(c.nearest_gt_action, []).append(c.nearest_gt_dist)
    for action, gaps in sorted(fp_near_gt.items(), key=lambda x: -len(x[1])):
        avg_gap = statistics.mean(gaps)
        near_table.add_row(action, str(len(gaps)), f"{avg_gap:.1f}")
    console.print(near_table)

    # === Individual FP listing (sorted by velocity) ===
    console.print(f"\n[bold underline]All {len(fps)} FP Contacts (sorted by velocity)[/bold underline]\n")

    fp_table = Table()
    fp_table.add_column("Rally", style="dim", max_width=8)
    fp_table.add_column("Frame", justify="right")
    fp_table.add_column("Vel", justify="right")
    fp_table.add_column("Dir°", justify="right")
    fp_table.add_column("PlyrDist", justify="right")
    fp_table.add_column("Ball XY", justify="right")
    fp_table.add_column("Side")
    fp_table.add_column("Net?")
    fp_table.add_column("Pred")
    fp_table.add_column("GT Gap", justify="right")
    fp_table.add_column("Near GT")

    for c in sorted(fps, key=lambda c: c.contact.velocity):
        pdist = (
            f"{c.contact.player_distance:.3f}"
            if c.contact.player_distance < 1.0
            else "none"
        )
        fp_table.add_row(
            c.rally_id[:8],
            str(c.contact.frame),
            f"{c.contact.velocity:.4f}",
            f"{c.contact.direction_change_deg:.1f}",
            pdist,
            f"({c.contact.ball_x:.2f},{c.contact.ball_y:.2f})",
            c.contact.court_side,
            "Y" if c.contact.is_at_net else "",
            c.pred_action,
            str(c.nearest_gt_dist),
            c.nearest_gt_action,
        )

    console.print(fp_table)


if __name__ == "__main__":
    main()
