"""Diagnose why GT block contacts are being missed.

For each GT block, examines the surrounding window and reports:
- Whether any contact candidate fired near the GT frame
- Ball detection rate in the window (ball dropouts at the net?)
- Pose: any near-net player has raised arms (both wrists above shoulders)?
- Distance from GT player to net at the GT frame
- Whether the prior contact was an attack (sanity check the "after attack" assumption)

Usage:
    cd analysis
    uv run python scripts/diagnose_missing_blocks.py
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    _build_player_positions,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def _ball_density(ball_positions: list, start: int, end: int) -> float:
    """Fraction of frames in [start, end) with a ball detection."""
    if end <= start:
        return 0.0
    have = {bp.frame_number for bp in ball_positions if start <= bp.frame_number < end}
    return len(have) / (end - start)


def _nearest_player_at(
    positions: list, frame: int, search: int = 3,
) -> tuple[PlayerPos | None, float]:
    """Return (player, distance_to_net) for the player closest to the net at `frame`.

    Net assumed at y=0.5 (we use y distance to that line). Returns None if
    no player is found in the search window.
    """
    best_pp = None
    best_dist = float("inf")
    for pp in positions:
        if abs(pp.frame_number - frame) > search:
            continue
        d = abs(pp.y - 0.5)
        if d < best_dist:
            best_dist = d
            best_pp = pp
    return best_pp, best_dist


def _arms_raised(pp: PlayerPos | None) -> bool | None:
    """Check if both wrists are above shoulders (normalized image coords).

    Returns None if pose data unavailable.
    """
    if pp is None or pp.keypoints is None:
        return None
    kps = pp.keypoints  # list[list[float]] (17, 3)
    # COCO indices: left/right shoulder (5/6), left/right wrist (9/10)
    try:
        ls_y, ls_c = kps[5][1], kps[5][2]
        rs_y, rs_c = kps[6][1], kps[6][2]
        lw_y, lw_c = kps[9][1], kps[9][2]
        rw_y, rw_c = kps[10][1], kps[10][2]
    except (IndexError, TypeError):
        return None
    if min(ls_c, rs_c, lw_c, rw_c) < 0.3:
        return None
    # In image coords y INCREASES downward — "above" means smaller y
    return (lw_y < ls_y) and (rw_y < rs_y)


def main() -> None:
    rallies = load_rallies_with_action_gt()
    block_rallies = [
        r for r in rallies if any(g.action == "block" for g in r.gt_labels)
    ]
    n_blocks = sum(
        1 for r in rallies for g in r.gt_labels if g.action == "block"
    )
    console.print(
        f"[bold]Loaded {len(rallies)} rallies, "
        f"{n_blocks} GT blocks across {len(block_rallies)} rallies[/bold]"
    )

    table = Table(title="GT Block Diagnosis", show_lines=False)
    table.add_column("Rally", max_width=10)
    table.add_column("BlkF", justify="right")  # GT block frame
    table.add_column("AtkF", justify="right")  # GT attack frame
    table.add_column("Δa-b", justify="right")  # frames from attack to block
    table.add_column("Δcand", justify="right")  # nearest candidate offset from BLOCK
    table.add_column("CandSrc", justify="left")  # which GT does the candidate match (atk/blk/other)
    table.add_column("Ball%", justify="right")  # ball density ±10
    table.add_column("DistNet", justify="right")  # nearest player to net at GT
    table.add_column("Arms↑", justify="center")  # arms raised

    summary = {
        "total": 0,
        "with_candidate": 0,
        "ball_dropout": 0,
        "arms_raised_yes": 0,
        "arms_raised_no": 0,
        "arms_raised_unknown": 0,
        "after_attack": 0,
    }

    config = ContactDetectionConfig(use_pose_attribution=False)

    for ri, rally in enumerate(block_rallies):
        gt_blocks = [g for g in rally.gt_labels if g.action == "block"]
        if not gt_blocks:
            continue

        # Build positions and run pose-free contact detection to see candidates
        ball_positions = []
        if rally.ball_positions_json:
            for bp in rally.ball_positions_json:
                if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
                    ball_positions.append(BallPos(
                        frame_number=bp["frameNumber"],
                        x=bp["x"], y=bp["y"],
                        confidence=bp.get("confidence", 1.0),
                    ))

        player_positions = (
            _build_player_positions(rally.positions_json, rally_id=rally.rally_id)
            if rally.positions_json
            else []
        )
        if not ball_positions or not player_positions:
            continue

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=config,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )
        contacts = contact_seq.contacts

        # Match GT to detected (gives us prev-action context)
        tolerance_frames = max(1, round(rally.fps * 167 / 1000))
        pred_actions = [
            {"frame": c.frame, "action": "unknown", "playerTrackId": c.player_track_id}
            for c in contacts
        ]
        matches, _ = match_contacts(
            rally.gt_labels, pred_actions, tolerance=tolerance_frames,
        )

        for gt in gt_blocks:
            summary["total"] += 1

            # Nearest contact candidate to BLOCK
            nearest_cand_offset = None
            nearest_cand_frame = None
            for c in contacts:
                d = c.frame - gt.frame
                if nearest_cand_offset is None or abs(d) < abs(nearest_cand_offset):
                    nearest_cand_offset = d
                    nearest_cand_frame = c.frame
            has_cand_5 = (
                nearest_cand_offset is not None
                and abs(nearest_cand_offset) <= 5
            )
            if has_cand_5:
                summary["with_candidate"] += 1

            # GT attack frame (immediately preceding the block)
            prev_actions = sorted(
                [g for g in rally.gt_labels if g.frame < gt.frame],
                key=lambda g: g.frame,
            )
            prev_action_lbl = prev_actions[-1] if prev_actions else None
            atk_frame = (
                prev_action_lbl.frame if prev_action_lbl and prev_action_lbl.action == "attack"
                else None
            )
            atk_to_blk = atk_frame is not None and (gt.frame - atk_frame)

            # Which GT does the nearest candidate belong to (closer in frames)?
            cand_src = "—"
            if nearest_cand_frame is not None and atk_frame is not None:
                d_atk = abs(nearest_cand_frame - atk_frame)
                d_blk = abs(nearest_cand_frame - gt.frame)
                if d_atk < d_blk:
                    cand_src = "atk"
                elif d_blk < d_atk:
                    cand_src = "blk"
                else:
                    cand_src = "tie"
            elif nearest_cand_frame is not None:
                cand_src = "blk?"

            # Ball density in ±10 frames
            ball_pct = _ball_density(
                ball_positions, gt.frame - 10, gt.frame + 11,
            )
            if ball_pct < 0.5:
                summary["ball_dropout"] += 1

            # Pose check on nearest-to-net player
            pp, dist_net = _nearest_player_at(player_positions, gt.frame)
            arms = _arms_raised(pp)
            if arms is True:
                summary["arms_raised_yes"] += 1
            elif arms is False:
                summary["arms_raised_no"] += 1
            else:
                summary["arms_raised_unknown"] += 1

            if prev_action_lbl and prev_action_lbl.action == "attack":
                summary["after_attack"] += 1

            table.add_row(
                rally.rally_id[:8],
                str(gt.frame),
                str(atk_frame) if atk_frame is not None else "—",
                f"{atk_to_blk:+d}" if isinstance(atk_to_blk, int) else "—",
                f"{nearest_cand_offset:+d}" if nearest_cand_offset is not None else "—",
                cand_src,
                f"{ball_pct:.0%}",
                f"{dist_net:.3f}" if dist_net != float("inf") else "—",
                "✓" if arms is True else ("✗" if arms is False else "?"),
            )

        if (ri + 1) % 5 == 0:
            console.print(f"  [{ri+1}/{len(block_rallies)}] processed")

    console.print(table)
    console.print()
    console.print("[bold]Summary[/bold]")
    t = summary["total"]
    if t == 0:
        console.print("  No blocks found")
        return
    console.print(f"  Total GT blocks: {t}")
    console.print(
        f"  With contact candidate (±5f): {summary['with_candidate']}/{t} "
        f"({summary['with_candidate']/t:.0%})"
    )
    console.print(
        f"  Ball dropout (<50% in ±10f): {summary['ball_dropout']}/{t} "
        f"({summary['ball_dropout']/t:.0%})"
    )
    console.print(
        f"  Arms raised (pose ✓): {summary['arms_raised_yes']}/{t} "
        f"({summary['arms_raised_yes']/t:.0%})"
    )
    console.print(
        f"  Arms NOT raised: {summary['arms_raised_no']}/{t} "
        f"({summary['arms_raised_no']/t:.0%})"
    )
    console.print(
        f"  Pose unavailable: {summary['arms_raised_unknown']}/{t} "
        f"({summary['arms_raised_unknown']/t:.0%})"
    )
    console.print(
        f"  After an attack: {summary['after_attack']}/{t} "
        f"({summary['after_attack']/t:.0%})"
    )


if __name__ == "__main__":
    main()
