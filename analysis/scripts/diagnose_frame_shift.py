"""Diagnose: how many attribution errors are fixable by shifting the contact frame.

For each attribution error, checks whether the nearest player at frame ± {1,2,3,5}
would be the correct (GT) player. Quantifies how much contact timing jitter
contributes to attribution errors.

Usage:
    cd analysis
    uv run python scripts/diagnose_frame_shift.py
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
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

OFFSETS = [-5, -3, -2, -1, 1, 2, 3, 5]


def _nearest_player_at_frame(
    frame: int,
    ball_x: float,
    ball_y: float,
    positions: list[PlayerPos],
    search_frames: int = 5,
) -> tuple[int, float]:
    """Find nearest player to ball at frame. Returns (track_id, distance)."""
    best_tid = -1
    best_dist = float("inf")
    for p in positions:
        if abs(p.frame_number - frame) > search_frames:
            continue
        px = p.x
        py = p.y - p.height * 0.25  # upper-quarter of bbox
        d = math.sqrt((ball_x - px) ** 2 + (ball_y - py) ** 2)
        if d < best_dist:
            best_dist = d
            best_tid = p.track_id
    return best_tid, best_dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", type=str)
    parser.add_argument("--tolerance-ms", type=int, default=167)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT.[/red]")
        return

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

    console.print(f"\n[bold]Frame-shift diagnosis across {len(rallies)} rallies[/bold]\n")

    total = 0
    correct_at_0 = 0
    errors = 0
    # For each offset, how many errors become correct
    fixed_by_offset: dict[int, int] = {o: 0 for o in OFFSETS}
    fixed_by_any_offset = 0
    best_offset_for_fix: list[int] = []  # smallest |offset| that fixes each error

    # Per-error details
    error_details: list[dict] = []

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

        ball_by_frame = {bp.frame_number: bp for bp in ball_positions}

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

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

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
                correct_at_0 += 1
                continue

            # Error case — try shifting
            errors += 1
            this_fixed = False
            this_best_offset = None

            for offset in sorted(OFFSETS, key=abs):
                shifted_frame = m.pred_frame + offset
                # Get ball position at shifted frame (or nearest)
                bp = ball_by_frame.get(shifted_frame)
                if bp is None:
                    # Try ±1 from shifted
                    bp = ball_by_frame.get(shifted_frame + 1) or ball_by_frame.get(shifted_frame - 1)
                if bp is None:
                    continue

                shifted_tid, shifted_dist = _nearest_player_at_frame(
                    shifted_frame, bp.x, bp.y, positions, search_frames=5,
                )

                if shifted_tid == gt_tid:
                    fixed_by_offset[offset] += 1
                    if not this_fixed:
                        this_fixed = True
                        this_best_offset = offset

            if this_fixed:
                fixed_by_any_offset += 1
                best_offset_for_fix.append(this_best_offset)

            error_details.append({
                "rally_id": rally.rally_id[:8],
                "frame": m.pred_frame,
                "gt_tid": gt_tid,
                "pred_tid": pred_tid,
                "fixed": this_fixed,
                "best_offset": this_best_offset,
                "action": m.gt_action if hasattr(m, "gt_action") else "?",
            })

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts, {errors} errors so far")

    # Results
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]")
    console.print(f"  Correct at frame+0: {correct_at_0} ({correct_at_0/max(1,total):.1%})")
    console.print(f"  Errors: {errors}")
    console.print(f"  Fixed by ANY offset (±1-5): {fixed_by_any_offset} ({fixed_by_any_offset/max(1,errors):.1%} of errors)")

    table = Table(title="Errors fixed per offset")
    table.add_column("Offset")
    table.add_column("Fixes")
    table.add_column("% of errors")
    for offset in OFFSETS:
        n = fixed_by_offset[offset]
        table.add_row(
            f"{offset:+d}",
            str(n),
            f"{n/max(1,errors):.1%}",
        )
    console.print(table)

    if best_offset_for_fix:
        from collections import Counter
        offset_dist = Counter(best_offset_for_fix)
        console.print("\n[bold cyan]Best offset distribution (smallest |offset| that fixes):[/bold cyan]")
        for off, cnt in sorted(offset_dist.items(), key=lambda x: abs(x[0])):
            console.print(f"  offset={off:+d}: {cnt} errors")

    console.print(f"\n[bold green]Ceiling if we could pick best frame per contact:[/bold green]")
    ceiling = correct_at_0 + fixed_by_any_offset
    console.print(f"  {ceiling}/{total} = {ceiling/max(1,total):.1%}")

    # Show first 20 error details
    if error_details:
        det_table = Table(title="Error details (first 30)")
        det_table.add_column("Rally")
        det_table.add_column("Frame")
        det_table.add_column("GT")
        det_table.add_column("Pred")
        det_table.add_column("Fixed?")
        det_table.add_column("Best Offset")
        for d in error_details[:30]:
            det_table.add_row(
                d["rally_id"],
                str(d["frame"]),
                f"T{d['gt_tid']}",
                f"T{d['pred_tid']}",
                "YES" if d["fixed"] else "no",
                f"{d['best_offset']:+d}" if d["best_offset"] is not None else "-",
            )
        console.print(det_table)


if __name__ == "__main__":
    main()
