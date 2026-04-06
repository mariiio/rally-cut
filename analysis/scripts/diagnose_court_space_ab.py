"""A/B comparison: court-space vs image-space server position detection.

For each rally, runs _find_server_by_position both WITH and WITHOUT the
calibrator, compares results against GT, and categorizes the impact.

Usage:
    cd analysis
    uv run python scripts/diagnose_court_space_ab.py
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration

from eval_sequence_enriched import (
    RallyBundle,
    prepare_rallies,
)

from rallycut.tracking.action_classifier import (
    _find_server_by_position,
)

console = Console()


@dataclass
class ABResult:
    rally_id: str
    video_id: str
    gt_side: str  # "near"/"far"

    # Image-space result
    img_tid: int
    img_side: str
    img_conf: float
    img_correct: bool

    # Court-space result
    court_tid: int
    court_side: str
    court_conf: float
    court_correct: bool

    # Court-space projection details
    court_y_values: dict[int, float]  # track_id → court_y
    court_used: bool  # True if court-space was actually used (not fallback)

    # Impact classification
    impact: str  # "improved", "regressed", "both_correct", "both_wrong", "no_change"


def run_ab(bundles: list[RallyBundle]) -> list[ABResult]:
    """Run A/B comparison for all rallies."""
    # Load calibrations
    video_ids = {b.rally.video_id for b in bundles}
    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal

    console.print(f"Calibrations loaded: {len(calibrators)}/{len(video_ids)} videos")

    # Load team assignments for GT side
    from rallycut.tracking.contact_detector import detect_contacts, ContactDetectionConfig

    results: list[ABResult] = []
    for i, bundle in enumerate(bundles):
        # Get GT serve side
        gt_serves = [gt for gt in bundle.gt_labels if gt.action == "serve"]
        if not gt_serves:
            continue
        gt_s = gt_serves[0]
        if not bundle.match_teams or gt_s.player_track_id < 0:
            continue
        gt_team = bundle.match_teams.get(gt_s.player_track_id)
        if gt_team is None:
            continue
        gt_side = "near" if gt_team == 0 else "far"

        # Get player positions and net_y from contact detection
        contact_seq = detect_contacts(
            ball_positions=bundle.ball_positions,
            player_positions=bundle.player_positions,
            config=ContactDetectionConfig(),
            frame_count=bundle.rally.frame_count,
        )
        player_positions = contact_seq.player_positions or []
        start_frame = contact_seq.rally_start_frame
        net_y = contact_seq.net_y

        if not player_positions:
            continue

        # A: Image-space (no calibrator)
        img_tid, img_side, img_conf = _find_server_by_position(
            player_positions, start_frame, net_y,
            calibrator=None,
        )

        # B: Court-space (with calibrator)
        cal = calibrators.get(bundle.rally.video_id)
        court_tid, court_side, court_conf = _find_server_by_position(
            player_positions, start_frame, net_y,
            calibrator=cal,
        )

        # Also get court-y projections for diagnostics
        court_y_values: dict[int, float] = {}
        court_used = False
        if cal is not None and cal.is_calibrated:
            from collections import defaultdict as dd
            track_foot_ys: dict[int, list[float]] = dd(list)
            track_foot_xs: dict[int, list[float]] = dd(list)
            end_frame = start_frame + 45
            for p in player_positions:
                if start_frame <= p.frame_number < end_frame:
                    track_foot_ys[p.track_id].append(p.y + p.height / 2.0)
                    track_foot_xs[p.track_id].append(p.x)
            for tid in track_foot_ys:
                mean_x = sum(track_foot_xs[tid]) / len(track_foot_xs[tid])
                mean_y = sum(track_foot_ys[tid]) / len(track_foot_ys[tid])
                try:
                    _, cy = cal.image_to_court((mean_x, mean_y), 1, 1)
                    court_y_values[tid] = cy
                except Exception:
                    pass

            # Determine if court-space was actually used
            # (if court result differs from image result, court-space was used)
            court_used = (court_tid != img_tid or court_side != img_side
                          or abs(court_conf - img_conf) > 0.01)
            # More precise: court_used if calibrator was provided and didn't fall back
            # We can detect fallback by checking if results are identical
            if not court_used and cal is not None:
                # Results identical — could be fallback OR could be agreement
                # Check if any court_y is out of bounds (would trigger fallback)
                for cy in court_y_values.values():
                    if cy < -2 or cy > 20:
                        court_used = False
                        break
                else:
                    court_used = True  # All in bounds, court-space was likely used

        img_correct = img_side == gt_side if img_tid >= 0 else False
        court_correct = court_side == gt_side if court_tid >= 0 else False

        # For no-detection cases, also consider "no detection" as incorrect
        if img_tid < 0:
            img_correct = False
        if court_tid < 0:
            court_correct = False

        # Classify impact
        if court_correct and not img_correct:
            impact = "improved"
        elif not court_correct and img_correct:
            impact = "regressed"
        elif court_correct and img_correct:
            impact = "both_correct"
        elif not court_correct and not img_correct:
            impact = "both_wrong"
        else:
            impact = "no_change"

        results.append(ABResult(
            rally_id=bundle.rally.rally_id,
            video_id=bundle.rally.video_id,
            gt_side=gt_side,
            img_tid=img_tid, img_side=img_side, img_conf=img_conf,
            img_correct=img_correct,
            court_tid=court_tid, court_side=court_side, court_conf=court_conf,
            court_correct=court_correct,
            court_y_values=court_y_values,
            court_used=court_used,
            impact=impact,
        ))

        status_str = {
            "improved": "[green]IMPROVED[/green]",
            "regressed": "[red]REGRESSED[/red]",
            "both_correct": "[dim]both_ok[/dim]",
            "both_wrong": "[yellow]both_wrong[/yellow]",
        }.get(impact, impact)

        if impact in ("improved", "regressed", "both_wrong"):
            cy_str = " ".join(
                f"t{tid}={cy:.1f}" for tid, cy in sorted(court_y_values.items())
            )
            console.print(
                f"  [{i+1}] {bundle.rally.rally_id[:8]} gt={gt_side:4s} "
                f"img={img_side or '-':4s} court={court_side or '-':4s} "
                f"used={'Y' if court_used else 'N'} "
                f"→ {status_str}  court_y=[{cy_str}]"
            )

    return results


def main() -> None:
    import logging
    logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

    bundles = prepare_rallies(label_spread=2)
    if not bundles:
        console.print("[red]No rallies loaded.[/red]")
        return

    console.print(f"\n[bold]Court-Space A/B Comparison[/bold] ({len(bundles)} rallies)\n")
    results = run_ab(bundles)

    # Summary
    total = len(results)
    impacts = Counter(r.impact for r in results)

    console.print(f"\n[bold]Summary ({total} rallies with GT serve side)[/bold]")
    table = Table()
    table.add_column("Impact")
    table.add_column("Count", justify="right")
    table.add_column("%", justify="right")
    for cat in ["improved", "regressed", "both_correct", "both_wrong"]:
        c = impacts.get(cat, 0)
        style = {"improved": "green", "regressed": "red", "both_wrong": "yellow"}.get(cat, "")
        table.add_row(cat, str(c), f"{c/total:.1%}", style=style)
    console.print(table)

    img_correct = sum(1 for r in results if r.img_correct)
    court_correct = sum(1 for r in results if r.court_correct)
    console.print(f"\n  Image-space position accuracy:  {img_correct}/{total} ({img_correct/total:.1%})")
    console.print(f"  Court-space position accuracy:  {court_correct}/{total} ({court_correct/total:.1%})")

    court_used = sum(1 for r in results if r.court_used)
    console.print(f"  Court-space actually used: {court_used}/{total}")

    # Regressions detail
    regressions = [r for r in results if r.impact == "regressed"]
    if regressions:
        console.print(f"\n[bold red]Regressions ({len(regressions)})[/bold red]")
        reg_table = Table()
        reg_table.add_column("Rally")
        reg_table.add_column("Video")
        reg_table.add_column("GT Side")
        reg_table.add_column("Img Side")
        reg_table.add_column("Court Side")
        reg_table.add_column("Court TID")
        reg_table.add_column("Court Y Values")
        reg_table.add_column("Net Y=8 Side Check")
        for r in regressions:
            cy_str = ", ".join(f"t{tid}={cy:.1f}" for tid, cy in sorted(r.court_y_values.items()))
            # Check if court-y assignments make sense
            side_check = []
            for tid, cy in r.court_y_values.items():
                side = "near" if cy < 8 else "far"
                side_check.append(f"t{tid}={side}")
            reg_table.add_row(
                r.rally_id[:8], r.video_id[:8],
                r.gt_side, r.img_side, r.court_side,
                str(r.court_tid),
                cy_str,
                ", ".join(side_check),
            )
        console.print(reg_table)

    # Improvements detail
    improvements = [r for r in results if r.impact == "improved"]
    if improvements:
        console.print(f"\n[bold green]Improvements ({len(improvements)})[/bold green]")
        imp_table = Table()
        imp_table.add_column("Rally")
        imp_table.add_column("Video")
        imp_table.add_column("GT Side")
        imp_table.add_column("Img Side")
        imp_table.add_column("Court Side")
        imp_table.add_column("Court Y Values")
        for r in improvements:
            cy_str = ", ".join(f"t{tid}={cy:.1f}" for tid, cy in sorted(r.court_y_values.items()))
            imp_table.add_row(
                r.rally_id[:8], r.video_id[:8],
                r.gt_side, r.img_side, r.court_side,
                cy_str,
            )
        console.print(imp_table)

    # Per-video regression analysis
    vid_regressions = Counter(r.video_id[:8] for r in regressions)
    if vid_regressions:
        console.print("\n[bold]Regressions by Video[/bold]")
        for vid, count in vid_regressions.most_common():
            total_vid = sum(1 for r in results if r.video_id[:8] == vid)
            console.print(f"  {vid}: {count}/{total_vid} regressed")

    # Court-Y distribution analysis for regressions
    if regressions:
        console.print("\n[bold]Court-Y Analysis for Regressions[/bold]")
        for r in regressions:
            console.print(f"\n  Rally {r.rally_id[:8]} (video {r.video_id[:8]}):")
            console.print(f"    GT side: {r.gt_side}")
            console.print(f"    Image: server=t{r.img_tid} side={r.img_side} conf={r.img_conf:.2f}")
            console.print(f"    Court: server=t{r.court_tid} side={r.court_side} conf={r.court_conf:.2f}")
            if r.court_y_values:
                for tid, cy in sorted(r.court_y_values.items()):
                    dist_from_net = abs(cy - 8.0)
                    side = "near" if cy < 8 else "far"
                    marker = " ← SERVER" if tid == r.court_tid else ""
                    console.print(f"    t{tid}: court_y={cy:6.1f} ({side}, {dist_from_net:.1f}m from net){marker}")


if __name__ == "__main__":
    main()
