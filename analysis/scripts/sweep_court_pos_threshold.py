"""Sweep confidence threshold for court-space position exemption from team overwrite.

For each rally, runs the full pipeline with different confidence thresholds
for when court_side_source="court_position" should block the team overwrite.

Usage:
    cd analysis
    uv run python scripts/sweep_court_pos_threshold.py
"""

from __future__ import annotations

import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import (
    ActionType,
    _find_server_by_position,
)
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts

from eval_sequence_enriched import RallyBundle, prepare_rallies

logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()


@dataclass
class RallyRecord:
    rally_id: str
    video_id: str
    gt_side: str

    # Position detection results
    court_side: str  # court-space detected side ("near"/"far"/"")
    court_conf: float  # confidence from court-space detection
    court_tid: int

    # What the team overwrite would say
    team_side: str  # team-based side for the server ("near"/"far"/"")

    # Is position detection correct?
    pos_correct: bool

    # Is team overwrite correct for the serve?
    team_correct: bool


def collect_records(bundles: list[RallyBundle]) -> list[RallyRecord]:
    """Collect per-rally records with both position and team signals."""
    video_ids = {b.rally.video_id for b in bundles}
    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal

    console.print(f"Calibrations: {len(calibrators)}/{len(video_ids)} videos")

    records: list[RallyRecord] = []
    for i, bundle in enumerate(bundles):
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

        # Contact detection to get player positions and net_y
        contact_seq = detect_contacts(
            ball_positions=bundle.ball_positions,
            player_positions=bundle.player_positions,
            config=ContactDetectionConfig(),
            frame_count=bundle.rally.frame_count,
        )
        player_positions = contact_seq.player_positions or []
        if not player_positions:
            continue

        cal = calibrators.get(bundle.rally.video_id)

        # Court-space position detection
        court_tid, court_side, court_conf = _find_server_by_position(
            player_positions, contact_seq.rally_start_frame,
            contact_seq.net_y, calibrator=cal,
        )

        # Team-based side for the detected server
        team_side = ""
        if court_tid >= 0 and bundle.match_teams:
            team = bundle.match_teams.get(court_tid)
            if team is not None:
                team_side = "near" if team == 0 else "far"

        pos_correct = court_side == gt_side if court_tid >= 0 else False
        team_correct = team_side == gt_side if team_side else False

        records.append(RallyRecord(
            rally_id=bundle.rally.rally_id,
            video_id=bundle.rally.video_id,
            gt_side=gt_side,
            court_side=court_side,
            court_conf=court_conf,
            court_tid=court_tid,
            team_side=team_side,
            pos_correct=pos_correct,
            team_correct=team_correct,
        ))

    return records


def sweep_thresholds(records: list[RallyRecord]) -> None:
    """Sweep confidence thresholds and report serve side accuracy."""
    # For each rally, determine serve side accuracy under different strategies:
    # 1. Always use team overwrite (baseline)
    # 2. Never use team overwrite for serves with court_position (no threshold)
    # 3. Only exempt from overwrite when court_conf >= threshold

    # Filter to records where court-space detected something AND team assignment exists
    has_both = [r for r in records if r.court_tid >= 0 and r.team_side]
    has_neither = [r for r in records if r.court_tid < 0 or not r.team_side]

    console.print(f"\nRecords with court-space detection + team assignment: {len(has_both)}")
    console.print(f"Records missing one or both: {len(has_neither)}")

    # Show confidence distribution for correct vs incorrect detections
    correct_confs = [r.court_conf for r in has_both if r.pos_correct]
    wrong_confs = [r.court_conf for r in has_both if not r.pos_correct]

    console.print(f"\n[bold]Confidence Distribution[/bold]")
    console.print(f"  Correct detections ({len(correct_confs)}): "
                  f"min={min(correct_confs):.2f} median={np.median(correct_confs):.2f} "
                  f"mean={np.mean(correct_confs):.2f} max={max(correct_confs):.2f}")
    if wrong_confs:
        console.print(f"  Wrong detections ({len(wrong_confs)}): "
                      f"min={min(wrong_confs):.2f} median={np.median(wrong_confs):.2f} "
                      f"mean={np.mean(wrong_confs):.2f} max={max(wrong_confs):.2f}")

    # Histogram
    bins = [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 1.01]
    console.print(f"\n[bold]Confidence Histogram[/bold]")
    hist_table = Table()
    hist_table.add_column("Bin")
    hist_table.add_column("Correct", justify="right")
    hist_table.add_column("Wrong", justify="right")
    hist_table.add_column("Precision", justify="right")
    for j in range(len(bins) - 1):
        lo, hi = bins[j], bins[j + 1]
        c = sum(1 for conf in correct_confs if lo <= conf < hi)
        w = sum(1 for conf in wrong_confs if lo <= conf < hi)
        prec = f"{c/(c+w):.0%}" if (c + w) > 0 else "-"
        hist_table.add_row(f"[{lo:.2f}, {hi:.2f})", str(c), str(w), prec)
    console.print(hist_table)

    # Disagreement analysis: where position and team disagree
    disagree = [r for r in has_both if r.court_side != r.team_side]
    agree = [r for r in has_both if r.court_side == r.team_side]
    console.print(f"\n[bold]Agreement Analysis[/bold]")
    console.print(f"  Agree: {len(agree)} (pos correct: {sum(1 for r in agree if r.pos_correct)}, "
                  f"team correct: {sum(1 for r in agree if r.team_correct)})")
    console.print(f"  Disagree: {len(disagree)} (pos correct: {sum(1 for r in disagree if r.pos_correct)}, "
                  f"team correct: {sum(1 for r in disagree if r.team_correct)})")

    if disagree:
        dis_confs = [r.court_conf for r in disagree]
        console.print(f"  Disagree confidence: min={min(dis_confs):.2f} "
                      f"median={np.median(dis_confs):.2f} max={max(dis_confs):.2f}")

    # Sweep thresholds
    thresholds = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]

    console.print(f"\n[bold]Threshold Sweep[/bold]")
    console.print("For each threshold: if court_conf >= threshold AND court ≠ team, use court; else use team")
    sweep_table = Table()
    sweep_table.add_column("Threshold")
    sweep_table.add_column("Exempted", justify="right")
    sweep_table.add_column("Correct (of exempted)", justify="right")
    sweep_table.add_column("Total Correct", justify="right")
    sweep_table.add_column("Accuracy", justify="right")
    sweep_table.add_column("Δ vs baseline", justify="right")

    # Baseline: always use team overwrite
    baseline_correct = 0
    for r in has_both:
        if r.team_correct:
            baseline_correct += 1
    # Add records without both signals (use their existing outcome)
    total = len(has_both)

    for thresh in thresholds:
        n_exempted = 0
        n_exempt_correct = 0
        n_correct = 0
        for r in has_both:
            if r.court_conf >= thresh and r.court_side != r.team_side:
                # Exempt from overwrite: use court-space side
                n_exempted += 1
                if r.pos_correct:
                    n_exempt_correct += 1
                    n_correct += 1
                # else: court was wrong, team would have been right or wrong
            else:
                # Use team overwrite as usual
                if r.team_correct:
                    n_correct += 1

        delta = n_correct - baseline_correct
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        style = "green" if delta > 0 else ("red" if delta < 0 else "")
        label = f"≥{thresh:.2f}" if thresh < 1.0 else "never exempt"
        exempt_acc = f"{n_exempt_correct}/{n_exempted}" if n_exempted > 0 else "-"
        sweep_table.add_row(
            label,
            str(n_exempted),
            exempt_acc,
            str(n_correct),
            f"{n_correct/total:.1%}",
            delta_str,
            style=style,
        )

    console.print(sweep_table)


def main() -> None:
    bundles = prepare_rallies(label_spread=2)
    if not bundles:
        console.print("[red]No rallies loaded.[/red]")
        return

    console.print(f"\n[bold]Court-Space Confidence Threshold Sweep[/bold]")
    records = collect_records(bundles)
    console.print(f"  {len(records)} rallies with evaluable serve side")
    sweep_thresholds(records)


if __name__ == "__main__":
    main()
