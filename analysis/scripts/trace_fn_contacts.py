"""Trace specific FN contacts through the pipeline step by step.

Shows candidate generation, refinement, feature extraction, and classifier
scores for GT contacts that are currently FN. Helps diagnose why "easy"
contacts are being missed.

Usage:
    cd analysis
    uv run python scripts/trace_fn_contacts.py
    uv run python scripts/trace_fn_contacts.py --rally <id>
"""

from __future__ import annotations

import argparse
import csv
import math

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_classifier import (
    CandidateFeatures,
    load_contact_classifier,
)
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _compute_acceleration,
    _compute_trajectory_curvature,
    _compute_velocities,
    _compute_velocity_ratio,
    _count_consecutive_detections,
    _check_net_crossing,
    _filter_noise_spikes,
    _find_nearest_player,
    _smooth_signal,
    compute_direction_change,
    detect_contacts,
    estimate_net_position,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
)
from scripts.train_contact_classifier import extract_candidate_features

console = Console()

_CONFIDENCE_THRESHOLD = 0.3


def load_feedback(path: str = "/Users/mario/Downloads/review_feedback.csv") -> dict[tuple[str, int], dict]:
    """Load feedback CSV keyed by (rally_id, gt_frame)."""
    feedback: dict[tuple[str, int], dict] = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["rally_id"], int(row["gt_frame"]))
                feedback[key] = row
    except FileNotFoundError:
        pass
    return feedback


def trace_rally(
    rally: RallyData,
    cfg: ContactDetectionConfig,
    classifier: object | None,
    feedback: dict[tuple[str, int], dict],
    show_all: bool = False,
) -> int:
    """Trace FN contacts in a rally. Returns count of FNs traced."""
    if not rally.ball_positions_json or not rally.gt_labels:
        return 0

    # Run detection to get current contacts
    ball_positions_raw = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]
    if not ball_positions_raw:
        return 0

    player_positions: list[PlayerPos] = []
    if rally.positions_json:
        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
                keypoints=pp.get("keypoints"),
            )
            for pp in rally.positions_json
        ]

    # Get current detected contacts
    contacts = detect_contacts(
        ball_positions_raw, player_positions,
        frame_count=rally.frame_count,
    )
    detected_frames = {c.frame for c in contacts.contacts}

    # Extract candidate features (with refinement)
    features_list, candidate_frames = extract_candidate_features(rally, config=cfg)

    # Score all candidates with current classifier
    score_by_frame: dict[int, float] = {}
    if classifier is not None and features_list:
        preds = classifier.predict(features_list)
        for feat, (_, score) in zip(features_list, preds):
            score_by_frame[feat.frame] = score

    # Ball by frame for trajectory analysis
    ball_positions = ball_positions_raw
    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)
    ball_by_frame = {
        bp.frame_number: bp for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }

    estimated_net_y = (
        rally.court_split_y if rally.court_split_y is not None
        else estimate_net_position(ball_positions)
    )

    velocities = _compute_velocities(ball_positions)
    frames_sorted = sorted(velocities.keys())
    if len(frames_sorted) < 3:
        return 0
    speeds = [velocities[f][0] for f in frames_sorted]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)
    velocity_lookup = dict(zip(frames_sorted, smoothed))

    traced = 0
    for gt in rally.gt_labels:
        # Check if this GT contact is FN (not detected)
        is_fn = not any(abs(gt.frame - d) <= 5 for d in detected_frames)
        if not is_fn and not show_all:
            continue

        # Get feedback tag if available
        fb = feedback.get((rally.rally_id, gt.frame), {})
        tag = fb.get("tag", "")
        notes = fb.get("notes", "")

        # Find nearest candidate
        nearest_cand = None
        nearest_dist = float("inf")
        nearest_score = None
        for cf in candidate_frames:
            d = abs(cf - gt.frame)
            if d < nearest_dist:
                nearest_dist = d
                nearest_cand = cf
                nearest_score = score_by_frame.get(cf)

        # Ball data at GT frame
        ball_at_gt = ball_by_frame.get(gt.frame)
        ball_nearby = None
        if ball_at_gt is None:
            for off in range(1, 6):
                for sign in [-1, 1]:
                    b = ball_by_frame.get(gt.frame + sign * off)
                    if b is not None:
                        ball_nearby = (gt.frame + sign * off, b)
                        break
                if ball_nearby:
                    break

        # Direction change at GT frame
        dir_change_gt = compute_direction_change(ball_by_frame, gt.frame, cfg.direction_check_frames)

        # Velocity at GT frame
        vel_gt = velocity_lookup.get(gt.frame, 0.0)

        # Nearest player at GT frame
        player_dist_gt = float("inf")
        if player_positions and ball_at_gt:
            _, player_dist_gt, _ = _find_nearest_player(
                gt.frame, ball_at_gt.x, ball_at_gt.y, player_positions,
                search_frames=cfg.player_search_frames,
            )
        elif player_positions and ball_nearby:
            _, f_ball = ball_nearby
            _, player_dist_gt, _ = _find_nearest_player(
                gt.frame, f_ball.x, f_ball.y, player_positions,
                search_frames=cfg.player_search_frames,
            )

        # Print trace
        status = "[red]FN[/red]" if is_fn else "[green]TP[/green]"
        tag_str = f" [{tag}]" if tag else ""
        console.print(f"\n{status} {rally.rally_id[:8]} f={gt.frame} {gt.action}{tag_str}")

        # Ball data
        if ball_at_gt:
            console.print(f"  Ball at GT frame: ({ball_at_gt.x:.3f}, {ball_at_gt.y:.3f}) conf={ball_at_gt.confidence:.2f}")
        elif ball_nearby:
            f_near, b_near = ball_nearby
            console.print(f"  Ball at GT frame: [yellow]MISSING[/yellow], nearest at f={f_near} ({b_near.x:.3f}, {b_near.y:.3f})")
        else:
            console.print(f"  Ball at GT frame: [red]NO BALL within ±5 frames[/red]")

        # Trajectory signals at GT frame
        console.print(f"  Velocity: {vel_gt:.4f} (min={cfg.min_candidate_velocity})")
        console.print(f"  Direction change: {dir_change_gt:.1f}° (min={cfg.min_direction_change_deg}°)")
        console.print(f"  Player dist: {player_dist_gt:.3f}" + (" [red]> 0.15[/red]" if player_dist_gt > 0.15 else ""))

        # Nearest candidate
        if nearest_cand is not None:
            console.print(f"  Nearest candidate: f={nearest_cand} (offset={nearest_cand - gt.frame:+d})")
            if nearest_score is not None:
                color = "green" if nearest_score >= 0.25 else "yellow" if nearest_score >= 0.15 else "red"
                console.print(f"  Classifier score: [{color}]{nearest_score:.3f}[/{color}] (threshold=0.25)")

                # Show direction change at candidate frame
                dir_at_cand = compute_direction_change(ball_by_frame, nearest_cand, cfg.direction_check_frames)
                vel_at_cand = velocity_lookup.get(nearest_cand, 0.0)
                console.print(f"  At candidate: vel={vel_at_cand:.4f}, dir={dir_at_cand:.1f}°")
            else:
                console.print(f"  Classifier score: [red]not scored (filtered before GBM)[/red]")
        else:
            console.print(f"  Nearest candidate: [red]NONE[/red]")

        if notes:
            console.print(f"  Notes: {notes}")

        traced += 1

    return traced


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace FN contacts through pipeline")
    parser.add_argument("--rally", type=str, help="Specific rally ID prefix")
    parser.add_argument("--tag", type=str, help="Only show FNs with this feedback tag")
    parser.add_argument("--limit", type=int, default=30, help="Max FNs to show")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    feedback = load_feedback()

    # Filter to rallies with feedback FN_contacts
    fn_rallies = set()
    for (rid, frame), fb in feedback.items():
        if fb.get("error_class") == "FN_contact":
            if args.tag is None or fb.get("tag") == args.tag:
                fn_rallies.add(rid)

    if args.rally:
        rallies = [r for r in rallies if r.rally_id.startswith(args.rally)]
    elif fn_rallies:
        rallies = [r for r in rallies if r.rally_id in fn_rallies]

    if not rallies:
        console.print("[red]No rallies found.[/red]")
        return

    console.print(f"[bold]Tracing FN contacts in {len(rallies)} rallies[/bold]")

    cfg = ContactDetectionConfig()
    classifier = load_contact_classifier()

    total_traced = 0
    for rally in rallies:
        n = trace_rally(rally, cfg, classifier, feedback)
        total_traced += n
        if total_traced >= args.limit:
            console.print(f"\n[yellow]Reached limit of {args.limit} FNs. Use --limit to show more.[/yellow]")
            break

    console.print(f"\n[bold]Total FNs traced: {total_traced}[/bold]")


if __name__ == "__main__":
    main()
