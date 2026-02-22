"""Diagnose false negative contacts — categorize each missed GT contact by failure mode.

For each FN, determines WHY the contact was missed:
- no_ball_data:       No ball position within ±5 frames — upstream ball tracking failure
- no_candidate:       Ball data exists but no candidate generator fired — generator gap
- rejected_candidate: Candidate existed but classifier/gates rejected it — threshold issue

Outputs:
1. Category summary (counts + percentages)
2. Per-action breakdown
3. Per-rally breakdown (top 15 worst)
4. Signal statistics (velocity, direction change, player distance distributions)
5. Rejected candidate analysis (classifier confidence buckets, generator coverage)
6. Full per-FN detail table

Usage:
    cd analysis
    uv run python scripts/diagnose_fn_contacts.py
    uv run python scripts/diagnose_fn_contacts.py --rally <id>
    uv run python scripts/diagnose_fn_contacts.py --tolerance-ms 200
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.signal import find_peaks

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_classifier import ContactClassifier, load_contact_classifier
from rallycut.tracking.contact_detector import (
    _CONFIDENCE_THRESHOLD,
    ContactDetectionConfig,
    _compute_direction_change,
    _compute_velocities,
    _filter_noise_spikes,
    _find_deceleration_candidates,
    _find_inflection_candidates,
    _find_nearest_player,
    _find_net_crossing_candidates,
    _find_parabolic_breakpoints,
    _find_velocity_reversal_candidates,
    _merge_candidates,
    _smooth_signal,
    detect_contacts,
    estimate_net_position,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

BALL_SEARCH_WINDOW = 5  # ±frames to check for ball data near GT frame


@dataclass
class FNDiagnostic:
    """Diagnostic info for a single false negative contact."""

    rally_id: str
    gt_frame: int
    gt_action: str

    # Category
    category: str  # no_ball_data | no_candidate | rejected_candidate

    # Ball data
    ball_present: bool  # Ball detected at GT frame
    ball_gap_frames: int  # Distance to nearest ball detection (0 = at frame)
    ball_confidence: float  # Ball confidence at GT frame (0 if absent)

    # Trajectory signals at GT frame
    velocity: float  # Smoothed speed
    direction_change_deg: float  # Angle change
    player_distance: float  # Nearest player distance
    player_present: bool  # Any player within 0.15

    # Candidate info
    generators_fired: list[str]  # Which generators produced a candidate nearby
    nearest_candidate_frame: int | None  # Closest candidate to GT frame
    nearest_candidate_distance: int  # Frame distance to nearest candidate
    nearest_candidate_accepted: bool  # Was it accepted by classifier?
    classifier_confidence: float  # Classifier score of nearest candidate (0 if none)


def _reconstruct_ball_player_data(
    rally: RallyData,
) -> tuple[list[BallPos], list[PlayerPos]]:
    """Reconstruct BallPosition and PlayerPosition lists from rally JSON data."""
    ball_positions: list[BallPos] = []
    if rally.ball_positions_json:
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

    player_positions: list[PlayerPos] = []
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

    return ball_positions, player_positions


def _get_all_candidates_with_scores(
    ball_positions: list[BallPos],
    player_positions: list[PlayerPos],
    net_y: float | None,
    frame_count: int | None,
    classifier: ContactClassifier | None,
) -> list[tuple[int, float, bool]]:
    """Run detection with threshold=0 to get ALL candidates with their scores.

    Returns list of (frame, confidence, accepted_at_default_threshold).
    """
    if classifier is None or not classifier.is_trained:
        # Fall back to hand-tuned: run once normally, once with permissive config
        normal_result = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=net_y,
            frame_count=frame_count,
            use_classifier=False,
        )
        accepted_frames = {c.frame for c in normal_result.contacts}

        # Run with very permissive hand-tuned config to get all candidates
        permissive_cfg = ContactDetectionConfig(
            min_peak_velocity=0.001,
            min_peak_prominence=0.0,
            min_direction_change_deg=0.0,
            high_velocity_threshold=0.001,
            min_candidate_velocity=0.001,
        )
        permissive_result = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=permissive_cfg,
            net_y=net_y,
            frame_count=frame_count,
            use_classifier=False,
        )
        return [
            (c.frame, c.confidence, c.frame in accepted_frames)
            for c in permissive_result.contacts
        ]

    # Run with threshold=0 to accept everything
    permissive_clf = deepcopy(classifier)
    permissive_clf.threshold = 0.0

    permissive_result = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        net_y=net_y,
        frame_count=frame_count,
        classifier=permissive_clf,
        use_classifier=True,
    )

    # Run with normal threshold
    normal_result = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        net_y=net_y,
        frame_count=frame_count,
        classifier=classifier,
        use_classifier=True,
    )
    accepted_frames = {c.frame for c in normal_result.contacts}

    return [
        (c.frame, c.confidence, c.frame in accepted_frames)
        for c in permissive_result.contacts
    ]


def _get_raw_candidates(
    ball_positions: list[BallPos],
    net_y: float | None,
    cfg: ContactDetectionConfig,
) -> list[int]:
    """Reproduce the candidate generation step (no validation) to get raw candidates."""
    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)

    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return []

    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return []

    speeds = [velocities[f][0] for f in frames]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)

    peak_indices, _ = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )
    velocity_peak_frames = [frames[idx] for idx in peak_indices]

    ball_by_frame: dict[int, BallPos] = {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    confident_frames = sorted(ball_by_frame.keys())

    inflection_frames: list[int] = []
    if cfg.enable_inflection_detection:
        inflection_frames = _find_inflection_candidates(
            ball_by_frame, confident_frames,
            min_angle_deg=cfg.min_inflection_angle_deg,
            check_frames=cfg.inflection_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    reversal_frames = _find_velocity_reversal_candidates(
        velocities, frames, cfg.min_peak_distance_frames
    )

    deceleration_frames: list[int] = []
    if cfg.enable_deceleration_detection:
        deceleration_frames = _find_deceleration_candidates(
            velocities, frames, smoothed,
            cfg.min_peak_distance_frames,
            min_speed_before=cfg.deceleration_min_speed_before,
            min_speed_drop_ratio=cfg.deceleration_min_drop_ratio,
            window=cfg.deceleration_window,
        )

    parabolic_frames: list[int] = []
    if cfg.enable_parabolic_detection:
        parabolic_frames, _ = _find_parabolic_breakpoints(
            ball_by_frame, confident_frames,
            window_frames=cfg.parabolic_window_frames,
            stride=cfg.parabolic_stride,
            min_residual=cfg.parabolic_min_residual,
            min_prominence=cfg.parabolic_min_prominence,
            min_distance_frames=cfg.min_peak_distance_frames,
        )

    estimated_net_y = net_y if net_y is not None else estimate_net_position(ball_positions)
    net_crossing_frames = _find_net_crossing_candidates(
        ball_by_frame, confident_frames, estimated_net_y, cfg.min_peak_distance_frames
    )

    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    traditional = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )
    with_deceleration = _merge_candidates(
        traditional, deceleration_frames, cfg.min_peak_distance_frames
    )
    with_parabolic = _merge_candidates(
        with_deceleration, parabolic_frames, cfg.min_peak_distance_frames
    )
    return _merge_candidates(
        with_parabolic, net_crossing_frames, cfg.min_peak_distance_frames
    )


def _identify_generators_near_frame(
    gt_frame: int,
    ball_by_frame: dict[int, BallPos],
    confident_frames: list[int],
    velocities: dict[int, tuple[float, float, float]],
    velocity_frames: list[int],
    smoothed_speeds: list[float],
    net_y: float,
    tolerance: int,
    cfg: ContactDetectionConfig,
) -> list[str]:
    """Check which individual candidate generators fire within ±tolerance of gt_frame."""
    fired: list[str] = []

    # Velocity peaks
    if len(smoothed_speeds) >= 3:
        peak_indices, _ = find_peaks(
            smoothed_speeds,
            height=cfg.min_peak_velocity,
            prominence=cfg.min_peak_prominence,
            distance=cfg.min_peak_distance_frames,
        )
        vel_peak_frames = [velocity_frames[i] for i in peak_indices]
        if any(abs(f - gt_frame) <= tolerance for f in vel_peak_frames):
            fired.append("velocity_peak")

    # Inflections
    if cfg.enable_inflection_detection and len(confident_frames) >= 3:
        infl_frames = _find_inflection_candidates(
            ball_by_frame, confident_frames,
            min_angle_deg=cfg.min_inflection_angle_deg,
            check_frames=cfg.inflection_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
        )
        if any(abs(f - gt_frame) <= tolerance for f in infl_frames):
            fired.append("inflection")

    # Reversals
    if len(velocity_frames) >= 3:
        rev_frames = _find_velocity_reversal_candidates(
            velocities, velocity_frames, cfg.min_peak_distance_frames,
        )
        if any(abs(f - gt_frame) <= tolerance for f in rev_frames):
            fired.append("reversal")

    # Deceleration
    if cfg.enable_deceleration_detection and len(velocity_frames) >= cfg.deceleration_window * 2 + 1:
        decel_frames = _find_deceleration_candidates(
            velocities, velocity_frames, smoothed_speeds,
            cfg.min_peak_distance_frames,
            min_speed_before=cfg.deceleration_min_speed_before,
            min_speed_drop_ratio=cfg.deceleration_min_drop_ratio,
            window=cfg.deceleration_window,
        )
        if any(abs(f - gt_frame) <= tolerance for f in decel_frames):
            fired.append("deceleration")

    # Parabolic
    if cfg.enable_parabolic_detection and len(confident_frames) >= cfg.parabolic_window_frames:
        para_frames, _ = _find_parabolic_breakpoints(
            ball_by_frame, confident_frames,
            window_frames=cfg.parabolic_window_frames,
            stride=cfg.parabolic_stride,
            min_residual=cfg.parabolic_min_residual,
            min_prominence=cfg.parabolic_min_prominence,
            min_distance_frames=cfg.min_peak_distance_frames,
        )
        if any(abs(f - gt_frame) <= tolerance for f in para_frames):
            fired.append("parabolic")

    # Net crossing
    if len(confident_frames) >= 2:
        net_frames = _find_net_crossing_candidates(
            ball_by_frame, confident_frames, net_y, cfg.min_peak_distance_frames,
        )
        if any(abs(f - gt_frame) <= tolerance for f in net_frames):
            fired.append("net_crossing")

    return fired


def diagnose_rally_fns(
    rally: RallyData,
    fn_labels: list[GtLabel],
    classifier: ContactClassifier | None,
    tolerance_frames: int,
) -> list[FNDiagnostic]:
    """Compute diagnostics for each FN contact in a rally."""
    if not fn_labels:
        return []

    ball_positions, player_positions = _reconstruct_ball_player_data(rally)
    cfg = ContactDetectionConfig()

    if not ball_positions:
        return [
            FNDiagnostic(
                rally_id=rally.rally_id,
                gt_frame=gt.frame,
                gt_action=gt.action,
                category="no_ball_data",
                ball_present=False,
                ball_gap_frames=9999,
                ball_confidence=0.0,
                velocity=0.0,
                direction_change_deg=0.0,
                player_distance=float("inf"),
                player_present=False,
                generators_fired=[],
                nearest_candidate_frame=None,
                nearest_candidate_distance=9999,
                nearest_candidate_accepted=False,
                classifier_confidence=0.0,
            )
            for gt in fn_labels
        ]

    # Pre-filter noise spikes (same as detect_contacts)
    if cfg.enable_noise_filter:
        filtered_bp = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)
    else:
        filtered_bp = ball_positions

    # Build ball lookup
    ball_by_frame: dict[int, BallPos] = {
        bp.frame_number: bp
        for bp in filtered_bp
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    confident_frames = sorted(ball_by_frame.keys())

    # Compute velocities and smooth
    velocities = _compute_velocities(filtered_bp)
    velocity_frames = sorted(velocities.keys())
    speeds = [velocities[f][0] for f in velocity_frames]
    smoothed_speeds = _smooth_signal(speeds, cfg.smoothing_window) if speeds else []
    speed_lookup = dict(zip(velocity_frames, smoothed_speeds))

    # Net Y
    net_y = rally.court_split_y or 0.5

    # Get all candidates with classifier scores
    all_candidates = _get_all_candidates_with_scores(
        ball_positions, player_positions, net_y, rally.frame_count or None, classifier,
    )

    # Also get raw candidates (pre-validation) for "no_candidate" check
    raw_candidates = _get_raw_candidates(ball_positions, net_y, cfg)

    diagnostics: list[FNDiagnostic] = []

    for gt in fn_labels:
        frame = gt.frame

        # Ball data check
        ball_at_frame = ball_by_frame.get(frame)
        ball_present = ball_at_frame is not None
        ball_confidence = ball_at_frame.confidence if ball_at_frame else 0.0

        # Find nearest ball detection
        ball_gap = 0
        if not ball_present:
            if confident_frames:
                ball_gap = min(abs(f - frame) for f in confident_frames)
            else:
                ball_gap = 9999

        has_ball_nearby = ball_gap <= BALL_SEARCH_WINDOW

        # Trajectory signals at GT frame (check nearby frames if exact frame missing)
        velocity = speed_lookup.get(frame, 0.0)
        if velocity == 0.0 and frame not in speed_lookup:
            for offset in range(1, 4):
                if frame + offset in speed_lookup:
                    velocity = speed_lookup[frame + offset]
                    break
                if frame - offset in speed_lookup:
                    velocity = speed_lookup[frame - offset]
                    break

        direction_change = 0.0
        if has_ball_nearby:
            direction_change = _compute_direction_change(ball_by_frame, frame, check_frames=8)

        # Player distance
        player_distance = float("inf")
        player_present = False
        ball_x = ball_at_frame.x if ball_at_frame else (gt.ball_x or 0.5)
        ball_y = ball_at_frame.y if ball_at_frame else (gt.ball_y or 0.5)
        if player_positions and (ball_at_frame or gt.ball_x is not None):
            _, player_distance, _ = _find_nearest_player(
                frame, ball_x, ball_y, player_positions, search_frames=5,
            )
            player_present = player_distance <= 0.15

        # Which generators fired near this frame?
        generators_fired: list[str] = []
        if has_ball_nearby:
            generators_fired = _identify_generators_near_frame(
                frame, ball_by_frame, confident_frames,
                velocities, velocity_frames, smoothed_speeds,
                net_y, tolerance_frames, cfg,
            )

        # Nearest candidate info (from permissive run)
        nearest_candidate_frame = None
        nearest_candidate_distance = 9999
        nearest_candidate_accepted = False
        classifier_confidence = 0.0

        if all_candidates:
            best_dist = 9999
            for cand_frame, cand_conf, cand_accepted in all_candidates:
                dist = abs(cand_frame - frame)
                if dist < best_dist:
                    best_dist = dist
                    nearest_candidate_frame = cand_frame
                    nearest_candidate_distance = dist
                    nearest_candidate_accepted = cand_accepted
                    classifier_confidence = cand_conf

        # Check if raw candidates (pre-validation) exist near this frame
        raw_near = any(abs(c - frame) <= tolerance_frames for c in raw_candidates)

        # Categorize
        if not has_ball_nearby:
            category = "no_ball_data"
        elif nearest_candidate_distance <= tolerance_frames and not nearest_candidate_accepted:
            # A candidate was generated but rejected by the classifier
            category = "rejected_candidate"
        elif raw_near and nearest_candidate_distance > tolerance_frames:
            # Raw candidate existed but got filtered by velocity floor/warmup/dedup
            category = "rejected_candidate"
        elif generators_fired:
            # Generators fired individually but no merged candidate survived
            category = "rejected_candidate"
        else:
            category = "no_candidate"

        diagnostics.append(FNDiagnostic(
            rally_id=rally.rally_id,
            gt_frame=frame,
            gt_action=gt.action,
            category=category,
            ball_present=ball_present,
            ball_gap_frames=ball_gap,
            ball_confidence=ball_confidence,
            velocity=velocity,
            direction_change_deg=direction_change,
            player_distance=player_distance,
            player_present=player_present,
            generators_fired=generators_fired,
            nearest_candidate_frame=nearest_candidate_frame,
            nearest_candidate_distance=nearest_candidate_distance,
            nearest_candidate_accepted=nearest_candidate_accepted,
            classifier_confidence=classifier_confidence,
        ))

    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose false negative contacts")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument(
        "--tolerance-ms", type=int, default=167,
        help="Time tolerance in ms for matching (default: 167, ~5 frames at 30fps)",
    )
    args = parser.parse_args()

    # Load classifier
    classifier = load_contact_classifier()
    if classifier is None:
        console.print("[yellow]No trained contact classifier found — using hand-tuned gates[/yellow]")
    else:
        console.print(f"[dim]Loaded contact classifier (threshold={classifier.threshold:.2f})[/dim]")

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies found with action ground truth labels.[/red]")
        return

    console.print(f"\n[bold]Diagnosing FN contacts across {len(rallies)} rallies[/bold]\n")

    all_diagnostics: list[FNDiagnostic] = []
    total_gt = 0
    total_tp = 0
    total_fn = 0

    for i, rally in enumerate(rallies):
        console.print(f"  [{i + 1}/{len(rallies)}] {rally.rally_id[:8]}...", end="\r")

        ball_positions, player_positions = _reconstruct_ball_player_data(rally)
        pred_actions: list[dict] = []

        if ball_positions:
            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=rally.court_split_y,
                frame_count=rally.frame_count or None,
                classifier=classifier,
                use_classifier=True,
            )
            # Use contact frames directly for matching (action type not needed)
            pred_actions = [c.to_dict() for c in contacts.contacts]

        tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))

        matches, _unmatched = match_contacts(
            rally.gt_labels, pred_actions, tolerance=tolerance_frames,
        )

        fn_labels = [
            GtLabel(
                frame=m.gt_frame,
                action=m.gt_action,
                player_track_id=-1,
                ball_x=next(
                    (gt.ball_x for gt in rally.gt_labels if gt.frame == m.gt_frame), None
                ),
                ball_y=next(
                    (gt.ball_y for gt in rally.gt_labels if gt.frame == m.gt_frame), None
                ),
            )
            for m in matches if m.pred_frame is None
        ]

        total_gt += len(matches)
        total_tp += sum(1 for m in matches if m.pred_frame is not None)
        total_fn += len(fn_labels)

        if fn_labels:
            diags = diagnose_rally_fns(rally, fn_labels, classifier, tolerance_frames)
            all_diagnostics.extend(diags)

    console.print()  # Clear progress line

    if not all_diagnostics:
        console.print("[green]No false negatives found![/green]")
        return

    # === 1. Summary ===
    console.print(
        f"[bold]Total: {total_gt} GT, {total_tp} TP, {total_fn} FN "
        f"(Recall {total_tp / max(1, total_gt):.1%})[/bold]\n"
    )

    # === 2. Category breakdown ===
    cat_counts: dict[str, int] = Counter(d.category for d in all_diagnostics)

    cat_table = Table(title="FN Categories")
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("Count", justify="right")
    cat_table.add_column("%", justify="right")
    for cat in ["no_ball_data", "no_candidate", "rejected_candidate"]:
        count = cat_counts.get(cat, 0)
        pct = count / len(all_diagnostics) * 100
        cat_table.add_row(cat, str(count), f"{pct:.1f}%")
    console.print(cat_table)

    # === 3. Per-action breakdown ===
    action_cat: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in all_diagnostics:
        action_cat[d.gt_action][d.category] += 1
        action_cat[d.gt_action]["total"] += 1

    action_table = Table(title="\nFN by Action Type")
    action_table.add_column("Action", style="bold")
    action_table.add_column("Total FN", justify="right")
    action_table.add_column("no_ball", justify="right")
    action_table.add_column("no_cand", justify="right")
    action_table.add_column("rejected", justify="right")
    for action in sorted(action_cat.keys(), key=lambda a: -action_cat[a]["total"]):
        cats = action_cat[action]
        action_table.add_row(
            action,
            str(cats["total"]),
            str(cats.get("no_ball_data", 0)),
            str(cats.get("no_candidate", 0)),
            str(cats.get("rejected_candidate", 0)),
        )
    console.print(action_table)

    # === 4. Per-rally breakdown ===
    rally_fns: dict[str, list[FNDiagnostic]] = defaultdict(list)
    for d in all_diagnostics:
        rally_fns[d.rally_id].append(d)

    rally_table = Table(title="\nFN by Rally (top 15)")
    rally_table.add_column("Rally", style="dim", max_width=10)
    rally_table.add_column("FN", justify="right")
    rally_table.add_column("no_ball", justify="right")
    rally_table.add_column("no_cand", justify="right")
    rally_table.add_column("rejected", justify="right")
    rally_table.add_column("Actions", max_width=30)

    sorted_rallies = sorted(rally_fns.items(), key=lambda x: -len(x[1]))
    for rally_id, diags in sorted_rallies[:15]:
        rally_cats = Counter(d.category for d in diags)
        rally_actions = Counter(d.gt_action for d in diags)
        action_str = ", ".join(f"{a}:{n}" for a, n in rally_actions.most_common())
        rally_table.add_row(
            rally_id[:8],
            str(len(diags)),
            str(rally_cats.get("no_ball_data", 0)),
            str(rally_cats.get("no_candidate", 0)),
            str(rally_cats.get("rejected_candidate", 0)),
            action_str,
        )
    console.print(rally_table)

    # === 5. Signal statistics for FNs with ball data ===
    signal_fns = [d for d in all_diagnostics if d.category != "no_ball_data"]
    if signal_fns:
        console.print("\n[bold]Signal Statistics (FNs with ball data)[/bold]")

        velocities_vals = [d.velocity for d in signal_fns if d.velocity > 0]
        dir_changes = [d.direction_change_deg for d in signal_fns if d.direction_change_deg > 0]
        player_dists = [d.player_distance for d in signal_fns if math.isfinite(d.player_distance)]

        if velocities_vals:
            console.print(
                f"  Velocity: median={np.median(velocities_vals):.4f}, "
                f"mean={np.mean(velocities_vals):.4f}, "
                f"range=[{min(velocities_vals):.4f}, {max(velocities_vals):.4f}]"
            )
        if dir_changes:
            console.print(
                f"  Direction change: median={np.median(dir_changes):.1f}deg, "
                f"mean={np.mean(dir_changes):.1f}deg, "
                f"range=[{min(dir_changes):.1f}, {max(dir_changes):.1f}]"
            )
        if player_dists:
            within = sum(1 for d in player_dists if d < 0.15)
            console.print(
                f"  Player distance: median={np.median(player_dists):.3f}, "
                f"mean={np.mean(player_dists):.3f}, "
                f"<0.15={within}/{len(player_dists)} ({within / len(player_dists):.0%})"
            )

        # No-candidate specific stats
        no_cand = [d for d in all_diagnostics if d.category == "no_candidate"]
        if no_cand:
            console.print(f"\n  [bold]no_candidate ({len(no_cand)}) breakdown:[/bold]")
            nc_vel = [d.velocity for d in no_cand if d.velocity > 0]
            nc_dir = [d.direction_change_deg for d in no_cand if d.direction_change_deg > 0]
            if nc_vel:
                console.print(
                    f"    Velocity: median={np.median(nc_vel):.4f}, "
                    f"<0.008 (peak threshold): {sum(1 for v in nc_vel if v < 0.008)}/{len(nc_vel)}"
                )
            else:
                console.print("    Velocity: all zero (no ball data at exact frame)")
            if nc_dir:
                console.print(
                    f"    Direction: median={np.median(nc_dir):.1f}deg, "
                    f"<15 (inflection threshold): {sum(1 for v in nc_dir if v < 15)}/{len(nc_dir)}"
                )

    # === 6. Rejected candidate analysis ===
    rejected = [d for d in all_diagnostics if d.category == "rejected_candidate"]
    if rejected:
        console.print(f"\n[bold]Rejected Candidates ({len(rejected)})[/bold]")
        confidences = [d.classifier_confidence for d in rejected if d.classifier_confidence > 0]
        if confidences:
            console.print(
                f"  Classifier confidence: median={np.median(confidences):.3f}, "
                f"mean={np.mean(confidences):.3f}, "
                f"max={max(confidences):.3f}"
            )
            buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
            bucket_strs = []
            for lo, hi in buckets:
                n = sum(1 for c in confidences if lo <= c < hi)
                if n > 0:
                    bucket_strs.append(f"[{lo:.1f},{hi:.1f}):{n}")
            if bucket_strs:
                console.print(f"  Confidence buckets: {', '.join(bucket_strs)}")
        else:
            console.print("  No classifier confidence data (hand-tuned gates)")

        # Generator coverage
        gen_counts: dict[str, int] = defaultdict(int)
        no_gen = 0
        for d in rejected:
            if d.generators_fired:
                for g in d.generators_fired:
                    gen_counts[g] += 1
            else:
                no_gen += 1
        console.print("  Generator coverage:")
        for gen, count in sorted(gen_counts.items(), key=lambda x: -x[1]):
            console.print(f"    {gen}: {count}")
        if no_gen:
            console.print(f"    (no generator fired): {no_gen}")

        # Nearest candidate distance distribution
        dists = [d.nearest_candidate_distance for d in rejected if d.nearest_candidate_frame is not None]
        if dists:
            console.print(
                f"  Nearest candidate distance: median={int(np.median(dists))}f, "
                f"mean={np.mean(dists):.1f}f, "
                f"within tolerance: {sum(1 for d in dists if d <= 5)}/{len(dists)}"
            )

    # === 7. Full per-FN detail table ===
    detail_table = Table(title=f"\nAll FN Contacts ({len(all_diagnostics)})")
    detail_table.add_column("Rally", style="dim", max_width=8)
    detail_table.add_column("Frame", justify="right")
    detail_table.add_column("Action", max_width=7)
    detail_table.add_column("Category", max_width=12)
    detail_table.add_column("Ball", justify="center", max_width=4)
    detail_table.add_column("Gap", justify="right", max_width=4)
    detail_table.add_column("Vel", justify="right", max_width=7)
    detail_table.add_column("Dir", justify="right", max_width=5)
    detail_table.add_column("PlrDst", justify="right", max_width=6)
    detail_table.add_column("Generators", max_width=22)
    detail_table.add_column("CandDst", justify="right", max_width=7)
    detail_table.add_column("Conf", justify="right", max_width=5)

    for d in sorted(all_diagnostics, key=lambda x: (x.category, x.rally_id, x.gt_frame)):
        plr_str = f"{d.player_distance:.3f}" if math.isfinite(d.player_distance) else "inf"
        cand_str = str(d.nearest_candidate_distance) if d.nearest_candidate_frame is not None else "-"
        conf_str = f"{d.classifier_confidence:.2f}" if d.classifier_confidence > 0 else "-"
        gen_str = ",".join(d.generators_fired) if d.generators_fired else "-"

        detail_table.add_row(
            d.rally_id[:8],
            str(d.gt_frame),
            d.gt_action,
            d.category.replace("_candidate", ""),
            "Y" if d.ball_present else "N",
            str(d.ball_gap_frames) if d.ball_gap_frames > 0 else "-",
            f"{d.velocity:.4f}" if d.velocity > 0 else "-",
            f"{d.direction_change_deg:.0f}" if d.direction_change_deg > 0 else "-",
            plr_str,
            gen_str,
            cand_str,
            conf_str,
        )

    console.print(detail_table)


if __name__ == "__main__":
    main()
