"""Diagnose trajectory refinement effect on contact detection.

Compares candidate frames WITH and WITHOUT refinement for every GT contact.
Shows per-FN impact: did refinement move candidates closer/farther from GT?
Which candidates got eaten by refinement dedup?

Usage:
    cd analysis
    uv run python scripts/diagnose_refinement.py
    uv run python scripts/diagnose_refinement.py --rally <id>
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.signal import find_peaks

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_classifier import (
    CandidateFeatures,
    load_contact_classifier,
)
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _compute_velocities,
    _filter_noise_spikes,
    _find_deceleration_candidates,
    _find_inflection_candidates,
    _find_net_crossing_candidates,
    _find_parabolic_breakpoints,
    _find_proximity_frame,
    _find_velocity_reversal_candidates,
    _merge_candidates,
    _refine_candidates_to_trajectory_peak,
    _smooth_signal,
    compute_direction_change,
    estimate_net_position,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
)
from scripts.train_contact_classifier import extract_candidate_features

console = Console()

_CONFIDENCE_THRESHOLD = 0.3
GT_TOLERANCE = 5  # ±frames for GT matching


@dataclass
class RefinementEffect:
    """Per-GT-contact refinement analysis."""

    rally_id: str
    gt_frame: int
    gt_action: str
    # Before refinement
    nearest_before: int | None  # nearest candidate frame (None = no candidate)
    offset_before: int | None  # signed offset from GT
    # After refinement
    nearest_after: int | None
    offset_after: int | None
    # Change
    moved_closer: bool | None  # True = closer, False = farther, None = N/A
    eaten_by_dedup: bool  # candidate existed pre-dedup but removed
    # Classifier scores (current model, at original vs refined frame)
    score_before: float | None
    score_after: float | None


def _generate_trajectory_candidates(
    rally: RallyData, cfg: ContactDetectionConfig,
) -> tuple[list[int], dict[int, BallPos], list[PlayerPos]]:
    """Generate trajectory-based candidates (no proximity, no refinement)."""
    if not rally.ball_positions_json:
        return [], {}, []

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
    if not ball_positions:
        return [], {}, []

    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)

    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return [], {}, []

    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return [], {}, []

    speeds = [velocities[f][0] for f in frames]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)

    ball_by_frame = {
        bp.frame_number: bp for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    confident_frames = sorted(ball_by_frame.keys())

    # Velocity peaks
    peak_indices, _ = find_peaks(
        smoothed, height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence, distance=cfg.min_peak_distance_frames,
    )
    velocity_peak_frames = [frames[idx] for idx in peak_indices]

    estimated_net_y = (
        rally.court_split_y if rally.court_split_y is not None
        else estimate_net_position(ball_positions)
    )

    inflection_frames = _find_inflection_candidates(
        ball_by_frame, confident_frames,
        min_angle_deg=cfg.min_inflection_angle_deg,
        check_frames=cfg.inflection_check_frames,
        min_distance_frames=cfg.min_peak_distance_frames,
    )
    reversal_frames = _find_velocity_reversal_candidates(
        velocities, frames, cfg.min_peak_distance_frames,
    )
    deceleration_frames: list[int] = []
    if cfg.enable_deceleration_detection:
        deceleration_frames = _find_deceleration_candidates(
            velocities, frames, smoothed, cfg.min_peak_distance_frames,
            min_speed_before=cfg.deceleration_min_speed_before,
            min_speed_drop_ratio=cfg.deceleration_min_drop_ratio,
            window=cfg.deceleration_window,
        )
    parabolic_frames, _ = _find_parabolic_breakpoints(
        ball_by_frame, confident_frames,
        window_frames=cfg.parabolic_window_frames, stride=cfg.parabolic_stride,
        min_residual=cfg.parabolic_min_residual,
        min_prominence=cfg.parabolic_min_prominence,
        min_distance_frames=cfg.min_peak_distance_frames,
    )
    net_crossing_frames = _find_net_crossing_candidates(
        ball_by_frame, confident_frames, estimated_net_y, cfg.min_peak_distance_frames,
    )

    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames,
    )
    traditional = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames,
    )
    with_deceleration = _merge_candidates(
        traditional, deceleration_frames, cfg.min_peak_distance_frames,
    )
    with_parabolic = _merge_candidates(
        with_deceleration, parabolic_frames, cfg.min_peak_distance_frames,
    )
    candidate_frames = _merge_candidates(
        with_parabolic, net_crossing_frames, cfg.min_peak_distance_frames,
    )

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

    return candidate_frames, ball_by_frame, player_positions


def _nearest_candidate(gt_frame: int, candidates: list[int]) -> tuple[int | None, int | None]:
    """Find nearest candidate to GT frame. Returns (frame, signed_offset)."""
    if not candidates:
        return None, None
    dists = [(abs(c - gt_frame), c - gt_frame, c) for c in candidates]
    dists.sort()
    return dists[0][2], dists[0][1]


def analyze_rally(
    rally: RallyData,
    cfg: ContactDetectionConfig,
    classifier: object | None,
) -> list[RefinementEffect]:
    """Analyze refinement effect for every GT contact in a rally."""
    results: list[RefinementEffect] = []

    trajectory_candidates, ball_by_frame, player_positions = (
        _generate_trajectory_candidates(rally, cfg)
    )
    if not trajectory_candidates:
        for gt in rally.gt_labels:
            results.append(RefinementEffect(
                rally_id=rally.rally_id, gt_frame=gt.frame, gt_action=gt.action,
                nearest_before=None, offset_before=None,
                nearest_after=None, offset_after=None,
                moved_closer=None, eaten_by_dedup=False,
                score_before=None, score_after=None,
            ))
        return results

    # Add proximity candidates (same as main pipeline)
    candidate_set = set(trajectory_candidates)
    if player_positions and cfg.enable_proximity_candidates:
        for frame in list(trajectory_candidates):
            prox = _find_proximity_frame(
                frame, ball_by_frame, player_positions,
                search_window=cfg.proximity_search_window,
                player_search_frames=cfg.player_search_frames,
                max_distance=cfg.player_contact_radius,
            )
            if prox is not None and prox != frame and prox not in candidate_set:
                candidate_set.add(prox)

    candidates_no_refine = sorted(candidate_set)

    # Apply refinement to trajectory candidates only (not proximity)
    first_frame = min(f for bp in rally.ball_positions_json
                      if (f := bp["frameNumber"]) and (bp.get("x", 0) > 0 or bp.get("y", 0) > 0))
    refined_trajectory = _refine_candidates_to_trajectory_peak(
        trajectory_candidates, ball_by_frame,
        direction_check_frames=cfg.direction_check_frames,
        search_window=cfg.trajectory_refinement_window,
        first_frame=first_frame,
        serve_window_frames=cfg.serve_window_frames,
    )

    # What refinement shifted (no dedup now — function returns all refined)
    refined_no_dedup: list[int] = []
    for frame in trajectory_candidates:
        best_frame = frame
        best_dc = compute_direction_change(ball_by_frame, frame, cfg.direction_check_frames)
        for offset in range(-cfg.trajectory_refinement_window, cfg.trajectory_refinement_window + 1):
            if offset == 0:
                continue
            f = frame + offset
            if f not in ball_by_frame:
                continue
            dc = compute_direction_change(ball_by_frame, f, cfg.direction_check_frames)
            if dc > best_dc:
                best_dc = dc
                best_frame = f
        refined_no_dedup.append(best_frame)

    dedup_victims = set(refined_no_dedup) - set(refined_trajectory)

    # Build candidate set with refinement: refined trajectory + original proximity
    proximity_only = set(candidates_no_refine) - set(trajectory_candidates)
    candidates_with_refine = sorted(set(refined_trajectory) | proximity_only)

    # Get classifier scores if available
    # We extract features at both original and refined frames for comparison
    features_no_refine, frames_no_refine = extract_candidate_features(rally, config=cfg)
    score_by_frame_no_refine: dict[int, float] = {}
    if classifier is not None and features_no_refine:
        preds = classifier.predict(features_no_refine)
        for feat, (_, score) in zip(features_no_refine, preds):
            score_by_frame_no_refine[feat.frame] = score

    # For refined features, we need to build a config that triggers refinement
    # but extract_candidate_features doesn't support that yet. Instead, compute
    # direction_change at refined frames as a proxy for "feature quality"

    for gt in rally.gt_labels:
        near_before, off_before = _nearest_candidate(gt.frame, candidates_no_refine)
        near_after, off_after = _nearest_candidate(gt.frame, candidates_with_refine)

        # Did refinement move closer or farther?
        moved_closer: bool | None = None
        if off_before is not None and off_after is not None:
            if abs(off_after) < abs(off_before):
                moved_closer = True
            elif abs(off_after) > abs(off_before):
                moved_closer = False
            # else: same distance, None

        # Was a candidate eaten by dedup?
        eaten = False
        if near_before is not None and near_after is None:
            eaten = True
        elif near_before is not None and near_after is not None:
            # Check if the nearest-before was in dedup victims
            if near_before in dedup_victims:
                eaten = True

        score_before = score_by_frame_no_refine.get(near_before) if near_before else None
        # Score at refined frame from the un-refined classifier (shows distribution shift)
        score_after = score_by_frame_no_refine.get(near_after) if near_after else None

        results.append(RefinementEffect(
            rally_id=rally.rally_id, gt_frame=gt.frame, gt_action=gt.action,
            nearest_before=near_before, offset_before=off_before,
            nearest_after=near_after, offset_after=off_after,
            moved_closer=moved_closer, eaten_by_dedup=eaten,
            score_before=score_before, score_after=score_after,
        ))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose trajectory refinement effect")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    if args.rally:
        rallies = [r for r in rallies if r.rally_id.startswith(args.rally)]

    if not rallies:
        console.print("[red]No rallies found.[/red]")
        return

    console.print(f"[bold]Analyzing refinement effect on {len(rallies)} rallies[/bold]\n")

    cfg = ContactDetectionConfig()
    classifier = load_contact_classifier()

    all_effects: list[RefinementEffect] = []
    for i, rally in enumerate(rallies):
        effects = analyze_rally(rally, cfg, classifier)
        all_effects.extend(effects)
        if (i + 1) % 50 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] rallies analyzed")

    console.print(f"\nTotal GT contacts: {len(all_effects)}")

    # ── Summary: within-tolerance analysis ──
    # Focus on GT contacts that are FN in current pipeline (no candidate within tolerance)
    fn_current = [e for e in all_effects if e.offset_before is None or abs(e.offset_before) > GT_TOLERANCE]
    tp_current = [e for e in all_effects if e.offset_before is not None and abs(e.offset_before) <= GT_TOLERANCE]

    console.print(f"Current TP (candidate within ±{GT_TOLERANCE}f): {len(tp_current)}")
    console.print(f"Current FN (no candidate within ±{GT_TOLERANCE}f): {len(fn_current)}")

    # ── How refinement affects FNs ──
    fn_rescued = [e for e in fn_current if e.offset_after is not None and abs(e.offset_after) <= GT_TOLERANCE]
    console.print(f"\n[bold]FNs rescued by refinement[/bold] (no candidate → candidate within ±{GT_TOLERANCE}f): {len(fn_rescued)}")
    if fn_rescued:
        action_counts = Counter(e.gt_action for e in fn_rescued)
        console.print(f"  By action: {dict(action_counts)}")

    # ── How refinement affects TPs ──
    tp_killed = [e for e in tp_current if e.offset_after is None or abs(e.offset_after) > GT_TOLERANCE]
    console.print(f"\n[bold red]TPs killed by refinement[/bold red] (candidate within ±{GT_TOLERANCE}f → lost): {len(tp_killed)}")
    if tp_killed:
        action_counts = Counter(e.gt_action for e in tp_killed)
        console.print(f"  By action: {dict(action_counts)}")
        eaten_count = sum(1 for e in tp_killed if e.eaten_by_dedup)
        console.print(f"  Eaten by dedup: {eaten_count}")

    # ── Moved closer / farther ──
    closer = [e for e in all_effects if e.moved_closer is True]
    farther = [e for e in all_effects if e.moved_closer is False]
    unchanged = [e for e in all_effects if e.moved_closer is None and e.offset_before is not None]
    console.print(f"\n[bold]Direction change:[/bold]")
    console.print(f"  Moved closer to GT: {len(closer)}")
    console.print(f"  Moved farther from GT: {len(farther)}")
    console.print(f"  Unchanged distance: {len(unchanged)}")

    # ── Dedup victims ──
    eaten = [e for e in all_effects if e.eaten_by_dedup]
    console.print(f"\n[bold]Dedup victims:[/bold] {len(eaten)} GT contacts had their nearest candidate eaten by refinement dedup")

    # ── Offset distribution ──
    offsets_before = [abs(e.offset_before) for e in all_effects if e.offset_before is not None]
    offsets_after = [abs(e.offset_after) for e in all_effects if e.offset_after is not None]
    if offsets_before:
        console.print(f"\n[bold]Offset distribution (|candidate - GT| in frames):[/bold]")
        console.print(f"  Before refinement: median={np.median(offsets_before):.0f}, mean={np.mean(offsets_before):.1f}, p90={np.percentile(offsets_before, 90):.0f}")
    if offsets_after:
        console.print(f"  After refinement:  median={np.median(offsets_after):.0f}, mean={np.mean(offsets_after):.1f}, p90={np.percentile(offsets_after, 90):.0f}")

    # ── Candidate count change ──
    console.print(f"\n[bold]Net effect on candidate-GT matching:[/bold]")
    console.print(f"  TP candidates (within ±{GT_TOLERANCE}f): {len(tp_current)} → {len(tp_current) - len(tp_killed) + len(fn_rescued)}")
    console.print(f"  Delta: {len(fn_rescued) - len(tp_killed):+d}")

    # ── Detail table: TPs killed ──
    if tp_killed:
        t = Table(title=f"\nTPs KILLED by refinement (top 20)")
        t.add_column("Rally", max_width=10)
        t.add_column("GT Frame", justify="right")
        t.add_column("Action")
        t.add_column("Before (offset)", justify="right")
        t.add_column("After (offset)", justify="right")
        t.add_column("Dedup?")
        t.add_column("Score@before", justify="right")

        for e in tp_killed[:20]:
            off_b = f"{e.nearest_before} ({e.offset_before:+d})" if e.nearest_before else "—"
            off_a = f"{e.nearest_after} ({e.offset_after:+d})" if e.nearest_after else "—"
            score = f"{e.score_before:.3f}" if e.score_before is not None else "—"
            t.add_row(e.rally_id[:8], str(e.gt_frame), e.gt_action, off_b, off_a, "Y" if e.eaten_by_dedup else "", score)
        console.print(t)

    # ── Detail table: FNs rescued ──
    if fn_rescued:
        t = Table(title=f"\nFNs RESCUED by refinement (top 20)")
        t.add_column("Rally", max_width=10)
        t.add_column("GT Frame", justify="right")
        t.add_column("Action")
        t.add_column("Before (offset)", justify="right")
        t.add_column("After (offset)", justify="right")
        t.add_column("Score@after", justify="right")

        for e in fn_rescued[:20]:
            off_b = f"{e.nearest_before} ({e.offset_before:+d})" if e.nearest_before else "—"
            off_a = f"{e.nearest_after} ({e.offset_after:+d})" if e.nearest_after else "—"
            score = f"{e.score_after:.3f}" if e.score_after is not None else "—"
            t.add_row(e.rally_id[:8], str(e.gt_frame), e.gt_action, off_b, off_a, score)
        console.print(t)

    # ── Detail table: Moved farther (top 20) ──
    farther_sorted = sorted(farther, key=lambda e: abs(e.offset_after or 0) - abs(e.offset_before or 0), reverse=True)
    if farther_sorted:
        t = Table(title=f"\nMoved FARTHER from GT (top 20 worst)")
        t.add_column("Rally", max_width=10)
        t.add_column("GT Frame", justify="right")
        t.add_column("Action")
        t.add_column("Before (offset)", justify="right")
        t.add_column("After (offset)", justify="right")
        t.add_column("Delta", justify="right")

        for e in farther_sorted[:20]:
            off_b = f"{e.nearest_before} ({e.offset_before:+d})" if e.nearest_before else "—"
            off_a = f"{e.nearest_after} ({e.offset_after:+d})" if e.nearest_after else "—"
            delta = abs(e.offset_after or 0) - abs(e.offset_before or 0)
            t.add_row(e.rally_id[:8], str(e.gt_frame), e.gt_action, off_b, off_a, f"+{delta}")
        console.print(t)


if __name__ == "__main__":
    main()
