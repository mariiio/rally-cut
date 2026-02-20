"""Evaluate court-plane identity resolution on labeled rallies.

Loads stored tracking predictions from DB, applies court identity
resolution as a post-hoc step, and compares before/after against GT.

Usage:
    uv run python scripts/eval_court_identity.py
    uv run python scripts/eval_court_identity.py --rally <rally-id>
"""

from __future__ import annotations

import argparse
import logging
import sys

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.evaluation.tracking.metrics import (
    TrackingEvaluationResult,
    evaluate_rally,
)
from rallycut.tracking.court_identity import CourtIdentityConfig, resolve_court_identity
from rallycut.tracking.player_filter import (
    PlayerFilterConfig,
    classify_teams,
    compute_court_split,
)
from rallycut.tracking.player_tracker import PlayerPosition, PlayerTrackingResult

logger = logging.getLogger(__name__)


def _create_calibrator(
    corners_json: list[dict[str, float]],
) -> CourtCalibrator | None:
    """Create a CourtCalibrator from DB corner JSON."""
    if not corners_json or len(corners_json) != 4:
        return None

    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in corners_json]
    calibrator.calibrate(image_corners)
    return calibrator


def _compute_team_assignments(
    positions: list[PlayerPosition],
    ball_positions: list[object] | None = None,
) -> dict[int, int]:
    """Compute team assignments from Y positions."""
    config = PlayerFilterConfig()
    split_y = compute_court_split(
        ball_positions or [], config, player_positions=positions
    )
    if split_y is None:
        return {}
    return classify_teams(positions, split_y)


def _apply_court_identity(
    positions: list[PlayerPosition],
    calibrator: CourtCalibrator,
    video_width: int,
    video_height: int,
    ball_positions: list[object] | None = None,
    config: CourtIdentityConfig | None = None,
) -> tuple[list[PlayerPosition], int, list[object]]:
    """Apply court identity resolution to existing positions."""
    team_assignments = _compute_team_assignments(positions, ball_positions)
    if not team_assignments:
        return positions, 0, []

    return resolve_court_identity(
        positions,
        team_assignments,
        calibrator,
        video_width=video_width,
        video_height=video_height,
        config=config,
    )


def _format_result(result: TrackingEvaluationResult) -> dict[str, float | int]:
    """Extract key metrics from evaluation result."""
    agg = result.aggregate
    hota = result.hota_metrics
    return {
        "HOTA": hota.hota * 100 if hota else 0.0,
        "DetA": hota.deta * 100 if hota else 0.0,
        "AssA": hota.assa * 100 if hota else 0.0,
        "MOTA": agg.mota * 100,
        "F1": agg.f1 * 100,
        "IDsw": agg.num_id_switches,
        "Frag": agg.num_fragmentations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate court identity resolution")
    parser.add_argument("--rally", help="Specific rally ID")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--margin", type=float, default=0.1, help="Confidence margin")
    args = parser.parse_args()

    config = CourtIdentityConfig(confidence_margin=args.margin)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # Load labeled rallies
    rallies = load_labeled_rallies(rally_id=args.rally)
    if not rallies:
        print("No labeled rallies found")
        sys.exit(1)

    print(f"Loaded {len(rallies)} labeled rally(s)\n")

    # Track aggregated results
    total_baseline_idsw = 0
    total_court_idsw = 0
    total_baseline_hota = 0.0
    total_court_hota = 0.0
    total_interactions = 0
    total_swaps = 0
    rallies_with_calibration = 0

    header = (
        f"{'Rally':<14} {'Calib':>5} "
        f"{'Base HOTA':>9} {'Court HOTA':>10} {'Δ HOTA':>7} "
        f"{'Base IDsw':>9} {'Court IDsw':>10} {'Δ IDsw':>7} "
        f"{'Interactions':>12} {'Swaps':>6}"
    )
    print(header)
    print("-" * len(header))

    for rally in rallies:
        rally_short = rally.rally_id[:12]
        predictions = rally.predictions
        if predictions is None:
            print(f"{rally_short:<14} {'N/A':>5} (no predictions)")
            continue

        # Baseline evaluation (stored predictions)
        baseline_result = evaluate_rally(
            rally.rally_id,
            rally.ground_truth,
            predictions,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )

        # Check if court calibration is available
        calibrator = None
        if rally.court_calibration_json:
            calibrator = _create_calibrator(rally.court_calibration_json)

        if calibrator is None or not calibrator.is_calibrated:
            # No calibration — court identity cannot run
            bm = _format_result(baseline_result)
            print(
                f"{rally_short:<14} {'No':>5} "
                f"{bm['HOTA']:>8.1f}% {'-':>10} {'-':>7} "
                f"{bm['IDsw']:>9} {'-':>10} {'-':>7} "
                f"{'-':>12} {'-':>6}"
            )
            total_baseline_idsw += baseline_result.aggregate.num_id_switches
            total_court_idsw += baseline_result.aggregate.num_id_switches
            hota_val = baseline_result.hota_metrics.hota if baseline_result.hota_metrics else 0.0
            total_baseline_hota += hota_val
            total_court_hota += hota_val
            continue

        rallies_with_calibration += 1

        # Apply court identity to stored predictions
        # Deep copy positions to avoid modifying original
        positions_copy = [
            PlayerPosition(
                frame_number=p.frame_number,
                track_id=p.track_id,
                x=p.x,
                y=p.y,
                width=p.width,
                height=p.height,
                confidence=p.confidence,
            )
            for p in predictions.positions
        ]

        ball_positions = predictions.ball_positions

        corrected_positions, num_swaps, decisions = _apply_court_identity(
            positions_copy,
            calibrator,
            rally.video_width,
            rally.video_height,
            ball_positions=ball_positions,
            config=config,
        )

        num_interactions = len(decisions)
        total_interactions += num_interactions
        total_swaps += num_swaps

        # Create corrected prediction result
        corrected_predictions = PlayerTrackingResult(
            positions=corrected_positions,
            frame_count=predictions.frame_count,
            video_fps=predictions.video_fps,
            ball_positions=predictions.ball_positions,
            primary_track_ids=predictions.primary_track_ids,
            court_split_y=predictions.court_split_y,
        )

        # Evaluate corrected predictions
        court_result = evaluate_rally(
            rally.rally_id,
            rally.ground_truth,
            corrected_predictions,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )

        bm = _format_result(baseline_result)
        cm = _format_result(court_result)
        delta_hota = cm["HOTA"] - bm["HOTA"]
        delta_idsw = cm["IDsw"] - bm["IDsw"]

        delta_hota_str = f"{delta_hota:+.1f}%"
        delta_idsw_str = f"{delta_idsw:+d}"

        print(
            f"{rally_short:<14} {'Yes':>5} "
            f"{bm['HOTA']:>8.1f}% {cm['HOTA']:>9.1f}% {delta_hota_str:>7} "
            f"{bm['IDsw']:>9} {cm['IDsw']:>9d} {delta_idsw_str:>7} "
            f"{num_interactions:>12} {num_swaps:>6}"
        )

        # Log decision details in verbose mode
        if args.verbose:
            for d in decisions:
                interaction = d.interaction
                print(
                    f"  interaction: tracks {interaction.track_a},{interaction.track_b} "
                    f"frames {interaction.start_frame}-{interaction.end_frame} "
                    f"swap={d.should_swap} confident={d.confident} "
                    f"score_gap={d.swap_score - d.no_swap_score:+.3f}"
                )

        total_baseline_idsw += baseline_result.aggregate.num_id_switches
        total_court_idsw += court_result.aggregate.num_id_switches
        base_hota = baseline_result.hota_metrics.hota if baseline_result.hota_metrics else 0.0
        court_hota = court_result.hota_metrics.hota if court_result.hota_metrics else 0.0
        total_baseline_hota += base_hota
        total_court_hota += court_hota

    # Summary
    n = len(rallies)
    print(f"\n{'='*80}")
    print(f"Summary ({n} rallies, {rallies_with_calibration} with calibration)")
    print(f"  Baseline: HOTA={total_baseline_hota/n*100:.1f}%, IDsw={total_baseline_idsw}")
    print(f"  Court ID: HOTA={total_court_hota/n*100:.1f}%, IDsw={total_court_idsw}")
    print(f"  Delta:    HOTA={((total_court_hota-total_baseline_hota)/n*100):+.1f}%, "
          f"IDsw={total_court_idsw-total_baseline_idsw:+d}")
    print(f"  Net interactions detected: {total_interactions}")
    print(f"  Swaps applied: {total_swaps}")


if __name__ == "__main__":
    main()
