"""Diagnose net interactions and their relationship to ID switches.

Loads labeled rallies, re-tracks each, and cross-references detected net
interactions with per-frame ID switches to determine what fraction of the
15 remaining IDsw are net-interaction-related.

Usage:
    uv run python scripts/diagnose_net_interactions.py
    uv run python scripts/diagnose_net_interactions.py --rally <rally-id>
    uv run python scripts/diagnose_net_interactions.py --stride 2  # faster for 60fps
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict

import cv2

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    get_video_path,
    load_labeled_rallies,
)
from rallycut.evaluation.tracking.metrics import evaluate_rally
from rallycut.tracking.color_repair import detect_convergence_periods
from rallycut.tracking.court_identity import (
    CourtIdentityConfig,
    CourtIdentityResolver,
)
from rallycut.tracking.player_filter import classify_teams, compute_court_split
from rallycut.tracking.player_tracker import PlayerTracker, PlayerTrackingResult

logger = logging.getLogger(__name__)

# Proximity window: IDsw within this many frames of a net interaction are
# considered "net-interaction-related"
PROXIMITY_FRAMES = 15


def _create_calibrator(
    corners_json: list[dict[str, float]] | None,
) -> CourtCalibrator | None:
    if not corners_json or len(corners_json) != 4:
        return None
    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in corners_json]
    calibrator.calibrate(image_corners)
    return calibrator


def _adjust_frame_numbers(
    result: PlayerTrackingResult,
    start_ms: int,
    video_path: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps:
        video_fps = 30.0
    cap.release()

    start_frame = int((start_ms / 1000.0) * video_fps)
    for pos in result.positions:
        pos.frame_number -= start_frame
    for pos in result.raw_positions:
        pos.frame_number -= start_frame
    if result.ball_positions:
        for bp in result.ball_positions:
            bp.frame_number -= start_frame


def _get_idsw_frames(
    rally: TrackingEvaluationRally,
    predictions: PlayerTrackingResult,
) -> list[int]:
    """Get frame numbers where ID switches occur."""
    result = evaluate_rally(
        rally.rally_id,
        rally.ground_truth,
        predictions,
        video_width=rally.video_width,
        video_height=rally.video_height,
    )
    agg = result.aggregate
    # Per-frame metrics have id_switches counts
    idsw_frames: list[int] = []
    for pf in result.per_frame:
        if pf.id_switches > 0:
            for _ in range(pf.id_switches):
                idsw_frames.append(pf.frame_number)
    return idsw_frames


def _retrack_rally(
    rally: TrackingEvaluationRally,
    stride: int = 1,
) -> PlayerTrackingResult | None:
    video_path = get_video_path(rally.video_id)
    if video_path is None:
        return None

    calibrator = _create_calibrator(rally.court_calibration_json)

    tracker = PlayerTracker()
    result = tracker.track_video(
        video_path=video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
        stride=stride,
        filter_enabled=True,
        court_calibrator=calibrator,
    )

    _adjust_frame_numbers(result, rally.start_ms, str(video_path))
    return result


def analyze_rally(
    rally: TrackingEvaluationRally,
    predictions: PlayerTrackingResult,
) -> dict:
    """Analyze net interactions and IDsw for a single rally."""
    calibrator = _create_calibrator(rally.court_calibration_json)
    has_calibration = calibrator is not None and calibrator.is_calibrated

    # Get IDsw frames
    idsw_frames = _get_idsw_frames(rally, predictions)
    total_idsw = len(idsw_frames)

    # Detect convergence periods (bbox overlap)
    convergence = detect_convergence_periods(predictions.positions)

    # Detect net interactions (court-space, if calibrated)
    net_interactions = []
    if has_calibration:
        config = CourtIdentityConfig(net_approach_distance=3.0)
        resolver = CourtIdentityResolver(
            calibrator,
            config,
            video_width=rally.video_width or 1920,
            video_height=rally.video_height or 1080,
        )

        # Build team assignments
        from rallycut.tracking.player_filter import PlayerFilterConfig

        split_result = compute_court_split(
            [], PlayerFilterConfig(), player_positions=predictions.positions
        )
        if split_result is not None:
            split_y = split_result[0]
            team_assignments = classify_teams(predictions.positions, split_y)
            court_tracks = resolver._build_court_tracks(predictions.positions)
            effective_net_y = resolver._estimate_effective_net_y(
                court_tracks, team_assignments
            )
            net_interactions = resolver._detect_net_interactions(
                court_tracks, team_assignments, effective_net_y
            )

    # Cross-reference: which IDsw fall within/near a net interaction?
    idsw_during = 0
    idsw_near = 0
    idsw_outside = 0

    for idsw_frame in idsw_frames:
        is_during = False
        is_near = False
        for ni in net_interactions:
            if ni.start_frame <= idsw_frame <= ni.end_frame:
                is_during = True
                break
            if (ni.start_frame - PROXIMITY_FRAMES <= idsw_frame
                    <= ni.end_frame + PROXIMITY_FRAMES):
                is_near = True

        if is_during:
            idsw_during += 1
        elif is_near:
            idsw_near += 1
        else:
            idsw_outside += 1

    # Also cross-reference with convergence periods
    idsw_during_convergence = 0
    for idsw_frame in idsw_frames:
        for cp in convergence:
            if (cp.start_frame - PROXIMITY_FRAMES <= idsw_frame
                    <= cp.end_frame + PROXIMITY_FRAMES):
                idsw_during_convergence += 1
                break

    return {
        "rally_id": rally.rally_id[:12],
        "has_calibration": has_calibration,
        "total_idsw": total_idsw,
        "net_interactions": len(net_interactions),
        "convergence_periods": len(convergence),
        "idsw_during_interaction": idsw_during,
        "idsw_near_interaction": idsw_near,
        "idsw_outside_interaction": idsw_outside,
        "idsw_during_convergence": idsw_during_convergence,
        "idsw_frames": idsw_frames,
        "interaction_details": [
            {
                "track_a": ni.track_a,
                "track_b": ni.track_b,
                "start": ni.start_frame,
                "end": ni.end_frame,
                "duration": ni.end_frame - ni.start_frame + 1,
            }
            for ni in net_interactions
        ],
        "convergence_details": [
            {
                "track_a": cp.track_a,
                "track_b": cp.track_b,
                "start": cp.start_frame,
                "end": cp.end_frame,
                "duration": cp.end_frame - cp.start_frame + 1,
            }
            for cp in convergence
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose net interactions and ID switches"
    )
    parser.add_argument("--rally", help="Specific rally ID")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)

    rallies = load_labeled_rallies(rally_id=args.rally)
    if not rallies:
        print("No labeled rallies found")
        sys.exit(1)

    print(f"Loaded {len(rallies)} labeled rally(s)\n")

    results = []
    total_idsw = 0
    total_during = 0
    total_near = 0
    total_outside = 0
    total_convergence = 0

    for rally in rallies:
        rally_short = rally.rally_id[:12]
        print(f"Processing {rally_short}...", end=" ", flush=True)

        t0 = time.time()
        predictions = _retrack_rally(rally, stride=args.stride)
        elapsed = time.time() - t0

        if predictions is None:
            print("FAILED (no video)")
            continue

        analysis = analyze_rally(rally, predictions)
        results.append(analysis)

        total_idsw += analysis["total_idsw"]
        total_during += analysis["idsw_during_interaction"]
        total_near += analysis["idsw_near_interaction"]
        total_outside += analysis["idsw_outside_interaction"]
        total_convergence += analysis["idsw_during_convergence"]

        print(
            f"IDsw={analysis['total_idsw']} "
            f"(during={analysis['idsw_during_interaction']}, "
            f"near={analysis['idsw_near_interaction']}, "
            f"outside={analysis['idsw_outside_interaction']}) "
            f"interactions={analysis['net_interactions']} "
            f"convergence={analysis['convergence_periods']} "
            f"[{elapsed:.1f}s]"
        )

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    header = (
        f"{'Rally':<14} {'Calib':>5} {'IDsw':>5} "
        f"{'During':>7} {'Near':>5} {'Outside':>7} "
        f"{'NetInt':>6} {'Conv':>5}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        calib = "Yes" if r["has_calibration"] else "No"
        print(
            f"{r['rally_id']:<14} {calib:>5} {r['total_idsw']:>5} "
            f"{r['idsw_during_interaction']:>7} {r['idsw_near_interaction']:>5} "
            f"{r['idsw_outside_interaction']:>7} "
            f"{r['net_interactions']:>6} {r['convergence_periods']:>5}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<14} {'':>5} {total_idsw:>5} "
        f"{total_during:>7} {total_near:>5} {total_outside:>7}"
    )

    if total_idsw > 0:
        pct_interaction = (total_during + total_near) / total_idsw * 100
        pct_convergence = total_convergence / total_idsw * 100
        print(
            f"\n{total_during + total_near}/{total_idsw} "
            f"({pct_interaction:.0f}%) IDsw during/near net interactions"
        )
        print(
            f"{total_convergence}/{total_idsw} "
            f"({pct_convergence:.0f}%) IDsw during/near convergence periods"
        )

    # Detailed interaction info
    print(f"\n{'='*80}")
    print("INTERACTION DETAILS")
    print(f"{'='*80}")

    for r in results:
        if not r["interaction_details"] and not r["convergence_details"]:
            continue
        print(f"\n{r['rally_id']} (IDsw at frames: {r['idsw_frames']})")
        for ni in r["interaction_details"]:
            print(
                f"  Net interaction: tracks {ni['track_a']}<->{ni['track_b']} "
                f"f{ni['start']}-{ni['end']} ({ni['duration']}f)"
            )
        for cp in r["convergence_details"]:
            print(
                f"  Convergence:     tracks {cp['track_a']}<->{cp['track_b']} "
                f"f{cp['start']}-{cp['end']} ({cp['duration']}f)"
            )


if __name__ == "__main__":
    main()
