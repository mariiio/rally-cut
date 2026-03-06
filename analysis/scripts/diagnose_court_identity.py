"""Diagnose court identity IDsw failures.

Re-tracks each labeled rally with skip_court_identity=True, then manually
runs resolve_court_identity() to compare raw vs resolved IDsw. Classifies
each IDsw by failure mode and analyzes per-component scoring.

Usage:
    uv run python scripts/diagnose_court_identity.py
    uv run python scripts/diagnose_court_identity.py --rally <rally-id>
    uv run python scripts/diagnose_court_identity.py --stride 2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

from rallycut.cli.commands.compare_tracking import _match_detections
from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.court_identity import (
    SwapDecision,
    resolve_court_identity,
)
from rallycut.tracking.player_tracker import PlayerPosition, PlayerTracker, PlayerTrackingResult

logger = logging.getLogger(__name__)

PROXIMITY_FRAMES = 15
CONFIDENCE_MARGIN = 0.15  # Threshold for confident vs uncertain


@dataclass
class IDSwitch:
    """A single ID switch event with GT/pred details."""

    frame: int
    gt_id: int
    old_pred_id: int
    new_pred_id: int
    duration: int = 0  # How many frames the new mapping persists before next switch


@dataclass
class FailureClassification:
    """Classification of an IDsw failure."""

    idsw: IDSwitch
    mode: str  # no_interaction, wrong_keep_confident, wrong_keep_uncertain,
    #            wrong_swap_confident, wrong_swap_uncertain, tracker_fragmentation
    decision: SwapDecision | None = None  # Associated decision if near interaction


@dataclass
class RallyResult:
    """Diagnosis result for one rally."""

    rally_id: str
    idsw_raw: list[IDSwitch] = field(default_factory=list)
    idsw_ci: list[IDSwitch] = field(default_factory=list)
    decisions: list[SwapDecision] = field(default_factory=list)
    failures: list[FailureClassification] = field(default_factory=list)
    fixed: list[IDSwitch] = field(default_factory=list)
    introduced: list[IDSwitch] = field(default_factory=list)
    remaining: list[IDSwitch] = field(default_factory=list)
    color_store: ColorHistogramStore | None = None


def _create_calibrator(
    corners_json: list[dict[str, float]] | None,
) -> CourtCalibrator | None:
    if not corners_json or len(corners_json) != 4:
        return None
    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in corners_json]
    calibrator.calibrate(image_corners)
    return calibrator


def _extract_idsw_events(
    rally: TrackingEvaluationRally,
    predictions: PlayerTrackingResult,
) -> list[IDSwitch]:
    """Extract detailed IDsw events with GT/pred track ID info."""
    gt = rally.ground_truth
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for p in gt.player_positions:
        gt_by_frame.setdefault(p.frame_number, []).append(
            (p.track_id, p.x, p.y, p.width, p.height)
        )

    pred_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    for pp in predictions.positions:
        pred_by_frame.setdefault(pp.frame_number, []).append(
            (pp.track_id, pp.x, pp.y, pp.width, pp.height)
        )

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))
    last_pred_id: dict[int, int] = {}
    events: list[IDSwitch] = []

    for frame in all_frames:
        gt_boxes = gt_by_frame.get(frame, [])
        pred_boxes = pred_by_frame.get(frame, [])
        if not gt_boxes or not pred_boxes:
            continue

        matches, _, _ = _match_detections(gt_boxes, pred_boxes, iou_threshold=0.5)
        for gt_id, pred_id in matches:
            if gt_id in last_pred_id and last_pred_id[gt_id] != pred_id:
                events.append(IDSwitch(
                    frame=frame,
                    gt_id=gt_id,
                    old_pred_id=last_pred_id[gt_id],
                    new_pred_id=pred_id,
                ))
            last_pred_id[gt_id] = pred_id

    # Compute duration: frames until next IDsw for same gt_id (or end of rally)
    max_frame = max(all_frames) if all_frames else 0
    gt_events: dict[int, list[IDSwitch]] = {}
    for e in events:
        gt_events.setdefault(e.gt_id, []).append(e)
    for gt_id, gt_list in gt_events.items():
        for i, e in enumerate(gt_list):
            if i + 1 < len(gt_list):
                e.duration = gt_list[i + 1].frame - e.frame
            else:
                e.duration = max_frame - e.frame

    return events


def _retrack_rally(
    rally: TrackingEvaluationRally,
    stride: int = 1,
) -> PlayerTrackingResult | None:
    """Re-track with skip_court_identity=True to get raw positions + color_store."""
    video_path = get_video_path(rally.video_id)
    if video_path is None:
        return None

    calibrator = _create_calibrator(rally.court_calibration_json)

    # Run ball tracking
    from rallycut.tracking.ball_tracker import create_ball_tracker

    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path, start_ms=rally.start_ms, end_ms=rally.end_ms,
    )

    tracker = PlayerTracker()
    result = tracker.track_video(
        video_path=video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
        stride=stride,
        filter_enabled=True,
        court_calibrator=calibrator,
        ball_positions=ball_result.positions,
        skip_court_identity=True,
    )

    if ball_result.positions:
        result.ball_positions = ball_result.positions

    return result


def _find_nearest_decision(
    frame: int,
    decisions: list[SwapDecision],
) -> SwapDecision | None:
    """Find the decision whose interaction window is closest to frame."""
    best: SwapDecision | None = None
    best_dist = PROXIMITY_FRAMES + 1
    for d in decisions:
        i = d.interaction
        if i.start_frame - PROXIMITY_FRAMES <= frame <= i.end_frame + PROXIMITY_FRAMES:
            # Distance = 0 if inside, else gap to nearest edge
            if i.start_frame <= frame <= i.end_frame:
                dist = 0
            else:
                dist = min(abs(frame - i.start_frame), abs(frame - i.end_frame))
            if dist < best_dist:
                best_dist = dist
                best = d
    return best


def _classify_idsw(
    idsw: IDSwitch,
    decisions: list[SwapDecision],
    is_in_raw: bool,
) -> FailureClassification:
    """Classify a single IDsw failure mode."""
    decision = _find_nearest_decision(idsw.frame, decisions)

    if decision is None:
        if is_in_raw:
            mode = "tracker_fragmentation"
        else:
            mode = "no_interaction"
        return FailureClassification(idsw=idsw, mode=mode)

    margin = decision.margin
    confident = margin >= CONFIDENCE_MARGIN

    if is_in_raw:
        # IDsw existed before CI and still exists — CI didn't fix it
        mode = "tracker_fragmentation"
    elif decision.should_swap:
        # CI swapped and this IDsw appeared (CI introduced it)
        mode = "wrong_swap_confident" if confident else "wrong_swap_uncertain"
    else:
        # CI kept and IDsw persists
        mode = "wrong_keep_confident" if confident else "wrong_keep_uncertain"

    return FailureClassification(idsw=idsw, mode=mode, decision=decision)


def analyze_rally(
    rally: TrackingEvaluationRally,
    stride: int = 1,
) -> RallyResult | None:
    """Full diagnosis for one rally."""
    # Step 1: Re-track with skip_court_identity=True
    result = _retrack_rally(rally, stride=stride)
    if result is None:
        return None

    # Step 2: Evaluate raw positions (without CI)
    idsw_raw = _extract_idsw_events(rally, result)

    # Step 3: Copy positions, run CI manually
    positions_copy = [
        PlayerPosition(
            frame_number=p.frame_number,
            track_id=p.track_id,
            x=p.x, y=p.y, width=p.width, height=p.height,
            confidence=p.confidence,
        )
        for p in result.positions
    ]

    calibrator = _create_calibrator(rally.court_calibration_json)
    decisions: list[SwapDecision] = []

    if (
        calibrator is not None
        and calibrator.is_calibrated
        and result.team_assignments
    ):
        cs = result.color_store
        if cs is not None and cs.has_data():
            cs_tids = cs.track_ids()
            pos_tids = {p.track_id for p in positions_copy}
            missing = pos_tids - cs_tids
            total_hists = sum(
                len(cs.get_track_histograms(tid)) for tid in cs_tids
            )
            logger.debug(
                f"Color store: {len(cs_tids)} tracks, "
                f"{total_hists} histograms, "
                f"position tracks: {sorted(pos_tids)}, "
                f"color tracks: {sorted(cs_tids)}, "
                f"missing: {sorted(missing) if missing else 'none'}"
            )
        else:
            logger.debug("Color store empty or None")

        positions_copy, _num_swaps, decisions = resolve_court_identity(
            positions_copy,
            dict(result.team_assignments),
            calibrator,
            video_width=rally.video_width or 1920,
            video_height=rally.video_height or 1080,
            color_store=result.color_store,
        )

    # Step 4: Evaluate resolved positions
    resolved_result = PlayerTrackingResult(
        positions=positions_copy,
        frame_count=result.frame_count,
        video_fps=result.video_fps,
        video_width=result.video_width,
        video_height=result.video_height,
        ball_positions=result.ball_positions,
        team_assignments=result.team_assignments,
    )
    idsw_ci = _extract_idsw_events(rally, resolved_result)

    # Step 5: Classify fixed/introduced/remaining
    raw_set = {(e.frame, e.gt_id) for e in idsw_raw}
    ci_set = {(e.frame, e.gt_id) for e in idsw_ci}

    fixed_keys = raw_set - ci_set
    introduced_keys = ci_set - raw_set
    remaining_keys = raw_set & ci_set

    fixed = [e for e in idsw_raw if (e.frame, e.gt_id) in fixed_keys]
    introduced = [e for e in idsw_ci if (e.frame, e.gt_id) in introduced_keys]
    remaining = [e for e in idsw_ci if (e.frame, e.gt_id) in remaining_keys]

    # Step 6: Classify each CI IDsw
    failures: list[FailureClassification] = []
    for e in idsw_ci:
        is_in_raw = (e.frame, e.gt_id) in raw_set
        failures.append(_classify_idsw(e, decisions, is_in_raw))

    return RallyResult(
        rally_id=rally.rally_id,
        idsw_raw=idsw_raw,
        idsw_ci=idsw_ci,
        decisions=decisions,
        failures=failures,
        fixed=fixed,
        introduced=introduced,
        remaining=remaining,
        color_store=result.color_store,
    )


def _print_decision_detail(
    d: SwapDecision,
    failures: list[FailureClassification],
    color_store: ColorHistogramStore | None = None,
) -> None:
    """Print per-component scores for one interaction."""
    i = d.interaction
    ns = d.no_swap_hypothesis
    sw = d.swap_hypothesis

    action = "SWAP" if d.should_swap else "KEEP"
    conf = "confident" if d.confident else "uncertain"
    print(f"  Tracks {i.track_a}<->{i.track_b} f{i.start_frame}-{i.end_frame}", end="")

    # Show color histogram counts around the interaction
    if color_store is not None:
        pre_window, post_window = 30, 30
        for tid in [i.track_a, i.track_b]:
            hists = color_store.get_track_histograms(tid)
            pre = sum(1 for fn, _ in hists if i.start_frame - pre_window <= fn < i.start_frame)
            post = sum(1 for fn, _ in hists if i.end_frame < fn <= i.end_frame + post_window)
            print(f"  [t{tid}: {pre}pre/{post}post]", end="")
    print()

    if ns and sw:
        print(
            f"    side={ns.side_of_net_score:.2f}/{sw.side_of_net_score:.2f}  "
            f"motion={ns.motion_smoothness_score:.2f}/{sw.motion_smoothness_score:.2f}  "
            f"bbox={ns.bbox_size_score:.2f}/{sw.bbox_size_score:.2f}  "
            f"color={ns.color_score:.2f}/{sw.color_score:.2f}"
        )
        if ns.grammar_score != 0.0 or sw.grammar_score != 0.0:
            print(
                f"    grammar={ns.grammar_score:.2f}/{sw.grammar_score:.2f}  "
                f"serve={ns.serve_anchor_score:.2f}/{sw.serve_anchor_score:.2f}"
            )

    print(
        f"    Total: no_swap={d.no_swap_score:.3f}  swap={d.swap_score:.3f}  "
        f"-> {action} ({conf}, margin={d.margin:.3f})"
    )

    # Show associated IDsw
    related = [
        f for f in failures
        if f.decision is not None
        and f.decision.interaction == d.interaction
    ]
    if related:
        for f in related:
            e = f.idsw
            tag = "PERSISTENT" if e.duration > 10 else "transient"
            print(
                f"    IDsw f{e.frame}: gt{e.gt_id} pred {e.old_pred_id}->{e.new_pred_id}  "
                f"dur={e.duration}f [{tag}] [{f.mode}]"
            )
    else:
        print("    (no IDsw in window)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose court identity IDsw failures"
    )
    parser.add_argument("--rally", nargs="+", help="Specific rally ID(s)")
    parser.add_argument("--worst", action="store_true",
                        help="Run only known high-IDsw rallies: fad29c31, 87ce7bff, 0d84f858")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.player_tracker").setLevel(logging.WARNING)
    logging.getLogger("rallycut.tracking.player_filter").setLevel(logging.WARNING)

    WORST_RALLIES = ["fad29c31", "87ce7bff", "0d84f858"]

    all_rallies = load_labeled_rallies()

    if args.worst:
        prefixes = WORST_RALLIES
    elif args.rally:
        prefixes = args.rally
    else:
        prefixes = None

    if prefixes:
        rallies = [
            r for r in all_rallies
            if any(r.rally_id.startswith(p) for p in prefixes)
        ]
    else:
        rallies = all_rallies

    if not rallies:
        print("No labeled rallies found")
        sys.exit(1)

    print(f"Loaded {len(rallies)} labeled rally(s), stride={args.stride}\n")

    results: list[RallyResult] = []
    total_raw = 0
    total_ci = 0
    total_fixed = 0
    total_introduced = 0
    total_interactions = 0

    for rally in rallies:
        rally_short = rally.rally_id[:12]
        print(f"Processing {rally_short}...", end=" ", flush=True)
        t0 = time.time()

        rr = analyze_rally(rally, stride=args.stride)
        elapsed = time.time() - t0

        if rr is None:
            print("FAILED (no video)")
            continue

        results.append(rr)
        total_raw += len(rr.idsw_raw)
        total_ci += len(rr.idsw_ci)
        total_fixed += len(rr.fixed)
        total_introduced += len(rr.introduced)
        total_interactions += len(rr.decisions)

        print(
            f"IDsw raw={len(rr.idsw_raw)} CI={len(rr.idsw_ci)} "
            f"fixed={len(rr.fixed)} intro={len(rr.introduced)} "
            f"interactions={len(rr.decisions)} [{elapsed:.1f}s]"
        )

    if not results:
        print("No results")
        sys.exit(1)

    # ===== Table 1: Per-rally summary =====
    print(f"\n{'='*80}")
    print("TABLE 1: Per-rally summary")
    print(f"{'='*80}")
    print(
        f"{'Rally':<14} {'IDsw(raw)':>9} {'IDsw(CI)':>9} {'Fixed':>6} "
        f"{'Intro':>6} {'Interactions':>12}"
    )
    print("-" * 62)
    for rr in results:
        if len(rr.idsw_raw) == 0 and len(rr.idsw_ci) == 0:
            continue
        print(
            f"{rr.rally_id[:12]:<14} {len(rr.idsw_raw):>9} {len(rr.idsw_ci):>9} "
            f"{len(rr.fixed):>6} {len(rr.introduced):>6} {len(rr.decisions):>12}"
        )
    print("-" * 62)
    print(
        f"{'TOTAL':<14} {total_raw:>9} {total_ci:>9} "
        f"{total_fixed:>6} {total_introduced:>6} {total_interactions:>12}"
    )

    # ===== Table 2: Per-interaction score breakdown (rallies with IDsw) =====
    print(f"\n{'='*80}")
    print("TABLE 2: Per-interaction score breakdown (rallies with IDsw)")
    print(f"{'='*80}")
    for rr in results:
        if len(rr.idsw_ci) == 0:
            continue
        print(f"\nRally {rr.rally_id[:12]}:")
        for d in rr.decisions:
            _print_decision_detail(d, rr.failures, rr.color_store)

    # ===== Table 3: Failure mode summary =====
    print(f"\n{'='*80}")
    print("TABLE 3: Failure mode summary")
    print(f"{'='*80}")
    mode_counts: dict[str, int] = {}
    for rr in results:
        for f in rr.failures:
            mode_counts[f.mode] = mode_counts.get(f.mode, 0) + 1

    total_failures = sum(mode_counts.values())
    mode_order = [
        "no_interaction",
        "wrong_keep_confident",
        "wrong_keep_uncertain",
        "wrong_swap_confident",
        "wrong_swap_uncertain",
        "tracker_fragmentation",
    ]
    for mode in mode_order:
        count = mode_counts.get(mode, 0)
        pct = 100.0 * count / total_failures if total_failures > 0 else 0
        print(f"  {mode:<26} {count:>4}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<26} {total_failures:>4}")

    # ===== Table 4: Color opportunity =====
    print(f"\n{'='*80}")
    print("TABLE 4: Color signal analysis")
    print(f"{'='*80}")
    total_ambig = 0
    total_ambig_idsw = 0
    all_failures: list[FailureClassification] = []
    for rr in results:
        all_failures.extend(rr.failures)

    for rr in results:
        for d in rr.decisions:
            ns = d.no_swap_hypothesis
            sw = d.swap_hypothesis
            if ns and sw:
                delta = abs(ns.color_score - sw.color_score)
                if delta < 0.10:
                    total_ambig += 1
                    has_idsw = any(
                        f.decision is not None
                        and f.decision.interaction == d.interaction
                        for f in rr.failures
                    )
                    if has_idsw:
                        total_ambig_idsw += 1

    # Count theoretically fixable: wrong_keep + wrong_swap (not tracker_frag or no_interaction)
    fixable = sum(
        1 for f in all_failures
        if f.mode in (
            "wrong_keep_confident", "wrong_keep_uncertain",
            "wrong_swap_confident", "wrong_swap_uncertain",
        )
    )

    print(f"  Ambiguous color interactions: {total_ambig}/{total_interactions} "
          f"({100.0 * total_ambig / total_interactions if total_interactions else 0:.1f}%)")
    print(f"    With IDsw: {total_ambig_idsw}/{total_ambig if total_ambig else 1}")
    print(f"  IDsw theoretically fixable by better scoring: {fixable}/{total_ci}")
    print(f"  IDsw from tracker fragmentation: "
          f"{mode_counts.get('tracker_fragmentation', 0)}/{total_ci}")
    print(f"  IDsw not near any interaction: "
          f"{mode_counts.get('no_interaction', 0)}/{total_ci}")

    # ===== Table 5: Transient vs persistent IDsw =====
    PERSISTENT_THRESHOLD = 10  # frames
    print(f"\n{'='*80}")
    print(f"TABLE 5: Transient vs persistent IDsw (threshold={PERSISTENT_THRESHOLD} frames)")
    print(f"{'='*80}")
    all_ci_idsw: list[IDSwitch] = []
    for rr in results:
        all_ci_idsw.extend(rr.idsw_ci)

    transient = [e for e in all_ci_idsw if e.duration <= PERSISTENT_THRESHOLD]
    persistent = [e for e in all_ci_idsw if e.duration > PERSISTENT_THRESHOLD]
    print(f"  Transient (≤{PERSISTENT_THRESHOLD}f, self-correcting): "
          f"{len(transient)}/{len(all_ci_idsw)}")
    print(f"  Persistent (>{PERSISTENT_THRESHOLD}f, real swap):       "
          f"{len(persistent)}/{len(all_ci_idsw)}")
    if persistent:
        durations = [e.duration for e in persistent]
        print(f"    Duration: median={sorted(durations)[len(durations)//2]}f, "
              f"max={max(durations)}f")
        print(f"    These are the IDsw that court identity could theoretically fix.")
    if transient:
        durations = [e.duration for e in transient]
        print(f"    Transient durations: {sorted(durations)}")

    # Per-rally breakdown for persistent
    for rr in results:
        p_count = sum(1 for e in rr.idsw_ci if e.duration > PERSISTENT_THRESHOLD)
        t_count = sum(1 for e in rr.idsw_ci if e.duration <= PERSISTENT_THRESHOLD)
        if rr.idsw_ci:
            print(f"  {rr.rally_id[:12]}: {p_count} persistent, {t_count} transient")
            for e in rr.idsw_ci:
                tag = "PERSISTENT" if e.duration > PERSISTENT_THRESHOLD else "transient"
                print(f"    f{e.frame}: gt{e.gt_id} {e.old_pred_id}->{e.new_pred_id} "
                      f"dur={e.duration}f [{tag}]")


if __name__ == "__main__":
    main()
