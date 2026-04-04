"""Diagnose the player tracking post-processing pipeline.

Answers 4 questions:
1. Oracle ceiling: best possible HOTA with perfect fragment→player assignment
2. Per-stage impact: which pipeline stages help vs hurt
3. Root cause patterns: what fails in bad rallies
4. Failure mode distribution for pipeline redesign

Usage:
    cd analysis && uv run python scripts/diagnose_postprocessing_pipeline.py
"""

from __future__ import annotations

import copy
import hashlib
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RallyData:
    rally: Any  # TrackingEvaluationRally
    cached_positions: list[Any]  # list[PlayerPosition]
    ball_positions: list[Any] | None
    color_store: Any  # ColorHistogramStore
    appearance_store: Any  # AppearanceDescriptorStore
    video_fps: float
    video_width: int
    video_height: int
    frame_count: int
    court_calibrator: Any | None


@dataclass
class OracleResult:
    rally_id: str
    num_raw_fragments: int
    num_raw_positions: int
    oracle_hota: float
    oracle_deta: float
    oracle_assa: float
    oracle_f1: float
    # Per-GT-player: which raw fragments cover them
    gt_player_fragments: dict[int, list[FragmentInfo]] = field(default_factory=dict)
    # Per-GT-player: oracle coverage fraction
    gt_player_coverage: dict[int, float] = field(default_factory=dict)


@dataclass
class FragmentInfo:
    track_id: int
    start_frame: int
    end_frame: int
    num_frames: int
    coverage_of_gt: float  # fraction of GT frames this fragment covers


@dataclass
class StageResult:
    stage_name: str
    hota: float
    deta: float
    assa: float
    f1: float
    id_switches: int
    num_positions: int
    num_tracks: int


@dataclass
class RootCause:
    gt_track_id: int
    classification: str  # detection_gap, fragmentation, misassignment, over_filtering
    detail: str


@dataclass
class RallyDiagnostic:
    rally_id: str
    oracle: OracleResult
    stages: list[StageResult]
    root_causes: list[RootCause]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_tracker_config_hash() -> str:
    from rallycut.tracking.player_tracker import (
        DEFAULT_CONFIDENCE, DEFAULT_COURT_ROI, DEFAULT_IMGSZ,
        DEFAULT_IOU, DEFAULT_TRACKER, MODEL_NAME,
    )
    config_parts = [
        f"model={MODEL_NAME}", f"conf={DEFAULT_CONFIDENCE}",
        f"iou={DEFAULT_IOU}", f"imgsz={DEFAULT_IMGSZ}",
        f"tracker={DEFAULT_TRACKER}", f"roi={DEFAULT_COURT_ROI}",
    ]
    return hashlib.sha256("|".join(config_parts).encode()).hexdigest()[:16]


def _build_court_calibrator(cal_json: list[dict] | None) -> Any:
    if not cal_json or not isinstance(cal_json, list) or len(cal_json) != 4:
        return None
    try:
        from rallycut.court.calibration import CourtCalibrator
        cc = CourtCalibrator()
        cc.calibrate([(c["x"], c["y"]) for c in cal_json])
        return cc if cc.is_calibrated else None
    except Exception:
        return None


def _make_tracking_result(positions, frame_count, video_fps, video_width=0,
                          video_height=0, ball_positions=None):
    """Build a minimal PlayerTrackingResult for evaluation."""
    from rallycut.tracking.player_tracker import PlayerTrackingResult
    return PlayerTrackingResult(
        positions=positions,
        frame_count=frame_count,
        video_fps=video_fps,
        video_width=video_width,
        video_height=video_height,
        ball_positions=ball_positions or [],
    )


def _copy_positions(positions):
    """Shallow-copy position list (PlayerPosition is mutable)."""
    from rallycut.tracking.player_tracker import PlayerPosition
    return [
        PlayerPosition(
            frame_number=p.frame_number, track_id=p.track_id,
            x=p.x, y=p.y, width=p.width, height=p.height,
            confidence=p.confidence,
        )
        for p in positions
    ]


# ---------------------------------------------------------------------------
# Part 1: Oracle ceiling
# ---------------------------------------------------------------------------

def compute_oracle_ceiling(rd: RallyData) -> OracleResult:
    from rallycut.cli.commands.compare_tracking import _match_detections
    from rallycut.evaluation.tracking.metrics import evaluate_rally, smart_interpolate_gt
    from rallycut.tracking.player_tracker import PlayerPosition, PlayerTrackingResult

    gt = rd.rally.ground_truth

    # Build a dummy predictions result from raw positions for smart interpolation
    raw_result = _make_tracking_result(
        rd.cached_positions, rd.frame_count, rd.video_fps,
        rd.video_width, rd.video_height,
    )
    interp_gt = smart_interpolate_gt(gt, raw_result, rd.frame_count)

    # Group raw and GT by frame
    raw_by_frame: dict[int, list] = defaultdict(list)
    for p in rd.cached_positions:
        raw_by_frame[p.frame_number].append(p)

    gt_by_frame: dict[int, list] = defaultdict(list)
    for p in interp_gt.player_positions:
        gt_by_frame[p.frame_number].append(p)

    # Per-frame oracle matching
    oracle_positions: list[PlayerPosition] = []
    # Track which raw fragments map to which GT players
    frag_gt_matches: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    all_frames = sorted(set(raw_by_frame.keys()) | set(gt_by_frame.keys()))
    for frame in all_frames:
        gt_boxes = [
            (p.track_id, p.x, p.y, p.width, p.height)
            for p in gt_by_frame.get(frame, [])
        ]
        raw_boxes = [
            (p.track_id, p.x, p.y, p.width, p.height)
            for p in raw_by_frame.get(frame, [])
        ]
        if not gt_boxes or not raw_boxes:
            continue

        matches, _, _ = _match_detections(gt_boxes, raw_boxes, iou_threshold=0.3)
        raw_lookup = {p.track_id: p for p in raw_by_frame[frame]}

        for gt_id, raw_id in matches:
            raw_p = raw_lookup[raw_id]
            oracle_positions.append(PlayerPosition(
                frame_number=frame,
                track_id=gt_id,
                x=raw_p.x, y=raw_p.y,
                width=raw_p.width, height=raw_p.height,
                confidence=raw_p.confidence,
            ))
            frag_gt_matches[raw_id][gt_id] += 1

    # Build oracle result and evaluate
    oracle_pred = _make_tracking_result(
        oracle_positions, rd.frame_count, rd.video_fps,
        rd.video_width, rd.video_height,
    )
    eval_result = evaluate_rally(
        rd.rally.rally_id, interp_gt, oracle_pred,
        interpolate_gt=False,
        video_width=rd.video_width, video_height=rd.video_height,
    )

    agg = eval_result.aggregate
    f1 = agg.f1

    # Count raw fragments
    raw_tracks: dict[int, list[int]] = defaultdict(list)
    for p in rd.cached_positions:
        raw_tracks[p.track_id].append(p.frame_number)
    num_fragments = len(raw_tracks)

    # Build per-GT-player fragment info
    gt_player_fragments: dict[int, list[FragmentInfo]] = defaultdict(list)
    gt_player_coverage: dict[int, float] = {}
    gt_track_ids = interp_gt.unique_player_tracks

    for gt_id in gt_track_ids:
        gt_frames = {p.frame_number for p in interp_gt.player_positions
                     if p.track_id == gt_id}
        total_gt_frames = len(gt_frames)

        # Find fragments that predominantly map to this GT player
        matched_frags = []
        for raw_tid, gt_counts in frag_gt_matches.items():
            if gt_id in gt_counts and gt_counts[gt_id] == max(gt_counts.values()):
                frames_for_frag = sorted(raw_tracks[raw_tid])
                coverage = gt_counts[gt_id] / total_gt_frames if total_gt_frames > 0 else 0
                matched_frags.append(FragmentInfo(
                    track_id=raw_tid,
                    start_frame=frames_for_frag[0],
                    end_frame=frames_for_frag[-1],
                    num_frames=len(frames_for_frag),
                    coverage_of_gt=coverage,
                ))
        gt_player_fragments[gt_id] = sorted(matched_frags, key=lambda f: f.start_frame)

        # Coverage = oracle matched frames / GT frames
        oracle_frames_for_gt = sum(
            1 for p in oracle_positions if p.track_id == gt_id
        )
        gt_player_coverage[gt_id] = (
            oracle_frames_for_gt / total_gt_frames if total_gt_frames > 0 else 0
        )

    return OracleResult(
        rally_id=rd.rally.rally_id,
        num_raw_fragments=num_fragments,
        num_raw_positions=len(rd.cached_positions),
        oracle_hota=eval_result.hota_metrics.hota if eval_result.hota_metrics else 0,
        oracle_deta=eval_result.hota_metrics.deta if eval_result.hota_metrics else 0,
        oracle_assa=eval_result.hota_metrics.assa if eval_result.hota_metrics else 0,
        oracle_f1=f1,
        gt_player_fragments=dict(gt_player_fragments),
        gt_player_coverage=dict(gt_player_coverage),
    )


# ---------------------------------------------------------------------------
# Part 2: Per-stage evaluation
# ---------------------------------------------------------------------------

STAGE_NAMES = [
    "raw",
    "remove_bg",
    "spatial_jump",
    "height_swap",
    "color_split",
    "spatial_relink",
    "appearance_link",
    "stabilize_ids",
    "player_filter",
    "team_classify",
    "global_identity",
    "convergence_swap",
    "interpolate",
]


def _eval_positions(positions, rd: RallyData, interp_gt) -> StageResult | None:
    """Evaluate a snapshot of positions against interpolated GT."""
    from rallycut.evaluation.tracking.metrics import evaluate_rally

    if not positions:
        return None

    pred = _make_tracking_result(
        positions, rd.frame_count, rd.video_fps,
        rd.video_width, rd.video_height,
    )
    result = evaluate_rally(
        rd.rally.rally_id, interp_gt, pred,
        interpolate_gt=False,
        video_width=rd.video_width, video_height=rd.video_height,
    )
    agg = result.aggregate
    hota = result.hota_metrics
    ident = result.identity_metrics
    unique_tids = len({p.track_id for p in positions})

    return StageResult(
        stage_name="",
        hota=hota.hota if hota else 0,
        deta=hota.deta if hota else 0,
        assa=hota.assa if hota else 0,
        f1=agg.f1,
        id_switches=ident.num_switches if ident else agg.num_id_switches,
        num_positions=len(positions),
        num_tracks=unique_tids,
    )


def replay_pipeline_stages(rd: RallyData) -> list[StageResult]:
    """Replay post-processing stages one by one, evaluating after each."""
    from rallycut.evaluation.tracking.metrics import smart_interpolate_gt
    from rallycut.tracking.player_filter import (
        PlayerFilter, PlayerFilterConfig, classify_teams,
        compute_court_split, interpolate_player_gaps,
        recover_missing_players, remove_stationary_background_tracks,
        stabilize_track_ids,
    )
    from rallycut.tracking.spatial_consistency import enforce_spatial_consistency

    # Deep-copy mutable inputs
    positions = _copy_positions(rd.cached_positions)
    raw_positions = _copy_positions(rd.cached_positions)
    color_store = copy.deepcopy(rd.color_store) if rd.color_store else None
    appearance_store = copy.deepcopy(rd.appearance_store) if rd.appearance_store else None
    config = PlayerFilterConfig().scaled_for_fps(rd.video_fps)

    # Pre-interpolate GT once using raw positions
    raw_result = _make_tracking_result(
        rd.cached_positions, rd.frame_count, rd.video_fps,
        rd.video_width, rd.video_height,
    )
    interp_gt = smart_interpolate_gt(
        rd.rally.ground_truth, raw_result, rd.frame_count,
    )

    results: list[StageResult] = []

    def snapshot(name: str):
        sr = _eval_positions(positions, rd, interp_gt)
        if sr:
            sr.stage_name = name
            results.append(sr)

    # Stage 0: raw
    snapshot("raw")

    # Stage 1: remove_bg
    positions, _ = remove_stationary_background_tracks(
        positions, config, total_frames=rd.frame_count,
    )
    snapshot("remove_bg")

    # Stage 2: spatial consistency (jump only)
    positions, _ = enforce_spatial_consistency(
        positions, color_store=color_store,
        appearance_store=appearance_store,
        video_fps=rd.video_fps, drift_detection=False,
    )
    snapshot("spatial_jump")

    # Stage 3: height swap
    from rallycut.tracking.height_consistency import fix_height_swaps
    positions, _ = fix_height_swaps(
        positions, color_store=color_store,
        appearance_store=appearance_store,
    )
    snapshot("height_swap")

    # Stages 4-6: color split, spatial relink, primary relink, appearance link
    if color_store is not None and color_store.has_data():
        from rallycut.tracking.color_repair import split_tracks_by_color
        from rallycut.tracking.tracklet_link import (
            link_tracklets_by_appearance,
            relink_primary_fragments,
            relink_spatial_splits,
        )

        positions, _ = split_tracks_by_color(positions, color_store)
        snapshot("color_split")

        positions, _ = relink_spatial_splits(
            positions, color_store, appearance_store=appearance_store,
        )
        snapshot("spatial_relink")

        # Primary fragment linking: identify primaries, link non-primary
        # fragments to nearest primary before appearance-based merging
        pre_pf = PlayerFilter(
            ball_positions=rd.ball_positions,
            total_frames=rd.frame_count,
            config=config,
            court_calibrator=rd.court_calibrator,
        )
        pre_pf.analyze_tracks(positions)
        pre_primary_ids = sorted(pre_pf.primary_tracks)
        positions, pre_primary_ids, _ = relink_primary_fragments(
            positions, pre_primary_ids, color_store,
            appearance_store=appearance_store,
        )
        snapshot("primary_relink")

        positions, _ = link_tracklets_by_appearance(
            positions, color_store, appearance_store=appearance_store,
        )
        snapshot("appearance_link")
    else:
        snapshot("color_split")
        snapshot("spatial_relink")
        snapshot("primary_relink")
        snapshot("appearance_link")

    # Stage 7: stabilize IDs
    positions, id_mapping = stabilize_track_ids(positions, config)
    if id_mapping:
        if color_store is not None:
            color_store.remap_ids(id_mapping)
        if appearance_store is not None:
            appearance_store.remap_ids(id_mapping)
    snapshot("stabilize_ids")

    # Stage 8: player filter (analyze + group + per-frame filter)
    player_filter = PlayerFilter(
        ball_positions=rd.ball_positions,
        total_frames=rd.frame_count,
        config=config,
        court_calibrator=rd.court_calibrator,
    )
    player_filter.analyze_tracks(positions)
    primary_track_ids = sorted(player_filter.primary_tracks)

    # Recover missing players
    if len(primary_track_ids) < config.max_players and raw_positions:
        positions, recovered_primary_set, _ = recover_missing_players(
            pipeline_positions=positions,
            raw_positions=raw_positions,
            primary_track_ids=player_filter.primary_tracks,
            total_frames=rd.frame_count,
            ball_positions=rd.ball_positions,
            config=config,
        )
        player_filter.primary_tracks = recovered_primary_set
        primary_track_ids = sorted(recovered_primary_set)

    # Group and filter per frame
    frames: dict[int, list] = {}
    for p in positions:
        frames.setdefault(p.frame_number, []).append(p)
    filtered: list = []
    for fn in sorted(frames.keys()):
        filtered.extend(player_filter.filter(frames[fn]))
    positions = filtered
    snapshot("player_filter")

    # Stage 9: team classification
    split_result = compute_court_split(
        rd.ball_positions or [], config,
        player_positions=positions,
        court_calibrator=rd.court_calibrator,
    )
    split_y = split_result[0] if split_result else None
    split_confidence = split_result[1] if split_result else None
    precomputed_teams = split_result[2] if split_result else None

    team_assignments: dict[int, int] = {}
    if split_y is not None and split_confidence == "high":
        team_assignments = classify_teams(
            positions, split_y,
            precomputed_assignments=precomputed_teams,
        )
    elif precomputed_teams and len(set(precomputed_teams.values())) >= 2:
        team_assignments = dict(precomputed_teams)
    snapshot("team_classify")

    # Stage 10: global identity
    if (color_store is not None and color_store.has_data() and team_assignments):
        from rallycut.tracking.global_identity import optimize_global_identity

        pre_global = {(p.frame_number, id(p)): p.track_id for p in positions}
        positions, global_result = optimize_global_identity(
            positions, team_assignments, color_store,
            court_split_y=split_y,
            appearance_store=appearance_store,
        )
        if global_result.num_remapped > 0:
            remap_keys: dict[tuple[int, int], int] = {}
            for p in positions:
                key = (p.frame_number, id(p))
                old_tid = pre_global.get(key)
                if old_tid is not None and old_tid != p.track_id:
                    remap_keys[(old_tid, p.frame_number)] = p.track_id
            if remap_keys:
                if color_store is not None:
                    color_store.remap_per_frame(remap_keys)
                if appearance_store is not None:
                    appearance_store.remap_per_frame(remap_keys)
    snapshot("global_identity")

    # Stage 11: convergence swap
    if len(primary_track_ids) >= 4:
        from rallycut.tracking.convergence_swap import detect_convergence_swaps
        positions, _ = detect_convergence_swaps(
            positions, primary_track_ids,
            color_store=color_store,
            upstream_split_y=split_y,
            upstream_teams=team_assignments,
        )
    snapshot("convergence_swap")

    # Stage 12: interpolate
    if primary_track_ids:
        positions, _ = interpolate_player_gaps(
            positions, primary_track_ids, config=config,
        )
    snapshot("interpolate")

    return results


# ---------------------------------------------------------------------------
# Part 3: Root cause analysis
# ---------------------------------------------------------------------------

def analyze_root_causes(
    rd: RallyData,
    oracle: OracleResult,
    stages: list[StageResult],
) -> list[RootCause]:
    """Classify failure modes for each GT player in a bad rally."""
    causes: list[RootCause] = []
    gt = rd.rally.ground_truth
    gt_track_ids = gt.unique_player_tracks

    final = stages[-1] if stages else None
    if not final:
        return causes

    # For each GT player, check oracle coverage and pipeline behavior
    for gt_id in sorted(gt_track_ids):
        oracle_cov = oracle.gt_player_coverage.get(gt_id, 0.0)
        frags = oracle.gt_player_fragments.get(gt_id, [])

        # Detection gap: oracle can't cover this player
        if oracle_cov < 0.80:
            causes.append(RootCause(
                gt_track_id=gt_id,
                classification="detection_gap",
                detail=f"Oracle coverage {oracle_cov:.0%}, "
                       f"{len(frags)} frag(s). Raw data insufficient.",
            ))
            continue

        # Check if multiple fragments → fragmentation risk
        if len(frags) > 1:
            total_frag_coverage = sum(f.coverage_of_gt for f in frags)
            frag_summary = ", ".join(
                f"T{f.track_id}(f{f.start_frame}-{f.end_frame})"
                for f in frags
            )

            # Check if pipeline manages to link them by looking at AssA
            # If oracle AssA is high but final AssA is low, it's a pipeline problem
            if final.assa < 0.90:
                causes.append(RootCause(
                    gt_track_id=gt_id,
                    classification="fragmentation",
                    detail=f"{len(frags)} frags [{frag_summary}], "
                           f"oracle cov={oracle_cov:.0%}. "
                           f"Pipeline didn't link them (final AssA={final.assa:.1%}).",
                ))
                continue

        # Check for misassignment (oracle is good, but final AssA is low)
        if final.assa < 0.90 and oracle.oracle_assa > 0.95:
            causes.append(RootCause(
                gt_track_id=gt_id,
                classification="misassignment",
                detail=f"Oracle AssA={oracle.oracle_assa:.1%}, "
                       f"final AssA={final.assa:.1%}. "
                       f"Pipeline assigned fragment to wrong player.",
            ))
            continue

        # Check for over-filtering (oracle DetA >> final DetA)
        if oracle.oracle_deta - final.deta > 0.10:
            causes.append(RootCause(
                gt_track_id=gt_id,
                classification="over_filtering",
                detail=f"Oracle DetA={oracle.oracle_deta:.1%}, "
                       f"final DetA={final.deta:.1%}. "
                       f"Correct data removed by pipeline.",
            ))

    return causes


def find_first_bad_stage(stages: list[StageResult]) -> str | None:
    """Find the first stage where HOTA starts degrading vs. the best so far."""
    if len(stages) < 2:
        return None
    # After player_filter, look for stages that reduce HOTA
    pf_idx = next((i for i, s in enumerate(stages) if s.stage_name == "player_filter"), None)
    if pf_idx is None:
        return None
    best_hota = stages[pf_idx].hota
    for s in stages[pf_idx + 1:]:
        if s.hota < best_hota - 0.005:
            return s.stage_name
        best_hota = max(best_hota, s.hota)
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_data() -> list[RallyData]:
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.retrack_cache import RetrackCache

    config_hash = _compute_tracker_config_hash()
    cache = RetrackCache()

    print(f"Config hash: {config_hash}")
    print(f"Cache stats: {cache.stats()}")

    rallies = load_labeled_rallies()
    print(f"Loaded {len(rallies)} GT rallies from DB")

    data: list[RallyData] = []
    skipped = 0
    for r in rallies:
        entry = cache.get(r.rally_id, config_hash)
        if entry is None:
            print(f"  SKIP {r.rally_id[:8]}: not in retrack cache")
            skipped += 1
            continue

        cached_data, color_store, appearance_store = entry
        court_cal = _build_court_calibrator(r.court_calibration_json)
        ball_positions = None
        if r.predictions is not None:
            ball_positions = r.predictions.ball_positions or None

        data.append(RallyData(
            rally=r,
            cached_positions=cached_data.positions,
            ball_positions=ball_positions,
            color_store=color_store,
            appearance_store=appearance_store,
            video_fps=cached_data.video_fps,
            video_width=cached_data.video_width,
            video_height=cached_data.video_height,
            frame_count=cached_data.frame_count,
            court_calibrator=court_cal,
        ))

    if skipped:
        print(f"\nWARNING: {skipped} rallies not in cache. "
              f"Run: uv run rallycut evaluate-tracking --all --retrack --cached")
    print(f"Proceeding with {len(data)} rallies\n")
    return data


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_oracle_table(diagnostics: list[RallyDiagnostic]):
    print("=" * 100)
    print("ORACLE CEILING ANALYSIS")
    print("=" * 100)
    print(f"{'Rally':<14} | {'Frames':>6} | {'Frags':>5} | "
          f"{'Oracle HOTA':>11} | {'Oracle DetA':>11} | {'Oracle AssA':>11} | "
          f"{'Final HOTA':>10} | {'Gap':>7}")
    print("-" * 100)

    oracle_hotas, oracle_detas, oracle_assas, final_hotas = [], [], [], []

    for d in sorted(diagnostics, key=lambda d: d.oracle.oracle_hota):
        o = d.oracle
        final_hota = d.stages[-1].hota if d.stages else 0
        gap = o.oracle_hota - final_hota
        oracle_hotas.append(o.oracle_hota)
        oracle_detas.append(o.oracle_deta)
        oracle_assas.append(o.oracle_assa)
        final_hotas.append(final_hota)

        print(f"{o.rally_id[:12]:<14} | {o.num_raw_positions // max(o.num_raw_fragments, 1):>6} | "
              f"{o.num_raw_fragments:>5} | "
              f"{o.oracle_hota:>10.1%} | {o.oracle_deta:>10.1%} | {o.oracle_assa:>10.1%} | "
              f"{final_hota:>9.1%} | {gap:>+6.1%}")

    print("-" * 100)
    n = len(diagnostics)
    if n > 0:
        print(f"{'MEAN':<14} | {'':>6} | {'':>5} | "
              f"{sum(oracle_hotas)/n:>10.1%} | {sum(oracle_detas)/n:>10.1%} | "
              f"{sum(oracle_assas)/n:>10.1%} | "
              f"{sum(final_hotas)/n:>9.1%} | "
              f"{(sum(oracle_hotas)-sum(final_hotas))/n:>+6.1%}")
    print()


def print_stage_table(diagnostics: list[RallyDiagnostic]):
    print("=" * 100)
    print("PER-STAGE PIPELINE EVALUATION (averaged)")
    print("=" * 100)

    # Collect all stage names from first diagnostic
    if not diagnostics:
        return
    stage_names = [s.stage_name for s in diagnostics[0].stages]

    print(f"{'Stage':<20} | {'HOTA':>7} | {'DetA':>7} | {'AssA':>7} | "
          f"{'F1':>7} | {'IDsw':>5} | {'Tracks':>6}")
    print("-" * 80)

    for sname in stage_names:
        hotas, detas, assas, f1s, idsws, tracks = [], [], [], [], [], []
        for d in diagnostics:
            sr = next((s for s in d.stages if s.stage_name == sname), None)
            if sr:
                hotas.append(sr.hota)
                detas.append(sr.deta)
                assas.append(sr.assa)
                f1s.append(sr.f1)
                idsws.append(sr.id_switches)
                tracks.append(sr.num_tracks)

        n = len(hotas)
        if n == 0:
            continue
        print(f"{sname:<20} | {sum(hotas)/n:>6.1%} | {sum(detas)/n:>6.1%} | "
              f"{sum(assas)/n:>6.1%} | {sum(f1s)/n:>6.1%} | "
              f"{sum(idsws):>5} | {sum(tracks)/n:>5.0f}")
    print()


def print_stage_detail_for_rally(d: RallyDiagnostic):
    """Print per-stage breakdown for a single rally."""
    print(f"\n  Rally {d.rally_id[:8]} "
          f"(final HOTA={d.stages[-1].hota:.1%}, "
          f"{d.stages[-1].id_switches} IDsw, "
          f"oracle={d.oracle.oracle_hota:.1%}):")
    print(f"  {'Stage':<20} | {'HOTA':>7} | {'DetA':>7} | {'AssA':>7} | "
          f"{'IDsw':>5} | {'Trks':>5} | {'Delta HOTA':>10}")
    print(f"  {'-'*75}")

    prev_hota = 0.0
    for s in d.stages:
        delta = s.hota - prev_hota
        delta_str = f"{delta:>+9.1%}" if prev_hota > 0 else ""
        print(f"  {s.stage_name:<20} | {s.hota:>6.1%} | {s.deta:>6.1%} | "
              f"{s.assa:>6.1%} | {s.id_switches:>5} | {s.num_tracks:>5} | "
              f"{delta_str}")
        prev_hota = s.hota


def print_root_cause_report(diagnostics: list[RallyDiagnostic]):
    print("=" * 100)
    print("ROOT CAUSE ANALYSIS (rallies with HOTA < 90% or IDsw > 0)")
    print("=" * 100)

    bad_rallies = [
        d for d in diagnostics
        if (d.stages and (d.stages[-1].hota < 0.90 or d.stages[-1].id_switches > 0))
    ]

    if not bad_rallies:
        print("No bad rallies found!")
        return

    for d in sorted(bad_rallies, key=lambda x: x.stages[-1].hota):
        final = d.stages[-1]
        first_bad = find_first_bad_stage(d.stages)
        print(f"\nRally {d.rally_id[:8]} "
              f"(HOTA={final.hota:.1%}, {final.id_switches} IDsw, "
              f"oracle={d.oracle.oracle_hota:.1%}"
              f"{f', first regression at {first_bad}' if first_bad else ''}):")

        # Show fragment structure for each GT player
        o = d.oracle
        for gt_id in sorted(o.gt_player_fragments.keys()):
            frags = o.gt_player_fragments[gt_id]
            cov = o.gt_player_coverage.get(gt_id, 0)
            frag_str = ", ".join(
                f"T{f.track_id}(f{f.start_frame}-{f.end_frame}, "
                f"{f.coverage_of_gt:.0%})"
                for f in frags
            )
            print(f"  GT Player {gt_id}: oracle cov={cov:.0%}, "
                  f"frags=[{frag_str}]")

        # Show root causes
        if d.root_causes:
            for rc in d.root_causes:
                print(f"    -> {rc.classification.upper()} (GT {rc.gt_track_id}): "
                      f"{rc.detail}")
        else:
            print(f"    -> No specific per-player root cause identified "
                  f"(gap may be distributed)")

        # Show per-stage detail
        print_stage_detail_for_rally(d)

    # Summary distribution
    all_causes = [rc for d in bad_rallies for rc in d.root_causes]
    print("\n" + "=" * 60)
    print("FAILURE MODE DISTRIBUTION")
    print("=" * 60)
    mode_counts: dict[str, int] = defaultdict(int)
    for rc in all_causes:
        mode_counts[rc.classification] += 1
    total = len(all_causes) or 1
    for mode in ["detection_gap", "fragmentation", "misassignment", "over_filtering"]:
        c = mode_counts.get(mode, 0)
        print(f"  {mode:<20}: {c:>3} ({c/total:>5.0%})")
    print(f"  {'TOTAL':<20}: {len(all_causes):>3} across {len(bad_rallies)} rallies")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    all_data = load_all_data()
    if not all_data:
        print("No data loaded. Exiting.")
        sys.exit(1)

    diagnostics: list[RallyDiagnostic] = []
    total = len(all_data)

    for i, rd in enumerate(all_data):
        rally_t0 = time.time()
        rally_id = rd.rally.rally_id

        # Part 1: Oracle ceiling
        oracle = compute_oracle_ceiling(rd)

        # Part 2: Per-stage evaluation
        stages = replay_pipeline_stages(rd)

        # Part 3: Root causes (for bad rallies)
        final = stages[-1] if stages else None
        root_causes: list[RootCause] = []
        if final and (final.hota < 0.90 or final.id_switches > 0):
            root_causes = analyze_root_causes(rd, oracle, stages)

        diag = RallyDiagnostic(
            rally_id=rally_id,
            oracle=oracle,
            stages=stages,
            root_causes=root_causes,
        )
        diagnostics.append(diag)

        elapsed = time.time() - rally_t0
        final_hota = stages[-1].hota if stages else 0
        final_idsw = stages[-1].id_switches if stages else 0
        gap = oracle.oracle_hota - final_hota
        print(f"[{i+1}/{total}] {rally_id[:8]}: "
              f"oracle HOTA={oracle.oracle_hota:.1%}, "
              f"final HOTA={final_hota:.1%}, "
              f"gap={gap:+.1%}, "
              f"{final_idsw} IDsw, "
              f"{elapsed:.1f}s")

    print(f"\nTotal time: {time.time() - t0:.1f}s\n")

    # Print all reports
    print_oracle_table(diagnostics)
    print_stage_table(diagnostics)
    print_root_cause_report(diagnostics)

    # Brief redesign analysis
    print("=" * 100)
    print("PIPELINE REDESIGN ANALYSIS")
    print("=" * 100)

    oracle_hotas = [d.oracle.oracle_hota for d in diagnostics]
    final_hotas = [d.stages[-1].hota for d in diagnostics if d.stages]
    mean_oracle = sum(oracle_hotas) / len(oracle_hotas) if oracle_hotas else 0
    mean_final = sum(final_hotas) / len(final_hotas) if final_hotas else 0
    mean_gap = mean_oracle - mean_final

    print(f"\nMean oracle ceiling: {mean_oracle:.1%}")
    print(f"Mean current HOTA:  {mean_final:.1%}")
    print(f"Mean improvement room: {mean_gap:.1%}")
    print(f"\nThe oracle ceiling represents {mean_oracle:.1%} HOTA achievable")
    print(f"with perfect fragment-to-player assignment using existing raw data.")
    print(f"The gap of {mean_gap:.1%} is recoverable through better post-processing.\n")

    # Count how many rallies have >5pp gap
    big_gap = sum(1 for d in diagnostics
                  if d.stages and d.oracle.oracle_hota - d.stages[-1].hota > 0.05)
    print(f"Rallies with >5pp gap (most improvable): {big_gap}/{total}")

    # Check per-stage patterns
    pf_hotas = []
    gi_hotas = []
    for d in diagnostics:
        pf = next((s for s in d.stages if s.stage_name == "player_filter"), None)
        gi = next((s for s in d.stages if s.stage_name == "global_identity"), None)
        if pf:
            pf_hotas.append(pf.hota)
        if gi:
            gi_hotas.append(gi.hota)

    if pf_hotas and gi_hotas:
        pf_mean = sum(pf_hotas) / len(pf_hotas)
        gi_mean = sum(gi_hotas) / len(gi_hotas)
        print(f"\nPlayer filter stage avg HOTA: {pf_mean:.1%}")
        print(f"After global identity avg HOTA: {gi_mean:.1%}")
        print(f"Global identity contribution: {gi_mean - pf_mean:+.1%}")

    print()


if __name__ == "__main__":
    main()
