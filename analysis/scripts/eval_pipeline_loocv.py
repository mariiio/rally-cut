"""Leave-one-video-out CV of the full action detection + attribution pipeline.

For each held-out video:
  1. Train action GBM on remaining videos' GT contacts (from pre-extracted features)
  2. Train temporal attribution model on remaining videos (from pre-extracted features)
  3. Run full pipeline (detect_contacts → classify_rally_actions) on held-out rallies
  4. Measure attribution accuracy + action accuracy

Then ablate: disable each post-classification stage one at a time and re-run.

Usage:
    cd analysis
    uv run python scripts/eval_pipeline_loocv.py
    uv run python scripts/eval_pipeline_loocv.py --skip-ablation   # full pipeline only
    uv run python scripts/eval_pipeline_loocv.py --only-video abc  # debug single fold
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    assign_court_side_from_teams,
    correct_team_from_propagation,
    propagate_court_side,
    reattribute_players,
    repair_action_sequence,
    validate_action_sequence,
    viterbi_decode_actions,
)
from rallycut.tracking.action_type_classifier import ActionTypeClassifier
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.temporal_attribution.inference import (
    TemporalAttributionInference,
)
from rallycut.tracking.temporal_attribution.training import (
    TrainingConfig,
    train_model as train_temporal_model,
)
from scripts.eval_action_detection import (
    MatchResult,
    RallyData,
    _load_match_team_assignments,
    _match_synthetic_serves,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.train_action_classifier import extract_features_for_rally
from scripts.train_temporal_attribution import (
    extract_training_data as extract_temporal_training_data,
)
from scripts.train_temporal_attribution import (
    load_rallies_with_action_gt as load_rallies_temporal_format,
)

# Suppress verbose logging from contact detection during feature extraction
logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.temporal_attribution").setLevel(logging.WARNING)

console = Console()


# ---------------------------------------------------------------------------
# Pipeline configuration for ablation
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Toggle individual post-classification stages."""
    name: str = "full"
    use_temporal_attribution: bool = True
    use_propagate_court_side: bool = True
    use_repair_action_sequence: bool = True
    use_viterbi: bool = True
    use_reattribute_players: bool = True
    use_correct_team: bool = True


ABLATION_CONFIGS = [
    PipelineConfig(name="full"),
    PipelineConfig(name="no_temporal_attr", use_temporal_attribution=False),
    PipelineConfig(name="no_propagate_cs", use_propagate_court_side=False),
    PipelineConfig(name="no_repair_seq", use_repair_action_sequence=False),
    PipelineConfig(name="no_viterbi", use_viterbi=False),
    PipelineConfig(name="no_reattribute", use_reattribute_players=False),
    PipelineConfig(name="no_correct_team", use_correct_team=False),
]

# Simplified pipeline: matches current classify_rally_actions() after cleanup
SIMPLIFIED_CONFIG = PipelineConfig(
    name="simplified",
    use_propagate_court_side=False,
    use_repair_action_sequence=False,
    use_correct_team=False,
)


# ---------------------------------------------------------------------------
# Custom pipeline with stage toggles
# ---------------------------------------------------------------------------

def run_pipeline(
    contact_sequence: Any,
    rally_id: str,
    action_classifier: ActionTypeClassifier | None,
    match_team_assignments: dict[int, int] | None,
    cfg: PipelineConfig,
) -> Any:
    """Run classify_rally_actions with individual stage toggles."""
    ac = ActionClassifier(ActionClassifierConfig())

    result = ac.classify_rally(
        contact_sequence, rally_id,
        classifier=action_classifier,
        match_team_assignments=match_team_assignments,
    )

    if cfg.use_propagate_court_side:
        result.actions = propagate_court_side(result.actions)

    serve_tid = -1
    for a in result.actions:
        if a.action_type.value == "serve":
            if a.is_synthetic:
                serve_tid = a.player_track_id
            break

    if cfg.use_repair_action_sequence:
        result.actions = repair_action_sequence(
            result.actions,
            net_y=contact_sequence.net_y,
            ball_positions=contact_sequence.ball_positions,
            rally_start_frame=contact_sequence.rally_start_frame,
            server_track_id=serve_tid,
        )

    if cfg.use_viterbi:
        result.actions = viterbi_decode_actions(result.actions)

    result.actions = validate_action_sequence(result.actions, rally_id)

    propagated_sides = [a.court_side for a in result.actions]

    if match_team_assignments:
        assign_court_side_from_teams(result.actions, match_team_assignments)

    pre_reattrib_tids = [a.player_track_id for a in result.actions]

    if cfg.use_reattribute_players:
        result.actions = reattribute_players(
            result.actions, contact_sequence.contacts, match_team_assignments,
            max_distance_ratio=1.5,
        )

    if cfg.use_correct_team and match_team_assignments:
        result.actions = correct_team_from_propagation(
            result.actions, contact_sequence.contacts,
            propagated_sides, match_team_assignments,
            pre_reattrib_tids=pre_reattrib_tids,
        )
        assign_court_side_from_teams(result.actions, match_team_assignments)

    return result


# ---------------------------------------------------------------------------
# Pre-extracted feature cache
# ---------------------------------------------------------------------------

@dataclass
class RallyFeatures:
    """Pre-extracted features for one rally."""
    rally_id: str
    video_id: str
    # Action GBM features
    action_features: list[np.ndarray]
    action_labels: list[str]
    # Temporal attribution features
    temporal_features: list[np.ndarray]
    temporal_labels: list[int]


def preextract_all_features(
    rallies: list[RallyData],
    rallies_temporal: list[dict],
    match_teams_by_rally: dict[str, dict[int, int]] | None = None,
) -> list[RallyFeatures]:
    """Pre-extract features for all rallies once."""
    console.print("[bold]Pre-extracting features for all rallies...[/bold]")

    # Index temporal rallies by rally_id for matching
    temporal_by_id: dict[str, dict] = {}
    for r in rallies_temporal:
        temporal_by_id[r["rally_id"]] = r

    results: list[RallyFeatures] = []
    t_start = time.monotonic()

    for i, rally in enumerate(rallies):
        # Action GBM features (runs detect_contacts internally)
        rally_teams = (match_teams_by_rally or {}).get(rally.rally_id)
        action_feats, action_labels, _ = extract_features_for_rally(
            rally, tolerance=5, team_assignments=rally_teams,
        )

        # Temporal attribution features for this rally only
        temporal_rally = temporal_by_id.get(rally.rally_id)
        if temporal_rally:
            t_feats, t_labels, _ = extract_temporal_training_data(
                [temporal_rally], use_predicted=False,
            )
        else:
            t_feats, t_labels = [], []

        results.append(RallyFeatures(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            action_features=action_feats,
            action_labels=action_labels,
            temporal_features=t_feats,
            temporal_labels=t_labels,
        ))

        if (i + 1) % 20 == 0 or i == len(rallies) - 1:
            elapsed = time.monotonic() - t_start
            n_action = sum(len(rf.action_features) for rf in results)
            n_temporal = sum(len(rf.temporal_features) for rf in results)
            print(
                f"  [{i + 1}/{len(rallies)}] "
                f"{n_action} action + {n_temporal} temporal samples "
                f"({elapsed:.0f}s)",
                flush=True,
            )

    n_action = sum(len(rf.action_features) for rf in results)
    n_temporal = sum(len(rf.temporal_features) for rf in results)
    console.print(f"  Total: {n_action} action samples, {n_temporal} temporal samples")
    return results


# ---------------------------------------------------------------------------
# Training from cached features
# ---------------------------------------------------------------------------

def train_action_gbm_from_cache(
    all_features: list[RallyFeatures],
    exclude_video: str,
) -> ActionTypeClassifier | None:
    """Train action GBM from pre-extracted features, excluding one video."""
    feats: list[np.ndarray] = []
    labels: list[str] = []

    for rf in all_features:
        if rf.video_id == exclude_video:
            continue
        feats.extend(rf.action_features)
        labels.extend(rf.action_labels)

    if len(feats) < 10:
        return None

    classifier = ActionTypeClassifier()
    classifier.train(np.array(feats), np.array(labels))
    return classifier


def train_temporal_from_cache(
    all_features: list[RallyFeatures],
    exclude_video: str,
) -> TemporalAttributionInference | None:
    """Train temporal attribution from pre-extracted features, excluding one video."""
    feats: list[np.ndarray] = []
    labels: list[int] = []

    for rf in all_features:
        if rf.video_id == exclude_video:
            continue
        feats.extend(rf.temporal_features)
        labels.extend(rf.temporal_labels)

    if len(feats) < 10:
        return None

    config = TrainingConfig(max_iter=200, max_depth=4, learning_rate=0.1, seed=42)

    all_f = np.array(feats)
    all_l = np.array(labels)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp_path = Path(f.name)

    train_temporal_model(
        train_features=all_f,
        train_labels=all_l,
        val_features=all_f[:1],
        val_labels=all_l[:1],
        config=config,
        output_path=tmp_path,
    )

    inference = TemporalAttributionInference(tmp_path)
    tmp_path.unlink(missing_ok=True)
    return inference


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Results for one fold (one held-out video)."""
    video_id: str
    n_rallies: int = 0
    n_gt: int = 0
    n_tp: int = 0
    n_fp: int = 0
    n_fn: int = 0
    action_correct: int = 0
    player_correct: int = 0
    player_evaluable: int = 0
    player_evaluable_correct: int = 0


def evaluate_fold(
    rallies: list[RallyData],
    action_classifier: ActionTypeClassifier | None,
    match_teams_by_rally: dict[str, dict[int, int]],
    calibrators: dict[str, CourtCalibrator | None],
    pipeline_cfg: PipelineConfig,
    tolerance_ms: int = 167,
) -> FoldResult:
    """Evaluate pipeline on held-out rallies for one config."""
    result = FoldResult(
        video_id=rallies[0].video_id if rallies else "",
        n_rallies=len(rallies),
    )

    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    contact_config = ContactDetectionConfig(
        use_temporal_attribution=pipeline_cfg.use_temporal_attribution,
    )

    for rally in rallies:
        if not rally.ball_positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
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
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        match_teams = match_teams_by_rally.get(rally.rally_id)

        contacts = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=contact_config,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            team_assignments=match_teams,
            court_calibrator=calibrators.get(rally.video_id),
        )

        rally_actions = run_pipeline(
            contacts, rally.rally_id,
            action_classifier=action_classifier,
            match_team_assignments=match_teams,
            cfg=pipeline_cfg,
        )

        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        synth_serves = [
            a for a in pred_actions
            if a.get("isSynthetic") and a.get("action") == "serve"
        ]

        tolerance_frames = max(1, round(rally.fps * tolerance_ms / 1000))
        avail_tids: set[int] | None = None
        if rally.positions_json:
            avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, unmatched = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tolerance_frames,
            available_track_ids=avail_tids,
            team_assignments=match_teams,
        )

        if synth_serves:
            synth_tol = max(tolerance_frames, round(rally.fps * 1.0))
            _match_synthetic_serves(
                matches, synth_serves, rally.gt_labels,
                synth_tol, avail_tids, match_teams,
            )

        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    metrics = compute_metrics(all_matches, all_unmatched)
    result.n_gt = metrics["total_gt"]
    result.n_tp = metrics["tp"]
    result.n_fp = metrics["fp"]
    result.n_fn = metrics["fn"]

    matched = [m for m in all_matches if m.pred_frame is not None]
    result.action_correct = sum(1 for m in matched if m.gt_action == m.pred_action)
    result.player_correct = sum(1 for m in matched if m.player_correct)

    evaluable = [m for m in matched if m.player_evaluable]
    result.player_evaluable = len(evaluable)
    result.player_evaluable_correct = sum(1 for m in evaluable if m.player_correct)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-video-out CV of full action pipeline with ablations"
    )
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Only run full pipeline, skip ablation configs")
    parser.add_argument("--simplified", action="store_true",
                        help="Run simplified pipeline config (no repair/propagate/correct)")
    parser.add_argument("--only-video", type=str,
                        help="Only run one fold (debug)")
    parser.add_argument("--tolerance-ms", type=int, default=167,
                        help="Frame tolerance in ms (default: 167)")
    args = parser.parse_args()

    t_total_start = time.monotonic()

    # --- Load data ---
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    rallies_temporal = load_rallies_temporal_format()
    console.print(f"  {len(rallies)} rallies, {len(rallies_temporal)} temporal-format")

    # Group by video
    video_ids = sorted({r.video_id for r in rallies})
    rallies_by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    console.print(f"  {len(video_ids)} unique videos")

    if args.only_video:
        video_ids = [v for v in video_ids if v.startswith(args.only_video)]
        if not video_ids:
            console.print(f"[red]No video matching '{args.only_video}'[/red]")
            return
        console.print(f"  Filtered to {len(video_ids)} video(s)")

    # Load court calibrations
    calibrators: dict[str, CourtCalibrator | None] = {}
    all_video_ids = {r.video_id for r in rallies}
    for vid in all_video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    # Load match-level team assignments
    match_teams_by_rally = _load_match_team_assignments(all_video_ids, min_confidence=0.70)
    n_with_match = sum(1 for r in rallies if r.rally_id in match_teams_by_rally)
    console.print(f"  Match teams: {n_with_match}/{len(rallies)} rallies")

    # --- Pre-extract features (one-time cost) ---
    cached_features = preextract_all_features(
        rallies, rallies_temporal, match_teams_by_rally,
    )

    # Index by video for fast fold splitting
    features_by_video: dict[str, list[RallyFeatures]] = defaultdict(list)
    for rf in cached_features:
        features_by_video[rf.video_id].append(rf)

    # --- Determine configs to run ---
    if args.simplified:
        configs = [SIMPLIFIED_CONFIG]
    elif args.skip_ablation:
        configs = [PipelineConfig(name="full")]
    else:
        configs = ABLATION_CONFIGS

    console.print(f"\n[bold]Running {len(configs)} config(s) × {len(video_ids)} folds[/bold]\n")

    # --- LOO-CV ---
    results: dict[str, list[FoldResult]] = {c.name: [] for c in configs}

    for fold_idx, held_out_video in enumerate(video_ids):
        t_fold_start = time.monotonic()
        held_out_rallies = rallies_by_video[held_out_video]

        print(
            f"[{fold_idx + 1}/{len(video_ids)}] "
            f"{held_out_video[:8]} ({len(held_out_rallies)} rallies)",
            end="",
            flush=True,
        )

        # Train action GBM from cached features
        action_clf = train_action_gbm_from_cache(cached_features, held_out_video)

        # Train temporal attribution from cached features
        temporal_model = train_temporal_from_cache(cached_features, held_out_video)

        # Run each config
        for cfg in configs:
            # Inject temporal model into cache
            import rallycut.tracking.contact_detector as cd
            if cfg.use_temporal_attribution and temporal_model is not None:
                cd._temporal_attributor_cache["default"] = temporal_model
            else:
                cd._temporal_attributor_cache["default"] = None

            fold_result = evaluate_fold(
                held_out_rallies,
                action_classifier=action_clf,
                match_teams_by_rally=match_teams_by_rally,
                calibrators=calibrators,
                pipeline_cfg=cfg,
                tolerance_ms=args.tolerance_ms,
            )
            fold_result.video_id = held_out_video
            results[cfg.name].append(fold_result)

        # Print summary for this fold (first config)
        fr = results[configs[0].name][-1]
        action_acc = fr.action_correct / max(1, fr.n_tp)
        attr_acc = fr.player_evaluable_correct / max(1, fr.player_evaluable)
        elapsed = time.monotonic() - t_fold_start
        print(
            f" → action={action_acc:.0%} attr={attr_acc:.0%} "
            f"(TP={fr.n_tp} eval={fr.player_evaluable}) [{elapsed:.1f}s]",
            flush=True,
        )

    # --- Restore default temporal model cache ---
    import rallycut.tracking.contact_detector as cd
    cd._temporal_attributor_cache.clear()

    # --- Aggregate and report ---
    print(flush=True)

    # Per-fold detail table (first config)
    primary_name = configs[0].name
    fold_table = Table(title=f"Per-Video LOO-CV Results ({primary_name})")
    fold_table.add_column("Video", style="cyan", max_width=10)
    fold_table.add_column("Rallies", justify="right")
    fold_table.add_column("GT", justify="right")
    fold_table.add_column("TP", justify="right")
    fold_table.add_column("F1", justify="right")
    fold_table.add_column("Action Acc", justify="right")
    fold_table.add_column("Attr (eval)", justify="right")
    fold_table.add_column("Eval N", justify="right")

    full_folds = results[primary_name]
    for fr in full_folds:
        action_acc = fr.action_correct / max(1, fr.n_tp)
        attr_acc = fr.player_evaluable_correct / max(1, fr.player_evaluable)
        f1 = 0.0
        if fr.n_tp > 0:
            p = fr.n_tp / max(1, fr.n_tp + fr.n_fp)
            r = fr.n_tp / max(1, fr.n_tp + fr.n_fn)
            f1 = 2 * p * r / max(1e-9, p + r)

        style = "green" if attr_acc > 0.80 else "red" if attr_acc < 0.65 else ""
        fold_table.add_row(
            fr.video_id[:8],
            str(fr.n_rallies),
            str(fr.n_gt),
            str(fr.n_tp),
            f"{f1:.1%}",
            f"{action_acc:.1%}",
            f"{attr_acc:.1%}",
            str(fr.player_evaluable),
            style=style,
        )

    console.print(fold_table)

    # Summary table: all configs
    summary_table = Table(title="Pipeline LOO-CV: Ablation Summary")
    summary_table.add_column("Config", style="cyan")
    summary_table.add_column("Contact F1", justify="right")
    summary_table.add_column("Action Acc", justify="right")
    summary_table.add_column("Attr (eval)", justify="right")
    summary_table.add_column("Attr Correct", justify="right")
    summary_table.add_column("Attr Eval N", justify="right")
    summary_table.add_column("Δ Attr", justify="right")

    full_eval_correct = sum(fr.player_evaluable_correct for fr in full_folds)
    full_eval_total = sum(fr.player_evaluable for fr in full_folds)
    full_attr_acc = full_eval_correct / max(1, full_eval_total)

    for cfg_name in [c.name for c in configs]:
        folds = results[cfg_name]
        total_tp = sum(fr.n_tp for fr in folds)
        total_fp = sum(fr.n_fp for fr in folds)
        total_fn = sum(fr.n_fn for fr in folds)
        total_action_correct = sum(fr.action_correct for fr in folds)
        total_eval_correct = sum(fr.player_evaluable_correct for fr in folds)
        total_eval_n = sum(fr.player_evaluable for fr in folds)

        p = total_tp / max(1, total_tp + total_fp)
        r = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        action_acc = total_action_correct / max(1, total_tp)
        attr_acc = total_eval_correct / max(1, total_eval_n)
        delta = attr_acc - full_attr_acc

        delta_str = f"{delta:+.1%}" if cfg_name != "full" else "—"
        style = ""
        if cfg_name != "full":
            style = "green" if delta > 0.005 else "red" if delta < -0.005 else "dim"

        summary_table.add_row(
            cfg_name,
            f"{f1:.1%}",
            f"{action_acc:.1%}",
            f"{attr_acc:.1%}",
            f"{total_eval_correct}/{total_eval_n}",
            str(total_eval_n),
            delta_str,
            style=style,
        )

    console.print(summary_table)

    # Final summary
    total_tp = sum(fr.n_tp for fr in full_folds)
    total_action = sum(fr.action_correct for fr in full_folds)
    total_elapsed = time.monotonic() - t_total_start
    console.print(f"\n[bold]{primary_name} pipeline LOO-CV:[/bold]")
    console.print(f"  Action accuracy: {total_action}/{total_tp} = {total_action/max(1,total_tp):.1%}")
    console.print(f"  Attribution (eval): {full_eval_correct}/{full_eval_total} = {full_attr_acc:.1%}")
    console.print(f"  Folds: {len(full_folds)} videos, {sum(fr.n_rallies for fr in full_folds)} rallies")
    console.print(f"  Total time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
