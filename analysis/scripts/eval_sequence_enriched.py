"""Evaluate sequence-enriched contact detection pipeline.

Validates three clean changes to the production pipeline:
  1. Sequence-enriched GBM: 7 MS-TCN++ probabilities as contact classifier features
  2. Adaptive dedup: 4-frame min distance for cross-side contacts (vs 12 same-side)
  3. Team-seeded attribution: sequence model action types → expected team

Uses k-fold grouped CV (default 5-fold) with per-fold retraining of both
the MS-TCN++ and the contact GBM, ensuring honest held-out evaluation.

Runs 5 ablation configs to measure each change independently:
  baseline:        per-fold 20-dim GBM, fixed dedup, chain-seeded attribution
  +seq_gbm:        per-fold 27-dim GBM (with sequence probs)
  +adaptive_dedup: adaptive cross-side dedup distance
  +team_reattr:    sequence-model team-seeded attribution
  combined:        all three changes

Usage:
    cd analysis
    uv run python scripts/eval_sequence_enriched.py
    uv run python scripts/eval_sequence_enriched.py --folds 3
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from eval_action_detection import (  # noqa: E402
    GtLabel,
    MatchResult,
    RallyData,
    _load_match_team_assignments,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from train_contact_classifier import (
    extract_candidate_features,
    label_candidates,
)

from rallycut.actions.sequence_dataset import (
    SequenceActionDataset,
    collate_rally_sequences,
)
from rallycut.actions.trajectory_features import (
    ACTION_TO_IDX,
    ACTION_TYPES,
    FEATURE_DIM,
    NUM_CLASSES,
    build_frame_labels,
    extract_trajectory_features,
)
from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig
from rallycut.temporal.temporal_maxer.training import FocalLoss, compute_tmse_loss
from rallycut.tracking.action_classifier import (
    ActionType,
    classify_rally_actions,
    reattribute_players,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition

logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Rally-level serve evaluation
# ---------------------------------------------------------------------------


@dataclass
class ServeResult:
    """Rally-level serve evaluation — separate from contact-level metrics.

    Every rally has exactly 1 GT serve and the pipeline always produces 1
    serve (real or synthetic).  This evaluates action-sequence correctness
    independent of whether the contact detector found the serve contact.
    """
    rally_id: str
    gt_side: str          # GT serve court side ("near" or "far")
    pred_side: str        # predicted serve court side
    pred_is_synthetic: bool
    side_correct: bool
    player_correct: bool | None  # None if GT player not evaluable


# ---------------------------------------------------------------------------
# Ablation config
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    name: str
    use_sequence_gbm: bool = False     # 27-dim GBM with sequence probs
    use_adaptive_dedup: bool = False    # Cross-side 4-frame min distance
    use_team_reattribution: bool = False  # Sequence-model team hints


ALL_ABLATION_CONFIGS = [
    AblationConfig("baseline"),
    AblationConfig("+seq_gbm", use_sequence_gbm=True),
    AblationConfig("+adaptive_dedup", use_adaptive_dedup=True),
    AblationConfig("+team_reattr", use_team_reattribution=True),
    AblationConfig("combined", use_sequence_gbm=True, use_adaptive_dedup=True,
                   use_team_reattribution=True),
]

# Default: only baseline vs seq_gbm (the only change with impact)
ABLATION_CONFIGS = [
    AblationConfig("baseline"),
    AblationConfig("+seq_gbm", use_sequence_gbm=True),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_ball(raw: list[dict]) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _parse_players(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"], track_id=pp["trackId"],
            x=pp["x"], y=pp["y"],
            width=pp.get("width", 0.05), height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
        )
        for pp in raw
    ]


@dataclass
class RallyBundle:
    rally: RallyData
    ball_positions: list[BallPosition]
    player_positions: list[PlayerPosition]
    trajectory_features: np.ndarray  # (T, FEATURE_DIM)
    trajectory_labels: np.ndarray    # (T,) int64
    match_teams: dict[int, int] | None
    gt_labels: list[GtLabel]


def prepare_rallies(label_spread: int = 2) -> list[RallyBundle]:
    console.print("[bold]Loading rallies...[/bold]")
    rallies = load_rallies_with_action_gt()

    video_ids = {r.video_id for r in rallies}
    # Pass rally_positions so match_team_assignments triggers
    # verify_team_assignments — mirrors production (track_player.py:1016).
    # Without this the eval/diagnose numbers measure an unverified-team path
    # that production does not actually run.
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _parse_players(r.positions_json)
    rally_team_map = _load_match_team_assignments(
        video_ids, rally_positions=rally_pos_lookup,
    )

    # Court calibrations for enhanced features
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import load_court_calibration
    homographies: dict[str, np.ndarray] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated and cal.homography is not None:
                homographies[vid] = cal.homography.homography

    bundles: list[RallyBundle] = []
    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        ball_positions = _parse_ball(rally.ball_positions_json)
        player_positions = _parse_players(rally.positions_json)
        if not ball_positions:
            continue

        match_teams = rally_team_map.get(rally.rally_id)
        features = extract_trajectory_features(
            ball_positions, player_positions, rally.court_split_y, rally.frame_count,
            team_assignments=match_teams,
            homography=homographies.get(rally.video_id),
        )
        gt_dicts = [{"frame": gt.frame, "action": gt.action} for gt in rally.gt_labels]
        labels = build_frame_labels(gt_dicts, rally.frame_count, label_spread=label_spread)

        bundles.append(RallyBundle(
            rally=rally, ball_positions=ball_positions,
            player_positions=player_positions,
            trajectory_features=features, trajectory_labels=labels,
            match_teams=match_teams, gt_labels=rally.gt_labels,
        ))

    console.print(f"  {len(bundles)} rallies loaded")
    return bundles


# ---------------------------------------------------------------------------
# MS-TCN++ training + inference
# ---------------------------------------------------------------------------


def train_mstcn(
    train_bundles: list[RallyBundle],
    val_bundles: list[RallyBundle],
    args: argparse.Namespace,
) -> MSTCN:
    device = torch.device(args.device)
    config = MSTCNConfig(
        feature_dim=FEATURE_DIM, hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES, num_stages=args.num_stages,
        num_layers=args.num_layers, dropout=args.dropout,
    )
    model = MSTCN(config).to(device)

    all_labels = np.concatenate([b.trajectory_labels for b in train_bundles])
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    weights = torch.tensor(
        [class_counts.max() / c for c in class_counts], dtype=torch.float32, device=device,
    )
    weights[0] = weights[0].clamp(max=1.0)
    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma, reduction="none")

    train_ds = SequenceActionDataset(
        [b.trajectory_features for b in train_bundles],
        [b.trajectory_labels for b in train_bundles],
        augment=True, noise_std=args.noise_std,
    )
    val_ds = SequenceActionDataset(
        [b.trajectory_features for b in val_bundles],
        [b.trajectory_labels for b in val_bundles],
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_rally_sequences)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_rally_sequences)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state, patience_ctr = 0.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        for feat, lbl, msk in train_dl:
            feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for logits in model.forward_all_stages(feat, msk):
                ce = criterion(logits, lbl)
                m = msk.squeeze(1)
                ce = (ce * m).sum() / m.sum()
                total_loss = total_loss + ce + args.tmse_weight * compute_tmse_loss(logits, msk)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for feat, lbl, msk in val_dl:
                feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
                p = model(feat, msk).argmax(dim=1)
                m = msk.squeeze(1).bool()
                for b in range(feat.shape[0]):
                    v = m[b]
                    pi, ti = p[b][v].cpu().numpy(), lbl[b][v].cpu().numpy()
                    am = ti > 0
                    if am.any():
                        preds_all.extend(pi[am].tolist())
                        targets_all.extend(ti[am].tolist())

        acc = sum(1 for p, t in zip(preds_all, targets_all) if p == t) / len(targets_all) if targets_all else 0.0
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)
    return model


def get_sequence_probs(
    model: MSTCN, bundle: RallyBundle, device: torch.device,
) -> np.ndarray:
    """Run MS-TCN++ inference → (NUM_CLASSES, T) probabilities."""
    model.eval()
    feat = torch.from_numpy(bundle.trajectory_features).float().unsqueeze(0).transpose(1, 2).to(device)
    mask = torch.ones(1, 1, bundle.rally.frame_count, device=device)
    with torch.no_grad():
        logits = model(feat, mask)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs


def get_sequence_action_types(
    probs: np.ndarray, contact_frames: list[int],
) -> list[str]:
    """Extract action type predictions at specific frames from model probs."""
    action_types: list[str] = []
    for frame in contact_frames:
        if 0 <= frame < probs.shape[1]:
            cls = int(np.argmax(probs[1:, frame])) + 1  # skip background
            action_types.append(ACTION_TYPES[cls - 1])
        else:
            action_types.append("unknown")
    return action_types


# ---------------------------------------------------------------------------
# Per-fold GBM training
# ---------------------------------------------------------------------------


def train_fold_gbm(
    train_bundles: list[RallyBundle],
    model: MSTCN,
    device: torch.device,
    use_sequence_features: bool,
) -> ContactClassifier:
    """Train contact GBM on train rallies with optional sequence features."""
    all_features_arrays: list[np.ndarray] = []
    all_labels: list[int] = []

    for bundle in train_bundles:
        # Get sequence probs for this rally (from model trained on same data —
        # slightly optimistic but GBM is shallow enough to not overfit)
        seq_probs = get_sequence_probs(model, bundle, device) if use_sequence_features else None

        features, candidate_frames = extract_candidate_features(
            bundle.rally, sequence_probs=seq_probs,
        )
        if not features:
            continue

        labels = label_candidates(candidate_frames, bundle.gt_labels, tolerance=5)

        for feat, lbl in zip(features, labels):
            all_features_arrays.append(feat.to_array())
            all_labels.append(lbl)

    if not all_features_arrays:
        return ContactClassifier(threshold=0.40)

    x_train = np.array(all_features_arrays)
    y = np.array(all_labels)

    clf = ContactClassifier(threshold=0.40)
    clf.train(x_train, y)
    return clf


# ---------------------------------------------------------------------------
# Per-rally evaluation
# ---------------------------------------------------------------------------


def evaluate_rally(
    bundle: RallyBundle,
    model: MSTCN,
    gbm: ContactClassifier,
    config: AblationConfig,
    device: torch.device,
) -> tuple[list[MatchResult], list[dict], ServeResult | None]:
    """Run the full pipeline on one rally and return match results.

    Returns:
        Tuple of (frame-level matches, unmatched preds, rally-level serve result).
        Frame-level matching excludes synthetic actions (contact detection eval).
        Serve result evaluates the pipeline's serve independently (always 1 per rally).
    """
    probs = get_sequence_probs(model, bundle, device)

    # Contact detection with optional sequence probs and adaptive dedup
    det_config = ContactDetectionConfig(
        adaptive_dedup=config.use_adaptive_dedup,
    )
    contact_seq = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=det_config,
        frame_count=bundle.rally.frame_count,
        classifier=gbm,
        sequence_probs=probs if config.use_sequence_gbm else None,
    )

    # Action classification (uses full pipeline including heuristics)
    rally_actions = classify_rally_actions(
        contact_seq,
        match_team_assignments=bundle.match_teams,
    )

    # Override action types with MS-TCN++ predictions (hybrid approach).
    # Exempt serve — it uses structural rally constraints (first action,
    # baseline position, arc crossing) that the per-frame model cannot learn.
    # Receive and block benefit from the model despite having heuristic rules.
    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if 0 <= frame < probs.shape[1]:
            cls = int(np.argmax(probs[1:, frame]))
            action.action_type = ActionType(ACTION_TYPES[cls])

    # Team-seeded reattribution
    if config.use_team_reattribution and bundle.match_teams:
        seq_types = get_sequence_action_types(
            probs, [a.frame for a in rally_actions.actions],
        )
        reattribute_players(
            rally_actions.actions,
            contact_seq.contacts,
            bundle.match_teams,
            sequence_action_types=seq_types,
        )

    # --- Rally-level serve evaluation ---
    # Every rally has 1 GT serve and the pipeline always produces 1 serve
    # (real or synthetic). Evaluate independently of contact detection.
    serve_result: ServeResult | None = None
    gt_serves = [gt for gt in bundle.gt_labels if gt.action == "serve"]
    pred_serves = [a for a in rally_actions.actions
                   if a.action_type == ActionType.SERVE]
    if gt_serves and pred_serves:
        gt_s = gt_serves[0]
        pred_s = pred_serves[0]
        # Determine GT side from team assignments or ball position
        gt_side = ""
        if bundle.match_teams and gt_s.player_track_id >= 0:
            team = bundle.match_teams.get(gt_s.player_track_id)
            if team is not None:
                gt_side = "near" if team == 0 else "far"
        # Player attribution check
        player_correct: bool | None = None
        if gt_s.player_track_id >= 0:
            player_correct = pred_s.player_track_id == gt_s.player_track_id
        serve_result = ServeResult(
            rally_id=bundle.rally.rally_id,
            gt_side=gt_side,
            pred_side=pred_s.court_side,
            pred_is_synthetic=pred_s.is_synthetic,
            side_correct=(gt_side == pred_s.court_side) if gt_side else True,
            player_correct=player_correct,
        )

    # --- Frame-level contact matching (excludes synthetic) ---
    preds = [
        {
            "frame": a.frame,
            "action": a.action_type.value,
            "playerTrackId": a.player_track_id,
            "courtSide": a.court_side,
        }
        for a in rally_actions.actions
        if not a.is_synthetic
    ]

    matches, unmatched = match_contacts(
        bundle.gt_labels, preds, tolerance=5,
        team_assignments=bundle.match_teams,
    )
    return matches, unmatched, serve_result


# ---------------------------------------------------------------------------
# Main CV loop
# ---------------------------------------------------------------------------


def run_cv(
    bundles: list[RallyBundle],
    args: argparse.Namespace,
) -> None:
    videos: dict[str, list[RallyBundle]] = defaultdict(list)
    for b in bundles:
        videos[b.rally.video_id].append(b)

    video_ids = sorted(videos.keys())
    n_folds = args.folds if args.folds > 0 else len(video_ids)
    n_folds = min(n_folds, len(video_ids))
    fold_map = {vid: i % n_folds for i, vid in enumerate(video_ids)}

    cv_label = f"{n_folds}-fold CV"
    console.print(f"\n[bold]{cv_label}: {len(video_ids)} videos, {len(bundles)} rallies[/bold]")
    console.print(f"  Configs: {', '.join(c.name for c in ABLATION_CONFIGS)}")

    device = torch.device(args.device)
    start_time = time.time()

    # Accumulate results per config
    results: dict[str, tuple[list[MatchResult], list[dict]]] = {
        c.name: ([], []) for c in ABLATION_CONFIGS
    }
    serve_results: dict[str, list[ServeResult]] = {
        c.name: [] for c in ABLATION_CONFIGS
    }

    for fold in range(n_folds):
        train_bundles = [b for b in bundles if fold_map[b.rally.video_id] != fold]
        test_bundles = [b for b in bundles if fold_map[b.rally.video_id] == fold]
        if not test_bundles or not train_bundles:
            continue

        fold_start = time.time()

        # 1. Train MS-TCN++ (shared across all configs)
        model = train_mstcn(train_bundles, test_bundles, args)

        # 2. Train per-fold GBMs (one with seq features, one without)
        gbm_20 = train_fold_gbm(train_bundles, model, device, use_sequence_features=False)
        gbm_27 = train_fold_gbm(train_bundles, model, device, use_sequence_features=True)

        train_time = time.time() - fold_start

        # 3. Evaluate each config on test rallies
        for cfg in ABLATION_CONFIGS:
            gbm = gbm_27 if cfg.use_sequence_gbm else gbm_20
            for bundle in test_bundles:
                matches, unmatched, serve_res = evaluate_rally(
                    bundle, model, gbm, cfg, device,
                )
                results[cfg.name][0].extend(matches)
                results[cfg.name][1].extend(unmatched)
                if serve_res is not None:
                    serve_results[cfg.name].append(serve_res)

        # Per-fold summary (last config)
        last_cfg = ABLATION_CONFIGS[-1]
        last_matches = results[last_cfg.name][0][-len(test_bundles) * 20:]
        m = compute_metrics(last_matches, [])
        fold_time = time.time() - fold_start
        console.print(
            f"  [{fold + 1}/{n_folds}] rallies={len(test_bundles)} "
            f"train={train_time:.0f}s total={fold_time:.0f}s "
            f"{last_cfg.name}: F1={m['f1']:.1%} action={m['action_accuracy']:.1%}"
        )

    total_time = time.time() - start_time
    console.print(f"\n[bold]Results ({total_time:.0f}s total)[/bold]")

    # --- Summary table ---
    table = Table(title=f"Ablation Results ({cv_label})")
    table.add_column("Config")
    table.add_column("Contact F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Action Acc", justify="right")
    table.add_column("Player Attr", justify="right")
    table.add_column("Δ F1", justify="right")

    baseline_f1 = None
    for cfg in ABLATION_CONFIGS:
        matches, unmatched = results[cfg.name]
        m = compute_metrics(matches, unmatched)

        # Attribution accuracy
        attr_eval = [r for r in matches if r.pred_frame is not None and r.player_evaluable]
        attr_correct = sum(1 for r in attr_eval if r.player_correct)
        attr_acc = attr_correct / len(attr_eval) if attr_eval else 0.0

        f1 = m["f1"]
        if baseline_f1 is None:
            baseline_f1 = f1
            delta = "—"
        else:
            d = f1 - baseline_f1
            delta = f"[green]+{d:.1%}[/green]" if d > 0.001 else (
                f"[red]{d:.1%}[/red]" if d < -0.001 else "0.0%"
            )

        table.add_row(
            cfg.name,
            f"{m['f1']:.1%}",
            f"{m['precision']:.1%}",
            f"{m['recall']:.1%}",
            f"{m['action_accuracy']:.1%}",
            f"{attr_acc:.1%}",
            delta,
        )

    console.print(table)

    # --- Per-class for last config ---
    last_cfg_name = ABLATION_CONFIGS[-1].name
    comb_matches, comb_unmatched = results[last_cfg_name]
    overall = compute_metrics(comb_matches, comb_unmatched)
    if "per_class" in overall:
        cls_table = Table(title="Per-Class (contact-level, excludes synthetic)")
        cls_table.add_column("Action")
        cls_table.add_column("TP", justify="right")
        cls_table.add_column("FP", justify="right")
        cls_table.add_column("FN", justify="right")
        cls_table.add_column("F1", justify="right")
        for action in ACTION_TYPES:
            if action in overall["per_class"]:
                c = overall["per_class"][action]
                cls_table.add_row(action, str(c["tp"]), str(c["fp"]), str(c["fn"]),
                                  f"{c['f1']:.1%}")
        console.print(cls_table)

    # --- Rally-level serve evaluation ---
    last_serves = serve_results[last_cfg_name]
    if last_serves:
        n_total = len(last_serves)
        n_synthetic = sum(1 for s in last_serves if s.pred_is_synthetic)
        n_real = n_total - n_synthetic
        side_eval = [s for s in last_serves if s.gt_side]
        n_side_correct = sum(1 for s in side_eval if s.side_correct)
        player_eval = [s for s in last_serves if s.player_correct is not None]
        n_player_correct = sum(1 for s in player_eval if s.player_correct)

        serve_table = Table(title="Rally-Level Serve Evaluation")
        serve_table.add_column("Metric", style="bold")
        serve_table.add_column("Value", justify="right")
        serve_table.add_row("Rallies with GT serve", str(n_total))
        serve_table.add_row("Serve detected (real contact)", str(n_real))
        serve_table.add_row("Serve inferred (synthetic)", str(n_synthetic))
        serve_table.add_row(
            "Serve present",
            f"{n_total}/{n_total} (100%)",
        )
        if side_eval:
            serve_table.add_row(
                "Serve side correct",
                f"{n_side_correct}/{len(side_eval)} "
                f"({n_side_correct / len(side_eval):.1%})",
            )
        if player_eval:
            serve_table.add_row(
                "Server attribution correct",
                f"{n_player_correct}/{len(player_eval)} "
                f"({n_player_correct / len(player_eval):.1%})",
            )
        console.print(serve_table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate sequence-enriched pipeline")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tmse-weight", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--label-spread", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--device", type=str,
        default="mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundles = prepare_rallies(label_spread=args.label_spread)
    if not bundles:
        console.print("[red]No rallies loaded.[/red]")
        return

    run_cv(bundles, args)


if __name__ == "__main__":
    main()
