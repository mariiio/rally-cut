"""Evaluate per-candidate pose attribution with leave-one-video-out CV.

Compares per-candidate binary classifier (with and without pose features)
against proximity baseline and the existing canonical-slot temporal model.

Usage:
    cd analysis
    uv run python scripts/eval_pose_attribution.py
    uv run python scripts/eval_pose_attribution.py --spatial-only   # No pose features
    uv run python scripts/eval_pose_attribution.py --rally <id>     # Debug single rally
    uv run python scripts/eval_pose_attribution.py --ablation       # Full ablation study
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.pose_attribution.features import (
    FEATURE_NAMES,
    NUM_FEATURES,
    POSE_FEATURE_COUNT,
    extract_candidate_features,
)
from rallycut.tracking.pose_attribution.pose_cache import load_pose_cache
from rallycut.tracking.pose_attribution.training import TrainingConfig, train_model
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
)

console = Console()


@dataclass
class ContactSample:
    """One GT contact with per-candidate features and labels."""

    rally_id: str
    video_id: str
    gt_frame: int
    gt_action: str
    gt_track_id: int
    candidate_features: list[np.ndarray]  # per candidate
    candidate_track_ids: list[int]
    proximity_track_id: int  # nearest player (baseline)


@dataclass
class EvalResult:
    """Evaluation results for one configuration."""

    name: str
    accuracy: float
    n_correct: int
    n_total: int
    per_action: dict[str, tuple[int, int]] = field(default_factory=dict)
    auc: float = 0.0


def _parse_positions(rally: RallyData) -> tuple[list[BallPosition], list[PlayerPosition]]:
    """Parse rally data into typed position lists."""
    ball_positions = []
    if rally.ball_positions_json:
        for bp in rally.ball_positions_json:
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
                ball_positions.append(BallPosition(
                    frame_number=bp["frameNumber"],
                    x=bp["x"],
                    y=bp["y"],
                    confidence=bp.get("confidence", 1.0),
                ))

    player_positions = []
    if rally.positions_json:
        for pp in rally.positions_json:
            player_positions.append(PlayerPosition(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"],
                y=pp["y"],
                width=pp["width"],
                height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            ))

    return ball_positions, player_positions


def extract_samples(
    rallies: list[RallyData],
    use_pose: bool = True,
) -> list[ContactSample]:
    """Extract per-candidate features for all GT contacts.

    For each GT contact, detect contacts from the pipeline (to match the
    evaluation protocol), then find the detected contact nearest to the GT
    frame and extract per-candidate features at that frame.
    """
    samples: list[ContactSample] = []
    n_skipped = 0

    for rally in rallies:
        ball_positions, player_positions = _parse_positions(rally)
        if not ball_positions or not player_positions:
            continue

        # Load pose cache if available
        pose_data = None
        if use_pose:
            pose_data = load_pose_cache(rally.rally_id)

        # Detect contacts to get contact sequence context
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )

        # Build contact frame lookup for sequence context
        contact_frames = [c.frame for c in contact_seq.contacts]

        for gt in rally.gt_labels:
            if gt.player_track_id < 0:
                n_skipped += 1
                continue

            # Find the detected contact nearest to this GT frame
            # Use GT frame directly for feature extraction
            gt_frame = gt.frame

            # Compute contact_index and side_count from detected contacts
            contact_index = 0
            side_count = 1
            for i, cf in enumerate(contact_frames):
                if cf <= gt_frame:
                    contact_index = i

            # Side count: how many consecutive contacts on same side
            if contact_seq.contacts:
                for prev_c in reversed(contact_seq.contacts[:contact_index + 1]):
                    if prev_c.court_side == "unknown":
                        break
                    side_count += 1
                    if side_count >= 3:
                        break

            # Extract per-candidate features
            result = extract_candidate_features(
                contact_frame=gt_frame,
                ball_positions=ball_positions,
                player_positions=player_positions,
                contact_index=contact_index,
                side_count=min(side_count, 3),
                pose_data=pose_data,
            )

            if result is None:
                n_skipped += 1
                continue

            candidate_tids = [tid for tid, _ in result]
            candidate_feats = [feat for _, feat in result]

            # Check that GT track is among candidates
            if gt.player_track_id not in candidate_tids:
                n_skipped += 1
                continue

            samples.append(ContactSample(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                gt_frame=gt_frame,
                gt_action=gt.action,
                gt_track_id=gt.player_track_id,
                candidate_features=candidate_feats,
                candidate_track_ids=candidate_tids,
                proximity_track_id=candidate_tids[0],  # nearest
            ))

    if n_skipped > 0:
        console.print(f"  [dim]Skipped {n_skipped} contacts (no track/insufficient data)[/dim]")

    return samples


def run_loocv(
    samples: list[ContactSample],
    feature_mask: np.ndarray | None = None,
    config: TrainingConfig | None = None,
) -> EvalResult:
    """Run leave-one-video-out CV with per-candidate binary classifier.

    Args:
        samples: Contact samples with per-candidate features.
        feature_mask: Boolean mask for feature selection. If None, use all.
        config: Training hyperparameters.

    Returns:
        EvalResult with accuracy and per-action breakdown.
    """
    if config is None:
        config = TrainingConfig()

    # Group by video
    by_video: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_video[s.video_id].append(i)

    video_ids = sorted(by_video.keys())

    correct = 0
    total = 0
    per_action_correct: dict[str, int] = defaultdict(int)
    per_action_total: dict[str, int] = defaultdict(int)
    all_probs: list[float] = []
    all_labels: list[int] = []

    for fold_idx, held_out_vid in enumerate(video_ids):
        held_out_indices = set(by_video[held_out_vid])

        # Build training data: per-candidate samples from other videos
        train_X: list[np.ndarray] = []
        train_y: list[int] = []

        for i, s in enumerate(samples):
            if i in held_out_indices:
                continue
            for feat, tid in zip(s.candidate_features, s.candidate_track_ids):
                f = feat if feature_mask is None else feat[feature_mask]
                train_X.append(f)
                train_y.append(1 if tid == s.gt_track_id else 0)

        train_X_arr = np.stack(train_X)
        train_y_arr = np.array(train_y, dtype=np.int32)

        # Train binary classifier
        clf, _ = train_model(train_X_arr, train_y_arr, config)

        # Evaluate on held-out video
        for i in by_video[held_out_vid]:
            s = samples[i]

            # Score each candidate
            feats = [
                f if feature_mask is None else f[feature_mask]
                for f in s.candidate_features
            ]
            X = np.stack(feats)
            probs = clf.predict_proba(X)

            # P(touching) for class 1
            class_idx = list(clf.classes_).index(1) if 1 in clf.classes_ else -1
            if class_idx < 0:
                touch_probs = np.ones(len(X)) * 0.5
            else:
                touch_probs = probs[:, class_idx]

            # Pick highest-scoring candidate
            best_idx = int(touch_probs.argmax())
            pred_tid = s.candidate_track_ids[best_idx]

            is_correct = pred_tid == s.gt_track_id
            if is_correct:
                correct += 1
                per_action_correct[s.gt_action] += 1
            total += 1
            per_action_total[s.gt_action] += 1

            # Collect for AUC
            for feat, tid in zip(feats, s.candidate_track_ids):
                all_labels.append(1 if tid == s.gt_track_id else 0)
            all_probs.extend(touch_probs.tolist())

    accuracy = correct / total if total > 0 else 0.0

    # Compute AUC
    auc = 0.0
    if all_probs and all_labels:
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            pass

    per_action = {
        action: (per_action_correct[action], per_action_total[action])
        for action in sorted(per_action_total.keys())
    }

    return EvalResult(
        name="per-candidate",
        accuracy=accuracy,
        n_correct=correct,
        n_total=total,
        per_action=per_action,
        auc=auc,
    )


def compute_proximity_baseline(samples: list[ContactSample]) -> EvalResult:
    """Compute proximity baseline: always pick nearest player."""
    correct = 0
    total = len(samples)
    per_action_correct: dict[str, int] = defaultdict(int)
    per_action_total: dict[str, int] = defaultdict(int)

    for s in samples:
        is_correct = s.proximity_track_id == s.gt_track_id
        if is_correct:
            correct += 1
            per_action_correct[s.gt_action] += 1
        per_action_total[s.gt_action] += 1

    per_action = {
        action: (per_action_correct[action], per_action_total[action])
        for action in sorted(per_action_total.keys())
    }

    return EvalResult(
        name="proximity",
        accuracy=correct / total if total > 0 else 0.0,
        n_correct=correct,
        n_total=total,
        per_action=per_action,
    )


def print_results(results: list[EvalResult]) -> None:
    """Print comparison table."""
    console.print()

    # Summary table
    table = Table(title="Attribution Accuracy (LOO-CV)", show_header=True)
    table.add_column("Method")
    table.add_column("Accuracy", justify="right")
    table.add_column("Correct/Total", justify="right")
    table.add_column("AUC", justify="right")
    table.add_column("Delta", justify="right")

    baseline_acc = results[0].accuracy if results else 0.0

    for r in results:
        delta = r.accuracy - baseline_acc
        delta_str = f"{delta:+.1%}" if r.name != "proximity" else "—"
        table.add_row(
            r.name,
            f"{r.accuracy:.1%}",
            f"{r.n_correct}/{r.n_total}",
            f"{r.auc:.3f}" if r.auc > 0 else "—",
            delta_str,
        )

    console.print(table)

    # Per-action table
    actions = sorted(set(
        action for r in results for action in r.per_action
    ))

    if actions:
        action_table = Table(title="Per-Action Accuracy", show_header=True)
        action_table.add_column("Action")
        for r in results:
            action_table.add_column(r.name, justify="right")

        for action in actions:
            row = [action]
            for r in results:
                c, t = r.per_action.get(action, (0, 0))
                row.append(f"{c/t:.1%} ({c}/{t})" if t > 0 else "—")
            action_table.add_row(*row)

        console.print()
        console.print(action_table)


def print_feature_importance(samples: list[ContactSample], feature_mask: np.ndarray | None = None) -> None:
    """Train on all data and print feature importance."""
    all_X: list[np.ndarray] = []
    all_y: list[int] = []

    for s in samples:
        for feat, tid in zip(s.candidate_features, s.candidate_track_ids):
            f = feat if feature_mask is None else feat[feature_mask]
            all_X.append(f)
            all_y.append(1 if tid == s.gt_track_id else 0)

    X = np.stack(all_X)
    y = np.array(all_y, dtype=np.int32)

    clf, result = train_model(X, y)

    # Map importances to correct feature names
    names = FEATURE_NAMES if feature_mask is None else [
        FEATURE_NAMES[i] for i in range(len(FEATURE_NAMES)) if feature_mask[i]
    ]

    from sklearn.inspection import permutation_importance

    console.print("\n[bold]Feature Importance (permutation, 5 repeats)[/bold]")
    perm_imp = permutation_importance(clf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    pairs = sorted(
        zip(names, perm_imp.importances_mean),
        key=lambda x: x[1],
        reverse=True,
    )
    table = Table(show_header=True)
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for name, imp in pairs[:15]:
        table.add_row(name, f"{imp:.4f}")
    console.print(table)

    console.print(f"\nTrain AUC: {result.train_auc:.3f}")
    console.print(f"Positive: {result.num_positive}, Negative: {result.num_negative}")


def print_error_analysis(
    samples: list[ContactSample],
    proximity_result: EvalResult,
) -> None:
    """Analyze which proximity errors the model fixes and which it introduces."""
    # Identify proximity errors
    prox_errors = []
    prox_correct = []
    for s in samples:
        if s.proximity_track_id != s.gt_track_id:
            prox_errors.append(s)
        else:
            prox_correct.append(s)

    console.print(f"\n[bold]Error Analysis[/bold]")
    console.print(f"Proximity errors: {len(prox_errors)}/{len(samples)}")
    console.print(f"Proximity correct: {len(prox_correct)}/{len(samples)}")

    # Show action distribution of proximity errors
    error_actions: dict[str, int] = defaultdict(int)
    for s in prox_errors:
        error_actions[s.gt_action] += 1

    console.print("\nProximity errors by action:")
    for action in sorted(error_actions.keys()):
        console.print(f"  {action}: {error_actions[action]}")

    # Show distance rank distribution for GT track in proximity errors
    gt_ranks: dict[int, int] = defaultdict(int)
    for s in prox_errors:
        for rank, tid in enumerate(s.candidate_track_ids):
            if tid == s.gt_track_id:
                gt_ranks[rank] += 1
                break

    console.print("\nGT player's rank among candidates (in proximity errors):")
    for rank in sorted(gt_ranks.keys()):
        console.print(f"  Rank {rank}: {gt_ranks[rank]} ({gt_ranks[rank]/len(prox_errors)*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate per-candidate pose attribution")
    parser.add_argument("--rally", type=str, help="Evaluate single rally ID")
    parser.add_argument("--spatial-only", action="store_true", help="Spatial features only (no pose)")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--importance", action="store_true", help="Print feature importance")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    console.print(f"Loaded {len(rallies)} rallies with action GT")

    n_contacts = sum(len(r.gt_labels) for r in rallies)
    n_videos = len(set(r.video_id for r in rallies))
    console.print(f"  {n_contacts} GT contacts, {n_videos} videos")

    # Check if pose cache exists
    has_pose = False
    if not args.spatial_only:
        n_cached = sum(1 for r in rallies if load_pose_cache(r.rally_id) is not None)
        has_pose = n_cached > 0
        if has_pose:
            console.print(f"  Pose cache: {n_cached}/{len(rallies)} rallies")
        else:
            console.print("  [yellow]No pose cache found — using spatial-only mode[/yellow]")
            console.print("  [dim]Run extract_pose_cache.py first to enable pose features[/dim]")

    use_pose = has_pose and not args.spatial_only

    # Extract samples
    console.print("\nExtracting per-candidate features...")
    t0 = time.time()
    samples = extract_samples(rallies, use_pose=use_pose)
    console.print(f"  {len(samples)} evaluable contacts ({time.time() - t0:.1f}s)")

    if not samples:
        console.print("[red]No evaluable samples[/red]")
        sys.exit(1)

    # Proximity baseline
    proximity = compute_proximity_baseline(samples)

    results: list[EvalResult] = [proximity]

    if args.ablation:
        # Ablation: spatial-only (features 12-26)
        console.print("\nRunning ablation: spatial-only...")
        spatial_mask = np.zeros(NUM_FEATURES, dtype=bool)
        spatial_mask[POSE_FEATURE_COUNT:] = True
        spatial_result = run_loocv(samples, feature_mask=spatial_mask)
        spatial_result.name = "spatial-only"
        results.append(spatial_result)

        if use_pose:
            # Ablation: pose-only (features 0-11)
            console.print("Running ablation: pose-only...")
            pose_mask = np.zeros(NUM_FEATURES, dtype=bool)
            pose_mask[:POSE_FEATURE_COUNT] = True
            pose_result = run_loocv(samples, feature_mask=pose_mask)
            pose_result.name = "pose-only"
            results.append(pose_result)

            # Full model
            console.print("Running ablation: full (pose+spatial)...")
            full_result = run_loocv(samples)
            full_result.name = "pose+spatial"
            results.append(full_result)
    else:
        # Single run: either spatial-only or full model
        mode = "pose+spatial" if use_pose else "spatial-only"
        console.print(f"\nRunning LOO-CV ({mode})...")

        if not use_pose:
            # Mask out pose features
            spatial_mask = np.zeros(NUM_FEATURES, dtype=bool)
            spatial_mask[POSE_FEATURE_COUNT:] = True
            result = run_loocv(samples, feature_mask=spatial_mask)
            result.name = mode
        else:
            result = run_loocv(samples)
            result.name = mode

        results.append(result)

    print_results(results)
    print_error_analysis(samples, proximity)

    if args.importance:
        mask = None
        if not use_pose:
            mask = np.zeros(NUM_FEATURES, dtype=bool)
            mask[12:] = True
        print_feature_importance(samples, feature_mask=mask)


if __name__ == "__main__":
    main()
