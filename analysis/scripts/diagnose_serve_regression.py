"""Diagnose serve detection: trace each GT serve through the pipeline.

For each GT serve, captures:
  - Heuristic label from classify_rally_actions() (before MS-TCN++ override)
  - MS-TCN++ probability at the serve frame
  - Whether a synthetic serve was generated (for off-screen/occluded serves)
  - Final matched label and correctness

Usage:
    cd analysis
    uv run python scripts/diagnose_serve_regression.py
    uv run python scripts/diagnose_serve_regression.py --folds 0  # LOO-CV
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from eval_action_detection import (  # noqa: E402
    MatchResult,
    match_contacts,
)
from eval_sequence_enriched import (
    RallyBundle,
    get_sequence_probs,
    prepare_rallies,
    train_fold_gbm,
    train_mstcn,
)

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.temporal.ms_tcn.model import MSTCN
from rallycut.tracking.action_classifier import (
    ActionType,
    classify_rally_actions,
)
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)

logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()


# ---------------------------------------------------------------------------
# Per-serve diagnostic record
# ---------------------------------------------------------------------------


@dataclass
class ServeDiagnostic:
    """Trace of one GT serve through the pipeline."""
    rally_id: str
    video_id: str
    gt_frame: int

    # Stage A: heuristic label (from classify_rally_actions, before override)
    heuristic_label: str = "unmatched"
    is_phantom: bool = False  # synthetic serve was inserted
    is_synthetic: bool = False  # this action itself is synthetic

    # Stage B: MS-TCN++ probabilities at this frame
    mstcn_serve_prob: float = 0.0
    mstcn_argmax_class: str = ""
    mstcn_argmax_prob: float = 0.0
    mstcn_probs: dict[str, float] = field(default_factory=dict)

    # Stage C: final result
    final_label: str = "unmatched"
    matched: bool = False  # GT serve matched to a prediction
    correct: bool = False  # final label == "serve"

    # Synthetic serve info
    has_synthetic_serve: bool = False  # rally has a synthetic serve
    synthetic_serve_frame: int = -1  # frame of synthetic serve
    synthetic_frame_gap: int = -1  # |gt_frame - synthetic_frame|

    # Categorization
    category: str = ""  # filled after analysis


# ---------------------------------------------------------------------------
# Instrumented evaluation
# ---------------------------------------------------------------------------


def evaluate_rally_instrumented(
    bundle: RallyBundle,
    model: MSTCN,
    gbm: ContactClassifier,
    device: torch.device,
) -> tuple[list[MatchResult], list[dict], list[ServeDiagnostic]]:
    """Run the full pipeline with serve-level instrumentation."""
    probs = get_sequence_probs(model, bundle, device)

    # Contact detection with sequence probs (27-dim GBM)
    contact_seq = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=ContactDetectionConfig(),
        frame_count=bundle.rally.frame_count,
        classifier=gbm,
        sequence_probs=probs,
    )

    # Stage A: heuristic classification
    rally_actions = classify_rally_actions(
        contact_seq,
        match_team_assignments=bundle.match_teams,
    )

    # Capture heuristic labels BEFORE override
    heuristic_map: dict[int, tuple[str, bool]] = {}  # frame → (label, is_synthetic)
    synthetic_serve_frame = -1
    has_phantom = False
    for a in rally_actions.actions:
        heuristic_map[a.frame] = (a.action_type.value, a.is_synthetic)
        if a.is_synthetic and a.action_type == ActionType.SERVE:
            has_phantom = True
            synthetic_serve_frame = a.frame

    # Stage B: hybrid MS-TCN++ override — exempt serve only.
    # Serve uses structural constraints (first action, baseline position) the
    # model can't learn. Receive/block benefit from the model.
    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if 0 <= frame < probs.shape[1]:
            cls = int(np.argmax(probs[1:, frame]))
            action.action_type = ActionType(ACTION_TYPES[cls])

    # Convert to match format (excluding synthetic)
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

    # Stage C: match GT to predictions
    matches, unmatched = match_contacts(
        bundle.gt_labels, preds, tolerance=5,
        team_assignments=bundle.match_teams,
    )

    # Build per-GT-serve diagnostics
    diagnostics: list[ServeDiagnostic] = []
    for mr in matches:
        if mr.gt_action != "serve":
            continue

        diag = ServeDiagnostic(
            rally_id=bundle.rally.rally_id[:8],
            video_id=bundle.rally.video_id[:8],
            gt_frame=mr.gt_frame,
        )

        # Find closest heuristic action to GT frame
        best_frame = None
        best_dist = 999
        for f in heuristic_map:
            dist = abs(f - mr.gt_frame)
            if dist < best_dist and dist <= 5:
                best_dist = dist
                best_frame = f

        if best_frame is not None:
            h_label, h_synth = heuristic_map[best_frame]
            diag.heuristic_label = h_label
            diag.is_synthetic = h_synth
            diag.is_phantom = has_phantom and not h_synth
        else:
            diag.heuristic_label = "no_match"

        # Synthetic serve info
        diag.has_synthetic_serve = has_phantom
        if has_phantom:
            diag.synthetic_serve_frame = synthetic_serve_frame
            diag.synthetic_frame_gap = abs(mr.gt_frame - synthetic_serve_frame)

        # MS-TCN++ probs at GT frame
        if 0 <= mr.gt_frame < probs.shape[1]:
            frame_probs = probs[1:, mr.gt_frame]  # skip background
            diag.mstcn_serve_prob = float(probs[1, mr.gt_frame])  # serve = index 1
            argmax_idx = int(np.argmax(frame_probs))
            diag.mstcn_argmax_class = ACTION_TYPES[argmax_idx]
            diag.mstcn_argmax_prob = float(frame_probs[argmax_idx])
            diag.mstcn_probs = {
                ACTION_TYPES[i]: float(frame_probs[i])
                for i in range(len(ACTION_TYPES))
            }

        # Final result
        if mr.pred_frame is not None:
            diag.matched = True
            diag.final_label = mr.pred_action or "unknown"
            diag.correct = mr.pred_action == "serve"
        else:
            diag.matched = False
            diag.final_label = "unmatched"

        # Categorize
        if diag.correct:
            diag.category = "correct"
        elif not diag.matched:
            diag.category = "unmatched_FN"
        elif diag.heuristic_label == "serve" and diag.final_label != "serve":
            diag.category = "override_killed"  # Mechanism B: heuristic correct, override wrong
        elif diag.heuristic_label == "receive" and diag.is_phantom:
            diag.category = "phantom_receive"  # Mechanism A: phantom detection
        elif diag.heuristic_label != "serve":
            diag.category = "heuristic_wrong"  # Neither identified it as serve
        else:
            diag.category = "other"

        diagnostics.append(diag)

    return matches, unmatched, diagnostics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_diagnostic(bundles: list[RallyBundle], args: argparse.Namespace) -> None:
    videos: dict[str, list[RallyBundle]] = defaultdict(list)
    for b in bundles:
        videos[b.rally.video_id].append(b)

    video_ids = sorted(videos.keys())
    n_folds = args.folds if args.folds > 0 else len(video_ids)
    n_folds = min(n_folds, len(video_ids))
    fold_map = {vid: i % n_folds for i, vid in enumerate(video_ids)}

    console.print("\n[bold]Serve Regression Diagnostic[/bold]")
    console.print(f"  {n_folds}-fold CV, {len(video_ids)} videos, {len(bundles)} rallies")

    device = torch.device(args.device)
    all_diagnostics: list[ServeDiagnostic] = []
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    for fold in range(n_folds):
        train_bundles = [b for b in bundles if fold_map[b.rally.video_id] != fold]
        test_bundles = [b for b in bundles if fold_map[b.rally.video_id] == fold]
        if not test_bundles or not train_bundles:
            continue

        fold_start = time.time()

        # Train MS-TCN++ and 27-dim GBM for this fold
        model = train_mstcn(train_bundles, test_bundles, args)
        gbm = train_fold_gbm(train_bundles, model, device, use_sequence_features=True)

        # Evaluate each test rally
        fold_diags: list[ServeDiagnostic] = []
        for bundle in test_bundles:
            matches, unmatched, diags = evaluate_rally_instrumented(
                bundle, model, gbm, device,
            )
            all_matches.extend(matches)
            all_unmatched.extend(unmatched)
            fold_diags.extend(diags)
            all_diagnostics.extend(diags)

        fold_time = time.time() - fold_start
        n_serves = len(fold_diags)
        n_correct = sum(1 for d in fold_diags if d.correct)
        console.print(
            f"  [{fold + 1}/{n_folds}] rallies={len(test_bundles)} "
            f"serves={n_serves} correct={n_correct} "
            f"({fold_time:.0f}s)"
        )

    # --- Per-serve table ---
    console.print(f"\n[bold]Per-Serve Diagnostic ({len(all_diagnostics)} GT serves)[/bold]")

    # Show only failures
    failures = [d for d in all_diagnostics if not d.correct]

    if failures:
        table = Table(title=f"Serve Failures ({len(failures)})")
        table.add_column("Rally", style="dim")
        table.add_column("GT Frame", justify="right")
        table.add_column("Heuristic", style="cyan")
        table.add_column("Phantom?")
        table.add_column("MS-TCN++ Serve P", justify="right")
        table.add_column("MS-TCN++ Argmax", style="yellow")
        table.add_column("Argmax P", justify="right")
        table.add_column("Final", style="red")
        table.add_column("Category", style="bold")

        for d in sorted(failures, key=lambda x: x.category):
            phantom = "YES" if d.is_phantom else ""
            table.add_row(
                d.rally_id,
                str(d.gt_frame),
                d.heuristic_label,
                phantom,
                f"{d.mstcn_serve_prob:.3f}",
                d.mstcn_argmax_class,
                f"{d.mstcn_argmax_prob:.3f}",
                d.final_label,
                d.category,
            )

        console.print(table)

    # --- Aggregate summary ---
    console.print("\n[bold]Summary[/bold]")
    total = len(all_diagnostics)
    correct = sum(1 for d in all_diagnostics if d.correct)
    console.print(f"  Total GT serves: {total}")
    console.print(f"  Correctly classified: {correct} ({correct/total:.1%})" if total else "  No serves")

    # Category breakdown
    cats = Counter(d.category for d in all_diagnostics)
    console.print("\n  Category breakdown:")
    for cat in ["correct", "override_killed", "phantom_receive", "heuristic_wrong", "unmatched_FN", "other"]:
        if cats[cat]:
            console.print(f"    {cat:20s}: {cats[cat]:3d} ({cats[cat]/total:.1%})")

    # Mechanism attribution
    n_override = cats["override_killed"]
    n_phantom = cats["phantom_receive"]
    n_heuristic_wrong = cats["heuristic_wrong"]
    console.print("\n  [bold]Root cause attribution:[/bold]")
    console.print(f"    Mechanism B (hybrid override killed correct heuristic): {n_override}")
    console.print(f"    Mechanism A (phantom → receive): {n_phantom}")
    console.print(f"    Heuristic wrong (not serve, not phantom): {n_heuristic_wrong}")
    console.print(f"    Unmatched (contact FN): {cats['unmatched_FN']}")

    # Heuristic-only accuracy (what if we removed the override?)
    heuristic_correct = sum(
        1 for d in all_diagnostics
        if d.heuristic_label == "serve" and not d.is_synthetic
    )
    console.print("\n  [bold]Counterfactual:[/bold]")
    console.print(f"    Heuristic serve recall (no override): {heuristic_correct}/{total}"
                  f" ({heuristic_correct/total:.1%})" if total else "")
    console.print(f"    MS-TCN++ serve at GT frames: {sum(1 for d in all_diagnostics if d.mstcn_argmax_class == 'serve')}/{total}"
                  f" ({sum(1 for d in all_diagnostics if d.mstcn_argmax_class == 'serve')/total:.1%})" if total else "")
    console.print(f"    Current (hybrid): {correct}/{total}"
                  f" ({correct/total:.1%})" if total else "")

    # Synthetic serve analysis
    unmatched = [d for d in all_diagnostics if not d.matched]
    with_synth = [d for d in unmatched if d.has_synthetic_serve]
    console.print("\n  [bold]Synthetic serve analysis (unmatched GT serves):[/bold]")
    console.print(f"    Unmatched GT serves: {len(unmatched)}")
    console.print(f"    Rally has synthetic serve: {len(with_synth)} ({len(with_synth)/max(len(unmatched),1):.0%})")
    if with_synth:
        gaps = [d.synthetic_frame_gap for d in with_synth]
        console.print(f"    Frame gap |GT - synthetic|: median={sorted(gaps)[len(gaps)//2]}, "
                      f"mean={sum(gaps)/len(gaps):.0f}, max={max(gaps)}, "
                      f"within ±5: {sum(1 for g in gaps if g <= 5)}, "
                      f"within ±15: {sum(1 for g in gaps if g <= 15)}")
    without_synth = [d for d in unmatched if not d.has_synthetic_serve]
    console.print(f"    No synthetic serve: {len(without_synth)} (pipeline treats first contact as real serve)")

    # Non-serve actions in rallies with unmatched GT serves (what did the pipeline call the serve?)
    console.print("\n  [bold]What pipeline did with unmatched GT serves:[/bold]")
    has_synth_count = len(with_synth)
    no_synth_count = len(without_synth)
    console.print(f"    Phantom → synthetic serve: {has_synth_count} (serve exists but filtered from eval)")
    console.print(f"    First contact = serve: {no_synth_count} (pipeline found serve via passes 0-2)")

    # Final label distribution for failures
    if failures:
        final_dist = Counter(d.final_label for d in failures)
        console.print(f"\n  Final labels for {len(failures)} failures: {dict(final_dist)}")

        # Heuristic label distribution for failures
        heur_dist = Counter(d.heuristic_label for d in failures)
        console.print(f"  Heuristic labels for failures: {dict(heur_dist)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose serve regression")
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

    run_diagnostic(bundles, args)


if __name__ == "__main__":
    main()
