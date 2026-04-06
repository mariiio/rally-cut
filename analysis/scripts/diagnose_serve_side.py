"""Diagnose serve side accuracy: trace why 51/231 rallies get wrong team labels.

For each rally, captures the serve detection path (position detection, serve pass,
phantom/synthetic, team overwrite) and categorizes side failures by root cause.

Usage:
    cd analysis
    uv run python scripts/diagnose_serve_side.py
    uv run python scripts/diagnose_serve_side.py --folds 0  # LOO-CV
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from eval_sequence_enriched import (  # noqa: E402
    RallyBundle,
    get_sequence_probs,
    prepare_rallies,
    train_fold_gbm,
    train_mstcn,
)

from rallycut.actions.trajectory_features import ACTION_TYPES  # noqa: E402
from rallycut.temporal.ms_tcn.model import MSTCN  # noqa: E402
from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    _find_server_by_position,
    _infer_serve_side,
    classify_rally_actions,
)
from rallycut.tracking.contact_classifier import ContactClassifier  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    detect_contacts,
)

logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()


# ---------------------------------------------------------------------------
# Per-rally diagnostic record
# ---------------------------------------------------------------------------


@dataclass
class ServeSideDiag:
    """Trace of serve side determination for one rally."""

    rally_id: str
    video_id: str

    # Ground truth
    gt_player_tid: int
    gt_team: int | None  # 0=near, 1=far in match_teams
    gt_side: str  # "near"/"far"/""

    # Position detection stage
    server_pos_tid: int
    server_pos_side: str  # "near"/"far"/""
    pos_conf: float

    # Serve detection stage
    serve_pass: int  # 0-4, -1 if no serve
    serve_index: int  # index into contacts

    # Serve action (final pipeline output)
    pred_is_synthetic: bool
    pred_player_tid: int
    pred_player_team: int | None  # team of pred player in match_teams
    pred_side: str  # final court_side (post-overwrite)
    was_overwritten: bool  # team overwrite changed court_side

    # Pre-overwrite side (reconstructed)
    pre_overwrite_side: str

    # Result
    side_correct: bool

    # Root cause (filled for failures)
    root_cause: str = ""


def categorize_failure(d: ServeSideDiag) -> str:
    """Assign root cause bucket to a wrong-side rally."""
    if d.serve_pass == -1:
        return "no_serve"

    # Correct player attributed but team mapping is wrong
    if (
        d.pred_player_tid >= 0
        and d.pred_player_tid == d.gt_player_tid
        and d.pred_player_team is not None
        and d.gt_team is not None
        and d.pred_player_team != d.gt_team
    ):
        return "team_mapping_wrong"

    # Synthetic serve
    if d.pred_is_synthetic:
        if d.pred_player_tid >= 0 and d.was_overwritten:
            # Position detection found a player, team overwrite applied
            return "pos_wrong_player"
        else:
            # No player (tid=-1), inferred side was wrong
            return "inference_wrong"

    # Real serve
    if d.was_overwritten:
        # Team overwrite applied — attributed player is on wrong team
        return "player_wrong_team"
    else:
        # No overwrite — raw heuristic court_side was wrong
        return "heuristic_side_wrong"


# ---------------------------------------------------------------------------
# Instrumented evaluation
# ---------------------------------------------------------------------------


def evaluate_rally_serve_side(
    bundle: RallyBundle,
    model: MSTCN,
    gbm: ContactClassifier,
    device: torch.device,
    calibrator: CourtCalibrator | None = None,
) -> ServeSideDiag | None:
    """Run the full pipeline and trace serve side determination."""
    probs = get_sequence_probs(model, bundle, device)

    # Contact detection (same as eval)
    contact_seq = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=ContactDetectionConfig(),
        frame_count=bundle.rally.frame_count,
        classifier=gbm,
        sequence_probs=probs,
    )

    # --- Diagnostic: call internal functions directly ---
    player_positions = contact_seq.player_positions or []
    server_pos_tid, server_pos_side, pos_conf = _find_server_by_position(
        player_positions,
        contact_seq.rally_start_frame,
        contact_seq.net_y,
        calibrator=calibrator,
    )

    ac = ActionClassifier(ActionClassifierConfig())
    ball_positions = contact_seq.ball_positions or None
    if contact_seq.contacts:
        serve_index, serve_pass = ac._find_serve_index(
            contact_seq.contacts,
            contact_seq.rally_start_frame,
            contact_seq.net_y,
            ball_positions=ball_positions,
            server_pos_tid=server_pos_tid,
        )
    else:
        serve_index, serve_pass = -1, -1

    # --- Full pipeline (same as eval_sequence_enriched.py) ---
    rally_actions = classify_rally_actions(
        contact_seq,
        match_team_assignments=bundle.match_teams,
        calibrator=calibrator,
    )

    # MS-TCN++ override (exempt serve)
    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if 0 <= frame < probs.shape[1]:
            cls = int(np.argmax(probs[1:, frame]))
            action.action_type = ActionType(ACTION_TYPES[cls])

    # Find GT and predicted serves
    gt_serves = [gt for gt in bundle.gt_labels if gt.action == "serve"]
    pred_serves = [
        a for a in rally_actions.actions if a.action_type == ActionType.SERVE
    ]

    if not gt_serves:
        return None

    gt_s = gt_serves[0]

    # GT side from team assignments
    gt_team: int | None = None
    gt_side = ""
    if bundle.match_teams and gt_s.player_track_id >= 0:
        gt_team = bundle.match_teams.get(gt_s.player_track_id)
        if gt_team is not None:
            gt_side = "near" if gt_team == 0 else "far"

    if not gt_side:
        return None  # Can't evaluate side without GT team assignment

    if not pred_serves:
        return ServeSideDiag(
            rally_id=bundle.rally.rally_id,
            video_id=bundle.rally.video_id,
            gt_player_tid=gt_s.player_track_id,
            gt_team=gt_team,
            gt_side=gt_side,
            server_pos_tid=server_pos_tid,
            server_pos_side=server_pos_side,
            pos_conf=pos_conf,
            serve_pass=-1,
            serve_index=-1,
            pred_is_synthetic=False,
            pred_player_tid=-1,
            pred_player_team=None,
            pred_side="",
            was_overwritten=False,
            pre_overwrite_side="",
            side_correct=False,
            root_cause="no_serve",
        )

    pred_s = pred_serves[0]

    # Determine if team overwrite happened
    pred_player_team: int | None = None
    was_overwritten = False
    if bundle.match_teams and pred_s.player_track_id >= 0:
        pred_player_team = bundle.match_teams.get(pred_s.player_track_id)
        if pred_player_team is not None:
            was_overwritten = True

    # Reconstruct pre-overwrite side
    if was_overwritten:
        if pred_s.is_synthetic:
            # Synthetic: would have been server_pos_side or inferred
            if server_pos_side:
                pre_overwrite = server_pos_side
            else:
                # Would have used _infer_serve_side or opposite of 1st contact
                first_contact = (
                    contact_seq.contacts[0] if contact_seq.contacts else None
                )
                inferred = None
                if first_contact:
                    inferred = _infer_serve_side(
                        first_contact, ball_positions, contact_seq.net_y,
                    )
                if inferred:
                    pre_overwrite = inferred
                elif first_contact and first_contact.court_side in ("near", "far"):
                    pre_overwrite = (
                        "far" if first_contact.court_side == "near" else "near"
                    )
                else:
                    pre_overwrite = "unknown"
        else:
            # Real serve: would have been contact.court_side or server_pos_side
            if 0 <= serve_index < len(contact_seq.contacts):
                pre_overwrite = contact_seq.contacts[serve_index].court_side
            else:
                pre_overwrite = "unknown"
    else:
        pre_overwrite = pred_s.court_side

    side_correct = gt_side == pred_s.court_side

    diag = ServeSideDiag(
        rally_id=bundle.rally.rally_id,
        video_id=bundle.rally.video_id,
        gt_player_tid=gt_s.player_track_id,
        gt_team=gt_team,
        gt_side=gt_side,
        server_pos_tid=server_pos_tid,
        server_pos_side=server_pos_side,
        pos_conf=pos_conf,
        serve_pass=serve_pass,
        serve_index=serve_index,
        pred_is_synthetic=pred_s.is_synthetic,
        pred_player_tid=pred_s.player_track_id,
        pred_player_team=pred_player_team,
        pred_side=pred_s.court_side,
        was_overwritten=was_overwritten,
        pre_overwrite_side=pre_overwrite,
        side_correct=side_correct,
    )

    if not side_correct:
        diag.root_cause = categorize_failure(diag)

    return diag


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

    # Load court calibrations per video for court-space server detection
    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal
    console.print(f"  Court calibrations: {len(calibrators)}/{len(video_ids)} videos")

    console.print("\n[bold]Serve Side Diagnostic[/bold]")
    console.print(
        f"  {n_folds}-fold CV, {len(video_ids)} videos, {len(bundles)} rallies"
    )

    device = torch.device(args.device)
    all_diags: list[ServeSideDiag] = []
    idx = 0

    for fold in range(n_folds):
        train_bundles = [
            b for b in bundles if fold_map[b.rally.video_id] != fold
        ]
        test_bundles = [
            b for b in bundles if fold_map[b.rally.video_id] == fold
        ]
        if not test_bundles or not train_bundles:
            continue

        fold_start = time.time()
        model = train_mstcn(train_bundles, test_bundles, args)
        gbm = train_fold_gbm(
            train_bundles, model, device, use_sequence_features=True,
        )

        fold_diags: list[ServeSideDiag] = []
        for bundle in test_bundles:
            cal = calibrators.get(bundle.rally.video_id)
            diag = evaluate_rally_serve_side(bundle, model, gbm, device, calibrator=cal)
            if diag is None:
                continue
            fold_diags.append(diag)
            all_diags.append(diag)
            idx += 1
            status = (
                "[green]OK[/green]"
                if diag.side_correct
                else f"[red]WRONG ({diag.root_cause})[/red]"
            )
            synth = "synth" if diag.pred_is_synthetic else "real "
            console.print(
                f"  [{idx}] {diag.rally_id[:8]} pass={diag.serve_pass} "
                f"{synth} pos={diag.server_pos_side or '-':4s} "
                f"conf={diag.pos_conf:.2f} → {status}"
            )

        fold_time = time.time() - fold_start
        n_correct = sum(1 for d in fold_diags if d.side_correct)
        console.print(
            f"  Fold [{fold + 1}/{n_folds}] rallies={len(test_bundles)} "
            f"evaluable={len(fold_diags)} correct={n_correct} "
            f"({fold_time:.0f}s)"
        )

    # --- Results ---
    total = len(all_diags)
    correct = sum(1 for d in all_diags if d.side_correct)
    failures = [d for d in all_diags if not d.side_correct]

    console.print(f"\n[bold]Serve Side Accuracy: {correct}/{total} "
                  f"({correct / total:.1%})[/bold]")
    console.print(f"  Failures: {len(failures)}")

    # --- Failure detail table ---
    if failures:
        table = Table(title=f"Wrong-Side Rallies ({len(failures)})")
        table.add_column("Rally", style="dim")
        table.add_column("Video", style="dim")
        table.add_column("Pass", justify="right")
        table.add_column("Synth?")
        table.add_column("GT Side")
        table.add_column("Pred Side", style="red")
        table.add_column("Pre-Overwrite")
        table.add_column("Overwritten?")
        table.add_column("Pos TID", justify="right")
        table.add_column("Pos Side")
        table.add_column("Pos Conf", justify="right")
        table.add_column("Pred TID", justify="right")
        table.add_column("Pred Team", justify="right")
        table.add_column("GT TID", justify="right")
        table.add_column("GT Team", justify="right")
        table.add_column("Root Cause", style="bold")

        for d in sorted(failures, key=lambda x: x.root_cause):
            table.add_row(
                d.rally_id[:8],
                d.video_id[:8],
                str(d.serve_pass),
                "Y" if d.pred_is_synthetic else "",
                d.gt_side,
                d.pred_side,
                d.pre_overwrite_side,
                "Y" if d.was_overwritten else "",
                str(d.server_pos_tid) if d.server_pos_tid >= 0 else "-",
                d.server_pos_side or "-",
                f"{d.pos_conf:.2f}",
                str(d.pred_player_tid) if d.pred_player_tid >= 0 else "-",
                str(d.pred_player_team) if d.pred_player_team is not None else "-",
                str(d.gt_player_tid) if d.gt_player_tid >= 0 else "-",
                str(d.gt_team) if d.gt_team is not None else "-",
                d.root_cause,
            )

        console.print(table)

    # --- Bucket summary ---
    bucket_counts = Counter(d.root_cause for d in failures)
    buckets = [
        "player_wrong_team", "pos_wrong_player", "inference_wrong",
        "heuristic_side_wrong", "team_mapping_wrong", "no_serve",
    ]
    bucket_table = Table(title="Root Cause Summary")
    bucket_table.add_column("Root Cause")
    bucket_table.add_column("Count", justify="right")
    bucket_table.add_column("% of Failures", justify="right")
    bucket_table.add_column("% of Total", justify="right")
    for b in buckets:
        c = bucket_counts.get(b, 0)
        if c > 0:
            bucket_table.add_row(
                b,
                str(c),
                f"{c / len(failures):.1%}",
                f"{c / total:.1%}",
            )
    # Any uncategorized
    other = sum(c for b, c in bucket_counts.items() if b not in buckets and b)
    if other:
        bucket_table.add_row("other", str(other),
                             f"{other / len(failures):.1%}",
                             f"{other / total:.1%}")
    console.print(bucket_table)

    # --- Cross-tab: serve_pass × correct/wrong ---
    xtab: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for d in all_diags:
        key = f"synth" if d.pred_is_synthetic else f"pass_{d.serve_pass}"
        xtab[key]["correct" if d.side_correct else "wrong"] += 1

    xtab_table = Table(title="Serve Type × Side Accuracy")
    xtab_table.add_column("Serve Type")
    xtab_table.add_column("Correct", justify="right")
    xtab_table.add_column("Wrong", justify="right")
    xtab_table.add_column("Accuracy", justify="right")
    for key in sorted(xtab.keys()):
        c = xtab[key]["correct"]
        w = xtab[key]["wrong"]
        xtab_table.add_row(key, str(c), str(w), f"{c / (c + w):.1%}")
    console.print(xtab_table)

    # --- Position detection analysis ---
    console.print("\n[bold]Position Detection Analysis[/bold]")
    has_pos = [d for d in all_diags if d.server_pos_tid >= 0]
    no_pos = [d for d in all_diags if d.server_pos_tid < 0]
    console.print(f"  With position detection: {len(has_pos)} "
                  f"(side correct: {sum(1 for d in has_pos if d.side_correct)}/"
                  f"{len(has_pos)})")
    console.print(f"  Without position detection: {len(no_pos)} "
                  f"(side correct: {sum(1 for d in no_pos if d.side_correct)}/"
                  f"{len(no_pos)})")

    # Position detection: correct side vs wrong side
    if has_pos:
        pos_side_matches_gt = sum(
            1 for d in has_pos
            if d.server_pos_side == d.gt_side
        )
        console.print(f"  Position side matches GT: {pos_side_matches_gt}/{len(has_pos)} "
                      f"({pos_side_matches_gt / len(has_pos):.1%})")

    # Video-level clustering
    console.print("\n[bold]Failures by Video[/bold]")
    vid_failures: dict[str, int] = Counter(d.video_id[:8] for d in failures)
    vid_totals: dict[str, int] = Counter(d.video_id[:8] for d in all_diags)
    for vid in sorted(vid_failures.keys(), key=lambda v: -vid_failures[v]):
        f_count = vid_failures[vid]
        t_count = vid_totals[vid]
        console.print(f"  {vid}: {f_count}/{t_count} wrong "
                      f"({f_count / t_count:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose serve side accuracy")
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
        "--device",
        type=str,
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
