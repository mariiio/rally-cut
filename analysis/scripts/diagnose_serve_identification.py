"""Session 6 — serve identification bucket diagnostic.

For every GT-labeled rally with a serve, run the current pipeline and bucket
the prediction into one of:
    correct                — predicted player == GT serve player
    right_team_wrong_player — pred player on GT team, different player
    wrong_team             — pred player on opposite team
    late_tracked           — GT server first appears AFTER serve frame
    untracked              — GT server never appears in player_positions
    other                  — anything else (no_serve, missing teams, etc.)

Also computes a counterfactual: if we forced the predicted serving team to
match the previous rally's GT point winner, how many wrong predictions flip
to a correct *team*?

Gate (Session 6, playbook): right_team_wrong_player must be ≥ 40% of wrong
cases for the rotation-prior design to be the right lever. Below that, the
fix is upstream (rally-winner / team assignment), not here.

Usage:
    cd analysis
    uv run python scripts/diagnose_serve_identification.py
    uv run python scripts/diagnose_serve_identification.py --folds 5
"""

from __future__ import annotations

import argparse
import csv
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

from diagnose_serve_side import evaluate_rally_serve_side  # noqa: E402
from eval_sequence_enriched import (  # noqa: E402
    RallyBundle,
    prepare_rallies,
    train_fold_gbm,
    train_mstcn,
)

from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: E402

logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()


# ---------------------------------------------------------------------------
# Per-rally bucket record
# ---------------------------------------------------------------------------


@dataclass
class ServeIdentRow:
    rally_id: str
    video_id: str
    start_ms: int

    gt_player_tid: int
    gt_team: int | None         # 0=near, 1=far
    pred_player_tid: int
    pred_player_team: int | None

    bucket: str                 # see module docstring
    player_correct: bool

    # Counterfactual
    prev_winner_team: str | None  # 'A'/'B' from gt_point_winner
    pred_team_label: str | None   # 'A'/'B' derived from pred_player_team
    gt_team_label: str | None     # 'A'/'B' derived from gt_team
    counterfactual_team_correct: bool  # team match if forced to prev winner


# Convention: team 0 = "A" (near), team 1 = "B" (far). gt_serving_team /
# gt_point_winner are stored as 'A'/'B' in DB; we treat A↔near, B↔far. This is
# only used for the counterfactual chain — sign flips wash out per video.
TEAM_INT_TO_LABEL = {0: "A", 1: "B"}


def _classify_tracking(
    bundle: RallyBundle, gt_player_tid: int, serve_frame: int,
) -> str | None:
    """Return 'untracked' or 'late_tracked' or None if GT player tracked in time."""
    if gt_player_tid < 0:
        return "untracked"
    pos_frames = [
        p.frame_number for p in bundle.player_positions if p.track_id == gt_player_tid
    ]
    if not pos_frames:
        return "untracked"
    earliest = min(pos_frames)
    # 'Late' = first appears strictly after the serve frame.
    if earliest > serve_frame:
        return "late_tracked"
    return None


def _bucket_row(
    bundle: RallyBundle,
    gt_player_tid: int,
    gt_team: int | None,
    pred_player_tid: int,
    pred_player_team: int | None,
    serve_frame: int,
) -> tuple[str, bool]:
    """Return (bucket, player_correct)."""
    player_correct = pred_player_tid >= 0 and pred_player_tid == gt_player_tid
    if player_correct:
        return "correct", True

    # Tracking-of-GT-server check first (this is independent of pred quality)
    tracking_bucket = _classify_tracking(bundle, gt_player_tid, serve_frame)
    if tracking_bucket is not None:
        return tracking_bucket, False

    if pred_player_tid < 0 or pred_player_team is None or gt_team is None:
        return "other", False

    if pred_player_team == gt_team:
        return "right_team_wrong_player", False
    return "wrong_team", False


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(bundles: list[RallyBundle], args: argparse.Namespace) -> None:
    videos: dict[str, list[RallyBundle]] = defaultdict(list)
    for b in bundles:
        videos[b.rally.video_id].append(b)
    for vid in videos:
        videos[vid].sort(key=lambda b: b.rally.start_ms)

    video_ids = sorted(videos.keys())
    n_folds = args.folds if args.folds > 0 else len(video_ids)
    n_folds = min(n_folds, len(video_ids))
    fold_map = {vid: i % n_folds for i, vid in enumerate(video_ids)}

    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal
    console.print(
        f"  Court calibrations: {len(calibrators)}/{len(video_ids)} videos"
    )

    console.print("\n[bold]Session 6 — Serve Identification Diagnostic[/bold]")
    console.print(
        f"  {n_folds}-fold CV, {len(video_ids)} videos, {len(bundles)} rallies"
    )

    device = torch.device(args.device)
    rows: list[ServeIdentRow] = []
    idx = 0

    for fold in range(n_folds):
        train_bundles = [b for b in bundles if fold_map[b.rally.video_id] != fold]
        test_bundles = [b for b in bundles if fold_map[b.rally.video_id] == fold]
        if not test_bundles or not train_bundles:
            continue

        fold_start = time.time()
        model = train_mstcn(train_bundles, test_bundles, args)
        gbm = train_fold_gbm(
            train_bundles, model, device, use_sequence_features=True,
        )

        for bundle in test_bundles:
            cal = calibrators.get(bundle.rally.video_id)
            diag = evaluate_rally_serve_side(bundle, model, gbm, device, calibrator=cal)
            if diag is None:
                continue

            # GT serve frame
            gt_serves = [gt for gt in bundle.gt_labels if gt.action == "serve"]
            serve_frame = gt_serves[0].frame if gt_serves else 0

            bucket, player_correct = _bucket_row(
                bundle,
                gt_player_tid=diag.gt_player_tid,
                gt_team=diag.gt_team,
                pred_player_tid=diag.pred_player_tid,
                pred_player_team=diag.pred_player_team,
                serve_frame=serve_frame,
            )

            row = ServeIdentRow(
                rally_id=bundle.rally.rally_id,
                video_id=bundle.rally.video_id,
                start_ms=bundle.rally.start_ms,
                gt_player_tid=diag.gt_player_tid,
                gt_team=diag.gt_team,
                pred_player_tid=diag.pred_player_tid,
                pred_player_team=diag.pred_player_team,
                bucket=bucket,
                player_correct=player_correct,
                prev_winner_team=None,         # filled in chain pass
                pred_team_label=(
                    TEAM_INT_TO_LABEL.get(diag.pred_player_team)
                    if diag.pred_player_team is not None else None
                ),
                gt_team_label=(
                    TEAM_INT_TO_LABEL.get(diag.gt_team)
                    if diag.gt_team is not None else None
                ),
                counterfactual_team_correct=False,
            )
            rows.append(row)
            idx += 1
            tag = "[green]✓[/green]" if player_correct else f"[red]{bucket}[/red]"
            console.print(
                f"  [{idx}] {row.rally_id[:8]} gt_tid={row.gt_player_tid} "
                f"pred_tid={row.pred_player_tid} → {tag}"
            )

        fold_time = time.time() - fold_start
        console.print(
            f"  Fold [{fold + 1}/{n_folds}] rallies={len(test_bundles)} "
            f"evaluable={sum(1 for r in rows if r.video_id in {b.rally.video_id for b in test_bundles})} "  # noqa: E501
            f"({fold_time:.0f}s)"
        )

    # ----- Counterfactual chain (per video, ordered by start_ms) -----
    by_video: dict[str, list[ServeIdentRow]] = defaultdict(list)
    for r in rows:
        by_video[r.video_id].append(r)
    for vid, vrows in by_video.items():
        vrows.sort(key=lambda r: r.start_ms)
        # Build a parallel list of GT winner labels indexed off bundles
        rally_to_winner = {
            b.rally.rally_id: b.rally.gt_point_winner for b in videos[vid]
        }
        for i, r in enumerate(vrows):
            if i == 0:
                continue
            prev_winner = rally_to_winner.get(vrows[i - 1].rally_id)
            r.prev_winner_team = prev_winner
            if prev_winner and r.gt_team_label:
                r.counterfactual_team_correct = (prev_winner == r.gt_team_label)

    _report(rows, args)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _report(rows: list[ServeIdentRow], args: argparse.Namespace) -> None:
    total = len(rows)
    if total == 0:
        console.print("[red]No evaluable rallies.[/red]")
        return

    correct = sum(1 for r in rows if r.player_correct)
    wrong = [r for r in rows if not r.player_correct]
    console.print(
        f"\n[bold]Player-level serve_id accuracy: {correct}/{total} "
        f"({correct / total:.1%})[/bold]"
    )
    console.print(f"  Wrong: {len(wrong)}")

    bucket_order = [
        "correct",
        "right_team_wrong_player",
        "wrong_team",
        "late_tracked",
        "untracked",
        "other",
    ]
    bcounts = Counter(r.bucket for r in rows)

    bt = Table(title="Bucket Distribution")
    bt.add_column("Bucket")
    bt.add_column("Count", justify="right")
    bt.add_column("% of Total", justify="right")
    bt.add_column("% of Wrong", justify="right")
    for b in bucket_order:
        n = bcounts.get(b, 0)
        pct_total = n / total
        pct_wrong = (n / len(wrong)) if (wrong and b != "correct") else 0.0
        bt.add_row(
            b,
            str(n),
            f"{pct_total:.1%}",
            f"{pct_wrong:.1%}" if b != "correct" else "—",
        )
    console.print(bt)

    # ---- Gate ----
    rtw = bcounts.get("right_team_wrong_player", 0)
    rtw_share = (rtw / len(wrong)) if wrong else 0.0
    gate_pass = rtw_share >= 0.40
    color = "green" if gate_pass else "red"
    verdict = "PASS" if gate_pass else "FAIL"
    console.print(
        f"\n[bold {color}]Session 6 Gate ({verdict}): "
        f"right_team_wrong_player = {rtw}/{len(wrong)} = {rtw_share:.1%} "
        f"(threshold ≥ 40%)[/bold {color}]"
    )
    if not gate_pass:
        console.print(
            "  → STOP. The rotation-prior design will not move serve_id "
            "meaningfully. Recommend investigating upstream "
            "(rally-winner prediction or team assignment) instead."
        )

    # ---- Counterfactual: previous-winner-serves rule ----
    eligible = [
        r for r in rows
        if r.prev_winner_team is not None and r.gt_team_label is not None
    ]
    if eligible:
        gt_team_match = sum(
            1 for r in eligible if r.prev_winner_team == r.gt_team_label
        )
        # How many wrong rows would have been *team*-correct under the rule?
        wrong_eligible = [r for r in eligible if not r.player_correct]
        # The rule fixes a wrong row when prev_winner == gt_team AND
        # current pred_team != gt_team (otherwise team was already right).
        rule_helps = sum(
            1 for r in wrong_eligible
            if r.prev_winner_team == r.gt_team_label
            and r.pred_team_label != r.gt_team_label
        )
        rule_hurts = sum(
            1 for r in eligible
            if r.player_correct
            and r.pred_team_label == r.gt_team_label
            and r.prev_winner_team != r.gt_team_label
        )
        console.print("\n[bold]Counterfactual: 'previous winner's team serves'[/bold]")
        console.print(
            f"  GT prior accuracy (rule vs GT team): "
            f"{gt_team_match}/{len(eligible)} ({gt_team_match / len(eligible):.1%})"
        )
        console.print(
            f"  Wrong rows the rule would FIX (team-level): {rule_helps}"
        )
        console.print(
            f"  Currently-correct rows the rule would BREAK (team-level): "
            f"{rule_hurts}"
        )

    # ---- Per-video table ----
    vt = Table(title="Per-Video Buckets")
    vt.add_column("Video")
    vt.add_column("N", justify="right")
    vt.add_column("✓", justify="right")
    vt.add_column("RtWp", justify="right")
    vt.add_column("WT", justify="right")
    vt.add_column("Late", justify="right")
    vt.add_column("Untrk", justify="right")
    by_video: dict[str, list[ServeIdentRow]] = defaultdict(list)
    for r in rows:
        by_video[r.video_id].append(r)
    for vid in sorted(by_video.keys()):
        vr = by_video[vid]
        vc = Counter(r.bucket for r in vr)
        vt.add_row(
            vid[:8],
            str(len(vr)),
            str(vc.get("correct", 0)),
            str(vc.get("right_team_wrong_player", 0)),
            str(vc.get("wrong_team", 0)),
            str(vc.get("late_tracked", 0)),
            str(vc.get("untracked", 0)),
        )
    console.print(vt)

    # ---- CSV dump ----
    out_dir = Path("outputs/serve_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "session6_buckets.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rally_id", "video_id", "start_ms",
            "gt_player_tid", "gt_team", "pred_player_tid", "pred_player_team",
            "bucket", "player_correct",
            "prev_winner_team", "pred_team_label", "gt_team_label",
            "counterfactual_team_correct",
        ])
        for r in rows:
            w.writerow([
                r.rally_id, r.video_id, r.start_ms,
                r.gt_player_tid, r.gt_team, r.pred_player_tid, r.pred_player_team,
                r.bucket, int(r.player_correct),
                r.prev_winner_team or "", r.pred_team_label or "",
                r.gt_team_label or "", int(r.counterfactual_team_correct),
            ])
    console.print(f"\n[dim]Wrote {csv_path}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Session 6 serve identification bucket diagnostic"
    )
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

    run(bundles, args)


if __name__ == "__main__":
    main()
