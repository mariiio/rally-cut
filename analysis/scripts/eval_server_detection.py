"""Train and evaluate the GBM server identification classifier.

For each rally with action GT:
  1. Build PlayerPosition list (with injected keypoints)
  2. Build BallPosition list
  3. Load court calibration for the video (when available)
  4. Extract per-player ServerFeatures from the first 60 frames
  5. Label sample y=1 for the GT server track_id, y=0 for the rest

Then run leave-one-rally-out cross-validation. Report:
  - Rally-level server identification accuracy
  - Rally-level serve side accuracy (target: >90%)
  - Stratified by server visibility (tracked at start / late) and court calibration
  - Feature importance
  - Heuristic baseline comparison (existing _identify_server)
  - Failures

Finally, train on all data and save weights to:
  weights/server_classifier/server_classifier.pkl

Usage:
    cd analysis
    uv run python scripts/eval_server_detection.py
    uv run python scripts/eval_server_detection.py --no-save  # Skip saving model
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))

from eval_action_detection import (  # noqa: E402
    _build_player_positions,
    load_rallies_with_action_gt,
)

from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.tracking.db import load_court_calibration  # noqa: E402
from rallycut.tracking.action_classifier import _identify_server  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.server_classifier import (  # noqa: E402
    DEFAULT_WINDOW_FRAMES,
    ServerClassifier,
    extract_server_features,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)
console = Console()


def _parse_ball(ball_json: list[dict]) -> list[BallPosition]:
    out: list[BallPosition] = []
    for b in ball_json:
        out.append(BallPosition(
            frame_number=b.get("frameNumber", 0),
            x=b.get("x", 0.0),
            y=b.get("y", 0.0),
            confidence=b.get("confidence", 1.0),
        ))
    return out


def _gt_serve(rally) -> tuple[int, int] | None:
    """Extract (server_track_id, serve_frame) from a rally's gt_labels.

    Returns None if no serve label or track_id is -1.
    """
    for gt in rally.gt_labels:
        if gt.action == "serve" and gt.player_track_id >= 0:
            return gt.player_track_id, gt.frame
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving trained model")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_FRAMES,
                        help=f"Feature window in frames (default: {DEFAULT_WINDOW_FRAMES})")
    parser.add_argument("--rally", type=str, default=None,
                        help="Evaluate a single rally (debug)")
    args = parser.parse_args()

    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt(args.rally)
    console.print(f"  Loaded {len(rallies)} rallies")

    # Load court calibrations per video
    video_ids = sorted({r.video_id for r in rallies})
    calibrators: dict[str, CourtCalibrator | None] = {}
    n_calibrated = 0
    for vid in video_ids:
        try:
            corners = load_court_calibration(vid)
        except Exception:
            corners = None
        cal: CourtCalibrator | None = None
        if corners and len(corners) == 4:
            try:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])
                if not cal.is_calibrated:
                    cal = None
            except Exception:
                cal = None
        calibrators[vid] = cal
        if cal is not None:
            n_calibrated += 1
    console.print(f"  Court calibrations: {n_calibrated}/{len(video_ids)} videos")

    # ----- Feature extraction -----
    console.print("[bold]Extracting features...[/bold]")
    x_list: list[np.ndarray] = []
    y_list: list[int] = []
    rally_id_list: list[str] = []
    track_id_list: list[int] = []
    foot_y_list: list[float] = []

    gt_server_tids: dict[str, int] = {}
    gt_serve_sides: dict[str, str] = {}
    net_ys: dict[str, float] = {}
    rally_meta: dict[str, dict] = {}  # for stratification

    n_skip_no_gt = 0
    n_skip_no_data = 0
    n_skip_server_not_tracked = 0

    t0 = time.time()
    for i, rally in enumerate(rallies):
        if not rally.positions_json or not rally.ball_positions_json:
            n_skip_no_data += 1
            continue
        gt = _gt_serve(rally)
        if gt is None:
            n_skip_no_gt += 1
            continue
        gt_tid, _gt_serve_frame = gt

        net_y = rally.court_split_y if rally.court_split_y is not None else 0.5

        player_positions = _build_player_positions(
            rally.positions_json, rally.rally_id,
        )
        ball_positions = _parse_ball(rally.ball_positions_json)

        if not player_positions:
            n_skip_no_data += 1
            continue

        # Rally start frame: smallest frame number among player positions
        start_frame = min(p.frame_number for p in player_positions)

        cal = calibrators.get(rally.video_id)
        feats = extract_server_features(
            player_positions=player_positions,
            ball_positions=ball_positions,
            net_y=net_y,
            start_frame=start_frame,
            window_frames=args.window,
            calibrator=cal,
        )

        if not feats:
            n_skip_no_data += 1
            continue

        # GT server must be tracked in the window
        if gt_tid not in feats:
            n_skip_server_not_tracked += 1
            continue

        # GT serve side from GT server's foot position
        gt_foot_y = feats[gt_tid].foot_y_mean
        gt_side = "near" if gt_foot_y > net_y else "far"

        gt_server_tids[rally.rally_id] = gt_tid
        gt_serve_sides[rally.rally_id] = gt_side
        net_ys[rally.rally_id] = float(net_y)

        # Stratification metadata
        rally_meta[rally.rally_id] = {
            "video_id": rally.video_id,
            "has_calibration": cal is not None,
            "gt_late": feats[gt_tid].first_frame_presence < 0.95,
            "gt_side": gt_side,
            "n_players": len(feats),
            # Save inputs for heuristic baseline
            "player_positions": player_positions,
            "start_frame": start_frame,
            "net_y": float(net_y),
            "calibrator": cal,
            "gt_tid": gt_tid,
        }

        for tid, f in feats.items():
            x_list.append(f.to_array())
            y_list.append(1 if tid == gt_tid else 0)
            rally_id_list.append(rally.rally_id)
            track_id_list.append(tid)
            foot_y_list.append(f.foot_y_mean)

        if (i + 1) % 50 == 0:
            console.print(
                f"  [{i + 1}/{len(rallies)}] kept={len(gt_server_tids)} "
                f"samples={len(x_list)}"
            )

    console.print(
        f"  Done in {time.time() - t0:.1f}s. "
        f"Kept {len(gt_server_tids)} rallies, {len(x_list)} samples."
    )
    console.print(
        f"  Skipped: no_gt={n_skip_no_gt} no_data={n_skip_no_data} "
        f"server_not_tracked={n_skip_server_not_tracked}"
    )

    if not x_list:
        console.print("[red]No samples extracted, aborting.[/red]")
        return

    x = np.array(x_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int64)
    rally_ids_arr = np.array(rally_id_list)
    track_ids_arr = np.array(track_id_list, dtype=np.int64)
    foot_y_arr = np.array(foot_y_list, dtype=np.float64)

    console.print(
        f"  Class balance: {int(np.sum(y))} positive / "
        f"{int(np.sum(1 - y))} negative"
    )

    # ----- LOO-CV -----
    console.print("\n[bold]Running LOO-CV...[/bold]")
    classifier = ServerClassifier()
    t0 = time.time()
    loo = classifier.loo_cv(
        x=x, y=y,
        rally_ids=rally_ids_arr,
        track_ids=track_ids_arr,
        gt_server_tids=gt_server_tids,
        gt_serve_sides=gt_serve_sides,
        foot_y_means=foot_y_arr,
        net_ys=net_ys,
        positive_weight=3.0,
        progress=False,
    )
    console.print(f"  LOO-CV done in {time.time() - t0:.0f}s")

    console.print(
        f"\n[bold green]Server ID accuracy:[/bold green] "
        f"{loo['n_id_correct']}/{loo['n_evaluated']} "
        f"({loo['id_accuracy']:.1%})"
    )
    console.print(
        f"[bold green]Serve side accuracy:[/bold green] "
        f"{loo['n_side_correct']}/{loo['n_evaluated']} "
        f"({loo['side_accuracy']:.1%})"
    )

    # ----- Stratified breakdown -----
    by_strat: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "id_ok": 0, "side_ok": 0},
    )
    for r in loo["per_rally"]:
        meta = rally_meta.get(r["rally_id"])
        if meta is None:
            continue
        cal_key = "cal" if meta["has_calibration"] else "no_cal"
        late_key = "late" if meta["gt_late"] else "early"
        for k in [(cal_key, "*"), ("*", late_key), (cal_key, late_key),
                  ("*", meta["gt_side"])]:
            by_strat[k]["n"] += 1
            if r["id_correct"]:
                by_strat[k]["id_ok"] += 1
            if r["side_correct"]:
                by_strat[k]["side_ok"] += 1

    table = Table(title="Stratified accuracy")
    table.add_column("Stratum")
    table.add_column("N", justify="right")
    table.add_column("Server ID", justify="right")
    table.add_column("Serve Side", justify="right")
    for key in sorted(by_strat.keys()):
        s = by_strat[key]
        if s["n"] == 0:
            continue
        table.add_row(
            f"{key[0]} / {key[1]}",
            str(s["n"]),
            f"{s['id_ok'] / s['n']:.1%}",
            f"{s['side_ok'] / s['n']:.1%}",
        )
    console.print(table)

    # ----- Heuristic baseline -----
    console.print("\n[bold]Heuristic baseline (existing _identify_server)...[/bold]")
    h_id_correct = 0
    h_side_correct = 0
    h_n = 0
    for rid, meta in rally_meta.items():
        h_n += 1
        h_tid, h_side, _conf = _identify_server(
            meta["player_positions"], meta["start_frame"], meta["net_y"],
            calibrator=meta["calibrator"],
        )
        if h_tid == meta["gt_tid"]:
            h_id_correct += 1
        if h_side and h_side == gt_serve_sides[rid]:
            h_side_correct += 1
    console.print(
        f"  Heuristic Server ID: {h_id_correct}/{h_n} ({h_id_correct / max(1, h_n):.1%})"
    )
    console.print(
        f"  Heuristic Serve Side: {h_side_correct}/{h_n} ({h_side_correct / max(1, h_n):.1%})"
    )

    # ----- Train final model on all data -----
    console.print("\n[bold]Training final model on all data...[/bold]")
    metrics = classifier.train(x, y, positive_weight=3.0)
    console.print(
        f"  Train F1={metrics['train_f1']:.3f} "
        f"P={metrics['train_precision']:.3f} R={metrics['train_recall']:.3f}"
    )

    # ----- Feature importance -----
    importance = classifier.feature_importance()
    sorted_feats = sorted(importance.items(), key=lambda x: -x[1])
    table = Table(title="Top 10 feature importances")
    table.add_column("Feature")
    table.add_column("Importance", justify="right")
    for name, val in sorted_feats[:10]:
        table.add_row(name, f"{val:.3f}")
    console.print(table)

    # ----- Save -----
    if not args.no_save:
        save_path = Path("weights/server_classifier/server_classifier.pkl")
        classifier.save(save_path)
        console.print(f"[green]Saved model to {save_path}[/green]")
    else:
        console.print("[yellow]--no-save: skipping model save[/yellow]")


if __name__ == "__main__":
    main()
