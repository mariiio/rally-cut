"""Phase 2: Linear probe on VideoMAE features (contact / no-contact binary).

Goal: answer "is there ANY contact signal in VideoMAE CLS features?" with
the smallest possible head. If this passes the 60% F1 hard-gate we move on
to Phase 3's 7-class MSTCN head; if not, we stop and pivot.

Design:
- One logistic regression (and optional 768→64→2 MLP) per LOO-per-video fold.
- Labels: each stride=4 window is POSITIVE if any GT contact frame falls
  within ±3 effective-fps frames of the window centre. Radius absorbs the
  ±16-frame window coverage + small annotation jitter.
- Decode: peak-NMS over per-window probabilities (min_distance=3 windows =
  12 frames ≈ 400ms), threshold swept on each fold's held-out.
- Scoring: feed predictions through the production
  ``scripts.eval_action_detection.match_contacts`` Hungarian matcher with
  the same ±7-frame tolerance used by the Phase 0 baseline. Apples-to-
  apples with the 88.0% F1 baseline.
- FPS handling: features are extracted at ``effective_fps`` (30 after auto
  subsampling for >40fps videos). GT frames are stored at ``rally.fps`` —
  we project both into the effective-fps frame space before matching
  windows to contacts.

Outputs:
- ``reports/videomae_probe_loo_video_2026_04_19.md`` — per-fold + aggregate
  P/R/F1 tables and threshold sweep.
- Optional ``--out-json`` companion file for diffing vs later runs.

Usage (cd analysis):
    uv run python scripts/train_videomae_contact_probe.py               # all 68
    uv run python scripts/train_videomae_contact_probe.py --limit 5     # smoke
    uv run python scripts/train_videomae_contact_probe.py --mlp         # 768→64→2 MLP
    uv run python scripts/train_videomae_contact_probe.py --radius 5    # ±5 frame label

Go/no-go per plan: aggregate contact F1 ≥ 60% → proceed to Phase 3.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

# Silence sanity warnings we don't care about here
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)

console = Console()

STRIDE_DEFAULT = 4
BACKBONE_DEFAULT = "videomae-v1"
WINDOW_SIZE = 16  # VideoMAE window width in frames
DEFAULT_RADIUS_FRAMES = 3  # effective-fps frames
DEFAULT_TOLERANCE_MS = 233  # ±7 frames @ 30fps — matches Phase 0 baseline


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

@dataclass
class VideoWindows:
    """Windows + labels for a single video."""

    video_id: str
    features: np.ndarray  # (num_windows, 768)
    labels: np.ndarray    # (num_windows,) 0/1
    effective_fps: float
    stride: int
    rallies: list[RallyData]
    # Per-rally start frame in effective-fps video-absolute coordinates
    rally_start_eff: dict[str, int] = field(default_factory=dict)


def _collect_video_windows(
    video: EvaluationVideo,
    rallies: list[RallyData],
    cache: FeatureCache,
    stride: int,
    backbone: str,
    radius_frames: int,
) -> VideoWindows | None:
    """Load features and label each window from this video's GT rallies."""
    cached = cache.get(video.content_hash, stride=stride, backbone=backbone)
    if cached is None:
        return None
    features, meta = cached
    eff_fps = float(meta.fps)
    num_windows = features.shape[0]
    labels = np.zeros(num_windows, dtype=np.int64)

    # window i covers effective-fps frames [i*stride, i*stride + WINDOW_SIZE)
    window_centers = np.arange(num_windows) * stride + WINDOW_SIZE // 2

    rally_start_eff: dict[str, int] = {}
    for rally in rallies:
        rally_fps = float(rally.fps or eff_fps)
        rally_start_eff_frame = int(round((rally.start_ms / 1000.0) * eff_fps))
        rally_start_eff[rally.rally_id] = rally_start_eff_frame

        for gt in rally.gt_labels:
            # gt.frame is rally-relative at rally.fps. Convert to video-absolute
            # effective-fps frame (same space as window_centers).
            gt_eff_frame = rally_start_eff_frame + int(
                round(gt.frame * eff_fps / rally_fps)
            )
            if gt_eff_frame < 0 or gt_eff_frame >= num_windows * stride + WINDOW_SIZE:
                continue
            # Label every window whose centre is within ±radius of this GT
            diffs = np.abs(window_centers - gt_eff_frame)
            labels[diffs <= radius_frames] = 1

    return VideoWindows(
        video_id=video.id,
        features=features.astype(np.float32),
        labels=labels,
        effective_fps=eff_fps,
        stride=stride,
        rallies=rallies,
        rally_start_eff=rally_start_eff,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _train_logistic(x_train: np.ndarray, y_train: np.ndarray):
    """Class-balanced logistic regression (fast, zero-dependency baseline)."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=500,
        class_weight="balanced",
        C=1.0,
    )
    clf.fit(x_train, y_train)
    return clf


def _predict_logistic(clf, x: np.ndarray) -> np.ndarray:
    """Return positive-class probability, (N,)."""
    return clf.predict_proba(x)[:, 1]


def _train_mlp(
    x_train: np.ndarray, y_train: np.ndarray, device: str, epochs: int = 20,
):
    """Optional 768 → 64 → 2 MLP with focal-ish weighted BCE."""
    import torch

    x = torch.from_numpy(x_train).to(device)
    y = torch.from_numpy(y_train).to(device).long()

    net = torch.nn.Sequential(
        torch.nn.Linear(768, 64),
        torch.nn.GELU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(64, 2),
    ).to(device)

    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    weight = torch.tensor([1.0, float(pos_weight)], device=device)
    optim = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    batch_size = 4096
    num_samples = len(x)
    for _ep in range(epochs):
        perm = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            idx = perm[start:start + batch_size]
            logits = net(x[idx])
            loss = loss_fn(logits, y[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
    net.eval()
    return net


def _predict_mlp(net, x: np.ndarray, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        x_t = torch.from_numpy(x).to(device)
        probs = torch.softmax(net(x_t), dim=-1)[:, 1]
    return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Decoding + scoring
# ---------------------------------------------------------------------------

def _decode_peaks(
    probs: np.ndarray, threshold: float, min_distance_windows: int,
) -> list[int]:
    """Peak-NMS: return window indices of accepted contact candidates."""
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(
        probs, height=threshold, distance=min_distance_windows,
    )
    return list(peaks)


def _score_fold(
    vw: VideoWindows,
    probs: np.ndarray,
    threshold: float,
    min_distance_windows: int,
    tolerance_ms: int,
) -> tuple[int, int, int, int]:
    """Decode + match against GT; return (tp, fp, fn, gt_count)."""
    stride = vw.stride
    eff_fps = vw.effective_fps
    accepted_windows = _decode_peaks(probs, threshold, min_distance_windows)
    accepted_abs_frames = [w * stride + WINDOW_SIZE // 2 for w in accepted_windows]

    tp = fp = fn = 0
    gt_total = 0

    for rally in vw.rallies:
        rally_fps = float(rally.fps or eff_fps)
        rally_start_eff = vw.rally_start_eff[rally.rally_id]

        # Predicted frames that land inside this rally's span
        # Rally span in effective-fps frames
        rally_end_eff = rally_start_eff + int(
            round((rally.frame_count or 0) * eff_fps / rally_fps)
        )
        rally_preds = []
        for af in accepted_abs_frames:
            if rally_start_eff <= af < rally_end_eff:
                # Convert back to rally-relative at rally_fps for match_contacts
                rally_rel = int(round((af - rally_start_eff) * rally_fps / eff_fps))
                rally_preds.append({
                    "frame": rally_rel,
                    "action": "contact",
                    "playerTrackId": -1,
                })

        tolerance_frames = max(1, round(rally_fps * tolerance_ms / 1000))
        matches, unmatched = match_contacts(
            rally.gt_labels, rally_preds, tolerance=tolerance_frames,
        )

        gt_total += len(rally.gt_labels)
        for m in matches:
            if m.pred_frame is None:
                fn += 1
            else:
                tp += 1
        fp += len(unmatched)

    return tp, fp, fn, gt_total


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# Main LOO loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--backbone", type=str, default=BACKBONE_DEFAULT)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS_FRAMES,
                        help="Label positive if any GT within ±radius eff-fps frames")
    parser.add_argument("--tolerance-ms", type=int, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--min-dist-windows", type=int, default=3,
                        help="Peak-NMS min distance in windows (stride*N frames)")
    parser.add_argument("--mlp", action="store_true",
                        help="Use 768→64→2 MLP instead of logistic regression")
    parser.add_argument("--mlp-epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default=None,
                        help="torch device for MLP (auto: cuda > mps > cpu)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N videos (smoke-test)")
    parser.add_argument("--out", type=str,
                        default="reports/videomae_probe_loo_video_2026_04_19.md")
    parser.add_argument("--out-json", type=str,
                        default="reports/videomae_probe_loo_video_2026_04_19.json")
    args = parser.parse_args()

    if args.mlp and args.device is None:
        import torch

        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    console.print("[bold]Loading GT rallies + features...[/bold]")
    t0 = time.time()
    all_rallies = load_rallies_with_action_gt()
    action_ids = {r.video_id for r in all_rallies}
    all_videos = [
        v for v in load_evaluation_videos(require_ground_truth=True)
        if v.id in action_ids
    ]
    console.print(f"  {len(all_rallies)} rallies across {len(all_videos)} videos")

    cache = FeatureCache()
    by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in all_rallies:
        by_video[r.video_id].append(r)

    # Build per-video windows
    videos_with_features: list[EvaluationVideo] = []
    video_windows: list[VideoWindows] = []
    for v in all_videos:
        vw = _collect_video_windows(
            v, by_video[v.id], cache, args.stride, args.backbone, args.radius,
        )
        if vw is None:
            console.print(f"  [yellow]skip {v.id[:8]} {v.filename}: no features[/yellow]")
            continue
        videos_with_features.append(v)
        video_windows.append(vw)
    console.print(f"  {len(video_windows)}/{len(all_videos)} videos with features "
                  f"[{time.time()-t0:.0f}s]")

    if args.limit:
        video_windows = video_windows[: args.limit]
        videos_with_features = videos_with_features[: args.limit]

    total_pos = sum(int(vw.labels.sum()) for vw in video_windows)
    total_windows = sum(int(vw.labels.size) for vw in video_windows)
    console.print(
        f"  Windows: {total_windows:,} total; {total_pos:,} positive "
        f"({100 * total_pos / max(1, total_windows):.2f}%)"
    )

    # LOO loop
    console.print("\n[bold]LOO-per-video (linear probe)...[/bold]")

    # Threshold sweep candidates — same grid on every fold, aggregated at the end
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    agg_tp = {t: 0 for t in thresholds}
    agg_fp = {t: 0 for t in thresholds}
    agg_fn = {t: 0 for t in thresholds}
    per_fold: list[dict] = []

    for fold_idx, held in enumerate(video_windows, start=1):
        t_fold = time.time()
        train_parts = [vw for vw in video_windows if vw.video_id != held.video_id]
        x_train = np.concatenate([vw.features for vw in train_parts], axis=0)
        y_train = np.concatenate([vw.labels for vw in train_parts], axis=0)

        if args.mlp:
            net = _train_mlp(x_train, y_train, args.device, epochs=args.mlp_epochs)
            probs = _predict_mlp(net, held.features, args.device)
        else:
            clf = _train_logistic(x_train, y_train)
            probs = _predict_logistic(clf, held.features)

        fold_row = {
            "video_id": held.video_id,
            "n_windows": int(held.labels.size),
            "n_positive": int(held.labels.sum()),
            "n_rallies": len(held.rallies),
            "threshold_scores": {},
        }
        for thr in thresholds:
            tp, fp, fn, _gt_total = _score_fold(
                held, probs, thr, args.min_dist_windows, args.tolerance_ms,
            )
            agg_tp[thr] += tp
            agg_fp[thr] += fp
            agg_fn[thr] += fn
            fold_row["threshold_scores"][thr] = {"tp": tp, "fp": fp, "fn": fn}

        per_fold.append(fold_row)
        best_thr = max(
            thresholds,
            key=lambda t: _f1(fold_row["threshold_scores"][t]["tp"],
                              fold_row["threshold_scores"][t]["fp"],
                              fold_row["threshold_scores"][t]["fn"])[2],
        )
        best = fold_row["threshold_scores"][best_thr]
        _, _, fold_f1 = _f1(best["tp"], best["fp"], best["fn"])
        sum(agg_tp[best_thr] for _ in [0])
        _, _, cum_f1 = _f1(agg_tp[best_thr], agg_fp[best_thr], agg_fn[best_thr])
        console.print(
            f"  [{fold_idx}/{len(video_windows)}] {held.video_id[:8]} "
            f"({len(held.rallies)}r, pos={fold_row['n_positive']}): "
            f"best@thr={best_thr} F1={fold_f1:.1%} "
            f"| cum@{best_thr}={cum_f1:.1%} "
            f"({time.time()-t_fold:.1f}s)"
        )

    # Summary tables
    console.print("\n[bold]Threshold sweep (aggregated across folds)[/bold]")
    table = Table(show_header=True, header_style="bold")
    for col in ("Thr", "TP", "FP", "FN", "P", "R", "F1"):
        table.add_column(col, justify="right")
    best_agg = (0.0, -1.0, 0, 0, 0)
    for thr in thresholds:
        p, r, f1 = _f1(agg_tp[thr], agg_fp[thr], agg_fn[thr])
        if f1 > best_agg[1]:
            best_agg = (thr, f1, agg_tp[thr], agg_fp[thr], agg_fn[thr])
        table.add_row(
            f"{thr:.2f}", str(agg_tp[thr]), str(agg_fp[thr]), str(agg_fn[thr]),
            f"{p:.1%}", f"{r:.1%}", f"{f1:.1%}",
        )
    console.print(table)

    best_thr, best_f1, btp, bfp, bfn = best_agg
    bp, br, _ = _f1(btp, bfp, bfn)
    verdict = "PASS" if best_f1 >= 0.60 else "FAIL"
    color = "green" if verdict == "PASS" else "red"
    console.print(
        f"\n[bold]Best: thr={best_thr:.2f} F1=[{color}]{best_f1:.1%}[/{color}] "
        f"(P={bp:.1%} R={br:.1%})  Phase-2 gate ≥60% → [{color}]{verdict}[/{color}][/bold]"
    )

    # Write report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 2 — VideoMAE linear probe (LOO-per-video)")
    lines.append("")
    lines.append(f"- Model: {'MLP 768→64→2' if args.mlp else 'LogisticRegression (class-balanced)'}")
    lines.append(f"- Stride: {args.stride}, Backbone: {args.backbone}")
    lines.append(f"- Label radius: ±{args.radius} effective-fps frames")
    lines.append(f"- Match tolerance: ±{args.tolerance_ms} ms")
    lines.append(f"- Peak-NMS min distance: {args.min_dist_windows} windows")
    lines.append(f"- Total windows: {total_windows:,} ({total_pos:,} positive)")
    lines.append("")
    lines.append("## Aggregate (all folds, best threshold)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Threshold | {best_thr:.2f} |")
    lines.append(f"| Contact F1 | **{best_f1:.1%}** |")
    lines.append(f"| Precision | {bp:.1%} |")
    lines.append(f"| Recall | {br:.1%} |")
    lines.append(f"| TP / FP / FN | {btp} / {bfp} / {bfn} |")
    lines.append(f"| Gate (≥60% → Phase 3) | **{verdict}** |")
    lines.append("")
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append("| Thr | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for thr in thresholds:
        p, r, f1 = _f1(agg_tp[thr], agg_fp[thr], agg_fn[thr])
        lines.append(
            f"| {thr:.2f} | {agg_tp[thr]} | {agg_fp[thr]} | {agg_fn[thr]} "
            f"| {p:.1%} | {r:.1%} | {f1:.1%} |"
        )
    lines.append("")
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")

    if args.out_json:
        json_path = Path(args.out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "args": {
                "stride": args.stride, "backbone": args.backbone,
                "radius": args.radius, "tolerance_ms": args.tolerance_ms,
                "min_dist_windows": args.min_dist_windows,
                "model": "mlp" if args.mlp else "logreg",
            },
            "aggregate_best": {
                "threshold": best_thr, "f1": best_f1, "precision": bp, "recall": br,
                "tp": btp, "fp": bfp, "fn": bfn,
            },
            "threshold_sweep": {
                f"{t:.2f}": {
                    "tp": agg_tp[t], "fp": agg_fp[t], "fn": agg_fn[t],
                    **dict(zip(["p", "r", "f1"], _f1(agg_tp[t], agg_fp[t], agg_fn[t]))),
                }
                for t in thresholds
            },
            "per_fold": per_fold,
            "verdict": verdict,
        }
        json_path.write_text(json.dumps(payload, indent=2))
        console.print(f"[green]JSON: {json_path}[/green]")


if __name__ == "__main__":
    main()
