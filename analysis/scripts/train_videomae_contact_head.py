"""Phase 3: 7-class MSTCN head on VideoMAE features (LOO-per-video).

Trains a small MS-TCN++ (~200K params) over cached stride=4 VideoMAE features
for each held-out video, decodes per-frame probs into discrete contact events
via per-class peak-NMS, and scores against the production Hungarian matcher.

Why this is the right next step after the Phase 2 linear probe (F1 36.9%):
- Phase 2 proved VideoMAE CLS features carry contact signal (recall 54.7%
  at threshold 0.30). It also showed the signal is NOT linearly separable on
  single 16-frame windows — precision collapses to 27.8% because adjacent
  windows activate on the same ball-player interaction.
- MSTCN's dilated temporal conv aggregates context across ~256 frames at
  stride=4 (~8.5s at 30fps). That explicitly targets the precision gap:
  only *one* window per interaction is assigned to a class.
- 7-class prediction (including per-action labels) doubles as a check: if
  the head learns action-type signal from visual features alone, Phase 5
  fusion with the trajectory GBM is materially meaningful. If it can't
  separate serve from receive, the ceiling is close to the probe.

Design:
- Soft Gaussian targets centred on each GT contact frame (σ=3 effective-fps
  frames) on the class channel, background = 1 − max_k other.
- Soft cross-entropy loss (KL divergence form) + per-class weighting
  inversely proportional to √(class count) to counter the heavy
  background skew (~99% background).
- Auxiliary loss from every MSTCN stage (deep supervision).
- AdamW lr=3e-4, cosine schedule, EMA decay=0.999, early-stop by LOO
  train-loss plateau (no val set — LOO itself is the evaluation).

Scoring:
- For each rally in the held-out video: per-class peak-NMS on predicted
  probs, decoded into {frame, action} events, matched to GT via
  ``match_contacts`` with ±7f tolerance. Also produces a binary contact
  F1 collapsing all non-background classes, for a direct apples-to-apples
  number vs the Phase 0 88.0% baseline and the Phase 2 probe.

Hard gates per plan:
- Binary contact F1 ≥ 82% LOO-per-video → continue to Phase 4/5.
- 65-82% → flagged, still worth Phase 5 fusion but unlikely to beat Phase 0.
- < 65% → STOP, reconsider (stride=2, mean-pooling, E2E-Spot pivot).

Usage (cd analysis):
    uv run python scripts/train_videomae_contact_head.py                 # 68 folds
    uv run python scripts/train_videomae_contact_head.py --limit 5       # smoke
    uv run python scripts/train_videomae_contact_head.py --epochs 40     # longer training
    uv run python scripts/train_videomae_contact_head.py --device cuda   # force CUDA

Writes ``reports/videomae_mstcn_loo_video_2026_04_19.{md,json}``.
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

from rallycut.actions.videomae_contact_head import (
    CLASS_TO_IDX,
    CONTACT_CLASSES,
    NUM_CLASSES,
    build_contact_head,
)
from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

STRIDE_DEFAULT = 4
BACKBONE_DEFAULT = "videomae-v1"
WINDOW_SIZE = 16
SIGMA_FRAMES_DEFAULT = 3.0
DEFAULT_TOLERANCE_MS = 233  # ±7 frames @ 30fps — matches Phase 0 baseline
ACTION_CLASSES_POSITIVE = CONTACT_CLASSES[1:]  # skip background


# ---------------------------------------------------------------------------
# Per-video data preparation
# ---------------------------------------------------------------------------

@dataclass
class VideoData:
    video_id: str
    features: np.ndarray  # (T, 768)
    soft_targets: np.ndarray  # (T, 7) Gaussian-labelled soft targets
    effective_fps: float
    stride: int
    rallies: list[RallyData]
    rally_start_eff: dict[str, int] = field(default_factory=dict)


def _soft_targets_for_video(
    num_windows: int,
    stride: int,
    eff_fps: float,
    rallies: list[RallyData],
    sigma_frames: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Gaussian-target matrix (T, 7). Index 0 reserved for background.

    For each GT contact (class c, effective-fps frame f_gt) add
    ``exp(-(c - d)^2 / (2 sigma^2))`` to the c-th channel of every window
    whose centre d is within 3 sigma of f_gt. Background is the complement
    ``1 - max_c positive_c`` — ensures the soft distribution sums to 1 and
    unambiguous non-contact frames stay at target=[1, 0, 0, ...].
    """
    window_centers = np.arange(num_windows) * stride + WINDOW_SIZE // 2
    target = np.zeros((num_windows, NUM_CLASSES), dtype=np.float32)
    rally_start_eff: dict[str, int] = {}

    for rally in rallies:
        rally_fps = float(rally.fps or eff_fps)
        rally_start = int(round((rally.start_ms / 1000.0) * eff_fps))
        rally_start_eff[rally.rally_id] = rally_start

        for gt in rally.gt_labels:
            cls_idx = CLASS_TO_IDX.get(gt.action, 0)
            if cls_idx == 0:
                continue  # unknown action label; skip
            gt_eff = rally_start + int(round(gt.frame * eff_fps / rally_fps))
            # Only consider windows within 3σ to avoid O(T*N) blowup
            diffs = window_centers - gt_eff
            mask = np.abs(diffs) <= 3.0 * sigma_frames
            if not mask.any():
                continue
            g = np.exp(-(diffs[mask] ** 2) / (2.0 * sigma_frames ** 2))
            # Max-merge so overlapping contacts on the same class don't
            # double-count
            target[mask, cls_idx] = np.maximum(target[mask, cls_idx], g)

    # Background channel = 1 - max over positive classes, clamped >= 0
    max_pos = target[:, 1:].max(axis=1)
    target[:, 0] = np.clip(1.0 - max_pos, 0.0, 1.0)
    # Normalise so each frame's target distribution sums to 1 (guards
    # against overlap between classes at the same frame)
    denom = target.sum(axis=1, keepdims=True).clip(min=1e-6)
    target /= denom
    return target, rally_start_eff


def _load_video_data(
    cache: FeatureCache,
    video_id: str,
    content_hash: str,
    stride: int,
    backbone: str,
    rallies: list[RallyData],
    sigma_frames: float,
) -> VideoData | None:
    cached = cache.get(content_hash, stride=stride, backbone=backbone)
    if cached is None:
        return None
    features, meta = cached
    eff_fps = float(meta.fps)
    num_windows = features.shape[0]
    soft, rally_start_eff = _soft_targets_for_video(
        num_windows, stride, eff_fps, rallies, sigma_frames,
    )
    return VideoData(
        video_id=video_id,
        features=features.astype(np.float32),
        soft_targets=soft,
        effective_fps=eff_fps,
        stride=stride,
        rallies=rallies,
        rally_start_eff=rally_start_eff,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _detect_device(override: str | None) -> str:
    import torch

    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _class_weights(train_videos: list[VideoData], device: str):
    import torch

    total = np.zeros(NUM_CLASSES, dtype=np.float64)
    for vd in train_videos:
        total += vd.soft_targets.sum(axis=0)
    # Inverse sqrt frequency, normalised so mean weight ≈ 1
    weights = 1.0 / np.sqrt(np.clip(total, 1.0, None))
    weights *= NUM_CLASSES / weights.sum()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_one_fold(
    train_videos: list[VideoData],
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    ema_decay: float,
):
    """Train a fresh MSTCN on ``train_videos``. Returns (model, ema_model)."""
    import torch
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    model = build_contact_head().to(device)
    ema = build_contact_head().to(device)
    ema.load_state_dict(model.state_dict())
    ema.eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=epochs)
    weights = _class_weights(train_videos, device)

    # Each video is one training sample; shuffle per epoch.
    train_tensors = []
    for vd in train_videos:
        x = torch.from_numpy(vd.features).to(device).transpose(0, 1).unsqueeze(0)  # (1, 768, T)
        y = torch.from_numpy(vd.soft_targets).to(device).unsqueeze(0)  # (1, T, 7)
        train_tensors.append((x, y))

    rng = np.random.default_rng(42)
    for ep in range(epochs):
        order = rng.permutation(len(train_tensors))
        running_loss = 0.0
        for idx in order:
            x, y_soft = train_tensors[idx]
            stage_logits = model.forward_all_stages(x)
            total_loss = torch.zeros((), device=device)
            for logits in stage_logits:
                # logits: (1, 7, T); softmax along class dim
                log_probs = torch.log_softmax(logits, dim=1)
                # Weighted soft-CE: -sum_c w_c * target_c * log p_c
                w = weights.view(1, -1, 1)
                tgt = y_soft.transpose(1, 2)  # (1, 7, T)
                loss = -(w * tgt * log_probs).sum(dim=1).mean()
                total_loss = total_loss + loss
            total_loss = total_loss / len(stage_logits)
            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            # EMA update
            with torch.no_grad():
                for ep_p, p in zip(ema.parameters(), model.parameters()):
                    ep_p.mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)
            running_loss += float(total_loss.item())
        scheduler.step()
    return model, ema


def _predict(model, features: np.ndarray, device: str) -> np.ndarray:
    import torch

    with torch.no_grad():
        x = torch.from_numpy(features).to(device).transpose(0, 1).unsqueeze(0)
        logits = model(x)  # (1, 7, T)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (7, T)
    return probs.T  # (T, 7)


# ---------------------------------------------------------------------------
# Decoding + scoring
# ---------------------------------------------------------------------------

def _decode_events(
    probs: np.ndarray,  # (T, 7)
    stride: int,
    min_distance_windows: int,
    threshold: float,
) -> list[tuple[int, str, float]]:
    """Per-class peak-NMS → list of (abs_eff_frame, action, score)."""
    from scipy.signal import find_peaks

    events: list[tuple[int, str, float]] = []
    for cls_idx in range(1, NUM_CLASSES):  # skip background
        peaks, props = find_peaks(
            probs[:, cls_idx], height=threshold, distance=min_distance_windows,
        )
        for w_idx, h in zip(peaks, props.get("peak_heights", [])):
            abs_frame = int(w_idx * stride + WINDOW_SIZE // 2)
            events.append((abs_frame, CONTACT_CLASSES[cls_idx], float(h)))
    events.sort(key=lambda e: e[0])
    return events


def _score_fold(
    vd: VideoData,
    probs: np.ndarray,
    threshold: float,
    min_distance_windows: int,
    tolerance_ms: int,
) -> dict:
    """Decode and match against GT. Returns per-class + binary tallies."""
    events = _decode_events(probs, vd.stride, min_distance_windows, threshold)

    per_class: dict[str, dict[str, int]] = {
        c: {"tp": 0, "fp": 0, "fn": 0} for c in ACTION_CLASSES_POSITIVE
    }
    bin_tp = bin_fp = bin_fn = 0
    gt_total = 0
    action_correct = 0
    action_total = 0

    for rally in vd.rallies:
        rally_fps = float(rally.fps or vd.effective_fps)
        rally_start = vd.rally_start_eff[rally.rally_id]
        rally_end = rally_start + int(round((rally.frame_count or 0) * vd.effective_fps / rally_fps))

        rally_preds: list[dict] = []
        for abs_frame, action, score in events:
            if not (rally_start <= abs_frame < rally_end):
                continue
            rally_rel = int(round((abs_frame - rally_start) * rally_fps / vd.effective_fps))
            rally_preds.append({
                "frame": rally_rel,
                "action": action,
                "playerTrackId": -1,
            })

        tolerance_frames = max(1, round(rally_fps * tolerance_ms / 1000))
        matches, unmatched = match_contacts(
            rally.gt_labels, rally_preds, tolerance=tolerance_frames,
        )

        gt_total += len(rally.gt_labels)
        for m in matches:
            if m.pred_frame is None:
                bin_fn += 1
                if m.gt_action in per_class:
                    per_class[m.gt_action]["fn"] += 1
            else:
                bin_tp += 1
                action_total += 1
                if m.gt_action == m.pred_action:
                    action_correct += 1
                    if m.gt_action in per_class:
                        per_class[m.gt_action]["tp"] += 1
                else:
                    if m.gt_action in per_class:
                        per_class[m.gt_action]["fn"] += 1
                    if m.pred_action and m.pred_action in per_class:
                        per_class[m.pred_action]["fp"] += 1

        bin_fp += len(unmatched)
        for u in unmatched:
            a = u.get("action")
            if a in per_class:
                per_class[a]["fp"] += 1

    return {
        "binary": {"tp": bin_tp, "fp": bin_fp, "fn": bin_fn},
        "per_class": per_class,
        "action_accuracy": {"correct": action_correct, "total": action_total},
        "gt_total": gt_total,
    }


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return p, r, f1


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    parser.add_argument("--backbone", type=str, default=BACKBONE_DEFAULT)
    parser.add_argument("--sigma", type=float, default=SIGMA_FRAMES_DEFAULT,
                        help="Target Gaussian σ in eff-fps frames")
    parser.add_argument("--tolerance-ms", type=int, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--min-dist-windows", type=int, default=3,
                        help="Per-class peak-NMS min distance in windows")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N folds (smoke-test)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=str,
                        default="reports/videomae_mstcn_loo_video_2026_04_19.md")
    parser.add_argument("--out-json", type=str,
                        default="reports/videomae_mstcn_loo_video_2026_04_19.json")
    args = parser.parse_args()

    device = _detect_device(args.device)
    console.print(f"[dim]Device: {device}[/dim]")

    console.print("[bold]Loading GT + features...[/bold]")
    t0 = time.time()
    all_rallies = load_rallies_with_action_gt()
    action_ids = {r.video_id for r in all_rallies}
    all_videos = [
        v for v in load_evaluation_videos(require_ground_truth=True)
        if v.id in action_ids
    ]
    by_video: dict[str, list[RallyData]] = defaultdict(list)
    for r in all_rallies:
        by_video[r.video_id].append(r)

    cache = FeatureCache()
    data: list[VideoData] = []
    for v in all_videos:
        vd = _load_video_data(
            cache, v.id, v.content_hash, args.stride, args.backbone,
            by_video[v.id], args.sigma,
        )
        if vd is None:
            console.print(f"  [yellow]skip {v.id[:8]}: no features[/yellow]")
            continue
        data.append(vd)
    console.print(f"  {len(data)}/{len(all_videos)} videos ready "
                  f"[{time.time()-t0:.0f}s]")

    if args.limit:
        data = data[: args.limit]
    console.print(f"[bold]LOO-per-video ({len(data)} folds, {args.epochs} epochs each)[/bold]")

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # Aggregated per-class/overall tallies at each threshold
    agg_binary = {t: {"tp": 0, "fp": 0, "fn": 0} for t in thresholds}
    agg_per_class = {
        t: {c: {"tp": 0, "fp": 0, "fn": 0} for c in ACTION_CLASSES_POSITIVE}
        for t in thresholds
    }
    agg_action_acc = {t: {"correct": 0, "total": 0} for t in thresholds}
    per_fold: list[dict] = []

    import torch

    for fold_idx, held in enumerate(data, start=1):
        t_fold = time.time()
        train_vids = [vd for vd in data if vd.video_id != held.video_id]
        model, ema = _train_one_fold(
            train_vids, device, args.epochs, args.lr,
            args.weight_decay, args.ema_decay,
        )
        probs = _predict(ema, held.features, device)

        fold_row = {"video_id": held.video_id, "per_threshold": {}}
        for thr in thresholds:
            r = _score_fold(held, probs, thr, args.min_dist_windows, args.tolerance_ms)
            for k in ("tp", "fp", "fn"):
                agg_binary[thr][k] += r["binary"][k]
            for c in ACTION_CLASSES_POSITIVE:
                for k in ("tp", "fp", "fn"):
                    agg_per_class[thr][c][k] += r["per_class"][c][k]
            agg_action_acc[thr]["correct"] += r["action_accuracy"]["correct"]
            agg_action_acc[thr]["total"] += r["action_accuracy"]["total"]
            fold_row["per_threshold"][thr] = r

        # Free GPU memory between folds
        del model, ema
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

        # Pick best binary F1 for this fold display
        best_thr, best_f1 = 0.0, -1.0
        for thr in thresholds:
            _, _, f1 = _f1(**fold_row["per_threshold"][thr]["binary"])
            if f1 > best_f1:
                best_thr, best_f1 = thr, f1
        _, _, cum_f1 = _f1(**agg_binary[best_thr])
        console.print(
            f"  [{fold_idx}/{len(data)}] {held.video_id[:8]} "
            f"(gt={held.soft_targets[:, 1:].sum():.0f}): "
            f"bin F1@{best_thr:.2f}={best_f1:.1%} "
            f"| cum@{best_thr:.2f}={cum_f1:.1%} "
            f"({time.time()-t_fold:.0f}s)"
        )
        per_fold.append(fold_row)

    # Summary: best threshold by aggregate binary F1
    best_thr = max(thresholds, key=lambda t: _f1(**agg_binary[t])[2])
    bp, br, bf1 = _f1(**agg_binary[best_thr])
    acc = agg_action_acc[best_thr]
    action_acc_val = acc["correct"] / max(1, acc["total"])

    console.print("\n[bold]Aggregate (all folds, best threshold)[/bold]")
    table = Table(show_header=True, header_style="bold")
    for c in ("Thr", "TP", "FP", "FN", "P", "R", "F1"):
        table.add_column(c, justify="right")
    for thr in thresholds:
        p, r, f1 = _f1(**agg_binary[thr])
        style = "bold" if thr == best_thr else ""
        table.add_row(
            f"{thr:.2f}",
            str(agg_binary[thr]["tp"]),
            str(agg_binary[thr]["fp"]),
            str(agg_binary[thr]["fn"]),
            f"{p:.1%}", f"{r:.1%}", f"{f1:.1%}",
            style=style,
        )
    console.print(table)

    # Per-class at best threshold
    pc = agg_per_class[best_thr]
    pc_table = Table(title=f"Per-class F1 @ threshold {best_thr:.2f}")
    pc_table.add_column("Class", style="bold")
    for col in ("TP", "FP", "FN", "P", "R", "F1"):
        pc_table.add_column(col, justify="right")
    for c in ACTION_CLASSES_POSITIVE:
        p, r, f1 = _f1(**pc[c])
        pc_table.add_row(
            c, str(pc[c]["tp"]), str(pc[c]["fp"]), str(pc[c]["fn"]),
            f"{p:.1%}", f"{r:.1%}", f"{f1:.1%}",
        )
    console.print(pc_table)

    gate = (
        "PASS" if bf1 >= 0.82 else
        "FLAG" if bf1 >= 0.65 else
        "FAIL"
    )
    color = {"PASS": "green", "FLAG": "yellow", "FAIL": "red"}[gate]
    console.print(
        f"\n[bold]Best: thr={best_thr:.2f}  binary F1=[{color}]{bf1:.1%}[/{color}]  "
        f"(P={bp:.1%} R={br:.1%})  action_acc={action_acc_val:.1%}  "
        f"Phase-3 gate → [{color}]{gate}[/{color}][/bold]"
    )

    # Reports
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase 3 — VideoMAE MSTCN head (LOO-per-video)")
    lines.append("")
    lines.append(f"- Model: MSTCN (2 stages × 8 layers, hidden=64) on 768-dim VideoMAE @ stride={args.stride}")
    lines.append(f"- σ={args.sigma}, tolerance=±{args.tolerance_ms}ms, NMS min_dist={args.min_dist_windows} windows")
    lines.append(f"- Epochs/fold: {args.epochs}, LR={args.lr}, EMA={args.ema_decay}")
    lines.append(f"- Folds: {len(data)}")
    lines.append("")
    lines.append("## Aggregate binary F1 (all classes collapsed)")
    lines.append("")
    lines.append("| Thr | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for thr in thresholds:
        p, r, f1 = _f1(**agg_binary[thr])
        marker = " ← best" if thr == best_thr else ""
        lines.append(
            f"| {thr:.2f} | {agg_binary[thr]['tp']} | {agg_binary[thr]['fp']} | "
            f"{agg_binary[thr]['fn']} | {p:.1%} | {r:.1%} | {f1:.1%}{marker} |"
        )
    lines.append("")
    lines.append(f"## Per-class F1 @ best threshold {best_thr:.2f}")
    lines.append("")
    lines.append("| Class | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in ACTION_CLASSES_POSITIVE:
        p, r, f1 = _f1(**pc[c])
        lines.append(
            f"| {c} | {pc[c]['tp']} | {pc[c]['fp']} | {pc[c]['fn']} "
            f"| {p:.1%} | {r:.1%} | {f1:.1%} |"
        )
    lines.append("")
    lines.append(f"## Gate: {gate}")
    lines.append("")
    lines.append("- ≥82% binary F1 → Phase 4/5 (PASS)")
    lines.append("- 65–82% → flagged (FLAG)")
    lines.append("- <65% → STOP (FAIL)")
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")

    if args.out_json:
        payload = {
            "args": vars(args),
            "best_threshold": best_thr,
            "aggregate_binary": {
                "f1": bf1, "precision": bp, "recall": br,
                **agg_binary[best_thr],
            },
            "action_accuracy": action_acc_val,
            "per_class_at_best": {
                c: {**pc[c], "precision": _f1(**pc[c])[0],
                    "recall": _f1(**pc[c])[1], "f1": _f1(**pc[c])[2]}
                for c in ACTION_CLASSES_POSITIVE
            },
            "threshold_sweep": {
                f"{t:.2f}": {
                    **agg_binary[t],
                    **dict(zip(["precision", "recall", "f1"], _f1(**agg_binary[t]))),
                }
                for t in thresholds
            },
            "verdict": gate,
        }
        json_path = Path(args.out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, default=float))
        console.print(f"[green]JSON: {json_path}[/green]")


if __name__ == "__main__":
    main()
