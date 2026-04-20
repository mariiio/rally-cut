"""Phase 1 crop-head frozen-backbone probe.

Trains a simple MLP head over frozen ResNet-18 features extracted from
pre-cached player + ball crops. Validates against the pre-registered
ship gates in docs/superpowers/plans/2026-04-21-crop-head-validation.md:

  Gate 1: Test AUC ≥ 0.75 on held-out videos
  Gate 2: Orthogonality + neg-control absolute gap ≥ 0.15
  Gate 3: Hard-negative AUC ≥ 0.65 (not just "player present")

All three must pass for PASS verdict. Any fail → NO-GO.

Video-level split: 53 train / 5 val / 10 test, deterministic selection
(sorted video_ids, every 7th to test, next 5 to val, rest to train).

Usage (cd analysis):
    uv run python scripts/train_crop_head_probe.py
    uv run python scripts/train_crop_head_probe.py --epochs 20 --seed 1337
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.ml.crop_head.backbone import FrozenResNet18
from rallycut.ml.crop_head.dataset import CropContactDataset
from rallycut.ml.crop_head.head import CropHeadMLP

console = Console()

CACHE_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "crop_dataset"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

GATE_AUC = 0.75
GATE_ORTHO_GAP = 0.15
GATE_HARD_AUC = 0.65


class CropHeadModel(nn.Module):
    """Combined model: frozen backbone + trainable MLP head.

    Takes a batch of (player_crop[T,3,64,64], ball_patch[T,3,32,32]),
    runs the backbone on each frame × both inputs, concatenates features,
    pools across time, and returns (B,) logits.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = FrozenResNet18()
        self.head = CropHeadMLP(d_in=1024, d_hidden=256)

    def forward(self, player_crops: torch.Tensor, ball_patches: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = player_crops.shape
        player_flat = player_crops.reshape(b * t, c, h, w)
        ball_flat = ball_patches.reshape(b * t, c, ball_patches.shape[-2], ball_patches.shape[-1])
        player_feat = self.backbone(player_flat).reshape(b, t, 512)
        ball_feat = self.backbone(ball_flat).reshape(b, t, 512)
        combined = torch.cat([player_feat, ball_feat], dim=-1)  # (B, T, 1024)
        return self.head(combined)


def _split_video_ids(all_vids: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Deterministic 53/5/10 split: sorted ids, every 7th → test, next 5 → val."""
    vids = sorted(all_vids)
    test = vids[::7][:10]
    remaining = [v for v in vids if v not in test]
    # Deterministic val pick: every 10th of the remaining
    val = remaining[::10][:5]
    train = [v for v in remaining if v not in val]
    return {"train": train, "val": val, "test": test}


def _collate(batch: list[dict]) -> dict:
    """Custom collate: stack tensors, keep lists for strings."""
    return {
        "player_crop": torch.stack([b["player_crop"] for b in batch]),
        "ball_patch": torch.stack([b["ball_patch"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
        "gbm_conf": torch.tensor([b["gbm_conf"] for b in batch], dtype=torch.float32),
        "source": [b["source"] for b in batch],
        "video_id": [b["video_id"] for b in batch],
        "rally_id": [b["rally_id"] for b in batch],
        "frame": [b["frame"] for b in batch],
    }


@torch.no_grad()
def _eval_model(
    model: CropHeadModel,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Return per-sample probs + labels + sources."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    all_sources: list[str] = []
    all_conf: list[float] = []
    for batch in loader:
        pc = batch["player_crop"].to(device)
        bp = batch["ball_patch"].to(device)
        logits = model(pc, bp)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(batch["label"].numpy().astype(int).tolist())
        all_sources.extend(batch["source"])
        all_conf.extend(batch["gbm_conf"].numpy().tolist())
    return {
        "probs": np.array(all_probs),
        "labels": np.array(all_labels),
        "sources": all_sources,
        "gbm_conf": np.array(all_conf),
    }


def _orthogonality_test(eval_out: dict) -> dict:
    """VideoMAE-style: P(model fires at GT) − P(model fires at non-contact)."""
    probs = eval_out["probs"]
    labels = eval_out["labels"]
    sources = np.array(eval_out["sources"])

    gt_mask = (labels == 1) & (sources == "gt_positive")
    nc_mask = (labels == 0) & ((sources == "hard_negative") | (sources == "random_negative"))

    gt_probs = probs[gt_mask]
    nc_probs = probs[nc_mask]

    gt_ge_05 = float((gt_probs >= 0.5).mean()) if len(gt_probs) else 0.0
    nc_ge_05 = float((nc_probs >= 0.5).mean()) if len(nc_probs) else 0.0

    return {
        "n_gt": int(len(gt_probs)),
        "n_non_contact": int(len(nc_probs)),
        "gt_ge_05": gt_ge_05,
        "nc_ge_05": nc_ge_05,
        "absolute_gap": gt_ge_05 - nc_ge_05,
        "gate_pass": (gt_ge_05 - nc_ge_05) >= GATE_ORTHO_GAP,
    }


def _hard_neg_auc(eval_out: dict) -> dict:
    """AUC on GT-positives vs hard-negatives only."""
    sources = np.array(eval_out["sources"])
    mask = (sources == "gt_positive") | (sources == "hard_negative")
    if mask.sum() < 10:
        return {"auc": float("nan"), "n": int(mask.sum()), "gate_pass": False,
                "reason": "insufficient samples"}
    labels = eval_out["labels"][mask]
    probs = eval_out["probs"][mask]
    if len(set(labels.tolist())) < 2:
        return {"auc": float("nan"), "n": int(mask.sum()), "gate_pass": False,
                "reason": "single-class subset"}
    auc = float(roc_auc_score(labels, probs))
    return {
        "auc": auc,
        "n": int(mask.sum()),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "gate_pass": auc >= GATE_HARD_AUC,
    }


def _threshold_sweep(eval_out: dict) -> list[dict]:
    probs = eval_out["probs"]
    labels = eval_out["labels"]
    rows = []
    for th in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (probs >= th).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        rows.append({"threshold": th, "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1})
    return rows


def _train_one_epoch(
    model: CropHeadModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    # Backbone stays in eval mode (BN fixed). Only head trains.
    model.backbone.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        pc = batch["player_crop"].to(device)
        bp = batch["ball_patch"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(pc, bp)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * y.size(0)
        n += y.size(0)
    return total_loss / max(1, n)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-report", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    console.print(f"[bold]device:[/bold] {device}")

    # --- Load splits ---
    all_vids = sorted({p.name for p in args.cache_root.iterdir() if p.is_dir()})
    if not all_vids:
        console.print(f"[red]No videos found in {args.cache_root}. Run extract_crop_dataset.py first.[/red]")
        return 1
    splits = _split_video_ids(all_vids)
    console.print(
        f"[bold]Split:[/bold] train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )

    train_ds = CropContactDataset(args.cache_root, splits["train"])
    val_ds = CropContactDataset(args.cache_root, splits["val"])
    test_ds = CropContactDataset(args.cache_root, splits["test"])

    # Class balance for pos_weight
    train_labels = np.array([int(np.load(p)["label"]) for _, p in train_ds.items])
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    console.print(f"  train pos={n_pos} neg={n_neg} pos_weight={pos_weight.item():.2f}")
    console.print(f"  val n={len(val_ds)}  test n={len(test_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate,
    )

    # --- Model ---
    model = CropHeadModel().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training loop ---
    best_val_auc = -1.0
    best_state = None
    epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_out = _eval_model(model, val_loader, device)
        val_auc = float(roc_auc_score(val_out["labels"], val_out["probs"])) \
            if len(set(val_out["labels"].tolist())) == 2 else float("nan")
        dt = time.time() - t0
        console.print(
            f"  [epoch {epoch:2d}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_auc={val_auc:.4f} ({dt:.1f}s)"
        )
        epoch_rows.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc, "dt_s": dt})
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # --- Load best model + eval on test ---
    if best_state is not None:
        model.load_state_dict(best_state)
    test_out = _eval_model(model, test_loader, device)

    test_auc = float(roc_auc_score(test_out["labels"], test_out["probs"])) \
        if len(set(test_out["labels"].tolist())) == 2 else float("nan")
    sweep = _threshold_sweep(test_out)
    ortho = _orthogonality_test(test_out)
    hard = _hard_neg_auc(test_out)

    # --- Per-source breakdown ---
    sources = np.array(test_out["sources"])
    probs = test_out["probs"]
    per_src = {}
    for src in ("gt_positive", "hard_negative", "random_negative"):
        mask = sources == src
        if mask.sum() == 0:
            per_src[src] = {"n": 0, "mean_prob": float("nan"), "p_ge_05": float("nan")}
        else:
            per_src[src] = {
                "n": int(mask.sum()),
                "mean_prob": float(probs[mask].mean()),
                "p_ge_05": float((probs[mask] >= 0.5).mean()),
            }

    # --- Decision ---
    gate1_pass = test_auc >= GATE_AUC
    gate2_pass = ortho["gate_pass"]
    gate3_pass = hard["gate_pass"]
    all_pass = gate1_pass and gate2_pass and gate3_pass
    verdict = "PASS" if all_pass else "NO-GO"

    # --- Write report ---
    today = date.today().isoformat()
    report_path = Path(args.out_report) if args.out_report else (
        REPORTS_DIR / f"crop_head_phase1_probe_{today}.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Crop-Head Phase 1 Probe — {verdict}")
    lines.append("")
    lines.append(f"- Date: {today}")
    lines.append(f"- Device: {device}")
    lines.append(f"- Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    lines.append(f"- Seed: {args.seed}")
    lines.append("")
    lines.append("## Splits (video-level)")
    lines.append(f"- Train: {len(splits['train'])} videos, {len(train_ds)} samples ({n_pos} pos + {n_neg} neg)")
    lines.append(f"- Val: {len(splits['val'])} videos, {len(val_ds)} samples")
    lines.append(f"- Test: {len(splits['test'])} videos, {len(test_ds)} samples")
    lines.append("")
    lines.append("## Ship gates (pre-registered)")
    lines.append("")
    lines.append("| Gate | Threshold | Observed | PASS? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Test AUC | ≥ {GATE_AUC} | **{test_auc:.4f}** | {'YES' if gate1_pass else 'NO'} |")
    lines.append(
        f"| Orthogonality gap | ≥ {GATE_ORTHO_GAP} | "
        f"**{ortho['absolute_gap']:+.4f}** "
        f"(GT {ortho['gt_ge_05']:.3f} − NC {ortho['nc_ge_05']:.3f}) | "
        f"{'YES' if gate2_pass else 'NO'} |"
    )
    hard_auc_str = f"**{hard['auc']:.4f}**" if not np.isnan(hard['auc']) else 'NaN'
    lines.append(
        f"| Hard-neg AUC | ≥ {GATE_HARD_AUC} | {hard_auc_str} "
        f"(n={hard['n']}) | {'YES' if gate3_pass else 'NO'} |"
    )
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    if all_pass:
        lines.append("All three pre-registered gates passed. Phase 2 (architecture "
                     "ablations: T window, pooling, input combinations) is scheduled "
                     "as a separate plan in a subsequent session.")
    else:
        failed = []
        if not gate1_pass:
            failed.append(f"Gate 1 (Test AUC {test_auc:.4f} < {GATE_AUC})")
        if not gate2_pass:
            failed.append(f"Gate 2 (Ortho gap {ortho['absolute_gap']:+.4f} < {GATE_ORTHO_GAP})")
        if not gate3_pass:
            failed.append(f"Gate 3 (Hard-neg AUC {hard.get('auc', float('nan')):.4f} < {GATE_HARD_AUC})")
        lines.append("Failed: " + "; ".join(failed) + ".")
        lines.append("")
        lines.append("Per the Phase 0 memo, this indicates the frozen-backbone architecture "
                     "does not linearly separate contact-vs-non-contact on these crops. "
                     "Redirect to dedicated pose-event classifier (2-3 days) or GT "
                     "attribution repair per memory/crop_head_phase0_2026_04_20.md.")

    lines.append("")
    lines.append("## Per-source breakdown (test)")
    lines.append("")
    lines.append("| Source | n | Mean prob | P(≥0.5) |")
    lines.append("|---|---|---|---|")
    for src, stats in per_src.items():
        lines.append(
            f"| {src} | {stats['n']} | {stats['mean_prob']:.4f} | {stats['p_ge_05']:.4f} |"
        )
    lines.append("")

    lines.append("## Threshold sweep (test)")
    lines.append("")
    lines.append("| Threshold | TP | FP | FN | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in sweep:
        lines.append(
            f"| {row['threshold']:.2f} | {row['tp']} | {row['fp']} | {row['fn']} "
            f"| {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} |"
        )
    lines.append("")

    lines.append("## Training curve")
    lines.append("")
    lines.append("| Epoch | Train Loss | Val AUC | Time (s) |")
    lines.append("|---|---|---|---|")
    for row in epoch_rows:
        lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | "
            f"{row['val_auc']:.4f} | {row['dt_s']:.1f} |"
        )
    lines.append("")
    lines.append("## Test split video IDs")
    for v in splits["test"]:
        lines.append(f"- `{v}`")
    lines.append("")

    report_path.write_text("\n".join(lines))
    console.print(f"\n[bold]Report:[/bold] {report_path}")
    console.print(f"[bold]Verdict:[/bold] {verdict}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({
            "verdict": verdict,
            "test_auc": test_auc,
            "ortho": ortho,
            "hard_neg": hard,
            "per_source": per_src,
            "threshold_sweep": sweep,
            "train_curve": epoch_rows,
            "splits": splits,
        }, indent=2, default=str))

    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
