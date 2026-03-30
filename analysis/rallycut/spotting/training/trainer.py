"""Training loop for E2E-Spot action spotting model."""

from __future__ import annotations

import functools
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rallycut.spotting.config import E2ESpotConfig
from rallycut.spotting.data.beach import RallyInfo
from rallycut.spotting.data.clip_dataset import ClipDataset, collate_clips, preload_all_frames
from rallycut.spotting.data.transforms import ClipTransform
from rallycut.spotting.model.e2e_spot import E2ESpot
from rallycut.spotting.training.losses import (
    FocalLoss,
    OffsetLoss,
    compute_class_weights,
)

# Force unbuffered output for Modal log streaming
print = functools.partial(print, flush=True)



WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "weights" / "spotting"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(
    train_rallies: list[RallyInfo],
    val_rallies: list[RallyInfo],
    config: E2ESpotConfig,
    device: torch.device,
    pretrained_path: Path | None = None,
    output_dir: Path | None = None,
    mirror_dir: Path | None = None,
) -> tuple[E2ESpot, dict]:
    """Train E2E-Spot model.

    Args:
        train_rallies: Training rally data.
        val_rallies: Validation rally data.
        config: Full model + training configuration.
        device: Training device.
        pretrained_path: Optional path to pretrained checkpoint for finetuning.
        output_dir: Directory for checkpoints and best model. Defaults to WEIGHTS_DIR.
        mirror_dir: Optional second directory to mirror best.pt and checkpoint.pt
            (e.g. network volume for persistence when output_dir is ephemeral local disk).

    Returns:
        Tuple of (best model, training stats dict).
    """
    tc = config.training
    save_dir = output_dir or WEIGHTS_DIR
    _set_seed(tc.seed)

    def _save(filename: str, data: object) -> None:
        """Save to save_dir and optionally mirror to persistent volume."""
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(data, save_dir / filename)
        if mirror_dir is not None:
            mirror_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(save_dir / filename), str(mirror_dir / filename))

    # Preload all frames into memory (eliminates disk I/O during training)
    print("  Preloading frames into memory...")
    all_rallies = train_rallies + val_rallies
    frame_cache = preload_all_frames(all_rallies)

    # Build datasets (with in-memory cache, no workers needed)
    train_transform = ClipTransform(size=224, is_train=True)
    val_transform = ClipTransform(size=224, is_train=False)

    train_dataset = ClipDataset(
        train_rallies,
        clip_length=tc.clip_length,
        transform=train_transform,
        oversample_events=3,
        is_train=True,
        frame_cache=frame_cache,
    )
    val_dataset = ClipDataset(
        val_rallies,
        clip_length=tc.clip_length,
        transform=val_transform,
        oversample_events=1,
        is_train=False,
        frame_cache=frame_cache,
    )

    print(f"  Train: {len(train_dataset)} clips from {len(train_rallies)} rallies")
    print(f"  Val:   {len(val_dataset)} clips from {len(val_rallies)} rallies")

    # num_workers=0: frames are in memory, transforms are fast batched ops.
    # Multiprocessing with large frame cache causes deadlocks in containers.
    train_loader = DataLoader(
        train_dataset,
        batch_size=tc.batch_size,
        shuffle=True,
        collate_fn=collate_clips,
        num_workers=0,
        pin_memory=False,
        generator=torch.Generator().manual_seed(tc.seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, tc.batch_size // 2),
        shuffle=False,
        collate_fn=collate_clips,
        num_workers=0,
        pin_memory=False,
    )

    # Build model
    model = E2ESpot(config).to(device)

    if pretrained_path is not None:
        print(f"  Loading pretrained weights: {pretrained_path}")
        state = torch.load(pretrained_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARN]Missing keys: {len(missing)}", flush=True)
        if unexpected:
            print(f"  [WARN]Unexpected keys: {len(unexpected)}", flush=True)

    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {param_count:,} params ({trainable:,} trainable)")

    # Class weights
    train_labels = [torch.from_numpy(r.labels) for r in train_rallies]
    class_weights = compute_class_weights(
        train_labels, config.head.num_classes
    ).to(device)
    print(f"  Class weights: {[f'{w:.1f}' for w in class_weights.tolist()]}")

    # Losses
    cls_criterion = FocalLoss(weight=class_weights, gamma=tc.focal_gamma)
    offset_criterion = OffsetLoss()

    # Optimizer with optional backbone LR scaling
    if tc.freeze_backbone_epochs > 0:
        model.freeze_backbone()
        print(f"  Backbone frozen for first {tc.freeze_backbone_epochs} epochs")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
    else:
        param_groups = model.get_param_groups(tc.backbone_lr_scale)
        optimizer = torch.optim.AdamW(
            [
                {"params": pg["params"], "lr": tc.lr * pg["lr_scale"]}
                for pg in param_groups
            ],
            weight_decay=tc.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.epochs)

    # Warmup
    warmup_scheduler = None
    if tc.warmup_epochs > 0:
        warmup_steps = tc.warmup_epochs * len(train_loader)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        )

    # Mixed precision
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Training loop
    best_val_f1 = 0.0
    best_model_state: dict | None = None
    best_epoch = 0
    patience_counter = 0
    start_epoch = 0
    start_time = time.time()

    # Resume from checkpoint if available
    checkpoint_path = save_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        print(f"  Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        patience_counter = ckpt.get("patience_counter", 0)
        if ckpt.get("best_model_state") is not None:
            best_model_state = ckpt["best_model_state"]

        # Rebuild optimizer with correct params (handles backbone freeze/unfreeze)
        if start_epoch >= tc.freeze_backbone_epochs and tc.freeze_backbone_epochs > 0:
            model.unfreeze_backbone()
        param_groups = model.get_param_groups(tc.backbone_lr_scale)
        optimizer = torch.optim.AdamW(
            [{"params": pg["params"], "lr": tc.lr * pg["lr_scale"]} for pg in param_groups],
            weight_decay=tc.weight_decay,
        )
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, KeyError):
            print("  [WARN]Optimizer state mismatch, using fresh optimizer", flush=True)

        # Rebuild scheduler from resumed epoch (skip warmup — already done)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=tc.epochs, last_epoch=start_epoch - 1
        )
        warmup_scheduler = None  # warmup already completed

        if scaler is not None and "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        print(
            f"  Resumed at epoch {start_epoch}, "
            f"best_val_f1={best_val_f1:.3f} (epoch {best_epoch + 1})"
        )

    print(f"  Starting training loop: epochs {start_epoch}-{tc.epochs}, {len(train_loader)} batches/epoch")

    for epoch in range(start_epoch, tc.epochs):
        # Unfreeze backbone after warmup period
        if epoch == tc.freeze_backbone_epochs and tc.freeze_backbone_epochs > 0:
            model.unfreeze_backbone()
            # Rebuild optimizer with all params
            param_groups = model.get_param_groups(tc.backbone_lr_scale)
            optimizer = torch.optim.AdamW(
                [
                    {"params": pg["params"], "lr": tc.lr * pg["lr_scale"]}
                    for pg in param_groups
                ],
                weight_decay=tc.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=tc.epochs - epoch
            )
            print(f"  Backbone unfrozen at epoch {epoch + 1}", flush=True)

        # --- Train ---
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"    batch {batch_idx}/{len(train_loader)}")
            clips = batch["clip"].to(device)        # (B, T, 3, H, W)
            labels = batch["labels"].to(device)      # (B, T)
            offsets_gt = batch["offsets"].to(device)  # (B, T)
            event_mask = batch["event_mask"].to(device)  # (B, T)

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(clips)
                    cls_loss = cls_criterion(out["logits"], labels)
                    off_loss = offset_criterion(out["offsets"], offsets_gt, event_mask)
                    loss = cls_loss + tc.offset_weight * off_loss

                scaler.scale(loss).backward()  # type: ignore[union-attr]
                scaler.unscale_(optimizer)  # type: ignore[union-attr]
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                scaler.step(optimizer)  # type: ignore[union-attr]
                scaler.update()  # type: ignore[union-attr]
            else:
                out = model(clips)
                cls_loss = cls_criterion(out["logits"], labels)
                off_loss = offset_criterion(out["offsets"], offsets_gt, event_mask)
                loss = cls_loss + tc.offset_weight * off_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()

            if warmup_scheduler is not None:
                warmup_scheduler.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(num_batches, 1)

        # --- Validate ---
        val_f1, val_action_acc = _validate(model, val_loader, device, config)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            # Save best weights (lightweight, for inference/finetuning)
            _save("best.pt", best_model_state)
        else:
            patience_counter += 1

        # Print every epoch (not just every 5th) for Modal log visibility
        elapsed = time.time() - start_time
        marker = "*" if patience_counter == 0 else ""
        print(
            f"  Epoch {epoch + 1:3d}/{tc.epochs}: loss={avg_loss:.4f}  "
            f"contact_F1={val_f1:.3f}  action_acc={val_action_acc:.3f}  "
            f"{marker}  [{elapsed:.0f}s]",
            flush=True,
        )

        # Save full checkpoint every 5 epochs (for resume / continued training)
        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "patience_counter": patience_counter,
                "best_model_state": best_model_state,
                "config": {
                    "clip_length": tc.clip_length,
                    "hidden_dim": config.temporal.hidden_dim,
                    "num_layers": config.temporal.num_layers,
                    "num_classes": config.head.num_classes,
                },
            }
            if scaler is not None:
                ckpt["scaler"] = scaler.state_dict()
            _save("checkpoint.pt", ckpt)

        if patience_counter >= tc.patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    # Final saves
    _save("best.pt", model.state_dict())
    print(f"  Saved best model to {save_dir / 'best.pt'}")
    # Remove checkpoint (training complete)
    for d in [save_dir, mirror_dir]:
        if d is not None:
            p = d / "checkpoint.pt"
            if p.exists():
                p.unlink()
    print("  Removed checkpoint (training complete)")

    stats = {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch + 1,
        "total_epochs": epoch + 1,
        "train_clips": len(train_dataset),
        "val_clips": len(val_dataset),
    }
    return model, stats


def _validate(
    model: E2ESpot,
    val_loader: DataLoader,
    device: torch.device,
    config: E2ESpotConfig,
) -> tuple[float, float]:
    """Run validation and compute contact F1 + action accuracy.

    Uses per-frame classification: any non-background prediction = contact.
    """
    model.eval()
    all_pred_cls: list[int] = []
    all_true_cls: list[int] = []

    with torch.no_grad():
        for batch in val_loader:
            clips = batch["clip"].to(device)
            labels = batch["labels"]

            out = model(clips)
            preds = out["logits"].argmax(dim=-1).cpu()  # (B, T)

            for b in range(clips.shape[0]):
                all_pred_cls.extend(preds[b].tolist())
                all_true_cls.extend(labels[b].tolist())

    y_true = np.array(all_true_cls)
    y_pred = np.array(all_pred_cls)

    # Contact F1: binary (event vs background)
    from sklearn.metrics import f1_score

    binary_true = (y_true > 0).astype(int)
    binary_pred = (y_pred > 0).astype(int)
    contact_f1 = f1_score(binary_true, binary_pred, zero_division=0)

    # Action accuracy (among true event frames)
    event_mask = y_true > 0
    if event_mask.sum() > 0:
        action_acc = (y_true[event_mask] == y_pred[event_mask]).mean()
    else:
        action_acc = 0.0

    return float(contact_f1), float(action_acc)
