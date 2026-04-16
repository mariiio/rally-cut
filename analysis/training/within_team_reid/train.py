"""Training loop for one variant.

`run_variant(cfg)` builds dataset/sampler/model, runs N epochs with mixed
precision, validates each epoch against the eval cache, and writes:
- weights/within_team_reid/variant_<id>/best.pt + last.pt
- weights/within_team_reid/variant_<id>/epochs.jsonl
- updates the global best.pt if this variant wins (subject to CF guard)

Best-checkpoint rule: max held-out rank-1 SUBJECT TO cross_rally >= 0.683.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from .config import (
    CROSS_RALLY_GUARD,
    EVAL_CACHE_META,
    EVAL_CACHE_PATH,
    HELDOUT_TARGET,
    REPORTS_ROOT,
    TRAINING_DATA_ROOT,
    WEIGHTS_ROOT,
    TrainConfig,
)
from .data.dataset import (
    PairDataset,
    build_identity_encoding,
    build_team_encoding,
)
from .data.manifest import load_manifest
from .data.sampler import PairBatchSampler
from .eval.cache import load_cache
from .eval.cross_rally import evaluate as evaluate_cross_rally
from .eval.held_out import evaluate as evaluate_held_out
from .losses.supcon import SupConLoss
from .losses.teammate_margin import TeammateMarginLoss
from .model.backbone import BackboneRunner
from .model.head import MLPHead

logger = logging.getLogger("within_team_reid.train")

TIER_POSITIVE = 0
TIER_EASY_NEG = 1
TIER_MID = 2
TIER_GOLD = 3


@dataclass
class EpochMetrics:
    epoch: int
    loss_total: float
    loss_supcon: float
    loss_tm: float
    teammate_margin_train: float
    embedding_variance_train: float
    held_out_rank1: float
    held_out_n_scored: int
    held_out_n_positive: int
    held_out_tm_mean: float
    cross_rally_rank1: float
    cross_rally_n_queries: int
    cf_guard_passed: bool
    seconds: float


@dataclass
class VariantResult:
    variant_id: str
    best_epoch: int
    best_held_out_rank1: float
    best_cross_rally_rank1: float
    best_passed_target: bool
    best_passed_cf_guard: bool
    final_epoch: int
    halted_reason: str | None
    epochs: list[EpochMetrics]


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _stack_pair_batch(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Interleave (crop_a, crop_b) into a single (2N, ...) tensor with labels.

    Returns (crops, identities, teams, tier_idx_per_crop) — each crop is one row.
    """
    crops = torch.cat([batch["crop_a"], batch["crop_b"]], dim=0)
    identities = torch.cat([batch["identity_a"], batch["identity_b"]], dim=0)
    teams = torch.cat([batch["team_a"], batch["team_b"]], dim=0)
    tiers = torch.cat([batch["tier_idx"], batch["tier_idx"]], dim=0)
    return crops, identities, teams, tiers


def _per_sample_supcon_weights(
    tier_per_crop: torch.Tensor,
    label_smoothing_mid: float,
) -> torch.Tensor:
    """Mid-tier crops contribute (1 - smoothing) to SupCon; everything else 1.0."""
    weights = torch.ones_like(tier_per_crop, dtype=torch.float32)
    weights[tier_per_crop == TIER_MID] = 1.0 - label_smoothing_mid
    return weights


def run_variant(
    cfg: TrainConfig,
    smoke: int = 0,
    train_pairs_cap: int = 0,
) -> VariantResult:
    _seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    logger.info("=== Variant %s on %s ===", cfg.variant_id, device)
    logger.info("Config: %s", json.dumps(asdict(cfg), default=str, indent=2))

    # ---------- Manifests + encodings ----------
    corpus_root = TRAINING_DATA_ROOT
    manifest = load_manifest(corpus_root)

    train_pairs = manifest.train_pairs
    if train_pairs_cap > 0:
        logger.info("CAP: training pairs limited to %d", train_pairs_cap)
        train_pairs = train_pairs[:train_pairs_cap]

    # Identity & team encodings span the full corpus (train + val) so eval-time
    # encoding is consistent.
    all_pairs = manifest.train_pairs + manifest.val_pairs
    id_enc = build_identity_encoding(all_pairs)
    team_enc = build_team_encoding(all_pairs)
    logger.info("Identity classes: %d, Team groups: %d", len(id_enc), len(team_enc))

    train_ds = PairDataset(
        pairs=train_pairs,
        corpus_root=corpus_root,
        identity_encoding=id_enc,
        team_encoding=team_enc,
        seed=cfg.seed,
        train_mode=True,
    )
    sampler = PairBatchSampler(
        pairs=train_pairs,
        pos_per_batch=cfg.pos_pairs_per_batch,
        easy_neg_per_batch=cfg.easy_neg_pairs_per_batch,
        hard_neg_per_batch=cfg.hard_neg_pairs_per_batch,
        seed=cfg.seed,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )

    steps_per_epoch = len(sampler)
    logger.info(
        "Train: %d pairs, %d steps/epoch, %d crops/batch",
        len(train_pairs), steps_per_epoch, cfg.batch_crops,
    )

    # ---------- Model + losses + optimizer ----------
    backbone = BackboneRunner(device=device)
    head = MLPHead(in_dim=384, hidden_dim=cfg.head_hidden, out_dim=cfg.head_out).to(device)
    supcon = SupConLoss(tau=cfg.tau).to(device)
    teammate_margin = TeammateMarginLoss(margin=cfg.teammate_margin_m).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg.epochs * steps_per_epoch),
    )

    use_amp = cfg.mixed_precision and device.type in {"cuda"}
    if cfg.mixed_precision and device.type == "mps":
        logger.info("AMP requested but MPS — disabling (autocast on MPS is partial)")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------- Eval cache ----------
    eval_cache = load_cache(EVAL_CACHE_PATH, EVAL_CACHE_META)
    logger.info(
        "Eval cache: %d held-out (%d scorable), %d cross-rally entries",
        len(eval_cache.held_out_events),
        sum(1 for m in eval_cache.held_out_events if m.is_scorable()),
        len(eval_cache.cross_rally_entries),
    )

    # ---------- Output dirs ----------
    variant_dir = WEIGHTS_ROOT / f"variant_{cfg.variant_id}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    epochs_log = variant_dir / "epochs.jsonl"
    epochs_log.write_text("")  # truncate

    manifest_sha = _file_sha(corpus_root / "manifest.json")
    pairs_sha = _file_sha(corpus_root / "candidate_pairs.jsonl")
    git_sha = _git_sha()

    # ---------- Training loop ----------
    epochs_metrics: list[EpochMetrics] = []
    best_epoch = -1
    best_rank1 = -1.0
    best_cross = -1.0
    best_passed_target = False
    best_passed_cf_guard = False
    halted_reason: str | None = None
    no_improve_streak = 0

    for epoch in range(1, cfg.epochs + 1):
        sampler.set_epoch(epoch)
        head.train()
        backbone._model.eval()  # belt-and-suspenders

        ep_t0 = time.time()
        ep_loss, ep_supcon, ep_tm = 0.0, 0.0, 0.0
        ep_tm_mean_running, ep_tm_count = 0.0, 0
        ep_var_running = 0.0
        n_batches_done = 0

        for step, batch in enumerate(loader):
            crops, identities, teams, tiers = _stack_pair_batch(batch)
            crops = crops.to(device, non_blocking=True)
            identities = identities.to(device, non_blocking=True)
            teams = teams.to(device, non_blocking=True)
            tiers_per_crop = tiers.to(device)

            with torch.no_grad():
                feats_backbone = backbone.forward(crops)  # (2N, 384)

            optimizer.zero_grad(set_to_none=True)
            amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if use_amp else _NullCtx()
            with amp_ctx:
                head_feats = head(feats_backbone)         # (2N, 128)
                weights = _per_sample_supcon_weights(tiers_per_crop, cfg.label_smoothing_mid)
                loss_supcon = supcon(head_feats, identities, weights=weights)
                tm_out = teammate_margin(head_feats, identities, teams)
                loss_tm = tm_out.loss
                loss = loss_supcon + cfg.lam_tm * loss_tm

            if not torch.isfinite(loss):
                logger.warning(
                    "epoch=%d step=%d non-finite loss — skipping batch", epoch, step,
                )
                continue

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                # Embedding variance (trace of covariance)
                center = head_feats - head_feats.mean(dim=0, keepdim=True)
                var = (center.pow(2).sum() / max(1, head_feats.shape[0] - 1)).item()

            ep_loss += float(loss.item())
            ep_supcon += float(loss_supcon.item())
            ep_tm += float(loss_tm.item())
            if tm_out.n_valid > 0:
                ep_tm_mean_running += tm_out.teammate_margin_mean
                ep_tm_count += 1
            ep_var_running += var
            n_batches_done += 1

            if smoke > 0 and n_batches_done >= smoke:
                logger.info(
                    "[smoke] step %d: loss=%.4f supcon=%.4f tm=%.4f tm_mean=%+.4f var=%.4f",
                    step, loss.item(), loss_supcon.item(), loss_tm.item(),
                    tm_out.teammate_margin_mean, var,
                )
                # Verify head got grad
                head_grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in head.parameters() if p.grad is not None
                ) ** 0.5
                logger.info("[smoke] head grad norm = %.6f", head_grad_norm)
                return VariantResult(
                    variant_id=cfg.variant_id,
                    best_epoch=epoch,
                    best_held_out_rank1=0.0,
                    best_cross_rally_rank1=0.0,
                    best_passed_target=False,
                    best_passed_cf_guard=False,
                    final_epoch=epoch,
                    halted_reason="smoke",
                    epochs=[],
                )

        if n_batches_done == 0:
            logger.error("epoch=%d no batches completed; aborting", epoch)
            break

        avg_loss = ep_loss / n_batches_done
        avg_supcon = ep_supcon / n_batches_done
        avg_tm = ep_tm / n_batches_done
        avg_tm_mean = ep_tm_mean_running / max(1, ep_tm_count)
        avg_var = ep_var_running / n_batches_done

        # Validation
        head.eval()
        ho = evaluate_held_out(eval_cache, head, device)
        cr = evaluate_cross_rally(eval_cache, head, device)
        cf_pass = cr.rank1 >= CROSS_RALLY_GUARD

        ep_metrics = EpochMetrics(
            epoch=epoch,
            loss_total=avg_loss,
            loss_supcon=avg_supcon,
            loss_tm=avg_tm,
            teammate_margin_train=avg_tm_mean,
            embedding_variance_train=avg_var,
            held_out_rank1=ho.rank1,
            held_out_n_scored=ho.n_scored,
            held_out_n_positive=ho.n_positive,
            held_out_tm_mean=ho.teammate_margin_mean,
            cross_rally_rank1=cr.rank1,
            cross_rally_n_queries=cr.n_queries,
            cf_guard_passed=cf_pass,
            seconds=time.time() - ep_t0,
        )
        epochs_metrics.append(ep_metrics)

        with epochs_log.open("a") as f:
            f.write(json.dumps(asdict(ep_metrics)) + "\n")

        logger.info(
            "epoch=%d loss=%.4f supcon=%.4f tm=%.4f tm_train=%+.4f var=%.4f"
            " | heldout=%.3f (%d/%d pos, scored=%d) tm_mean=%+.4f"
            " | cross=%.4f (n=%d) %s | %.1fs",
            epoch, avg_loss, avg_supcon, avg_tm, avg_tm_mean, avg_var,
            ho.rank1, ho.n_positive, ho.n_total, ho.n_scored, ho.teammate_margin_mean,
            cr.rank1, cr.n_queries, "✓" if cf_pass else "⚠CF",
            ep_metrics.seconds,
        )

        # Best checkpoint (subject to CF guard)
        if cf_pass and ho.rank1 > best_rank1:
            best_rank1 = ho.rank1
            best_cross = cr.rank1
            best_epoch = epoch
            best_passed_cf_guard = True
            best_passed_target = ho.rank1 >= HELDOUT_TARGET
            no_improve_streak = 0
            _save_checkpoint(
                variant_dir / "best.pt", head, cfg, ep_metrics, manifest_sha,
                pairs_sha, git_sha,
            )
            if best_passed_target:
                _maybe_update_global_best(
                    variant_dir / "best.pt", WEIGHTS_ROOT / "best.pt", ho.rank1, cr.rank1,
                )
        else:
            no_improve_streak += 1

        _save_checkpoint(
            variant_dir / "last.pt", head, cfg, ep_metrics, manifest_sha,
            pairs_sha, git_sha,
        )

        # Stop conditions per plan
        if epoch >= 10 and avg_tm_mean <= 0:
            halted_reason = (
                f"teammate_margin_train ≤ 0 after epoch {epoch} — "
                "data cannot invert anti-correlation prior"
            )
            logger.warning("HALT: %s", halted_reason)
            break
        if no_improve_streak >= cfg.early_stop_patience:
            halted_reason = (
                f"early stop: no improvement under CF guard for {no_improve_streak} epochs"
            )
            logger.info("STOP: %s", halted_reason)
            break

    return VariantResult(
        variant_id=cfg.variant_id,
        best_epoch=best_epoch,
        best_held_out_rank1=best_rank1 if best_rank1 >= 0 else 0.0,
        best_cross_rally_rank1=best_cross if best_cross >= 0 else 0.0,
        best_passed_target=best_passed_target,
        best_passed_cf_guard=best_passed_cf_guard,
        final_epoch=epochs_metrics[-1].epoch if epochs_metrics else 0,
        halted_reason=halted_reason,
        epochs=epochs_metrics,
    )


def _save_checkpoint(
    path: Path,
    head: torch.nn.Module,
    cfg: TrainConfig,
    metrics: EpochMetrics,
    manifest_sha: str,
    pairs_sha: str,
    git_sha: str,
) -> None:
    payload = {
        "head_state_dict": head.state_dict(),
        "config": asdict(cfg),
        "metrics": asdict(metrics),
        "manifest_sha": manifest_sha,
        "pairs_sha": pairs_sha,
        "git_sha": git_sha,
    }
    torch.save(payload, path)


def _maybe_update_global_best(
    variant_best: Path,
    global_best: Path,
    primary: float,
    cross_rally: float,
) -> None:
    """Overwrite global best if this variant beats it on primary metric."""
    if not global_best.exists():
        torch.save(torch.load(variant_best, map_location="cpu", weights_only=False), global_best)
        logger.info(
            "  Set global best: primary=%.3f cross=%.3f", primary, cross_rally,
        )
        return
    existing = torch.load(global_best, map_location="cpu", weights_only=False)
    existing_primary = float(existing["metrics"]["held_out_rank1"])
    if primary > existing_primary:
        torch.save(torch.load(variant_best, map_location="cpu", weights_only=False), global_best)
        logger.info(
            "  Updated global best: primary %.3f → %.3f cross=%.3f",
            existing_primary, primary, cross_rally,
        )


class _NullCtx:
    def __enter__(self) -> None: return None
    def __exit__(self, *a) -> None: return None
