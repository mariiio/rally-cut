"""Quick validation: train a BINARY focal-BCE MSTCN on 1 LOO fold.

Purpose: verify that fixing the class-weight bug (MSTCN collapsing to predict
background) actually lifts F1 above the 36.9% linear-probe ceiling.

If this hits ≥60% binary F1 on a held-out video after 20 epochs → the fix
works, proceed to full 68-fold. If it still floors at ~40% → feature
ceiling is real, pivot to E2E-Spot.
"""
# ruff: noqa: E701, E702  # terse multi-statement lines acceptable in debug probes
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tnf

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig
from scripts.eval_action_detection import load_rallies_with_action_gt, match_contacts

STRIDE = 4
BACKBONE = "videomae-v1"
WINDOW_SIZE = 16
SIGMA = 3.0


def build_binary_head() -> MSTCN:
    return MSTCN(MSTCNConfig(
        feature_dim=768, num_stages=2, num_layers=8, hidden_dim=64,
        num_classes=2, dropout=0.3, ball_feature_dim=0,
    ))


def _binary_target(num_windows, stride, eff_fps, rallies, sigma):
    """Binary soft target: max Gaussian over all GT contacts, any class."""
    centers = np.arange(num_windows) * stride + WINDOW_SIZE // 2
    tgt = np.zeros(num_windows, dtype=np.float32)
    for r in rallies:
        rally_fps = float(r.fps or eff_fps)
        rstart = int(round((r.start_ms / 1000.0) * eff_fps))
        for gt in r.gt_labels:
            gt_eff = rstart + int(round(gt.frame * eff_fps / rally_fps))
            diffs = centers - gt_eff
            m = np.abs(diffs) <= 3 * sigma
            if not m.any(): continue
            g = np.exp(-(diffs[m] ** 2) / (2 * sigma * sigma))
            tgt[m] = np.maximum(tgt[m], g)
    return tgt


def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    """Soft focal BCE. logits: (T,), targets: (T,) in [0,1]."""
    p = torch.sigmoid(logits)
    bce = tnf.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    # pt: probability assigned to the "correct" direction for soft targets
    pt = targets * p + (1 - targets) * (1 - p)
    focal = (1 - pt).clamp(min=1e-6) ** gamma
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return (alpha_t * focal * bce).mean()


def weighted_bce(logits, targets, pos_weight=50.0):
    """Standard pos_weight BCE. Canonical fix for extreme class imbalance."""
    pw = torch.tensor(pos_weight, device=logits.device)
    return tnf.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pw, reduction="mean",
    )


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device={device}")

    all_rallies = load_rallies_with_action_gt()
    action_ids = {r.video_id for r in all_rallies}
    videos = [v for v in load_evaluation_videos(require_ground_truth=True) if v.id in action_ids]
    by_v = defaultdict(list)
    for r in all_rallies: by_v[r.video_id].append(r)

    cache = FeatureCache()
    videos_loaded = []
    for v in videos:
        c = cache.get(v.content_hash, stride=STRIDE, backbone=BACKBONE)
        if c is None: continue
        features, meta = c
        tgt = _binary_target(features.shape[0], STRIDE, meta.fps, by_v[v.id], SIGMA)
        videos_loaded.append({
            "id": v.id, "features": features.astype(np.float32),
            "targets": tgt, "eff_fps": float(meta.fps), "rallies": by_v[v.id],
        })
    print(f"loaded {len(videos_loaded)} videos")

    # Hold out video 0 (073cb11b / wuwu)
    held = videos_loaded[0]
    train = videos_loaded[1:]
    print(f"held out: {held['id'][:8]} "
          f"({sum((held['targets']>0.5).astype(int))} pos frames, "
          f"{held['features'].shape[0]} total)")
    total_pos = sum((vd['targets']>0.5).sum() for vd in train)
    total_frames = sum(vd['features'].shape[0] for vd in train)
    print(f"train: {len(train)} videos, {total_pos} pos / {total_frames} total = {100*total_pos/total_frames:.2f}%")

    model = build_binary_head().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20)

    # Pre-load tensors
    train_tensors = []
    for vd in train:
        x = torch.from_numpy(vd["features"]).transpose(0, 1).unsqueeze(0).to(device)  # (1, 768, T)
        y = torch.from_numpy(vd["targets"]).unsqueeze(0).to(device)  # (1, T)
        train_tensors.append((x, y))

    rng = np.random.default_rng(42)
    for ep in range(20):
        order = rng.permutation(len(train_tensors))
        total_loss = 0.0
        for idx in order:
            x, y_soft = train_tensors[idx]
            stage_logits = model.forward_all_stages(x)  # list of (1, 2, T)
            loss = torch.zeros((), device=device)
            for lg in stage_logits:
                # Binary: use class-1 logit (contact), or diff of class-1 and class-0
                # MSTCN was configured with num_classes=2. Treat channel-1 as contact logit.
                logit_contact = lg[:, 1, :] - lg[:, 0, :]  # (1, T) — margin logit
                loss = loss + weighted_bce(logit_contact.squeeze(0), y_soft.squeeze(0), pos_weight=50.0)
            loss = loss / len(stage_logits)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += float(loss.item())
        scheduler.step()
        if ep in (0, 4, 9, 14, 19):
            print(f"  ep{ep:02d} loss={total_loss/len(train_tensors):.4f}")

    # Evaluate on held
    model.eval()
    with torch.no_grad():
        x_h = torch.from_numpy(held["features"]).transpose(0, 1).unsqueeze(0).to(device)
        stage_logits = model.forward_all_stages(x_h)
        lg = stage_logits[-1]
        logit_contact = lg[0, 1, :] - lg[0, 0, :]
        prob = torch.sigmoid(logit_contact).cpu().numpy()
    print(f"\npred prob stats: max={prob.max():.3f} mean={prob.mean():.3f} "
          f"frac>0.5={(prob>0.5).mean():.3%} frac>0.3={(prob>0.3).mean():.3%}")

    # Decode + score via match_contacts
    from scipy.signal import find_peaks

    best_f1 = 0.0
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6]:
        peaks, _ = find_peaks(prob, height=thr, distance=3)
        accepted_abs = [int(w * STRIDE + WINDOW_SIZE // 2) for w in peaks]
        eff_fps = held["eff_fps"]
        tp = fp = fn = 0
        for rally in held["rallies"]:
            rf = float(rally.fps or eff_fps)
            rstart = int(round((rally.start_ms / 1000.0) * eff_fps))
            rend = rstart + int(round((rally.frame_count or 0) * eff_fps / rf))
            rp = []
            for af in accepted_abs:
                if rstart <= af < rend:
                    rp.append({"frame": int(round((af - rstart) * rf / eff_fps)), "action": "contact", "playerTrackId": -1})
            tol = max(1, round(rf * 0.233))
            matches, unmatched = match_contacts(rally.gt_labels, rp, tolerance=tol)
            for m in matches:
                if m.pred_frame is None: fn += 1
                else: tp += 1
            fp += len(unmatched)
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn); f1 = 2*p*r / max(1e-9, p+r)
        mark = " ← best" if f1 > best_f1 else ""
        if f1 > best_f1: best_f1 = f1
        print(f"  thr={thr:.2f}  tp={tp} fp={fp} fn={fn}  P={p:.1%} R={r:.1%} F1={f1:.1%}{mark}")

    verdict = "FIX WORKS" if best_f1 >= 0.50 else "STILL UNDERPOWERED"
    print(f"\nheld-out best binary F1 = {best_f1:.1%} → {verdict}")


if __name__ == "__main__":
    main()
