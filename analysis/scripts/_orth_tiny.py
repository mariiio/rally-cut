"""Minimal negative-control: 5 folds only, write results immediately."""
# ruff: noqa: N806  # sklearn convention: X/Y for feature/label matrices
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.train_videomae_contact_probe import _collect_video_windows

STRIDE = 4
WINDOW_SIZE = 16
TOLERANCE_MS = 233


def main():
    out = Path("reports/orth_tiny_2026_04_19.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(msg: str):
        print(msg, flush=True)
        lines.append(msg)
        out.write_text("\n".join(lines))

    t0 = time.time()
    rallies = load_rallies_with_action_gt()
    action_ids = {r.video_id for r in rallies}
    videos = [v for v in load_evaluation_videos(require_ground_truth=True)
              if v.id in action_ids]
    by_video = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)

    cache = FeatureCache()
    log("loading 5 videos for LOO...")
    loaded = []
    for v in videos[:5]:
        vw = _collect_video_windows(v, by_video[v.id], cache, STRIDE, "videomae-v1", radius_frames=3)
        if vw is None:
            continue
        loaded.append(vw)
    log(f"loaded {len(loaded)} videos ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(42)
    all_gt = []
    all_neg = []

    for held in loaded:
        t_fold = time.time()
        train = [vw for vw in loaded if vw.video_id != held.video_id]
        X = np.concatenate([vw.features for vw in train], axis=0)
        y = np.concatenate([vw.labels for vw in train], axis=0)
        clf = LogisticRegression(solver="lbfgs", max_iter=500, class_weight="balanced")
        clf.fit(X, y)
        probs = clf.predict_proba(held.features)[:, 1]
        log(f"  fold {held.video_id[:8]}: T={len(probs)} mean_prob={probs.mean():.3f} max={probs.max():.3f} ({time.time()-t_fold:.0f}s)")

        # Build GT eff-frames
        gt_eff = set()
        rally_spans = []
        for r in by_video[held.video_id]:
            rf = float(r.fps or held.effective_fps)
            rstart = int(round((r.start_ms / 1000.0) * held.effective_fps))
            rend = rstart + int(round((r.frame_count or 0) * held.effective_fps / rf))
            rally_spans.append((rstart, rend))
            for gt in r.gt_labels:
                gt_eff.add(rstart + int(round(gt.frame * held.effective_fps / rf)))

        tol_eff = int(round(TOLERANCE_MS * held.effective_fps / 1000))

        def _max_near(f_eff):
            lo = max(0, (f_eff - tol_eff - WINDOW_SIZE) // STRIDE)
            hi = min(len(probs), (f_eff + tol_eff + WINDOW_SIZE) // STRIDE + 1)
            return float(probs[lo:hi].max()) if hi > lo else 0.0

        for f in gt_eff:
            all_gt.append(_max_near(f))

        # Random non-contact in-rally frames, > 1s from any GT
        for rstart, rend in rally_spans:
            if rend - rstart < 60:
                continue
            for _ in range(10):
                attempt = 0
                while attempt < 15:
                    rf = rng.integers(rstart, rend)
                    if all(abs(rf - g) > int(held.effective_fps) for g in gt_eff):
                        all_neg.append(_max_near(rf))
                        break
                    attempt += 1

    log("")
    log(f"GT contacts (n={len(all_gt)}):")
    if all_gt:
        a = np.array(all_gt)
        log(f"  mean={a.mean():.3f} p25={np.percentile(a,25):.3f} median={np.percentile(a,50):.3f} p75={np.percentile(a,75):.3f}")
        for thr in [0.3, 0.5, 0.7, 0.9]:
            log(f"  frac ≥ {thr}: {(a >= thr).mean():.1%}")
    log("")
    log(f"Random in-rally non-contact frames (n={len(all_neg)}):")
    if all_neg:
        a = np.array(all_neg)
        log(f"  mean={a.mean():.3f} p25={np.percentile(a,25):.3f} median={np.percentile(a,50):.3f} p75={np.percentile(a,75):.3f}")
        for thr in [0.3, 0.5, 0.7, 0.9]:
            log(f"  frac ≥ {thr}: {(a >= thr).mean():.1%}")

    log(f"\nTotal time: {time.time()-t0:.0f}s")
    log(f"→ {out}")


if __name__ == "__main__":
    main()
