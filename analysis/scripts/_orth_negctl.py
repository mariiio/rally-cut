"""Negative control for the orthogonality probe: is the high coverage real?

We saw 98.7% of GT contacts have VideoMAE prob ≥ 0.30 within ±tolerance.
But class-balanced logistic regression could be pushing probs high
EVERYWHERE. Check: sample random non-contact frames and see their max
prob in ±tolerance too. If near-identical to GT contacts → signal is
diffuse, fusion won't help. If substantially lower → signal is real.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.orthogonality_probe import _videomae_probs_per_video

STRIDE = 4
WINDOW_SIZE = 16
TOLERANCE_MS = 233


def main() -> None:
    rallies = load_rallies_with_action_gt()
    action_ids = {r.video_id for r in rallies}
    videos = [v for v in load_evaluation_videos(require_ground_truth=True)
              if v.id in action_ids]

    cache = FeatureCache()
    probs_per_video, eff_fps = _videomae_probs_per_video(videos, cache)

    # Build GT contact frame sets per video (absolute eff-fps frames)
    by_video = defaultdict(list)
    for r in rallies:
        by_video[r.video_id].append(r)

    rng = np.random.default_rng(42)
    gt_probs = []
    neg_probs = []
    random_probs = []

    for v in videos:
        if v.id not in probs_per_video:
            continue
        probs = probs_per_video[v.id]
        e_fps = eff_fps[v.id]

        gt_eff_frames = set()
        all_rally_span = []
        for rally in by_video[v.id]:
            rf = float(rally.fps or e_fps)
            rstart = int(round((rally.start_ms / 1000.0) * e_fps))
            rend = rstart + int(round((rally.frame_count or 0) * e_fps / rf))
            all_rally_span.append((rstart, rend))
            for gt in rally.gt_labels:
                f_eff = rstart + int(round(gt.frame * e_fps / rf))
                gt_eff_frames.add(f_eff)

        tol_eff = int(round(TOLERANCE_MS * e_fps / 1000))

        def _max_prob_near(f_eff: int) -> float:
            lo = max(0, (f_eff - tol_eff - WINDOW_SIZE) // STRIDE)
            hi = min(len(probs), (f_eff + tol_eff + WINDOW_SIZE) // STRIDE + 1)
            return float(probs[lo:hi].max()) if hi > lo else 0.0

        # GT contact probs
        for f in gt_eff_frames:
            gt_probs.append(_max_prob_near(f))

        # Negative control 1: sample random in-rally frames > 1s from any GT
        for rstart, rend in all_rally_span:
            span_frames = max(0, rend - rstart)
            if span_frames < 60:
                continue
            for _ in range(5):
                attempt = 0
                while attempt < 10:
                    rf = rng.integers(rstart, rend)
                    if all(abs(rf - g) > int(e_fps) for g in gt_eff_frames):
                        neg_probs.append(_max_prob_near(rf))
                        break
                    attempt += 1

        # Negative control 2: uniformly random windows anywhere in video
        n_total = len(probs) * STRIDE
        for _ in range(20):
            rf = int(rng.integers(WINDOW_SIZE, max(WINDOW_SIZE + 1, n_total - WINDOW_SIZE)))
            random_probs.append(_max_prob_near(rf))

    out_lines: list[str] = []

    def _summary(label: str, arr: list[float]) -> None:
        a = np.array(arr)
        out_lines.append(
            f"{label}: n={len(a)}  mean={a.mean():.3f}  p25={np.percentile(a,25):.3f}  "
            f"median={np.percentile(a,50):.3f}  p75={np.percentile(a,75):.3f}"
        )
        for thr in [0.3, 0.5, 0.7, 0.9]:
            out_lines.append(f"  frac ≥ {thr}: {(a >= thr).mean():.1%}")

    _summary("GT contacts", gt_probs)
    out_lines.append("")
    _summary("Random in-rally non-contact frames", neg_probs)
    out_lines.append("")
    _summary("Uniform random frames", random_probs)

    out_path = Path("reports/orthogonality_negctl_2026_04_19.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines))
    for line in out_lines:
        print(line, flush=True)
    print(f"\n→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
