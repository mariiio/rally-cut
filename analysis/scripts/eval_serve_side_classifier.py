#!/usr/bin/env python3
"""VideoMAE linear probe for serve-side classification.

Tests whether cached VideoMAE features (768-dim, stride=12) contain enough
signal to predict which court side served, using LOO-video cross-validation.

Sweeps frame offset as a hyperparameter:
  Tier 1 (single window): t ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0}s from rally_start
  Tier 2 (clip mean):     windows covering [-1s,+1s], [0,+2s], [0,+3s] from rally_start

Reports accuracy per offset/window so we can see where signal peaks.

Gate: >=85% LOO accuracy -> scope production integration.
      70-85% -> error analysis, try fine-tuning.
      <70%  -> signal not in VideoMAE representation.

Read-only experiment. No production code modified.

Usage:
    cd analysis
    uv run python scripts/eval_serve_side_classifier.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.temporal.features import FeatureCache, FeatureMetadata

STRIDE = 12
WINDOW_SIZE = 16

# Tier 1: single-window offsets from rally_start (seconds)
SINGLE_OFFSETS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

# Tier 2: clip windows [start_offset, end_offset] from rally_start (seconds)
CLIP_WINDOWS = [(-1.0, 1.0), (0.0, 2.0), (0.0, 3.0)]


@dataclass
class RallyBase:
    """Rally metadata + label, features extracted on demand per offset."""

    rally_id: str
    video_id: str
    start_ms: int
    serve_near: bool  # True if near side served (after side-switch correction)
    # Full video features + metadata for on-demand window extraction
    all_features: np.ndarray  # (num_windows, 768) — shared ref, not copied
    fps: float
    stride: int


def _load_side_switches() -> dict[str, list[int]]:
    """Load sideSwitches per video from player_matching_gt_json."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json
            FROM videos
            WHERE id IN (
                SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL
            )
        """)
        result: dict[str, list[int]] = {}
        for vid, gt in cur.fetchall():
            if isinstance(gt, dict):
                result[vid] = list(
                    gt.get("sideSwitches", gt.get("side_switches", []))
                )
            else:
                result[vid] = []
    return result


def _load_rallies_with_serving_team() -> list[dict]:
    """Load all rallies that have gt_serving_team, ordered by video + start_ms."""
    query = """
        SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team
        FROM rallies r
        WHERE r.gt_serving_team IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """
    rows = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(query)
        for rid, vid, start_ms, gt_team in cur.fetchall():
            rows.append({
                "rally_id": rid,
                "video_id": vid,
                "start_ms": start_ms or 0,
                "gt_serving_team": gt_team,
            })
    return rows


def _compute_side_flipped(rally_index: int, switch_indices: list[int]) -> bool:
    """Return True if near/far->A/B mapping is flipped at this rally index."""
    flipped = False
    for si in switch_indices:
        if si <= rally_index:
            flipped = not flipped
    return flipped


def _sec_to_window(sec: float, fps: float, stride: int, n_windows: int) -> int:
    """Convert a time in seconds to the nearest feature window index."""
    idx = int(sec * fps / stride)
    return max(0, min(idx, n_windows - 1))


def extract_feature_single(rally: RallyBase, offset_s: float) -> np.ndarray:
    """Extract single-window feature at rally_start + offset_s."""
    target_sec = rally.start_ms / 1000.0 + offset_s
    idx = _sec_to_window(target_sec, rally.fps, rally.stride, len(rally.all_features))
    return rally.all_features[idx]


def extract_feature_clip(
    rally: RallyBase, start_offset_s: float, end_offset_s: float,
) -> np.ndarray:
    """Extract mean feature over a time window relative to rally_start."""
    start_sec = rally.start_ms / 1000.0 + start_offset_s
    end_sec = rally.start_ms / 1000.0 + end_offset_s
    n = len(rally.all_features)
    i0 = _sec_to_window(start_sec, rally.fps, rally.stride, n)
    i1 = _sec_to_window(end_sec, rally.fps, rally.stride, n)
    # Ensure at least 1 window
    if i1 <= i0:
        i1 = i0 + 1
    i1 = min(i1, n)
    return rally.all_features[i0:i1].mean(axis=0)


def build_rally_bases(verbose: bool = True) -> list[RallyBase]:
    """Load all rallies with gt_serving_team and attach full video features."""
    if verbose:
        print("Loading rallies with gt_serving_team...")
    rallies = _load_rallies_with_serving_team()
    if verbose:
        print(f"  {len(rallies)} rallies across "
              f"{len({r['video_id'] for r in rallies})} videos")

    if verbose:
        print("Loading side switches...")
    side_switches = _load_side_switches()

    if verbose:
        print("Loading evaluation videos (for content_hash)...")
    eval_videos = load_evaluation_videos(require_ground_truth=False)
    video_map: dict[str, EvaluationVideo] = {v.id: v for v in eval_videos}

    cache = FeatureCache()

    by_video: dict[str, list[dict]] = defaultdict(list)
    for r in rallies:
        by_video[r["video_id"]].append(r)

    bases: list[RallyBase] = []
    skipped_no_video = 0
    skipped_no_features = 0

    for vid, vid_rallies in sorted(by_video.items()):
        ev = video_map.get(vid)
        if ev is None:
            skipped_no_video += len(vid_rallies)
            continue

        cached = cache.get(ev.content_hash, stride=STRIDE)
        if cached is None:
            skipped_no_features += len(vid_rallies)
            if verbose:
                print(f"  SKIP {ev.filename}: no cached features")
            continue

        features, metadata = cached
        switches = side_switches.get(vid, [])

        for idx, r in enumerate(vid_rallies):
            flipped = _compute_side_flipped(idx, switches)
            gt_team = r["gt_serving_team"]
            # Convention: A = near, B = far. If flipped, swap.
            serve_near = (gt_team == "A") if not flipped else (gt_team == "B")

            bases.append(RallyBase(
                rally_id=r["rally_id"],
                video_id=vid,
                start_ms=r["start_ms"],
                serve_near=serve_near,
                all_features=features,
                fps=metadata.fps,
                stride=STRIDE,
            ))

    if verbose:
        print(f"  Built {len(bases)} rally bases "
              f"(skipped: {skipped_no_video} no-video, "
              f"{skipped_no_features} no-features)")
        if bases:
            near = sum(1 for b in bases if b.serve_near)
            far = len(bases) - near
            print(f"  Class balance: near={near} far={far} "
                  f"majority-class floor={max(near, far)/len(bases)*100:.1f}%")

    return bases


def run_loo_cv(
    bases: list[RallyBase],
    feature_fn: callable,
    classifier_name: str = "logistic",
    verbose: bool = False,
) -> tuple[float, list[tuple[str, int, int]]]:
    """Run LOO-video CV. Returns (accuracy, per_video_results)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    by_video: dict[str, list[RallyBase]] = defaultdict(list)
    for b in bases:
        by_video[b.video_id].append(b)

    video_ids = sorted(by_video.keys())
    total_correct = 0
    total_count = 0
    per_video: list[tuple[str, int, int]] = []

    for held_out_vid in video_ids:
        test = by_video[held_out_vid]
        train = [b for vid in video_ids if vid != held_out_vid for b in by_video[vid]]

        X_train = np.array([feature_fn(b) for b in train])
        y_train = np.array([int(b.serve_near) for b in train])
        X_test = np.array([feature_fn(b) for b in test])
        y_test = np.array([int(b.serve_near) for b in test])

        if len(np.unique(y_train)) < 2:
            continue

        if classifier_name == "logistic":
            clf = LogisticRegression(max_iter=1000, C=1.0)
        elif classifier_name == "mlp":
            clf = MLPClassifier(
                hidden_layer_sizes=(128,), max_iter=500,
                early_stopping=True, random_state=42,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        correct = int((preds == y_test).sum())
        n = len(y_test)

        per_video.append((held_out_vid, correct, n))
        total_correct += correct
        total_count += n

        if verbose:
            print(f"    {held_out_vid[:8]}: {correct}/{n} ({correct/n*100:.0f}%)")

    acc = total_correct / total_count if total_count > 0 else 0.0
    return acc, per_video


def main() -> int:
    bases = build_rally_bases()
    if not bases:
        print("ERROR: no rally bases built")
        return 1

    # ── Tier 1: single-window sweep ──────────────────────────────────
    print("\n" + "=" * 70)
    print("TIER 1: Single-window probe (sweep offset from rally_start)")
    print("=" * 70)

    tier1_results: list[tuple[str, float, list[tuple[str, int, int]]]] = []

    for offset in SINGLE_OFFSETS:
        label = f"t=+{offset:.2f}s"
        print(f"\n--- {label} + logistic ---")
        acc, per_vid = run_loo_cv(
            bases,
            feature_fn=lambda b, o=offset: extract_feature_single(b, o),
            classifier_name="logistic",
            verbose=True,
        )
        tier1_results.append((label, acc, per_vid))
        print(f"  => {acc*100:.1f}%")

    # ── Tier 2: clip-mean sweep ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("TIER 2: Clip-mean probe (sweep window around rally_start)")
    print("=" * 70)

    tier2_results: list[tuple[str, float, list[tuple[str, int, int]]]] = []

    for start_off, end_off in CLIP_WINDOWS:
        label = f"[{start_off:+.1f}s, {end_off:+.1f}s]"
        print(f"\n--- {label} + logistic ---")
        acc, per_vid = run_loo_cv(
            bases,
            feature_fn=lambda b, s=start_off, e=end_off: extract_feature_clip(b, s, e),
            classifier_name="logistic",
            verbose=True,
        )
        tier2_results.append((label, acc, per_vid))
        print(f"  => {acc*100:.1f}%")

    # ── Best offset with MLP ─────────────────────────────────────────
    all_results = tier1_results + tier2_results
    best_label, best_acc, _ = max(all_results, key=lambda x: x[1])

    print("\n" + "=" * 70)
    print(f"MLP(128) on best offset: {best_label}")
    print("=" * 70)

    # Re-run best with MLP
    # Find the matching feature_fn
    best_is_clip = best_label.startswith("[")
    if best_is_clip:
        # Parse clip window from label
        for start_off, end_off in CLIP_WINDOWS:
            if f"[{start_off:+.1f}s, {end_off:+.1f}s]" == best_label:
                best_fn = lambda b, s=start_off, e=end_off: extract_feature_clip(b, s, e)
                break
    else:
        for offset in SINGLE_OFFSETS:
            if f"t=+{offset:.2f}s" == best_label:
                best_fn = lambda b, o=offset: extract_feature_single(b, o)
                break

    mlp_acc, mlp_per_vid = run_loo_cv(
        bases, feature_fn=best_fn, classifier_name="mlp", verbose=True,
    )
    print(f"  => {mlp_acc*100:.1f}%")

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Config':<30s} {'Acc':>6s}  Gate")
    print(f"  {'-'*30} {'-'*6}  {'-'*11}")

    for label, acc, _ in tier1_results:
        gate = "PASS" if acc >= 0.85 else ("INVESTIGATE" if acc >= 0.70 else "FAIL")
        print(f"  {label + ' (logistic)':<30s} {acc*100:5.1f}%  [{gate}]")

    for label, acc, _ in tier2_results:
        gate = "PASS" if acc >= 0.85 else ("INVESTIGATE" if acc >= 0.70 else "FAIL")
        print(f"  {label + ' (logistic)':<30s} {acc*100:5.1f}%  [{gate}]")

    gate = "PASS" if mlp_acc >= 0.85 else ("INVESTIGATE" if mlp_acc >= 0.70 else "FAIL")
    print(f"  {best_label + ' (MLP)':<30s} {mlp_acc*100:5.1f}%  [{gate}]")

    # Overall best
    final_best_acc = max(
        max(r[1] for r in all_results),
        mlp_acc,
    )
    print(f"\n  Best overall: {final_best_acc*100:.1f}%")

    if final_best_acc >= 0.85:
        print("  -> Gate PASSED. Scope production integration.")
    elif final_best_acc >= 0.70:
        print("  -> Investigate per-video errors. Consider Tier 2 fine-tuning.")
    else:
        print("  -> Signal not found in VideoMAE features. Revisit assumptions.")

    # ── Per-video error breakdown for best config ────────────────────
    _, _, best_per_vid = max(all_results, key=lambda x: x[1])
    print(f"\n  Per-video breakdown (best logistic):")
    best_per_vid.sort(key=lambda x: x[1] / x[2] if x[2] > 0 else 1.0)
    for vid, c, n in best_per_vid:
        print(f"    {vid[:8]}: {c}/{n} ({c/n*100:.0f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
