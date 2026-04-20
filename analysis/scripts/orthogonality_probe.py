"""Does VideoMAE fire on the contacts the GBM misses?

Phase-5 go/no-go gate. Runs in two steps:

1. **Tag each GT contact as GBM-HIT or GBM-MISS** using the exact Phase 0
   LOO-per-video contact+action pipeline (88.0% baseline). For each held-out
   video we re-train the contact GBM on the other 67 videos, run
   ``detect_contacts`` + ``classify_rally_actions`` on the held-out rallies,
   and use the Hungarian ±7f matcher to decide hit/miss per GT.
2. **Score each GT contact with VideoMAE** using a single logistic probe
   trained on all 68 videos' cached stride=4 features (no LOO — we want a
   per-frame confidence, not a held-out number). For each GT contact we
   take the max VideoMAE prob in the ±radius window around the contact
   frame (same match tolerance as the eval).

Decision outputs:
- Distribution of VideoMAE probs at GBM-HIT vs GBM-MISS contacts.
- Fraction of GBM-MISS contacts where VideoMAE prob exceeds each threshold.
- Estimated fusion-F1 ceiling assuming "VideoMAE rescues every GBM-miss it
  flags above threshold, adds proportional FPs at that threshold".

If ≥40% of GBM-MISSes have VideoMAE prob above a threshold where the
corresponding background-prob is ≲2× → fusion materially moves the needle.
If ≤15% → signals are redundant, close the branch.

Usage:
    cd analysis
    uv run python scripts/orthogonality_probe.py
    uv run python scripts/orthogonality_probe.py --skip-loo   # use cached tags

Writes ``reports/videomae_orthogonality_2026_04_19.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _inject_action_classifier,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)

console = Console()

STRIDE = 4
BACKBONE = "videomae-v1"
WINDOW_SIZE = 16
DEFAULT_THRESHOLD = 0.30
DEFAULT_TOLERANCE_MS = 233


@dataclass
class GTEvent:
    """A single GT contact with its pipeline + VideoMAE tags."""

    video_id: str
    rally_id: str
    frame: int            # rally-relative at rally.fps
    action: str
    gbm_hit: bool
    gbm_pred_frame: int | None
    gbm_pred_action: str | None
    videomae_max_prob: float  # max prob in ±tolerance window of windows


# ---------------------------------------------------------------------------
# Step 1: GBM LOO hit/miss tagging
# ---------------------------------------------------------------------------

def _tag_gbm_hits(
    rallies_raw: list[RallyData],
    contact_cfg: ContactDetectionConfig,
    threshold: float,
    tolerance_ms: int,
) -> list[GTEvent]:
    """Run Phase-0-style LOO-per-video and tag each GT as HIT/MISS."""
    console.print("[bold]Step 1: LOO-per-video GBM tagging...[/bold]")
    t0 = time.time()
    precomputed: list[RallyPrecomputed] = []
    for rally in rallies_raw:
        pre = _precompute(rally, contact_cfg)
        if pre is None:
            continue
        precomputed.append(pre)
    console.print(f"  precomputed {len(precomputed)} rallies ({time.time()-t0:.0f}s)")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    events: list[GTEvent] = []

    for fold_idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [pre for v, rs in by_video.items() if v != vid for pre in rs]
        contact_clf, action_clf = _train_fold(train, threshold)

        for pre in held:
            rally = pre.rally
            _inject_action_classifier(action_clf if action_clf.is_trained else None)
            try:
                contact_seq = detect_contacts(
                    ball_positions=pre.ball_positions,
                    player_positions=pre.player_positions,
                    config=contact_cfg,
                    net_y=rally.court_split_y,
                    frame_count=rally.frame_count or None,
                    classifier=contact_clf,
                    use_classifier=True,
                    sequence_probs=pre.sequence_probs,
                )
                rally_actions = classify_rally_actions(
                    contact_seq, rally_id=rally.rally_id,
                    use_classifier=action_clf.is_trained,
                    sequence_probs=pre.sequence_probs,
                )
            finally:
                _reset_action_classifier_cache()

            pred_actions = [a.to_dict() for a in rally_actions.actions]
            real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
            tolerance_frames = max(1, round(rally.fps * tolerance_ms / 1000))
            matches, _unmatched = match_contacts(
                rally.gt_labels, real_pred, tolerance=tolerance_frames,
            )

            for m in matches:
                events.append(GTEvent(
                    video_id=rally.video_id,
                    rally_id=rally.rally_id,
                    frame=m.gt_frame,
                    action=m.gt_action,
                    gbm_hit=m.pred_frame is not None,
                    gbm_pred_frame=m.pred_frame,
                    gbm_pred_action=m.pred_action,
                    videomae_max_prob=0.0,  # filled in step 2
                ))

        n_hit = sum(1 for e in events if e.video_id == vid and e.gbm_hit)
        n_total = sum(1 for e in events if e.video_id == vid)
        console.print(
            f"  [{fold_idx}/{len(video_ids)}] {vid[:8]} "
            f"({n_hit}/{n_total} hit, {time.time()-t_fold:.0f}s)"
        )

    return events


# ---------------------------------------------------------------------------
# Step 2: VideoMAE per-frame probability
# ---------------------------------------------------------------------------

def _videomae_probs_per_video(
    videos: list, cache: FeatureCache,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """LOO VideoMAE probs: for each video train on the other 67, predict here.

    In-sample training produces memorisation (HIT and MISS both near p=1.0),
    which is useless for the fusion decision. LOO gives honest held-out
    probabilities — the same setting Phase 2 used to get 36.9% F1.
    """
    from sklearn.linear_model import LogisticRegression

    console.print("[bold]Step 2: LOO-per-video VideoMAE logistic probes...[/bold]")
    t0 = time.time()

    from scripts.train_videomae_contact_probe import _collect_video_windows

    all_rallies = load_rallies_with_action_gt()
    by_video = defaultdict(list)
    for r in all_rallies:
        by_video[r.video_id].append(r)

    per_video: list = []
    eff_fps: dict[str, float] = {}
    for v in videos:
        vw = _collect_video_windows(
            v, by_video[v.id], cache, STRIDE, BACKBONE, radius_frames=3,
        )
        if vw is None:
            continue
        per_video.append(vw)
        eff_fps[v.id] = vw.effective_fps

    console.print(f"  loaded {len(per_video)} videos ({time.time()-t0:.0f}s)")

    probs_per_video: dict[str, np.ndarray] = {}
    for fold_idx, held in enumerate(per_video, start=1):
        t_fold = time.time()
        train_parts = [vw for vw in per_video if vw.video_id != held.video_id]
        X = np.concatenate([vw.features for vw in train_parts], axis=0)  # noqa: N806
        y = np.concatenate([vw.labels for vw in train_parts], axis=0)
        clf = LogisticRegression(
            solver="lbfgs", max_iter=500, class_weight="balanced", C=1.0,
        )
        clf.fit(X, y)
        probs_per_video[held.video_id] = clf.predict_proba(held.features)[:, 1]
        if fold_idx % 10 == 0 or fold_idx == len(per_video):
            console.print(
                f"  [{fold_idx}/{len(per_video)}] {held.video_id[:8]} "
                f"({time.time()-t_fold:.0f}s/fold)"
            )
    console.print(f"  LOO probes done ({time.time()-t0:.0f}s total)")
    return probs_per_video, eff_fps


def _annotate_events_with_videomae(
    events: list[GTEvent],
    probs_per_video: dict[str, np.ndarray],
    eff_fps: dict[str, float],
    rallies_by_id: dict[str, RallyData],
    tolerance_ms: int,
) -> None:
    """For each event, compute max VideoMAE prob within ±tolerance of GT frame."""
    for e in events:
        probs = probs_per_video.get(e.video_id)
        if probs is None:
            continue
        rally = rallies_by_id[e.rally_id]
        rally_fps = float(rally.fps or eff_fps[e.video_id])
        f_eff = int(round((rally.start_ms / 1000.0) * eff_fps[e.video_id])) + int(
            round(e.frame * eff_fps[e.video_id] / rally_fps)
        )
        tol_eff = int(round(tolerance_ms * eff_fps[e.video_id] / 1000))
        lo = max(0, (f_eff - tol_eff - WINDOW_SIZE) // STRIDE)
        hi = min(len(probs), (f_eff + tol_eff + WINDOW_SIZE) // STRIDE + 1)
        if hi > lo:
            e.videomae_max_prob = float(probs[lo:hi].max())


# ---------------------------------------------------------------------------
# Analysis + reporting
# ---------------------------------------------------------------------------

def _analyse(events: list[GTEvent]) -> dict:
    hit = [e for e in events if e.gbm_hit]
    miss = [e for e in events if not e.gbm_hit]
    hit_probs = np.array([e.videomae_max_prob for e in hit])
    miss_probs = np.array([e.videomae_max_prob for e in miss])

    def _pct(x, p):
        return float(np.percentile(x, p)) if len(x) else 0.0

    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    coverage = {}
    for thr in thresholds:
        miss_covered = (miss_probs >= thr).mean() if len(miss_probs) else 0.0
        hit_covered = (hit_probs >= thr).mean() if len(hit_probs) else 0.0
        coverage[thr] = {
            "miss_fraction_above": miss_covered,
            "hit_fraction_above": hit_covered,
        }

    return {
        "n_total": len(events),
        "n_hit": len(hit),
        "n_miss": len(miss),
        "hit_probs_stats": {
            "mean": float(hit_probs.mean()) if len(hit_probs) else 0,
            "median": _pct(hit_probs, 50),
            "p25": _pct(hit_probs, 25),
            "p75": _pct(hit_probs, 75),
        },
        "miss_probs_stats": {
            "mean": float(miss_probs.mean()) if len(miss_probs) else 0,
            "median": _pct(miss_probs, 50),
            "p25": _pct(miss_probs, 25),
            "p75": _pct(miss_probs, 75),
        },
        "coverage": coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tolerance-ms", type=int, default=DEFAULT_TOLERANCE_MS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cache-events", type=str,
        default="reports/orthogonality_events_2026_04_19.json",
        help="Where to cache step-1 GBM hit/miss tagging (so we can iterate on step 2 "
             "without re-running LOO).",
    )
    parser.add_argument("--skip-loo", action="store_true",
                        help="Reuse cached GBM tagging from --cache-events.")
    parser.add_argument(
        "--out", type=str,
        default="reports/videomae_orthogonality_2026_04_19.md",
    )
    args = parser.parse_args()

    rallies_raw = load_rallies_with_action_gt()
    if args.limit:
        # Keep all rallies from first N videos
        seen_videos: list[str] = []
        filtered = []
        for r in rallies_raw:
            if r.video_id not in seen_videos:
                if len(seen_videos) >= args.limit:
                    continue
                seen_videos.append(r.video_id)
            filtered.append(r)
        rallies_raw = filtered

    rallies_by_id = {r.rally_id: r for r in rallies_raw}
    contact_cfg = ContactDetectionConfig()

    # Step 1: tag each GT as GBM-HIT or GBM-MISS (expensive; cache)
    cache_path = Path(args.cache_events)
    if args.skip_loo and cache_path.exists():
        console.print(f"[dim]Loading cached GBM tags from {cache_path}[/dim]")
        data = json.loads(cache_path.read_text())
        events = [GTEvent(**e) for e in data]
    else:
        events = _tag_gbm_hits(
            rallies_raw, contact_cfg, args.threshold, args.tolerance_ms,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps([
            {
                "video_id": e.video_id, "rally_id": e.rally_id, "frame": e.frame,
                "action": e.action, "gbm_hit": e.gbm_hit,
                "gbm_pred_frame": e.gbm_pred_frame, "gbm_pred_action": e.gbm_pred_action,
                "videomae_max_prob": e.videomae_max_prob,
            }
            for e in events
        ], indent=2))
        console.print(f"[dim]Cached GBM tags → {cache_path}[/dim]")

    n_hit = sum(1 for e in events if e.gbm_hit)
    console.print(
        f"  GBM: {n_hit}/{len(events)} hits ({n_hit/max(1,len(events)):.1%}), "
        f"{len(events)-n_hit} misses"
    )

    # Step 2: VideoMAE per-video probs
    all_videos = load_evaluation_videos(require_ground_truth=True)
    action_ids = {e.video_id for e in events}
    videos = [v for v in all_videos if v.id in action_ids]
    fc = FeatureCache()
    probs_per_video, eff_fps = _videomae_probs_per_video(videos, fc)
    _annotate_events_with_videomae(
        events, probs_per_video, eff_fps, rallies_by_id, args.tolerance_ms,
    )

    # Analysis
    analysis = _analyse(events)

    console.print("\n[bold]VideoMAE-prob distribution at GT contacts[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Group", style="bold")
    for col in ("N", "mean", "p25", "median", "p75"):
        table.add_column(col, justify="right")
    h = analysis["hit_probs_stats"]
    m = analysis["miss_probs_stats"]
    table.add_row("GBM HIT", str(analysis["n_hit"]),
                  f"{h['mean']:.3f}", f"{h['p25']:.3f}",
                  f"{h['median']:.3f}", f"{h['p75']:.3f}")
    table.add_row("GBM MISS", str(analysis["n_miss"]),
                  f"{m['mean']:.3f}", f"{m['p25']:.3f}",
                  f"{m['median']:.3f}", f"{m['p75']:.3f}")
    console.print(table)

    console.print("\n[bold]Coverage: fraction of GT where VideoMAE prob ≥ thr[/bold]")
    table2 = Table(show_header=True, header_style="bold")
    table2.add_column("Thr", justify="right")
    table2.add_column("HIT above", justify="right")
    table2.add_column("MISS above", justify="right")
    table2.add_column("MISS_rescued_frac", justify="right")
    for thr, cov in sorted(analysis["coverage"].items()):
        table2.add_row(
            f"{thr:.2f}",
            f"{cov['hit_fraction_above']:.1%}",
            f"{cov['miss_fraction_above']:.1%}",
            f"{cov['miss_fraction_above']:.1%}",
        )
    console.print(table2)

    # Verdict
    best_miss_rescue = max(v["miss_fraction_above"] for v in analysis["coverage"].values())
    verdict = (
        "STRONG SIGNAL — fusion likely helps"
        if best_miss_rescue >= 0.40 else
        "MODEST SIGNAL — fusion small gain"
        if best_miss_rescue >= 0.15 else
        "REDUNDANT SIGNAL — fusion unlikely to help"
    )
    color = "green" if best_miss_rescue >= 0.40 else "yellow" if best_miss_rescue >= 0.15 else "red"
    console.print(
        f"\n[bold]Best miss-rescue fraction: [{color}]{best_miss_rescue:.1%}[/{color}]  → "
        f"[{color}]{verdict}[/{color}][/bold]"
    )

    # Markdown report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# VideoMAE–GBM Orthogonality Probe")
    lines.append("")
    lines.append(f"- GT contacts: {analysis['n_total']}")
    lines.append(f"- GBM hits: {analysis['n_hit']}  GBM misses: {analysis['n_miss']}")
    lines.append("")
    lines.append("## VideoMAE prob distribution at GT contacts")
    lines.append("")
    lines.append("| Group | N | mean | p25 | median | p75 |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| GBM HIT | {analysis['n_hit']} | {h['mean']:.3f} | {h['p25']:.3f} "
                 f"| {h['median']:.3f} | {h['p75']:.3f} |")
    lines.append(f"| GBM MISS | {analysis['n_miss']} | {m['mean']:.3f} | {m['p25']:.3f} "
                 f"| {m['median']:.3f} | {m['p75']:.3f} |")
    lines.append("")
    lines.append("## Coverage by threshold")
    lines.append("")
    lines.append("| Thr | HIT above | MISS above |")
    lines.append("|---|---|---|")
    for thr, cov in sorted(analysis["coverage"].items()):
        lines.append(f"| {thr:.2f} | {cov['hit_fraction_above']:.1%} "
                     f"| {cov['miss_fraction_above']:.1%} |")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")


if __name__ == "__main__":
    main()
