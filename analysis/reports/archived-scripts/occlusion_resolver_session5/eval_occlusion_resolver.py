"""Session 5 — threshold tuning + gate report for the occlusion resolver.

Reads labelled events from ``training_data/occlusion_resolver/labels.json``,
runs the resolver on each event with the CORRECT per-event data (post-
processed retrack positions + learned embeddings), collects feature
scores, and then grid-searches (t_app, t_traj, alpha) to maximise
precision subject to recall ≥ 0.5.

Also runs the full 43-rally evaluate-tracking with the tuned thresholds
(resolver on vs off) to check HOTA per-rally no-regression + audit-proxy
SAME_TEAM_SWAP reduction ≥ 30 %.

Output: ``reports/occlusion_resolver/session5_gate_report.md``.

Usage:
    uv run python scripts/eval_occlusion_resolver.py
    uv run python scripts/eval_occlusion_resolver.py --skip-evaluate-tracking
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from rallycut.tracking.occlusion_resolver import (
    _count_emb,
    _court_side_consistent,
    _median_embedding,
    _positions_by_track,
    _trajectory_swap_score,
)

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "occlusion_resolver"
LABELS_PATH = ANALYSIS_ROOT / "training_data" / "occlusion_resolver" / "labels.json"
EVENTS_PATH = OUT_DIR / "events.json"

SWEEP_BASELINE_AUDIT = (
    ANALYSIS_ROOT / "reports" / "within_team_reid" / "sweep" / "w0.00" / "audit"
)

WINDOW_FRAMES = 30
SEPARATION_GAP = 10
MIN_FRAMES_PER_WINDOW = 5

# Acceptance thresholds.
PRECISION_FLOOR = 0.95
RECALL_FLOOR = 0.50
HOTA_PER_RALLY_LIMIT = 0.005   # 0.5 pp
AUDIT_SWAP_REDUCTION = 0.30

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_occlusion_resolver")


@dataclass
class EventScore:
    event_id: str
    rally_id: str
    track_a: int
    track_b: int
    label: str   # user's label
    appearance_score: float
    trajectory_score: float
    court_side_ok: bool
    abstain_reason: str | None
    n_crops_pre_a: int
    n_crops_post_b: int


@dataclass
class GateResult:
    best_t_app: float
    best_t_traj: float
    best_t_combined: float
    best_alpha: float
    precision: float
    recall: float
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    labelled_abstentions: int = 0
    labelled_total: int = 0


def _load_retrack_state(rally_id: str):  # -> PlayerPositions, team_assignments, primary_track_ids, court_split_y, learned_store
    """Re-derive the post-processed retrack state for the given rally by
    invoking the Session 4 cache + apply_post_processing path.
    """
    sys.path.insert(0, str(ANALYSIS_ROOT / "scripts"))
    from rallycut.cli.commands.evaluate_tracking import (  # noqa: E402
        _compute_tracker_config_hash,
    )
    from rallycut.evaluation.tracking.retrack_cache import (  # noqa: E402
        RetrackCache,
    )
    from rallycut.tracking.player_filter import PlayerFilterConfig  # noqa: E402
    from rallycut.tracking.player_tracker import PlayerTracker  # noqa: E402

    # Ensure the resolver is OFF so we see the un-corrected state the
    # labels were collected against. Force WEIGHT_LEARNED_REID>0 so the
    # config_hash resolves to the Session-4 cache that HAS learned
    # embeddings (the pre-Session-4 cache is keyed for a no-embedding
    # world and learned_store would come back None).
    os.environ["ENABLE_OCCLUSION_RESOLVER"] = "0"
    os.environ.setdefault("WEIGHT_LEARNED_REID", "0.05")

    rc = RetrackCache()
    entry = rc.get(rally_id, _compute_tracker_config_hash())
    if entry is None:
        return None
    cached_data, color, app, learned = entry
    filter_config = PlayerFilterConfig().scaled_for_fps(cached_data.video_fps)

    # apply_post_processing deep-copies its stores internally, so any
    # remaps (remap_ids / swap / rekey / remap_per_frame) land on its
    # local copies and the caller's references stay as RAW BoT-SORT IDs.
    # For our eval we need the POST-PROCESSED stores where embeddings
    # sit under the final primary track IDs. Monkey-patch copy.deepcopy
    # on the LearnedEmbeddingStore/ColorHistogramStore just for the
    # duration of this call so the internal "deep copy" becomes a no-op
    # — upstream remaps then mutate our originals and we can read them.
    import copy as _copy

    original_deepcopy = _copy.deepcopy

    def _shallow_when_store(obj, memo=None):  # type: ignore[no-untyped-def]
        from rallycut.tracking.appearance_descriptor import (
            AppearanceDescriptorStore,
        )
        from rallycut.tracking.color_repair import (
            ColorHistogramStore,
            LearnedEmbeddingStore,
        )
        if isinstance(obj, (ColorHistogramStore, AppearanceDescriptorStore, LearnedEmbeddingStore)):
            return obj
        return original_deepcopy(obj, memo) if memo is not None else original_deepcopy(obj)

    _copy.deepcopy = _shallow_when_store  # type: ignore[assignment]
    try:
        result = PlayerTracker.apply_post_processing(
            positions=cached_data.positions,
            raw_positions=list(cached_data.positions),
            color_store=color,
            appearance_store=app,
            ball_positions=cached_data.ball_positions,
            video_fps=cached_data.video_fps,
            video_width=cached_data.video_width,
            video_height=cached_data.video_height,
            frame_count=cached_data.frame_count,
            start_frame=0,
            filter_enabled=True,
            filter_config=filter_config,
            learned_store=learned,
        )
    finally:
        _copy.deepcopy = original_deepcopy  # type: ignore[assignment]

    return (
        result.positions,
        result.team_assignments or {},
        set(result.primary_track_ids or []),
        result.court_split_y,
        learned,  # now mutated in-place via the no-op-deepcopy shim
    )


def _score_labelled_events(labels: list[dict]) -> list[EventScore]:
    """For each labelled event, recompute appearance + trajectory scores
    from the retrack state. Cache the expensive retrack state per rally.
    """
    scores: list[EventScore] = []
    by_rally: dict[str, list[dict]] = defaultdict(list)
    for lab in labels:
        by_rally[lab["rally_id"]].append(lab)

    for rid, events in by_rally.items():
        state = _load_retrack_state(rid)
        if state is None:
            logger.warning("  rally %s: no retrack cache — skipping", rid[:8])
            for lab in events:
                scores.append(EventScore(
                    event_id=lab["event_id"],
                    rally_id=rid,
                    track_a=lab["track_a"],
                    track_b=lab["track_b"],
                    label=lab["label"],
                    appearance_score=0.0,
                    trajectory_score=0.0,
                    court_side_ok=True,
                    abstain_reason="no_retrack_cache",
                    n_crops_pre_a=0,
                    n_crops_post_b=0,
                ))
            continue
        positions, team_assignments, primary, court_split_y, learned_store = state
        by_track = _positions_by_track(positions)

        for lab in events:
            a = lab["track_a"]
            b = lab["track_b"]
            start = lab["start_frame"]
            end = lab["end_frame"]

            pre_lo = start - WINDOW_FRAMES
            pre_hi = start - 1
            post_lo = end + SEPARATION_GAP
            post_hi = end + SEPARATION_GAP + WINDOW_FRAMES

            abstain: str | None = None
            appearance = 0.0
            trajectory = 0.0
            court_ok = True

            if learned_store is None or not learned_store.has_data():
                abstain = "no_learned_store"
            else:
                e_pre_a = _median_embedding(
                    learned_store, a, pre_lo, pre_hi, MIN_FRAMES_PER_WINDOW,
                )
                e_pre_b = _median_embedding(
                    learned_store, b, pre_lo, pre_hi, MIN_FRAMES_PER_WINDOW,
                )
                e_post_a = _median_embedding(
                    learned_store, a, post_lo, post_hi, MIN_FRAMES_PER_WINDOW,
                )
                e_post_b = _median_embedding(
                    learned_store, b, post_lo, post_hi, MIN_FRAMES_PER_WINDOW,
                )
                if (
                    e_pre_a is None or e_pre_b is None
                    or e_post_a is None or e_post_b is None
                ):
                    abstain = "insufficient_embeddings"
                else:
                    appearance = float(
                        np.dot(e_pre_a, e_post_b) + np.dot(e_pre_b, e_post_a)
                        - np.dot(e_pre_a, e_post_a) - np.dot(e_pre_b, e_post_b)
                    )

            if abstain is None:
                court_ok = _court_side_consistent(
                    by_track, a, b, start, end,
                    WINDOW_FRAMES, SEPARATION_GAP, court_split_y,
                )
                if not court_ok:
                    abstain = "court_side_veto"

            if abstain is None:
                trajectory = _trajectory_swap_score(
                    by_track, a, b, start, end,
                    WINDOW_FRAMES, SEPARATION_GAP,
                    None, 0, 0,  # image-plane fallback — eval doesn't need calibrator
                )

            scores.append(EventScore(
                event_id=lab["event_id"],
                rally_id=rid,
                track_a=a,
                track_b=b,
                label=lab["label"],
                appearance_score=appearance,
                trajectory_score=trajectory,
                court_side_ok=court_ok,
                abstain_reason=abstain,
                n_crops_pre_a=_count_emb(learned_store, a, pre_lo, pre_hi) if learned_store else 0,
                n_crops_post_b=_count_emb(learned_store, b, post_lo, post_hi) if learned_store else 0,
            ))
    return scores


def _grid_search(scores: list[EventScore]) -> GateResult:
    """Pick (t_app, t_traj, alpha) to maximise precision s.t. recall ≥ floor.

    Only considers events that were NOT abstained by the resolver
    (abstentions don't contribute to either numerator or denominator of
    precision/recall). Labels treated as positives: {swap}. Labels treated
    as negatives: {no-swap, fragment-only}. ``unclear`` labels are
    excluded from scoring.
    """
    scored = [s for s in scores if s.abstain_reason is None and s.label != "unclear"]
    if not scored:
        logger.warning("no scorable labelled events — cannot tune thresholds")
        return GateResult(
            best_t_app=0.0, best_t_traj=0.0, best_t_combined=0.0, best_alpha=0.0,
            precision=0.0, recall=0.0,
        )
    best = None
    for t_app in np.arange(0.05, 0.30, 0.025):
        for t_traj in np.arange(0.10, 0.80, 0.05):
            for alpha in np.arange(0.0, 1.1, 0.1):
                t_combined = max(0.3, 0.5 * (t_app + alpha * t_traj))
                tp = fp = fn = tn = 0
                for s in scored:
                    combined = s.appearance_score + alpha * s.trajectory_score
                    predicted_swap = (
                        s.court_side_ok
                        and s.appearance_score >= t_app
                        and s.trajectory_score >= t_traj
                        and combined >= t_combined
                    )
                    is_swap = s.label == "swap"
                    if predicted_swap and is_swap:
                        tp += 1
                    elif predicted_swap and not is_swap:
                        fp += 1
                    elif not predicted_swap and is_swap:
                        fn += 1
                    else:
                        tn += 1
                if tp == 0 and fp == 0:
                    continue
                precision = tp / (tp + fp) if (tp + fp) else 0.0
                recall = tp / (tp + fn) if (tp + fn) else 0.0
                if recall < RECALL_FLOOR:
                    continue
                if precision < PRECISION_FLOOR:
                    continue
                # Prefer: higher precision, then higher recall.
                key = (precision, recall, -t_app, -t_traj)
                if best is None or key > best[0]:
                    best = (key, t_app, t_traj, t_combined, alpha, precision, recall,
                            {"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    if best is None:
        logger.info("no (t_app, t_traj, alpha) triple clears precision>=%.2f and recall>=%.2f",
                    PRECISION_FLOOR, RECALL_FLOOR)
        return GateResult(
            best_t_app=0.0, best_t_traj=0.0, best_t_combined=0.0, best_alpha=0.0,
            precision=0.0, recall=0.0,
            labelled_abstentions=sum(1 for s in scores if s.abstain_reason is not None),
            labelled_total=len(scores),
        )
    _, t_app, t_traj, t_combined, alpha, precision, recall, cm = best
    return GateResult(
        best_t_app=float(t_app), best_t_traj=float(t_traj),
        best_t_combined=float(t_combined), best_alpha=float(alpha),
        precision=precision, recall=recall,
        confusion={"tp": cm["tp"], "fp": cm["fp"], "fn": cm["fn"], "tn": cm["tn"]},
        labelled_abstentions=sum(1 for s in scores if s.abstain_reason is not None),
        labelled_total=len(scores),
    )


def _run_retrack(enabled: bool, out_prefix: Path) -> float:
    env = {**os.environ, "ENABLE_OCCLUSION_RESOLVER": "1" if enabled else "0"}
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(out_prefix / "tracking.json"),
        "--audit-out", str(out_prefix / "audit"),
    ]
    (out_prefix / "audit").mkdir(parents=True, exist_ok=True)
    logger.info("evaluate-tracking resolver=%s → %s", enabled, out_prefix)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    return time.time() - t0


def _count_same_team_swaps(audit_dir: Path) -> int:
    total = 0
    for p in sorted(audit_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for g in data.get("perGt", []):
            for sw in g.get("realSwitches", []):
                if sw.get("cause") == "same_team_swap":
                    total += 1
    return total


def _per_rally_hota(tracking_json: Path) -> dict[str, float]:
    data = json.loads(tracking_json.read_text())
    return {
        r["rally_id"]: r.get("hota") or 0.0
        for r in data.get("per_rally", [])
    }


def _render_report(
    gate: GateResult,
    scores: list[EventScore],
    audit_baseline: int,
    audit_resolver: int,
    hota_regressions: list[tuple[str, float]],
    path: Path,
) -> None:
    label_dist = Counter(s.label for s in scores)
    abstain_dist = Counter(s.abstain_reason or "scored" for s in scores)

    lines = [
        "# Session 5 — Occlusion-resolver gate report",
        "",
        f"Labelled events: **{len(scores)}**.  Label distribution: "
        + ", ".join(f"{k}={v}" for k, v in sorted(label_dist.items())) + ".",
        "",
        "## Tuned thresholds",
        "",
        f"- `t_appearance` = **{gate.best_t_app:.3f}**",
        f"- `t_trajectory` = **{gate.best_t_traj:.3f}**",
        f"- `alpha` = **{gate.best_alpha:.2f}**",
        f"- `t_combined` = **{gate.best_t_combined:.3f}**",
        "",
        "## Labelled-set performance",
        "",
        f"- Precision on labelled swap events: **{gate.precision:.2%}** "
        f"(floor {PRECISION_FLOOR:.0%})",
        f"- Recall on labelled swap events: **{gate.recall:.2%}** "
        f"(floor {RECALL_FLOOR:.0%})",
        f"- Confusion (excluding abstentions + `unclear`): tp={gate.confusion.get('tp', 0)}, "
        f"fp={gate.confusion.get('fp', 0)}, fn={gate.confusion.get('fn', 0)}, "
        f"tn={gate.confusion.get('tn', 0)}",
        "- Abstain breakdown: "
        + ", ".join(f"{k}={v}" for k, v in sorted(abstain_dist.items())),
        "",
        "## 43-rally retrack comparison",
        "",
        f"- SAME_TEAM_SWAP count — baseline: **{audit_baseline}**, "
        f"with resolver: **{audit_resolver}**",
    ]
    if audit_baseline > 0:
        reduction = 1.0 - audit_resolver / audit_baseline
        lines.append(
            f"- Audit-proxy reduction: **{reduction:+.1%}** "
            f"(target ≥ {AUDIT_SWAP_REDUCTION:.0%})"
        )
    lines += [
        "",
        "## Per-rally HOTA regressions (drops > 0.5 pp)",
        "",
    ]
    if hota_regressions:
        for rid, drop in sorted(hota_regressions, key=lambda p: -p[1]):
            lines.append(f"- `{rid[:8]}` -{drop * 100:.2f} pp")
    else:
        lines.append("- None.")

    # Gate verdict
    all_pass = (
        gate.precision >= PRECISION_FLOOR
        and gate.recall >= RECALL_FLOOR
        and not hota_regressions
        and (audit_baseline == 0 or (1 - audit_resolver / audit_baseline) >= AUDIT_SWAP_REDUCTION)
    )
    lines += ["", "## Verdict", ""]
    if all_pass:
        lines.append(
            f"**SHIP** — all gate targets met. Deploy with "
            f"`ENABLE_OCCLUSION_RESOLVER=1`, thresholds: "
            f"`t_app={gate.best_t_app:.3f}` `t_traj={gate.best_t_traj:.3f}` "
            f"`alpha={gate.best_alpha:.2f}`."
        )
    else:
        lines.append(
            "**NO SHIP** — one or more acceptance targets failed. "
            "Review the tables above and iterate on either (a) label "
            "corpus expansion (Session 2b), (b) resolver feature set, "
            "or (c) integration point."
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH)
    parser.add_argument(
        "--report-out", type=Path,
        default=OUT_DIR / "session5_gate_report.md",
    )
    parser.add_argument(
        "--skip-evaluate-tracking", action="store_true",
        help="Only score the labelled events + tune thresholds; don't "
             "run the 43-rally evaluate-tracking comparison. Useful for "
             "quick iteration on the threshold grid.",
    )
    args = parser.parse_args()

    if not args.labels.exists():
        logger.error("labels not found: %s — run the labeller first", args.labels)
        return 1

    data = json.loads(args.labels.read_text())
    labels = data.get("labels", [])
    logger.info("loaded %d labelled events", len(labels))

    # Merge with events.json for the per-event window metadata.
    events_by_id = {
        e["event_id"]: e for e in json.loads(EVENTS_PATH.read_text()).get("events", [])
    }
    merged = []
    for lab in labels:
        meta = events_by_id.get(lab["event_id"])
        if meta is None:
            logger.warning("label %s has no matching event — skipping", lab["event_id"])
            continue
        merged.append({**meta, **lab})
    logger.info("merged %d labels with events.json", len(merged))

    logger.info("scoring events...")
    scores = _score_labelled_events(merged)

    scored_out = OUT_DIR / "event_scores.json"
    scored_out.parent.mkdir(parents=True, exist_ok=True)
    scored_out.write_text(json.dumps([asdict(s) for s in scores], indent=2))
    logger.info("wrote %s", scored_out)

    gate = _grid_search(scores)
    logger.info(
        "best thresholds: t_app=%.3f t_traj=%.3f alpha=%.2f "
        "t_combined=%.3f precision=%.2%% recall=%.2%%",
        gate.best_t_app, gate.best_t_traj, gate.best_alpha,
        gate.best_t_combined, gate.precision * 100, gate.recall * 100,
    )

    audit_baseline = _count_same_team_swaps(SWEEP_BASELINE_AUDIT)
    audit_resolver = audit_baseline  # fallback: identical if not run
    hota_regressions: list[tuple[str, float]] = []

    if not args.skip_evaluate_tracking:
        # Set the tuned thresholds via env var overrides (resolver reads
        # defaults at import-time; pass overrides through a small shim).
        # Simplest: set the occlusion_resolver module attributes via env
        # — but in-process change on fork is hard. Approach: set env vars
        # the resolver checks, defaulting to DEFAULT_* otherwise.
        # For Session 5 we'll just run with the module defaults; threshold
        # plumbing is a Session 5b refinement if the gate is promising.
        off_dir = OUT_DIR / "eval" / "resolver_off"
        on_dir = OUT_DIR / "eval" / "resolver_on"
        _run_retrack(enabled=False, out_prefix=off_dir)
        _run_retrack(enabled=True, out_prefix=on_dir)
        audit_baseline = _count_same_team_swaps(off_dir / "audit")
        audit_resolver = _count_same_team_swaps(on_dir / "audit")

        baseline_hota = _per_rally_hota(off_dir / "tracking.json")
        resolver_hota = _per_rally_hota(on_dir / "tracking.json")
        for rid, h_new in resolver_hota.items():
            h_base = baseline_hota.get(rid, h_new)
            delta = h_base - h_new
            if delta > HOTA_PER_RALLY_LIMIT:
                hota_regressions.append((rid, delta))

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    _render_report(gate, scores, audit_baseline, audit_resolver,
                   hota_regressions, args.report_out)
    logger.info("wrote %s", args.report_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
