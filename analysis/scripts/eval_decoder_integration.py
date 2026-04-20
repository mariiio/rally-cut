"""A/B: baseline vs decoder overlay on LOO-per-video.

Runs the same 68-fold loop as eval_loo_video.py, but evaluates each
held-out rally TWICE: once with ``decoder_contacts=None`` (baseline) and
once with the decoder output threaded through the overlay. Reports
side-by-side F1, Action Acc, per-class metrics, and the decision against
the 3 pre-registered ship gates:

  1. F1 delta >= -0.3pp (no regression)
  2. Action Acc delta >= +2.0pp
  3. Per-class F1 regression <= 2pp (block exempt: up to -12pp)

Both arms consume the SAME ``detect_contacts()`` output - only the
``decoder_contacts`` kwarg differs. Guarantees apples-to-apples
comparison: attribution is identical across arms, only action labels
can diverge.

Usage (cd analysis):
    uv run python scripts/eval_decoder_integration.py --limit 5    # smoke
    uv run python scripts/eval_decoder_integration.py              # full 68 folds
    uv run python scripts/eval_decoder_integration.py --skip-penalty 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.candidate_decoder import TransitionMatrix
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.crop_head_emitter import CropHeadContactClassifier
from rallycut.tracking.decoder_runtime import run_decoder_over_rally
from scripts.eval_action_detection import (
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _f1,
    _inject_action_classifier,
    _per_class_tally,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]

GATE_F1_DELTA = -0.003
GATE_ACC_DELTA = 0.020
GATE_PER_CLASS_REGRESSION = -0.020
BLOCK_EXEMPT_FLOOR = -0.120
BASELINE_SANITY_F1 = 0.880
BASELINE_SANITY_TOL = 0.003


@dataclass
class ArmResult:
    tp: int
    fp: int
    fn: int
    acc_correct: int
    acc_total: int
    class_tally: dict[str, dict[str, int]]


def _eval_rally_both_arms(
    pre: RallyPrecomputed,
    contact_clf: Any,
    action_clf: Any,
    contact_cfg: ContactDetectionConfig,
    transitions: TransitionMatrix,
    tolerance_ms: int,
    skip_penalty: float,
    emitter_factory: Callable[[RallyPrecomputed], Any] | None = None,
) -> tuple[ArmResult, ArmResult]:
    """Evaluate ONE rally under both arms.

    Same fold-trained classifiers for both arms; the only difference is
    whether ``decoder_contacts`` is passed to ``classify_rally_actions``.

    When ``emitter_factory`` is provided, the experimental arm uses the
    factory-produced emitter for ``run_decoder_over_rally`` while the
    baseline arm keeps using the GBM ``contact_clf`` for
    ``detect_contacts``. This swaps ONLY the decoder's internal emission
    source — the ``detect_contacts`` path stays on GBM in both arms.
    """
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
        baseline_actions = classify_rally_actions(
            contact_seq, rally_id=rally.rally_id,
            use_classifier=action_clf.is_trained,
            sequence_probs=pre.sequence_probs,
        )

        decoder_emitter = (
            emitter_factory(pre) if emitter_factory is not None else contact_clf
        )
        decoder_contacts = run_decoder_over_rally(
            ball_positions=pre.ball_positions,
            player_positions=pre.player_positions,
            sequence_probs=pre.sequence_probs,
            classifier=decoder_emitter,
            contact_config=contact_cfg,
            gt_frames=[gt.frame for gt in rally.gt_labels],
            transitions=transitions,
            skip_penalty=skip_penalty,
        )
        overlay_actions = classify_rally_actions(
            contact_seq, rally_id=rally.rally_id,
            use_classifier=action_clf.is_trained,
            sequence_probs=pre.sequence_probs,
            decoder_contacts=decoder_contacts,
        )
    finally:
        _reset_action_classifier_cache()

    tol = max(1, round(rally.fps * tolerance_ms / 1000))

    def _score(rally_actions: Any) -> ArmResult:
        pred = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred if not a.get("isSynthetic")]
        matches, unmatched = match_contacts(rally.gt_labels, real_pred, tolerance=tol)
        m = compute_metrics(matches, unmatched)
        ct = _per_class_tally(matches, unmatched)
        correct = sum(
            1 for mm in matches
            if mm.pred_action is not None and mm.gt_action == mm.pred_action
        )
        total = sum(1 for mm in matches if mm.pred_action is not None)
        return ArmResult(
            tp=m["tp"], fp=m["fp"], fn=m["fn"],
            acc_correct=correct, acc_total=total, class_tally=ct,
        )

    return _score(baseline_actions), _score(overlay_actions)


def _aggregate(arms: list[ArmResult]) -> ArmResult:
    total_tally = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in ACTION_TYPES}
    tp = fp = fn = ac = at = 0
    for a in arms:
        tp += a.tp
        fp += a.fp
        fn += a.fn
        ac += a.acc_correct
        at += a.acc_total
        for cls in ACTION_TYPES:
            for k in ("tp", "fp", "fn"):
                total_tally[cls][k] += a.class_tally[cls][k]
    return ArmResult(
        tp=tp, fp=fp, fn=fn, acc_correct=ac, acc_total=at,
        class_tally=total_tally,
    )


def _check_gates(
    base: ArmResult,
    over: ArmResult,
) -> tuple[str, list[str], dict[str, Any]]:
    """Evaluate the 3 pre-registered ship gates.

    Returns (verdict, reasons, details) where verdict is "PASS" or "NO-GO".
    """
    _, _, base_f1 = _f1(base.tp, base.fp, base.fn)
    _, _, over_f1 = _f1(over.tp, over.fp, over.fn)
    base_acc = base.acc_correct / max(1, base.acc_total)
    over_acc = over.acc_correct / max(1, over.acc_total)

    f1_delta = over_f1 - base_f1
    acc_delta = over_acc - base_acc

    reasons: list[str] = []
    per_class_details: dict[str, dict[str, float]] = {}
    worst_regression_cls: str | None = None
    worst_regression_val = 0.0

    # Gate 1: F1 delta >= -0.3pp
    if f1_delta < GATE_F1_DELTA:
        reasons.append(
            f"Gate 1 FAIL: F1 delta {f1_delta:+.2%} < {GATE_F1_DELTA:+.2%}"
        )
    # Gate 2: Action Acc delta >= +2.0pp
    if acc_delta < GATE_ACC_DELTA:
        reasons.append(
            f"Gate 2 FAIL: Action Acc delta {acc_delta:+.2%} < {GATE_ACC_DELTA:+.2%}"
        )
    # Gate 3: per-class F1 regression <= 2pp (block exempt)
    for cls in ACTION_TYPES:
        b = base.class_tally[cls]
        o = over.class_tally[cls]
        _, _, bf1 = _f1(b["tp"], b["fp"], b["fn"])
        _, _, of1 = _f1(o["tp"], o["fp"], o["fn"])
        delta = of1 - bf1
        per_class_details[cls] = {
            "base_f1": bf1, "over_f1": of1, "delta": delta,
        }
        floor = (
            BLOCK_EXEMPT_FLOOR if cls == "block"
            else GATE_PER_CLASS_REGRESSION
        )
        if delta < floor:
            reasons.append(
                f"Gate 3 FAIL ({cls}): per-class F1 delta {delta:+.2%} "
                f"< floor {floor:+.2%}"
            )
        if delta < worst_regression_val:
            worst_regression_val = delta
            worst_regression_cls = cls

    verdict = "PASS" if not reasons else "NO-GO"
    details = {
        "base_f1": base_f1,
        "over_f1": over_f1,
        "base_acc": base_acc,
        "over_acc": over_acc,
        "f1_delta": f1_delta,
        "acc_delta": acc_delta,
        "per_class": per_class_details,
        "worst_regression_class": worst_regression_cls,
        "worst_regression_delta": worst_regression_val,
    }
    return verdict, reasons, details


def _render_markdown(
    out_path: Path,
    args: argparse.Namespace,
    base: ArmResult,
    over: ArmResult,
    fold_rows: list[dict[str, Any]],
    total_seconds: float,
    verdict: str,
    reasons: list[str],
    details: dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, _, base_f1 = _f1(base.tp, base.fp, base.fn)
    _, _, over_f1 = _f1(over.tp, over.fp, over.fn)
    base_acc = base.acc_correct / max(1, base.acc_total)
    over_acc = over.acc_correct / max(1, over.acc_total)

    # Baseline-drift sanity
    drift = base_f1 - BASELINE_SANITY_F1
    sanity_note = (
        f"Baseline F1 {base_f1:.2%} is within "
        f"{BASELINE_SANITY_TOL:.1%} of canonical {BASELINE_SANITY_F1:.1%} "
        f"(delta {drift:+.2%})."
        if abs(drift) <= BASELINE_SANITY_TOL
        else (
            f"**WARN**: baseline F1 {base_f1:.2%} drifts "
            f"{drift:+.2%} from canonical {BASELINE_SANITY_F1:.1%} "
            f"(tolerance {BASELINE_SANITY_TOL:.1%}). Likely per-fold "
            "train/eval wiring change; investigate before trusting delta."
        )
    )

    lines: list[str] = []
    lines.append("# Decoder integration A/B eval (LOO-per-video)")
    lines.append("")
    lines.append(f"- Folds: {len(fold_rows)}")
    lines.append(f"- Contact threshold: {args.threshold}")
    lines.append(f"- Match tolerance: plus/minus {args.tolerance_ms} ms")
    lines.append(f"- Skip penalty: {args.skip_penalty}")
    lines.append(f"- Wall-clock: {total_seconds/60:.1f} min")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    if reasons:
        lines.append("Gate failures:")
        for r in reasons:
            lines.append(f"- {r}")
    else:
        lines.append("All 3 ship gates pass.")
    lines.append("")
    lines.append(f"Sanity: {sanity_note}")
    lines.append("")
    lines.append("## Aggregate (baseline vs overlay)")
    lines.append("")
    lines.append("| Metric | Baseline | Overlay | Delta |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| Contact F1 | {base_f1:.2%} | {over_f1:.2%} | "
        f"**{details['f1_delta']:+.2%}** |"
    )
    lines.append(
        f"| Action Acc | {base_acc:.2%} | {over_acc:.2%} | "
        f"**{details['acc_delta']:+.2%}** |"
    )
    lines.append(
        f"| TP / FP / FN | {base.tp} / {base.fp} / {base.fn} | "
        f"{over.tp} / {over.fp} / {over.fn} | "
        f"{over.tp - base.tp:+d} / {over.fp - base.fp:+d} / "
        f"{over.fn - base.fn:+d} |"
    )
    lines.append(
        f"| Action correct / total | "
        f"{base.acc_correct} / {base.acc_total} | "
        f"{over.acc_correct} / {over.acc_total} | "
        f"{over.acc_correct - base.acc_correct:+d} / "
        f"{over.acc_total - base.acc_total:+d} |"
    )
    lines.append("")
    lines.append("## Per-class F1 (baseline vs overlay)")
    lines.append("")
    lines.append("| Class | Base TP/FP/FN | Over TP/FP/FN | Base F1 | Over F1 | Delta |")
    lines.append("|---|---|---|---|---|---|")
    for cls in ACTION_TYPES:
        b = base.class_tally[cls]
        o = over.class_tally[cls]
        pc = details["per_class"][cls]
        lines.append(
            f"| {cls} "
            f"| {b['tp']}/{b['fp']}/{b['fn']} "
            f"| {o['tp']}/{o['fp']}/{o['fn']} "
            f"| {pc['base_f1']:.2%} "
            f"| {pc['over_f1']:.2%} "
            f"| **{pc['delta']:+.2%}** |"
        )
    lines.append("")
    lines.append("## Per-fold summary")
    lines.append("")
    lines.append(
        "| Video | Rallies | Base F1 | Over F1 | dF1 | "
        "Base Acc | Over Acc | dAcc | Seconds |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for f in sorted(fold_rows, key=lambda r: r["video_id"]):
        lines.append(
            f"| {f['video_id'][:8]} "
            f"| {f['n_rallies']} "
            f"| {f['base_f1']:.1%} "
            f"| {f['over_f1']:.1%} "
            f"| {f['f1_delta']:+.1%} "
            f"| {f['base_acc']:.1%} "
            f"| {f['over_acc']:.1%} "
            f"| {f['acc_delta']:+.1%} "
            f"| {f['seconds']:.0f} |"
        )
    lines.append("")
    out_path.write_text("\n".join(lines))


def _serialize(arm: ArmResult) -> dict[str, Any]:
    _, _, f1 = _f1(arm.tp, arm.fp, arm.fn)
    acc = arm.acc_correct / max(1, arm.acc_total)
    per_class: dict[str, dict[str, float]] = {}
    for cls in ACTION_TYPES:
        c = arm.class_tally[cls]
        p, r, f = _f1(c["tp"], c["fp"], c["fn"])
        per_class[cls] = {
            "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
            "precision": p, "recall": r, "f1": f,
        }
    return {
        "tp": arm.tp, "fp": arm.fp, "fn": arm.fn,
        "contact_f1": f1,
        "action_correct": arm.acc_correct,
        "action_total": arm.acc_total,
        "action_accuracy": acc,
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Contact classifier threshold (default: 0.30 — matches production)")
    parser.add_argument("--tolerance-ms", type=int, default=233,
                        help="Match tolerance in ms (default: 233 = plus/minus 7 frames @ 30fps)")
    parser.add_argument("--skip-penalty", type=float, default=1.0,
                        help="Decoder skip penalty (default: 1.0 — ship config)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N folds for smoke-testing")
    parser.add_argument("--out", type=str, default=None,
                        help="Markdown report path (default: "
                             "reports/decoder_integration_<date>.md)")
    parser.add_argument("--out-json", type=str, default=None,
                        help="JSON report path (default: "
                             "reports/decoder_integration_<date>.json)")
    parser.add_argument(
        "--emitter",
        choices=["gbm", "crop_head"],
        default="gbm",
        help="Decoder emission source. 'gbm' (default) keeps current "
             "production. 'crop_head' swaps the decoder's internal emitter "
             "for CropHeadContactClassifier — the baseline arm stays on GBM.",
    )
    parser.add_argument(
        "--crop-head-weights",
        type=str,
        default=None,
        help="Path to the crop-head checkpoint .pt. Required with --emitter=crop_head.",
    )
    args = parser.parse_args()

    if args.emitter == "crop_head" and not args.crop_head_weights:
        parser.error("--emitter=crop_head requires --crop-head-weights PATH")

    today = date.today().isoformat().replace("-", "_")
    default_md = f"reports/decoder_integration_{today}.md"
    default_json = f"reports/decoder_integration_{today}.json"
    md_path = Path(args.out) if args.out else Path(default_md)
    json_path = Path(args.out_json) if args.out_json else Path(default_json)

    t_start = time.time()
    console.print("[bold]Loading GT rallies...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(
        f"  {len(rallies)} rallies across "
        f"{len({r.video_id for r in rallies})} videos"
    )

    console.print("[bold]Pre-computing features for all rallies...[/bold]")
    t_pre = time.time()
    contact_cfg = ContactDetectionConfig()
    precomputed: list[RallyPrecomputed] = []
    for i, rally in enumerate(rallies):
        pre = _precompute(rally, contact_cfg)
        if pre is None:
            continue
        precomputed.append(pre)
        if (i + 1) % 25 == 0 or (i + 1) == len(rallies):
            console.print(
                f"  [{i+1}/{len(rallies)}] "
                f"({time.time()-t_pre:.0f}s elapsed)"
            )
    if not precomputed:
        console.print("[red]No rallies with precomputed features — aborting.[/red]")
        return 1
    console.print(
        f"  {len(precomputed)} rallies with features "
        f"({time.time()-t_pre:.0f}s)"
    )

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]

    transitions = TransitionMatrix.default()
    console.print(
        f"[dim]Loaded transitions: {len(transitions.probs)} contexts[/dim]"
    )

    # --- Experimental-arm emitter factory ---
    emitter_factory: Callable[[RallyPrecomputed], Any] | None = None
    video_path_cache: dict[str, Path] = {}
    if args.emitter == "crop_head":
        ckpt_path = Path(args.crop_head_weights)
        if not ckpt_path.exists():
            console.print(f"[red]Crop-head checkpoint missing: {ckpt_path}[/red]")
            return 1
        console.print(
            f"[bold]Emitter:[/bold] crop_head weights={ckpt_path}"
        )

        def _crop_head_factory(pre: RallyPrecomputed) -> Any:
            video_id = pre.rally.video_id
            vpath = video_path_cache.get(video_id)
            if vpath is None:
                resolved = get_video_path(video_id)
                if resolved is None:
                    raise RuntimeError(
                        f"Could not resolve video path for {video_id}"
                    )
                vpath = resolved
                video_path_cache[video_id] = vpath
            rally_start_frame = round(pre.rally.start_ms * pre.rally.fps / 1000)
            return CropHeadContactClassifier(
                checkpoint_path=ckpt_path,
                video_path=vpath,
                rally_start_frame=rally_start_frame,
                ball_positions=pre.ball_positions,
                player_positions=pre.player_positions,
            )
        emitter_factory = _crop_head_factory
    else:
        console.print("[bold]Emitter:[/bold] gbm (default)")

    console.print(
        f"[bold]Running {len(video_ids)}-fold A/B "
        "(baseline vs decoder overlay)...[/bold]"
    )

    base_fold_results: list[ArmResult] = []
    over_fold_results: list[ArmResult] = []
    fold_rows: list[dict[str, Any]] = []
    arms_differ_fold_count = 0

    for fold_idx, held_out_video in enumerate(video_ids, start=1):
        t_fold = time.time()
        held_rallies = by_video[held_out_video]
        train_rallies = [
            pre for vid, rs in by_video.items()
            if vid != held_out_video
            for pre in rs
        ]
        contact_clf, action_clf = _train_fold(train_rallies, args.threshold)

        fold_base_arms: list[ArmResult] = []
        fold_over_arms: list[ArmResult] = []
        for pre in held_rallies:
            base_arm, over_arm = _eval_rally_both_arms(
                pre, contact_clf, action_clf, contact_cfg,
                transitions, args.tolerance_ms, args.skip_penalty,
                emitter_factory=emitter_factory,
            )
            fold_base_arms.append(base_arm)
            fold_over_arms.append(over_arm)

        fold_base = _aggregate(fold_base_arms)
        fold_over = _aggregate(fold_over_arms)
        _, _, fold_base_f1 = _f1(fold_base.tp, fold_base.fp, fold_base.fn)
        _, _, fold_over_f1 = _f1(fold_over.tp, fold_over.fp, fold_over.fn)
        fold_base_acc = fold_base.acc_correct / max(1, fold_base.acc_total)
        fold_over_acc = fold_over.acc_correct / max(1, fold_over.acc_total)

        if fold_base_acc != fold_over_acc or fold_base_f1 != fold_over_f1:
            arms_differ_fold_count += 1

        base_fold_results.extend(fold_base_arms)
        over_fold_results.extend(fold_over_arms)

        fold_seconds = time.time() - t_fold
        fold_rows.append({
            "video_id": held_out_video,
            "n_rallies": len(held_rallies),
            "base_f1": fold_base_f1,
            "over_f1": fold_over_f1,
            "f1_delta": fold_over_f1 - fold_base_f1,
            "base_acc": fold_base_acc,
            "over_acc": fold_over_acc,
            "acc_delta": fold_over_acc - fold_base_acc,
            "base_tp": fold_base.tp, "base_fp": fold_base.fp, "base_fn": fold_base.fn,
            "over_tp": fold_over.tp, "over_fp": fold_over.fp, "over_fn": fold_over.fn,
            "seconds": fold_seconds,
        })

        console.print(
            f"  [{fold_idx}/{len(video_ids)}] {held_out_video[:8]} "
            f"({len(held_rallies)}r): "
            f"F1 {fold_base_f1:.1%}->{fold_over_f1:.1%} "
            f"acc {fold_base_acc:.1%}->{fold_over_acc:.1%} "
            f"({fold_seconds:.0f}s)"
        )

    total_seconds = time.time() - t_start

    base_agg = _aggregate(base_fold_results)
    over_agg = _aggregate(over_fold_results)
    verdict, reasons, details = _check_gates(base_agg, over_agg)

    # Console summary
    console.print(
        f"\n[bold]Completed {len(video_ids)} folds in "
        f"{total_seconds/60:.1f} min[/bold]"
    )
    _, _, agg_base_f1 = _f1(base_agg.tp, base_agg.fp, base_agg.fn)
    _, _, agg_over_f1 = _f1(over_agg.tp, over_agg.fp, over_agg.fn)
    agg_base_acc = base_agg.acc_correct / max(1, base_agg.acc_total)
    agg_over_acc = over_agg.acc_correct / max(1, over_agg.acc_total)
    console.print(
        f"  Baseline: F1={agg_base_f1:.2%}  Acc={agg_base_acc:.2%}"
    )
    console.print(
        f"  Overlay : F1={agg_over_f1:.2%}  Acc={agg_over_acc:.2%}"
    )
    console.print(
        f"  Delta  : dF1={details['f1_delta']:+.2%}  "
        f"dAcc={details['acc_delta']:+.2%}"
    )
    console.print(
        f"  Folds where arms differ: "
        f"{arms_differ_fold_count}/{len(fold_rows)}"
    )

    color = "green" if verdict == "PASS" else "red"
    console.print(f"\n[bold][{color}]Verdict: {verdict}[/{color}][/bold]")
    for r in reasons:
        console.print(f"  [red]{r}[/red]")

    table = Table(title="Per-class F1 delta (overlay - baseline)")
    table.add_column("Class", style="bold")
    for col in ("Base F1", "Over F1", "Delta"):
        table.add_column(col, justify="right")
    for cls in ACTION_TYPES:
        pc = details["per_class"][cls]
        delta_color = (
            "red" if pc["delta"] < GATE_PER_CLASS_REGRESSION
            and cls != "block" else "white"
        )
        table.add_row(
            cls,
            f"{pc['base_f1']:.2%}",
            f"{pc['over_f1']:.2%}",
            f"[{delta_color}]{pc['delta']:+.2%}[/{delta_color}]",
        )
    console.print(table)

    # Write reports
    _render_markdown(
        md_path, args, base_agg, over_agg, fold_rows,
        total_seconds, verdict, reasons, details,
    )
    console.print(f"\n[green]Markdown: {md_path}[/green]")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "args": {
            "threshold": args.threshold,
            "tolerance_ms": args.tolerance_ms,
            "skip_penalty": args.skip_penalty,
            "limit": args.limit,
        },
        "verdict": verdict,
        "reasons": reasons,
        "baseline": _serialize(base_agg),
        "overlay": _serialize(over_agg),
        "deltas": {
            "f1": details["f1_delta"],
            "acc": details["acc_delta"],
        },
        "baseline_sanity": {
            "observed_f1": agg_base_f1,
            "canonical_f1": BASELINE_SANITY_F1,
            "tolerance": BASELINE_SANITY_TOL,
            "drift": agg_base_f1 - BASELINE_SANITY_F1,
            "within_tolerance": abs(agg_base_f1 - BASELINE_SANITY_F1)
            <= BASELINE_SANITY_TOL,
        },
        "per_fold": fold_rows,
        "folds_where_arms_differ": arms_differ_fold_count,
        "wall_clock_seconds": total_seconds,
    }, indent=2))
    console.print(f"[green]JSON: {json_path}[/green]")

    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
