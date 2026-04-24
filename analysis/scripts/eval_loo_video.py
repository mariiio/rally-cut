"""Phase 0 baseline: LOO-per-video eval of the current contact + action pipeline.

Current `ContactClassifier.loo_cv` in contact_classifier.py is per-rally — rallies
from the same video land in both train and test folds, which leaks style,
lighting, and player identities across splits. This script re-measures the
baseline under **video-level** LOO so Phase 5 fusion has an honest ceiling to
beat.

What this does:
1. Pre-extracts candidate features (contact-side) + action features (for
   GT-matched contacts) for all 364 rallies once. Sequence probs are cached.
2. For each of the 68 videos:
   a. Trains a fresh contact GBM on candidates from the other 67 videos' rallies.
   b. Trains a fresh action GBM on matched contacts from the other 67 videos.
   c. Runs `detect_contacts(classifier=fold_contact)` + `classify_rally_actions`
      (with the fold action classifier injected) on every rally in the held-out
      video.
   d. Matches predictions to GT via the production Hungarian ±7f matcher.
3. Aggregates per-class F1 + contact F1 + action accuracy across folds and
   writes a markdown report under `reports/`.

Not included (intentionally): match-teams, ReID, visual-attribution,
calibrator-aware server detection. Those are orthogonal axes that inflate
scoring complexity without changing the contact F1 / action accuracy signal
we need for the VideoMAE baseline.

Usage (cd analysis):
    uv run python scripts/eval_loo_video.py                     # Full 68-fold LOO
    uv run python scripts/eval_loo_video.py --limit 5           # Smoke-test 5 folds
    uv run python scripts/eval_loo_video.py --tolerance-ms 233  # Match tolerance
    uv run python scripts/eval_loo_video.py --threshold 0.30    # Contact threshold
    uv run python scripts/eval_loo_video.py --out reports/my.md
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

import rallycut.tracking.action_classifier as ac_mod
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.action_type_classifier import ActionTypeClassifier
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.sequence_action_runtime import get_sequence_probs
from scripts.eval_action_detection import (
    RallyData,
    _match_synthetic_serves,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.train_action_classifier import extract_features_for_rally
from scripts.train_contact_classifier import extract_candidate_features, label_candidates

# Silence per-rally sanity warnings ("consecutive net-crossing", ">3 contacts on
# near side") from action_classifier — they are diagnostics, not errors, and
# would produce ~5k lines across 68 folds.
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)

console = Console()

ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]


@dataclass
class RallyPrecomputed:
    """Cached per-rally training inputs for the LOO loop."""

    rally: RallyData
    ball_positions: list[BallPos]
    player_positions: list[PlayerPos]
    sequence_probs: np.ndarray | None
    # Contact side
    candidate_features: np.ndarray  # (N_cand, 26)
    candidate_labels: np.ndarray  # (N_cand,) 0/1
    # Action side
    action_features: np.ndarray  # (N_matched, D)
    action_labels: np.ndarray  # (N_matched,) str


def _build_rally_positions(rally: RallyData) -> tuple[list[BallPos], list[PlayerPos]]:
    ball = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in (rally.ball_positions_json or [])
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]
    players = [
        PlayerPos(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"], y=pp["y"],
            width=pp["width"], height=pp["height"],
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        )
        for pp in (rally.positions_json or [])
    ]
    return ball, players


def _precompute(rally: RallyData, contact_cfg: ContactDetectionConfig) -> RallyPrecomputed | None:
    """Precompute seq_probs + candidate features + action features for a rally.

    Returns None when the rally cannot produce any candidates (missing tracking).
    """
    ball, players = _build_rally_positions(rally)
    if not ball or not players:
        return None

    seq_probs = get_sequence_probs(
        ball_positions=ball,
        player_positions=players,
        court_split_y=rally.court_split_y,
        frame_count=rally.frame_count or 0,
        team_assignments=None,
    )

    # Candidate features for contact GBM training. Pass gt_frames so
    # `frames_since_last` uses the inference-equivalent signal.
    gt_frames = [gt.frame for gt in rally.gt_labels]
    cand_feats, cand_frames = extract_candidate_features(
        rally, config=contact_cfg, gt_frames=gt_frames, sequence_probs=seq_probs,
    )
    if not cand_feats:
        return None
    cand_labels = label_candidates(cand_frames, rally.gt_labels)

    # Action features on GT-matched contacts. Uses the same extractor as
    # production training. Skips match-teams/calibrator for simplicity — these
    # are orthogonal axes that don't change the action-type signal being
    # measured here.
    action_feats, action_labels, _rids = extract_features_for_rally(
        rally, config=contact_cfg, tolerance=5,
        team_assignments=None, inject_pose=False,
        calibrator=None, camera_height=0.0,
    )

    return RallyPrecomputed(
        rally=rally,
        ball_positions=ball,
        player_positions=players,
        sequence_probs=seq_probs,
        candidate_features=np.array([f.to_array() for f in cand_feats], dtype=np.float64),
        candidate_labels=np.array(cand_labels, dtype=np.int64),
        action_features=np.array(action_feats, dtype=np.float64) if action_feats else np.zeros((0, 1)),
        action_labels=np.array(action_labels, dtype=object) if action_labels else np.array([], dtype=object),
    )


def _train_fold(
    train_rallies: list[RallyPrecomputed],
    threshold: float,
) -> tuple[ContactClassifier, ActionTypeClassifier]:
    """Train contact + action GBMs on the training folds."""
    X_contact = np.concatenate([r.candidate_features for r in train_rallies], axis=0)  # noqa: N806
    y_contact = np.concatenate([r.candidate_labels for r in train_rallies], axis=0)
    contact_clf = ContactClassifier(threshold=threshold)
    contact_clf.train(X_contact, y_contact)

    X_action = np.concatenate(  # noqa: N806
        [r.action_features for r in train_rallies if len(r.action_labels) > 0],
        axis=0,
    )
    y_action = np.concatenate(
        [r.action_labels for r in train_rallies if len(r.action_labels) > 0],
        axis=0,
    )
    action_clf = ActionTypeClassifier()
    if len(y_action) > 0:
        action_clf.train(X_action, y_action)
    return contact_clf, action_clf


def _inject_action_classifier(clf: ActionTypeClassifier | None) -> None:
    """Force `classify_rally_actions` to use the fold-trained classifier.

    Bypasses `load_action_type_classifier()` which reads from disk.
    """
    ac_mod._default_action_classifier_cache["default"] = clf


def _reset_action_classifier_cache() -> None:
    ac_mod._default_action_classifier_cache.clear()


def _eval_rally(
    pre: RallyPrecomputed,
    contact_clf: ContactClassifier,
    action_clf: ActionTypeClassifier,
    contact_cfg: ContactDetectionConfig,
    tolerance_ms: int,
    include_synthetic: bool = False,
    use_decoder: bool = False,
    decoder_skip_penalty: float = 1.0,
) -> dict:
    """Run detect_contacts → classify_rally_actions → match against GT.

    Returns the compute_metrics() dict plus matched/unmatched lists so the
    caller can aggregate per-class stats.

    When `use_decoder=True`, runs `detect_contacts_via_decoder` and skips
    `classify_rally_actions` (the decoder emits action labels directly via
    its Viterbi grammar). Phase 3 of the parallel-decoder ship plan
    (`docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`).
    """
    rally = pre.rally
    _inject_action_classifier(action_clf if action_clf.is_trained else None)
    try:
        if use_decoder:
            from rallycut.tracking.contact_detector import (
                detect_contacts_via_decoder,
            )
            contact_seq = detect_contacts_via_decoder(
                ball_positions=pre.ball_positions,
                player_positions=pre.player_positions,
                config=contact_cfg,
                frame_count=rally.frame_count or None,
                classifier=contact_clf,
                use_classifier=True,
                sequence_probs=pre.sequence_probs,
                skip_penalty=decoder_skip_penalty,
                # Eval-only: use GT frames for `frames_since_last` semantics
                # so this measurement matches the validated
                # `eval_candidate_decoder.py` methodology. Production wiring
                # in Phase 4 will need a two-pass scheme — see the plan §5.
                _eval_gt_frames=[gt.frame for gt in rally.gt_labels],
            )
            # Decoder emits action labels via the `decoder_action` field;
            # build pred_actions directly without running the legacy
            # `classify_rally_actions` pipeline.
            pred_actions = [
                {
                    "frame": c.frame,
                    "action": c.decoder_action or "unknown",
                    "playerTrackId": c.player_track_id,
                    "courtSide": c.court_side,
                    # No synthetic serves emitted on this path
                    "isSynthetic": False,
                }
                for c in contact_seq.contacts
                if c.decoder_action is not None
            ]
        else:
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
            pred_actions = [a.to_dict() for a in rally_actions.actions]
    finally:
        _reset_action_classifier_cache()

    real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
    synth_pred = [a for a in pred_actions if a.get("isSynthetic")]

    tolerance_frames = max(1, round(rally.fps * tolerance_ms / 1000))
    matches, unmatched = match_contacts(rally.gt_labels, real_pred, tolerance=tolerance_frames)

    # Pass 2: match synthetic serves against unmatched GT serves with wider
    # tolerance (~1s) — matches production-aligned eval in eval_action_detection.py.
    # Opt-in via --include-synthetic flag; default off preserves existing LOO numbers.
    if include_synthetic and synth_pred:
        synth_tolerance = max(tolerance_frames, round(rally.fps * 1.0))
        _match_synthetic_serves(
            matches, synth_pred, rally.gt_labels, synth_tolerance,
        )

    metrics = compute_metrics(matches, unmatched)
    return {
        "metrics": metrics,
        "matches": matches,
        "unmatched": unmatched,
    }


def _per_class_tally(matches: list, unmatched: list) -> dict[str, dict[str, int]]:
    """Per-class TP/FP/FN over matches + unmatched-pred list."""
    tally: dict[str, dict[str, int]] = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in ACTION_TYPES}

    for m in matches:
        gt = m.gt_action
        pred = m.pred_action
        if pred is None:
            # Unmatched GT → FN for GT class
            if gt in tally:
                tally[gt]["fn"] += 1
        else:
            if gt == pred:
                if gt in tally:
                    tally[gt]["tp"] += 1
            else:
                # Matched but wrong class: FN for gt, FP for pred
                if gt in tally:
                    tally[gt]["fn"] += 1
                if pred in tally:
                    tally[pred]["fp"] += 1

    # Unmatched predictions are FPs for the predicted class
    for p in unmatched:
        pred = p.get("action")
        if pred in tally:
            tally[pred]["fp"] += 1

    return tally


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return p, r, f1


def _write_report(
    out_path: Path,
    args: argparse.Namespace,
    fold_results: list[dict],
    agg_tp: int,
    agg_fp: int,
    agg_fn: int,
    agg_acc_correct: int,
    agg_acc_total: int,
    class_tally: dict[str, dict[str, int]],
    total_seconds: float,
) -> None:
    p, r, f1 = _f1(agg_tp, agg_fp, agg_fn)
    action_acc = (agg_acc_correct / max(1, agg_acc_total))

    lines: list[str] = []
    lines.append("# Phase 0 — LOO-per-video baseline")
    lines.append("")
    lines.append(f"- Folds: {len(fold_results)}")
    lines.append(f"- Contact threshold: {args.threshold}")
    lines.append(f"- Match tolerance: ±{args.tolerance_ms} ms")
    lines.append(f"- Wall-clock: {total_seconds/60:.1f} min")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Contact F1 | **{f1:.1%}** |")
    lines.append(f"| Contact Precision | {p:.1%} |")
    lines.append(f"| Contact Recall | {r:.1%} |")
    lines.append(f"| Action Accuracy | **{action_acc:.1%}** ({agg_acc_correct}/{agg_acc_total}) |")
    lines.append(f"| TP / FP / FN | {agg_tp} / {agg_fp} / {agg_fn} |")
    lines.append("")
    lines.append("## Per-class F1")
    lines.append("")
    lines.append("| Class | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cls in ACTION_TYPES:
        c = class_tally[cls]
        p_c, r_c, f1_c = _f1(c["tp"], c["fp"], c["fn"])
        lines.append(
            f"| {cls} | {c['tp']} | {c['fp']} | {c['fn']} "
            f"| {p_c:.1%} | {r_c:.1%} | {f1_c:.1%} |"
        )
    lines.append("")
    lines.append("## Per-fold (held-out video)")
    lines.append("")
    lines.append("| Video | Rallies | TP | FP | FN | Contact F1 | Action Acc |")
    lines.append("|---|---|---|---|---|---|---|")
    for fold in sorted(fold_results, key=lambda f: f["video_id"]):
        lines.append(
            f"| {fold['video_id'][:8]} | {fold['n_rallies']} "
            f"| {fold['tp']} | {fold['fp']} | {fold['fn']} "
            f"| {fold['contact_f1']:.1%} | {fold['action_acc']:.1%} |"
        )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Contact classifier threshold (default: 0.30 — matches production)")
    parser.add_argument("--tolerance-ms", type=int, default=233,
                        help="Match tolerance in ms (default: 233 = ±7 frames @ 30fps)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to N folds for smoke-testing")
    parser.add_argument("--out", type=str,
                        default="reports/contact_baseline_loo_video_2026_04_19.md")
    parser.add_argument("--include-synthetic", action="store_true",
                        help="Match synthetic serves against unmatched GT serves with ±1s tolerance (production-aligned). Default off preserves historical LOO baseline numbers.")
    parser.add_argument("--out-json", type=str, default=None,
                        help="Also write raw per-fold JSON for later comparison")
    parser.add_argument("--use-decoder", action="store_true",
                        help="Route contact detection through the parallel Viterbi decoder "
                             "(`detect_contacts_via_decoder`). Skips classify_rally_actions; "
                             "the decoder emits action labels directly. Phase 3 of the "
                             "parallel-decoder ship plan.")
    parser.add_argument("--decoder-skip-penalty", type=float, default=1.0,
                        help="Skip-penalty for the decoder (default 1.0 = production-validated).")
    args = parser.parse_args()

    t_start = time.time()
    console.print("[bold]Loading GT rallies...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies across {len({r.video_id for r in rallies})} videos")

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
    console.print(
        f"  {len(precomputed)} rallies with features "
        f"({time.time()-t_pre:.0f}s)"
    )

    # Group by video
    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]
    console.print(f"[bold]Running {len(video_ids)}-fold LOO-per-video...[/bold]")

    fold_results: list[dict] = []
    agg_tp = agg_fp = agg_fn = 0
    agg_acc_correct = 0
    agg_acc_total = 0
    class_tally: dict[str, dict[str, int]] = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in ACTION_TYPES}

    for fold_idx, held_out_video in enumerate(video_ids, start=1):
        t_fold = time.time()
        held_rallies = by_video[held_out_video]
        train_rallies = [
            pre for vid, rs in by_video.items()
            if vid != held_out_video
            for pre in rs
        ]

        contact_clf, action_clf = _train_fold(train_rallies, args.threshold)

        fold_tp = fold_fp = fold_fn = 0
        fold_acc_correct = 0
        fold_acc_total = 0
        fold_matches: list = []
        fold_unmatched: list = []
        for pre in held_rallies:
            result = _eval_rally(
                pre, contact_clf, action_clf, contact_cfg, args.tolerance_ms,
                include_synthetic=args.include_synthetic,
                use_decoder=args.use_decoder,
                decoder_skip_penalty=args.decoder_skip_penalty,
            )
            m = result["metrics"]
            fold_tp += m["tp"]
            fold_fp += m["fp"]
            fold_fn += m["fn"]
            # compute_metrics already tracks action_accuracy via matched contacts
            matched_correct = sum(
                1 for mr in result["matches"]
                if mr.pred_action is not None and mr.gt_action == mr.pred_action
            )
            matched_total = sum(
                1 for mr in result["matches"] if mr.pred_action is not None
            )
            fold_acc_correct += matched_correct
            fold_acc_total += matched_total
            fold_matches.extend(result["matches"])
            fold_unmatched.extend(result["unmatched"])

        _, _, fold_f1 = _f1(fold_tp, fold_fp, fold_fn)
        fold_action_acc = fold_acc_correct / max(1, fold_acc_total)

        # Per-class accumulation
        cls_tally_fold = _per_class_tally(fold_matches, fold_unmatched)
        for cls, c in cls_tally_fold.items():
            for k in ("tp", "fp", "fn"):
                class_tally[cls][k] += c[k]

        agg_tp += fold_tp
        agg_fp += fold_fp
        agg_fn += fold_fn
        agg_acc_correct += fold_acc_correct
        agg_acc_total += fold_acc_total

        fold_results.append({
            "video_id": held_out_video,
            "n_rallies": len(held_rallies),
            "tp": fold_tp, "fp": fold_fp, "fn": fold_fn,
            "contact_f1": fold_f1,
            "action_acc": fold_action_acc,
        })

        _, _, cum_f1 = _f1(agg_tp, agg_fp, agg_fn)
        cum_acc = agg_acc_correct / max(1, agg_acc_total)
        console.print(
            f"  [{fold_idx}/{len(video_ids)}] {held_out_video[:8]} "
            f"({len(held_rallies)}r): F1={fold_f1:.1%} acc={fold_action_acc:.1%} "
            f"| cum F1={cum_f1:.1%} acc={cum_acc:.1%} ({time.time()-t_fold:.0f}s)"
        )

    total = time.time() - t_start
    console.print(f"\n[bold]Completed {len(fold_results)} folds in {total/60:.1f} min[/bold]")

    # Final summary
    p, r, f1 = _f1(agg_tp, agg_fp, agg_fn)
    console.print(
        f"  Contact F1: [bold]{f1:.1%}[/bold] (P={p:.1%}, R={r:.1%})  "
        f"TP={agg_tp} FP={agg_fp} FN={agg_fn}"
    )
    console.print(
        f"  Action accuracy: [bold]{agg_acc_correct/max(1,agg_acc_total):.1%}[/bold] "
        f"({agg_acc_correct}/{agg_acc_total})"
    )
    table = Table(title="Per-class F1 (LOO-per-video)")
    table.add_column("Class", style="bold")
    for col in ("TP", "FP", "FN", "P", "R", "F1"):
        table.add_column(col, justify="right")
    for cls in ACTION_TYPES:
        c = class_tally[cls]
        p_c, r_c, f1_c = _f1(c["tp"], c["fp"], c["fn"])
        table.add_row(
            cls, str(c["tp"]), str(c["fp"]), str(c["fn"]),
            f"{p_c:.1%}", f"{r_c:.1%}", f"{f1_c:.1%}",
        )
    console.print(table)

    out_path = Path(args.out)
    _write_report(
        out_path, args, fold_results,
        agg_tp, agg_fp, agg_fn,
        agg_acc_correct, agg_acc_total,
        class_tally, total,
    )
    console.print(f"\n[green]Report: {out_path}[/green]")

    if args.out_json:
        json_path = Path(args.out_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({
            "args": {"threshold": args.threshold, "tolerance_ms": args.tolerance_ms, "include_synthetic": args.include_synthetic},
            "aggregate": {
                "contact_f1": f1, "contact_precision": p, "contact_recall": r,
                "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
                "action_accuracy": agg_acc_correct / max(1, agg_acc_total),
                "action_correct": agg_acc_correct, "action_total": agg_acc_total,
            },
            "per_class": {
                cls: {
                    **class_tally[cls],
                    "precision": _f1(**class_tally[cls])[0],
                    "recall": _f1(**class_tally[cls])[1],
                    "f1": _f1(**class_tally[cls])[2],
                }
                for cls in ACTION_TYPES
            },
            "folds": fold_results,
            "wall_clock_seconds": total,
        }, indent=2))
        console.print(f"[green]JSON: {json_path}[/green]")


if __name__ == "__main__":
    main()
