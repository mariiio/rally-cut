"""Phase CRF-1: LOO-per-video eval of the Viterbi candidate decoder.

Drop-in replacement for ``eval_loo_video.py``'s classifier decision: instead
of accepting candidates above a threshold, run Viterbi MAP decode with the
learned transition matrix from Phase CRF-0 as structural prior.

Reuses:
- ``_precompute`` + ``_train_fold`` + ``_eval_rally`` plumbing from
  ``eval_loo_video.py``.
- ``TransitionMatrix`` from ``reports/transition_matrix_2026_04_20.json``.
- ``match_contacts`` Hungarian ±7f from ``eval_action_detection.py``.

Apples-to-apples vs the 88.0% Phase-0 baseline.

Usage (cd analysis):
    uv run python scripts/eval_candidate_decoder.py              # full 68 folds
    uv run python scripts/eval_candidate_decoder.py --limit 5    # smoke
    uv run python scripts/eval_candidate_decoder.py --skip-penalty 0.5
    uv run python scripts/eval_candidate_decoder.py --min-accept-prob 0.02
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

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.candidate_decoder import (
    ACTIONS,
    CandidateFeatures,
    TransitionMatrix,
    decode_rally,
    infer_team_from_player_track,
)
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    RallyData,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _f1,
    _precompute,
    _train_fold,
)
from scripts.train_contact_classifier import (
    extract_candidate_features,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

ACTION_TYPES = ACTIONS  # alias for metrics tables


@dataclass
class CandidateMeta:
    """Candidate + its context for decoder features + nearest-player team."""
    frame: int
    team: int
    gbm_contact_prob: float
    action_probs: np.ndarray


def _build_candidate_features(
    rally: RallyData,
    candidate_frames: list[int],
    gbm_probs: np.ndarray,
    sequence_probs: np.ndarray | None,
    players: list[PlayerPos],
) -> list[CandidateFeatures]:
    """Convert per-rally candidates into decoder input format.

    - gbm_probs: output of ContactClassifier.predict_proba, aligned with candidate_frames.
    - sequence_probs: MS-TCN++ output (NUM_CLASSES, T) where channel 0 = bg,
      1-6 match ACTIONS. If None, action probs come from uniform prior.
    - team: inferred from the nearest active player at each candidate frame.
    """
    # Cache player positions by frame for fast nearest-player lookup
    by_frame: dict[int, list[PlayerPos]] = defaultdict(list)
    for pp in players:
        by_frame[pp.frame_number].append(pp)

    # Ball positions by frame for proximity calculation
    ball_by_frame: dict[int, BallPos] = {}
    for bp in (rally.ball_positions_json or []):
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
            ball_by_frame[bp["frameNumber"]] = BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )

    results: list[CandidateFeatures] = []
    for i, frame in enumerate(candidate_frames):
        # Team from nearest player at that frame
        team = -1
        ball = ball_by_frame.get(frame)
        if ball is not None:
            best = None
            best_d = 1e9
            for pp in by_frame.get(frame, []):
                d = (pp.x - ball.x) ** 2 + (pp.y - ball.y) ** 2
                if d < best_d:
                    best_d = d
                    best = pp
            if best is not None and 1 <= best.track_id <= 4:
                team = infer_team_from_player_track(best.track_id)

        # Action probs from seq_probs at this frame (bg at index 0, actions 1..6)
        if sequence_probs is not None and sequence_probs.size > 0:
            f = max(0, min(sequence_probs.shape[1] - 1, frame))
            p = sequence_probs[:, f]
            # Drop bg (index 0), renormalize over the 6 action channels
            pos = p[1:]
            s = float(pos.sum())
            if s > 1e-6:
                action_probs = pos / s
            else:
                action_probs = np.ones(len(ACTIONS)) / len(ACTIONS)
        else:
            action_probs = np.ones(len(ACTIONS)) / len(ACTIONS)

        results.append(CandidateFeatures(
            frame=frame,
            gbm_contact_prob=float(gbm_probs[i]),
            action_probs=action_probs,
            team=team,
        ))
    return results


def _per_class_tally(matches: list, unmatched: list) -> dict[str, dict[str, int]]:
    tally = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ACTION_TYPES}
    for m in matches:
        gt = m.gt_action
        pred = m.pred_action
        if pred is None:
            if gt in tally:
                tally[gt]["fn"] += 1
        else:
            if gt == pred:
                if gt in tally:
                    tally[gt]["tp"] += 1
            else:
                if gt in tally:
                    tally[gt]["fn"] += 1
                if pred in tally:
                    tally[pred]["fp"] += 1
    for p in unmatched:
        a = p.get("action")
        if a in tally:
            tally[a]["fp"] += 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="GBM threshold (only used for training target; "
                             "decoder uses raw probs)")
    parser.add_argument("--tolerance-ms", type=int, default=233)
    parser.add_argument("--skip-penalty", type=float, default=0.0,
                        help="Extra log-cost for skipping a candidate")
    parser.add_argument("--min-accept-prob", type=float, default=0.0,
                        help="GBM prob floor below which candidates are "
                             "pre-filtered")
    parser.add_argument("--transitions", type=str,
                        default="reports/transition_matrix_2026_04_20.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str,
                        default="reports/candidate_decoder_loo_2026_04_20.md")
    parser.add_argument("--out-json", type=str,
                        default="reports/candidate_decoder_loo_2026_04_20.json")
    args = parser.parse_args()

    t_start = time.time()
    transitions = TransitionMatrix.from_json(args.transitions)
    console.print(f"[dim]Loaded transitions: {len(transitions.probs)} contexts[/dim]")

    rallies = load_rallies_with_action_gt()
    contact_cfg = ContactDetectionConfig()

    console.print(f"[bold]Pre-computing features for {len(rallies)} rallies...[/bold]")
    t_pre = time.time()
    precomputed: list[RallyPrecomputed] = []
    cand_frames_by_rally: dict[str, list[int]] = {}
    for rally in rallies:
        pre = _precompute(rally, contact_cfg)
        if pre is None:
            continue
        precomputed.append(pre)
        _feats, cand_frames = extract_candidate_features(
            rally, config=contact_cfg,
            gt_frames=[gt.frame for gt in rally.gt_labels],
            sequence_probs=pre.sequence_probs,
        )
        cand_frames_by_rally[rally.rally_id] = list(cand_frames)
    console.print(f"  {len(precomputed)} rallies ready ({time.time()-t_pre:.0f}s)")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]

    console.print(f"\n[bold]LOO-per-video decoder eval ({len(video_ids)} folds)[/bold]")

    agg_tp = agg_fp = agg_fn = 0
    agg_acc_correct = 0
    agg_acc_total = 0
    class_tally = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ACTION_TYPES}
    fold_rows: list[dict] = []

    for fold_idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [pre for v, rs in by_video.items() if v != vid for pre in rs]
        contact_clf, action_clf = _train_fold(train, args.threshold)

        fold_tp = fold_fp = fold_fn = 0
        fold_acc_correct = 0
        fold_acc_total = 0
        fold_matches: list = []
        fold_unmatched: list = []

        for pre in held:
            rally = pre.rally

            # Get raw GBM probs on this rally's candidates
            feats_list, cand_frames = extract_candidate_features(
                rally, config=contact_cfg,
                gt_frames=[gt.frame for gt in rally.gt_labels],
                sequence_probs=pre.sequence_probs,
            )
            if not feats_list:
                continue
            X = np.array([f.to_array() for f in feats_list], dtype=np.float64)  # noqa: N806
            # ContactClassifier.predict returns (is_contact, prob) tuples.
            # We want the raw prob for decoder emission.
            if contact_clf.model is None:
                continue
            probs = contact_clf.model.predict_proba(X)[:, 1]

            cand_features = _build_candidate_features(
                rally, cand_frames, probs, pre.sequence_probs, pre.player_positions,
            )
            accepted = decode_rally(
                cand_features, transitions,
                skip_penalty=args.skip_penalty,
                min_accept_prob=args.min_accept_prob,
            )

            # Build pred_actions in the same schema as match_contacts expects
            pred_actions = [
                {"frame": d.frame, "action": d.action, "playerTrackId": -1}
                for d in accepted
            ]
            tol = max(1, round(rally.fps * args.tolerance_ms / 1000))
            matches, unmatched = match_contacts(rally.gt_labels, pred_actions, tolerance=tol)
            metrics = compute_metrics(matches, unmatched)

            fold_tp += metrics["tp"]
            fold_fp += metrics["fp"]
            fold_fn += metrics["fn"]
            matched_correct = sum(
                1 for m in matches
                if m.pred_action is not None and m.gt_action == m.pred_action
            )
            matched_total = sum(1 for m in matches if m.pred_action is not None)
            fold_acc_correct += matched_correct
            fold_acc_total += matched_total
            fold_matches.extend(matches)
            fold_unmatched.extend(unmatched)

        _, _, fold_f1 = _f1(fold_tp, fold_fp, fold_fn)
        fold_acc = fold_acc_correct / max(1, fold_acc_total)

        cls_fold = _per_class_tally(fold_matches, fold_unmatched)
        for c in ACTION_TYPES:
            for k in ("tp", "fp", "fn"):
                class_tally[c][k] += cls_fold[c][k]

        agg_tp += fold_tp
        agg_fp += fold_fp
        agg_fn += fold_fn
        agg_acc_correct += fold_acc_correct
        agg_acc_total += fold_acc_total

        fold_rows.append({
            "video_id": vid, "n_rallies": len(held),
            "tp": fold_tp, "fp": fold_fp, "fn": fold_fn,
            "f1": fold_f1, "action_acc": fold_acc,
        })

        _, _, cum_f1 = _f1(agg_tp, agg_fp, agg_fn)
        cum_acc = agg_acc_correct / max(1, agg_acc_total)
        console.print(
            f"  [{fold_idx}/{len(video_ids)}] {vid[:8]} ({len(held)}r): "
            f"F1={fold_f1:.1%} acc={fold_acc:.1%} | "
            f"cum F1={cum_f1:.1%} acc={cum_acc:.1%} ({time.time()-t_fold:.0f}s)"
        )

    total = time.time() - t_start
    p, r, f1 = _f1(agg_tp, agg_fp, agg_fn)
    acc = agg_acc_correct / max(1, agg_acc_total)

    console.print(f"\n[bold]Aggregate ({len(fold_rows)} folds, {total/60:.1f} min)[/bold]")
    console.print(
        f"  Contact F1: [bold]{f1:.1%}[/bold] (P={p:.1%}, R={r:.1%})  "
        f"TP={agg_tp} FP={agg_fp} FN={agg_fn}"
    )
    console.print(f"  Action accuracy: [bold]{acc:.1%}[/bold] ({agg_acc_correct}/{agg_acc_total})")

    table = Table(title="Per-class F1 (candidate decoder, LOO-per-video)")
    table.add_column("Class", style="bold")
    for col in ("TP", "FP", "FN", "P", "R", "F1"):
        table.add_column(col, justify="right")
    for c in ACTION_TYPES:
        cp, cr, cf1 = _f1(class_tally[c]["tp"], class_tally[c]["fp"],
                          class_tally[c]["fn"])
        table.add_row(
            c, str(class_tally[c]["tp"]), str(class_tally[c]["fp"]),
            str(class_tally[c]["fn"]),
            f"{cp:.1%}", f"{cr:.1%}", f"{cf1:.1%}",
        )
    console.print(table)

    # Compare vs Phase 0 baseline
    baseline_f1 = 0.880
    baseline_acc = 0.912
    delta_f1 = f1 - baseline_f1
    delta_acc = acc - baseline_acc
    verdict = (
        "STRONG PASS" if delta_f1 >= 0.03 else
        "PASS" if delta_f1 >= 0.015 else
        "WEAK" if delta_f1 >= 0.005 else
        "NO LIFT" if delta_f1 >= -0.005 else
        "REGRESSION"
    )
    color = {"STRONG PASS": "green", "PASS": "green", "WEAK": "yellow",
             "NO LIFT": "yellow", "REGRESSION": "red"}[verdict]
    console.print(
        f"\n[bold]Vs Phase 0 (88.0% F1 / 91.2% acc): "
        f"[{color}]ΔF1={delta_f1:+.1%}[/{color}] "
        f"[{color}]Δacc={delta_acc:+.1%}[/{color}] → [{color}]{verdict}[/{color}][/bold]"
    )

    # Write report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase CRF-1 — Viterbi Candidate Decoder Eval (LOO-per-video)")
    lines.append("")
    lines.append(f"- Folds: {len(fold_rows)}")
    lines.append(f"- Skip penalty: {args.skip_penalty}")
    lines.append(f"- Min accept prob: {args.min_accept_prob}")
    lines.append(f"- Transition matrix: {args.transitions}")
    lines.append(f"- Wall-clock: {total/60:.1f} min")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Metric | Decoder | Phase 0 baseline | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Contact F1 | **{f1:.1%}** | 88.0% | **{delta_f1:+.1%}** |")
    lines.append(f"| Action accuracy | **{acc:.1%}** | 91.2% | **{delta_acc:+.1%}** |")
    lines.append(f"| TP / FP / FN | {agg_tp} / {agg_fp} / {agg_fn} | 1781/174/314 | |")
    lines.append("")
    lines.append(f"## Verdict: {verdict}")
    lines.append("")
    lines.append("## Per-class F1")
    lines.append("")
    lines.append("| Class | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in ACTION_TYPES:
        cp, cr, cf1 = _f1(class_tally[c]["tp"], class_tally[c]["fp"],
                          class_tally[c]["fn"])
        lines.append(f"| {c} | {class_tally[c]['tp']} | {class_tally[c]['fp']} "
                     f"| {class_tally[c]['fn']} | {cp:.1%} | {cr:.1%} | {cf1:.1%} |")
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")

    if args.out_json:
        payload = {
            "args": vars(args),
            "aggregate": {
                "contact_f1": f1, "precision": p, "recall": r,
                "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
                "action_accuracy": acc,
                "action_correct": agg_acc_correct, "action_total": agg_acc_total,
            },
            "baseline": {"contact_f1": baseline_f1, "action_accuracy": baseline_acc},
            "delta_f1": delta_f1,
            "delta_acc": delta_acc,
            "verdict": verdict,
            "per_class": {
                c: {
                    **class_tally[c],
                    "precision": _f1(**class_tally[c])[0],
                    "recall": _f1(**class_tally[c])[1],
                    "f1": _f1(**class_tally[c])[2],
                }
                for c in ACTION_TYPES
            },
            "folds": fold_rows,
        }
        Path(args.out_json).write_text(json.dumps(payload, indent=2, default=float))
        console.print(f"[green]JSON: {args.out_json}[/green]")


if __name__ == "__main__":
    main()
