"""Phase CRF-1 tuning sweep: skip_penalty × action-emission-on/off.

Problem diagnosed in the baseline decoder run:
1. Decoder never predicts block (0% F1) because MS-TCN action emission
   suppresses the small block signal after renormalisation.
2. Decoder is too conservative overall (91 "lost" real contacts vs GBM
   baseline); skipping is costless while accepting requires
   emission × emission × transition product.

This script trains the 68 LOO-per-video GBMs ONCE and caches per-rally
candidate GBM probabilities. Then it runs the Viterbi decoder under
multiple (skip_penalty, use_action_emission) configurations, reporting a
single summary table across configs.

Usage (cd analysis):
    uv run python scripts/sweep_candidate_decoder.py               # all configs
    uv run python scripts/sweep_candidate_decoder.py --limit 10    # smoke
    uv run python scripts/sweep_candidate_decoder.py --configs "0.0,1.0:action 0.5,1.0,1.5:no_action"
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
from rallycut.tracking.contact_detector import ContactDetectionConfig
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
from scripts.train_contact_classifier import extract_candidate_features

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

ACTION_TYPES = ACTIONS


@dataclass
class RallyProbs:
    """Cached GBM+action probs for one rally (under one LOO fold)."""
    rally_id: str
    candidate_frames: list[int]
    gbm_probs: np.ndarray        # (N,)
    action_probs: np.ndarray     # (N, 6) MS-TCN-derived action distribution
    teams: list[int]             # (N,)
    fps: float
    start_ms: int
    frame_count: int
    court_split_y: float | None
    gt_labels: list


def _build_action_probs_and_teams(
    rally: RallyData,
    candidate_frames: list[int],
    sequence_probs: np.ndarray | None,
    players: list[PlayerPos],
) -> tuple[np.ndarray, list[int]]:
    by_frame: dict[int, list[PlayerPos]] = defaultdict(list)
    for pp in players:
        by_frame[pp.frame_number].append(pp)

    ball_by_frame: dict[int, BallPos] = {}
    for bp in (rally.ball_positions_json or []):
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0:
            ball_by_frame[bp["frameNumber"]] = BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )

    action_probs = np.zeros((len(candidate_frames), len(ACTIONS)), dtype=np.float32)
    teams: list[int] = []
    for i, frame in enumerate(candidate_frames):
        ball = ball_by_frame.get(frame)
        team = -1
        if ball is not None:
            best_d = 1e9
            best = None
            for pp in by_frame.get(frame, []):
                d = (pp.x - ball.x) ** 2 + (pp.y - ball.y) ** 2
                if d < best_d:
                    best_d = d
                    best = pp
            if best is not None and 1 <= best.track_id <= 4:
                team = infer_team_from_player_track(best.track_id)
        teams.append(team)

        if sequence_probs is not None and sequence_probs.size > 0:
            f = max(0, min(sequence_probs.shape[1] - 1, frame))
            p = sequence_probs[:, f]
            pos = p[1:]  # skip bg
            s = float(pos.sum())
            if s > 1e-6:
                action_probs[i] = pos / s
            else:
                action_probs[i] = np.ones(len(ACTIONS)) / len(ACTIONS)
        else:
            action_probs[i] = np.ones(len(ACTIONS)) / len(ACTIONS)

    return action_probs, teams


def _per_class_tally(matches, unmatched):
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
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tolerance-ms", type=int, default=233)
    parser.add_argument("--min-accept-prob", type=float, default=0.0)
    parser.add_argument("--transitions", type=str,
                        default="reports/transition_matrix_2026_04_20.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str,
                        default="reports/candidate_decoder_sweep_2026_04_20.md")
    args = parser.parse_args()

    configs = [
        {"name": "sp0 / action", "skip_penalty": 0.0, "use_action_emission": True},
        {"name": "sp1.0 / action", "skip_penalty": 1.0, "use_action_emission": True},
        {"name": "sp1.5 / action", "skip_penalty": 1.5, "use_action_emission": True},
        {"name": "sp0 / no-act", "skip_penalty": 0.0, "use_action_emission": False},
        {"name": "sp1.0 / no-act", "skip_penalty": 1.0, "use_action_emission": False},
        {"name": "sp1.5 / no-act", "skip_penalty": 1.5, "use_action_emission": False},
    ]

    transitions = TransitionMatrix.from_json(args.transitions)
    console.print(f"[dim]Transitions: {len(transitions.probs)} contexts[/dim]")

    rallies = load_rallies_with_action_gt()
    contact_cfg = ContactDetectionConfig()

    console.print(f"[bold]Pre-computing features for {len(rallies)} rallies...[/bold]")
    t0 = time.time()
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
    console.print(f"  {len(precomputed)} rallies ({time.time()-t0:.0f}s)")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]

    # === Step 1: train 68 LOO GBMs + cache per-rally GBM probs ===
    console.print(
        f"\n[bold]LOO GBM training ({len(video_ids)} folds, cache probs)[/bold]"
    )
    probs_by_rally: dict[str, RallyProbs] = {}
    t_train = time.time()
    for fold_idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [pre for v, rs in by_video.items() if v != vid for pre in rs]
        contact_clf, _action_clf = _train_fold(train, args.threshold)

        for pre in held:
            rally = pre.rally
            feats_list, cand_frames = extract_candidate_features(
                rally, config=contact_cfg,
                gt_frames=[gt.frame for gt in rally.gt_labels],
                sequence_probs=pre.sequence_probs,
            )
            if not feats_list:
                continue
            X = np.array([f.to_array() for f in feats_list], dtype=np.float64)  # noqa: N806
            gbm_probs = contact_clf.model.predict_proba(X)[:, 1]
            action_probs, teams = _build_action_probs_and_teams(
                rally, cand_frames, pre.sequence_probs, pre.player_positions,
            )
            probs_by_rally[rally.rally_id] = RallyProbs(
                rally_id=rally.rally_id,
                candidate_frames=cand_frames,
                gbm_probs=gbm_probs,
                action_probs=action_probs,
                teams=teams,
                fps=rally.fps,
                start_ms=rally.start_ms or 0,
                frame_count=rally.frame_count or 0,
                court_split_y=rally.court_split_y,
                gt_labels=rally.gt_labels,
            )
        console.print(
            f"  [{fold_idx}/{len(video_ids)}] {vid[:8]} "
            f"({time.time()-t_fold:.0f}s, {len(probs_by_rally)} rallies cached)"
        )
    console.print(f"  Total training time: {(time.time()-t_train)/60:.1f} min")

    # === Step 2: run each decoder config against cached probs ===
    console.print(f"\n[bold]Running {len(configs)} decoder configs[/bold]")
    config_results: list[dict] = []

    for cfg in configs:
        t_cfg = time.time()
        agg_tp = agg_fp = agg_fn = 0
        agg_acc_correct = 0
        agg_acc_total = 0
        class_tally = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ACTION_TYPES}

        for rp in probs_by_rally.values():
            # Build CandidateFeatures per rally
            cf_list = []
            for i, frame in enumerate(rp.candidate_frames):
                if cfg["use_action_emission"]:
                    action_probs = rp.action_probs[i]
                else:
                    # Uniform prior over actions — transition-argmax will assign
                    action_probs = np.ones(len(ACTIONS)) / len(ACTIONS)
                cf_list.append(CandidateFeatures(
                    frame=frame,
                    gbm_contact_prob=float(rp.gbm_probs[i]),
                    action_probs=action_probs,
                    team=rp.teams[i],
                ))

            accepted = decode_rally(
                cf_list, transitions,
                skip_penalty=cfg["skip_penalty"],
                min_accept_prob=args.min_accept_prob,
            )
            pred_actions = [
                {"frame": d.frame, "action": d.action, "playerTrackId": -1}
                for d in accepted
            ]
            tol = max(1, round(rp.fps * args.tolerance_ms / 1000))
            matches, unmatched = match_contacts(rp.gt_labels, pred_actions, tolerance=tol)
            metrics = compute_metrics(matches, unmatched)
            agg_tp += metrics["tp"]
            agg_fp += metrics["fp"]
            agg_fn += metrics["fn"]
            matched_correct = sum(
                1 for m in matches
                if m.pred_action is not None and m.gt_action == m.pred_action
            )
            matched_total = sum(1 for m in matches if m.pred_action is not None)
            agg_acc_correct += matched_correct
            agg_acc_total += matched_total

            cls = _per_class_tally(matches, unmatched)
            for c in ACTION_TYPES:
                for k in ("tp", "fp", "fn"):
                    class_tally[c][k] += cls[c][k]

        p, r, f1 = _f1(agg_tp, agg_fp, agg_fn)
        acc = agg_acc_correct / max(1, agg_acc_total)
        result = {
            "config": cfg["name"],
            "skip_penalty": cfg["skip_penalty"],
            "use_action_emission": cfg["use_action_emission"],
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
            "precision": p, "recall": r, "f1": f1,
            "action_accuracy": acc,
            "per_class": {
                c: {**class_tally[c],
                    "p": _f1(**class_tally[c])[0],
                    "r": _f1(**class_tally[c])[1],
                    "f1": _f1(**class_tally[c])[2]}
                for c in ACTION_TYPES
            },
        }
        config_results.append(result)
        console.print(
            f"  {cfg['name']}: F1={f1:.1%} P={p:.1%} R={r:.1%} acc={acc:.1%} "
            f"TP={agg_tp} FP={agg_fp} FN={agg_fn}  ({time.time()-t_cfg:.0f}s)"
        )

    # === Summary table ===
    console.print("\n[bold]Sweep summary (vs Phase 0 baseline F1=88.0% acc=91.2%)[/bold]")
    table = Table(show_header=True, header_style="bold")
    for col in ("Config", "F1", "ΔF1", "P", "R", "Action Acc", "Δacc",
                "TP", "FP", "FN", "block F1"):
        table.add_column(col, justify="right" if col != "Config" else "left")
    baseline_f1 = 0.880
    baseline_acc = 0.912

    best_f1 = max(r["f1"] for r in config_results)
    for r in config_results:
        delta_f1 = r["f1"] - baseline_f1
        delta_acc = r["action_accuracy"] - baseline_acc
        style = ""
        if r["f1"] == best_f1:
            style = "bold green" if delta_f1 > 0 else "bold"
        table.add_row(
            r["config"],
            f"{r['f1']:.1%}",
            f"{delta_f1:+.1%}",
            f"{r['precision']:.1%}",
            f"{r['recall']:.1%}",
            f"{r['action_accuracy']:.1%}",
            f"{delta_acc:+.1%}",
            str(r["tp"]), str(r["fp"]), str(r["fn"]),
            f"{r['per_class']['block']['f1']:.1%}",
            style=style,
        )
    console.print(table)

    # Markdown report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Phase CRF-1 Decoder Sweep (LOO-per-video)")
    lines.append("")
    lines.append(f"- Folds: {len(video_ids)}")
    lines.append(f"- Transition matrix: {args.transitions}")
    lines.append("- Baseline: Phase 0 GBM F1=88.0%, action_acc=91.2%")
    lines.append("")
    lines.append("## Config results")
    lines.append("")
    lines.append("| Config | F1 | ΔF1 | P | R | Action Acc | Δacc | TP | FP | FN | block F1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in config_results:
        marker = " ← best" if r["f1"] == best_f1 else ""
        lines.append(
            f"| {r['config']} | {r['f1']:.1%} | {r['f1']-baseline_f1:+.1%} "
            f"| {r['precision']:.1%} | {r['recall']:.1%} "
            f"| {r['action_accuracy']:.1%} | {r['action_accuracy']-baseline_acc:+.1%} "
            f"| {r['tp']} | {r['fp']} | {r['fn']} "
            f"| {r['per_class']['block']['f1']:.1%}{marker} |"
        )
    lines.append("")
    # Per-class for best config
    best_cfg = max(config_results, key=lambda r: r["f1"])
    lines.append(f"## Per-class F1 (best config: {best_cfg['config']})")
    lines.append("")
    lines.append("| Class | TP | FP | FN | P | R | F1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in ACTION_TYPES:
        pc = best_cfg["per_class"][c]
        lines.append(
            f"| {c} | {pc['tp']} | {pc['fp']} | {pc['fn']} "
            f"| {pc['p']:.1%} | {pc['r']:.1%} | {pc['f1']:.1%} |"
        )
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")

    json_path = Path(args.out.replace(".md", ".json"))
    json_path.write_text(json.dumps({
        "baseline_f1": baseline_f1,
        "baseline_acc": baseline_acc,
        "configs": config_results,
    }, indent=2, default=float))
    console.print(f"[green]JSON: {json_path}[/green]")


if __name__ == "__main__":
    main()
