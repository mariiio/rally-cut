"""Phase CRF-1 emission retrain: GBM WITHOUT `frames_since_last` feature.

Hypothesis: in the sequence-decoder architecture, the decoder's gap-bucketed
transitions enforce "reasonable gap" constraints better than the GBM's
`frames_since_last` feature (which blindly kills all rapid candidates at 37%
feature importance). Removing that feature should preserve real rapid
cross-side contacts that the baseline GBM wrongly rejects, while the decoder
structurally filters the rapid noise the feature was guarding against.

Memory notes this drops standalone F1 to 75.7%. We test it in combination
with the decoder.

Implementation: zero out column 16 (`frames_since_last`) in the feature
matrix before training AND before prediction. The GBM can't split on a
constant feature, so it's effectively removed.

Usage:
    uv run python scripts/sweep_decoder_no_fsl.py
"""
# ruff: noqa: N803, N806, E701, E702  # sklearn X/Y convention + terse debug lines
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
from rallycut.tracking.contact_classifier import (
    CandidateFeatures as ContactFeatureVec,
)
from rallycut.tracking.contact_classifier import (
    ContactClassifier,
)
from rallycut.tracking.contact_detector import ContactDetectionConfig
from scripts.eval_action_detection import (
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from scripts.eval_loo_video import RallyPrecomputed, _f1, _precompute
from scripts.train_contact_classifier import extract_candidate_features

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

ACTION_TYPES = ACTIONS
FSL_FEATURE_IDX = ContactFeatureVec.feature_names().index("frames_since_last")


def _mask_fsl(X: np.ndarray) -> np.ndarray:
    """Zero out the frames_since_last column (GBM can't split on constant)."""
    Y = X.copy()
    Y[:, FSL_FEATURE_IDX] = 0.0
    return Y


def _train_fold_masked(
    train_rallies: list[RallyPrecomputed],
    threshold: float,
) -> ContactClassifier:
    """Train contact GBM with frames_since_last column masked."""
    X = np.concatenate([r.candidate_features for r in train_rallies], axis=0)
    y = np.concatenate([r.candidate_labels for r in train_rallies], axis=0)
    X_masked = _mask_fsl(X)
    clf = ContactClassifier(threshold=threshold)
    clf.train(X_masked, y)
    return clf


@dataclass
class RallyProbs:
    rally_id: str
    candidate_frames: list[int]
    gbm_probs: np.ndarray
    action_probs: np.ndarray
    teams: list[int]
    fps: float
    start_ms: int
    frame_count: int
    gt_labels: list


def _build_action_probs_and_teams(
    rally, candidate_frames, sequence_probs, players,
):
    by_frame: dict[int, list] = defaultdict(list)
    for pp in players:
        by_frame[pp.frame_number].append(pp)
    ball_by_frame = {}
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
            best_d = 1e9; best = None
            for pp in by_frame.get(frame, []):
                d = (pp.x - ball.x) ** 2 + (pp.y - ball.y) ** 2
                if d < best_d:
                    best_d = d; best = pp
            if best is not None and 1 <= best.track_id <= 4:
                team = infer_team_from_player_track(best.track_id)
        teams.append(team)

        if sequence_probs is not None and sequence_probs.size > 0:
            f = max(0, min(sequence_probs.shape[1] - 1, frame))
            p = sequence_probs[:, f]
            pos = p[1:]
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
        gt, pred = m.gt_action, m.pred_action
        if pred is None:
            if gt in tally: tally[gt]["fn"] += 1
        elif gt == pred:
            if gt in tally: tally[gt]["tp"] += 1
        else:
            if gt in tally: tally[gt]["fn"] += 1
            if pred in tally: tally[pred]["fp"] += 1
    for p in unmatched:
        a = p.get("action")
        if a in tally: tally[a]["fp"] += 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tolerance-ms", type=int, default=233)
    parser.add_argument("--transitions", type=str,
                        default="reports/transition_matrix_2026_04_20.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str,
                        default="reports/decoder_no_fsl_sweep_2026_04_20.md")
    args = parser.parse_args()

    console.print(f"[dim]Zeroing out feature idx {FSL_FEATURE_IDX} "
                  f"({ContactFeatureVec.feature_names()[FSL_FEATURE_IDX]})[/dim]")

    transitions = TransitionMatrix.from_json(args.transitions)

    rallies = load_rallies_with_action_gt()
    contact_cfg = ContactDetectionConfig()

    console.print(f"[bold]Pre-computing for {len(rallies)} rallies...[/bold]")
    t0 = time.time()
    precomputed: list[RallyPrecomputed] = []
    for rally in rallies:
        pre = _precompute(rally, contact_cfg)
        if pre is None: continue
        precomputed.append(pre)
    console.print(f"  {len(precomputed)} rallies ({time.time()-t0:.0f}s)")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for pre in precomputed:
        by_video[pre.rally.video_id].append(pre)

    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]

    console.print(f"\n[bold]LOO GBM training (no FSL) — {len(video_ids)} folds[/bold]")
    probs_by_rally: dict[str, RallyProbs] = {}
    t_train = time.time()
    for fold_idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [pre for v, rs in by_video.items() if v != vid for pre in rs]
        contact_clf = _train_fold_masked(train, args.threshold)

        for pre in held:
            rally = pre.rally
            feats_list, cand_frames = extract_candidate_features(
                rally, config=contact_cfg,
                gt_frames=[gt.frame for gt in rally.gt_labels],
                sequence_probs=pre.sequence_probs,
            )
            if not feats_list:
                continue
            X = np.array([f.to_array() for f in feats_list], dtype=np.float64)
            X_masked = _mask_fsl(X)
            gbm_probs = contact_clf.model.predict_proba(X_masked)[:, 1]
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
                gt_labels=rally.gt_labels,
            )
        console.print(
            f"  [{fold_idx}/{len(video_ids)}] {vid[:8]} "
            f"({time.time()-t_fold:.0f}s)"
        )
    console.print(f"  Total training: {(time.time()-t_train)/60:.1f} min")

    # Also evaluate the *standalone* no-FSL GBM at threshold 0.30 for ablation context
    console.print("\n[bold]Standalone no-FSL GBM @ thr=0.30 (ablation sanity)[/bold]")
    s_tp = s_fp = s_fn = 0
    for rp in probs_by_rally.values():
        accepted = [i for i, p in enumerate(rp.gbm_probs) if p >= args.threshold]
        pred_actions = [{"frame": rp.candidate_frames[i], "action": "contact",
                         "playerTrackId": -1} for i in accepted]
        tol = max(1, round(rp.fps * args.tolerance_ms / 1000))
        matches, unmatched = match_contacts(rp.gt_labels, pred_actions, tolerance=tol)
        m = compute_metrics(matches, unmatched)
        s_tp += m["tp"]; s_fp += m["fp"]; s_fn += m["fn"]
    s_p, s_r, s_f1 = _f1(s_tp, s_fp, s_fn)
    console.print(f"  No-FSL GBM standalone: F1={s_f1:.1%} P={s_p:.1%} R={s_r:.1%} "
                  f"(action-agnostic, used only to validate memory's 75.7% claim)")

    configs = [
        {"name": "no-FSL + sp0.0", "skip_penalty": 0.0},
        {"name": "no-FSL + sp0.5", "skip_penalty": 0.5},
        {"name": "no-FSL + sp1.0", "skip_penalty": 1.0},
        {"name": "no-FSL + sp1.5", "skip_penalty": 1.5},
        {"name": "no-FSL + sp2.0", "skip_penalty": 2.0},
    ]
    console.print(f"\n[bold]Running {len(configs)} decoder configs (w/ action emission)[/bold]")
    results: list[dict] = []

    for cfg in configs:
        t_cfg = time.time()
        agg_tp = agg_fp = agg_fn = 0
        agg_acc_c = agg_acc_t = 0
        class_tally = {a: {"tp": 0, "fp": 0, "fn": 0} for a in ACTION_TYPES}

        for rp in probs_by_rally.values():
            cf_list = [
                CandidateFeatures(
                    frame=frame,
                    gbm_contact_prob=float(rp.gbm_probs[i]),
                    action_probs=rp.action_probs[i],
                    team=rp.teams[i],
                )
                for i, frame in enumerate(rp.candidate_frames)
            ]
            accepted = decode_rally(
                cf_list, transitions, skip_penalty=cfg["skip_penalty"],
            )
            pred_actions = [
                {"frame": d.frame, "action": d.action, "playerTrackId": -1}
                for d in accepted
            ]
            tol = max(1, round(rp.fps * args.tolerance_ms / 1000))
            matches, unmatched = match_contacts(rp.gt_labels, pred_actions, tolerance=tol)
            m = compute_metrics(matches, unmatched)
            agg_tp += m["tp"]; agg_fp += m["fp"]; agg_fn += m["fn"]
            mc = sum(1 for mm in matches
                     if mm.pred_action is not None and mm.gt_action == mm.pred_action)
            mt = sum(1 for mm in matches if mm.pred_action is not None)
            agg_acc_c += mc; agg_acc_t += mt
            cls = _per_class_tally(matches, unmatched)
            for c in ACTION_TYPES:
                for k in ("tp", "fp", "fn"):
                    class_tally[c][k] += cls[c][k]

        p, r, f1 = _f1(agg_tp, agg_fp, agg_fn)
        acc = agg_acc_c / max(1, agg_acc_t)
        results.append({
            "config": cfg["name"], "skip_penalty": cfg["skip_penalty"],
            "f1": f1, "precision": p, "recall": r, "action_accuracy": acc,
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn,
            "per_class": {
                c: {**class_tally[c], "f1": _f1(**class_tally[c])[2]}
                for c in ACTION_TYPES
            },
        })
        console.print(
            f"  {cfg['name']}: F1={f1:.1%} P={p:.1%} R={r:.1%} "
            f"acc={acc:.1%} TP={agg_tp} FP={agg_fp} FN={agg_fn} "
            f"({time.time()-t_cfg:.0f}s)"
        )

    baseline_f1 = 0.880; baseline_acc = 0.912
    best = max(results, key=lambda r: r["f1"])

    console.print("\n[bold]Summary (vs Phase 0 F1=88.0% acc=91.2%)[/bold]")
    tbl = Table(show_header=True, header_style="bold")
    for col in ("Config", "F1", "ΔF1", "P", "R", "Action Acc", "Δacc", "TP", "FP", "FN"):
        tbl.add_column(col, justify="right" if col != "Config" else "left")
    # Reference rows
    tbl.add_row("Phase 0 GBM (baseline)", "88.0%", "-", "91.1%", "85.0%",
                "91.2%", "-", "1781", "174", "314")
    tbl.add_row("GBM w/ FSL + decoder sp1.0 (prev best)", "88.2%", "+0.2%",
                "90.4%", "86.1%", "95.5%", "+4.3%", "1803", "192", "292")
    tbl.add_row("No-FSL GBM standalone @thr=0.30 (ablation)",
                f"{s_f1:.1%}", f"{s_f1-baseline_f1:+.1%}",
                f"{s_p:.1%}", f"{s_r:.1%}", "-", "-",
                str(s_tp), str(s_fp), str(s_fn))
    for r in results:
        style = "bold green" if r is best and r["f1"] > baseline_f1 else ""
        tbl.add_row(
            r["config"], f"{r['f1']:.1%}", f"{r['f1']-baseline_f1:+.1%}",
            f"{r['precision']:.1%}", f"{r['recall']:.1%}",
            f"{r['action_accuracy']:.1%}", f"{r['action_accuracy']-baseline_acc:+.1%}",
            str(r["tp"]), str(r["fp"]), str(r["fn"]),
            style=style,
        )
    console.print(tbl)

    # Best config per-class
    pc_tbl = Table(title=f"Per-class F1 (best: {best['config']})", show_header=True)
    pc_tbl.add_column("Class", style="bold")
    for col in ("TP", "FP", "FN", "F1"):
        pc_tbl.add_column(col, justify="right")
    for c in ACTION_TYPES:
        x = best["per_class"][c]
        pc_tbl.add_row(c, str(x["tp"]), str(x["fp"]), str(x["fn"]), f"{x['f1']:.1%}")
    console.print(pc_tbl)

    # Write report
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# No-FSL Emission + Decoder Sweep (LOO-per-video)")
    lines.append("")
    lines.append(f"Feature masked: `{ContactFeatureVec.feature_names()[FSL_FEATURE_IDX]}` "
                 f"(index {FSL_FEATURE_IDX})")
    lines.append("")
    lines.append("## Reference lines")
    lines.append("")
    lines.append("- Phase 0 GBM baseline (with FSL, no decoder): F1=88.0% P=91.1% R=85.0% acc=91.2%")
    lines.append("- Previous best decoder (with FSL, sp1.0): F1=88.2% P=90.4% R=86.1% acc=95.5%")
    lines.append(f"- No-FSL GBM standalone (no decoder): F1={s_f1:.1%} P={s_p:.1%} R={s_r:.1%}")
    lines.append("")
    lines.append("## No-FSL + decoder sweep")
    lines.append("")
    lines.append("| Config | F1 | ΔF1 | P | R | Action Acc | Δacc | TP | FP | FN |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        marker = " ← best" if r is best else ""
        lines.append(
            f"| {r['config']} | {r['f1']:.1%} | {r['f1']-baseline_f1:+.1%} "
            f"| {r['precision']:.1%} | {r['recall']:.1%} "
            f"| {r['action_accuracy']:.1%} | {r['action_accuracy']-baseline_acc:+.1%} "
            f"| {r['tp']} | {r['fp']} | {r['fn']}{marker} |"
        )
    lines.append("")
    lines.append(f"## Per-class F1 (best: {best['config']})")
    lines.append("")
    lines.append("| Class | TP | FP | FN | F1 |")
    lines.append("|---|---|---|---|---|")
    for c in ACTION_TYPES:
        x = best["per_class"][c]
        lines.append(f"| {c} | {x['tp']} | {x['fp']} | {x['fn']} | {x['f1']:.1%} |")
    out.write_text("\n".join(lines))
    console.print(f"\n[green]Report: {out}[/green]")

    json_path = Path(args.out.replace(".md", ".json"))
    json_path.write_text(json.dumps({
        "no_fsl_standalone": {"f1": s_f1, "p": s_p, "r": s_r,
                              "tp": s_tp, "fp": s_fp, "fn": s_fn},
        "configs": results,
        "baseline_f1": baseline_f1, "baseline_acc": baseline_acc,
    }, indent=2, default=float))
    console.print(f"[green]JSON: {json_path}[/green]")


if __name__ == "__main__":
    main()
