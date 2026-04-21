"""Per-FN stage attribution diagnostic.

For each FN in the reconciled corpus, trace the GT contact through each
pipeline stage and identify the FIRST stage that dropped it. This
distinguishes "classifier rejected" from "classifier accepted but dedup
dropped" from "dedup passed but action classifier returned None" from
"action classifier assigned but matching gave it to a different GT."

Stages:
  1. ball_tracked          Ball position within ±3f of GT, conf >= 0.3
  2. player_tracked        Any player position within ±3f of GT
  3. seq_signal            MS-TCN++ non-bg max within ±5f >= 0.3
  4. candidate_generated   Candidate emitted within ±tol of GT
  5. classifier_accepted   Any accepted candidate (post-GBM) within ±tol
  6. dedup_survived        detect_contacts output has a contact within ±tol
  7. action_labeled        rally_actions has a non-None action_type within ±tol
  8. matched_to_gt         match_contacts paired this GT with a contact
     (stage 10 in the brief; stages 5 and 9 are collapsed here since no
     separate observation of them is available in the existing pipeline
     state)

A FN's "lost_at_stage" = the first stage where the check fails.

Outputs:
    outputs/fn_stage_attribution.jsonl  — per-FN stage trace
    outputs/fn_stage_attribution_summary.json  — aggregate counts

Usage (cd analysis):
    uv run python scripts/diagnose_fn_stage_attribution.py
    uv run python scripts/diagnose_fn_stage_attribution.py --limit 5  # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.candidate_decoder import TransitionMatrix
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.decoder_runtime import run_decoder_over_rally
from scripts.eval_action_detection import load_rallies_with_action_gt, match_contacts
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _inject_action_classifier,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)
from scripts.train_contact_classifier import extract_candidate_features

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

STAGES = [
    "ball_tracked",
    "player_tracked",
    "seq_signal",
    "candidate_generated",
    "classifier_accepted",
    "dedup_survived",
    "action_labeled",
    "matched_to_gt",
]


@dataclass
class StageTrace:
    rally_id: str
    video_id: str
    fps: float
    gt_frame: int
    gt_action: str
    # Per-stage boolean checks
    ball_tracked: bool
    player_tracked: bool
    seq_signal: bool
    candidate_generated: bool
    classifier_accepted: bool
    dedup_survived: bool
    action_labeled: bool
    matched_to_gt: bool
    # Ancillary: the "lost at" stage and contextual evidence
    lost_at_stage: str  # first failing stage name, or "survived" (shouldn't happen for FNs)
    # Context for downstream triage
    nearest_candidate_distance: int  # frames to closest candidate (9999 if none)
    nearest_candidate_gbm: float  # GBM prob of nearest candidate (-1 if none)
    accepted_in_window_nearest_gbm: float  # best GBM prob of an accepted candidate in window (-1 if none)
    detected_contact_frames_in_window: list[int]
    rally_actions_in_window: list[dict]  # [{frame, action_type, player_track_id}]
    seq_peak_nonbg: float
    seq_peak_action: str
    # If action_labeled=True but matched_to_gt=False, indicates stage-10 steal.
    adjacent_gt_took_it_frame: int | None
    adjacent_gt_took_it_action: str | None


def _seq_peak_at_frame(sequence_probs, frame: int, window: int = 5) -> tuple[float, str]:
    if sequence_probs is None or sequence_probs.size == 0 or sequence_probs.ndim != 2:
        return 0.0, ""
    from rallycut.tracking.candidate_decoder import ACTIONS
    f_lo = max(0, frame - window)
    f_hi = min(sequence_probs.shape[1] - 1, frame + window)
    if f_hi < f_lo:
        return 0.0, ""
    window_probs = sequence_probs[1:, f_lo:f_hi + 1]
    if window_probs.size == 0:
        return 0.0, ""
    peak_idx = int(window_probs.argmax())
    action_idx = peak_idx // window_probs.shape[1]
    return float(window_probs.max()), (ACTIONS[action_idx] if action_idx < len(ACTIONS) else "")


def _trace_fn(
    rally,
    pre: RallyPrecomputed,
    gt_frame: int,
    gt_action: str,
    candidate_frames: list[int],
    gbm_probs: np.ndarray,
    contact_seq,
    rally_actions,
    matches,
    tolerance_frames: int,
    contact_clf,
) -> StageTrace:
    # Stage 1: ball tracker
    ball_tracked = any(
        abs(b.frame_number - gt_frame) <= 3 and b.confidence >= 0.3
        and (b.x > 0 or b.y > 0)
        for b in pre.ball_positions
    )

    # Stage 2: player tracker
    player_tracked = any(
        abs(p.frame_number - gt_frame) <= 3 for p in pre.player_positions
    )

    # Stage 3: seq signal
    seq_peak, seq_peak_action = _seq_peak_at_frame(pre.sequence_probs, gt_frame)
    seq_signal = seq_peak >= 0.3

    # Stage 4: candidate generated in window
    in_win = [
        (f, p) for f, p in zip(candidate_frames, gbm_probs, strict=True)
        if abs(f - gt_frame) <= tolerance_frames
    ]
    candidate_generated = bool(in_win)
    nearest_d = min((abs(f - gt_frame) for f, _ in in_win), default=9999)
    nearest_cand_gbm = -1.0
    if candidate_generated:
        nearest = min(in_win, key=lambda x: abs(x[0] - gt_frame))
        nearest_cand_gbm = float(nearest[1])

    # Stage 5: classifier_accepted — any candidate in window with gbm >= threshold
    threshold = contact_clf.threshold
    accepted_in_win = [(f, p) for f, p in in_win if p >= threshold]
    classifier_accepted = bool(accepted_in_win)
    best_accepted_gbm = max((p for _, p in accepted_in_win), default=-1.0)

    # Stage 6: dedup_survived — detect_contacts has a contact in window
    detected_in_win = [c for c in contact_seq.contacts if abs(c.frame - gt_frame) <= tolerance_frames]
    dedup_survived = bool(detected_in_win)

    # Stage 7: action_labeled — rally_actions entry in window with non-None action_type
    ra_entries = [
        a.to_dict() for a in rally_actions.actions if not a.to_dict().get("isSynthetic")
    ]
    ra_in_win = [a for a in ra_entries if abs(a["frame"] - gt_frame) <= tolerance_frames]
    # ClassifiedAction.to_dict emits the key "action" with ActionType.value;
    # UNKNOWN serializes as "unknown" (or None). Treat UNKNOWN/None as unlabeled.
    def _has_real_action(a: dict) -> bool:
        v = a.get("action")
        return v is not None and v != "unknown"
    action_labeled = any(_has_real_action(a) for a in ra_in_win)

    # Stage 8: matched_to_gt — the GT has pred_frame != None from match_contacts
    match_row = next((m for m in matches if m.gt_frame == gt_frame), None)
    matched_to_gt = bool(match_row and match_row.pred_frame is not None)

    # If action_labeled but not matched_to_gt, find which adjacent GT stole it
    adjacent_gt_took_frame: int | None = None
    adjacent_gt_took_action: str | None = None
    if action_labeled and not matched_to_gt:
        # Find if another GT got the nearest rally_actions entry
        for a in ra_in_win:
            for m in matches:
                if m.pred_frame == a["frame"] and m.gt_frame != gt_frame:
                    adjacent_gt_took_frame = m.gt_frame
                    adjacent_gt_took_action = m.gt_action
                    break
            if adjacent_gt_took_frame is not None:
                break

    # Decide first failing stage
    checks = [
        ("ball_tracked", ball_tracked),
        ("player_tracked", player_tracked),
        ("seq_signal", seq_signal),
        ("candidate_generated", candidate_generated),
        ("classifier_accepted", classifier_accepted),
        ("dedup_survived", dedup_survived),
        ("action_labeled", action_labeled),
        ("matched_to_gt", matched_to_gt),
    ]
    lost_at = "survived"
    for name, ok in checks:
        if not ok:
            lost_at = name
            break

    return StageTrace(
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        fps=rally.fps,
        gt_frame=gt_frame,
        gt_action=gt_action,
        ball_tracked=ball_tracked,
        player_tracked=player_tracked,
        seq_signal=seq_signal,
        candidate_generated=candidate_generated,
        classifier_accepted=classifier_accepted,
        dedup_survived=dedup_survived,
        action_labeled=action_labeled,
        matched_to_gt=matched_to_gt,
        lost_at_stage=lost_at,
        nearest_candidate_distance=nearest_d,
        nearest_candidate_gbm=nearest_cand_gbm,
        accepted_in_window_nearest_gbm=best_accepted_gbm,
        detected_contact_frames_in_window=[c.frame for c in detected_in_win],
        rally_actions_in_window=[
            {"frame": a["frame"], "action": a.get("action"),
             "pid": a.get("playerTrackId")} for a in ra_in_win
        ],
        seq_peak_nonbg=seq_peak,
        seq_peak_action=seq_peak_action,
        adjacent_gt_took_it_frame=adjacent_gt_took_frame,
        adjacent_gt_took_it_action=adjacent_gt_took_action,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tolerance-ms", type=int, default=233)
    parser.add_argument("--skip-penalty", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/fn_stage_attribution.jsonl")
    parser.add_argument("--summary", type=str, default="outputs/fn_stage_attribution_summary.json")
    args = parser.parse_args()

    t_start = time.time()
    contact_cfg = ContactDetectionConfig()
    transitions = TransitionMatrix.default()

    console.print("[bold]Loading GT rallies + precomputing features...[/bold]")
    rallies = load_rallies_with_action_gt()
    precomputed: list[RallyPrecomputed] = []
    for i, r in enumerate(rallies):
        pre = _precompute(r, contact_cfg)
        if pre is not None:
            precomputed.append(pre)
        if (i + 1) % 50 == 0 or (i + 1) == len(rallies):
            console.print(f"  [{i+1}/{len(rallies)}] precomputed ({time.time()-t_start:.0f}s)")
    console.print(f"  {len(precomputed)} rallies across {len({p.rally.video_id for p in precomputed})} videos")

    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for p in precomputed:
        by_video[p.rally.video_id].append(p)
    video_ids = sorted(by_video.keys())
    if args.limit:
        video_ids = video_ids[: args.limit]

    all_traces: list[StageTrace] = []
    total_gt = 0
    total_fn = 0

    for idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [p for v, rs in by_video.items() if v != vid for p in rs]
        contact_clf, action_clf = _train_fold(train, args.threshold)
        _inject_action_classifier(action_clf if action_clf.is_trained else None)

        fold_fn = 0
        try:
            for pre in held:
                rally = pre.rally
                tol = max(1, round(rally.fps * args.tolerance_ms / 1000))
                gt_frames = [gt.frame for gt in rally.gt_labels]

                # Candidates (pre-classifier view): we need the candidate frame
                # list + GBM probs so we can trace stages 4/5 independently.
                feats, cand_frames = extract_candidate_features(
                    rally, config=contact_cfg,
                    gt_frames=gt_frames,
                    sequence_probs=pre.sequence_probs,
                )
                if feats:
                    x = np.array([f.to_array() for f in feats], dtype=np.float64)
                    n_exp = contact_clf.model.n_features_in_
                    if x.shape[1] > n_exp:
                        x = x[:, :n_exp]
                    elif x.shape[1] < n_exp:
                        x = np.hstack([x, np.zeros((x.shape[0], n_exp - x.shape[1]))])
                    gbm_probs = contact_clf.model.predict_proba(x)[:, 1]
                else:
                    gbm_probs = np.zeros(0)

                # Detect_contacts (post-dedup view)
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

                # Decoder + overlay (post label-assignment view)
                decoder_contacts = run_decoder_over_rally(
                    ball_positions=pre.ball_positions,
                    player_positions=pre.player_positions,
                    sequence_probs=pre.sequence_probs,
                    classifier=contact_clf,
                    contact_config=contact_cfg,
                    gt_frames=gt_frames,
                    transitions=transitions,
                    skip_penalty=args.skip_penalty,
                )
                rally_actions = classify_rally_actions(
                    contact_seq, rally_id=rally.rally_id,
                    use_classifier=action_clf.is_trained,
                    sequence_probs=pre.sequence_probs,
                    decoder_contacts=decoder_contacts,
                )

                pred = [a.to_dict() for a in rally_actions.actions]
                real_pred = [a for a in pred if not a.get("isSynthetic")]
                synth_serves = [
                    a for a in pred
                    if a.get("isSynthetic") and a.get("action") == "serve"
                ]
                matches, _ = match_contacts(rally.gt_labels, real_pred, tolerance=tol)
                # Pass 2: synthetic serves — matches the production eval flow.
                from scripts.eval_action_detection import _match_synthetic_serves
                if synth_serves:
                    synth_tol = max(tol, round(rally.fps * 1.0))
                    _match_synthetic_serves(
                        matches, synth_serves, rally.gt_labels,
                        synth_tol,
                    )

                # For each FN, trace stages
                for i, m in enumerate(matches):
                    total_gt += 1
                    if m.pred_frame is None:
                        total_fn += 1
                        fold_fn += 1
                        gt = rally.gt_labels[i]
                        trace = _trace_fn(
                            rally, pre, gt.frame, gt.action,
                            list(cand_frames), gbm_probs,
                            contact_seq, rally_actions, matches,
                            tol, contact_clf,
                        )
                        all_traces.append(trace)
                    # STAGE-10-STEAL DETECTION: a GT that IS matched but whose
                    # nearest action-labeled contact was STOLEN by a different GT
                    # would still be counted as TP here. We focus on FN cases
                    # only for the diagnostic.
        finally:
            _reset_action_classifier_cache()

        console.print(f"  [{idx}/{len(video_ids)}] {vid[:8]} ({len(held)}r): fold_fn={fold_fn} "
                      f"({time.time()-t_fold:.0f}s)")

    # Aggregate
    lost_at_counts = Counter(t.lost_at_stage for t in all_traces)
    per_class = defaultdict(lambda: Counter())
    for t in all_traces:
        per_class[t.gt_action][t.lost_at_stage] += 1

    console.print(f"\n[bold]Wrote {len(all_traces)} FN traces[/bold]")
    console.print(f"Total GT: {total_gt}, Total FN: {total_fn} ({100*total_fn/max(1,total_gt):.1f}%)")

    # Display summary table
    table = Table(title="FN Lost-at-Stage Distribution")
    table.add_column("Stage", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("% of FN", justify="right")
    for stage in STAGES + ["survived"]:
        n = lost_at_counts.get(stage, 0)
        if n > 0:
            table.add_row(stage, str(n), f"{100*n/len(all_traces):.1f}%")
    console.print(table)

    table2 = Table(title="FN Lost-at-Stage by GT action")
    table2.add_column("Class", style="bold")
    for stage in STAGES:
        table2.add_column(stage.replace("_", "\n"), justify="right")
    for cls in sorted(per_class.keys()):
        row = [cls]
        for stage in STAGES:
            n = per_class[cls].get(stage, 0)
            row.append(str(n) if n else "—")
        table2.add_row(*row)
    console.print(table2)

    # Write outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in all_traces:
            f.write(json.dumps(asdict(t), default=str) + "\n")
    console.print(f"\nPer-FN traces: {out_path}")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "total_gt": total_gt,
        "total_fn": total_fn,
        "lost_at_counts": dict(lost_at_counts),
        "per_class": {k: dict(v) for k, v in per_class.items()},
        "wall_clock_seconds": time.time() - t_start,
    }, indent=2))
    console.print(f"Summary: {summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
