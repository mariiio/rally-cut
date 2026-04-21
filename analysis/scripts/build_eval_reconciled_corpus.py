"""Build an error corpus that mirrors the production eval harness exactly.

Unlike ``build_action_error_corpus.py`` (which uses the production-trained
classifier loaded from disk and evaluates on the SAME data it was trained
on), this script does proper 68-fold LOO:
  - for each held-out video, train a fresh GBM + action classifier on the
    other 67 videos (same as ``scripts/eval_loo_video.py``)
  - run detect_contacts + classify_rally_actions with decoder overlay
    (matching ``scripts/eval_decoder_integration.py`` overlay arm)
  - run match_contacts
  - emit error records per GT that is either unmatched (FN), matched with
    wrong action (wrong_action), or matched with wrong player (wrong_player)

The resulting corpus reflects the same error surface the honest 88% F1 /
93.4% Action Acc numbers are computed on — so the dashboard shows what
the eval actually sees, not what a train-on-eval classifier sees.

Outputs:
    outputs/action_errors/corpus_eval_reconciled.jsonl

Usage (cd analysis):
    uv run python scripts/build_eval_reconciled_corpus.py
    uv run python scripts/build_eval_reconciled_corpus.py --limit 5  # smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.candidate_decoder import TransitionMatrix
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts
from rallycut.tracking.decoder_runtime import run_decoder_over_rally
from scripts.build_action_error_corpus import compute_contact_quality
from scripts.diagnose_fn_contacts import diagnose_rally_fns
from scripts.eval_action_detection import load_rallies_with_action_gt, match_contacts
from scripts.eval_loo_video import (
    RallyPrecomputed,
    _inject_action_classifier,
    _precompute,
    _reset_action_classifier_cache,
    _train_fold,
)

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)
console = Console()

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "action_errors"


def _seq_peak_at_frame(sequence_probs, frame: int, window: int = 5) -> tuple[float, str, float]:
    """Max non-background action prob within ±window frames of frame."""
    if sequence_probs is None or sequence_probs.size == 0 or sequence_probs.ndim != 2:
        return 0.0, "", 0.0
    f_lo = max(0, frame - window)
    f_hi = min(sequence_probs.shape[1] - 1, frame + window)
    if f_hi < f_lo:
        return 0.0, "", 0.0
    window_probs = sequence_probs[:, f_lo:f_hi + 1]
    if window_probs.size == 0:
        return 0.0, "", 0.0
    from rallycut.tracking.candidate_decoder import ACTIONS
    pos = window_probs[1:]
    if pos.size == 0:
        return 0.0, "", 0.0
    peak_idx = int(pos.argmax())
    action_idx = peak_idx // pos.shape[1]
    peak_prob = float(pos.flat[peak_idx])
    action = ACTIONS[action_idx] if action_idx < len(ACTIONS) else ""
    nonbg_max = float(pos.max())
    return nonbg_max, action, peak_prob


def _process_fold(
    held_out_video: str,
    held: list[RallyPrecomputed],
    train: list[RallyPrecomputed],
    contact_cfg: ContactDetectionConfig,
    transitions: TransitionMatrix,
    threshold: float,
    tolerance_ms: int,
    skip_penalty: float,
) -> tuple[list[dict], dict[str, int]]:
    """Train classifiers on `train`, evaluate on `held`, return error records."""
    contact_clf, action_clf = _train_fold(train, threshold)
    _inject_action_classifier(action_clf if action_clf.is_trained else None)
    errors: list[dict] = []
    tallies = {"tp": 0, "fp": 0, "fn": 0, "wrong_action": 0, "wrong_player": 0, "gt_total": 0}
    try:
        for pre in held:
            rally = pre.rally
            tol_frames = max(1, round(rally.fps * tolerance_ms / 1000))

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

            decoder_contacts = run_decoder_over_rally(
                ball_positions=pre.ball_positions,
                player_positions=pre.player_positions,
                sequence_probs=pre.sequence_probs,
                classifier=contact_clf,
                contact_config=contact_cfg,
                gt_frames=[gt.frame for gt in rally.gt_labels],
                transitions=transitions,
                skip_penalty=skip_penalty,
            )

            rally_actions = classify_rally_actions(
                contact_seq, rally_id=rally.rally_id,
                use_classifier=action_clf.is_trained,
                sequence_probs=pre.sequence_probs,
                decoder_contacts=decoder_contacts,
            )

            pred = [a.to_dict() for a in rally_actions.actions]
            real_pred = [a for a in pred if not a.get("isSynthetic")]

            matches, unmatched_preds = match_contacts(
                rally.gt_labels, real_pred, tolerance=tol_frames,
            )

            # Diagnose FNs (reuses the existing diagnostic)
            fn_labels = [rally.gt_labels[i] for i, m in enumerate(matches) if m.pred_frame is None]
            fn_diagnostics = diagnose_rally_fns(
                rally, fn_labels, contact_clf, tol_frames,
                sequence_probs=pre.sequence_probs,
            )
            fn_diag_map = {d.gt_frame: d for d in fn_diagnostics}

            for i, m in enumerate(matches):
                tallies["gt_total"] += 1
                gt = rally.gt_labels[i]
                cq = compute_contact_quality(rally, m.gt_frame)

                if m.pred_frame is None:
                    tallies["fn"] += 1
                    diag = fn_diag_map.get(m.gt_frame)
                    rec = {
                        "rally_id": rally.rally_id,
                        "video_id": rally.video_id,
                        "fps": rally.fps,
                        "start_ms": rally.start_ms,
                        "gt_frame": m.gt_frame,
                        "gt_action": m.gt_action,
                        "gt_player_track_id": gt.player_track_id,
                        "pred_frame": None,
                        "pred_action": None,
                        "pred_player_track_id": None,
                        "error_class": "FN_contact",
                        "fn_subcategory": diag.category if diag else "unknown",
                        "classifier_conf": diag.classifier_confidence if diag else 0.0,
                        "nearest_cand_dist": diag.nearest_candidate_distance if diag else 9999,
                        "ball_gap_frames": diag.ball_gap_frames if diag else 9999,
                        "velocity": diag.velocity if diag else 0.0,
                        "direction_change_deg": diag.direction_change_deg if diag else 0.0,
                        "player_distance": diag.player_distance if diag else float("inf"),
                        "seq_peak_nonbg_within_5f": diag.seq_peak_nonbg_within_5f if diag else 0.0,
                        "seq_peak_action": diag.seq_peak_action if diag else "",
                        "seq_peak_action_prob": diag.seq_peak_action_prob if diag else 0.0,
                        **asdict(cq),
                    }
                    errors.append(rec)

                elif m.pred_action != m.gt_action:
                    tallies["tp"] += 1
                    tallies["wrong_action"] += 1
                    pred_row = next((p for p in real_pred if p.get("frame") == m.pred_frame), None)
                    seq_peak, seq_act, seq_prob = _seq_peak_at_frame(
                        pre.sequence_probs, m.gt_frame,
                    )
                    rec = {
                        "rally_id": rally.rally_id,
                        "video_id": rally.video_id,
                        "fps": rally.fps,
                        "start_ms": rally.start_ms,
                        "gt_frame": m.gt_frame,
                        "gt_action": m.gt_action,
                        "gt_player_track_id": gt.player_track_id,
                        "pred_frame": m.pred_frame,
                        "pred_action": m.pred_action,
                        "pred_player_track_id": pred_row.get("playerTrackId") if pred_row else None,
                        "error_class": "wrong_action",
                        "fn_subcategory": None,
                        "classifier_conf": (pred_row or {}).get("confidence", 0.0),
                        "nearest_cand_dist": abs(m.gt_frame - (m.pred_frame or 0)),
                        "ball_gap_frames": 0,
                        "velocity": 0.0,
                        "direction_change_deg": 0.0,
                        "player_distance": 0.0,
                        "seq_peak_nonbg_within_5f": seq_peak,
                        "seq_peak_action": seq_act,
                        "seq_peak_action_prob": seq_prob,
                        **asdict(cq),
                    }
                    errors.append(rec)

                else:
                    tallies["tp"] += 1

            tallies["fp"] += len(unmatched_preds)
    finally:
        _reset_action_classifier_cache()

    return errors, tallies


def reproduce_single_fold(
    video_id: str,
    *,
    threshold: float = 0.30,
    tolerance_ms: int = 233,
    skip_penalty: float = 1.0,
) -> list[dict]:
    """Reproduce a single video-fold's error records.

    Used by the freshness pre-flight in `rallycut.evaluation.corpus_freshness`
    to validate that a stored corpus still matches current code output for
    its canary video. Precomputes all rallies (~45s) + trains the held-out
    fold (~17s) then returns the errors for the canary video only.

    Wall clock: ~1 minute on the reference workstation.
    """
    contact_cfg = ContactDetectionConfig()
    transitions = TransitionMatrix.default()
    rallies = load_rallies_with_action_gt()
    precomputed: list[RallyPrecomputed] = []
    for r in rallies:
        pre = _precompute(r, contact_cfg)
        if pre is not None:
            precomputed.append(pre)
    by_video: dict[str, list[RallyPrecomputed]] = defaultdict(list)
    for p in precomputed:
        by_video[p.rally.video_id].append(p)
    held = by_video.get(video_id, [])
    train = [p for v, rs in by_video.items() if v != video_id for p in rs]
    if not held:
        raise ValueError(f"Canary video {video_id} not found in GT-labeled rallies.")
    errors, _ = _process_fold(
        video_id, held, train, contact_cfg, transitions,
        threshold, tolerance_ms, skip_penalty,
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tolerance-ms", type=int, default=233)
    parser.add_argument("--skip-penalty", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit to N folds for smoke-test")
    parser.add_argument("--out", type=str, default=str(OUTPUT_ROOT / "corpus_eval_reconciled.jsonl"))
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

    all_errors: list[dict] = []
    totals = {"tp": 0, "fp": 0, "fn": 0, "wrong_action": 0, "wrong_player": 0, "gt_total": 0}

    for idx, vid in enumerate(video_ids, start=1):
        t_fold = time.time()
        held = by_video[vid]
        train = [p for v, rs in by_video.items() if v != vid for p in rs]
        errors, tallies = _process_fold(
            vid, held, train, contact_cfg, transitions,
            args.threshold, args.tolerance_ms, args.skip_penalty,
        )
        all_errors.extend(errors)
        for k in totals:
            totals[k] += tallies[k]
        fn_cnt = tallies["fn"]
        wa_cnt = tallies["wrong_action"]
        console.print(
            f"  [{idx}/{len(video_ids)}] {vid[:8]} ({len(held)}r): "
            f"gt={tallies['gt_total']} tp={tallies['tp']} fn={fn_cnt} "
            f"wrong_action={wa_cnt} fp={tallies['fp']} "
            f"({time.time()-t_fold:.0f}s)"
        )

    # F1 sanity
    p = totals['tp'] / max(1, totals['tp'] + totals['fp'])
    r = totals['tp'] / max(1, totals['tp'] + totals['fn'])
    f1 = 2 * p * r / max(1e-9, p + r)
    non_wa_tp = totals['tp'] - totals['wrong_action']
    acc = non_wa_tp / max(1, totals['tp'])

    # Freshness _meta header: canary fingerprint lets downstream analyses
    # verify the corpus still reproduces under current code. Git-independent.
    from rallycut.evaluation.corpus_freshness import build_meta_header
    meta = build_meta_header(
        errors=all_errors,
        n_rallies=len(precomputed),
        n_gt=totals['gt_total'],
        tp=totals['tp'], fn=totals['fn'], fp=totals['fp'],
        wrong_action=totals['wrong_action'],
        f1=f1, action_acc=acc,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(json.dumps(meta, default=str) + "\n")
        for rec in all_errors:
            f.write(json.dumps(rec, default=str) + "\n")

    console.print(f"\n[bold]Wrote {len(all_errors)} error records to {out_path}[/bold]")
    console.print(f"Totals: gt={totals['gt_total']} tp={totals['tp']} "
                  f"fn={totals['fn']} wrong_action={totals['wrong_action']} fp={totals['fp']} "
                  f"({(time.time()-t_start)/60:.1f} min)")
    console.print(f"F1: {f1:.3%}  (tp/(tp+fp) = {p:.3%}, tp/(tp+fn) = {r:.3%})")
    console.print(f"Action Acc: {acc:.3%}  (correct action / matched contacts)")
    console.print(f"Canary: {meta['canary_video_id'][:8]}  "
                  f"fingerprint={meta['canary_fingerprint'][:20]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
