"""Fast retrain + eval for iteration. Skips LOO CV and action retrain.

Usage:
    cd analysis
    uv run python scripts/fast_retrain_eval.py
    uv run python scripts/fast_retrain_eval.py --threshold 0.30
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from rich.console import Console

from rallycut.tracking.contact_classifier import ContactClassifier, CandidateFeatures
from rallycut.tracking.contact_detector import ContactDetectionConfig
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.train_contact_classifier import extract_candidate_features, label_candidates

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.30)
    args = parser.parse_args()

    t0 = time.time()

    rallies = load_rallies_with_action_gt()
    cfg = ContactDetectionConfig()
    console.print(f"[bold]Fast retrain + eval: {len(rallies)} rallies[/bold]")

    # Pre-compute seq probs (cached after first call)
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    seq_cache: dict[str, np.ndarray | None] = {}
    for rally in rallies:
        bps = [BallPos(frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                       confidence=bp.get("confidence", 1.0))
               for bp in (rally.ball_positions_json or [])
               if bp.get("x", 0) > 0 or bp.get("y", 0) > 0]
        pps = [PlayerPos(frame_number=pp["frameNumber"], track_id=pp["trackId"],
                         x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                         confidence=pp.get("confidence", 1.0), keypoints=pp.get("keypoints"))
               for pp in (rally.positions_json or [])]
        seq_cache[rally.rally_id] = get_sequence_probs(
            bps, pps, rally.court_split_y, rally.frame_count or 0, None,
        )

    # Extract features + labels
    all_X, all_y = [], []
    for rally in rallies:
        gt_frames = [gt.frame for gt in rally.gt_labels]
        feats, frames = extract_candidate_features(
            rally, config=cfg, gt_frames=gt_frames,
            sequence_probs=seq_cache.get(rally.rally_id),
        )
        if not feats:
            continue
        labels = label_candidates(frames, rally.gt_labels)
        for f, lab in zip(feats, labels):
            all_X.append(f.to_array())
            all_y.append(lab)

    X = np.array(all_X)
    y = np.array(all_y)
    console.print(f"  Dataset: {len(y)} ({y.sum()} pos, {(1-y).sum()} neg) [{time.time()-t0:.0f}s]")

    # Train on all (no LOO)
    clf = ContactClassifier(threshold=args.threshold)
    clf.train(X, y)
    clf.save("weights/contact_classifier/contact_classifier.pkl")
    console.print(f"  Trained + saved [{time.time()-t0:.0f}s]")

    # Eval
    from rallycut.tracking.contact_detector import detect_contacts

    tp = fp = fn = 0
    for rally in rallies:
        if not rally.ball_positions_json:
            continue
        bps = [BallPos(frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                       confidence=bp.get("confidence", 1.0))
               for bp in rally.ball_positions_json if bp.get("x", 0) > 0 or bp.get("y", 0) > 0]
        pps = [PlayerPos(frame_number=pp["frameNumber"], track_id=pp["trackId"],
                         x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                         confidence=pp.get("confidence", 1.0), keypoints=pp.get("keypoints"))
               for pp in (rally.positions_json or [])]
        seq = seq_cache.get(rally.rally_id)
        result = detect_contacts(bps, pps, frame_count=rally.frame_count, sequence_probs=seq)
        det = [c.frame for c in result.contacts]

        for gt in rally.gt_labels:
            if any(abs(gt.frame - d) <= 7 for d in det):
                tp += 1
            else:
                fn += 1
        for d in det:
            if not any(abs(d - gt.frame) <= 7 for gt in rally.gt_labels):
                fp += 1

    prec = tp / max(1, tp + fp) * 100
    rec = tp / max(1, tp + fn) * 100
    f1 = 2 * prec * rec / max(0.1, prec + rec)

    console.print(f"\n[bold]Results ({time.time()-t0:.0f}s total):[/bold]")
    console.print(f"  Contact F1: {f1:.1f}% (P={prec:.1f}%, R={rec:.1f}%)")
    console.print(f"  TP={tp} FP={fp} FN={fn}")


if __name__ == "__main__":
    main()
