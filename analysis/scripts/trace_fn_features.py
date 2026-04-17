"""Trace feature values for anomalous FN contacts (strong signal, low score).

Helps identify which GBM feature is killing contacts that should be detected.

Usage:
    cd analysis
    uv run python scripts/trace_fn_features.py
"""

from __future__ import annotations

import csv
import math

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.tracking.contact_classifier import (
    CandidateFeatures,
    load_contact_classifier,
)
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
)
from scripts.train_contact_classifier import extract_candidate_features

console = Console()


def load_feedback(path: str = "/Users/mario/Downloads/review_feedback.csv") -> dict[tuple[str, int], dict]:
    feedback: dict[tuple[str, int], dict] = {}
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["rally_id"], int(row["gt_frame"]))
                feedback[key] = row
    except FileNotFoundError:
        pass
    return feedback


def main() -> None:
    rallies = load_rallies_with_action_gt()
    feedback = load_feedback()
    cfg = ContactDetectionConfig()
    classifier = load_contact_classifier()

    if classifier is None:
        console.print("[red]No classifier found[/red]")
        return

    # Get feature names and importances
    feature_names = CandidateFeatures.feature_names()
    importances = classifier.feature_importance()

    # Collect TP and FN feature distributions
    tp_features: list[np.ndarray] = []
    fn_features: list[np.ndarray] = []
    fn_details: list[tuple[str, int, str, float, CandidateFeatures]] = []

    fn_rallies = set()
    for (rid, frame), fb in feedback.items():
        if fb.get("error_class") == "FN_contact" and fb.get("tag") == "classifier_boundary":
            fn_rallies.add(rid)

    for rally in rallies:
        if rally.rally_id not in fn_rallies:
            continue

        features_list, candidate_frames = extract_candidate_features(rally, config=cfg)
        if not features_list:
            continue

        # Score all candidates
        preds = classifier.predict(features_list)

        # Get detected contacts
        ball_positions = [
            BallPos(frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                    confidence=bp.get("confidence", 1.0))
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]
        player_positions = [
            PlayerPos(frame_number=pp["frameNumber"], track_id=pp["trackId"],
                      x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                      confidence=pp.get("confidence", 1.0), keypoints=pp.get("keypoints"))
            for pp in (rally.positions_json or [])
        ]

        contacts = detect_contacts(ball_positions, player_positions, frame_count=rally.frame_count)
        detected_frames = {c.frame for c in contacts.contacts}

        for gt in rally.gt_labels:
            fb = feedback.get((rally.rally_id, gt.frame))
            if not fb or fb.get("error_class") != "FN_contact" or fb.get("tag") != "classifier_boundary":
                continue

            # Find nearest candidate
            best_feat = None
            best_score = None
            best_dist = float("inf")
            for feat, (_, score) in zip(features_list, preds):
                d = abs(feat.frame - gt.frame)
                if d < best_dist:
                    best_dist = d
                    best_feat = feat
                    best_score = score

            if best_feat is None or best_dist > 10:
                continue

            is_fn = not any(abs(gt.frame - d) <= 5 for d in detected_frames)
            if is_fn and best_score is not None and best_score < 0.25 and best_feat.direction_change_deg > 20:
                fn_features.append(best_feat.to_array())
                fn_details.append((rally.rally_id[:8], gt.frame, gt.action, best_score, best_feat))

        # Also collect TPs for comparison
        for feat, (is_valid, score) in zip(features_list, preds):
            if is_valid:
                tp_features.append(feat.to_array())

    if not fn_features:
        console.print("[yellow]No anomalous FN features found[/yellow]")
        return

    tp_arr = np.array(tp_features)
    fn_arr = np.array(fn_features)

    console.print(f"\n[bold]Anomalous FNs: {len(fn_features)} (dir>20° + score<0.25)[/bold]")
    console.print(f"TPs for comparison: {len(tp_features)}")

    # Compare feature distributions: which features differ most between TP and anomalous FN?
    t = Table(title="\nFeature Distribution: Anomalous FN vs TP")
    t.add_column("Feature")
    t.add_column("Importance", justify="right")
    t.add_column("FN median", justify="right")
    t.add_column("TP median", justify="right")
    t.add_column("FN/TP ratio", justify="right")
    t.add_column("Diagnosis")

    for i, name in enumerate(feature_names):
        imp = importances.get(name, 0.0)
        fn_med = float(np.median(fn_arr[:, i]))
        tp_med = float(np.median(tp_arr[:, i]))
        ratio = fn_med / tp_med if tp_med != 0 else float("inf")

        # Flag features where FN differs significantly from TP
        diag = ""
        if imp > 0.01:
            if name == "frames_since_last" and fn_med < tp_med * 0.5:
                diag = "FN much closer to prev cand → likely adjacent contact"
            elif name == "player_distance" and fn_med > tp_med * 1.5:
                diag = "FN player farther away"
            elif name == "direction_change_deg" and fn_med < tp_med * 0.5:
                diag = "FN lower direction change"
            elif name == "velocity" and fn_med < tp_med * 0.5:
                diag = "FN lower velocity"
            elif name == "ball_detection_density" and fn_med < tp_med * 0.8:
                diag = "FN sparse ball detections"

        t.add_row(
            name,
            f"{imp:.3f}",
            f"{fn_med:.3f}",
            f"{tp_med:.3f}",
            f"{ratio:.2f}",
            diag,
        )

    console.print(t)

    # Print the worst offenders: FNs with highest direction change but lowest score
    fn_sorted = sorted(fn_details, key=lambda x: x[3])  # Sort by score ascending
    t2 = Table(title="\nWorst Anomalies: High signal, lowest score (top 15)")
    t2.add_column("Rally", max_width=10)
    t2.add_column("Frame", justify="right")
    t2.add_column("Action")
    t2.add_column("Score", justify="right")
    t2.add_column("dir°", justify="right")
    t2.add_column("vel", justify="right")
    t2.add_column("p_dist", justify="right")
    t2.add_column("f_since", justify="right")
    t2.add_column("density", justify="right")
    t2.add_column("consec", justify="right")

    for rally, frame, action, score, feat in fn_sorted[:15]:
        p_dist = feat.player_distance if math.isfinite(feat.player_distance) else 1.0
        t2.add_row(
            rally, str(frame), action,
            f"{score:.3f}",
            f"{feat.direction_change_deg:.1f}",
            f"{feat.velocity:.4f}",
            f"{p_dist:.3f}",
            str(feat.frames_since_last),
            f"{feat.ball_detection_density:.2f}",
            str(feat.consecutive_detections),
        )

    console.print(t2)


if __name__ == "__main__":
    main()
