"""Evaluate action detection against ground truth labels.

Loads action ground truth from the database (saved via the web editor),
runs contact detection + action classification on each rally, and computes:
- Contact recall: % of GT contacts detected (±tolerance frame window)
- Contact precision: % of detected contacts matching a GT contact
- Action accuracy: % of matched contacts correctly classified
- Per-class F1: serve, receive, set, attack, block, dig
- Confusion matrix
- Per-rally and aggregate tables

Usage:
    cd analysis
    uv run python scripts/eval_action_detection.py
    uv run python scripts/eval_action_detection.py --tolerance-ms 150  # ±150ms window
    uv run python scripts/eval_action_detection.py --rally <id>        # Specific rally
    uv run python scripts/eval_action_detection.py --redetect --config '{"min_peak_velocity": 0.008}'
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.sanity_checks import SanityViolation
from rallycut.evaluation.sanity_checks import check_all as check_sanity
from rallycut.evaluation.split import add_split_argument, apply_split
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts

console = Console()

ACTION_TYPES = ["serve", "receive", "set", "attack", "block", "dig"]


def _build_player_positions(
    positions_json: list[dict],
    rally_id: str | None = None,
    inject_pose: bool = False,
) -> list:
    """Build PlayerPosition list from stored JSON, optionally injecting pose keypoints."""
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    pose_kps: dict[tuple[int, int], list[list[float]]] = {}
    if inject_pose and rally_id:
        from rallycut.tracking.pose_attribution.pose_cache import load_pose_cache

        pose_data = load_pose_cache(rally_id)
        if pose_data is not None and len(pose_data["frames"]) > 0:
            for i in range(len(pose_data["frames"])):
                key = (int(pose_data["frames"][i]), int(pose_data["track_ids"][i]))
                pose_kps[key] = pose_data["keypoints"][i].tolist()

    result = []
    for pp in positions_json:
        kps = pose_kps.get((pp["frameNumber"], pp["trackId"])) if pose_kps else None
        if kps is None and "keypoints" in pp:
            kps = pp["keypoints"]
        result.append(PlayerPos(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp["width"],
            height=pp["height"],
            confidence=pp.get("confidence", 1.0),
            keypoints=kps,
        ))
    return result


@dataclass
class GtLabel:
    frame: int
    action: str
    player_track_id: int
    ball_x: float | None = None
    ball_y: float | None = None


@dataclass
class RallyData:
    rally_id: str
    video_id: str
    gt_labels: list[GtLabel]
    ball_positions_json: list[dict] | None
    positions_json: list[dict] | None
    contacts_json: dict | None
    actions_json: dict | None
    frame_count: int
    fps: float
    court_split_y: float | None
    start_ms: int = 0
    # Session 5: per-rally score GT ('A' | 'B' | None each).
    gt_serving_team: str | None = None
    gt_point_winner: str | None = None


def load_rallies_with_action_gt(
    rally_id: str | None = None,
) -> list[RallyData]:
    """Load rallies that have action ground truth labels."""
    where_clauses = ["pt.action_ground_truth_json IS NOT NULL"]
    params: list[str] = []

    if rally_id:
        where_clauses.append("r.id = %s")
        params.append(rally_id)

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            r.id as rally_id,
            r.video_id,
            pt.action_ground_truth_json,
            pt.ball_positions_json,
            pt.positions_json,
            pt.contacts_json,
            pt.actions_json,
            pt.frame_count,
            pt.fps,
            pt.court_split_y,
            r.start_ms,
            r.gt_serving_team,
            r.gt_point_winner
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE {where_sql}
        ORDER BY r.video_id, r.start_ms
    """

    results: list[RallyData] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            for row in rows:
                (
                    rally_id_val,
                    video_id_val,
                    action_gt_json,
                    ball_positions_json,
                    positions_json,
                    contacts_json,
                    actions_json,
                    frame_count,
                    fps,
                    court_split_y,
                    start_ms_val,
                    gt_serving_team_val,
                    gt_point_winner_val,
                ) = row

                gt_labels = []
                if action_gt_json:
                    for label in action_gt_json:
                        gt_labels.append(GtLabel(
                            frame=label["frame"],
                            action=label["action"],
                            player_track_id=label.get("playerTrackId", -1),
                            ball_x=label.get("ballX"),
                            ball_y=label.get("ballY"),
                        ))

                results.append(RallyData(
                    rally_id=rally_id_val,
                    video_id=video_id_val,
                    gt_labels=gt_labels,
                    ball_positions_json=ball_positions_json,
                    positions_json=positions_json,
                    contacts_json=contacts_json,
                    actions_json=actions_json,
                    frame_count=frame_count or 0,
                    fps=fps or 30.0,
                    court_split_y=court_split_y,
                    start_ms=start_ms_val or 0,
                    gt_serving_team=gt_serving_team_val,
                    gt_point_winner=gt_point_winner_val,
                ))

    return results


def _load_match_team_assignments(
    video_ids: set[str],
    min_confidence: float = 0.70,
    rally_positions: dict[str, list[Any]] | None = None,
) -> dict[str, dict[int, int]]:
    """Load match-level team assignments from match_analysis_json.

    Only includes rallies where assignment confidence >= min_confidence.
    Convention: players 1-2 = team 0 (near), 3-4 = team 1 (far),
    flipped by cumulative side switches.

    When rally_positions is provided, each rally's team labels are verified
    against actual player Y positions (fixes ~8% inversions from wrong
    initial assignment or missed side switches).
    """
    from rallycut.tracking.match_tracker import build_match_team_assignments

    if not video_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """

    result: dict[str, dict[int, int]] = {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            rows = cur.fetchall()

    for _video_id_val, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        result.update(build_match_team_assignments(
            ma_json, min_confidence, rally_positions=rally_positions,
        ))

    return result


def _load_track_to_player_maps(
    video_ids: set[str],
) -> dict[str, dict[int, int]]:
    """Load track_to_player maps from match_analysis_json (all confidences).

    Returns rally_id -> {track_id: player_id (1-4)}.
    """
    if not video_ids:
        return {}

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """

    result: dict[str, dict[int, int]] = {}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            rows = cur.fetchall()

    for _video_id_val, ma_json in rows:
        if not isinstance(ma_json, dict):
            continue
        for rally_entry in ma_json.get("rallies", []):
            rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
            t2p = rally_entry.get("trackToPlayer") or rally_entry.get(
                "track_to_player", {}
            )
            if rid and t2p:
                result[rid] = {int(k): int(v) for k, v in t2p.items()}

    return result


def _train_reid_classifiers(
    video_ids: set[str],
) -> dict[str, Any]:
    """Train per-video ReID classifiers from reference crops.

    Returns video_id -> trained PlayerReIDClassifier (or missing if no crops).
    """
    from rallycut.evaluation.tracking.db import get_video_path

    classifiers: dict[str, Any] = {}

    for vid in video_ids:
        # Check for reference crops
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                       FROM player_reference_crops
                       WHERE video_id = %s
                       ORDER BY player_id, created_at""",
                    [vid],
                )
                crop_rows = cur.fetchall()

        if not crop_rows:
            continue

        video_path = get_video_path(vid)
        if video_path is None:
            continue

        from rallycut.tracking.reid_embeddings import (
            PlayerReIDClassifier,
            extract_crops_from_video,
        )

        crop_infos = [
            {
                "player_id": r[0], "frame_ms": r[1],
                "bbox_x": r[2], "bbox_y": r[3],
                "bbox_w": r[4], "bbox_h": r[5],
            }
            for r in crop_rows
        ]

        try:
            crops_by_player = extract_crops_from_video(video_path, crop_infos)
            if len(crops_by_player) < 2:
                continue

            classifier = PlayerReIDClassifier()
            stats = classifier.train(crops_by_player)
            n_crops = sum(len(c) for c in crops_by_player.values())
            console.print(
                f"  ReID [{vid[:8]}]: {n_crops} crops → "
                f"{len(crops_by_player)} players, acc={stats['train_acc']:.0%}"
            )
            classifiers[vid] = classifier
        except Exception:
            console.print(f"  [dim]ReID [{vid[:8]}]: training failed, skipping[/dim]")

    return classifiers


def _compute_reid_predictions_for_rally(
    classifier: Any,
    video_id: str,
    rally_start_ms: int,
    rally_fps: float,
    contacts: list,
    positions_json: list[dict],
    track_to_player: dict[int, int],
) -> dict[int, dict[str, Any]]:
    """Compute ReID predictions for contacts in a rally.

    Returns {contact_frame: {"best_tid": int, "margin": float}}.
    """
    import cv2
    import numpy as np

    from rallycut.evaluation.tracking.db import get_video_path

    video_path = get_video_path(video_id)
    if video_path is None:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    pos_by_frame_track: dict[tuple[int, int], dict] = {}
    for pp in positions_json:
        pos_by_frame_track[(pp["frameNumber"], pp["trackId"])] = pp

    rally_start_frame = int(rally_start_ms / 1000.0 * rally_fps)
    predictions: dict[int, dict[str, Any]] = {}

    try:
        for contact in contacts:
            if not contact.player_candidates or len(contact.player_candidates) < 2:
                continue

            candidate_info: list[tuple[int, int, np.ndarray]] = []

            for cand_tid, _dist in contact.player_candidates:
                cand_pid = track_to_player.get(cand_tid, -1)
                if cand_pid < 0:
                    continue

                best_pos = None
                for delta in range(6):
                    for fn in [contact.frame + delta, contact.frame - delta]:
                        if fn < 0:
                            continue
                        pos = pos_by_frame_track.get((fn, cand_tid))
                        if pos is not None:
                            best_pos = pos
                            break
                    if best_pos is not None:
                        break

                if best_pos is None:
                    continue

                abs_fn = rally_start_frame + best_pos["frameNumber"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_fn)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                h, w = frame.shape[:2]
                bx = best_pos["x"]
                by = best_pos["y"]
                bw = best_pos["width"]
                bh = best_pos["height"]
                x1 = max(0, int((bx - bw / 2) * w))
                y1 = max(0, int((by - bh / 2) * h))
                x2 = min(w, int((bx + bw / 2) * w))
                y2 = min(h, int((by + bh / 2) * h))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
                    continue

                candidate_info.append((cand_tid, cand_pid, crop))

            if len(candidate_info) < 2:
                continue

            crops = [c[2] for c in candidate_info]
            probs_list = classifier.predict(crops)

            best_tid = -1
            best_prob = -1.0
            second_prob = -1.0

            for idx, (cand_tid, cand_pid, _crop) in enumerate(candidate_info):
                prob = probs_list[idx].get(cand_pid, 0.0)
                if prob > best_prob:
                    second_prob = best_prob
                    best_prob = prob
                    best_tid = cand_tid
                elif prob > second_prob:
                    second_prob = prob

            margin = best_prob - second_prob if second_prob >= 0 else 0.0
            predictions[contact.frame] = {
                "best_tid": best_tid,
                "margin": margin,
            }
    finally:
        cap.release()

    return predictions


@dataclass
class MatchResult:
    """Result of matching GT to predicted contacts."""
    gt_frame: int
    gt_action: str
    pred_frame: int | None  # None if unmatched
    pred_action: str | None  # None if unmatched
    player_correct: bool = False
    player_evaluable: bool = True  # False if GT track_id doesn't exist in tracking
    court_side_correct: bool | None = None  # None if not evaluable


def match_contacts(
    gt_labels: list[GtLabel],
    pred_actions: list[dict],
    tolerance: int = 3,
    available_track_ids: set[int] | None = None,
    team_assignments: dict[int, int] | None = None,
) -> tuple[list[MatchResult], list[dict]]:
    """Match GT labels to predicted actions using frame tolerance.

    Args:
        gt_labels: Ground truth labels.
        pred_actions: Predicted action dicts.
        tolerance: Frame tolerance for matching.
        available_track_ids: Track IDs present in current tracking data.
            When provided, marks matches as player_evaluable=False if the GT
            track_id doesn't exist in the current tracking data.
        team_assignments: Optional track_id -> team (0=near, 1=far) for
            court_side accuracy evaluation.

    Returns:
        Tuple of (matched GT results, unmatched predictions).
    """
    from scipy.optimize import linear_sum_assignment

    results: list[MatchResult] = []

    # Sort GT and predictions by frame
    gt_sorted = sorted(gt_labels, key=lambda gt: gt.frame)
    pred_sorted = sorted(pred_actions, key=lambda a: a.get("frame", 0))

    team_to_side = {0: "near", 1: "far"}

    # Build cost matrix for Hungarian matching (optimal bipartite assignment)
    # Cost = frame distance; entries beyond tolerance get a large cost to prevent matching
    big_cost = tolerance + 1
    n_gt = len(gt_sorted)
    n_pred = len(pred_sorted)

    if n_gt > 0 and n_pred > 0:
        import numpy as np

        cost = np.full((n_gt, n_pred), big_cost, dtype=np.float64)
        for gi, gt in enumerate(gt_sorted):
            for pi, pred in enumerate(pred_sorted):
                dist = abs(gt.frame - pred.get("frame", 0))
                if dist <= tolerance:
                    cost[gi][pi] = dist

        gt_inds, pred_inds = linear_sum_assignment(cost)

        # Build match mapping (only keep pairs within tolerance)
        gt_to_pred: dict[int, int] = {}
        for gi, pi in zip(gt_inds, pred_inds):
            if cost[gi][pi] <= tolerance:
                gt_to_pred[gi] = pi
        used_preds = set(gt_to_pred.values())
    else:
        gt_to_pred = {}
        used_preds: set[int] = set()

    for gi, gt in enumerate(gt_sorted):
        # Determine if GT track_id is evaluable
        evaluable = True
        if available_track_ids is not None and gt.player_track_id >= 0:
            evaluable = gt.player_track_id in available_track_ids

        pi = gt_to_pred.get(gi)
        if pi is not None:
            pred = pred_sorted[pi]

            # Court-side accuracy: check if predicted court_side matches
            # expected side for the GT player's team
            cs_correct: bool | None = None
            if team_assignments and gt.player_track_id >= 0:
                gt_team = team_assignments.get(gt.player_track_id)
                pred_cs = pred.get("courtSide")
                if gt_team is not None and pred_cs in ("near", "far"):
                    cs_correct = pred_cs == team_to_side[gt_team]

            results.append(MatchResult(
                gt_frame=gt.frame,
                gt_action=gt.action,
                pred_frame=pred.get("frame"),
                pred_action=pred.get("action"),
                player_correct=(gt.player_track_id == pred.get("playerTrackId", -1)),
                player_evaluable=evaluable,
                court_side_correct=cs_correct,
            ))
        else:
            results.append(MatchResult(
                gt_frame=gt.frame,
                gt_action=gt.action,
                pred_frame=None,
                pred_action=None,
                player_evaluable=evaluable,
            ))

    # Collect unmatched predictions
    unmatched = [pred_sorted[i] for i in range(len(pred_sorted)) if i not in used_preds]

    return results, unmatched


def _match_synthetic_serves(
    matches: list[MatchResult],
    synth_serves: list[dict],
    gt_labels: list[GtLabel],
    synth_tolerance: int,
    available_track_ids: set[int] | None = None,
    team_assignments: dict[int, int] | None = None,
) -> None:
    """Match synthetic serves against unmatched GT serves (in-place).

    Pass 2 of the two-pass matching strategy. Real predictions are matched
    first (Pass 1); this function fills remaining GT serve slots with
    synthetic serves using a wider tolerance (~1s) to account for frame
    estimation uncertainty.

    Each synthetic serve can match at most one GT serve. GT contacts that
    already have a real prediction match are not affected.
    """
    team_to_side = {0: "near", 1: "far"}
    used_synth_indices: set[int] = set()

    for m_idx, m in enumerate(matches):
        if m.pred_frame is not None or m.gt_action != "serve":
            continue
        for s_idx, synth in enumerate(synth_serves):
            if s_idx in used_synth_indices:
                continue
            s_frame = synth.get("frame", 0)
            if abs(m.gt_frame - s_frame) > synth_tolerance:
                continue

            # Find GT label for player attribution evaluation
            gt_tid = -1
            for gt in gt_labels:
                if gt.frame == m.gt_frame:
                    gt_tid = gt.player_track_id
                    break

            evaluable = True
            if available_track_ids is not None and gt_tid >= 0:
                evaluable = gt_tid in available_track_ids

            cs_correct: bool | None = None
            if team_assignments and gt_tid >= 0:
                gt_team = team_assignments.get(gt_tid)
                pred_cs = synth.get("courtSide")
                if gt_team is not None and pred_cs in ("near", "far"):
                    cs_correct = pred_cs == team_to_side[gt_team]

            matches[m_idx] = MatchResult(
                gt_frame=m.gt_frame,
                gt_action=m.gt_action,
                pred_frame=s_frame,
                pred_action=synth.get("action"),
                player_correct=(gt_tid == synth.get("playerTrackId", -1)),
                player_evaluable=evaluable,
                court_side_correct=cs_correct,
            )
            used_synth_indices.add(s_idx)
            break


def compute_metrics(
    matches: list[MatchResult],
    unmatched_preds: list[dict],
) -> dict:
    """Compute contact detection and action classification metrics."""
    total_gt = len(matches)
    matched = [m for m in matches if m.pred_frame is not None]
    unmatched_gt = [m for m in matches if m.pred_frame is None]

    tp = len(matched)
    fn = len(unmatched_gt)
    fp = len(unmatched_preds)

    recall = tp / max(1, total_gt)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    # Action accuracy (among matched contacts)
    action_correct = sum(1 for m in matched if m.gt_action == m.pred_action)
    action_accuracy = action_correct / max(1, tp)

    # Player attribution accuracy
    player_correct = sum(1 for m in matched if m.player_correct)
    player_accuracy = player_correct / max(1, tp)

    # Evaluable player attribution (excluding GT labels with stale track_ids)
    evaluable_matched = [m for m in matched if m.player_evaluable]
    evaluable_correct = sum(1 for m in evaluable_matched if m.player_correct)
    evaluable_total = len(evaluable_matched)
    evaluable_accuracy = evaluable_correct / max(1, evaluable_total)
    evaluable_skipped = tp - evaluable_total

    # Per-class metrics
    per_class: dict[str, dict[str, float]] = {}
    for action in ACTION_TYPES:
        class_gt = [m for m in matches if m.gt_action == action]
        class_tp = [m for m in class_gt if m.pred_action == action]
        class_fn = [m for m in class_gt if m.pred_action != action]
        class_fp_matched = [m for m in matched if m.pred_action == action and m.gt_action != action]
        class_fp_unmatched = [p for p in unmatched_preds if p.get("action") == action]

        c_tp = len(class_tp)
        c_fn = len(class_fn)
        c_fp = len(class_fp_matched) + len(class_fp_unmatched)

        c_precision = c_tp / max(1, c_tp + c_fp)
        c_recall = c_tp / max(1, c_tp + c_fn)
        c_f1 = 2 * c_precision * c_recall / max(1e-9, c_precision + c_recall)

        per_class[action] = {
            "tp": c_tp,
            "fp": c_fp,
            "fn": c_fn,
            "precision": c_precision,
            "recall": c_recall,
            "f1": c_f1,
        }

    # Court-side accuracy (where evaluable)
    cs_evaluable = [m for m in matched if m.court_side_correct is not None]
    cs_correct = sum(1 for m in cs_evaluable if m.court_side_correct)
    cs_total = len(cs_evaluable)
    cs_accuracy = cs_correct / max(1, cs_total)

    return {
        "total_gt": total_gt,
        "total_pred": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "action_accuracy": action_accuracy,
        "player_accuracy": player_accuracy,
        "player_evaluable_accuracy": evaluable_accuracy,
        "player_evaluable_total": evaluable_total,
        "player_evaluable_correct": evaluable_correct,
        "player_evaluable_skipped": evaluable_skipped,
        "court_side_accuracy": cs_accuracy,
        "court_side_total": cs_total,
        "court_side_correct": cs_correct,
        "per_class": per_class,
    }


def build_confusion_matrix(
    matches: list[MatchResult],
) -> dict[str, dict[str, int]]:
    """Build confusion matrix from matched contacts."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for m in matches:
        if m.pred_action is not None:
            matrix[m.gt_action][m.pred_action] += 1
        else:
            matrix[m.gt_action]["MISS"] += 1

    return {k: dict(v) for k, v in matrix.items()}


def _run_threshold_sweep(
    rallies: list[RallyData],
    args: argparse.Namespace,
) -> None:
    """Sweep classifier thresholds and report P/R/F1 tradeoff."""
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.contact_classifier import load_contact_classifier
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

    # Load calibrators and match teams once
    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None
    # Build positions lookup for team assignment verification.
    # Use _build_player_positions so inline DB keypoints flow to pose_attribution —
    # parity with main() so sweep attribution metrics are comparable.
    rally_pos_lookup: dict[str, list[PlayerPos]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _build_player_positions(
                r.positions_json, rally_id=r.rally_id, inject_pose=args.pose
            )
    match_teams_by_rally = _load_match_team_assignments(
        video_ids, min_confidence=0.70, rally_positions=rally_pos_lookup,
    )

    console.print(f"\n[bold]Threshold sweep on {len(rallies)} rallies[/bold]\n")

    sweep_table = Table(title="Contact Classifier Threshold Sweep")
    sweep_table.add_column("Threshold", justify="right")
    sweep_table.add_column("TP", justify="right")
    sweep_table.add_column("FP", justify="right")
    sweep_table.add_column("FN", justify="right")
    sweep_table.add_column("Precision", justify="right")
    sweep_table.add_column("Recall", justify="right")
    sweep_table.add_column("F1", justify="right")
    sweep_table.add_column("Action Acc", justify="right")
    sweep_table.add_column("Attribution", justify="right")
    sweep_table.add_column("Court-Side", justify="right")

    for threshold in thresholds:
        # Load classifier with this threshold
        classifier = load_contact_classifier()
        if classifier is None:
            console.print("[red]No trained classifier found[/red]")
            return
        classifier.threshold = threshold

        all_matches_sweep: list[MatchResult] = []
        all_unmatched_sweep: list[dict] = []

        for rally in rallies:
            if not rally.ball_positions_json:
                continue

            ball_positions = [
                BallPos(
                    frame_number=bp["frameNumber"],
                    x=bp["x"], y=bp["y"],
                    confidence=bp.get("confidence", 1.0),
                )
                for bp in rally.ball_positions_json
                if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
            ]

            player_positions = []
            if rally.positions_json:
                player_positions = _build_player_positions(
                    rally.positions_json,
                    rally_id=rally.rally_id,
                    inject_pose=args.pose,
                )

            match_teams = match_teams_by_rally.get(rally.rally_id)

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=rally.court_split_y,
                frame_count=rally.frame_count or None,
                classifier=classifier,
                team_assignments=match_teams,
                court_calibrator=calibrators.get(rally.video_id),
            )

            rally_actions = classify_rally_actions(
                contacts, rally.rally_id,
                match_team_assignments=match_teams,
            )
            pred_actions = [a.to_dict() for a in rally_actions.actions]
            real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

            tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))
            avail_tids: set[int] | None = None
            if rally.positions_json:
                avail_tids = {pp["trackId"] for pp in rally.positions_json}

            matches, unmatched = match_contacts(
                rally.gt_labels, real_pred,
                tolerance=tolerance_frames,
                available_track_ids=avail_tids,
                team_assignments=match_teams,
            )

            # Pass 2: synthetic serves → unmatched GT serves
            sweep_synth = [
                a for a in pred_actions
                if a.get("isSynthetic") and a.get("action") == "serve"
            ]
            if sweep_synth:
                synth_tol = max(tolerance_frames, round(rally.fps * 1.0))
                _match_synthetic_serves(
                    matches, sweep_synth, rally.gt_labels,
                    synth_tol, avail_tids,
                    team_assignments=match_teams,
                )
            all_matches_sweep.extend(matches)
            all_unmatched_sweep.extend(unmatched)

        metrics = compute_metrics(all_matches_sweep, all_unmatched_sweep)
        is_current = threshold == 0.35
        style = "bold" if is_current else ""
        cs_acc = metrics.get("court_side_accuracy")
        cs_str = f"{cs_acc:.1%}" if cs_acc is not None else "-"
        attr_str = f"{metrics['player_accuracy']:.1%}"
        sweep_table.add_row(
            f"{threshold:.2f}" + (" *" if is_current else ""),
            str(metrics["tp"]),
            str(metrics["fp"]),
            str(metrics["fn"]),
            f"{metrics['precision']:.1%}",
            f"{metrics['recall']:.1%}",
            f"{metrics['f1']:.1%}",
            f"{metrics['action_accuracy']:.1%}",
            attr_str,
            cs_str,
            style=style,
        )
        console.print(
            f"  threshold={threshold:.2f}: F1={metrics['f1']:.1%} "
            f"ActAcc={metrics['action_accuracy']:.1%} "
            f"Attr={attr_str} CS={cs_str}"
        )

    console.print(sweep_table)
    console.print("\n[dim]* = current default threshold[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate action detection vs ground truth")
    parser.add_argument("--rally", type=str, help="Specific rally ID to evaluate")
    parser.add_argument("--tolerance-ms", type=int, default=167, help="Time tolerance in ms for matching (default: 167, ~5 frames at 30fps)")
    parser.add_argument("--stored", action="store_true", help="Use stored actions from DB instead of re-running detection (default: re-detect)")
    parser.add_argument("--config", type=str, help="JSON config overrides for ContactDetectionConfig")
    parser.add_argument("--classifier", type=str, help="Path to trained contact classifier model")
    parser.add_argument("--no-classifier", action="store_true", help="Disable auto-loading of trained contact classifier (force hand-tuned gates)")
    parser.add_argument("--no-action-classifier", action="store_true", help="Disable learned action type classifier (force rule-based state machine)")
    parser.add_argument("--sweep-thresholds", action="store_true", help="Sweep classifier thresholds [0.25-0.55] and report P/R/F1 tradeoff")
    parser.add_argument("--reid", action="store_true", help="Enable ReID re-attribution (Pass 3) using per-video reference crops")
    parser.add_argument("--visual", action="store_true", help="Enable visual attribution using VideoMAE per-player action classifier")
    parser.add_argument("--exclude-videos", type=str, help="Comma-separated video ID prefixes to exclude (held-out test set)")
    parser.add_argument("--only-videos", type=str, help="Comma-separated video ID prefixes to include (held-out test set eval)")
    parser.add_argument("--pose", action="store_true", help="Inject YOLO-Pose keypoints from cache into player positions (enables pose action features)")
    add_split_argument(parser)
    args = parser.parse_args()

    # Build ContactDetectionConfig from overrides
    contact_config: ContactDetectionConfig | None = None
    if args.config:
        try:
            overrides = json.loads(args.config)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --config: {e}[/red]")
            return
        try:
            contact_config = ContactDetectionConfig(**overrides)
        except TypeError as e:
            console.print(f"[red]Invalid config field: {e}[/red]")
            return

    # Load classifier if specified
    contact_classifier = None
    if args.classifier:
        from rallycut.tracking.contact_classifier import ContactClassifier
        contact_classifier = ContactClassifier.load(args.classifier)
        console.print(f"[bold]Using trained classifier: {args.classifier}[/bold]")

    rallies = load_rallies_with_action_gt(rally_id=args.rally)

    # Apply hash-based train/held-out split
    rallies = apply_split(rallies, args)

    # Filter by video ID prefixes (held-out test set support)
    if args.exclude_videos:
        prefixes = [p.strip() for p in args.exclude_videos.split(",")]
        before = len(rallies)
        rallies = [r for r in rallies if not any(r.video_id.startswith(p) for p in prefixes)]
        console.print(f"  Excluded {before - len(rallies)} rallies from {len(prefixes)} video(s)")
    if args.only_videos:
        prefixes = [p.strip() for p in args.only_videos.split(",")]
        before = len(rallies)
        rallies = [r for r in rallies if any(r.video_id.startswith(p) for p in prefixes)]
        console.print(f"  Filtered to {len(rallies)} rallies from {len(prefixes)} video(s) (was {before})")

    if not rallies:
        console.print("[red]No rallies found with action ground truth labels.[/red]")
        console.print("Label actions in the web editor first (Label Actions button).")
        return

    if args.sweep_thresholds:
        _run_threshold_sweep(rallies, args)
        return

    # Load court calibrations per video for perspective-corrected player attribution
    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    n_calibrated = 0
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
            n_calibrated += 1
        else:
            calibrators[vid] = None

    # Build positions lookup for team assignment verification
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

    rally_pos_lookup: dict[str, list[PlayerPos]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPos(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in r.positions_json
            ]

    # Load match-level team assignments for re-attribution (confidence-gated)
    match_teams_by_rally = _load_match_team_assignments(
        video_ids, min_confidence=0.70, rally_positions=rally_pos_lookup,
    )
    n_with_match = sum(1 for r in rallies if r.rally_id in match_teams_by_rally)

    # Load ReID classifiers if requested
    reid_classifiers: dict[str, Any] = {}
    track_to_player_maps: dict[str, dict[int, int]] = {}
    if args.reid:
        console.print("[bold]Training ReID classifiers from reference crops...[/bold]")
        reid_classifiers = _train_reid_classifiers(video_ids)
        if reid_classifiers:
            track_to_player_maps = _load_track_to_player_maps(video_ids)
        console.print(
            f"  ReID classifiers: {len(reid_classifiers)}/{len(video_ids)} videos\n"
        )

    # Load visual attribution classifier if requested
    visual_classifier = None
    if args.visual:
        from rallycut.tracking.visual_attribution import load_visual_attribution_classifier
        visual_classifier = load_visual_attribution_classifier()
        if visual_classifier is None:
            console.print("[yellow]Warning: No trained visual attribution model found.[/yellow]")

    use_pose = getattr(args, "pose", False)

    console.print(f"\n[bold]Evaluating {len(rallies)} rallies with action ground truth[/bold]")
    console.print(f"  Court calibration: {n_calibrated}/{len(video_ids)} videos")
    console.print(f"  Match teams (conf>=0.70): {n_with_match}/{len(rallies)} rallies")
    if args.reid:
        console.print(f"  ReID classifiers: {len(reid_classifiers)}/{len(video_ids)} videos")
    if visual_classifier is not None:
        console.print("  Visual attribution classifier: loaded")
    if use_pose:
        console.print("  Pose keypoint injection: enabled")
    console.print()

    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    total_gt_serves = 0
    serves_present = 0

    # Sanity violation tracking
    all_violations: list[SanityViolation] = []

    # Per-rally results table
    rally_table = Table(title="Per-Rally Contact Detection")
    rally_table.add_column("Rally ID", style="dim", max_width=12)
    rally_table.add_column("GT", justify="right")
    rally_table.add_column("Pred", justify="right")
    rally_table.add_column("TP", justify="right")
    rally_table.add_column("Recall", justify="right")
    rally_table.add_column("Precision", justify="right")
    rally_table.add_column("F1", justify="right")
    rally_table.add_column("Action Acc", justify="right")

    total_rallies = len(rallies)
    cumulative_tp = 0
    cumulative_fp = 0
    cumulative_fn = 0

    # Video file handles for visual attribution (reuse across rallies from same video)
    _video_caps: dict[str, Any] = {}  # video_id → cv2.VideoCapture
    _video_dims: dict[str, tuple[int, int, float]] = {}  # video_id → (w, h, fps)

    for rally_idx, rally in enumerate(rallies):
        t_start = time.monotonic()
        print(f"[{rally_idx + 1}/{total_rallies}] {rally.rally_id[:8]}...", end=" ", flush=True)

        # Get predicted actions — either from stored data or re-detect
        pred_actions: list[dict] = []

        if not args.stored and rally.ball_positions_json:
            # Re-run contact detection from ball/player positions
            from rallycut.tracking.ball_tracker import BallPosition as BallPos
            from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

            ball_positions = [
                BallPos(
                    frame_number=bp["frameNumber"],
                    x=bp["x"],
                    y=bp["y"],
                    confidence=bp.get("confidence", 1.0),
                )
                for bp in rally.ball_positions_json
                if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
            ]

            player_positions = (
                _build_player_positions(
                    rally.positions_json,
                    rally_id=rally.rally_id,
                    inject_pose=use_pose,
                )
                if rally.positions_json
                else []
            )

            match_teams = match_teams_by_rally.get(rally.rally_id)

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                config=contact_config,
                net_y=rally.court_split_y,
                frame_count=rally.frame_count or None,
                classifier=contact_classifier,
                use_classifier=not args.no_classifier,
                team_assignments=match_teams,
                court_calibrator=calibrators.get(rally.video_id),
            )

            # Build ReID predictions if classifier available for this video
            reid_preds: dict[int, dict[str, Any]] | None = None
            reid_cls = reid_classifiers.get(rally.video_id)
            if reid_cls is not None and rally.positions_json:
                t2p = track_to_player_maps.get(rally.rally_id, {})
                if t2p:
                    reid_preds = _compute_reid_predictions_for_rally(
                        reid_cls, rally.video_id, rally.start_ms, rally.fps,
                        contacts.contacts, rally.positions_json, t2p,
                    )

            # Prepare visual attribution context if classifier loaded
            vis_cap = None
            vis_positions = None
            vis_start_frame = 0
            vis_w = vis_h = 0
            if visual_classifier is not None and rally.positions_json:
                if rally.video_id not in _video_caps:
                    from rallycut.evaluation.tracking.db import get_video_path
                    vpath = get_video_path(rally.video_id)
                    if vpath is not None:
                        import cv2
                        cap = cv2.VideoCapture(str(vpath))
                        if cap.isOpened():
                            _video_caps[rally.video_id] = cap
                            _video_dims[rally.video_id] = (
                                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                cap.get(cv2.CAP_PROP_FPS) or 30.0,
                            )
                if rally.video_id in _video_caps:
                    vis_cap = _video_caps[rally.video_id]
                    w, h, fps = _video_dims[rally.video_id]
                    vis_w, vis_h = w, h
                    vis_positions = rally.positions_json
                    vis_start_frame = int((rally.start_ms or 0) / 1000.0 * fps)

            rally_actions = classify_rally_actions(
                contacts, rally.rally_id,
                use_classifier=not args.no_action_classifier,
                match_team_assignments=match_teams,
                reid_predictions=reid_preds,
                visual_classifier=visual_classifier if vis_cap else None,
                visual_video_cap=vis_cap,
                visual_positions_json=vis_positions,
                visual_rally_start_frame=vis_start_frame,
                visual_frame_w=vis_w,
                visual_frame_h=vis_h,
            )
            pred_actions = [a.to_dict() for a in rally_actions.actions]
        elif rally.actions_json:
            # Use stored actions
            pred_actions = rally.actions_json.get("actions", [])

        # Separate synthetic vs real predictions for metrics.
        real_pred_actions = [a for a in pred_actions if not a.get("isSynthetic")]
        synth_serves = [
            a for a in pred_actions
            if a.get("isSynthetic") and a.get("action") == "serve"
        ]
        # Includes synthetic serves — presence metric counts game-state inference
        has_pred_serve = any(a.get("action") == "serve" for a in pred_actions)
        has_gt_serve = any(gt.action == "serve" for gt in rally.gt_labels)

        if has_gt_serve:
            total_gt_serves += 1
            if has_pred_serve:
                serves_present += 1

        # FPS-adaptive tolerance: convert ms to frames for this rally
        tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))

        # Collect available track_ids for evaluable player attribution
        avail_tids: set[int] | None = None
        if rally.positions_json:
            avail_tids = {pp["trackId"] for pp in rally.positions_json}

        match_teams_for_cs = match_teams_by_rally.get(rally.rally_id)

        # Pass 1: match real predictions (original behavior)
        matches, unmatched = match_contacts(
            rally.gt_labels,
            real_pred_actions,
            tolerance=tolerance_frames,
            available_track_ids=avail_tids,
            team_assignments=match_teams_for_cs,
        )

        # Pass 2: match synthetic serves against unmatched GT serves.
        # Uses wider tolerance (~1s) since synthetic serve frame is
        # estimated from ball detection onset, not actual contact.
        # Only matches GT serves that real predictions couldn't reach.
        if synth_serves:
            synth_tolerance = max(tolerance_frames, round(rally.fps * 1.0))
            _match_synthetic_serves(
                matches, synth_serves, rally.gt_labels,
                synth_tolerance, avail_tids, match_teams_for_cs,
            )

        # Sanity checks on predicted action sequence
        pred_action_names = [a.get("action", "") for a in pred_actions]
        pred_frames = [a.get("frame", 0) for a in pred_actions]
        rally_violations = check_sanity(
            pred_action_names, pred_frames,
            rally_id=rally.rally_id, fps=rally.fps,
        )
        all_violations.extend(rally_violations)

        metrics = compute_metrics(matches, unmatched)

        rally_table.add_row(
            rally.rally_id[:8],
            str(metrics["total_gt"]),
            str(metrics["total_pred"]),
            str(metrics["tp"]),
            f"{metrics['recall']:.1%}",
            f"{metrics['precision']:.1%}",
            f"{metrics['f1']:.1%}",
            f"{metrics['action_accuracy']:.1%}",
        )

        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        cumulative_tp += metrics["tp"]
        cumulative_fp += metrics["fp"]
        cumulative_fn += metrics["fn"]
        elapsed = time.monotonic() - t_start
        cum_p = cumulative_tp / max(1, cumulative_tp + cumulative_fp)
        cum_r = cumulative_tp / max(1, cumulative_tp + cumulative_fn)
        cum_f1 = 2 * cum_p * cum_r / max(1e-9, cum_p + cum_r)
        print(
            f"F1={metrics['f1']:.1%} acc={metrics['action_accuracy']:.1%} "
            f"(cum F1={cum_f1:.1%} TP={cumulative_tp} FP={cumulative_fp} FN={cumulative_fn}) "
            f"({elapsed:.1f}s)"
        )


    console.print(rally_table)

    # Aggregate metrics
    if all_matches:
        agg_metrics = compute_metrics(all_matches, all_unmatched)

        console.print(f"\n[bold]Aggregate Results ({len(rallies)} rallies)[/bold]")
        console.print(f"  GT contacts:        {agg_metrics['total_gt']}")
        console.print(f"  Predicted contacts: {agg_metrics['total_pred']}")
        console.print(f"  True positives:     {agg_metrics['tp']}")
        console.print(f"  False positives:    {agg_metrics['fp']}")
        console.print(f"  False negatives:    {agg_metrics['fn']}")
        console.print(f"  [bold]Contact Recall:    {agg_metrics['recall']:.1%}[/bold]")
        console.print(f"  [bold]Contact Precision: {agg_metrics['precision']:.1%}[/bold]")
        console.print(f"  [bold]Contact F1:        {agg_metrics['f1']:.1%}[/bold]")
        console.print(f"  [bold]Action Accuracy:   {agg_metrics['action_accuracy']:.1%}[/bold]")
        console.print(f"  Player Attribution: {agg_metrics['player_accuracy']:.1%}")
        skipped = agg_metrics['player_evaluable_skipped']
        if skipped > 0:
            ev_acc = agg_metrics['player_evaluable_accuracy']
            ev_tot = agg_metrics['player_evaluable_total']
            ev_cor = agg_metrics['player_evaluable_correct']
            console.print(
                f"  Player Attr (eval): {ev_acc:.1%} "
                f"({ev_cor}/{ev_tot}, {skipped} skipped — stale GT track IDs)"
            )
        cs_tot = agg_metrics['court_side_total']
        if cs_tot > 0:
            cs_acc = agg_metrics['court_side_accuracy']
            cs_cor = agg_metrics['court_side_correct']
            console.print(
                f"  Court-Side Accuracy: {cs_acc:.1%} ({cs_cor}/{cs_tot})"
            )

        # Serve presence: does a serve action exist (real or synthetic)?
        if total_gt_serves > 0:
            serve_pct = serves_present / total_gt_serves
            console.print(
                f"  [bold]Serve Presence:    {serves_present}/{total_gt_serves} "
                f"({serve_pct:.1%})[/bold]  "
                f"[dim](includes synthetic)[/dim]"
            )

        # Per-class table
        class_table = Table(title="\nPer-Action Metrics")
        class_table.add_column("Action", style="bold")
        class_table.add_column("TP", justify="right")
        class_table.add_column("FP", justify="right")
        class_table.add_column("FN", justify="right")
        class_table.add_column("Precision", justify="right")
        class_table.add_column("Recall", justify="right")
        class_table.add_column("F1", justify="right")

        for action in ACTION_TYPES:
            c = agg_metrics["per_class"].get(action, {})
            if c.get("tp", 0) + c.get("fp", 0) + c.get("fn", 0) > 0:
                class_table.add_row(
                    action.capitalize(),
                    str(int(c.get("tp", 0))),
                    str(int(c.get("fp", 0))),
                    str(int(c.get("fn", 0))),
                    f"{c.get('precision', 0):.1%}",
                    f"{c.get('recall', 0):.1%}",
                    f"{c.get('f1', 0):.1%}",
                )

        console.print(class_table)

        # Confusion matrix
        conf_matrix = build_confusion_matrix(all_matches)
        if conf_matrix:
            all_labels = sorted(set(
                list(conf_matrix.keys()) +
                [pred for row in conf_matrix.values() for pred in row.keys()]
            ))

            cm_table = Table(title="\nConfusion Matrix (rows=GT, cols=Predicted)")
            cm_table.add_column("GT \\ Pred", style="bold")
            for label in all_labels:
                cm_table.add_column(label[:5], justify="right")

            for gt_label in all_labels:
                if gt_label in conf_matrix:
                    row_data = conf_matrix[gt_label]
                    cells = [str(row_data.get(pred, 0)) for pred in all_labels]
                    cm_table.add_row(gt_label[:8], *cells)

            console.print(cm_table)

        # Per-action player attribution breakdown
        matched_with_pred = [m for m in all_matches if m.pred_frame is not None]
        if matched_with_pred:
            attr_table = Table(title="\nPer-Action Player Attribution")
            attr_table.add_column("Action", style="bold")
            attr_table.add_column("Correct", justify="right")
            attr_table.add_column("Wrong", justify="right")
            attr_table.add_column("Total", justify="right")
            attr_table.add_column("Accuracy", justify="right")

            for action in ACTION_TYPES:
                action_matches = [
                    m for m in matched_with_pred
                    if m.gt_action == action and m.player_evaluable
                ]
                if not action_matches:
                    continue
                correct = sum(1 for m in action_matches if m.player_correct)
                total = len(action_matches)
                attr_table.add_row(
                    action.capitalize(),
                    str(correct),
                    str(total - correct),
                    str(total),
                    f"{correct / total:.1%}",
                )

            console.print(attr_table)

    # Sanity violation summary
    if all_violations:
        time_gaps = [v for v in all_violations if v.violation_type == "time_gap"]
        illegal_seq = [v for v in all_violations if v.violation_type == "same_action_repeat"]
        console.print("\n[bold]Sanity Violations[/bold]")
        console.print(f"  Time gaps (>3s):      {len(time_gaps)}")
        console.print(f"  Illegal sequences:    {len(illegal_seq)}")
        console.print(f"  Total violations:     {len(all_violations)}")
        rallies_with_violations = len({v.rally_id for v in all_violations})
        console.print(f"  Rallies with issues:  {rallies_with_violations}/{len(rallies)}")
        if len(all_violations) <= 20:
            for v in all_violations:
                console.print(f"    [{v.rally_id[:8]}] {v.violation_type}: {v.description}")
    else:
        console.print("\n[bold]Sanity Violations: 0[/bold] (all sequences clean)")

    # Clean up video file handles
    for cap in _video_caps.values():
        cap.release()


if __name__ == "__main__":
    main()
