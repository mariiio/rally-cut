"""Build a per-error corpus for action detection gap audit.

For each GT contact that is missed (FN) or misclassified (wrong action/player),
emits one JSONL record with full diagnostic context: FN subcategory, ball/player
quality signals, candidate features, and classifier confidence.

Also computes per-rally quality signals and joins them onto the corpus.

Outputs:
  outputs/action_errors/corpus.jsonl          — raw error records
  outputs/action_errors/rally_quality.json    — per-rally quality signals
  outputs/action_errors/corpus_annotated.jsonl — errors + quality signals joined

Usage:
    cd analysis
    uv run python scripts/build_action_error_corpus.py
    uv run python scripts/build_action_error_corpus.py --rally <id>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_classifier import load_contact_classifier
from rallycut.tracking.contact_detector import (
    _CONFIDENCE_THRESHOLD,
    detect_contacts,
)
from rallycut.tracking.sequence_action_runtime import get_sequence_probs
from scripts.diagnose_fn_contacts import diagnose_rally_fns
from scripts.eval_action_detection import (
    RallyData,
    _build_player_positions,
    _load_match_team_assignments,
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
    match_contacts,
)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "action_errors"
BALL_SEARCH_WINDOW = 5


@dataclass
class ErrorRecord:
    rally_id: str
    video_id: str
    fps: float
    start_ms: int
    gt_frame: int
    gt_action: str
    gt_player_track_id: int
    pred_frame: int | None
    pred_action: str | None
    pred_player_track_id: int | None
    error_class: str
    fn_subcategory: str | None
    classifier_conf: float
    nearest_cand_dist: int
    generators_fired: list[str]
    ball_present: bool
    ball_gap_frames: int
    ball_confidence: float
    velocity: float
    direction_change_deg: float
    player_distance: float
    player_present: bool


@dataclass
class RallyQuality:
    rally_id: str
    video_id: str
    frame_count: int
    fps: float
    ball_coverage_pct: float
    ball_max_gap_frames: int
    ball_mean_confidence: float
    ball_conf_p25: float
    player_track_count: int
    gt_contact_count: int
    # Number of predicted contacts that did not match any GT. This is the
    # corpus-level FP metric used by the Pattern-A rescue-gate sweep. Set by
    # `main()` after `match_contacts`; 0 for rallies without ball data.
    extra_predictions: int = 0


@dataclass
class ContactQuality:
    ball_present_at_frame: bool
    ball_gap_to_nearest: int
    player_bbox_present: bool
    player_track_continuity_8f: bool


def compute_rally_quality(rally: RallyData) -> RallyQuality:
    """Compute quality signals for a rally."""
    ball_positions = rally.ball_positions_json or []
    confident_balls = [
        bp for bp in ball_positions
        if bp.get("confidence", 1.0) >= _CONFIDENCE_THRESHOLD
        and (bp.get("x", 0) > 0 or bp.get("y", 0) > 0)
    ]

    frame_count = rally.frame_count or 1
    coverage = len(confident_balls) / max(1, frame_count)

    max_gap = 0
    if confident_balls:
        frames_sorted = sorted(bp["frameNumber"] for bp in confident_balls)
        for i in range(1, len(frames_sorted)):
            gap = frames_sorted[i] - frames_sorted[i - 1]
            max_gap = max(max_gap, gap)
    else:
        max_gap = frame_count

    confs = [bp.get("confidence", 1.0) for bp in confident_balls]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    p25_conf = float(np.percentile(confs, 25)) if confs else 0.0

    player_tracks: set[int] = set()
    if rally.positions_json:
        for pp in rally.positions_json:
            player_tracks.add(pp.get("trackId", 0))

    return RallyQuality(
        rally_id=rally.rally_id,
        video_id=rally.video_id,
        frame_count=frame_count,
        fps=rally.fps,
        ball_coverage_pct=round(coverage * 100, 1),
        ball_max_gap_frames=max_gap,
        ball_mean_confidence=round(mean_conf, 3),
        ball_conf_p25=round(p25_conf, 3),
        player_track_count=len(player_tracks),
        gt_contact_count=len(rally.gt_labels),
    )


def compute_contact_quality(
    rally: RallyData,
    gt_frame: int,
) -> ContactQuality:
    """Compute quality signals for a specific GT contact frame."""
    ball_positions = rally.ball_positions_json or []
    confident_frames = {
        bp["frameNumber"]
        for bp in ball_positions
        if bp.get("confidence", 1.0) >= _CONFIDENCE_THRESHOLD
        and (bp.get("x", 0) > 0 or bp.get("y", 0) > 0)
    }

    ball_present = gt_frame in confident_frames
    ball_gap = 0
    if not ball_present:
        if confident_frames:
            ball_gap = min(abs(f - gt_frame) for f in confident_frames)
        else:
            ball_gap = 9999

    player_at_frame = False
    continuity_8f = True
    if rally.positions_json:
        frames_with_player = {
            pp["frameNumber"]
            for pp in rally.positions_json
        }
        player_at_frame = gt_frame in frames_with_player
        for offset in range(-4, 5):
            if (gt_frame + offset) not in frames_with_player:
                continuity_8f = False
                break
    else:
        continuity_8f = False

    return ContactQuality(
        ball_present_at_frame=ball_present,
        ball_gap_to_nearest=ball_gap,
        player_bbox_present=player_at_frame,
        player_track_continuity_8f=continuity_8f,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build action detection error corpus")
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument(
        "--tolerance-ms", type=int, default=233,
        help="Frame matching tolerance in ms (default: 233, ~7 frames at 30fps)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading rallies with action GT...")
    rallies = load_rallies_with_action_gt(args.rally)
    print(f"  {len(rallies)} rallies loaded")

    if not rallies:
        print("No rallies with action GT found.")
        sys.exit(1)

    classifier = load_contact_classifier()
    if classifier:
        print(f"  Contact classifier loaded (threshold={classifier.threshold:.2f})")
    else:
        print("  No contact classifier — using hand-tuned gates")

    video_ids = {r.video_id for r in rallies}
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
    match_teams_by_rally = _load_match_team_assignments(
        video_ids, min_confidence=0.70, rally_positions=rally_pos_lookup,
    )
    track_to_player = _load_track_to_player_maps(video_ids)
    print(f"  Team assignments for {len(match_teams_by_rally)} rallies")

    corpus_path = OUTPUT_DIR / "corpus.jsonl"
    quality_path = OUTPUT_DIR / "rally_quality.json"
    annotated_path = OUTPUT_DIR / "corpus_annotated.jsonl"

    errors: list[dict] = []
    rally_qualities: dict[str, dict] = {}

    total_gt = 0
    total_tp = 0
    total_fn = 0
    total_wrong_action = 0
    total_wrong_player = 0
    total_extra_preds = 0

    with open(corpus_path, "w") as f_corpus:
        for idx, rally in enumerate(rallies):
            t0 = time.monotonic()
            rq = compute_rally_quality(rally)
            rally_qualities[rally.rally_id] = asdict(rq)

            if not rally.ball_positions_json:
                for gt in rally.gt_labels:
                    total_gt += 1
                    total_fn += 1
                    cq = compute_contact_quality(rally, gt.frame)
                    rec = {
                        "rally_id": rally.rally_id,
                        "video_id": rally.video_id,
                        "fps": rally.fps,
                        "start_ms": rally.start_ms,
                        "gt_frame": gt.frame,
                        "gt_action": gt.action,
                        "gt_player_track_id": gt.player_track_id,
                        "pred_frame": None,
                        "pred_action": None,
                        "pred_player_track_id": None,
                        "error_class": "FN_contact",
                        "fn_subcategory": "no_ball_data",
                        "classifier_conf": 0.0,
                        "nearest_cand_dist": 9999,
                        "generators_fired": [],
                        "ball_present": False,
                        "ball_gap_frames": 9999,
                        "ball_confidence": 0.0,
                        "velocity": 0.0,
                        "direction_change_deg": 0.0,
                        "player_distance": float("inf"),
                        "player_present": False,
                        **asdict(cq),
                    }
                    f_corpus.write(json.dumps(rec, default=str) + "\n")
                    errors.append(rec)
                elapsed = time.monotonic() - t0
                print(
                    f"[{idx+1}/{len(rallies)}] {rally.rally_id[:8]} "
                    f"NO_BALL_DATA ({len(rally.gt_labels)} FNs) [{elapsed:.1f}s]"
                )
                continue

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
                    inject_pose=True,
                )
                if rally.positions_json
                else []
            )

            match_teams = match_teams_by_rally.get(rally.rally_id)

            # Compute MS-TCN++ per-frame probs once per rally. Threading
            # this through both detect_contacts (for the two-signal rescue
            # gate at contact_detector.py:2135-2144) and classify_rally_actions
            # (for the internal apply_sequence_override) brings the corpus
            # evaluation in line with production CLI behavior. Returns None
            # if the sequence model weights are missing or the rally is too
            # short, in which case both callees degrade gracefully.
            sequence_probs = get_sequence_probs(
                ball_positions, player_positions, rally.court_split_y,
                rally.frame_count or 0, match_teams,
            )

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=rally.court_split_y,
                frame_count=rally.frame_count or None,
                classifier=classifier,
                team_assignments=match_teams,
                sequence_probs=sequence_probs,
            )

            rally_actions = classify_rally_actions(
                contacts, rally.rally_id,
                match_team_assignments=match_teams,
                sequence_probs=sequence_probs,
            )
            pred_actions = [a.to_dict() for a in rally_actions.actions]
            real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
            synth_serves = [
                a for a in pred_actions
                if a.get("isSynthetic") and a.get("action") == "serve"
            ]

            tolerance_frames = max(1, round(rally.fps * args.tolerance_ms / 1000))
            avail_tids: set[int] | None = None
            if rally.positions_json:
                avail_tids = {pp["trackId"] for pp in rally.positions_json}

            t2p = track_to_player.get(rally.rally_id)

            matches, unmatched_preds = match_contacts(
                rally.gt_labels,
                real_pred,
                tolerance=tolerance_frames,
                available_track_ids=avail_tids,
                team_assignments=match_teams,
                track_id_map=t2p,
            )

            from scripts.eval_action_detection import _match_synthetic_serves
            if synth_serves:
                synth_tol = max(tolerance_frames, round(rally.fps * 1.0))
                _match_synthetic_serves(
                    matches, synth_serves, rally.gt_labels,
                    synth_tol, avail_tids, match_teams, t2p,
                )

            # Predictions that did not match any GT are the FP count for this
            # rally. Record it on RallyQuality so sweep harnesses can compute
            # Δextra_pred per threshold cell.
            extra_preds_count = len(unmatched_preds)
            total_extra_preds += extra_preds_count
            rq.extra_predictions = extra_preds_count
            rally_qualities[rally.rally_id] = asdict(rq)

            fn_labels = [
                rally.gt_labels[i]
                for i, m in enumerate(matches)
                if m.pred_frame is None
            ]
            fn_diagnostics = diagnose_rally_fns(
                rally, fn_labels, classifier, tolerance_frames,
            )
            fn_diag_map = {d.gt_frame: d for d in fn_diagnostics}

            rally_fn = 0
            rally_tp = 0
            rally_wrong_action = 0
            rally_wrong_player = 0

            for i, m in enumerate(matches):
                total_gt += 1
                gt = rally.gt_labels[i]
                cq = compute_contact_quality(rally, m.gt_frame)

                if m.pred_frame is None:
                    total_fn += 1
                    rally_fn += 1
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
                        "generators_fired": diag.generators_fired if diag else [],
                        "ball_present": diag.ball_present if diag else False,
                        "ball_gap_frames": diag.ball_gap_frames if diag else 9999,
                        "ball_confidence": diag.ball_confidence if diag else 0.0,
                        "velocity": diag.velocity if diag else 0.0,
                        "direction_change_deg": diag.direction_change_deg if diag else 0.0,
                        "player_distance": diag.player_distance if diag else float("inf"),
                        "player_present": diag.player_present if diag else False,
                        **asdict(cq),
                    }
                    f_corpus.write(json.dumps(rec, default=str) + "\n")
                    errors.append(rec)

                elif m.pred_action != m.gt_action:
                    total_wrong_action += 1
                    rally_wrong_action += 1
                    pred = real_pred[0]
                    for p in real_pred:
                        if p.get("frame") == m.pred_frame:
                            pred = p
                            break

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
                        "pred_player_track_id": pred.get("playerTrackId"),
                        "error_class": "wrong_action",
                        "fn_subcategory": None,
                        "classifier_conf": pred.get("confidence", 0.0),
                        "nearest_cand_dist": abs(m.gt_frame - (m.pred_frame or 0)),
                        "generators_fired": [],
                        "ball_present": True,
                        "ball_gap_frames": 0,
                        "ball_confidence": 0.0,
                        "velocity": 0.0,
                        "direction_change_deg": 0.0,
                        "player_distance": 0.0,
                        "player_present": True,
                        **asdict(cq),
                    }
                    f_corpus.write(json.dumps(rec, default=str) + "\n")
                    errors.append(rec)

                elif m.player_evaluable and not m.player_correct:
                    total_wrong_player += 1
                    rally_wrong_player += 1
                    pred = real_pred[0]
                    for p in real_pred:
                        if p.get("frame") == m.pred_frame:
                            pred = p
                            break

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
                        "pred_player_track_id": pred.get("playerTrackId"),
                        "error_class": "wrong_player",
                        "fn_subcategory": None,
                        "classifier_conf": pred.get("confidence", 0.0),
                        "nearest_cand_dist": abs(m.gt_frame - (m.pred_frame or 0)),
                        "generators_fired": [],
                        "ball_present": True,
                        "ball_gap_frames": 0,
                        "ball_confidence": 0.0,
                        "velocity": 0.0,
                        "direction_change_deg": 0.0,
                        "player_distance": 0.0,
                        "player_present": True,
                        **asdict(cq),
                    }
                    f_corpus.write(json.dumps(rec, default=str) + "\n")
                    errors.append(rec)

                else:
                    total_tp += 1
                    rally_tp += 1

            elapsed = time.monotonic() - t0
            print(
                f"[{idx+1}/{len(rallies)}] {rally.rally_id[:8]} "
                f"GT={len(rally.gt_labels)} TP={rally_tp} FN={rally_fn} "
                f"wrong_action={rally_wrong_action} wrong_player={rally_wrong_player} "
                f"extra_pred={extra_preds_count} "
                f"ball_cov={rq.ball_coverage_pct:.0f}% [{elapsed:.1f}s]"
            )

    with open(quality_path, "w") as f:
        json.dump(rally_qualities, f, indent=2, default=str)

    with open(annotated_path, "w") as f_out:
        for rec in errors:
            rq_data = rally_qualities.get(rec["rally_id"], {})
            annotated = {**rec, "rally_quality": rq_data}
            f_out.write(json.dumps(annotated, default=str) + "\n")

    print(f"\n{'='*60}")
    print("CORPUS SUMMARY")
    print(f"{'='*60}")
    print(f"Total GT contacts:    {total_gt}")
    print(f"True positives:       {total_tp}")
    print(f"False negatives:      {total_fn}")
    print(f"Wrong action:         {total_wrong_action}")
    print(f"Wrong player:         {total_wrong_player}")
    print(f"Total errors:         {len(errors)}")
    print(f"Extra preds (FP):     {total_extra_preds}")
    print()

    fn_cats = Counter(r["fn_subcategory"] for r in errors if r["error_class"] == "FN_contact")
    if fn_cats:
        print("FN subcategories:")
        for cat, count in fn_cats.most_common():
            print(f"  {cat:30s} {count:4d}")

    action_errors = Counter(
        f"{r['gt_action']}→{r['pred_action']}"
        for r in errors if r["error_class"] == "wrong_action"
    )
    if action_errors:
        print("\nAction misclassifications:")
        for pair, count in action_errors.most_common():
            print(f"  {pair:30s} {count:4d}")

    print("\nOutputs:")
    print(f"  {corpus_path}")
    print(f"  {quality_path}")
    print(f"  {annotated_path}")


if __name__ == "__main__":
    main()
