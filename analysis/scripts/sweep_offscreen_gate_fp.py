"""Static FP-sweep — apply the proposed v1.3 prepend gate to every GT rally
and count how many fire, how many land within HIT_TOLERANCE of the GT serve,
and how many would be false positives.

The gate (corrected after visual validation):

  first_classified_action.type == "serve"
  AND there exists frame f in [rally_start, first_action.frame - 15] with:
        seq[serve_class][f] >= PEAK_THRESHOLD
        first_action.frame - f >= MIN_GAP
  AND seq[serve_class][first_action.frame] < FIRST_ACTION_SERVE_CEIL

When fires:
  TP : abs(peak_frame - gt_serve_frame) <= HIT_TOLERANCE
  FP : otherwise

We also probe alternate (PEAK_THRESHOLD, MIN_GAP) pairs to view the Pareto
frontier.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.evaluation.tracking.db import get_connection
from rallycut.training.action_gt_query import load_for_rallies
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15

# Candidate parameters to sweep
PEAK_THRESHOLDS = [0.85, 0.90, 0.95, 0.97]
MIN_GAPS = [25, 35, 50]
FIRST_ACTION_SERVE_CEIL = 0.50  # if first-action's serve-prob >= this, skip
GUARD_FRAMES = 15


def _bp_from_json(bp_json: Any) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=int(b.get("frameNumber", 0)),
            x=float(b.get("x", 0.0)),
            y=float(b.get("y", 0.0)),
            confidence=float(b.get("confidence", 0.0)),
        )
        for b in (bp_json or [])
        if isinstance(b, dict)
    ]


def _pp_from_json(pp_json: Any) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=int(p.get("frameNumber", 0)),
            track_id=int(p.get("trackId", -1)),
            x=float(p.get("x", 0.0)),
            y=float(p.get("y", 0.0)),
            width=float(p.get("width", 0.0)),
            height=float(p.get("height", 0.0)),
            confidence=float(p.get("confidence", 0.0)),
            keypoints=p.get("keypoints"),
        )
        for p in (pp_json or [])
        if isinstance(p, dict)
    ]


def _select_peak(
    seq: np.ndarray,
    upper_excl: int,
    floor: float,
) -> tuple[int, float]:
    """Return (argmax_frame, peak_prob) for serve-class in [0, upper_excl)."""
    if seq is None or upper_excl <= 0:
        return (-1, 0.0)
    serve_idx = ACTION_TYPES.index("serve") + 1
    row = seq[serve_idx, :upper_excl]
    if row.size == 0:
        return (-1, 0.0)
    peak = int(np.argmax(row))
    return (peak, float(row[peak]))


def _evaluate_gate(
    *,
    peak_threshold: float,
    min_gap: int,
    rallies: list[dict[str, Any]],
) -> dict[str, Any]:
    fires_tp = 0
    fires_fp = 0
    fp_details: list[dict[str, Any]] = []
    for rally in rallies:
        first_action_frame = rally["first_action_frame"]
        first_action_type = rally["first_action_type"]
        seq_at_first_action = rally["seq_at_first_action"]
        peak_f = rally["peak_f"]
        peak_p = rally["peak_p"]
        gt_serve_f = rally["gt_serve_f"]
        if first_action_type != "serve":
            continue
        if peak_f < 0 or peak_p < peak_threshold:
            continue
        if first_action_frame - peak_f < min_gap:
            continue
        if seq_at_first_action >= FIRST_ACTION_SERVE_CEIL:
            continue
        # gate fires
        if abs(peak_f - gt_serve_f) <= HIT_TOLERANCE:
            fires_tp += 1
        else:
            fires_fp += 1
            fp_details.append({
                "video": rally["video"],
                "rally_id": rally["rally_id"],
                "peak_f": peak_f,
                "gt_serve_f": gt_serve_f,
                "peak_minus_gt": peak_f - gt_serve_f,
                "peak_p": peak_p,
                "first_action_f": first_action_frame,
            })
    return {
        "peak_threshold": peak_threshold,
        "min_gap": min_gap,
        "tp": fires_tp,
        "fp": fires_fp,
        "fp_details": fp_details,
    }


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)
    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    print(f"Loading {len(gt['rallies'])} rallies + scoring serve-class signals...")
    rallies: list[dict[str, Any]] = []
    serve_idx = ACTION_TYPES.index("serve") + 1

    with get_connection() as conn:
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.fps, pt.frame_count, pt.court_split_y,
                              pt.ball_positions_json, pt.positions_json,
                              pt.actions_json, pt.primary_track_ids
                       FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, _fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row
            rid_str = str(rid)
            gt_labels = load_for_rallies(conn, [rid_str]).get(rid_str, [])

            gt_serve_f = next(
                (a["frame"] for a in gt_labels
                 if a.get("action") == "serve"),
                None,
            )
            if gt_serve_f is None:
                continue
            gt_serve_f = int(gt_serve_f)

            bp = _bp_from_json(bp_json)
            pp = _pp_from_json(pp_json)
            if not bp or not pp:
                continue

            ta_str = (aj or {}).get("teamAssignments", {}) or {}
            ta_int = {
                int(k): (0 if v == "A" else 1)
                for k, v in ta_str.items()
                if v in ("A", "B")
            }
            seq = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=None,
            )
            if seq is None:
                continue

            # Run production pipeline to get first classified action
            contact_seq = detect_contacts(
                ball_positions=bp,
                player_positions=pp,
                config=ContactDetectionConfig(),
                frame_count=fcount or None,
                team_assignments=ta_int,
                sequence_probs=seq,
                primary_track_ids=list(primary_raw or []) or None,
            )
            ra = classify_rally_actions(
                contact_seq,
                team_assignments=ta_int,
                sequence_probs=seq,
            )
            if not ra.actions:
                continue
            first_action = min(ra.actions, key=lambda a: a.frame)
            first_action_frame = first_action.frame
            first_action_type = first_action.action_type.value

            # MS-TCN++ peak in [rally_start, first_action_frame - GUARD_FRAMES]
            upper = max(first_action_frame - GUARD_FRAMES, 0)
            peak_f, peak_p = _select_peak(seq, upper, floor=0.0)

            # seq serve-prob AT the first detected contact frame
            if 0 <= first_action_frame < seq.shape[1]:
                seq_at_first_action = float(seq[serve_idx, first_action_frame])
            else:
                seq_at_first_action = 0.0

            rallies.append({
                "video": name,
                "rally_id": rid,
                "gt_serve_f": gt_serve_f,
                "first_action_frame": first_action_frame,
                "first_action_type": first_action_type,
                "seq_at_first_action": seq_at_first_action,
                "peak_f": peak_f,
                "peak_p": peak_p,
            })

    print(f"Scored {len(rallies)} rallies.\n")

    # Header for sweep
    print(f"{'peak_p':>7} {'min_gap':>8} {'TP':>4} {'FP':>4}  notes")
    print("-" * 60)
    all_results: list[dict[str, Any]] = []
    for pt in PEAK_THRESHOLDS:
        for mg in MIN_GAPS:
            res = _evaluate_gate(
                peak_threshold=pt, min_gap=mg, rallies=rallies,
            )
            all_results.append(res)
            note = ""
            if res["fp"] == 0 and res["tp"] >= 18:
                note = "*** clean ***"
            print(f"{pt:>7.2f} {mg:>8} {res['tp']:>4} {res['fp']:>4}  {note}")

    # Print FP details for the recommended gate
    rec = next(
        r for r in all_results
        if r["peak_threshold"] == 0.95 and r["min_gap"] == 35
    )
    print()
    print(f"=== Details for (peak_p>=0.95, min_gap>=35) ===")
    print(f"TP={rec['tp']}, FP={rec['fp']}")
    if rec["fp_details"]:
        print("FPs:")
        for fp in rec["fp_details"]:
            print(f"  {fp['video']:<8} {fp['rally_id'][:8]}  "
                  f"peak_f={fp['peak_f']} gt_serve_f={fp['gt_serve_f']} "
                  f"delta={fp['peak_minus_gt']:+d} peak_p={fp['peak_p']:.3f}")

    # Also list which rallies WOULD fire at the recommended gate
    print()
    print(f"=== Rallies that would fire (peak_p>=0.95, min_gap>=35) ===")
    n_fires = 0
    for rally in rallies:
        if rally["first_action_type"] != "serve":
            continue
        if rally["peak_p"] < 0.95:
            continue
        if rally["first_action_frame"] - rally["peak_f"] < 35:
            continue
        if rally["seq_at_first_action"] >= FIRST_ACTION_SERVE_CEIL:
            continue
        is_tp = abs(rally["peak_f"] - rally["gt_serve_f"]) <= HIT_TOLERANCE
        verdict = "TP" if is_tp else "FP"
        n_fires += 1
        print(
            f"  {rally['video']:<8} {rally['rally_id'][:8]}  "
            f"peak_f={rally['peak_f']:>4} gt_serve_f={rally['gt_serve_f']:>4} "
            f"delta={rally['peak_f'] - rally['gt_serve_f']:+5d} "
            f"peak_p={rally['peak_p']:.2f}  first_pred={rally['first_action_frame']:>4}  "
            f"[{verdict}]"
        )
    print(f"\nTotal fires: {n_fires}")

    out_path = Path("reports/offscreen_gate_sweep.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "rallies": rallies,
            "sweep_results": [
                {k: v for k, v in r.items() if k != "fp_details"}
                for r in all_results
            ],
        }, f, indent=2, default=str)
    print(f"\nWrote sweep results to {out_path}")


if __name__ == "__main__":
    main()
