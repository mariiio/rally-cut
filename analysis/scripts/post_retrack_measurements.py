"""Three post-retrack measurements (run after retrack_corrupted_rallies.py).

1. Post-retrack panel F1 — true post-v1.1 panel F1 with healthy data.
2. Fleet synth/real census — corrected synthetic-vs-real population breakdown.
3. Candidate-pool diagnostic for panel FNs — does detect_contacts produce a
   candidate within +-15 of each FN's GT frame? If yes, the FN is in-pool-but-
   filtered. If no, the FN is candidate-generator-miss. Tells us whether the
   '94% out_of_gap' finding from earlier in the session holds on healthy data.

Reads DB state directly. Doesn't re-run the pipeline (the retrack already
populated the DB with current-pipeline output).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.tracking.db import (
    get_connection,
    load_court_calibration,
)
from rallycut.training.action_gt_query import load_for_rallies

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15

PANEL_IDS = [
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0",
    "7d77980f-3006-40e0-adc0-db491a5bb659",
    "854bb250-3e91-47d2-944d-f62413e3cf45",
    "5c756c41-1cc1-4486-a95c-97398912cfbe",
    "073cb11b-c7ba-4fac-8cc9-b032b3152ad6",
]


def _match_actions(
    gt: list[dict[str, Any]], pred: list[dict[str, Any]]
) -> tuple[int, int, int, list[int]]:
    """Return (matched, fn, fp, fn_frames) by greedy frame-tolerance matching."""
    used: set[int] = set()
    matched = 0
    fn_frames: list[int] = []
    for g in gt:
        gf = int(g.get("frame", 0))
        best_i = -1
        best_d = HIT_TOLERANCE + 1
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(int(p.get("frame", 0)) - gf)
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d <= HIT_TOLERANCE:
            used.add(best_i)
            matched += 1
        else:
            fn_frames.append(gf)
    return matched, len(gt) - matched, len(pred) - matched, fn_frames


def _f1(matched: int, pred: int, gt: int) -> float:
    if not pred or not gt:
        return 0.0
    p = matched / pred
    r = matched / gt
    return 2 * p * r / (p + r) if (p + r) else 0.0


def measurement_1_panel_f1() -> None:
    print("=" * 80)
    print("MEASUREMENT 1: Post-retrack panel F1")
    print("=" * 80)

    with open(GT_PATH) as f:
        gt = json.load(f)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE id = ANY(%s)",
                [PANEL_IDS],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    hash_to_id: dict[str, str] = {
        cast(str, h): cast(str, vid) for vid, (h, _) in meta.items()
    }
    panel_gt = [
        (hash_to_id[r["video_content_hash"]], r)
        for r in gt["rallies"]
        if r["video_content_hash"] in hash_to_id
    ]

    totals = {"gt": 0, "pred": 0, "matched": 0, "fn": 0, "fp": 0}
    print(
        f"{'video':<6} {'rally':<10} {'#GT':>4} {'#Pred':>6} "
        f"{'matched':>8} {'FN':>4} {'FP':>4}"
    )
    print("-" * 60)

    with get_connection() as conn:
        for vid, gt_rally in panel_gt:
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.actions_json FROM rallies r
                       LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid = str(row[0])
            aj = cast(dict[str, Any] | None, row[1]) or {}
            pred_actions = list(aj.get("actions") or [])
            gt_actions = load_for_rallies(conn, [rid]).get(rid, [])
            matched, fn, fp, _ = _match_actions(gt_actions, pred_actions)
            print(
                f"{name:<6} {rid[:8]:<10} {len(gt_actions):>4} "
                f"{len(pred_actions):>6} {matched:>8} {fn:>4} {fp:>4}"
            )
            totals["gt"] += len(gt_actions)
            totals["pred"] += len(pred_actions)
            totals["matched"] += matched
            totals["fn"] += fn
            totals["fp"] += fp

    print("-" * 60)
    print(
        f"TOTAL: GT={totals['gt']}  Pred={totals['pred']}  "
        f"matched={totals['matched']}  FN={totals['fn']}  FP={totals['fp']}  "
        f"F1={_f1(totals['matched'], totals['pred'], totals['gt']):.3f}"
    )
    print()
    print("Reference baselines from earlier in session:")
    print("  Pre-GT-fix:           F1=0.862  35 FN  19 FP  (GT included drift on ruru:3655eb69)")
    print("  Post-GT-fix:          F1=0.898  28 FN  12 FP  (after fixing ruru entry)")
    print("  Post-deploy + drift:  F1=??     ??     ??    (corrupted rallies were 0-action)")
    print("  Post-retrack:         F1={:.3f}  {} FN  {} FP  (CURRENT)".format(
        _f1(totals["matched"], totals["pred"], totals["gt"]),
        totals["fn"], totals["fp"],
    ))


def measurement_2_fleet_census() -> None:
    print()
    print("=" * 80)
    print("MEASUREMENT 2: Fleet synth-vs-real serve census (post-retrack)")
    print("=" * 80)

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
    hash_to_id: dict[str, str] = {
        cast(str, h): cast(str, vid) for vid, (h, _) in meta.items()
    }

    n_synth = n_real = n_no_pred = 0
    n_synth_hit = n_real_hit = 0
    with get_connection() as conn:
        cur = conn.cursor()
        for r in gt["rallies"]:
            if r["video_content_hash"] not in hash_to_id:
                continue
            vid = hash_to_id[r["video_content_hash"]]
            cur.execute(
                """SELECT r.id, pt.actions_json FROM rallies r
                   LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s AND r.start_ms = %s""",
                [vid, r["rally_start_ms"]],
            )
            row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid2 = str(row[0])
            aj2 = cast(dict[str, Any] | None, row[1]) or {}
            actions = aj2.get("actions") or []
            pred_serve = next((a for a in actions if a.get("action") == "serve"), None)
            gt_labels2 = load_for_rallies(conn, [rid2]).get(rid2, [])
            gt_serve_f = next(
                (a["frame"] for a in gt_labels2 if a.get("action") == "serve"),
                None,
            )
            if gt_serve_f is None:
                continue
            if pred_serve is None:
                n_no_pred += 1
                continue
            is_synth = pred_serve.get("isSynthetic", False)
            diff = abs(pred_serve.get("frame", -999) - gt_serve_f)
            hit = diff <= HIT_TOLERANCE
            if is_synth:
                n_synth += 1
                if hit:
                    n_synth_hit += 1
            else:
                n_real += 1
                if hit:
                    n_real_hit += 1

    total_with_pred = n_synth + n_real
    total_with_gt = total_with_pred + n_no_pred
    total_hit = n_synth_hit + n_real_hit

    print(f"Fleet rallies with GT serve: {total_with_gt}")
    print(f"  with pred serve: {total_with_pred} "
          f"(synthetic: {n_synth}, real: {n_real})")
    print(f"  no pred serve:   {n_no_pred}")
    print()
    print(f"Hit rate (within +-{HIT_TOLERANCE} frames of GT):")
    if n_synth:
        print(f"  Synthetic: {n_synth_hit}/{n_synth} ({n_synth_hit/n_synth*100:.0f}%)")
    if n_real:
        print(f"  Real:      {n_real_hit}/{n_real} ({n_real_hit/n_real*100:.0f}%)")
    print(f"  Total:     {total_hit}/{total_with_pred} hit (vs {total_with_gt} GT-labeled)")
    print()
    print("Reference baselines from earlier in session:")
    print("  Pre-deploy stored:  44 synth + 194 real = 238 hit / 313 total / 22 no-pred")
    print("  Post-deploy stored: 37 synth + 213 real = 250 hit / 290 total / 45 no-pred")
    print(f"  Post-retrack:       {n_synth} synth + {n_real} real = {total_hit} hit / "
          f"{total_with_pred} total / {n_no_pred} no-pred")


def measurement_3_panel_fn_candidate_pool() -> None:
    print()
    print("=" * 80)
    print("MEASUREMENT 3: Panel FN candidate-pool diagnostic")
    print("(For each panel FN, does detect_contacts produce a candidate within +-15 of GT?)")
    print("=" * 80)

    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition as BallPos
    from rallycut.tracking.contact_detector import detect_contacts
    from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs

    with open(GT_PATH) as f:
        gt = json.load(f)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos WHERE id = ANY(%s)",
                [PANEL_IDS],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id: dict[str, str] = {
        cast(str, h): cast(str, vid) for vid, (h, _) in meta.items()
    }
    panel_gt = [
        (hash_to_id[r["video_content_hash"]], r)
        for r in gt["rallies"]
        if r["video_content_hash"] in hash_to_id
    ]

    n_fns = n_candidate_exists = n_no_candidate = 0
    print(f"{'video':<6} {'rally':<10} {'fn_frame':>9} {'candidate?':<11} {'nearest':<10}")
    print("-" * 55)

    cache_calibrators: dict[str, CourtCalibrator | None] = {}

    with get_connection() as conn:
        for vid, gt_rally in panel_gt:
            name = meta[vid][1]
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.ball_positions_json, pt.positions_json,
                              pt.frame_count, pt.court_split_y, pt.actions_json
                       FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, gt_rally["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid = str(row[0])
            bp_json = cast(list[dict[str, Any]] | None, row[1])
            pp_json = cast(list[dict[str, Any]] | None, row[2])
            fc = cast(int | None, row[3])
            csy = cast(float | None, row[4])
            aj = cast(dict[str, Any] | None, row[5])
            pred_actions = list((aj or {}).get("actions") or [])
            gt_actions = load_for_rallies(conn, [rid]).get(rid, [])

            _, fn, _, fn_frames = _match_actions(gt_actions, pred_actions)
            if not fn_frames:
                continue

            if vid not in cache_calibrators:
                corners = load_court_calibration(vid)
                cal = None
                if corners and len(corners) == 4:
                    cal = CourtCalibrator()
                    cal.calibrate([(c["x"], c["y"]) for c in corners])
                cache_calibrators[vid] = cal
            cal = cache_calibrators[vid]

            ball_positions = [
                BallPos(
                    frame_number=int(b.get("frameNumber", 0)),
                    x=float(b.get("x", 0.0)), y=float(b.get("y", 0.0)),
                    confidence=float(b.get("confidence", 1.0)),
                )
                for b in (bp_json or [])
                if isinstance(b, dict) and (b.get("x", 0) > 0 or b.get("y", 0) > 0)
            ]
            player_positions = [
                PlayerPos(
                    frame_number=int(p.get("frameNumber", 0)),
                    track_id=int(p.get("trackId", -1)),
                    x=float(p.get("x", 0.0)), y=float(p.get("y", 0.0)),
                    width=float(p.get("width", 0.0)), height=float(p.get("height", 0.0)),
                    confidence=float(p.get("confidence", 1.0)),
                )
                for p in (pp_json or [])
                if isinstance(p, dict)
            ]
            seq_probs = get_sequence_probs(
                ball_positions, player_positions, csy, fc or 0, None,
                calibrator=cal,
            )
            contact_seq = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
frame_count=fc or None,
                court_calibrator=cal, sequence_probs=seq_probs,
            )

            for fn_frame in fn_frames:
                n_fns += 1
                # Find nearest candidate frame
                near = None
                for c in contact_seq.contacts:
                    if abs(c.frame - fn_frame) <= HIT_TOLERANCE:
                        if near is None or abs(c.frame - fn_frame) < abs(near.frame - fn_frame):
                            near = c
                if near is not None:
                    n_candidate_exists += 1
                    print(
                        f"{name:<6} {rid[:8]:<10} {fn_frame:>9} "
                        f"{'EXISTS':<11} {f'f{near.frame}@{near.confidence:.2f}':<10}"
                    )
                else:
                    n_no_candidate += 1
                    print(
                        f"{name:<6} {rid[:8]:<10} {fn_frame:>9} "
                        f"{'NO_CAND':<11} {'-':<10}"
                    )

    print("-" * 55)
    print(f"Total panel FNs: {n_fns}")
    if n_fns:
        print(f"  Candidate exists (was rejected/filtered): {n_candidate_exists} "
              f"({n_candidate_exists/n_fns*100:.0f}%)")
        print(f"  No candidate generated:                    {n_no_candidate} "
              f"({n_no_candidate/n_fns*100:.0f}%)")
    print()
    print("Reference: earlier diagnostic (pre-retrack, possibly contaminated by")
    print("corrupted rallies) reported 94% out_of_gap (0 candidates near FN frames).")
    print("If post-retrack 'no_candidate' rate is much lower, that finding was")
    print("inflated by the corruption — recoverable signal exists on healthy data.")


def main() -> None:
    measurement_1_panel_f1()
    measurement_2_fleet_census()
    measurement_3_panel_fn_candidate_pool()


if __name__ == "__main__":
    main()
