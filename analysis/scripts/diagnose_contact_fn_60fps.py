"""Phase 1 diagnosis: classify contact FN by failure mode on 60fps vs 30fps.

For each GT action frame across kuku/lulu/wawa (60fps with GT) and
titi/toto/jaja (30fps comparison), check whether the v4 pipeline produced
a contact within ±10 frames of the GT frame. For each FN, classify the
failure mode:

  C  — ball had no confident detection at GT frame ± 2 (upstream FN)
  D  — no player within 0.15 of ball at GT frame ± 2 (attribution FN)
  A  — ball present, player near, _prepare_candidates DID generate a
       candidate within ±5f, but it was filtered out (GBM rejected it
       or dedup/refinement dropped it). FIX = GBM retrain.
  B  — ball present, player near, but _prepare_candidates did NOT
       generate any candidate within ±5f. The gates filtered too
       aggressively. FIX = scale the culprit gate.

Output: per-video and per-cohort tables of FN rate + classification.

Usage:
    cd analysis
    uv run python scripts/diagnose_contact_fn_60fps.py                  # DB v4 contacts
    uv run python scripts/diagnose_contact_fn_60fps.py --in-memory      # Run detect_contacts
                                                                        # inline with current
                                                                        # default classifier
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from typing import Any, cast

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _prepare_candidates,
    detect_contacts,
)
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

MATCH_WINDOW = 10  # GT contact considered "detected" if v4 contact within ±N frames
CANDIDATE_WINDOW = 5  # AB split: candidate considered "at GT" if within ±N
SUPPORT_WINDOW = 2  # FN diagnosis searches ±N frames for ball/player presence
PLAYER_CONTACT_RADIUS = 0.15  # normalized image units; same as ContactDetectionConfig default
BALL_CONFIDENCE_THRESHOLD = 0.3  # mirrors _CONFIDENCE_THRESHOLD in contact_detector

COHORTS = {
    # All 60fps videos in the GT corpus (19 videos, ~727 GT actions). caca has
    # only 4 GT and is excluded for noise reduction.
    "60fps": [
        "haha", "kaka", "kiki", "kuku", "lulu", "matchc", "ruru", "vivi",
        "vovo", "vuvu", "wawa", "wewe", "wiwi", "wowo", "wuwu", "yaya",
        "yeye", "yiyi", "yoyo",
    ],
    "30fps": ["titi", "toto", "jaja"],
}


def _load_rally_data(cur: Any, video_names: list[str]) -> dict[str, dict[str, Any]]:
    """Load contacts, ball, players, fps + court_split_y + frame_count per rally."""
    placeholders = ",".join(["%s"] * len(video_names))
    cur.execute(
        f"""
        SELECT r.id, r.video_id, v.name,
               pt.contacts_json, pt.ball_positions_json, pt.positions_json,
               COALESCE(pt.fps, v.fps) AS resolved_fps,
               pt.court_split_y, pt.frame_count
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON r.video_id = v.id
        WHERE v.name IN ({placeholders})
        """,
        video_names,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in cur:
        rid = str(row[0])
        out[rid] = {
            "video_id": str(row[1]),
            "video": row[2],
            "contacts_json": row[3],
            "ball_json": row[4],
            "positions_json": row[5],
            "fps": float(row[6]) if row[6] else 30.0,
            "court_split_y": row[7],
            "frame_count": row[8],
        }
    return out


def _load_match_teams(
    cur: Any, video_names: list[str], rally_data: dict[str, dict[str, Any]],
) -> dict[str, dict[int, int]]:
    """Load per-rally team assignments via build_match_team_assignments."""
    placeholders = ",".join(["%s"] * len(video_names))
    # Pre-load positions for team verification
    cur.execute(
        f"""
        SELECT pt.rally_id, pt.positions_json
        FROM player_tracks pt
        JOIN rallies r ON r.id = pt.rally_id
        JOIN videos v ON r.video_id = v.id
        WHERE v.name IN ({placeholders}) AND pt.positions_json IS NOT NULL
        """,
        video_names,
    )
    rally_positions: dict[str, list[PlayerPosition]] = {}
    for rid_raw, pos_raw in cur.fetchall():
        rid_str = str(rid_raw)
        pos_list = pos_raw if isinstance(pos_raw, list) else []
        rally_positions[rid_str] = [
            PlayerPosition(
                frame_number=p.get("frameNumber", 0),
                track_id=p.get("trackId", 0),
                x=p.get("x", 0), y=p.get("y", 0),
                width=p.get("width", 0), height=p.get("height", 0),
                confidence=p.get("confidence", 0),
            )
            for p in pos_list
            if isinstance(p, dict)
        ]

    cur.execute(
        f"SELECT v.id, v.match_analysis_json FROM videos v "
        f"WHERE v.name IN ({placeholders}) AND v.match_analysis_json IS NOT NULL",
        video_names,
    )
    out: dict[str, dict[int, int]] = {}
    for _vid, mj_raw in cur.fetchall():
        mj = cast(dict[str, Any], mj_raw)
        if not mj:
            continue
        out.update(
            build_match_team_assignments(
                mj, min_confidence=0.0, rally_positions=rally_positions,
            )
        )
    return out


def _load_gt(cur: Any, video_names: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Load GT actions per rally."""
    placeholders = ",".join(["%s"] * len(video_names))
    cur.execute(
        f"""
        SELECT r.id, gt.frame, gt.action, gt.snapshot_ball_x, gt.snapshot_ball_y,
               gt.resolved_track_id
        FROM rally_action_ground_truth gt
        JOIN rallies r ON gt.rally_id = r.id
        JOIN videos v ON r.video_id = v.id
        WHERE v.name IN ({placeholders})
        ORDER BY r.id, gt.frame
        """,
        video_names,
    )
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cur:
        rid = str(row[0])
        out[rid].append({
            "frame": row[1],
            "action": row[2],
            "ball_x": row[3],
            "ball_y": row[4],
            "track_id": row[5],
        })
    return out


def _parse_contacts(contacts_json: Any) -> list[dict[str, Any]]:
    if contacts_json is None:
        return []
    if isinstance(contacts_json, str):
        return cast(list[dict[str, Any]], json.loads(contacts_json).get("contacts", []))
    return cast(list[dict[str, Any]], cast(dict[str, Any], contacts_json).get("contacts", []))


def _ball_present(
    ball_positions: list[dict[str, Any]] | None, gt_frame: int,
) -> tuple[bool, float | None, float | None]:
    """Check if a confident ball detection exists within ±SUPPORT_WINDOW of gt_frame.

    Returns (present, x_at_nearest, y_at_nearest).
    """
    if not ball_positions:
        return False, None, None
    best_dist = math.inf
    best_x: float | None = None
    best_y: float | None = None
    for bp in ball_positions:
        f = bp.get("frameNumber")
        if f is None:
            continue
        d = abs(f - gt_frame)
        if d > SUPPORT_WINDOW:
            continue
        if bp.get("confidence", 0.0) < BALL_CONFIDENCE_THRESHOLD:
            continue
        if d < best_dist:
            best_dist = d
            best_x = bp.get("x")
            best_y = bp.get("y")
    return best_x is not None, best_x, best_y


def _nearest_player_dist(
    positions: list[dict[str, Any]] | None,
    gt_frame: int,
    ball_x: float,
    ball_y: float,
) -> float:
    """Min Euclidean distance from ball to any player bbox center within ±SUPPORT_WINDOW."""
    if not positions:
        return math.inf
    best = math.inf
    for p in positions:
        f = p.get("frameNumber")
        if f is None or abs(f - gt_frame) > SUPPORT_WINDOW:
            continue
        px = p.get("x")
        py = p.get("y")
        if px is None or py is None:
            continue
        d = math.hypot(px - ball_x, py - ball_y)
        if d < best:
            best = d
    return best


def _contact_matched(contacts: list[dict[str, Any]], gt_frame: int) -> bool:
    for c in contacts:
        if abs(c.get("frame", -10000) - gt_frame) <= MATCH_WINDOW:
            return True
    return False


def _build_ball_positions(ball_json: list[dict[str, Any]] | None) -> list[BallPosition]:
    if not ball_json:
        return []
    return [
        BallPosition(
            frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in ball_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _build_player_positions(
    positions_json: list[dict[str, Any]] | None,
) -> list[PlayerPosition]:
    if not positions_json:
        return []
    return [
        PlayerPosition(
            frame_number=p["frameNumber"], track_id=p["trackId"],
            x=p["x"], y=p["y"], width=p["width"], height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in positions_json
    ]


def _candidate_at_gt(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    gt_frame: int,
) -> tuple[bool, list[str]]:
    """Run _prepare_candidates and check whether any candidate fires near gt_frame.

    Returns (any_candidate_in_window, list_of_generators_that_fired).
    """
    cfg = ContactDetectionConfig()
    try:
        prep = _prepare_candidates(ball_positions, player_positions, cfg)
    except Exception:
        return False, []

    any_near = any(abs(f - gt_frame) <= CANDIDATE_WINDOW for f in prep.candidate_frames)
    if not any_near:
        return False, []

    fired: list[str] = []
    for label, frames in (
        ("velocity", prep.velocity_peak_frames),
        ("inflection", prep.inflection_frames),
        ("decel", prep.deceleration_frames),
        ("parabolic", prep.parabolic_frames),
        ("dirchange", prep.direction_change_frames),
        ("netcross", prep.net_crossing_frames),
    ):
        if any(abs(f - gt_frame) <= CANDIDATE_WINDOW for f in frames):
            fired.append(label)
    return True, fired


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-memory", action="store_true",
        help="Run detect_contacts inline using the current default classifier "
        "instead of reading contacts_json from DB. Used to A/B a freshly "
        "trained classifier without --apply DB writes.",
    )
    args = ap.parse_args()

    with get_connection() as conn:
        with conn.cursor() as cur:
            all_videos = COHORTS["60fps"] + COHORTS["30fps"]
            rally_data = _load_rally_data(cur, all_videos)
            gt_by_rally = _load_gt(cur, all_videos)
            match_teams_by_rally: dict[str, dict[int, int]] = {}
            calibrators: dict[str, Any] = {}
            if args.in_memory:
                match_teams_by_rally = _load_match_teams(cur, all_videos, rally_data)
                seen_vids = {d["video_id"] for d in rally_data.values()}
                for vid in seen_vids:
                    corners = load_court_calibration(vid)
                    if corners and len(corners) == 4:
                        cal = CourtCalibrator()
                        cal.calibrate([(c["x"], c["y"]) for c in corners])
                        calibrators[vid] = cal
                    else:
                        calibrators[vid] = None

    mode = "in-memory (current default classifier)" if args.in_memory else "DB v4 contacts"
    print(f"# Mode: {mode}")
    print()

    per_video: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    generator_fires: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for rid, gt_list in gt_by_rally.items():
        if rid not in rally_data:
            continue
        data = rally_data[rid]
        video = data["video"]
        ball_json = cast(list[dict[str, Any]] | None, data["ball_json"])
        positions_json = cast(list[dict[str, Any]] | None, data["positions_json"])
        # Build typed objects once per rally (reused for each GT row)
        ball_positions = _build_ball_positions(ball_json)
        player_positions = _build_player_positions(positions_json)

        if args.in_memory:
            # Re-run detect_contacts with current default classifier
            try:
                seq_probs = get_sequence_probs(
                    ball_positions, player_positions, data["court_split_y"],
                    data["frame_count"] or 0,
                    match_teams_by_rally.get(rid),
                    calibrator=calibrators.get(data["video_id"]),
                )
                cs = detect_contacts(
                    ball_positions=ball_positions,
                    player_positions=player_positions,
                    frame_count=data["frame_count"] or None,
                    court_calibrator=calibrators.get(data["video_id"]),
                    team_assignments=match_teams_by_rally.get(rid),
                    sequence_probs=seq_probs,
                )
                contacts = [
                    {"frame": c.frame} for c in cs.contacts
                ]
            except Exception as e:
                print(f"WARN rally {rid[:8]} in-memory detect failed: {e}", file=sys.stderr)
                contacts = []
        else:
            contacts = _parse_contacts(data["contacts_json"])

        for gt in gt_list:
            gt_frame = gt["frame"]
            per_video[video]["gt_total"] += 1

            if _contact_matched(contacts, gt_frame):
                per_video[video]["matched"] += 1
                continue

            # FN — classify
            per_video[video]["fn_total"] += 1
            ball_ok, bx, by = _ball_present(ball_json, gt_frame)
            if not ball_ok:
                per_video[video]["fn_class_C"] += 1
                continue
            search_x = bx if bx is not None else gt.get("ball_x")
            search_y = by if by is not None else gt.get("ball_y")
            if search_x is None or search_y is None:
                per_video[video]["fn_class_C"] += 1
                continue
            d = _nearest_player_dist(positions_json, gt_frame, search_x, search_y)
            if d > PLAYER_CONTACT_RADIUS:
                per_video[video]["fn_class_D"] += 1
                continue

            # AB split: did _prepare_candidates fire at the GT frame?
            had_candidate, gens = _candidate_at_gt(
                ball_positions, player_positions, gt_frame,
            )
            cohort = "60fps" if data["fps"] > 40.0 else "30fps"
            if had_candidate:
                per_video[video]["fn_class_A"] += 1  # GBM/dedup rejected
                for g in gens:
                    generator_fires[cohort][f"A_{g}"] += 1
            else:
                per_video[video]["fn_class_B"] += 1  # gates filtered all generators
                generator_fires[cohort]["B_no_generator_fired"] += 1

    # Per-video table
    print(f"{'video':<8} {'fps':>5} {'gt':>5} {'matched':>8} {'fn':>5} "
          f"{'fn%':>6} {'C':>4} {'D':>4} {'A':>4} {'B':>4}")
    cohort_totals: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    track_keys = (
        "gt_total", "matched", "fn_total",
        "fn_class_C", "fn_class_D", "fn_class_A", "fn_class_B",
    )
    for cohort_name, videos in COHORTS.items():
        for v in videos:
            row = per_video[v]
            n_gt = row["gt_total"]
            n_matched = row["matched"]
            n_fn = row["fn_total"]
            n_fn_c = row["fn_class_C"]
            n_fn_d = row["fn_class_D"]
            n_fn_a = row["fn_class_A"]
            n_fn_b = row["fn_class_B"]
            fn_pct = (n_fn / n_gt * 100) if n_gt > 0 else 0
            fps_avg = next(
                (d["fps"] for r, d in rally_data.items() if d["video"] == v), 0.0,
            )
            print(
                f"{v:<8} {fps_avg:>5.1f} {n_gt:>5d} {n_matched:>8d} {n_fn:>5d} "
                f"{fn_pct:>5.1f}% {n_fn_c:>4d} {n_fn_d:>4d} {n_fn_a:>4d} {n_fn_b:>4d}"
            )
            for k in track_keys:
                cohort_totals[cohort_name][k] += row[k]

    print()
    print(f"{'cohort':<10} {'gt':>5} {'matched':>8} {'fn':>5} "
          f"{'fn%':>6} {'C%':>5} {'D%':>5} {'A%':>5} {'B%':>5}")
    for cohort_name in ("60fps", "30fps"):
        c = cohort_totals[cohort_name]
        n_gt = c["gt_total"]
        n_fn = c["fn_total"]
        if n_gt == 0:
            continue
        fn_pct = n_fn / n_gt * 100
        c_pct = c["fn_class_C"] / n_gt * 100
        d_pct = c["fn_class_D"] / n_gt * 100
        a_pct = c["fn_class_A"] / n_gt * 100
        b_pct = c["fn_class_B"] / n_gt * 100
        print(
            f"{cohort_name:<10} {n_gt:>5d} {c['matched']:>8d} {n_fn:>5d} "
            f"{fn_pct:>5.1f}% {c_pct:>4.1f}% {d_pct:>4.1f}% {a_pct:>4.1f}% {b_pct:>4.1f}%"
        )

    print()
    print("Generator-fire breakdown on class A (which generator produced the "
          "candidate that the GBM/dedup then rejected):")
    for cohort_name in ("60fps", "30fps"):
        print(f"  [{cohort_name}]")
        for k in sorted(generator_fires[cohort_name].keys()):
            print(f"    {k}: {generator_fires[cohort_name][k]}")

    print()
    print()
    print("Legend:")
    print(f"  C = no confident ball detection within ±{SUPPORT_WINDOW}f of GT (upstream FN)")
    print(f"  D = ball present but no player within {PLAYER_CONTACT_RADIUS} (attribution FN)")
    print(f"  A = candidate WAS generated within ±{CANDIDATE_WINDOW}f, GBM/dedup rejected it (FIX = GBM retrain)")
    print(f"  B = NO candidate generated within ±{CANDIDATE_WINDOW}f, gates filtered everything (FIX = scale culprit gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
