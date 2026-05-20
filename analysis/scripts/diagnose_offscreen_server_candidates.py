"""Fleet diagnostic — find candidate off-screen-server rallies.

For every rally with action GT we ask: did the production pipeline miss the
real serve frame because the server was off-screen?

Per-rally signals collected (no production changes, diagnostic only):

  - first_pred_contact_frame   : earliest frame from current detect_contacts
  - gt_first_action_frame      : earliest action frame from beach_v11 GT
  - gt_first_action_type       : "serve" / "receive" / etc.
  - gap_pred_to_gt             : first_pred_contact_frame - gt_first_action_frame
                                  (positive = pipeline missed an earlier action)
  - pre_first_seq_serve_peak_f : argmax of MS-TCN++ serve-class probability in
                                  the window [rally_start, first_pred_contact-5]
  - pre_first_seq_serve_peak_p : peak probability
  - early_player_track_count   : distinct tracked player tracks present in
                                  frames [0, 60]
  - early_min_player_distance  : minimum ball-to-player distance in [0, 60]
                                  (court-normalised)

Then we classify each rally into one of these buckets:

  off_screen_candidate
    gap_pred_to_gt > 60 (≈1 sec)
    pre_first_seq_serve_peak_p >= 0.70
    early_player_track_count <= 1  OR  early_min_player_distance >= 0.10

  late_real_serve_no_off_screen
    gap_pred_to_gt > 60 BUT players are present near the early ball
    (probably the GT is just generous on the serve toss frame and the
    pipeline's first-contact frame is the real serve)

  pipeline_already_correct
    gap_pred_to_gt <= 15 (within HIT_TOLERANCE)

  pre_serve_artifact
    gap_pred_to_gt > 60 AND no MS-TCN++ serve peak before the first contact
    (pipeline isn't missing a serve, it's just labelling something pre-action)

The script prints one row per rally and a final cluster summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import (
    get_connection,
    load_court_calibration,
)
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs
from rallycut.training.action_gt_query import load_for_rallies

GT_PATH = Path("training_datasets/beach_v11/action_ground_truth.json")
HIT_TOLERANCE = 15
EARLY_WINDOW_FRAMES = 60  # frames [0, 60] — ~1s at 60fps
SERVE_PEAK_FLOOR = 0.50  # consider any MS-TCN++ peak above this
PLAYER_EMPTY_TRACK_COUNT = 1  # ≤1 distinct player track in early window
PLAYER_EMPTY_DISTANCE = 0.10  # ball is at least this far from nearest player
GAP_THRESHOLD = 60  # frames — pred-to-gt mismatch above this is "wide"


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


def _seq_peak_before(
    seq: np.ndarray | None,
    upper_excl: int,
    floor: float,
) -> tuple[int, float]:
    """Returns (peak_frame, peak_prob) for serve-class peak in [0, upper_excl).

    Returns (-1, 0.0) if no peak above floor.
    """
    if seq is None or upper_excl <= 0:
        return (-1, 0.0)
    serve_idx = ACTION_TYPES.index("serve") + 1
    serve_row = seq[serve_idx, :upper_excl]
    if serve_row.size == 0:
        return (-1, 0.0)
    peak_frame = int(np.argmax(serve_row))
    peak_prob = float(serve_row[peak_frame])
    if peak_prob < floor:
        return (-1, peak_prob)
    return (peak_frame, peak_prob)


def _early_window_signals(
    bp: list[BallPosition],
    pp: list[PlayerPosition],
    end_frame: int,
) -> tuple[int, float, float]:
    """Signals over frames [0, end_frame). Returns:
      - distinct_track_count    : number of distinct player track_ids present
      - min_ball_player_distance: normalised-image distance, inf if no valid ball
      - ball_visibility_fraction: fraction of frames in window with ball.x>0.01
                                  (proxy for "ball is in-frame")
    """
    tracks_in_window = {
        p.track_id for p in pp
        if 0 <= p.frame_number < end_frame and p.track_id >= 0
    }
    pp_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in pp:
        pp_by_frame.setdefault(p.frame_number, []).append(p)
    distances: list[float] = []
    ball_visible_frames = 0
    bp_by_frame: dict[int, BallPosition] = {b.frame_number: b for b in bp}
    for f in range(end_frame):
        b = bp_by_frame.get(f)
        if b is None:
            continue
        if b.x <= 0.01 and b.y <= 0.01:
            continue
        ball_visible_frames += 1
        for p in pp_by_frame.get(f, []):
            px = p.x + p.width / 2
            py = p.y + p.height / 2
            distances.append(float(np.hypot(b.x - px, b.y - py)))
    vis_fraction = ball_visible_frames / max(end_frame, 1)
    return (
        len(tracks_in_window),
        min(distances) if distances else float("inf"),
        vis_fraction,
    )


def main() -> None:
    with open(GT_PATH) as f:
        gt = json.load(f)

    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, content_hash, name FROM videos "
                "WHERE content_hash = ANY(%s)",
                [list(gt_hashes)],
            )
            meta = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, (h, _) in meta.items()}

    header = (
        f"{'video':<8} {'rally':<10} {'gt_act':<7} {'gt_f':>5} {'pred_f':>6} "
        f"{'gap':>5} {'peak_f':>6} {'peak_p':>6} {'trks':>4} {'pdist':>6}  "
        f"{'cluster':<28}"
    )
    print(header)
    print("-" * len(header))

    rows: list[dict[str, Any]] = []
    cluster_counts: dict[str, int] = {}

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
                       FROM rallies r LEFT JOIN player_tracks pt
                       ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND r.start_ms = %s""",
                    [vid, r["rally_start_ms"]],
                )
                row = cur.fetchone()
            if row is None or not row[1]:
                continue
            rid, _fps, fcount, csy, bp_json, pp_json, aj, primary_raw = row

            gt_by_rally = load_for_rallies(conn, [str(rid)])
            gt_actions = gt_by_rally.get(str(rid), [])
            if not gt_actions:
                continue
            gt_first = min(gt_actions, key=lambda a: a.get("frame", 0))
            gt_first_f = int(gt_first.get("frame", 0))
            gt_first_type = str(gt_first.get("action", "?"))

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

            corners = load_court_calibration(vid)
            cal: CourtCalibrator | None = None
            if corners and len(corners) == 4:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])

            seq = get_sequence_probs(
                bp, pp, csy, fcount or 0, ta_int, calibrator=cal,
            )

            # Run current production detect_contacts (NO rescue toggle —
            # this is the "what the pipeline sees today" baseline)
            contact_seq = detect_contacts(
                ball_positions=bp,
                player_positions=pp,
                config=ContactDetectionConfig(),
                frame_count=fcount or None,
                team_assignments=ta_int,
                sequence_probs=seq,
                primary_track_ids=list(primary_raw or []) or None,
            )
            if not contact_seq.contacts:
                first_pred_f = -1
            else:
                first_pred_f = min(c.frame for c in contact_seq.contacts)

            if first_pred_f >= 0:
                gap = first_pred_f - gt_first_f
                peak_f, peak_p = _seq_peak_before(
                    seq, max(first_pred_f - 5, 0), SERVE_PEAK_FLOOR,
                )
            else:
                gap = -gt_first_f
                peak_f, peak_p = (-1, 0.0)

            tracks_n, pdist = _early_window_signals(
                bp, pp, EARLY_WINDOW_FRAMES,
            )

            if first_pred_f >= 0 and abs(gap) <= HIT_TOLERANCE:
                cluster = "pipeline_already_correct"
            elif gap > GAP_THRESHOLD:
                # the pipeline missed something at the start
                has_peak = peak_f >= 0 and peak_p >= 0.70
                player_empty = (
                    tracks_n <= PLAYER_EMPTY_TRACK_COUNT
                    or pdist >= PLAYER_EMPTY_DISTANCE
                )
                if has_peak and player_empty:
                    cluster = "off_screen_candidate"
                elif has_peak and not player_empty:
                    cluster = "late_real_serve_no_off_screen"
                elif not has_peak:
                    cluster = "wide_gap_no_peak"
                else:
                    cluster = "uncategorised"
            elif gap < -GAP_THRESHOLD:
                # pipeline reported a contact way BEFORE GT first action
                cluster = "pre_serve_artifact"
            else:
                cluster = "near_miss_other"

            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
            rows.append({
                "video": name,
                "rally_id": rid,
                "gt_first_type": gt_first_type,
                "gt_first_f": gt_first_f,
                "first_pred_f": first_pred_f,
                "gap": gap,
                "peak_f": peak_f,
                "peak_p": peak_p,
                "tracks_n": tracks_n,
                "pdist": pdist,
                "cluster": cluster,
            })

            print(
                f"{name[:8]:<8} {rid[:8]:<10} {gt_first_type[:7]:<7} "
                f"{gt_first_f:>5} {first_pred_f:>6} {gap:>+5d} "
                f"{peak_f:>6} {peak_p:>6.2f} {tracks_n:>4} {pdist:>6.3f}  "
                f"{cluster:<28}"
            )

    print("-" * len(header))
    print()
    print("Cluster summary:")
    total = sum(cluster_counts.values())
    for cl, n in sorted(cluster_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cl:<32} {n:>4}  ({100 * n / total:.1f}%)")
    print(f"  {'TOTAL':<32} {total:>4}")

    out_path = Path("reports/offscreen_server_diagnostic.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"rows": rows, "cluster_counts": cluster_counts}, f, indent=2)
    print(f"\nWrote per-rally details to {out_path}")


if __name__ == "__main__":
    main()
