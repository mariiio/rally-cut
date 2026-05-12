"""Probe contact-detection gate failures on the 409-rally action-GT corpus.

For each GT action with no pipeline action within +/-10 frames, where the ball
IS tracked at the GT frame, compute the value of each contact-detection gate
at gt_frame and report which thresholds it fails. Output: per-gate failure
histogram + per-action-type stratification + per-case JSON trace.

Run from analysis/:
    uv run python -u scripts/probe_contact_gate_failures.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rallycut.evaluation.db import get_connection
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _depth_scale_at_y,
    _player_to_ball_dist,
    compute_direction_change,
)
from rallycut.tracking.player_tracker import PlayerPosition

OUT_MD = (
    Path(__file__).resolve().parent.parent
    / "reports" / "contact_detection_fn" / "probe_2026_05_12.md"
)
OUT_JSON = (
    Path(__file__).resolve().parent.parent
    / "reports" / "contact_detection_fn" / "probe_2026_05_12.json"
)
MATCH_TOLERANCE_FRAMES = 10
BALL_PRESENCE_HALF_WINDOW = 5  # ball within +/-5 frames counts as "tracked at"


def load_corpus_rallies() -> list[dict[str, Any]]:
    """Load all rallies with action GT. Returns rally records with everything
    needed for gate diagnosis."""
    rallies: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id::text, r.video_id::text, r.start_ms, r.end_ms,
                       pt.actions_json,
                       pt.contacts_json,
                       pt.ball_positions_json,
                       pt.positions_json,
                       pt.primary_track_ids
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE EXISTS (
                    SELECT 1 FROM rally_action_ground_truth gt
                    WHERE gt.rally_id = r.id
                )
                """
            )
            rows = cur.fetchall()
        # Load GT separately
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT rally_id::text, frame, action,
                       resolved_track_id, snapshot_track_id
                FROM rally_action_ground_truth
                ORDER BY rally_id, frame
                """
            )
            gt_rows = cur.fetchall()
    gt_by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rid, frame, action, resolved_tid, snap_tid in gt_rows:
        gt_by_rally[rid].append({
            "frame": int(frame),
            "action": str(action).lower() if action is not None else None,
            "track_id": resolved_tid if resolved_tid is not None else snap_tid,
        })
    for rid, vid, sm, em, actions_json, contacts_json, ball_json, pos_json, ptids in rows:
        rallies.append({
            "rally_id": rid,
            "video_id": vid,
            "start_ms": sm,
            "end_ms": em,
            "actions": (actions_json or {}).get("actions", []) if isinstance(actions_json, dict) else [],
            "contacts": (contacts_json or {}).get("contacts", []) if isinstance(contacts_json, dict) else [],
            "ball_json": ball_json or [],
            "pos_json": pos_json or [],
            "primary_track_ids": ptids or [],
            "gt": gt_by_rally.get(rid, []),
        })
    return rallies


def _build_ball_by_frame(ball_json: list[dict[str, Any]]) -> dict[int, BallPosition]:
    out: dict[int, BallPosition] = {}
    for bp in ball_json:
        fn = bp.get("frameNumber")
        if fn is None:
            continue
        out[int(fn)] = BallPosition(
            frame_number=int(fn),
            x=float(bp["x"]),
            y=float(bp["y"]),
            confidence=float(bp.get("confidence", 1.0)),
        )
    return out


def _build_players_by_frame(pos_json: list[dict[str, Any]]) -> list[PlayerPosition]:
    out: list[PlayerPosition] = []
    for p in pos_json:
        try:
            out.append(PlayerPosition(
                track_id=int(p["trackId"]),
                frame_number=int(p["frameNumber"]),
                x=float(p["x"]),
                y=float(p["y"]),
                width=float(p["width"]),
                height=float(p["height"]),
                confidence=float(p.get("confidence", 1.0)),
                keypoints=p.get("keypoints"),
            ))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _frame_velocity(
    ball_by_frame: dict[int, BallPosition], frame: int
) -> float:
    """Speed (units/frame) at `frame` from +/-1 ball positions."""
    before = ball_by_frame.get(frame - 1)
    after = ball_by_frame.get(frame + 1)
    if before is None or after is None:
        # Try wider
        for offset in range(2, 6):
            if before is None and (frame - offset) in ball_by_frame:
                before = ball_by_frame[frame - offset]
            if after is None and (frame + offset) in ball_by_frame:
                after = ball_by_frame[frame + offset]
            if before is not None and after is not None:
                dx = after.x - before.x
                dy = after.y - before.y
                gap = (after.frame_number - before.frame_number)
                return math.sqrt(dx * dx + dy * dy) / max(gap, 1)
        return 0.0
    dx = after.x - before.x
    dy = after.y - before.y
    return math.sqrt(dx * dx + dy * dy) / 2.0


def diagnose_gates(
    gt_frame: int,
    ball_by_frame: dict[int, BallPosition],
    players: list[PlayerPosition],
    primary_track_ids: list[int],
    cfg: ContactDetectionConfig,
) -> dict[str, Any]:
    """Compute each gate's value at gt_frame and report pass/fail."""
    # Direction change
    dir_change = compute_direction_change(
        ball_by_frame, gt_frame, check_frames=cfg.direction_check_frames
    )
    inflection_dir_change = compute_direction_change(
        ball_by_frame, gt_frame, check_frames=cfg.inflection_check_frames
    )
    # Velocity at gt_frame (use +/-1 frame to compute speed)
    velocity = _frame_velocity(ball_by_frame, gt_frame)
    # Nearest player
    bp_at = ball_by_frame.get(gt_frame)
    if bp_at is None:
        # Interpolate ball position
        nearest_before = max(
            (f for f in ball_by_frame if f < gt_frame), default=None
        )
        nearest_after = min(
            (f for f in ball_by_frame if f > gt_frame), default=None
        )
        if nearest_before is not None and nearest_after is not None:
            t = (gt_frame - nearest_before) / (nearest_after - nearest_before)
            ball_x = (
                ball_by_frame[nearest_before].x
                + t * (ball_by_frame[nearest_after].x - ball_by_frame[nearest_before].x)
            )
            ball_y = (
                ball_by_frame[nearest_before].y
                + t * (ball_by_frame[nearest_after].y - ball_by_frame[nearest_before].y)
            )
        elif nearest_before is not None:
            ball_x = ball_by_frame[nearest_before].x
            ball_y = ball_by_frame[nearest_before].y
        elif nearest_after is not None:
            ball_x = ball_by_frame[nearest_after].x
            ball_y = ball_by_frame[nearest_after].y
        else:
            ball_x = ball_y = float("nan")
    else:
        ball_x, ball_y = bp_at.x, bp_at.y

    nearest_dist = float("inf")
    nearest_tid = -1
    if not math.isnan(ball_x):
        # Match production: scan +/-player_search_frames window
        primary_set = set(int(t) for t in primary_track_ids) if primary_track_ids else None
        for p in players:
            if abs(p.frame_number - gt_frame) > cfg.player_search_frames:
                continue
            if primary_set is not None and p.track_id not in primary_set:
                continue
            d = _player_to_ball_dist(p, ball_x, ball_y)
            if d < nearest_dist:
                nearest_dist = d
                nearest_tid = p.track_id
        # Apply depth scaling per production. We pass court_calibrator=None
        # because we don't have per-rally calibration available cheaply here;
        # _depth_scale_at_y returns 1.0 in that case (matches rallies that
        # have no calibration available at attribution time).
        depth_scale = _depth_scale_at_y(ball_y, None)
        effective_radius = cfg.player_contact_radius * depth_scale
    else:
        effective_radius = cfg.player_contact_radius

    # Gate evaluations
    gates = {
        "direction_change_deg": {
            "value": dir_change,
            "threshold": cfg.min_direction_change_deg,
            "passes": dir_change >= cfg.min_direction_change_deg,
        },
        "inflection_angle_deg": {
            "value": inflection_dir_change,
            "threshold": cfg.min_inflection_angle_deg,
            "passes": inflection_dir_change >= cfg.min_inflection_angle_deg,
        },
        "velocity": {
            "value": velocity,
            "threshold": cfg.min_peak_velocity,
            "passes": velocity >= cfg.min_peak_velocity,
        },
        "min_candidate_velocity": {
            "value": velocity,
            "threshold": cfg.min_candidate_velocity,
            "passes": velocity >= cfg.min_candidate_velocity,
        },
        "player_radius_depth_scaled": {
            "value": nearest_dist,
            "threshold": effective_radius,
            "passes": nearest_dist <= effective_radius,
            "nearest_tid": nearest_tid,
        },
    }
    return {
        "gates": gates,
        "ball_xy": (ball_x, ball_y),
        "nearest_player_dist": nearest_dist,
        "nearest_player_tid": nearest_tid,
        "depth_scale_at_ball_y": (
            _depth_scale_at_y(ball_y, None) if not math.isnan(ball_y) else None
        ),
    }


def main() -> int:
    cfg = ContactDetectionConfig()
    rallies = load_corpus_rallies()
    print(f"[probe] {len(rallies)} rallies loaded", flush=True)

    # ---- PARITY CHECK: probe should say "all gates pass" for known TPs ----
    parity_failures: list[str] = []
    parity_checked = 0
    for r in rallies:
        if parity_checked >= 5:
            break
        ball_by_frame = _build_ball_by_frame(r["ball_json"])
        players = _build_players_by_frame(r["pos_json"])
        for c in r["contacts"][:1]:  # first contact in each rally
            cf = c.get("frame")
            if cf is None:
                continue
            cf = int(cf)
            diag = diagnose_gates(cf, ball_by_frame, players,
                                  r["primary_track_ids"], cfg)
            failing = [g for g, gd in diag["gates"].items() if not gd["passes"]]
            if failing:
                # The strict gates may legitimately fail for some real contacts
                # (sequence-rescue paths cover these). Log but don't error.
                # We're checking ORDER OF MAGNITUDE: if every contact fails
                # every gate, the probe is wrong.
                parity_failures.append(
                    f"  rally={r['rally_id'][:8]} f={cf} fails: {failing}"
                )
            parity_checked += 1
            break
    print(f"[probe parity] checked {parity_checked} known-true contacts; "
          f"{len(parity_failures)} had any failing gate", flush=True)
    for line in parity_failures[:5]:
        print(line, flush=True)
    if parity_failures and len(parity_failures) == parity_checked:
        print("[probe parity] WARN: every checked contact had a failing gate. "
              "This may indicate the probe's gate logic is wrong. Investigate "
              "before trusting the output.", flush=True)

    cases: list[dict[str, Any]] = []
    n_matched = 0
    n_missing = 0
    n_no_ball = 0
    n_ball_tracked = 0

    for r in rallies:
        ball_by_frame = _build_ball_by_frame(r["ball_json"])
        players = _build_players_by_frame(r["pos_json"])
        pipeline_actions = r["actions"]
        # Index pipeline actions by frame for tolerance lookup
        pa_frames = sorted(int(pa["frame"]) for pa in pipeline_actions if pa.get("frame") is not None)

        for gt in r["gt"]:
            gt_frame = gt["frame"]
            # Match within +/-tolerance to a pipeline action
            matched = any(
                abs(pf - gt_frame) <= MATCH_TOLERANCE_FRAMES for pf in pa_frames
            )
            if matched:
                n_matched += 1
                continue
            n_missing += 1
            # Ball present at gt_frame?
            ball_present = any(
                abs(f - gt_frame) <= BALL_PRESENCE_HALF_WINDOW for f in ball_by_frame
            )
            if not ball_present:
                n_no_ball += 1
                continue
            n_ball_tracked += 1
            diag = diagnose_gates(
                gt_frame, ball_by_frame, players,
                r["primary_track_ids"], cfg,
            )
            cases.append({
                "rally_id": r["rally_id"][:8],
                "rally_id_full": r["rally_id"],
                "video_id": r["video_id"],
                "gt_frame": gt_frame,
                "gt_action": gt["action"],
                "gt_track_id": gt["track_id"],
                **diag,
            })

    print(f"[probe] {n_matched} matched, {n_missing} missing, "
          f"{n_no_ball} no-ball, {n_ball_tracked} ball-tracked", flush=True)
    return _write_outputs(cases, n_matched, n_missing, n_no_ball)


def _write_outputs(
    cases: list[dict[str, Any]],
    n_matched: int,
    n_missing: int,
    n_no_ball: int,
) -> int:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    n = len(cases)

    # Per-gate failure histogram
    gate_fail_counts: Counter = Counter()
    for c in cases:
        for gname, gd in c["gates"].items():
            if not gd["passes"]:
                gate_fail_counts[gname] += 1

    # Per-action-type stratification
    by_action: dict[str, Counter] = defaultdict(Counter)
    by_action_n: Counter = Counter()
    for c in cases:
        action_key = c["gt_action"] or "?"
        by_action_n[action_key] += 1
        for gname, gd in c["gates"].items():
            if not gd["passes"]:
                by_action[action_key][gname] += 1

    # Print + write markdown
    lines = [
        "# Contact-Detection FN Probe -- 2026-05-12",
        "",
        "Corpus: 409-rally action-GT corpus.",
        f"- matched (within +/-{MATCH_TOLERANCE_FRAMES} frames): {n_matched}",
        f"- missing (no pipeline action within +/-{MATCH_TOLERANCE_FRAMES}): {n_missing}",
        f"  - of those, no ball at gt_frame (+/-{BALL_PRESENCE_HALF_WINDOW}): {n_no_ball}",
        f"  - of those, ball tracked at gt_frame: **{n}**  <- probe target",
        "",
        "## Gate-failure histogram (across all ball-tracked-no-contact cases)",
        "",
        f"| Gate | Failures | % of {n} |",
        "|---|---|---|",
    ]
    for gname, cnt in gate_fail_counts.most_common():
        pct = cnt / n * 100 if n else 0
        lines.append(f"| `{gname}` | {cnt} | {pct:.1f}% |")
    lines.extend(["", "## Per-action-type stratification", ""])
    actions = sorted(by_action_n.keys())
    if actions:
        gate_names = list(gate_fail_counts.keys())
        header = "| action | n | " + " | ".join(f"`{g}`" for g in gate_names) + " |"
        sep = "|---|---|" + "---|" * len(gate_names)
        lines.append(header)
        lines.append(sep)
        for a in actions:
            row = [a, str(by_action_n[a])]
            for g in gate_names:
                row.append(str(by_action[a].get(g, 0)))
            lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Per-case JSON: see `probe_2026_05_12.json` for full traces.")
    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))

    OUT_JSON.write_text(json.dumps({
        "n_matched": n_matched,
        "n_missing": n_missing,
        "n_no_ball": n_no_ball,
        "n_ball_tracked": n,
        "gate_fail_counts": dict(gate_fail_counts),
        "by_action_type_n": dict(by_action_n),
        "by_action_type_gate_fails": {
            a: dict(by_action[a]) for a in actions
        },
        "cases": cases,
    }, indent=2, default=str))
    print(f"\nWrote: {OUT_MD}")
    print(f"Wrote: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
