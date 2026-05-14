"""Fleet-wide S4 (anti-self-touch + trajectory-integral) flip-candidate sweep.

S4 = trajectory-integral (S3) with anti-self-touch tiebreak. For each contact
in a rally that has a previous contact, score each same-team candidate by the
mean L2 distance between their bbox center and the ball position over the
K=10 frames immediately preceding the contact frame. Pick the candidate with
the minimum mean distance. If that pick equals the previous toucher's
track_id AND the previous action is not BLOCK, exclude that candidate and
pick the next-best. Compare S4's pick to the pipeline's stored
`playerTrackId`; if they differ, this contact is a "flip candidate".

This script reads stored `actions_json`, `contacts_json`, `positions_json`,
and `ball_positions_json` from `player_tracks` — no detect/classify re-run.
It enumerates flip candidates across the fleet and writes them to JSON.

Usage:
    cd analysis
    uv run python scripts/measure_s4_fleet_rate.py
    uv run python scripts/measure_s4_fleet_rate.py --video <video_id>
    uv run python scripts/measure_s4_fleet_rate.py --limit 50
    uv run python scripts/measure_s4_fleet_rate.py --output <path.json>
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    HERE.parent / "reports" / "probe_b_sequence_aware" / "2026_05_14"
    / "s4_fleet_candidates.json"
)

K_PRE = 10                # frames before contact for trajectory window
MIN_BALL_PRE = 5          # require >= N ball positions in pre-window
ACTION_FRAME_TOL = 0      # actions are pinned to contact frames; require exact match


@dataclass
class FlipCandidate:
    rally_id: str
    rally_short: str
    video_id: str
    video_name: str
    rally_order: int
    rally_start_ms: int
    fps: float
    pl_frame: int                  # rally-relative frame
    action_idx: int
    action_type: str
    pipeline_pid: int              # current actions_json[i].playerTrackId
    s4_pid: int                    # S4's alternative pick
    pipeline_team: str | None
    s4_team: str | None
    prev_action_idx: int
    prev_action_type: str
    prev_action_frame: int
    prev_toucher_pid: int | None
    prev_toucher_team: str | None
    same_team_cands: list[tuple[int, float]]  # (tid, mean_dist_pre)
    n_pre_ball: int                # ball-position coverage in pre-window
    s3_integrals: dict[str, float] = field(default_factory=dict)
    note: str = ""


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _compute_s4_for_rally(
    *,
    rally_id: str,
    rally_short: str,
    video_id: str,
    video_name: str,
    rally_order: int,
    rally_start_ms: int,
    fps: float,
    actions_json: dict[str, Any],
    contacts_json: dict[str, Any],
    positions_json: list[dict[str, Any]],
    ball_positions_json: list[dict[str, Any]],
) -> list[FlipCandidate]:
    """For each contact (other than the first) in the rally, run S4 logic and
    emit a FlipCandidate when S4 picks differently from the pipeline.
    """
    actions = actions_json.get("actions", []) or []
    team_ass = actions_json.get("teamAssignments", {}) or {}
    contacts = contacts_json.get("contacts", []) or []
    if not actions or not contacts:
        return []

    # Build lookup: frame -> contact
    contacts_by_frame: dict[int, dict[str, Any]] = {}
    for cc in contacts:
        try:
            f = int(cc.get("frame", -1))
        except (TypeError, ValueError):
            continue
        contacts_by_frame[f] = cc

    # Build lookup: frame -> {trackId -> (cx, cy)} for player positions.
    # positions_json `x`,`y` are already bbox centers in normalized coords.
    positions_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for p in positions_json or []:
        try:
            f = int(p.get("frameNumber", -1))
            tid = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            continue
        positions_by_frame.setdefault(f, {})[tid] = (
            float(p.get("x", 0.0)),
            float(p.get("y", 0.0)),
        )

    # Ball positions by frame.
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_positions_json or []:
        try:
            f = int(bp["frameNumber"])
            bx = float(bp.get("x", 0.0))
            by = float(bp.get("y", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if bx <= 0 and by <= 0:
            continue
        ball_by_frame[f] = (bx, by)

    flips: list[FlipCandidate] = []

    for i, action in enumerate(actions):
        if i == 0:
            continue  # no previous contact

        try:
            pl_frame = int(action.get("frame", -1))
            pl_pid = int(action.get("playerTrackId") or -1)
        except (TypeError, ValueError):
            continue
        if pl_pid < 0 or pl_frame < 0:
            continue
        pl_team = team_ass.get(str(pl_pid))
        if not pl_team:
            continue

        prev = actions[i - 1]
        try:
            prev_pid_raw = prev.get("playerTrackId")
            prev_pid = int(prev_pid_raw) if prev_pid_raw is not None else None
            prev_frame = int(prev.get("frame", -1))
        except (TypeError, ValueError):
            continue
        prev_action_type = str(prev.get("action", "")).upper() or ""

        # Pre-window ball positions over [pl_frame - K, pl_frame - 1]
        pre_ball: list[tuple[int, float, float]] = []
        for f in range(pl_frame - K_PRE, pl_frame):
            if f in ball_by_frame:
                bx, by = ball_by_frame[f]
                pre_ball.append((f, bx, by))
        if len(pre_ball) < MIN_BALL_PRE:
            continue

        # Same-team candidates from the contact at pl_frame.
        cont = contacts_by_frame.get(pl_frame)
        if not cont:
            continue
        raw_cands = cont.get("playerCandidates", []) or []
        same_team_tids: list[int] = []
        for cd in raw_cands:
            if not cd or len(cd) < 2:
                continue
            try:
                tid = int(cd[0])
            except (TypeError, ValueError):
                continue
            if team_ass.get(str(tid)) == pl_team:
                same_team_tids.append(tid)
        if len(same_team_tids) < 2:
            continue  # need >= 2 to flip

        # S3 integrals: mean distance over pre-window frames where both
        # ball and candidate bbox exist.
        s3_integrals: dict[int, float] = {}
        for tid in same_team_tids:
            total = 0.0
            count = 0
            for f, bx, by in pre_ball:
                frame_positions = positions_by_frame.get(f)
                if not frame_positions:
                    continue
                bb = frame_positions.get(tid)
                if bb is None:
                    continue
                total += _dist(bb, (bx, by))
                count += 1
            if count > 0:
                s3_integrals[tid] = total / count

        if len(s3_integrals) < 2:
            continue

        # S3 pick = argmin
        s3_pick = min(s3_integrals.items(), key=lambda kv: kv[1])[0]

        # S4 pick = S3, but if S3 picks prev_toucher and prev_action != BLOCK,
        # pick next-best instead.
        s4_pick = s3_pick
        if (
            prev_pid is not None
            and s3_pick == prev_pid
            and prev_action_type != "BLOCK"
        ):
            filt = {t: v for t, v in s3_integrals.items() if t != prev_pid}
            if filt:
                s4_pick = min(filt.items(), key=lambda kv: kv[1])[0]

        if s4_pick == pl_pid:
            continue  # no flip

        # Build sorted (tid, mean_dist) for reporting
        sorted_cands = sorted(s3_integrals.items(), key=lambda kv: kv[1])

        flips.append(FlipCandidate(
            rally_id=rally_id,
            rally_short=rally_short,
            video_id=video_id,
            video_name=video_name,
            rally_order=rally_order,
            rally_start_ms=rally_start_ms,
            fps=fps,
            pl_frame=pl_frame,
            action_idx=i,
            action_type=str(action.get("action", "")).upper(),
            pipeline_pid=pl_pid,
            s4_pid=s4_pick,
            pipeline_team=pl_team,
            s4_team=team_ass.get(str(s4_pick)),
            prev_action_idx=i - 1,
            prev_action_type=prev_action_type,
            prev_action_frame=prev_frame,
            prev_toucher_pid=prev_pid,
            prev_toucher_team=(team_ass.get(str(prev_pid)) if prev_pid is not None else None),
            same_team_cands=[(t, float(d)) for t, d in sorted_cands],
            n_pre_ball=len(pre_ball),
            s3_integrals={str(t): float(d) for t, d in s3_integrals.items()},
        ))

    return flips


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fleet-wide S4 flip-candidate sweep (no DB writes)",
    )
    parser.add_argument("--video", type=str, help="Only this video ID")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after N rallies (0 = all)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output JSON path")
    args = parser.parse_args()

    where_clauses = [
        "pt.ball_positions_json IS NOT NULL",
        "pt.actions_json IS NOT NULL",
        "pt.contacts_json IS NOT NULL",
        "pt.positions_json IS NOT NULL",
    ]
    params: list[Any] = []
    if args.video:
        where_clauses.append("r.video_id = %s")
        params.append(args.video)
    where_sql = " AND ".join(where_clauses)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(f"""
            SELECT r.id, r.video_id, r."order", r.start_ms,
                   pt.actions_json, pt.contacts_json,
                   pt.positions_json, pt.ball_positions_json,
                   v.name, v.fps
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE {where_sql}
            ORDER BY v.name, r."order"
        """, params)
        rows = cur.fetchall()

    if args.limit:
        rows = rows[: args.limit]

    print(f"Fleet sweep: {len(rows)} rallies")
    t_start = time.monotonic()

    all_flips: list[FlipCandidate] = []
    errors = 0
    n_with_flips = 0

    for i, row in enumerate(rows):
        rally_id = str(row[0])
        video_id = str(row[1])
        rally_order = int(row[2]) if row[2] is not None else -1
        rally_start_ms = int(row[3] or 0)
        actions_json = cast(dict[str, Any], row[4]) or {}
        contacts_json = cast(dict[str, Any], row[5]) or {}
        positions_json = cast(list[dict[str, Any]], row[6]) or []
        ball_positions_json = cast(list[dict[str, Any]], row[7]) or []
        video_name = str(row[8] or "")
        fps = float(row[9]) if row[9] is not None else 30.0

        try:
            flips = _compute_s4_for_rally(
                rally_id=rally_id,
                rally_short=rally_id[:8],
                video_id=video_id,
                video_name=video_name,
                rally_order=rally_order,
                rally_start_ms=rally_start_ms,
                fps=fps,
                actions_json=actions_json,
                contacts_json=contacts_json,
                positions_json=positions_json,
                ball_positions_json=ball_positions_json,
            )
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  ERROR {video_name}/{rally_id[:8]}: {e}")
            continue

        if flips:
            n_with_flips += 1
            all_flips.extend(flips)
            elapsed = time.monotonic() - t_start
            tags = ",".join(
                f"f{f.pl_frame}:p{f.pipeline_pid}->p{f.s4_pid}({f.action_type}|prev={f.prev_action_type})"
                for f in flips
            )
            print(
                f"  [{i+1}/{len(rows)}] {video_name}/{rally_id[:8]} "
                f"({len(flips)} flips): {tags}  ({elapsed:.1f}s)"
            )
        else:
            elapsed = time.monotonic() - t_start
            print(f"  [{i+1}/{len(rows)}] {video_name}/{rally_id[:8]}: 0 flips ({elapsed:.1f}s)")

    elapsed = time.monotonic() - t_start
    print()
    print("=" * 60)
    print(f"Total rallies processed: {len(rows)}")
    print(f"Rallies with >=1 flip:   {n_with_flips}")
    print(f"Total flip candidates:   {len(all_flips)}")
    print(f"Errors:                  {errors}")
    print(f"Elapsed:                 {elapsed:.1f}s")

    # Stratify by shape for visibility:
    same_team_attacks = sum(
        1 for f in all_flips
        if f.action_type == "ATTACK" and f.prev_action_type == "ATTACK"
        and f.pipeline_team == f.prev_toucher_team
    )
    same_team_non_attack = sum(
        1 for f in all_flips
        if f.action_type in ("SET", "RECEIVE", "DIG")
        and f.pipeline_team == f.prev_toucher_team
    )
    cross_team = sum(
        1 for f in all_flips
        if f.pipeline_team != f.prev_toucher_team
    )
    print()
    print("Shape breakdown:")
    print(f"  same-team ATTACK after ATTACK: {same_team_attacks}")
    print(f"  same-team SET/RECEIVE/DIG:     {same_team_non_attack}")
    print(f"  cross-team (rare for S4):      {cross_team}")
    print(f"  other:                          {len(all_flips) - same_team_attacks - same_team_non_attack - cross_team}")

    # Write JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": {
            "K_PRE": K_PRE,
            "MIN_BALL_PRE": MIN_BALL_PRE,
        },
        "n_rallies_processed": len(rows),
        "n_rallies_with_flips": n_with_flips,
        "n_flip_candidates": len(all_flips),
        "errors": errors,
        "elapsed_s": elapsed,
        "shape_breakdown": {
            "same_team_attack_after_attack": same_team_attacks,
            "same_team_set_receive_dig": same_team_non_attack,
            "cross_team": cross_team,
        },
        "flips": [asdict(f) for f in all_flips],
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print()
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
