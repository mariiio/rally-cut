"""Oracle-ceiling probe — how much does PERFECT ball-side knowledge help
attribution?

For each click-GT action across the 8-fixture corpus:
  1. Get pipeline's predicted player + team.
  2. Get GT actor's player + team.
  3. If already correct → no lift.
  4. If wrong + same-team swap (pipeline_team == actor_team) → ball-side cannot
     disambiguate. No lift.
  5. If wrong + cross-team miss (pipeline_team != actor_team):
     a. Compute ball-side at the GT frame via net_line_estimator.
     b. Determine which team is on the near vs far side for that rally (from
        player positions' median y).
     c. If ball-side == actor_team's side → ORACLE-ELIGIBLE lift. A rule-cost
        gate that excludes cross-team candidates based on ball-side would
        have given the correct team the edge.

This is an UPPER BOUND — it assumes any flip goes to the correct player
within the correct team. In practice a different same-team player might be
picked. But upper bound is the right signal for a go/no-go decision before
building the full rule-cost decoder.

Gate:
  - Oracle lift ≥ 5 pp → pursue Phase 4 net-cross soft γ integration.
  - Oracle lift 3–5 pp → borderline, consider tighter gate.
  - Oracle lift < 3 pp → close workstream for contact-detection/attribution.

Outputs:
  analysis/reports/attribution_ballside_oracle_2026_04_23.json
  analysis/reports/attribution_ballside_oracle_2026_04_23.md
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from rallycut.court.keypoint_detector import CourtKeypointDetector
from rallycut.court.net_line_estimator import NetLine, estimate_net_line_from_s3
from rallycut.evaluation.db import get_connection
from scripts.measure_endtoend import (
    VIDEO_IDS,
    _find_verdicts_path,
    load_click_gt,
    load_pipeline_actions,
    load_production_track_to_player,
)

TOLERANCE = 10  # frames
AMBIG_BAND_NORM_Y = 0.010
HIGH_BALL_ABSTAIN_NORM_Y = 0.025
VERDICTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "session3" / "verdicts"
OUT_DIR = Path(__file__).resolve().parent.parent / "reports"


def _classify_ball_side(bx: float, by: float, nl: NetLine) -> str:
    """Net-TOP decisive classifier from probe2. Returns near|far|ambiguous."""
    # Interpolate net-top line at ball x
    tl = nl.top_left_xy
    tr = nl.top_right_xy
    x1, y1 = tl
    x2, y2 = tr
    if abs(x2 - x1) < 1e-8:
        ref_y = (y1 + y2) / 2.0
    else:
        t = (bx - x1) / (x2 - x1)
        ref_y = y1 + t * (y2 - y1)
    delta = by - ref_y
    if delta < -HIGH_BALL_ABSTAIN_NORM_Y:
        return "ambiguous"  # ball well above net-top — high in flight
    if abs(delta) < AMBIG_BAND_NORM_Y:
        return "ambiguous"
    return "near" if delta > 0 else "far"


def _load_video_meta(video_id: str) -> dict:
    """Load width/height/s3_key/match_analysis for a video."""
    q = """
        SELECT v.width, v.height, v.s3_key, v.processed_s3_key, v.proxy_s3_key,
               v.match_analysis_json
        FROM videos v WHERE v.id = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, [video_id])
        row = cur.fetchone()
    if row is None:
        return {}
    ma = row[5] if row[5] else {}
    team_templates = ma.get("teamTemplates", {}) if isinstance(ma, dict) else {}
    pid_to_team: dict[int, int] = {}
    for team_label, entry in team_templates.items():
        for pid in entry.get("playerIds", []):
            pid_to_team[int(pid)] = int(team_label)
    return {
        "width": int(row[0] or 1920),
        "height": int(row[1] or 1080),
        "s3_key": row[3] or row[4] or row[2],
        "pid_to_team": pid_to_team,
    }


def _load_rally_data(rally_ids: list[str]) -> dict[str, dict]:
    """Return per-rally {ball_positions, player_positions}."""
    if not rally_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(rally_ids))
    q = f"""
        SELECT r.id::text, pt.ball_positions_json, pt.positions_json
        FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id::text IN ({placeholders})
    """
    out: dict[str, dict] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, rally_ids)
        for rid, ball_j, pos_j in cur.fetchall():
            out[rid] = {
                "ball": ball_j if isinstance(ball_j, list) else [],
                "players": pos_j if isinstance(pos_j, list) else [],
            }
    return out


def _find_rally_full_ids(rally_shorts: list[str], video_id: str) -> dict[str, str]:
    """Map 8-char prefix → full rally id for the given video."""
    if not rally_shorts:
        return {}
    q = "SELECT r.id::text FROM rallies r WHERE r.video_id = %s"
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(q, [video_id])
        all_ids = [row[0] for row in cur.fetchall()]
    out: dict[str, str] = {}
    for rs in rally_shorts:
        for rid in all_ids:
            if rid.startswith(rs):
                out[rs] = rid
                break
    return out


def _team_side_at_frame(
    players: list[dict],
    frame: int,
    tids_in_team: dict[int, list[int]],
    trackToPlayer: dict[int, int],
    pid_to_team: dict[int, int],
    window: int = 10,
) -> dict[int, str] | None:
    """At `frame±window`, determine which team is on the near (larger image-y)
    side vs far (smaller image-y) side. Returns {0: "near"|"far", 1: ...} or
    None if insufficient data.
    """
    team_ys: dict[int, list[float]] = defaultdict(list)
    for p in players:
        fn = p.get("frameNumber", -1)
        if abs(fn - frame) > window:
            continue
        tid = p.get("trackId", -1)
        pid = trackToPlayer.get(int(tid))
        if pid is None:
            continue
        team = pid_to_team.get(int(pid))
        if team is None:
            continue
        y = p.get("y", 0.0) + p.get("height", 0.0) / 2.0
        team_ys[team].append(y)
    if 0 not in team_ys or 1 not in team_ys:
        return None
    m0 = float(np.median(team_ys[0]))
    m1 = float(np.median(team_ys[1]))
    if abs(m0 - m1) < 0.03:
        return None  # teams too close to reliably infer sides
    if m0 > m1:
        return {0: "near", 1: "far"}
    return {0: "far", 1: "near"}


def _ball_at_frame(ball_positions: list[dict], frame: int, tol: int = 5) -> tuple[float, float] | None:
    best = None
    best_d = 1 << 30
    for bp in ball_positions:
        if bp.get("x", 0) <= 0:
            continue
        fn = bp.get("frameNumber", -999)
        d = abs(fn - frame)
        if d < best_d:
            best_d = d
            best = (float(bp["x"]), float(bp["y"]))
    if best is None or best_d > tol:
        return None
    return best


def main() -> int:
    detector = CourtKeypointDetector()
    if not detector.model_exists:
        print("keypoint model not available", file=sys.stderr)
        return 2

    # Aggregate stats across all videos
    agg: Counter[str] = Counter()
    per_video_stats: dict[str, Counter[str]] = {}
    oracle_details: list[dict] = []

    for vshort, vid_full in VIDEO_IDS.items():
        print(f"\n=== {vshort} ===")
        vstats: Counter[str] = Counter()
        verdicts_path = _find_verdicts_path(VERDICTS_DIR, vshort)
        if verdicts_path is None:
            print(f"  [skip] no verdicts file")
            continue

        click_gt = load_click_gt(verdicts_path)
        if not click_gt:
            print(f"  [skip] empty click-GT")
            continue

        video_meta = _load_video_meta(vid_full)
        if not video_meta:
            print(f"  [skip] no video meta")
            continue

        s3_key = video_meta["s3_key"]
        pid_to_team = video_meta["pid_to_team"]
        if not pid_to_team:
            print(f"  [skip] no teamTemplates")
            continue

        # Net-line (cached per video)
        nl = estimate_net_line_from_s3(
            s3_key,
            video_id=vid_full,
            image_width=video_meta["width"],
            image_height=video_meta["height"],
            detector=detector,
            n_frames=30,
            duration_s=30.0,
        )
        if nl is None:
            print(f"  [skip] net-line estimation failed")
            continue

        pipeline_actions = load_pipeline_actions(vid_full)
        production_ttp = load_production_track_to_player(vid_full)

        # Load rally data
        rally_shorts = list(click_gt.keys())
        rally_full = _find_rally_full_ids(rally_shorts, vid_full)
        rally_data = _load_rally_data(list(rally_full.values()))

        # Team player IDs
        tids_in_team: dict[int, list[int]] = defaultdict(list)
        for pid, team in pid_to_team.items():
            tids_in_team[team].append(pid)

        for rally_short, actions in click_gt.items():
            prod_actions = pipeline_actions.get(rally_short, [])
            ttp = production_ttp.get(rally_short, {})
            full_rid = rally_full.get(rally_short)
            rdata = rally_data.get(full_rid, {}) if full_rid else {}
            ball_positions = rdata.get("ball", [])
            player_positions = rdata.get("players", [])

            for gt_frame, v in actions.items():
                actor_pid = int(v["actor_pid"])
                actor_team = pid_to_team.get(actor_pid)
                if actor_team is None:
                    vstats["actor_team_unknown"] += 1
                    continue
                vstats["n_gt"] += 1

                # Find pipeline action within tolerance
                best_action = None
                best_d = TOLERANCE + 1
                for pa in prod_actions:
                    f = int(pa.get("frame", -9999))
                    d = abs(f - gt_frame)
                    if d < best_d:
                        best_d = d
                        best_action = pa
                if best_action is None:
                    vstats["no_pipeline_action"] += 1
                    continue

                pipeline_tid = best_action.get("playerTrackId")
                if pipeline_tid is None:
                    vstats["pipeline_tid_missing"] += 1
                    continue
                pipeline_pid = ttp.get(int(pipeline_tid))
                if pipeline_pid is None:
                    vstats["pipeline_pid_missing"] += 1
                    continue
                pipeline_team = pid_to_team.get(int(pipeline_pid))
                if pipeline_team is None:
                    vstats["pipeline_team_unknown"] += 1
                    continue

                # Correct?
                if pipeline_pid == actor_pid:
                    vstats["correct"] += 1
                    continue

                # Wrong
                vstats["wrong"] += 1

                # Same-team swap? Ball-side can't disambiguate
                if pipeline_team == actor_team:
                    vstats["wrong_same_team"] += 1
                    continue

                # Cross-team miss — candidate for ball-side disambiguation
                vstats["wrong_cross_team"] += 1

                # Ball position at gt_frame
                ball_xy = _ball_at_frame(ball_positions, gt_frame)
                if ball_xy is None:
                    vstats["oracle_no_ball"] += 1
                    continue

                # Classify ball side
                ball_side = _classify_ball_side(ball_xy[0], ball_xy[1], nl)
                if ball_side == "ambiguous":
                    vstats["oracle_ambiguous"] += 1
                    continue

                # Team-to-side mapping at this frame
                team_side = _team_side_at_frame(
                    player_positions, gt_frame, tids_in_team, ttp, pid_to_team,
                )
                if team_side is None:
                    vstats["oracle_no_team_side"] += 1
                    continue

                actor_team_side = team_side[actor_team]
                # Oracle: ball-side flips the attribution if ball-side == actor team's side
                if ball_side == actor_team_side:
                    vstats["oracle_lift_eligible"] += 1
                    oracle_details.append({
                        "video": vshort, "rally": rally_short,
                        "frame": gt_frame, "actor_pid": actor_pid,
                        "pipeline_pid": pipeline_pid,
                        "actor_team": actor_team, "pipeline_team": pipeline_team,
                        "ball_side": ball_side,
                        "actor_team_side": actor_team_side,
                    })
                else:
                    vstats["oracle_ball_on_pipeline_side"] += 1

        per_video_stats[vshort] = vstats
        for k, v in vstats.items():
            agg[k] += v
        # Per-video print
        n = vstats.get("n_gt", 0)
        corr = vstats.get("correct", 0)
        lift = vstats.get("oracle_lift_eligible", 0)
        if n > 0:
            print(f"  n_gt={n} correct={corr} ({corr/n:.1%}) "
                  f"oracle_lift_eligible={lift} ({lift/n:.1%})")
        else:
            print(f"  no GT actions")

    # Aggregate
    print()
    print("=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    n_total = agg.get("n_gt", 0)
    correct = agg.get("correct", 0)
    wrong = agg.get("wrong", 0)
    wrong_same = agg.get("wrong_same_team", 0)
    wrong_cross = agg.get("wrong_cross_team", 0)
    lift = agg.get("oracle_lift_eligible", 0)
    oracle_no_ball = agg.get("oracle_no_ball", 0)
    oracle_ambig = agg.get("oracle_ambiguous", 0)
    oracle_wrong_side = agg.get("oracle_ball_on_pipeline_side", 0)
    oracle_no_ts = agg.get("oracle_no_team_side", 0)

    if n_total == 0:
        print("No actions to evaluate.")
        return 3

    baseline_acc = correct / n_total
    oracle_max_correct = correct + lift
    oracle_max_acc = oracle_max_correct / n_total
    oracle_lift_pp = (oracle_max_acc - baseline_acc) * 100

    print(f"Total GT actions:             {n_total}")
    print(f"Baseline correct:             {correct} ({baseline_acc:.1%})")
    print(f"Wrong:                        {wrong}")
    print(f"  same-team swap (ball cannot disambiguate): {wrong_same}")
    print(f"  cross-team miss:            {wrong_cross}")
    print(f"    oracle-lift-eligible (ball on actor-team side): {lift}")
    print(f"    ball on pipeline-team side (oracle hurts): {oracle_wrong_side}")
    print(f"    oracle indeterminate (ball ambiguous/missing/no team_side): "
          f"{oracle_ambig + oracle_no_ball + oracle_no_ts}")
    print()
    print(f"ORACLE CEILING: +{oracle_lift_pp:.2f} pp "
          f"({baseline_acc:.1%} → {oracle_max_acc:.1%})")
    print()
    gate_msg = (
        "PURSUE Phase 4 soft γ integration"
        if oracle_lift_pp >= 5.0
        else "BORDERLINE — consider tighter gate"
        if oracle_lift_pp >= 3.0
        else "CLOSE workstream — lift too thin"
    )
    print(f"GATE (≥5pp pursue, 3-5pp borderline, <3pp close): {gate_msg}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "aggregate": dict(agg),
        "per_video": {k: dict(v) for k, v in per_video_stats.items()},
        "baseline_accuracy": baseline_acc,
        "oracle_max_accuracy": oracle_max_acc,
        "oracle_lift_pp": oracle_lift_pp,
        "gate_verdict": gate_msg,
        "oracle_details_count": len(oracle_details),
    }
    (OUT_DIR / "attribution_ballside_oracle_2026_04_23.json").write_text(
        json.dumps(result, indent=2),
    )
    (OUT_DIR / "attribution_ballside_oracle_2026_04_23_details.jsonl").write_text(
        "\n".join(json.dumps(d) for d in oracle_details),
    )

    # Markdown summary
    lines = []
    lines.append("# Attribution ball-side oracle ceiling (2026-04-23)\n\n")
    lines.append(f"**Baseline accuracy:** {baseline_acc:.1%} ({correct}/{n_total})\n")
    lines.append(f"**Oracle ceiling:** {oracle_max_acc:.1%} "
                 f"(+{oracle_lift_pp:.2f} pp)\n")
    lines.append(f"**Gate verdict:** {gate_msg}\n\n")
    lines.append("## Breakdown\n\n")
    lines.append(f"- Wrong (any reason): {wrong}\n")
    lines.append(f"  - Same-team swap (ball-side can't help): {wrong_same}\n")
    lines.append(f"  - Cross-team miss: {wrong_cross}\n")
    lines.append(f"    - **Oracle-lift-eligible: {lift}**\n")
    lines.append(f"    - Ball on pipeline's side (oracle wouldn't help): {oracle_wrong_side}\n")
    lines.append(f"    - Ball ambiguous/missing: "
                 f"{oracle_ambig + oracle_no_ball + oracle_no_ts}\n\n")
    lines.append("## Per-video\n\n| video | n_gt | correct | wrong | "
                 "cross-team | oracle-lift-eligible |\n"
                 "|---|---:|---:|---:|---:|---:|\n")
    for vshort, vs in per_video_stats.items():
        n = vs.get("n_gt", 0)
        c = vs.get("correct", 0)
        w = vs.get("wrong", 0)
        ct = vs.get("wrong_cross_team", 0)
        oe = vs.get("oracle_lift_eligible", 0)
        lines.append(f"| {vshort} | {n} | {c} | {w} | {ct} | {oe} |\n")
    (OUT_DIR / "attribution_ballside_oracle_2026_04_23.md").write_text("".join(lines))
    print()
    print(f"wrote analysis/reports/attribution_ballside_oracle_2026_04_23.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
