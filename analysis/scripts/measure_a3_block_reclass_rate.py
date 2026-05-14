"""Fleet-wide A3 BLOCK reclassification rate measurement.

In-process — re-runs contacts + classification + A3 pass on every tracked
rally and counts ATTACKs reclassified to BLOCK. NO DB writes.

Ship gate: rate ≤ 3% (per spec — heuristic is not too aggressive).

Usage:
    cd analysis
    uv run python scripts/measure_a3_block_reclass_rate.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, cast

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import get_connection, load_court_calibration
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.tracking.action_classifier import (
    ActionType,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.block_reclassification import estimate_net_y_image
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the pose extractor from redetect_all_actions.py.
spec = importlib.util.spec_from_file_location(
    "redetect", HERE / "redetect_all_actions.py"
)
assert spec and spec.loader
red = importlib.util.module_from_spec(spec)
spec.loader.exec_module(red)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure A3 reclassification rate fleet-wide",
    )
    parser.add_argument("--video", type=str, help="Only this video")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Stop after N rallies (0 = all)",
    )
    args = parser.parse_args()

    where_clauses = ["pt.ball_positions_json IS NOT NULL"]
    params: list[Any] = []
    if args.video:
        where_clauses.append("r.video_id = %s")
        params.append(args.video)
    where_sql = " AND ".join(where_clauses)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT v.id, v.match_analysis_json FROM videos v "
            "WHERE v.match_analysis_json IS NOT NULL"
        )
        match_teams_by_rally: dict[str, dict[int, int]] = {}
        for vid, mj_raw in cur.fetchall():
            mj = cast(dict[str, Any], mj_raw)
            if not mj:
                continue
            match_teams_by_rally.update(
                build_match_team_assignments(mj, min_confidence=0.0)
            )

        cur.execute(f"""
            SELECT r.id, r.video_id, pt.id, pt.ball_positions_json,
                   pt.positions_json, pt.frame_count, pt.court_split_y,
                   r.start_ms, v.name
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            JOIN videos v ON v.id = r.video_id
            WHERE {where_sql}
            ORDER BY r.video_id, r.start_ms
        """, params)
        rows = cur.fetchall()

    if args.limit:
        rows = rows[: args.limit]

    print(f"Fleet measurement: {len(rows)} rallies")

    calibrators: dict[str, CourtCalibrator | None] = {}
    net_y_by_video: dict[str, float | None] = {}
    video_meta_cache: dict[str, dict[str, Any]] = {}
    resolver: VideoResolver | None = None

    total_attacks = 0
    total_reclassified = 0
    per_rally: list[dict[str, Any]] = []
    t_start = time.monotonic()
    errors = 0

    for i, row in enumerate(rows):
        rally_id = str(row[0])
        video_id = str(row[1])
        ball_json = cast(list[dict[str, Any]], row[3])
        positions_json = cast(list[dict[str, Any]] | None, row[4])
        frame_count = cast(int | None, row[5])
        court_split_y = cast(float | None, row[6])
        rally_start_ms = int(cast(int | None, row[7]) or 0)
        video_name = str(row[8])

        if video_id not in calibrators:
            corners = load_court_calibration(video_id)
            if corners and len(corners) == 4:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])
                calibrators[video_id] = cal
                net_y_by_video[video_id] = estimate_net_y_image(corners)
            else:
                calibrators[video_id] = None
                net_y_by_video[video_id] = None

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in ball_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]
        if not ball_positions:
            continue
        player_positions = []
        if positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in positions_json
            ]

        match_teams = match_teams_by_rally.get(rally_id)

        try:
            cal_obj = calibrators.get(video_id)
            seq = get_sequence_probs(
                ball_positions, player_positions, court_split_y,
                frame_count or 0, match_teams, calibrator=cal_obj,
            )
            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=court_split_y,
                frame_count=frame_count or None,
                court_calibrator=cal_obj,
                team_assignments=match_teams,
                sequence_probs=seq,
            )
            if not contacts.contacts:
                continue

            # Resolve video for pose
            net_y_image_norm = net_y_by_video.get(video_id)
            if video_id not in video_meta_cache:
                with get_connection() as vconn, vconn.cursor() as vcur:
                    vcur.execute(
                        "SELECT fps, s3_key, proxy_s3_key, processed_s3_key, "
                        "content_hash FROM videos WHERE id = %s",
                        (video_id,),
                    )
                    vrow = vcur.fetchone()
                if vrow is not None:
                    video_meta_cache[video_id] = {
                        "fps": float(cast(float, vrow[0])) if vrow[0] is not None else 30.0,
                        "s3_key": vrow[1],
                        "proxy_s3_key": vrow[2],
                        "processed_s3_key": vrow[3],
                        "content_hash": vrow[4],
                    }
            vmeta = video_meta_cache.get(video_id)
            video_path: Path | None = None
            if vmeta and vmeta.get("content_hash"):
                if resolver is None:
                    resolver = VideoResolver()
                for label_key in ("proxy_s3_key", "s3_key", "processed_s3_key"):
                    sk = vmeta.get(label_key)
                    if not sk:
                        continue
                    try:
                        video_path = resolver.resolve(
                            sk, vmeta["content_hash"]
                        )
                        break
                    except Exception:
                        continue
            pose_wrist: dict[tuple[int, int], float] | None = None
            if video_path is not None:
                contact_frames = sorted({c.frame for c in contacts.contacts})
                pose_wrist = red._extract_wrist_y_for_contacts(
                    video_path=video_path,
                    rally_start_ms=rally_start_ms,
                    fps=(vmeta or {}).get("fps", 30.0),
                    contact_frames=contact_frames,
                    player_positions=player_positions,
                )

            # Run classification with A3 OFF first (baseline)
            import os
            os.environ["USE_BLOCK_RECLASSIFICATION"] = "0"
            ra_off = classify_rally_actions(
                contacts, rally_id, use_classifier=True,
                team_assignments=match_teams,
                match_team_assignments=match_teams,
                sequence_probs=seq,
                net_y_image=net_y_image_norm,
                pose_wrist_by_frame_tid=pose_wrist,
            )
            n_attacks_off = sum(
                1 for a in ra_off.actions if a.action_type == ActionType.ATTACK
            )

            # Run again with A3 ON
            os.environ["USE_BLOCK_RECLASSIFICATION"] = "1"
            ra_on = classify_rally_actions(
                contacts, rally_id, use_classifier=True,
                team_assignments=match_teams,
                match_team_assignments=match_teams,
                sequence_probs=seq,
                net_y_image=net_y_image_norm,
                pose_wrist_by_frame_tid=pose_wrist,
            )
            n_attacks_on = sum(
                1 for a in ra_on.actions if a.action_type == ActionType.ATTACK
            )
            n_blocks_on = sum(
                1 for a in ra_on.actions if a.action_type == ActionType.BLOCK
            )
            n_blocks_off = sum(
                1 for a in ra_off.actions if a.action_type == ActionType.BLOCK
            )
            # Reclassified = those that were ATTACK in OFF but BLOCK in ON
            # on the same frame.
            off_frames_to_type = {a.frame: a.action_type for a in ra_off.actions}
            reclassified_frames = [
                a.frame for a in ra_on.actions
                if a.action_type == ActionType.BLOCK
                and off_frames_to_type.get(a.frame) == ActionType.ATTACK
            ]

            total_attacks += n_attacks_off
            n_reclass = len(reclassified_frames)
            total_reclassified += n_reclass
            per_rally.append({
                "rally_id": rally_id,
                "video": video_name,
                "n_attacks_off": n_attacks_off,
                "n_blocks_off": n_blocks_off,
                "n_attacks_on": n_attacks_on,
                "n_blocks_on": n_blocks_on,
                "reclassified_frames": reclassified_frames,
            })
            elapsed = time.monotonic() - t_start
            if n_reclass > 0:
                print(
                    f"  [{i+1}/{len(rows)}] {video_name}/{rally_id[:8]}: "
                    f"attacks_off={n_attacks_off} reclassified={n_reclass} "
                    f"frames={reclassified_frames} ({elapsed:.1f}s)"
                )
            else:
                print(
                    f"  [{i+1}/{len(rows)}] {video_name}/{rally_id[:8]}: "
                    f"attacks_off={n_attacks_off} reclassified=0 ({elapsed:.1f}s)"
                )

        except Exception as e:
            errors += 1
            print(f"  ERROR {rally_id[:8]}: {e}")

    elapsed = time.monotonic() - t_start
    print()
    print("=" * 60)
    print(f"Total rallies processed: {len(per_rally)}")
    print(f"Errors:                  {errors}")
    print(f"Total ATTACKs (A3 off):  {total_attacks}")
    print(f"Total reclassified:      {total_reclassified}")
    if total_attacks > 0:
        rate = total_reclassified / total_attacks * 100.0
        print(f"Rate:                    {rate:.2f}% "
              f"({total_reclassified} / {total_attacks})")
        ship = "PASS" if rate <= 3.0 else "FAIL"
        print(f"Spec gate (≤ 3%):        {ship}")
    print(f"Elapsed:                 {elapsed:.1f}s")


if __name__ == "__main__":
    main()
