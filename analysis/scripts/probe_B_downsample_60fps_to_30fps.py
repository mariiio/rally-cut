"""Probe B: downsample 60fps rallies to simulated-30fps and compare contact detection.

For each kuku/lulu/wawa rally with GT actions:
  1. Native run: detect_contacts on real 60fps ball/player positions, fps=59.94
  2. Downsampled run: keep only even-frame positions, re-index to half frame numbers,
     run detect_contacts with fps=30
  3. Compare against GT: contact recall, attribution match rate

If downsampled-30fps outperforms native-60fps on the SAME content, the cause IS fps.
If they perform equivalently, fps isn't the differential cause.

This is the cleanest experiment to separate "fps causation" from "video coincidence."
Read-only — no DB writes.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import psycopg

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
VIDEOS = ["kuku", "lulu", "wawa"]
MATCH_WINDOW = 10  # GT contact matched if within ±N frames of detected


def downsample(positions: list, target_attr: str = "frame_number") -> list:
    """Keep only even-numbered frames; re-index frame_number to frame_number // 2.

    Mimics what the video would look like if every other frame were dropped
    (capturing at half the temporal rate).
    """
    out = []
    for p in positions:
        f = getattr(p, target_attr)
        if f % 2 == 0:
            # Create a copy with halved frame
            if isinstance(p, BallPosition):
                out.append(BallPosition(
                    frame_number=f // 2, x=p.x, y=p.y, confidence=p.confidence,
                ))
            elif isinstance(p, PlayerPosition):
                out.append(PlayerPosition(
                    frame_number=f // 2, track_id=p.track_id,
                    x=p.x, y=p.y, width=p.width, height=p.height,
                    confidence=p.confidence, keypoints=p.keypoints,
                ))
    return out


def main() -> int:
    with psycopg.connect(DB_DSN) as conn:
        # Load GT for each 60fps video
        placeholders = ",".join(["%s"] * len(VIDEOS))
        cur = conn.execute(
            f"""
            SELECT v.id AS vid, v.name, r.id AS rid, COALESCE(pt.fps, v.fps) AS fps,
                   pt.ball_positions_json, pt.positions_json,
                   pt.court_split_y, pt.frame_count
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name IN ({placeholders})
              AND pt.ball_positions_json IS NOT NULL
              AND pt.positions_json IS NOT NULL
            """,
            VIDEOS,
        )
        rally_data = cur.fetchall()
        print(f"Loaded {len(rally_data)} rallies", flush=True)

        # Load GT actions per rally
        cur2 = conn.execute(
            f"""
            SELECT gt.rally_id, gt.frame
            FROM rally_action_ground_truth gt
            JOIN rallies r ON gt.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            WHERE v.name IN ({placeholders})
              AND gt.resolved_track_id IS NOT NULL
            """,
            VIDEOS,
        )
        gt_by_rally: dict[str, list[int]] = {}
        for rid, frame in cur2.fetchall():
            gt_by_rally.setdefault(str(rid), []).append(frame)

        # Load match team assignments
        cur3 = conn.execute(
            f"SELECT id, match_analysis_json FROM videos "
            f"WHERE name IN ({placeholders}) AND match_analysis_json IS NOT NULL",
            VIDEOS,
        )
        rally_positions: dict[str, list[PlayerPosition]] = {}
        cur4 = conn.execute(
            "SELECT rally_id, positions_json FROM player_tracks "
            "WHERE positions_json IS NOT NULL",
        )
        for rid_raw, pos_raw in cur4.fetchall():
            rid_s = str(rid_raw)
            pos_list = pos_raw if isinstance(pos_raw, list) else []
            rally_positions[rid_s] = [
                PlayerPosition(
                    frame_number=p.get("frameNumber", 0),
                    track_id=p.get("trackId", 0),
                    x=p.get("x", 0), y=p.get("y", 0),
                    width=p.get("width", 0), height=p.get("height", 0),
                    confidence=p.get("confidence", 0),
                )
                for p in pos_list if isinstance(p, dict)
            ]
        match_teams_by_rally: dict[str, dict[int, int]] = {}
        for _vid, mj_raw in cur3.fetchall():
            if not mj_raw:
                continue
            match_teams_by_rally.update(
                build_match_team_assignments(
                    mj_raw, min_confidence=0.0, rally_positions=rally_positions,
                )
            )

    results = {"native": Counter(), "downsampled": Counter()}
    per_rally = []

    for vid, vname, rid, fps, bj, pj, court_split_y, frame_count in rally_data:
        rid_s = str(rid)
        gt_frames = gt_by_rally.get(rid_s, [])
        if not gt_frames:
            continue

        # Build typed objects
        bj_list = bj if isinstance(bj, list) else json.loads(bj or '[]')
        pj_list = pj if isinstance(pj, list) else json.loads(pj or '[]')
        ball_native = [
            BallPosition(
                frame_number=b["frameNumber"], x=b["x"], y=b["y"],
                confidence=b.get("confidence", 1.0),
            )
            for b in bj_list
        ]
        players_native = [
            PlayerPosition(
                frame_number=p["frameNumber"], track_id=p["trackId"],
                x=p["x"], y=p["y"], width=p["width"], height=p["height"],
                confidence=p.get("confidence", 1.0),
                keypoints=p.get("keypoints"),
            )
            for p in pj_list
        ]

        # Native (60fps)
        try:
            seq_native = get_sequence_probs(
                ball_native, players_native, court_split_y,
                frame_count or 0, match_teams_by_rally.get(rid_s),
            )
            contacts_native = detect_contacts(
                ball_positions=ball_native,
                player_positions=players_native,
                net_y=court_split_y,
                frame_count=frame_count or None,
                team_assignments=match_teams_by_rally.get(rid_s),
                sequence_probs=seq_native,
            )
            native_frames = [c.frame for c in contacts_native.contacts]
        except Exception as e:
            print(f"  native fail {rid_s[:8]}: {e}", flush=True)
            native_frames = []

        # Downsampled (treat as 30fps)
        ball_down = downsample(ball_native)
        players_down = downsample(players_native)
        try:
            seq_down = get_sequence_probs(
                ball_down, players_down, court_split_y,
                (frame_count or 0) // 2, match_teams_by_rally.get(rid_s),
            )
            contacts_down = detect_contacts(
                ball_positions=ball_down,
                player_positions=players_down,
                net_y=court_split_y,
                frame_count=(frame_count or 0) // 2 or None,
                team_assignments=match_teams_by_rally.get(rid_s),
                sequence_probs=seq_down,
            )
            # Map downsampled contact frames back to native frame numbering
            down_frames_native = [c.frame * 2 for c in contacts_down.contacts]
        except Exception as e:
            print(f"  downsampled fail {rid_s[:8]}: {e}", flush=True)
            down_frames_native = []

        # Score against GT
        def matched(detected: list[int], gt: list[int]) -> int:
            return sum(
                1 for g in gt
                if any(abs(d - g) <= MATCH_WINDOW for d in detected)
            )

        n_gt = len(gt_frames)
        n_native_match = matched(native_frames, gt_frames)
        n_down_match = matched(down_frames_native, gt_frames)

        results["native"]["gt"] += n_gt
        results["native"]["matched"] += n_native_match
        results["downsampled"]["gt"] += n_gt
        results["downsampled"]["matched"] += n_down_match

        per_rally.append({
            "video": vname, "rally": rid_s[:8],
            "gt": n_gt, "native_match": n_native_match, "down_match": n_down_match,
            "native_n_contacts": len(native_frames),
            "down_n_contacts": len(down_frames_native),
        })

    print()
    print(f"{'video':<8} {'rally':<10} {'gt':>3} {'native_match':>13} {'down_match':>11} "
          f"{'native_n':>9} {'down_n':>7}")
    for r in per_rally:
        print(
            f"{r['video']:<8} {r['rally']:<10} {r['gt']:>3} "
            f"{r['native_match']:>13} {r['down_match']:>11} "
            f"{r['native_n_contacts']:>9} {r['down_n_contacts']:>7}"
        )

    print()
    n = results["native"]
    d = results["downsampled"]
    print(f"AGGREGATE:")
    print(f"  Native (60fps):       {n['matched']}/{n['gt']} = {n['matched']/n['gt']*100:.1f}%")
    print(f"  Downsampled (30fps):  {d['matched']}/{d['gt']} = {d['matched']/d['gt']*100:.1f}%")
    print(f"  Delta (down - native): {(d['matched'] - n['matched'])/n['gt']*100:+.1f}pp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
