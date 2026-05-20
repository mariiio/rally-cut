"""Option 3 probe: where does the 30fps Serve attribution regression come from?

Runs the pipeline twice on the 30fps subset (24 videos in trusted-31):
  - With v4 contact_classifier
  - With candidate contact_classifier (multi-fps retrain)

For each GT Serve action, compares both pipelines' predicted player to GT.
Classifies each Serve into:
  BOTH_CORRECT  — both v4 and candidate get it right
  BOTH_WRONG    — both get it wrong (often the same way)
  V4_ONLY       — v4 correct, candidate wrong  (the REGRESSION cases)
  CANDIDATE_ONLY — candidate correct, v4 wrong  (the WIN cases)

Outputs:
  - Per-video breakdown showing where regressions concentrate
  - For each V4_ONLY case: dump details (predicted players, candidates, frames,
    whether serve is synthetic, etc.)

Read-only — no DB writes. Production model files NOT touched.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import psycopg

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

# 30fps subset of trusted-31 (24 videos)
VIDEOS_30FPS = [
    "caco", "cece", "cici", "cuco", "gigi", "juju", "keke", "mame",
    "meme", "mimi", "moma", "mumu", "papa", "pepe", "pipi", "popo",
    "pupu", "titi", "veve", "koko", "toto", "gaga", "yeye", "gugu",
]

WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights" / "contact_classifier"
V4_PATH = WEIGHTS_DIR / "contact_classifier.pkl.backup_pre_2026_05_19_60fps_retrain"
CAND_PATH = WEIGHTS_DIR / "contact_classifier.pkl.candidate_2026_05_19_60fps_retrain"

MATCH_WINDOW = 10


def _run_pipeline(
    ball_positions: list, player_positions: list,
    classifier: ContactClassifier,
    court_split_y, frame_count, match_teams,
) -> list[tuple[int, int, float, float]]:
    """Run detect_contacts then classify_rally_actions; return list of
    (frame, player_track_id, ball_x, ball_y) for each Serve action.
    """
    from rallycut.tracking.action_classifier import classify_rally_actions

    seq_probs = get_sequence_probs(
        ball_positions, player_positions, court_split_y,
        frame_count or 0, match_teams,
    )
    contacts = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        frame_count=frame_count or None,
        classifier=classifier,
        team_assignments=match_teams,
        sequence_probs=seq_probs,
    )
    actions = classify_rally_actions(
        contacts, match_team_assignments=match_teams,
    )
    serves = []
    for a in actions.actions:
        if a.action_type.value == "serve":
            serves.append((a.frame, a.player_track_id, a.ball_x, a.ball_y))
    return serves


def main() -> int:
    print("Loading classifiers...", flush=True)
    v4_clf = ContactClassifier.load(str(V4_PATH))
    cand_clf = ContactClassifier.load(str(CAND_PATH))
    print(f"  v4 loaded from {V4_PATH.name}", flush=True)
    print(f"  candidate loaded from {CAND_PATH.name}", flush=True)
    print()

    with psycopg.connect(DB_DSN) as conn:
        # Load rally data + GT serves
        placeholders = ",".join(["%s"] * len(VIDEOS_30FPS))
        cur = conn.execute(
            f"""
            SELECT r.id, v.name, v.id AS vid,
                   pt.ball_positions_json, pt.positions_json,
                   pt.court_split_y, pt.frame_count
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name IN ({placeholders})
              AND pt.ball_positions_json IS NOT NULL
              AND pt.positions_json IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM rally_action_ground_truth gt
                WHERE gt.rally_id = r.id AND gt.action = 'SERVE'
                  AND gt.resolved_track_id IS NOT NULL
              )
            ORDER BY v.name, r.start_ms
            """,
            VIDEOS_30FPS,
        )
        rallies = cur.fetchall()
        print(f"Loaded {len(rallies)} rallies with GT serves", flush=True)

        # Load GT serves per rally
        cur2 = conn.execute(
            f"""
            SELECT gt.rally_id, gt.frame, gt.resolved_track_id
            FROM rally_action_ground_truth gt
            JOIN rallies r ON gt.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            WHERE v.name IN ({placeholders})
              AND gt.action = 'SERVE'
              AND gt.resolved_track_id IS NOT NULL
            """,
            VIDEOS_30FPS,
        )
        gt_serves: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for rid, frame, tid in cur2.fetchall():
            gt_serves[str(rid)].append((frame, tid))

        # Load match team assignments
        rally_positions: dict = {}
        cur3 = conn.execute(
            "SELECT rally_id, positions_json FROM player_tracks "
            "WHERE positions_json IS NOT NULL",
        )
        for rid_raw, pos_raw in cur3.fetchall():
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
        cur4 = conn.execute(
            f"SELECT id, match_analysis_json FROM videos "
            f"WHERE name IN ({placeholders}) AND match_analysis_json IS NOT NULL",
            VIDEOS_30FPS,
        )
        match_teams_by_rally: dict = {}
        for _vid, mj_raw in cur4.fetchall():
            if not mj_raw:
                continue
            match_teams_by_rally.update(
                build_match_team_assignments(
                    mj_raw, min_confidence=0.0, rally_positions=rally_positions,
                )
            )

    per_video_class: dict[str, Counter] = defaultdict(Counter)
    v4_only_failures: list[dict] = []

    for i, (rid, vname, vid, bj, pj, court_split_y, frame_count) in enumerate(rallies):
        rid_s = str(rid)
        gt_list = gt_serves.get(rid_s, [])
        if not gt_list:
            continue

        bj_list = bj if isinstance(bj, list) else json.loads(bj or '[]')
        pj_list = pj if isinstance(pj, list) else json.loads(pj or '[]')
        ball_positions = [
            BallPosition(
                frame_number=b["frameNumber"], x=b["x"], y=b["y"],
                confidence=b.get("confidence", 1.0),
            )
            for b in bj_list
        ]
        player_positions = [
            PlayerPosition(
                frame_number=p["frameNumber"], track_id=p["trackId"],
                x=p["x"], y=p["y"], width=p["width"], height=p["height"],
                confidence=p.get("confidence", 1.0),
                keypoints=p.get("keypoints"),
            )
            for p in pj_list
        ]
        match_teams = match_teams_by_rally.get(rid_s)

        try:
            v4_serves = _run_pipeline(
                ball_positions, player_positions, v4_clf,
                court_split_y, frame_count, match_teams,
            )
            cand_serves = _run_pipeline(
                ball_positions, player_positions, cand_clf,
                court_split_y, frame_count, match_teams,
            )
        except Exception as e:
            print(f"  rally {rid_s[:8]} ({vname}) FAIL: {e}", flush=True)
            continue

        for gt_frame, gt_tid in gt_list:
            # Find matched serve in each pipeline output
            v4_match = next(
                ((f, tid, bx, by) for f, tid, bx, by in v4_serves
                 if abs(f - gt_frame) <= MATCH_WINDOW),
                None,
            )
            cand_match = next(
                ((f, tid, bx, by) for f, tid, bx, by in cand_serves
                 if abs(f - gt_frame) <= MATCH_WINDOW),
                None,
            )
            v4_correct = v4_match is not None and v4_match[1] == gt_tid
            cand_correct = cand_match is not None and cand_match[1] == gt_tid

            if v4_correct and cand_correct:
                cls = "BOTH_CORRECT"
            elif not v4_correct and not cand_correct:
                cls = "BOTH_WRONG"
            elif v4_correct and not cand_correct:
                cls = "V4_ONLY"  # regression
            else:
                cls = "CANDIDATE_ONLY"  # win

            per_video_class[vname][cls] += 1

            if cls == "V4_ONLY":
                v4_only_failures.append({
                    "video": vname, "rally": rid_s[:8], "gt_frame": gt_frame,
                    "gt_player": gt_tid,
                    "v4": v4_match, "candidate": cand_match,
                })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rallies)}] processed", flush=True)

    print()
    print("Per-video breakdown:")
    print(f"{'video':<10} {'both_ok':>8} {'both_wrong':>11} {'v4_only(REG)':>13} {'cand_only(WIN)':>15} {'total':>6}")
    total = Counter()
    for v in sorted(per_video_class.keys()):
        c = per_video_class[v]
        tot = sum(c.values())
        print(
            f"{v:<10} {c.get('BOTH_CORRECT', 0):>8} {c.get('BOTH_WRONG', 0):>11} "
            f"{c.get('V4_ONLY', 0):>13} {c.get('CANDIDATE_ONLY', 0):>15} {tot:>6}"
        )
        for k, val in c.items():
            total[k] += val
    print(
        f"{'TOTAL':<10} {total.get('BOTH_CORRECT', 0):>8} {total.get('BOTH_WRONG', 0):>11} "
        f"{total.get('V4_ONLY', 0):>13} {total.get('CANDIDATE_ONLY', 0):>15} "
        f"{sum(total.values()):>6}"
    )
    print()
    n_reg = total.get('V4_ONLY', 0)
    n_win = total.get('CANDIDATE_ONLY', 0)
    n_tot = sum(total.values())
    print(f"NET (cand wins - regressions): {n_win - n_reg} ({(n_win - n_reg)/n_tot*100:+.1f}pp)")
    print()

    print(f"V4_ONLY (regression) cases — {len(v4_only_failures)} total:")
    for f in v4_only_failures:
        v4 = f["v4"]
        cand = f["candidate"]
        v4_str = f"frame={v4[0]} pid={v4[1]} ball=({v4[2]:.3f},{v4[3]:.3f})" if v4 else "NO_MATCH"
        cand_str = f"frame={cand[0]} pid={cand[1]} ball=({cand[2]:.3f},{cand[3]:.3f})" if cand else "NO_MATCH"
        # Mark synthetic if ballX == 0.5 exactly
        v4_synth = " [SYNTH]" if v4 and abs(v4[2] - 0.5) < 1e-9 else ""
        cand_synth = " [SYNTH]" if cand and abs(cand[2] - 0.5) < 1e-9 else ""
        print(
            f"  {f['video']:<8} {f['rally']} GT_frame={f['gt_frame']:>4} GT_pid={f['gt_player']}\n"
            f"    v4 :  {v4_str}{v4_synth}\n"
            f"    cand: {cand_str}{cand_synth}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
