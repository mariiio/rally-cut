"""Forensic probe: are the user's reported chimera cases actually BoT-SORT chimeras
(different physical players sequentially in one track) OR matcher-pipeline-ordering bugs
(track is internally consistent, matcher just assigned the wrong PID)?

Two-signal verdict per suspect track T_a:
  1. Within-track late-vs-early APPEARANCE distance, compared to the same metric
     on KNOWN-CLEAN tracks in the same rally.
     - High distance (well above clean baseline) → chimera (appearance changes mid-track).
     - Low distance (near clean baseline) → consistent track (matcher bug).

  2. POSITION trajectory at the candidate split frame (late-arriver's first frame).
     - Big position jump >> natural velocity → BoT-SORT swapped track to a new player.
     - Smooth trajectory → no track-id reuse event.

If both signals fire → genuine BoT-SORT chimera (within-rally split fix is the right path).
If both signals are clean → matcher-pipeline-ordering bug (different fix entirely).
If they disagree → escalate, deeper investigation needed.

Read-only — does not modify DB. Reproduces the same crop-extraction pipeline the
matcher uses.

Usage:
    uv run python scripts/forensic_user_chimera_cases.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_rallies_for_video,
)
from rallycut.tracking.match_tracker import extract_rally_appearances
from rallycut.tracking.player_features import (
    PlayerAppearanceProfile,
    TrackAppearanceStats,
    compute_appearance_similarity,
    compute_track_similarity,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.reid_general import GeneralReIDModel, WEIGHTS_PATH


# User-cited cases. (video_id, rally_id, suspect_tracks, label)
CASES = [
    (
        "5c756c41-1cc1-4486-a95c-97398912cfbe",
        "21d4cdf6-ca17-4a4a-a1a8-d567f4fa1d76",
        [1, 2],   # Memo says T1 (Rule A) + T2 (Rule B) suspects.
        "5c756c41 r10 — only P3+P4 visible, P1 occluded by P3, P2 off-camera serve",
    ),
    (
        "7d77980f-3006-40e0-adc0-db491a5bb659",
        "09553ef1-1b53-4756-9c29-a74db1f1a29e",
        [1],      # Memo says T1.
        "7d77980f r02 — only P1+P3 visible, P2 occluded by P1, P4 off-camera serve",
    ),
]


def to_player_positions(raw: list[dict[str, Any]], track_id: int) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=int(q["frameNumber"]),
            track_id=int(q["trackId"]),
            x=float(q["x"]),
            y=float(q["y"]),
            width=float(q.get("width", 0.0)),
            height=float(q.get("height", 0.0)),
            confidence=float(q.get("confidence", 1.0)),
        )
        for q in raw
        if int(q["trackId"]) == track_id
    ]


def load_pre_remap(rally_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pre_remap_state_json FROM player_tracks WHERE rally_id = %s",
            [rally_id],
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return []
    snap = row[0]
    if isinstance(snap, str):
        snap = json.loads(snap)
    return cast(list[dict[str, Any]], snap.get("positions") or [])


def load_match_state(video_id: str) -> tuple[dict[int, dict], dict[str, dict[int, int]]]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        return {}, {}
    ma = row[0]
    if isinstance(ma, str):
        ma = json.loads(ma)
    profiles_raw = ma.get("playerProfiles") or {}
    profiles: dict[int, PlayerAppearanceProfile] = {}
    for pid_str, p in profiles_raw.items():
        pid = int(pid_str)
        profiles[pid] = PlayerAppearanceProfile.from_dict({**p, "player_id": pid})
    afm_by_rally: dict[str, dict[int, int]] = {}
    for entry in ma.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id")
        if not rid:
            continue
        src = entry.get("appliedFullMapping") or entry.get("trackToPlayer") or {}
        afm_by_rally[rid] = {
            int(k): int(v) for k, v in src.items() if int(k) > 0
        }
    return profiles, afm_by_rally  # type: ignore[return-value]


def position_trajectory_summary(
    positions: list[dict[str, Any]],
    track_id: int,
    split_frame: int,
) -> dict[str, float]:
    """Position-jump signal at split_frame.

    Returns:
      pre_x, pre_y: track's mean pos in last K frames before split_frame
      post_x, post_y: track's mean pos in first K frames at/after split_frame
      jump: euclidean distance between pre and post centroids
      per_frame_velocity: median per-frame movement magnitude over track lifetime
      jump_to_velocity_ratio: jump / max(per_frame_velocity, ε)
    """
    pts = sorted(
        [q for q in positions if int(q["trackId"]) == track_id],
        key=lambda q: int(q["frameNumber"]),
    )
    if not pts:
        return {}
    K = 10
    pre = [q for q in pts if int(q["frameNumber"]) < split_frame][-K:]
    post = [q for q in pts if int(q["frameNumber"]) >= split_frame][:K]
    if not pre or not post:
        return {}
    pre_x = float(np.mean([float(q["x"]) for q in pre]))
    pre_y = float(np.mean([float(q["y"]) for q in pre]))
    post_x = float(np.mean([float(q["x"]) for q in post]))
    post_y = float(np.mean([float(q["y"]) for q in post]))
    jump = float(np.hypot(post_x - pre_x, post_y - pre_y))
    # Per-frame movement magnitude (track's natural-motion scale).
    velocities: list[float] = []
    for a, b in zip(pts, pts[1:]):
        if int(b["frameNumber"]) - int(a["frameNumber"]) != 1:
            continue
        velocities.append(
            float(np.hypot(float(b["x"]) - float(a["x"]),
                           float(b["y"]) - float(a["y"])))
        )
    median_v = float(np.median(velocities)) if velocities else 0.0
    return {
        "pre_x": pre_x, "pre_y": pre_y,
        "post_x": post_x, "post_y": post_y,
        "jump": jump,
        "median_velocity_per_frame": median_v,
        "jump_in_velocities": jump / median_v if median_v > 1e-6 else float("inf"),
    }


def find_split_frame(positions: list[dict[str, Any]]) -> int:
    """Earliest late-arriver's first frame."""
    by_tid: dict[int, list[int]] = defaultdict(list)
    for q in positions:
        tid = int(q["trackId"])
        if tid <= 0:
            continue
        by_tid[tid].append(int(q["frameNumber"]))
    f_min = min(min(fns) for fns in by_tid.values())
    EARLY = 30
    track_lens = sorted(by_tid.items(), key=lambda kv: -len(kv[1]))
    top4 = [tid for tid, _ in track_lens[:4]]
    early = [tid for tid in top4 if min(by_tid[tid]) <= f_min + EARLY]
    late = [tid for tid in top4 if min(by_tid[tid]) > f_min + EARLY]
    if not late:
        return f_min + 100
    return min(min(by_tid[tid]) for tid in late)


def main() -> None:
    print("Loading ReID model...")
    reid = GeneralReIDModel(weights_path=WEIGHTS_PATH)

    for video_id, rally_id, suspects, label in CASES:
        print(f"\n{'=' * 78}")
        print(f"=== {label}")
        print(f"=== video={video_id}  rally={rally_id}")
        print(f"{'=' * 78}")

        rallies = load_rallies_for_video(video_id)
        rally = next((r for r in rallies if r.rally_id == rally_id), None)
        if rally is None:
            print("  rally not found")
            continue

        positions = load_pre_remap(rally_id)
        if not positions:
            print("  no pre_remap_state_json")
            continue

        split_frame = find_split_frame(positions)
        f_min = min(int(q["frameNumber"]) for q in positions if int(q["trackId"]) > 0)
        f_max = max(int(q["frameNumber"]) for q in positions if int(q["trackId"]) > 0)
        print(f"  frame range [{f_min}, {f_max}]  split_frame={split_frame}")

        # Top-4 tracks by frame count (suspect + control set).
        by_tid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for q in positions:
            tid = int(q["trackId"])
            if tid > 0:
                by_tid[tid].append(q)
        top4 = sorted(by_tid, key=lambda t: -len(by_tid[t]))[:4]
        print(f"  top-4 tracks by frame count: {top4}")

        profiles, afm_by_rally = load_match_state(video_id)
        afm = afm_by_rally.get(rally_id, {})
        print(f"  matcher AFM: {afm}")

        video_path = get_video_path(video_id)
        if video_path is None:
            print("  no video file")
            continue

        # For each top-4 track, compute pre/post within-track appearance + position summary.
        # Suspect tracks (early) are flagged; remaining top-4 act as in-rally controls.
        per_track: list[dict[str, Any]] = []
        for tid in top4:
            track_positions = sorted(by_tid[tid], key=lambda q: int(q["frameNumber"]))
            n_frames = len(track_positions)
            first_frame = int(track_positions[0]["frameNumber"])
            last_frame = int(track_positions[-1]["frameNumber"])
            is_early = first_frame <= f_min + 30
            is_suspect = tid in suspects

            pre = [q for q in track_positions if int(q["frameNumber"]) < split_frame]
            post = [q for q in track_positions if int(q["frameNumber"]) >= split_frame]
            if len(pre) < 8 or len(post) < 8:
                print(f"  T{tid}: SKIP (pre={len(pre)} post={len(post)} < 8)")
                continue

            pre_player_positions = [
                PlayerPosition(
                    frame_number=int(q["frameNumber"]),
                    track_id=int(q["trackId"]),
                    x=float(q["x"]), y=float(q["y"]),
                    width=float(q.get("width", 0.0)),
                    height=float(q.get("height", 0.0)),
                    confidence=float(q.get("confidence", 1.0)),
                )
                for q in pre
            ]
            post_player_positions = [
                PlayerPosition(
                    frame_number=int(q["frameNumber"]),
                    track_id=int(q["trackId"]),
                    x=float(q["x"]), y=float(q["y"]),
                    width=float(q.get("width", 0.0)),
                    height=float(q.get("height", 0.0)),
                    confidence=float(q.get("confidence", 1.0)),
                )
                for q in post
            ]
            pre_ts = extract_rally_appearances(
                video_path=video_path, positions=pre_player_positions,
                primary_track_ids=[tid],
                start_ms=rally.start_ms, end_ms=rally.end_ms,
                num_samples=12, extract_reid=True, reid_model=reid,
            )
            post_ts = extract_rally_appearances(
                video_path=video_path, positions=post_player_positions,
                primary_track_ids=[tid],
                start_ms=rally.start_ms, end_ms=rally.end_ms,
                num_samples=12, extract_reid=True, reid_model=reid,
            )
            if tid not in pre_ts or tid not in post_ts:
                print(f"  T{tid}: extract failed")
                continue

            # SIGNAL 1: within-track late-vs-early appearance distance.
            within_track_d = float(
                compute_track_similarity(pre_ts[tid], post_ts[tid], reid_blend=0.5)
            )

            # SIGNAL 2: position trajectory at split frame.
            pos = position_trajectory_summary(positions, tid, split_frame)

            # Distance to cross-rally profiles, both halves.
            if profiles:
                pre_dist = {pid: float(compute_appearance_similarity(profiles[pid], pre_ts[tid]))
                            for pid in profiles}
                post_dist = {pid: float(compute_appearance_similarity(profiles[pid], post_ts[tid]))
                             for pid in profiles}
                pre_best = min(pre_dist, key=lambda p: pre_dist[p])
                post_best = min(post_dist, key=lambda p: post_dist[p])
            else:
                pre_dist, post_dist = {}, {}
                pre_best, post_best = -1, -1

            per_track.append({
                "track_id": tid,
                "is_suspect": is_suspect,
                "is_early": is_early,
                "n_pre": len(pre), "n_post": len(post),
                "first_frame": first_frame, "last_frame": last_frame,
                "matcher_pid": afm.get(tid),
                "within_track_distance": within_track_d,
                "position": pos,
                "pre_best_pid": pre_best, "post_best_pid": post_best,
                "pre_distances": pre_dist, "post_distances": post_dist,
            })

        # Compute baseline within-track distance from non-suspect tracks.
        controls = [r for r in per_track if not r["is_suspect"]]
        suspects_data = [r for r in per_track if r["is_suspect"]]
        if controls:
            ctrl_d = [r["within_track_distance"] for r in controls]
            print(f"\n  CONTROL within-track distances "
                  f"(non-suspect top-4): "
                  f"min={min(ctrl_d):.3f}  median={float(np.median(ctrl_d)):.3f}  "
                  f"max={max(ctrl_d):.3f}")

        print()
        for r in per_track:
            mark = "*** SUSPECT ***" if r["is_suspect"] else "    control   "
            tid = r["track_id"]
            pos = r["position"]
            pos_summary = (
                f"jump={pos.get('jump', 0):.3f}u  "
                f"velocity={pos.get('median_velocity_per_frame', 0):.4f}u/f  "
                f"jump/v={pos.get('jump_in_velocities', 0):.1f}×"
            ) if pos else "(insufficient pos data)"
            print(f"  {mark}  T{tid}  matcher_pid=P{r['matcher_pid']}  "
                  f"early_arrived={r['is_early']}  frames=[{r['first_frame']},{r['last_frame']}]")
            print(f"      within-track late-vs-early appearance dist = "
                  f"{r['within_track_distance']:.3f}")
            print(f"      position trajectory at split={pos_summary}")
            if r["pre_distances"]:
                print(f"      cross-rally profile match: "
                      f"pre best=P{r['pre_best_pid']} d={r['pre_distances'][r['pre_best_pid']]:.3f}  "
                      f"post best=P{r['post_best_pid']} d={r['post_distances'][r['post_best_pid']]:.3f}")

        # Verdict.
        if controls and suspects_data:
            ctrl_d = [r["within_track_distance"] for r in controls]
            ctrl_max = max(ctrl_d)
            ctrl_median = float(np.median(ctrl_d))
            print(f"\n  ====== VERDICT ======")
            for r in suspects_data:
                tid = r["track_id"]
                wd = r["within_track_distance"]
                pos = r["position"]
                jump_v = pos.get("jump_in_velocities", 0)
                appearance_anomaly = wd > ctrl_max  # exceeds the control max
                appearance_above_median = wd > 1.5 * ctrl_median
                position_anomaly = jump_v > 5.0  # >5x natural velocity
                if appearance_anomaly and position_anomaly:
                    verdict = "GENUINE BOT-SORT CHIMERA"
                elif not appearance_above_median and not position_anomaly:
                    verdict = "INTERNALLY CONSISTENT TRACK (matcher-pipeline bug, not chimera)"
                else:
                    verdict = (
                        f"AMBIGUOUS — appearance_anomaly={appearance_anomaly} "
                        f"appearance_above_median_x1.5={appearance_above_median} "
                        f"position_anomaly={position_anomaly}"
                    )
                print(f"  T{tid}: {verdict}")
                print(f"     within-track d={wd:.3f} (ctrl_median={ctrl_median:.3f}, "
                      f"ctrl_max={ctrl_max:.3f})")
                print(f"     position jump={jump_v:.1f}× natural velocity")


if __name__ == "__main__":
    main()
