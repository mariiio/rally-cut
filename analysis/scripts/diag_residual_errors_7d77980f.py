"""Per-rally residual diagnostic for 7d77980f's 8 errors.

For each rally with GT, derives the GT-correct track→PID mapping by
finding the closest track to each GT bbox center, then compares to:
  - the matcher's chosen AFM (appliedFullMapping)
  - the global Hungarian permutation across all GT samples on this video

Reports per rally:
  - GT-derived track→PID
  - Matcher AFM
  - Global perm expectation (= what matcher SHOULD have produced for
    this rally to be consistent with the rest of the video)
  - Direct/permuted hits per GT sample

Then checks each problem rally for:
  - Side classification (near/far) of each primary track
  - Server identity (which player served)
  - Per-track Y position (helps see if seed Y-sort would land
    differently than other rallies)
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, cast

import numpy as np
from scipy.optimize import linear_sum_assignment

from rallycut.evaluation.tracking.db import get_connection, load_rallies_for_video

VIDEO_ID = "7d77980f-3006-40e0-adc0-db491a5bb659"
PROBLEM_RALLIES = {"30ffb876", "613fde44", "1f9ce33a"}


def main() -> None:
    rallies = load_rallies_for_video(VIDEO_ID)
    chronological = sorted(rallies, key=lambda r: r.start_ms)
    rid_to_idx = {r.rally_id: i + 1 for i, r in enumerate(chronological)}

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT player_matching_gt_json, match_analysis_json "
            "FROM videos WHERE id = %s", [VIDEO_ID])
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"video not found: {VIDEO_ID}")
        gt_raw, ma_raw = row
        gt = gt_raw if isinstance(gt_raw, dict) else json.loads(cast(str, gt_raw))
        ma = ma_raw if isinstance(ma_raw, dict) else json.loads(cast(str, ma_raw))

        cur.execute(
            "SELECT pt.rally_id::text, pt.pre_remap_state_json "
            "FROM player_tracks pt JOIN rallies r ON pt.rally_id = r.id "
            "WHERE r.video_id = %s", [VIDEO_ID])
        snaps: dict[str, dict[str, Any]] = {}
        for rid, snap in cur.fetchall():
            if snap is not None:
                snaps[str(rid)] = (
                    snap if isinstance(snap, dict)
                    else json.loads(cast(str, snap))
                )

    afm_by_rally: dict[str, dict[int, int]] = {}
    side_switch_by_rally: dict[str, bool] = {}
    server_by_rally: dict[str, int | None] = {}
    for entry in ma.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id")
        if not rid:
            continue
        src = entry.get("appliedFullMapping") or entry.get("trackToPlayer") or {}
        afm_by_rally[rid] = {
            int(k): int(v) for k, v in src.items() if int(k) > 0
        }
        side_switch_by_rally[rid] = bool(entry.get("sideSwitchDetected"))
        server_by_rally[rid] = entry.get("serverPlayerId")

    # Pass 1: derive GT-correct track→PID per rally (and collect for global perm)
    gt_track_to_pid: dict[str, dict[int, int]] = {}
    samples: list[tuple[str, int, int, int]] = []
    for rid, entry in (gt.get("rallies", {}) or {}).items():
        snap = snaps.get(rid)
        if not snap:
            continue
        positions = snap.get("positions") or []
        per_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for q in positions:
            fn = q.get("frameNumber")
            if fn is None:
                continue
            per_frame[int(fn)].append(q)

        rally_map: dict[int, int] = {}
        for lbl in entry.get("labels", []):
            gt_pid = int(lbl["playerId"])
            cx, cy = float(lbl["cx"]), float(lbl["cy"])
            gt_frame = int(lbl["frame"])
            cands = per_frame.get(gt_frame, [])
            if not cands:
                for delta in range(1, 10):
                    if per_frame.get(gt_frame - delta):
                        cands = per_frame[gt_frame - delta]
                        break
                    if per_frame.get(gt_frame + delta):
                        cands = per_frame[gt_frame + delta]
                        break
            if not cands:
                continue
            closest = min(cands, key=lambda p: (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2)
            tid = int(closest["trackId"])
            matcher_pid = afm_by_rally.get(rid, {}).get(tid)
            if matcher_pid is None:
                continue
            rally_map[tid] = gt_pid  # track tid IS gt_pid in reality
            samples.append((rid, gt_frame, gt_pid, matcher_pid))
        gt_track_to_pid[rid] = rally_map

    # Compute global permutation matcher_pid -> gt_pid via Hungarian
    pids = sorted({s[2] for s in samples} | {s[3] for s in samples if 1 <= s[3] <= 4})
    n = max(4, len(pids))
    confusion = np.zeros((n, n), dtype=int)
    pid_to_idx = {p: i for i, p in enumerate(pids)}
    for _, _, gt_pid, mp in samples:
        if mp in pid_to_idx and gt_pid in pid_to_idx:
            confusion[pid_to_idx[mp], pid_to_idx[gt_pid]] += 1
    row_ind, col_ind = linear_sum_assignment(-confusion)
    perm: dict[int, int] = {pids[r]: pids[c] for r, c in zip(row_ind, col_ind)
                            if r < len(pids) and c < len(pids)}
    print(f"\nGlobal permutation matcher_pid → gt_pid: {perm}\n")

    # Pass 2: per-rally report
    print(f"{'idx':>3} {'rid':>8} {'ss':>2} {'srv':>3} | "
          f"{'pre_primary':22} | {'matcher_AFM':28} | "
          f"{'GT_correct_AFM':28} | issues")
    print("-" * 140)
    for r in chronological:
        rid = r.rally_id
        rid_short = rid[:8]
        idx = rid_to_idx[rid]
        snap = snaps.get(rid)
        pre_primary = snap.get("primaryTrackIds", []) if snap else []
        afm = afm_by_rally.get(rid, {})
        gt_map = gt_track_to_pid.get(rid, {})
        ss = "Y" if side_switch_by_rally.get(rid) else "."
        srv = server_by_rally.get(rid) or "-"

        # Format AFM as sorted track→pid, only primary tracks
        primary_set = set(int(t) for t in pre_primary)
        afm_str = "{" + ", ".join(
            f"{t}→{afm.get(t, '?')}" for t in sorted(primary_set)
        ) + "}"
        gt_str = "{" + ", ".join(
            f"{t}→{gt_map[t]}" for t in sorted(gt_map)
        ) + "}" if gt_map else "(no GT)"

        # Issue: does matcher AFM match GT after global perm?
        issues = []
        if gt_map:
            for tid, gt_pid in gt_map.items():
                m_pid = afm.get(tid)
                if m_pid is None:
                    continue
                if perm.get(m_pid, m_pid) != gt_pid:
                    issues.append(f"T{tid}: matcher P{m_pid} (perm {perm.get(m_pid, m_pid)}) ≠ GT P{gt_pid}")

        marker = "  ⚠" if rid_short in PROBLEM_RALLIES else ""
        issue_str = (" || " + " | ".join(issues)) if issues else ""
        print(f"{idx:>3} {rid_short:>8} {ss:>2} {srv!s:>3} | "
              f"{str(pre_primary):22} | {afm_str:28} | "
              f"{gt_str:28}{issue_str}{marker}")

    # Deep dive on the 3 problem rallies
    print("\n\n" + "=" * 80)
    print("DEEP DIVE: problem rallies")
    print("=" * 80)
    for rid_short in PROBLEM_RALLIES:
        rid = next(r.rally_id for r in chronological if r.rally_id.startswith(rid_short))
        snap = snaps[rid]
        positions = snap.get("positions") or []
        primary = [int(t) for t in (snap.get("primaryTrackIds") or [])]
        print(f"\n## {rid_short} (rally {rid_to_idx[rid]} of 21)")
        print(f"   pre_primary={primary}, matcher_AFM={afm_by_rally.get(rid)}")
        print(f"   GT-correct AFM={gt_track_to_pid.get(rid)}")
        # Per-track mean Y position (shows seed-init's Y-sort would land)
        track_y: dict[int, list[float]] = defaultdict(list)
        track_x: dict[int, list[float]] = defaultdict(list)
        for q in positions:
            tid = int(q.get("trackId", -1))
            if tid in primary:
                track_y[tid].append(float(q["y"]))
                track_x[tid].append(float(q["x"]))
        print("   Per-track mean position:")
        for tid in sorted(primary):
            ys, xs = track_y[tid], track_x[tid]
            if ys:
                print(f"     T{tid}: mean (x, y) = "
                      f"({sum(xs)/len(xs):.3f}, {sum(ys)/len(ys):.3f}), "
                      f"frames={len(ys)}")
        # GT samples for this rally
        rally_gt = (gt.get("rallies", {}) or {}).get(rid, {}).get("labels", [])
        if rally_gt:
            print("   GT labels (gt_pid, x, y, frame):")
            for lbl in sorted(rally_gt, key=lambda x: x.get("frame", 0)):
                print(f"     P{lbl['playerId']} @ frame {lbl['frame']}: "
                      f"({lbl['cx']:.3f}, {lbl['cy']:.3f})")


if __name__ == "__main__":
    main()
