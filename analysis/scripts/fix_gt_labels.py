#!/usr/bin/env python3
"""Fix player_matching_gt_json labels interactively or via batch commands.

Usage:
    # Show current GT for a video
    uv run python scripts/fix_gt_labels.py show <video_id>

    # Swap two players globally across all rallies in a video
    uv run python scripts/fix_gt_labels.py swap <video_id> <pid_a> <pid_b>

    # Swap two players in a specific rally only
    uv run python scripts/fix_gt_labels.py swap-rally <video_id> <rally_prefix> <pid_a> <pid_b>

    # Reassign a track in a specific rally
    uv run python scripts/fix_gt_labels.py assign <video_id> <rally_prefix> <track_id> <new_pid>

    # Remove a rally from GT (bad data)
    uv run python scripts/fix_gt_labels.py remove-rally <video_id> <rally_prefix>

    # Regenerate grid for a video after fixing
    uv run python scripts/fix_gt_labels.py grid <video_id>

    # Validate all GT: check for duplicates, missing players, missing tracks
    uv run python scripts/fix_gt_labels.py validate

    # Migrate pre-remap GT track IDs to current (post-remap) IDs
    uv run python scripts/fix_gt_labels.py migrate-gt
    uv run python scripts/fix_gt_labels.py migrate-gt --dry-run
"""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from rallycut.evaluation.db import get_connection


def load_gt(video_id: str) -> tuple[str, dict[str, Any]]:
    """Load GT for a video. Returns (full_video_id, gt_data)."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, player_matching_gt_json FROM videos "
                "WHERE id LIKE %s AND player_matching_gt_json IS NOT NULL",
                [f"{video_id}%"],
            )
            row = cur.fetchone()
    if not row:
        print(f"Error: No GT found for video {video_id}")
        sys.exit(1)
    return str(row[0]), cast(dict[str, Any], row[1])


def save_gt(video_id: str, gt_data: dict[str, Any]) -> None:
    """Save GT back to DB."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET player_matching_gt_json = %s WHERE id = %s",
                [json.dumps(gt_data), video_id],
            )
        conn.commit()


def find_rally(rallies: dict[str, Any], prefix: str) -> str | None:
    """Find rally ID by prefix."""
    matches = [rid for rid in rallies if rid.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        print(f"Ambiguous prefix '{prefix}', matches: {[r[:8] for r in matches]}")
    else:
        print(f"No rally matching prefix '{prefix}'")
    return None


def cmd_show(video_id: str) -> None:
    full_id, gt = load_gt(video_id)
    rallies = gt.get("rallies", {})
    switches = gt.get("sideSwitches", [])

    print(f"Video: {full_id[:8]}")
    print(f"Rallies: {len(rallies)}")
    if switches:
        print(f"Side switches at: {switches}")
    print()

    for i, (rid, mapping) in enumerate(rallies.items()):
        pids = sorted(set(mapping.values()))
        flag = " ⚠" if len(pids) < 4 else ""
        assigns = ", ".join(f"T{k}→P{v}" for k, v in sorted(mapping.items(), key=lambda x: x[1]))
        switch = " [SWITCH]" if i in switches else ""
        print(f"  {rid[:8]}: {assigns}{flag}{switch}")


def cmd_swap(video_id: str, pid_a: int, pid_b: int) -> None:
    full_id, gt = load_gt(video_id)
    rallies = gt.get("rallies", {})

    n_swapped = 0
    for rid, mapping in rallies.items():
        changed = False
        for tid in mapping:
            if mapping[tid] == pid_a:
                mapping[tid] = pid_b
                changed = True
            elif mapping[tid] == pid_b:
                mapping[tid] = pid_a
                changed = True
        if changed:
            n_swapped += 1

    save_gt(full_id, gt)
    print(f"Swapped P{pid_a} ↔ P{pid_b} in {n_swapped}/{len(rallies)} rallies")


def cmd_swap_rally(video_id: str, rally_prefix: str, pid_a: int, pid_b: int) -> None:
    full_id, gt = load_gt(video_id)
    rallies = gt.get("rallies", {})

    rid = find_rally(rallies, rally_prefix)
    if rid is None:
        return

    mapping = rallies[rid]
    for tid in mapping:
        if mapping[tid] == pid_a:
            mapping[tid] = pid_b
        elif mapping[tid] == pid_b:
            mapping[tid] = pid_a

    save_gt(full_id, gt)
    assigns = ", ".join(f"T{k}→P{v}" for k, v in sorted(mapping.items(), key=lambda x: x[1]))
    print(f"Swapped P{pid_a} ↔ P{pid_b} in rally {rid[:8]}: {assigns}")


def cmd_assign(video_id: str, rally_prefix: str, track_id: str, new_pid: int) -> None:
    full_id, gt = load_gt(video_id)
    rallies = gt.get("rallies", {})

    rid = find_rally(rallies, rally_prefix)
    if rid is None:
        return

    mapping = rallies[rid]
    old_pid = mapping.get(track_id)
    mapping[track_id] = new_pid

    save_gt(full_id, gt)
    print(f"Rally {rid[:8]}: T{track_id} reassigned P{old_pid} → P{new_pid}")


def cmd_remove_rally(video_id: str, rally_prefix: str) -> None:
    full_id, gt = load_gt(video_id)
    rallies = gt.get("rallies", {})

    rid = find_rally(rallies, rally_prefix)
    if rid is None:
        return

    del rallies[rid]
    save_gt(full_id, gt)
    print(f"Removed rally {rid[:8]} from GT ({len(rallies)} remaining)")


def cmd_grid(video_id: str) -> None:
    """Regenerate grid for a single video."""
    import subprocess

    result = subprocess.run(
        ["uv", "run", "python", "scripts/visualize_gt_grids.py", "--video-id", video_id],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)


def cmd_validate() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, player_matching_gt_json FROM videos "
                "WHERE player_matching_gt_json IS NOT NULL ORDER BY id"
            )
            rows = cur.fetchall()

            # Load position track IDs per rally (only GT videos)
            cur.execute(
                "SELECT r.id, pt.positions_json FROM rallies r "
                "JOIN player_tracks pt ON pt.rally_id = r.id "
                "JOIN videos v ON v.id = r.video_id "
                "WHERE pt.positions_json IS NOT NULL "
                "AND v.player_matching_gt_json IS NOT NULL"
            )
            rally_track_ids: dict[str, set[int]] = {}
            for rid, pos_json in cur.fetchall():
                tids: set[int] = set()
                positions = cast(list[dict[str, Any]], pos_json or [])
                for p in positions:
                    tid = p.get("trackId", p.get("track_id"))
                    if tid is not None:
                        tids.add(tid)
                rally_track_ids[str(rid)] = tids

    issues = 0
    for vid, gt_raw in rows:
        vid_str = str(vid)[:8]
        gt = cast(dict[str, Any], gt_raw)
        rallies = gt.get("rallies", {})
        for rid, mapping in rallies.items():
            pids = list(mapping.values())

            # Check for duplicate player IDs
            if len(pids) != len(set(pids)):
                dups = [p for p in set(pids) if pids.count(p) > 1]
                print(f"  DUP  {vid_str} {rid[:8]}: P{dups} assigned to multiple tracks")
                issues += 1

            # Check for out-of-range player IDs
            for p in pids:
                if p not in (1, 2, 3, 4):
                    print(f"  BAD  {vid_str} {rid[:8]}: player ID {p} out of range")
                    issues += 1

            # Check for GT track IDs missing from positions
            pos_tids = rally_track_ids.get(rid, set())
            if pos_tids:
                for tid_str in mapping:
                    if int(tid_str) not in pos_tids:
                        print(
                            f"  MISSING_TRACK  {vid_str} {rid[:8]}: "
                            f"GT track {tid_str} not in positions "
                            f"(have {sorted(pos_tids)})"
                        )
                        issues += 1

    total_rallies = sum(
        len(cast(dict[str, Any], g).get("rallies", {})) for _, g in rows
    )
    print(f"\n{issues} issues found across {len(rows)} videos, {total_rallies} rallies")


def cmd_migrate_gt(dry_run: bool = False) -> None:
    """Translate pre-remap GT track IDs to current (post-remap) IDs."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, player_matching_gt_json, match_analysis_json FROM videos "
                "WHERE player_matching_gt_json IS NOT NULL ORDER BY id"
            )
            rows = cur.fetchall()

    total_migrated = 0
    for vid, gt_raw, ma_raw in rows:
        vid_str = str(vid)[:8]
        if not ma_raw:
            continue
        gt = cast(dict[str, Any], gt_raw)
        ma_json = cast(dict[str, Any], ma_raw)

        # Build per-rally appliedFullMapping
        remap_by_rally: dict[str, dict[str, int]] = {}
        for entry in cast(list[dict[str, Any]], ma_json.get("rallies", [])):
            rid_key = entry.get("rallyId") or entry.get("rally_id", "")
            afm = entry.get("appliedFullMapping")
            if afm and entry.get("remapApplied"):
                remap_by_rally[rid_key] = cast(dict[str, int], afm)

        if not remap_by_rally:
            continue

        rallies = cast(dict[str, dict[str, Any]], gt.get("rallies", {}))
        video_migrated = 0
        for rid, mapping in rallies.items():
            afm = remap_by_rally.get(rid)
            if not afm:
                continue

            # Check if any GT track IDs need translation
            new_mapping: dict[str, Any] = {}
            changed = False
            for tid_str, pid in mapping.items():
                new_tid = afm.get(tid_str)
                if new_tid is not None and str(new_tid) != str(tid_str):
                    new_mapping[str(new_tid)] = pid
                    changed = True
                else:
                    new_mapping[tid_str] = pid

            if changed:
                rallies[rid] = new_mapping
                video_migrated += 1

        if video_migrated > 0:
            total_migrated += video_migrated
            action = "would migrate" if dry_run else "migrated"
            print(f"  {vid_str}: {action} {video_migrated}/{len(rallies)} rallies")
            if not dry_run:
                save_gt(str(vid), gt)

    print(f"\n{'Would migrate' if dry_run else 'Migrated'} "
          f"{total_migrated} rallies across {len(rows)} videos")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "show" and len(sys.argv) >= 3:
        cmd_show(sys.argv[2])
    elif cmd == "swap" and len(sys.argv) >= 5:
        cmd_swap(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "swap-rally" and len(sys.argv) >= 6:
        cmd_swap_rally(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif cmd == "assign" and len(sys.argv) >= 6:
        cmd_assign(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
    elif cmd == "remove-rally" and len(sys.argv) >= 4:
        cmd_remove_rally(sys.argv[2], sys.argv[3])
    elif cmd == "grid" and len(sys.argv) >= 3:
        cmd_grid(sys.argv[2])
    elif cmd == "validate":
        cmd_validate()
    elif cmd == "migrate-gt":
        dry_run = "--dry-run" in sys.argv
        cmd_migrate_gt(dry_run=dry_run)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
