"""Diagnose GT track_id mismatches in action ground truth labels.

Identifies rallies where GT playerTrackId values don't match current tracking
data, and attempts to auto-map old track_ids to current ones.

Usage:
    cd analysis
    uv run python scripts/diagnose_gt_track_mismatch.py
    uv run python scripts/diagnose_gt_track_mismatch.py --classify \
        --out reports/gt_integrity_diagnosis.json \
        --auto-fix-videos-out reports/gt_integrity_auto_fix_video_ids.txt
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.tracking.player_tracker import PlayerPosition
from scripts.eval_action_detection import (
    RallyData,
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
)

console = Console()


@dataclass
class MismatchInfo:
    rally_id: str
    video_id: str
    gt_track_ids: set[int]
    available_track_ids: set[int]
    missing_track_ids: set[int]
    n_gt_labels: int
    n_affected_labels: int
    has_stored_actions: bool
    stored_track_ids: set[int]


def find_nearest_track_at_frame(
    frame: int,
    x: float,
    y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 10,
) -> tuple[int, float] | None:
    """Find nearest track to a position at a given frame."""
    best_tid = -1
    best_dist = float("inf")

    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue
        dist = math.sqrt((x - p.x) ** 2 + (y - p.y) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_tid = p.track_id

    if best_tid < 0:
        return None
    return best_tid, best_dist


def _classify_rally(
    rally: RallyData,
    t2p: dict[int, int] | None,
) -> dict:
    """Classify rally GT integrity into clean / auto_fixable / gt_orphaned.

    GT `playerTrackId` is canonical (1-4) per eval_action_detection.py:493-497.
    `positions_json` trackIds may be raw (post-retrack) or canonical (post-remap).
    `trackToPlayer` (t2p) maps raw → canonical.

    Resolvability per avail track `t`: after remap, track contributes player
    `t` (direct) and/or `t2p[t]` (if mapped). Union across `avail_tids` gives
    the set of canonical player IDs reachable from current tracking.

    - clean: gt_tids ⊆ avail_tids (already aligned, no remap needed).
    - auto_fixable: broken AND gt_tids ⊆ resolvable (running
      `remap-track-ids` will align positions_json with GT).
    - gt_orphaned: gt_tids contain IDs not resolvable from current tracks.
      GT references a player that is no longer represented by any current
      track — neither remap nor pipeline rerun recovers them. These need
      S3 backup restore, OR position-based spatial repair (ballX/ballY),
      OR manual re-label.
    """
    gt_tids = {gt.player_track_id for gt in rally.gt_labels if gt.player_track_id >= 0}
    avail_tids: set[int] = set()
    if rally.positions_json:
        for pp in rally.positions_json:
            tid = pp.get("trackId")
            if isinstance(tid, int):
                avail_tids.add(tid)

    if not gt_tids:
        return {
            "status": "clean",
            "reason": "no GT labels with playerTrackId",
            "gt_tids": [],
            "avail_tids": sorted(avail_tids),
            "t2p_present": bool(t2p),
        }

    if gt_tids <= avail_tids:
        return {
            "status": "clean",
            "reason": "GT tids already in positions_json",
            "gt_tids": sorted(gt_tids),
            "avail_tids": sorted(avail_tids),
            "t2p_present": bool(t2p),
        }

    # Compute resolvable set: every avail_tid `t` contributes `t` itself
    # (if left unmapped) AND `t2p[t]` (if it's a key in the map, remap would
    # rewrite it). Union over all current tracks.
    resolvable: set[int] = set(avail_tids)
    if t2p:
        for t in avail_tids:
            if t in t2p:
                resolvable.add(t2p[t])

    unresolvable = gt_tids - resolvable

    if not unresolvable:
        return {
            "status": "auto_fixable",
            "reason": "remap-track-ids would align positions_json with GT",
            "gt_tids": sorted(gt_tids),
            "avail_tids": sorted(avail_tids),
            "resolvable": sorted(resolvable),
            "t2p_present": bool(t2p),
            "fix": "remap-track-ids",
        }

    return {
        "status": "gt_orphaned",
        "reason": "GT playerTrackIds do not correspond to any current track",
        "gt_tids": sorted(gt_tids),
        "avail_tids": sorted(avail_tids),
        "unresolvable_gt_tids": sorted(unresolvable),
        "resolvable": sorted(resolvable),
        "t2p_present": bool(t2p),
        "fix": "S3 backup restore OR spatial repair OR manual re-label",
    }


def run_classify(
    out_path: Path,
    auto_fix_videos_out: Path,
) -> None:
    """Classify every GT rally into clean / auto_fixable / needs_restore.

    Emits `out_path` (JSON) and `auto_fix_videos_out` (unique video IDs for
    auto_fixable rallies, one per line — feed into recover_match_state.py).
    """
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found.[/red]")
        return

    video_ids = {r.video_id for r in rallies}
    t2p_maps = _load_track_to_player_maps(video_ids)

    per_rally: list[dict] = []
    status_counts: dict[str, int] = defaultdict(int)
    auto_fix_videos: set[str] = set()
    orphaned_videos: set[str] = set()

    for idx, rally in enumerate(rallies):
        t2p = t2p_maps.get(rally.rally_id)
        cls = _classify_rally(rally, t2p)
        cls["rally_id"] = rally.rally_id
        cls["video_id"] = rally.video_id
        cls["n_gt_labels"] = len(rally.gt_labels)
        per_rally.append(cls)
        status_counts[cls["status"]] += 1
        if cls["status"] == "auto_fixable":
            auto_fix_videos.add(rally.video_id)
        elif cls["status"] == "gt_orphaned":
            orphaned_videos.add(rally.video_id)

        if (idx + 1) % 25 == 0 or idx == len(rallies) - 1:
            console.print(
                f"[{idx+1}/{len(rallies)}] {rally.rally_id[:8]} → {cls['status']}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "total_rallies": len(rallies),
                "status_counts": dict(status_counts),
                "auto_fix_video_count": len(auto_fix_videos),
                "orphaned_video_count": len(orphaned_videos),
                "per_rally": per_rally,
            },
            indent=2,
        )
    )

    auto_fix_videos_out.parent.mkdir(parents=True, exist_ok=True)
    auto_fix_videos_out.write_text(
        "\n".join(sorted(auto_fix_videos)) + ("\n" if auto_fix_videos else "")
    )

    console.print(f"\n[bold]GT Integrity Classification ({len(rallies)} rallies)[/bold]")
    for status in ("clean", "auto_fixable", "gt_orphaned"):
        console.print(f"  {status:20s} {status_counts.get(status, 0):4d}")
    console.print(f"\n  auto_fix videos: {len(auto_fix_videos)}")
    console.print(f"  orphaned videos: {len(orphaned_videos)}")
    console.print(f"\n  → {out_path}")
    console.print(f"  → {auto_fix_videos_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classify",
        action="store_true",
        help="Emit JSON classification (clean|auto_fixable|needs_restore) + video IDs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/gt_integrity_diagnosis.json"),
        help="Output JSON path when --classify is set",
    )
    parser.add_argument(
        "--auto-fix-videos-out",
        type=Path,
        default=Path("reports/gt_integrity_auto_fix_video_ids.txt"),
        help="Output TXT (one video_id per line) for auto_fixable rallies",
    )
    args = parser.parse_args()

    if args.classify:
        run_classify(args.out, args.auto_fix_videos_out)
        return

    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found.[/red]")
        return

    console.print(f"[bold]GT Track ID Mismatch Analysis ({len(rallies)} rallies)[/bold]\n")

    mismatches: list[MismatchInfo] = []
    total_gt_labels = 0
    total_affected = 0
    total_missing_tids = 0

    # Per-rally analysis
    rally_table = Table(title="Per-Rally GT Track Mismatch")
    rally_table.add_column("Rally", max_width=12)
    rally_table.add_column("Video", max_width=12)
    rally_table.add_column("GT labels", justify="right")
    rally_table.add_column("Affected", justify="right")
    rally_table.add_column("GT TIDs")
    rally_table.add_column("Available TIDs")
    rally_table.add_column("Missing TIDs", style="red")
    rally_table.add_column("Stored?")

    for rally in rallies:
        player_positions: list[PlayerPosition] = []
        if rally.positions_json:
            player_positions = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"],
                    y=pp["y"],
                    width=pp["width"],
                    height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        available_tids = set(p.track_id for p in player_positions)

        gt_tids: set[int] = set()
        affected = 0
        for gt in rally.gt_labels:
            if gt.player_track_id >= 0:
                gt_tids.add(gt.player_track_id)
                if gt.player_track_id not in available_tids:
                    affected += 1

        missing = gt_tids - available_tids

        # Check stored actions
        stored_tids: set[int] = set()
        has_stored = False
        if rally.actions_json and rally.actions_json.get("actions"):
            has_stored = True
            for a in rally.actions_json["actions"]:
                tid = a.get("playerTrackId", -1)
                if tid >= 0:
                    stored_tids.add(tid)

        total_gt_labels += len(rally.gt_labels)

        if missing:
            total_affected += affected
            total_missing_tids += len(missing)
            mismatches.append(MismatchInfo(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                gt_track_ids=gt_tids,
                available_track_ids=available_tids,
                missing_track_ids=missing,
                n_gt_labels=len(rally.gt_labels),
                n_affected_labels=affected,
                has_stored_actions=has_stored,
                stored_track_ids=stored_tids,
            ))
            rally_table.add_row(
                rally.rally_id[:12],
                rally.video_id[:12],
                str(len(rally.gt_labels)),
                str(affected),
                str(sorted(gt_tids)),
                str(sorted(available_tids)[:6]),
                str(sorted(missing)),
                "YES" if has_stored else "no",
            )

    console.print(rally_table)

    console.print("\n[bold]Summary[/bold]")
    console.print(f"  Total rallies: {len(rallies)}")
    console.print(f"  Rallies with mismatches: {len(mismatches)}")
    console.print(f"  Total GT labels: {total_gt_labels}")
    console.print(f"  Affected labels: {total_affected}")
    console.print(f"  Unique missing track IDs: {total_missing_tids}")

    # Group by video to see patterns
    console.print("\n[bold]Mismatches by Video[/bold]")
    by_video: dict[str, list[MismatchInfo]] = defaultdict(list)
    for m in mismatches:
        by_video[m.video_id].append(m)

    video_table = Table()
    video_table.add_column("Video ID", max_width=12)
    video_table.add_column("Rallies", justify="right")
    video_table.add_column("Affected labels", justify="right")
    video_table.add_column("Missing TIDs")
    video_table.add_column("Available TIDs (sample)")

    for vid, ms in sorted(by_video.items()):
        all_missing = set()
        all_available = set()
        total_aff = 0
        for m in ms:
            all_missing |= m.missing_track_ids
            all_available |= m.available_track_ids
            total_aff += m.n_affected_labels
        video_table.add_row(
            vid[:12],
            str(len(ms)),
            str(total_aff),
            str(sorted(all_missing)),
            str(sorted(all_available)[:8]),
        )
    console.print(video_table)

    # Try auto-mapping: for each rally with mismatches, check if stored actions
    # have track_ids that ARE in available_tids → those are the "current" mapping
    console.print("\n[bold]Auto-Mapping Feasibility[/bold]")

    # Check if stored actions use current track_ids
    stored_match = 0
    stored_mismatch = 0
    no_stored = 0
    for m in mismatches:
        if m.has_stored_actions and m.stored_track_ids:
            if m.stored_track_ids.issubset(m.available_track_ids | {-1}):
                stored_match += 1
            else:
                stored_mismatch += 1
        else:
            no_stored += 1

    console.print(f"  Stored actions use current TIDs: {stored_match}")
    console.print(f"  Stored actions also mismatched: {stored_mismatch}")
    console.print(f"  No stored actions: {no_stored}")

    # For rallies with stored actions that match current TIDs,
    # we could potentially use stored actions' playerTrackId as GT
    if stored_match > 0:
        console.print(f"\n  [bold green]{stored_match} rallies could use stored action TIDs as updated GT[/bold green]")

    # Check GT labels that have ballX/ballY — we can try to find the nearest
    # current track at that position
    console.print("\n[bold]Position-Based Track Mapping (using GT ball position)[/bold]")

    mappable = 0
    unmappable = 0
    mapping_results: list[dict] = []

    for m in mismatches:
        rally = None
        for r in rallies:
            if r.rally_id == m.rally_id:
                rally = r
                break
        if not rally:
            continue

        player_positions = []
        if rally.positions_json:
            player_positions = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"],
                    y=pp["y"],
                    width=pp["width"],
                    height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        for gt in rally.gt_labels:
            if gt.player_track_id < 0 or gt.player_track_id in m.available_track_ids:
                continue  # Not affected

            # Try to find nearest player at the GT frame using ball position
            if gt.ball_x is not None and gt.ball_y is not None:
                result = find_nearest_track_at_frame(
                    gt.frame, gt.ball_x, gt.ball_y, player_positions
                )
                if result:
                    mappable += 1
                    mapping_results.append({
                        "rally": m.rally_id[:8],
                        "frame": gt.frame,
                        "action": gt.action,
                        "old_tid": gt.player_track_id,
                        "new_tid": result[0],
                        "distance": result[1],
                    })
                else:
                    unmappable += 1
            else:
                unmappable += 1

    console.print(f"  Labels with ball position → nearest track: {mappable}")
    console.print(f"  Labels without ball position or no nearby track: {unmappable}")

    if mapping_results:
        map_table = Table(title="Proposed Track Mappings (sample)")
        map_table.add_column("Rally", max_width=8)
        map_table.add_column("Frame", justify="right")
        map_table.add_column("Action")
        map_table.add_column("Old TID", justify="right")
        map_table.add_column("→ New TID", justify="right")
        map_table.add_column("Distance", justify="right")

        for r in mapping_results[:20]:
            map_table.add_row(
                r["rally"],
                str(r["frame"]),
                r["action"],
                str(r["old_tid"]),
                str(r["new_tid"]),
                f"{r['distance']:.4f}",
            )
        console.print(map_table)

    # Final recommendation
    console.print("\n[bold]Recommendation[/bold]")
    console.print(f"  Option 1: Re-label {len(mismatches)} rallies in web editor (most reliable)")
    console.print(f"  Option 2: Auto-update {mappable} GT labels using nearest-track mapping")
    if stored_match > 0:
        console.print(f"  Option 3: Use stored action TIDs for {stored_match} rallies")
    console.print(f"\n  Impact: fixing {total_affected} affected labels across {len(mismatches)} rallies")


if __name__ == "__main__":
    main()
