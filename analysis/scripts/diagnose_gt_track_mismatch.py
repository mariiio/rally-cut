"""Diagnose GT track_id mismatches in action ground truth labels.

Identifies rallies where GT playerTrackId values don't match current tracking
data, and attempts to auto-map old track_ids to current ones.

Usage:
    cd analysis
    uv run python scripts/diagnose_gt_track_mismatch.py
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from scripts.eval_action_detection import load_rallies_with_action_gt
from rallycut.tracking.player_tracker import PlayerPosition

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


def main() -> None:
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

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Total rallies: {len(rallies)}")
    console.print(f"  Rallies with mismatches: {len(mismatches)}")
    console.print(f"  Total GT labels: {total_gt_labels}")
    console.print(f"  Affected labels: {total_affected}")
    console.print(f"  Unique missing track IDs: {total_missing_tids}")

    # Group by video to see patterns
    console.print(f"\n[bold]Mismatches by Video[/bold]")
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
    console.print(f"\n[bold]Auto-Mapping Feasibility[/bold]")

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
    console.print(f"\n[bold]Position-Based Track Mapping (using GT ball position)[/bold]")

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
    console.print(f"\n[bold]Recommendation[/bold]")
    console.print(f"  Option 1: Re-label {len(mismatches)} rallies in web editor (most reliable)")
    console.print(f"  Option 2: Auto-update {mappable} GT labels using nearest-track mapping")
    if stored_match > 0:
        console.print(f"  Option 3: Use stored action TIDs for {stored_match} rallies")
    console.print(f"\n  Impact: fixing {total_affected} affected labels across {len(mismatches)} rallies")


if __name__ == "__main__":
    main()
