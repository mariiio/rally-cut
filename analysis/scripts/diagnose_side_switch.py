"""Diagnose side switch detection for a video.

Runs match-players with verbose logging to show per-rally costs
for normal vs swapped team assignments using global Hungarian matching.

Usage:
    cd analysis && uv run python scripts/diagnose_side_switch.py <video-id>
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.cli.commands.remap_track_ids import _invert_mapping, _should_reverse
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
from rallycut.tracking.match_tracker import (
    SIDE_PENALTY,
    MatchPlayerTracker,
    extract_rally_appearances,
)
from rallycut.tracking.player_features import compute_appearance_similarity


def main() -> None:
    video_id = sys.argv[1] if len(sys.argv) > 1 else "07fedbd4-693e-4651-9fee-c616a1f4b413"

    # Load rallies
    rallies = load_rallies_for_video(video_id)
    print(f"Loaded {len(rallies)} rallies for video {video_id[:8]}")

    # Reverse appliedFullMapping if positions were remapped
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()

    if row and row[0]:
        match_analysis: dict[str, Any] = row[0]  # type: ignore[assignment]
        entries_by_id: dict[str, dict[str, Any]] = {}
        for ma_entry in match_analysis.get("rallies", []):
            rid = ma_entry.get("rallyId") or ma_entry.get("rally_id", "")
            if rid:
                entries_by_id[rid] = ma_entry

        reversed_count = 0
        for rally in rallies:
            old_entry = entries_by_id.get(rally.rally_id)
            if not old_entry:
                continue
            if not old_entry.get("remapApplied", False):
                continue
            applied_raw = old_entry.get("appliedFullMapping")
            if not applied_raw:
                continue
            applied = {int(k): int(v) for k, v in applied_raw.items()}
            pos_dicts: list[dict[str, Any]] = [
                {"trackId": p.track_id} for p in rally.positions
            ]
            if _should_reverse(pos_dicts, applied):
                inverse = _invert_mapping(applied)
                for p in rally.positions:
                    if p.track_id in inverse:
                        p.track_id = inverse[p.track_id]
                rally.primary_track_ids = [
                    inverse.get(t, t) for t in rally.primary_track_ids
                ]
                reversed_count += 1

        if reversed_count:
            print(f"Reversed previous remap on {reversed_count} rallies")

    # Get video path
    video_path = get_video_path(video_id)
    if not video_path:
        print("No video path found")
        return
    print(f"Video: {video_path}")

    tracker = MatchPlayerTracker()

    for i, rally in enumerate(rallies):
        print(f"\n{'='*60}")
        print(f"Rally {i+1} ({rally.rally_id[:8]})")
        print(f"  Primary tracks: {rally.primary_track_ids}")
        print(f"  Court split Y: {rally.court_split_y}")

        # Compute avg Y per track
        track_ys: dict[int, list[float]] = defaultdict(list)
        for p in rally.positions:
            if p.track_id >= 0:
                track_ys[p.track_id].append(p.y)
        for tid in sorted(track_ys.keys()):
            avg_y = float(np.mean(track_ys[tid]))
            side = "NEAR" if avg_y > (rally.court_split_y or 0.55) else "FAR"
            print(f"  Track {tid}: avg_y={avg_y:.3f} ({side})")

        # Extract appearances
        track_stats = extract_rally_appearances(
            video_path=video_path,
            positions=rally.positions,
            primary_track_ids=rally.primary_track_ids,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
            num_samples=12,
        )

        # Process rally
        result = tracker.process_rally(
            track_stats=track_stats,
            player_positions=rally.positions,
            ball_positions=rally.ball_positions,
            court_split_y=rally.court_split_y,
        )

        print(f"  Assignments: {result.track_to_player}")
        print(f"  Confidence: {result.assignment_confidence:.3f}")
        print(f"  Side switch: {result.side_switch_detected}")
        print(f"  Server: P{result.server_player_id}")

        # Show side assignments
        for pid, team in sorted(tracker.state.current_side_assignment.items()):
            side = "NEAR" if team == 0 else "FAR"
            print(f"  P{pid} → {side}")

        # After rally 2+, show global cost matrix details
        if i >= 2:
            # Get track side classification
            track_avg_y, track_court_sides = tracker._classify_track_sides(
                track_stats, rally.positions, rally.court_split_y
            )
            all_track_ids = list(track_court_sides.keys())
            top_tracks = tracker._top_tracks_by_frames(all_track_ids, track_stats, 4)

            all_player_ids = sorted(tracker.state.players.keys())

            print(f"\n  Global cost matrix (top_tracks={top_tracks}):")
            print(f"  Track sides: {track_court_sides}")

            # Print header
            header = "         " + "".join(f"  P{pid:d}    " for pid in all_player_ids)
            print(f"  {header}")

            for tid in top_tracks:
                if tid not in track_stats:
                    continue
                track_side = track_court_sides.get(tid)
                row = f"  T{tid:3d} "
                for pid in all_player_ids:
                    if pid not in tracker.state.players:
                        continue
                    app_cost = compute_appearance_similarity(
                        tracker.state.players[pid], track_stats[tid]
                    )
                    player_side = tracker.state.current_side_assignment.get(pid)
                    side_pen = SIDE_PENALTY if track_side != player_side else 0.0
                    total = app_cost + side_pen
                    pen_str = "+" if side_pen > 0 else " "
                    row += f" {total:.3f}{pen_str}"
                print(row)

            # Show flipped count
            flipped = 0
            total = 0
            for tid, pid in result.track_to_player.items():
                ts = track_court_sides.get(tid)
                ps = tracker.state.current_side_assignment.get(pid)
                if ts is not None and ps is not None:
                    total += 1
                    if ts != ps:
                        flipped += 1
            print(f"\n  Flipped: {flipped}/{total} (need >=3 for switch)")

    # Summary
    print(f"\n{'='*60}")
    print("Side switches detected:", tracker.state.side_switches)
    print("Final side assignments:")
    for pid, team in sorted(tracker.state.current_side_assignment.items()):
        side = "NEAR" if team == 0 else "FAR"
        print(f"  P{pid} → {side}")


if __name__ == "__main__":
    main()
