"""Batch-run match-players on videos missing match_analysis_json.

Usage:
    cd analysis
    uv run python scripts/batch_match_players.py
"""

from __future__ import annotations

import time

from rich.console import Console

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
from rallycut.tracking.match_tracker import match_players_across_rallies

console = Console()


def get_videos_needing_match() -> list[tuple[str, str, int]]:
    """Get video IDs that have action GT but no match_analysis_json."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT r.video_id, v.filename, COUNT(r.id)
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON v.id = r.video_id
                WHERE pt.action_ground_truth_json IS NOT NULL
                  AND v.match_analysis_json IS NULL
                GROUP BY r.video_id, v.filename
                ORDER BY COUNT(r.id) DESC
            """)
            return [(str(r[0]), str(r[1] or "?"), int(r[2])) for r in cur.fetchall()]


def save_match_analysis(video_id: str, result_json: dict) -> None:
    """Save match_analysis_json to the videos table."""
    import json

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                [json.dumps(result_json), video_id],
            )
        conn.commit()


def main() -> None:
    videos = get_videos_needing_match()
    if not videos:
        console.print("[green]All videos already have match_analysis_json.[/green]")
        return

    total_rallies = sum(n for _, _, n in videos)
    console.print(
        f"[bold]Batch match-players: {len(videos)} videos, "
        f"{total_rallies} rallies with action GT[/bold]\n"
    )

    succeeded = 0
    failed = 0

    for i, (video_id, filename, n_rallies) in enumerate(videos):
        t0 = time.monotonic()
        print(f"[{i + 1}/{len(videos)}] {filename[:30]:<30} ({n_rallies} rallies)...", end=" ", flush=True)

        try:
            rallies = load_rallies_for_video(video_id)
            if not rallies:
                print("SKIP (no tracked rallies)")
                continue

            video_path = get_video_path(video_id)
            if video_path is None:
                print("SKIP (no video file)")
                continue

            result = match_players_across_rallies(
                video_path=video_path,
                rallies=rallies,
                num_samples=12,
            )

            # Build JSON for DB storage (same format as CLI match-players output)
            result_json = {
                "videoId": video_id,
                "numRallies": len(result.rally_results),
                "rallies": [
                    {
                        "rallyId": rallies[r.rally_index].rally_id if r.rally_index < len(rallies) else "",
                        "rallyIndex": r.rally_index,
                        "trackToPlayer": {str(k): v for k, v in r.track_to_player.items()},
                        "assignmentConfidence": r.assignment_confidence,
                        "sideSwitchDetected": r.side_switch_detected,
                        "serverPlayerId": r.server_player_id,
                    }
                    for r in result.rally_results
                ],
                "playerProfiles": {
                    str(pid): profile.to_dict()
                    for pid, profile in result.player_profiles.items()
                    if profile.rally_count > 0
                },
            }

            save_match_analysis(video_id, result_json)
            elapsed = time.monotonic() - t0
            avg_conf = (
                sum(r.assignment_confidence for r in result.rally_results)
                / len(result.rally_results)
                if result.rally_results
                else 0
            )
            switches = sum(1 for r in result.rally_results if r.side_switch_detected)
            print(f"OK conf={avg_conf:.2f} switches={switches} ({elapsed:.1f}s)")
            succeeded += 1

        except Exception as e:
            elapsed = time.monotonic() - t0
            print(f"FAIL: {e} ({elapsed:.1f}s)")
            failed += 1

    console.print(f"\n[bold]Done: {succeeded} succeeded, {failed} failed[/bold]")


if __name__ == "__main__":
    main()
