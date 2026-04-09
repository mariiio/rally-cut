"""Find GT 'attack' labels that are likely overpasses (dig/set/receive going over the net).

Heuristic: An attack on the 1st or 2nd touch on a side is suspicious — real attacks
are typically the 3rd touch. We detect side changes using court_split_y.

Usage:
    cd analysis
    uv run python scripts/find_overpass_attacks.py
"""

from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection

console = Console()


def main() -> None:
    query = """
        SELECT
            r.id as rally_id,
            r.video_id,
            v.filename,
            pt.action_ground_truth_json,
            pt.court_split_y,
            r.start_ms,
            pt.fps
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.action_ground_truth_json IS NOT NULL
        ORDER BY v.filename, r.start_ms
    """

    suspects: list[dict] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    # Track rally number per video
    video_rally_counts: dict[str, int] = {}

    for row in rows:
        rally_id, video_id, filename, gt_json, court_split_y, start_ms, fps = row
        if not gt_json or not court_split_y:
            continue

        # Count rally number within video
        video_rally_counts[video_id] = video_rally_counts.get(video_id, 0) + 1
        rally_num = video_rally_counts[video_id]

        labels = gt_json
        net_y = court_split_y

        # Walk through actions, tracking which side has the ball and touch count
        current_side = None  # "near" or "far"
        touch_on_side = 0

        for label in labels:
            action = label["action"]
            ball_y = label.get("ballY")
            frame = label["frame"]

            # Determine side from ball Y position
            if ball_y is not None:
                side = "near" if ball_y > net_y else "far"
            else:
                side = None

            # Serve resets everything
            if action == "serve":
                current_side = side
                touch_on_side = 1
                continue

            # Receive is always 1st touch on the receiving side
            if action == "receive":
                current_side = side
                touch_on_side = 1
                continue

            # Check if side changed
            if side and current_side and side != current_side:
                # Ball crossed net — reset touch count
                current_side = side
                touch_on_side = 1
            else:
                touch_on_side += 1

            if action == "attack" and touch_on_side < 3:
                time_s = start_ms / 1000 + frame / (fps or 30)
                minutes = int(time_s // 60)
                seconds = time_s % 60

                expected = "dig" if touch_on_side == 1 else "set"

                suspects.append({
                    "filename": filename,
                    "video_id": video_id[:8],
                    "rally_num": rally_num,
                    "rally_id": rally_id[:8],
                    "touch": touch_on_side,
                    "expected": expected,
                    "frame": frame,
                    "time": f"{minutes}:{seconds:05.2f}",
                    "ball_y": f"{ball_y:.3f}" if ball_y else "?",
                    "net_y": f"{net_y:.3f}",
                })

    if not suspects:
        console.print("[green]No suspect attack labels found!")
        return

    console.print(f"\n[bold]Found {len(suspects)} suspect 'attack' labels (1st or 2nd touch on side):[/bold]\n")

    table = Table(show_lines=True)
    table.add_column("Video", style="cyan")
    table.add_column("Rally #", justify="center")
    table.add_column("Time", justify="right")
    table.add_column("Touch #", justify="center")
    table.add_column("Likely", style="yellow")
    table.add_column("Frame", justify="right")
    table.add_column("Ball Y", justify="right")
    table.add_column("Net Y", justify="right")
    table.add_column("Rally ID", style="dim")

    for s in suspects:
        table.add_row(
            s["filename"],
            str(s["rally_num"]),
            s["time"],
            str(s["touch"]),
            s["expected"],
            str(s["frame"]),
            s["ball_y"],
            s["net_y"],
            s["rally_id"],
        )

    console.print(table)

    # Summary by video
    console.print("\n[bold]Summary by video:[/bold]")
    by_video: dict[str, int] = {}
    for s in suspects:
        by_video[s["filename"]] = by_video.get(s["filename"], 0) + 1
    for fname, count in sorted(by_video.items()):
        console.print(f"  {fname}: {count} suspects")


if __name__ == "__main__":
    main()
