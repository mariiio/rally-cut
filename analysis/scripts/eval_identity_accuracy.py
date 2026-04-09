"""Measure per-frame identity classifier accuracy.

For each video with reference crops AND track_to_player mappings, trains the
DINOv2 identity classifier and measures what % of frames it correctly identifies
each player. This tells us whether the bottleneck is classifier quality or
integration quality.

Usage:
    cd analysis
    uv run python scripts/eval_identity_accuracy.py
    uv run python scripts/eval_identity_accuracy.py --video <video-id>
    uv run python scripts/eval_identity_accuracy.py --max-rallies 5  # quick test
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from typing import Any

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.identity_classifier import FrameIdentityClassifier
from rallycut.tracking.reid_embeddings import extract_crops_from_video

console = Console()


def _load_videos_with_crops_and_tracks() -> list[dict[str, Any]]:
    """Load videos that have both reference crops and match analysis."""
    query = """
        SELECT v.id as video_id, v.match_analysis_json
        FROM videos v
        WHERE v.match_analysis_json IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM player_reference_crops prc WHERE prc.video_id = v.id
          )
        ORDER BY v.id
    """
    results: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                video_id, ma_json = row
                if not isinstance(ma_json, dict):
                    continue
                results.append({
                    "video_id": video_id,
                    "match_analysis_json": ma_json,
                })
    return results


def _load_reference_crops(video_id: str) -> list[dict[str, Any]]:
    """Load reference crop info for a video."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                   FROM player_reference_crops
                   WHERE video_id = %s
                   ORDER BY player_id, created_at""",
                [video_id],
            )
            return [
                {
                    "player_id": r[0], "frame_ms": r[1],
                    "bbox_x": r[2], "bbox_y": r[3],
                    "bbox_w": r[4], "bbox_h": r[5],
                }
                for r in cur.fetchall()
            ]


def _load_rallies_for_video(video_id: str) -> list[dict[str, Any]]:
    """Load rallies with positions for a video."""
    query = """
        SELECT r.id, r.start_ms, pt.positions_json, pt.fps
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND pt.positions_json IS NOT NULL
        ORDER BY r.start_ms
    """
    results: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            for row in cur.fetchall():
                rally_id, start_ms, positions_json, fps = row
                if positions_json:
                    results.append({
                        "rally_id": rally_id,
                        "start_ms": start_ms or 0,
                        "positions_json": positions_json,
                        "fps": fps or 30.0,
                    })
    return results


def _derive_player_teams(
    ma_json: dict[str, Any],
) -> tuple[dict[int, int], dict[str, dict[int, int]]]:
    """Derive player_id → team and per-rally track_to_player from match analysis.

    Returns (player_teams, rally_t2p) where:
    - player_teams: {player_id: team (0/1)}
    - rally_t2p: {rally_id: {track_id: player_id}}
    """
    player_teams: dict[int, int] = {}
    rally_t2p: dict[str, dict[int, int]] = {}

    # Primary source: playerProfiles at video level
    profiles = ma_json.get("playerProfiles", {})
    for pid_str, profile in profiles.items():
        pid = int(pid_str)
        team = profile.get("team")
        if team is not None:
            player_teams[pid] = int(team)

    # Per-rally track_to_player
    for rally_entry in ma_json.get("rallies", []):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        t2p = rally_entry.get("trackToPlayer") or rally_entry.get("track_to_player", {})

        if rid and t2p:
            rally_t2p[rid] = {int(k): int(v) for k, v in t2p.items()}

    return player_teams, rally_t2p


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure per-frame identity accuracy")
    parser.add_argument("--video", type=str, help="Specific video ID")
    parser.add_argument("--max-rallies", type=int, default=0, help="Max rallies per video (0=all)")
    parser.add_argument("--every-n", type=int, default=3, help="Classify every Nth frame")
    args = parser.parse_args()

    videos = _load_videos_with_crops_and_tracks()
    if args.video:
        videos = [v for v in videos if v["video_id"] == args.video or v["video_id"].startswith(args.video)]

    if not videos:
        console.print("[red]No videos found with both reference crops and match analysis.[/red]")
        return

    console.print(f"\n[bold]Per-Frame Identity Accuracy ({len(videos)} videos)[/bold]\n")

    # Summary table
    summary_table = Table(title="Per-Video Identity Accuracy")
    summary_table.add_column("Video", style="dim", max_width=12)
    summary_table.add_column("Players", justify="right")
    summary_table.add_column("Rallies", justify="right")
    summary_table.add_column("Frames", justify="right")
    summary_table.add_column("Player Acc", justify="right")
    summary_table.add_column("Team Acc", justify="right")
    summary_table.add_column("Avg Conf", justify="right")
    summary_table.add_column("Time", justify="right")

    all_correct = 0
    all_team_correct = 0
    all_total = 0
    all_conf_sum = 0.0

    for vid_info in videos:
        vid = vid_info["video_id"]
        ma_json = vid_info["match_analysis_json"]
        t_start = time.monotonic()

        print(f"Processing {vid[:8]}...", end=" ", flush=True)

        # Load reference crops
        crop_infos = _load_reference_crops(vid)
        video_path = get_video_path(vid)
        if not crop_infos or video_path is None:
            print("skipped (no crops/video)")
            continue

        crops_by_player = extract_crops_from_video(video_path, crop_infos)
        if len(crops_by_player) < 2:
            print("skipped (<2 players)")
            continue

        # Derive teams and track_to_player
        player_teams, rally_t2p = _derive_player_teams(ma_json)

        # Train classifier
        classifier = FrameIdentityClassifier(
            player_teams=player_teams,
            classify_every_n=args.every_n,
        )
        stats = classifier.train(crops_by_player)
        n_players = len(crops_by_player)

        # Load rallies
        rallies = _load_rallies_for_video(vid)
        if args.max_rallies > 0:
            rallies = rallies[:args.max_rallies]

        vid_correct = 0
        vid_team_correct = 0
        vid_total = 0
        vid_conf_sum = 0.0
        n_rallies_eval = 0

        for rally in rallies:
            rid = rally["rally_id"]
            t2p = rally_t2p.get(rid)
            if not t2p:
                continue

            # Invert: player_id → set of track_ids (but usually 1:1)
            # We need track_id → player_id for evaluation
            positions_json = rally["positions_json"]
            rally_start_frame = int(rally["start_ms"] / 1000.0 * rally["fps"])

            # Reset classifier state for each rally
            classifier._track_states.clear()
            classifier._label_cache.clear()

            # Run per-frame identity classification
            identity_labels = classifier.classify_detections_batch(
                positions_json=positions_json,
                video_path=str(video_path),
                rally_start_frame=rally_start_frame,
            )

            # Evaluate: for each classified frame, check if identity matches t2p
            for tid, frame_labels in identity_labels.items():
                expected_pid = t2p.get(tid)
                if expected_pid is None:
                    continue  # Track not in t2p mapping

                for _frame, label in frame_labels.items():
                    vid_total += 1
                    vid_conf_sum += label.confidence

                    if label.player_id == expected_pid:
                        vid_correct += 1

                    # Team accuracy: even if wrong player, is the team correct?
                    expected_team = player_teams.get(expected_pid)
                    if expected_team is not None and label.team == expected_team:
                        vid_team_correct += 1

            n_rallies_eval += 1

        elapsed = time.monotonic() - t_start
        player_acc = vid_correct / max(1, vid_total)
        team_acc = vid_team_correct / max(1, vid_total)
        avg_conf = vid_conf_sum / max(1, vid_total)

        all_correct += vid_correct
        all_team_correct += vid_team_correct
        all_total += vid_total
        all_conf_sum += vid_conf_sum

        print(
            f"{n_players}p {n_rallies_eval}r {vid_total}f "
            f"player={player_acc:.1%} team={team_acc:.1%} "
            f"conf={avg_conf:.2f} ({elapsed:.1f}s)"
        )

        summary_table.add_row(
            vid[:8],
            str(n_players),
            str(n_rallies_eval),
            str(vid_total),
            f"{player_acc:.1%}",
            f"{team_acc:.1%}",
            f"{avg_conf:.2f}",
            f"{elapsed:.0f}s",
        )

    console.print(summary_table)

    if all_total > 0:
        console.print(f"\n[bold]Aggregate ({len(videos)} videos, {all_total} frames)[/bold]")
        console.print(f"  Player Accuracy: {all_correct / all_total:.1%} ({all_correct}/{all_total})")
        console.print(f"  Team Accuracy:   {all_team_correct / all_total:.1%} ({all_team_correct}/{all_total})")
        console.print(f"  Avg Confidence:  {all_conf_sum / all_total:.3f}")


if __name__ == "__main__":
    main()
