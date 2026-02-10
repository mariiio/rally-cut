#!/usr/bin/env python3
"""Compare tracking results with different configs against ground truth.

Usage:
    uv run python scripts/compare_tracking_configs.py
"""

import tempfile
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from rallycut.evaluation.tracking.db import load_labeled_rallies
from rallycut.evaluation.tracking.metrics import evaluate_rally
from rallycut.tracking.player_filter import PlayerFilter, PlayerFilterConfig
from rallycut.tracking.player_tracker import PlayerTracker


def get_minio_client():
    """Get MinIO S3 client."""
    return boto3.client(
        "s3",
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
        config=BotoConfig(signature_version="s3v4"),
    )


def download_video(s3_key: str, local_path: Path) -> None:
    """Download video from MinIO."""
    client = get_minio_client()
    client.download_file("rallycut-dev", s3_key, str(local_path))


def run_tracking_for_rally(
    video_path: Path,
    start_ms: int,
    end_ms: int,
    fps: float = 30.0,
) -> tuple[list, int]:
    """Run tracking for a specific time range and return positions."""
    from rallycut.tracking.player_tracker import PlayerPosition

    tracker = PlayerTracker()

    # Track the full video segment
    result = tracker.track_video(
        video_path=video_path,
        start_ms=start_ms,
        end_ms=end_ms,
        filter_enabled=True,  # Apply filtering
    )

    # Convert absolute frame numbers to rally-relative
    # Video starts at start_ms, so frame 0 of rally = frame at start_ms in video
    start_frame = int(start_ms / 1000.0 * fps)

    adjusted_positions = [
        PlayerPosition(
            frame_number=p.frame_number - start_frame,
            track_id=p.track_id,
            x=p.x,
            y=p.y,
            width=p.width,
            height=p.height,
            confidence=p.confidence,
        )
        for p in result.positions
    ]

    return adjusted_positions, result.frame_count


def main():
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Load labeled rallies
    console.print("[bold]Loading labeled rallies...[/bold]")
    rallies = load_labeled_rallies()
    console.print(f"Found {len(rallies)} rallies with ground truth\n")

    # Get S3 keys for videos using existing db module
    from rallycut.evaluation.db import get_connection

    results = []

    for rally in rallies:  # Test with all rallies
        console.print(f"[bold]Processing rally {rally.rally_id[:8]}...[/bold]")

        # Get video S3 key
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT v.s3_key FROM videos v JOIN rallies r ON r.video_id = v.id WHERE r.id = %s",
                    (rally.rally_id,)
                )
                row = cur.fetchone()
                if not row:
                    console.print(f"  [red]No video found, skipping[/red]")
                    continue
                s3_key = row[0]

        # Download video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = Path(f.name)

        console.print(f"  Downloading video...")
        try:
            download_video(s3_key, video_path)
        except Exception as e:
            console.print(f"  [red]Failed to download: {e}[/red]")
            continue

        try:
            # Run tracking
            console.print(f"  Running tracking (yolov8s + lowered thresholds)...")
            positions, frame_count = run_tracking_for_rally(
                video_path=video_path,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                fps=rally.video_fps,
            )

            # Create PlayerTrackingResult-like object for evaluation
            from rallycut.tracking.player_tracker import PlayerTrackingResult
            new_predictions = PlayerTrackingResult(
                positions=positions,
                frame_count=frame_count,
                video_fps=rally.video_fps,
            )

            # Evaluate new predictions
            new_result = evaluate_rally(
                rally_id=rally.rally_id,
                ground_truth=rally.ground_truth,
                predictions=new_predictions,
                video_width=rally.video_width,
                video_height=rally.video_height,
            )

            # Compare with old predictions (from database)
            if rally.predictions:
                old_result = evaluate_rally(
                    rally_id=rally.rally_id,
                    ground_truth=rally.ground_truth,
                    predictions=rally.predictions,
                    video_width=rally.video_width,
                    video_height=rally.video_height,
                )

                results.append({
                    "rally": rally.rally_id[:8],
                    "old_recall": old_result.aggregate.recall * 100,
                    "new_recall": new_result.aggregate.recall * 100,
                    "old_precision": old_result.aggregate.precision * 100,
                    "new_precision": new_result.aggregate.precision * 100,
                    "old_f1": old_result.aggregate.f1 * 100,
                    "new_f1": new_result.aggregate.f1 * 100,
                    "old_id_sw": old_result.aggregate.num_id_switches,
                    "new_id_sw": new_result.aggregate.num_id_switches,
                })

            console.print(f"  [green]Done![/green] New: recall={new_result.aggregate.recall:.1%}, "
                         f"precision={new_result.aggregate.precision:.1%}, "
                         f"F1={new_result.aggregate.f1:.1%}")

        finally:
            # Clean up
            video_path.unlink(missing_ok=True)

    # Display comparison table
    if results:
        console.print("\n[bold]Comparison Results[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Rally")
        table.add_column("Old Recall")
        table.add_column("New Recall")
        table.add_column("Δ Recall")
        table.add_column("Old F1")
        table.add_column("New F1")
        table.add_column("Δ F1")
        table.add_column("ID Sw")

        for r in results:
            recall_delta = r["new_recall"] - r["old_recall"]
            f1_delta = r["new_f1"] - r["old_f1"]

            recall_style = "green" if recall_delta > 0 else "red" if recall_delta < 0 else ""
            f1_style = "green" if f1_delta > 0 else "red" if f1_delta < 0 else ""

            table.add_row(
                r["rally"],
                f"{r['old_recall']:.1f}%",
                f"{r['new_recall']:.1f}%",
                f"[{recall_style}]{recall_delta:+.1f}%[/{recall_style}]",
                f"{r['old_f1']:.1f}%",
                f"{r['new_f1']:.1f}%",
                f"[{f1_style}]{f1_delta:+.1f}%[/{f1_style}]",
                f"{r['old_id_sw']}→{r['new_id_sw']}",
            )

        console.print(table)


if __name__ == "__main__":
    main()
