"""Restore training datasets into the application database.

Re-imports ground truth videos, rally annotations, and tracking ground truth
from a local dataset directory into PostgreSQL, so the web editor can access
them after a DB reset or fresh machine setup.

Expected files: manifest.json, ground_truth.json, tracking_ground_truth.json (optional),
action_ground_truth.json (optional).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psycopg
from platformdirs import user_cache_dir

from rallycut.evaluation.db import get_connection


@dataclass
class RestoreResult:
    """Summary of what was restored."""

    videos_inserted: int = 0
    videos_skipped: int = 0
    rallies_inserted: int = 0
    tracking_gt_restored: int = 0
    action_gt_restored: int = 0
    session_created: str = ""
    errors: list[str] = field(default_factory=list)


def _get_default_user_id(conn: psycopg.Connection[tuple[Any, ...]]) -> str:
    """Get the first user ID from the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM users ORDER BY created_at ASC LIMIT 1")
        row = cur.fetchone()
    if row is None:
        raise ValueError(
            "No users found in database. Create a user first or pass --user-id."
        )
    return str(row[0])


def _get_app_s3_client() -> tuple[Any, str]:
    """Create a boto3 client for the application's MinIO/S3 (not training S3).

    Returns:
        Tuple of (s3_client, bucket_name).
    """
    import boto3
    from botocore.config import Config as BotoConfig

    # Load api/.env for app S3 creds (same pattern as VideoResolver)
    api_env = Path(__file__).parents[3] / "api" / ".env"
    if api_env.exists():
        for line in api_env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value

    s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:9000")
    bucket = os.getenv("S3_BUCKET_NAME", "rallycut-dev")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        config=BotoConfig(signature_version="s3v4"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )
    return s3, bucket


def restore_dataset_to_db(
    dataset_dir: Path,
    name: str,
    user_id: str | None = None,
    dry_run: bool = False,
    upload_to_app_s3: bool = False,
) -> RestoreResult:
    """Restore a training dataset into the application database.

    Reads manifest.json and ground_truth.json from the dataset directory,
    inserts videos and rallies into the database, and creates a session
    for web editor access.

    Args:
        dataset_dir: Path to the dataset directory (e.g., training_datasets/beach_v2/).
        name: Dataset name (used for session naming).
        user_id: User ID to assign videos to. If None, auto-detects first user.
        dry_run: If True, preview what would happen without making changes.
        upload_to_app_s3: If True, upload video files to the app's MinIO/S3.

    Returns:
        RestoreResult with counts and any errors.
    """
    result = RestoreResult()

    # Load dataset files
    manifest_path = dataset_dir / "manifest.json"
    ground_truth_path = dataset_dir / "ground_truth.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {dataset_dir}")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"ground_truth.json not found in {dataset_dir}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)

    # Build lookup: video_id -> rallies
    gt_by_video: dict[str, list[dict[str, int]]] = {}
    for entry in ground_truth:
        gt_by_video[entry["video_id"]] = entry.get("rallies", [])

    videos = manifest.get("videos", [])
    if not videos:
        return result

    # Cache dir for finding local video files
    cache_dir = Path(user_cache_dir("rallycut")) / "evaluation"

    # Initialize app S3 client once if needed
    app_s3 = None
    app_bucket = ""
    if upload_to_app_s3:
        app_s3, app_bucket = _get_app_s3_client()

    conn = get_connection()
    try:
        # Auto-detect user_id if not provided
        if user_id is None:
            user_id = _get_default_user_id(conn)

        if dry_run:
            _preview_restore(conn, videos, gt_by_video, name, user_id, dataset_dir)
            conn.rollback()
            return result

        with conn.transaction():
            # Process each video
            video_id_map: dict[str, str] = {}  # old_video_id -> new_or_existing db id
            inserted_video_ids: set[str] = set()  # old_video_ids that were newly inserted

            for video_info in videos:
                content_hash = video_info["content_hash"]
                filename = video_info["filename"]
                old_video_id = video_info["video_id"]
                duration_ms = video_info.get("duration_ms")

                # Check if video already exists by content_hash
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM videos WHERE content_hash = %s AND deleted_at IS NULL LIMIT 1",
                        (content_hash,),
                    )
                    existing = cur.fetchone()

                if existing:
                    video_id_map[old_video_id] = str(existing[0])
                    result.videos_skipped += 1
                    continue

                # Generate new video record
                new_video_id = str(uuid.uuid4())
                s3_key = f"videos/{user_id}/{new_video_id}/{filename}"

                # Find local video file for size info
                local_path = _find_video_file(dataset_dir, cache_dir, content_hash, filename)
                file_size = local_path.stat().st_size if local_path else None

                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO videos (
                            id, user_id, name, filename, s3_key, content_hash,
                            status, duration_ms, file_size_bytes,
                            processing_status, created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s,
                            'DETECTED', %s, %s,
                            'SKIPPED', NOW(), NOW()
                        )
                        """,
                        (
                            new_video_id,
                            user_id,
                            filename,
                            filename,
                            s3_key,
                            content_hash,
                            duration_ms,
                            file_size,
                        ),
                    )

                video_id_map[old_video_id] = new_video_id
                inserted_video_ids.add(old_video_id)
                result.videos_inserted += 1

                # Upload to app S3 if requested
                if app_s3 and local_path:
                    try:
                        app_s3.upload_file(str(local_path), app_bucket, s3_key)
                    except Exception as e:
                        result.errors.append(f"S3 upload failed for {filename}: {e}")

            # Restore court calibration for all videos (new and existing)
            for video_info in videos:
                court_cal = video_info.get("court_calibration")
                if court_cal is not None:
                    old_video_id = video_info["video_id"]
                    if old_video_id in video_id_map:
                        db_video_id = video_id_map[old_video_id]
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE videos SET court_calibration_json = %s WHERE id = %s",
                                (json.dumps(court_cal), db_video_id),
                            )

            # Insert rallies only for newly inserted videos (skip existing to avoid duplicates)
            rally_params: list[tuple[str, str, int, int, int]] = []
            for video_info in videos:
                old_video_id = video_info["video_id"]
                if old_video_id not in inserted_video_ids:
                    continue

                db_video_id = video_id_map[old_video_id]
                rallies = gt_by_video.get(old_video_id, [])
                for order, rally in enumerate(rallies):
                    rally_params.append((
                        str(uuid.uuid4()),
                        db_video_id,
                        rally["start_ms"],
                        rally["end_ms"],
                        order,
                    ))

            if rally_params:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO rallies (
                            id, video_id, start_ms, end_ms,
                            confidence, status, "order",
                            created_at, updated_at
                        ) VALUES (
                            %s, %s, %s, %s,
                            NULL, 'CONFIRMED', %s,
                            NOW(), NOW()
                        )
                        """,
                        rally_params,
                    )
                result.rallies_inserted = len(rally_params)

            # Create session for web editor access
            session_id = str(uuid.uuid4())
            session_name = f"Training: {name}"

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, user_id, name, type, created_at, updated_at)
                    VALUES (%s, %s, %s, 'REGULAR', NOW(), NOW())
                    """,
                    (session_id, user_id, session_name),
                )

            # Link videos to session
            sv_params: list[tuple[str, str, str, int]] = []
            for order, video_info in enumerate(videos):
                old_video_id = video_info["video_id"]
                if old_video_id not in video_id_map:
                    continue
                sv_params.append((
                    str(uuid.uuid4()),
                    session_id,
                    video_id_map[old_video_id],
                    order,
                ))

            if sv_params:
                with conn.cursor() as cur:
                    cur.executemany(
                        """
                        INSERT INTO session_videos (id, session_id, video_id, "order", added_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        ON CONFLICT (session_id, video_id) DO NOTHING
                        """,
                        sv_params,
                    )

            result.session_created = session_name

            # Restore tracking ground truth if available
            tracking_gt_path = dataset_dir / "tracking_ground_truth.json"
            if tracking_gt_path.exists():
                _restore_tracking_gt(conn, tracking_gt_path, result)

            # Restore action ground truth if available
            action_gt_path = dataset_dir / "action_ground_truth.json"
            if action_gt_path.exists():
                _restore_action_gt(conn, action_gt_path, result)

    finally:
        conn.close()

    return result


def _preview_restore(
    conn: psycopg.Connection[tuple[Any, ...]],
    videos: list[dict[str, Any]],
    gt_by_video: dict[str, list[dict[str, int]]],
    name: str,
    user_id: str,
    dataset_dir: Path,
) -> None:
    """Print a preview of what restore would do (for --dry-run)."""
    from rich import print as rprint

    rprint("\n[bold]Dry Run Preview[/bold] (no changes will be made)\n")
    rprint(f"  User ID: [cyan]{user_id}[/cyan]")
    rprint(f"  Session: [cyan]Training: {name}[/cyan]")
    rprint()

    would_insert = 0
    would_skip = 0
    total_rallies = 0

    for video_info in videos:
        content_hash = video_info["content_hash"]
        filename = video_info["filename"]
        old_video_id = video_info["video_id"]

        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM videos WHERE content_hash = %s AND deleted_at IS NULL LIMIT 1",
                (content_hash,),
            )
            existing = cur.fetchone()

        rallies = gt_by_video.get(old_video_id, [])

        if existing:
            rprint(f"  [yellow]SKIP[/yellow] {filename} (already in DB)")
            would_skip += 1
        else:
            rprint(f"  [green]INSERT[/green] {filename} ({len(rallies)} rallies)")
            would_insert += 1

        total_rallies += len(rallies)

    rprint()
    rprint(f"  Would insert: [green]{would_insert}[/green] videos")
    rprint(f"  Would skip: [yellow]{would_skip}[/yellow] videos (already exist)")
    rprint(f"  Would insert: [green]{total_rallies}[/green] rally annotations")
    rprint(f"  Would create session: [cyan]Training: {name}[/cyan]")

    # Show tracking GT info if available
    tracking_gt_path = dataset_dir / "tracking_ground_truth.json"
    if tracking_gt_path.exists():
        with open(tracking_gt_path) as f:
            tracking_gt = json.load(f)
        tgt_stats = tracking_gt.get("stats", {})
        rprint(
            f"  Would restore: [green]{tgt_stats.get('total_rallies_with_tracking_gt', 0)}[/green]"
            f" tracking GT annotations"
        )

    # Show action GT info if available
    action_gt_path = dataset_dir / "action_ground_truth.json"
    if action_gt_path.exists():
        with open(action_gt_path) as f:
            action_gt = json.load(f)
        agt_stats = action_gt.get("stats", {})
        rprint(
            f"  Would restore: [green]{agt_stats.get('total_rallies_with_action_gt', 0)}[/green]"
            f" action GT annotations"
        )


def _restore_tracking_gt(
    conn: psycopg.Connection[tuple[Any, ...]],
    tracking_gt_path: Path,
    result: RestoreResult,
) -> None:
    """Restore tracking ground truth from tracking_ground_truth.json.

    For each entry, matches a rally by video content_hash + start_ms + end_ms,
    then upserts into player_tracks with ground_truth_json.
    """
    with open(tracking_gt_path) as f:
        tracking_gt = json.load(f)

    rallies = tracking_gt.get("rallies", [])
    if not rallies:
        return

    restored = 0
    for entry in rallies:
        content_hash = entry["video_content_hash"]
        start_ms = entry["rally_start_ms"]
        end_ms = entry["rally_end_ms"]
        gt_json = entry["ground_truth_json"]

        with conn.cursor() as cur:
            # Find rally by video content_hash + timing
            cur.execute(
                """
                SELECT r.id
                FROM rallies r
                JOIN videos v ON v.id = r.video_id
                WHERE v.content_hash = %s
                  AND r.start_ms = %s
                  AND r.end_ms = %s
                  AND v.deleted_at IS NULL
                LIMIT 1
                """,
                (content_hash, start_ms, end_ms),
            )
            row = cur.fetchone()

        if row is None:
            result.errors.append(
                f"No matching rally for {content_hash[:8]}... "
                f"({start_ms}-{end_ms}ms)"
            )
            continue

        rally_id = str(row[0])
        gt_json_str = json.dumps(gt_json)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO player_tracks (id, rally_id, status, ground_truth_json, created_at)
                VALUES (gen_random_uuid(), %s, 'COMPLETED', %s::jsonb, NOW())
                ON CONFLICT (rally_id) DO UPDATE
                    SET ground_truth_json = EXCLUDED.ground_truth_json
                """,
                (rally_id, gt_json_str),
            )

        restored += 1

    result.tracking_gt_restored = restored


def _restore_action_gt(
    conn: psycopg.Connection[tuple[Any, ...]],
    action_gt_path: Path,
    result: RestoreResult,
) -> None:
    """Restore action ground truth from action_ground_truth.json.

    For each entry, matches a rally by video content_hash + start_ms + end_ms,
    then upserts action_ground_truth_json on player_tracks.
    """
    with open(action_gt_path) as f:
        action_gt = json.load(f)

    rallies = action_gt.get("rallies", [])
    if not rallies:
        return

    restored = 0
    for entry in rallies:
        content_hash = entry["video_content_hash"]
        start_ms = entry["rally_start_ms"]
        end_ms = entry["rally_end_ms"]
        gt_json = entry["action_ground_truth_json"]

        with conn.cursor() as cur:
            # Find rally by video content_hash + timing
            cur.execute(
                """
                SELECT r.id
                FROM rallies r
                JOIN videos v ON v.id = r.video_id
                WHERE v.content_hash = %s
                  AND r.start_ms = %s
                  AND r.end_ms = %s
                  AND v.deleted_at IS NULL
                LIMIT 1
                """,
                (content_hash, start_ms, end_ms),
            )
            row = cur.fetchone()

        if row is None:
            result.errors.append(
                f"Action GT: no matching rally for {content_hash[:8]}... "
                f"({start_ms}-{end_ms}ms)"
            )
            continue

        rally_id = str(row[0])
        gt_json_str = json.dumps(gt_json)

        with conn.cursor() as cur:
            # Update existing player_tracks row, or insert if none exists
            cur.execute(
                """
                UPDATE player_tracks
                SET action_ground_truth_json = %s::jsonb
                WHERE rally_id = %s
                """,
                (gt_json_str, rally_id),
            )
            if cur.rowcount == 0:
                # No player_tracks row exists — insert one
                cur.execute(
                    """
                    INSERT INTO player_tracks (id, rally_id, status, action_ground_truth_json, created_at)
                    VALUES (gen_random_uuid(), %s, 'COMPLETED', %s::jsonb, NOW())
                    """,
                    (rally_id, gt_json_str),
                )

        restored += 1

    result.action_gt_restored = restored


def _find_video_file(
    dataset_dir: Path,
    cache_dir: Path,
    content_hash: str,
    filename: str,
) -> Path | None:
    """Find a video file locally, checking cache and dataset directory."""
    # Check cache first
    cache_path = cache_dir / f"{content_hash}.mp4"
    if cache_path.exists():
        return cache_path

    # Check dataset videos dir (resolve symlinks)
    video_path = dataset_dir / "videos" / filename
    if video_path.exists():
        return video_path.resolve()

    return None
