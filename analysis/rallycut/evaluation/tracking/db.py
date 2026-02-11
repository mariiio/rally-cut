"""Database loader for tracking ground truth and predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.db import get_connection
from rallycut.labeling.ground_truth import GroundTruthPosition, GroundTruthResult
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import (
    BallPhaseInfo,
    PlayerPosition,
    PlayerTrackingResult,
    ServerInfo,
)

logger = logging.getLogger(__name__)

# Videos with validated ball tracking ground truth
# Other videos have ground truth labels that were found to be inaccurate
# and should not be used for evaluation
VALID_BALL_GT_VIDEOS = {
    "a5866029-7cf4-42d6-adc2-8e28111ffd81",
    "1efa35cf-4edd-4504-b4a4-834eee9e5218",
    "70ab9d7f-8cc4-48cb-892a-1c36793cac72",
}


@dataclass
class TrackingEvaluationRally:
    """A rally with ground truth and predictions for evaluation."""

    rally_id: str
    video_id: str
    start_ms: int
    end_ms: int
    ground_truth: GroundTruthResult
    predictions: PlayerTrackingResult | None
    # Video metadata
    video_fps: float = 30.0
    video_width: int = 1920
    video_height: int = 1080
    # Raw positions before filtering (for parameter tuning)
    raw_positions: list[PlayerPosition] | None = None


def _parse_ground_truth(
    gt_json: dict[str, Any] | None,
    frame_offset: int = 0,
    fps_scale: float = 1.0,
) -> GroundTruthResult | None:
    """Parse ground truth JSON from database into GroundTruthResult.

    Args:
        gt_json: Ground truth JSON from database.
        frame_offset: Frame number to subtract from all positions to convert
            from absolute video frames to rally-relative frames. This should
            be calculated at Label Studio's FPS rate (30fps).
        fps_scale: Scale factor for frame numbers to match prediction FPS.
            E.g., 2.0 when predictions are at 60fps but GT was labeled at 30fps.

    Returns:
        Parsed GroundTruthResult with rally-relative frame numbers scaled to
        match prediction FPS.
    """
    if gt_json is None:
        return None

    # Label Studio exports 1-indexed frames at 30fps, predictions may be at different FPS
    # 1. Subtract frame_offset (at Label Studio 30fps) to get rally-relative frames
    # 2. Subtract 1 for 1-indexed to 0-indexed conversion
    # 3. Scale by fps_scale to match prediction FPS (e.g., 2x for 60fps predictions)
    positions = [
        GroundTruthPosition(
            frame_number=int((p["frameNumber"] - frame_offset - 1) * fps_scale),
            track_id=p["trackId"],
            label=p["label"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in gt_json.get("positions", [])
    ]

    return GroundTruthResult(
        positions=positions,
        frame_count=gt_json.get("frameCount", 0),
        video_width=gt_json.get("videoWidth", 0),
        video_height=gt_json.get("videoHeight", 0),
    )


def _parse_predictions(
    positions_json: list[dict[str, Any]] | None,
    pt_data: dict[str, Any],
) -> PlayerTrackingResult | None:
    """Parse predictions JSON from database into PlayerTrackingResult."""
    if positions_json is None:
        return None

    # positions_json is just the positions array in the database
    # Other fields come from the PlayerTrack row
    positions = [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            confidence=p["confidence"],
        )
        for p in positions_json
    ]

    # Parse ball positions if available
    ball_positions: list[BallPosition] = []
    ball_json = pt_data.get("ball_positions_json")
    if ball_json:
        for bp in ball_json:
            ball_positions.append(
                BallPosition(
                    frame_number=bp["frameNumber"],
                    x=bp["x"],
                    y=bp["y"],
                    confidence=bp["confidence"],
                )
            )

    # Parse ball phases if available
    ball_phases: list[BallPhaseInfo] = []
    phases_json = pt_data.get("ball_phases_json")
    if phases_json:
        for phase in phases_json:
            ball_phases.append(
                BallPhaseInfo(
                    phase=phase["phase"],
                    frame_start=phase["frameStart"],
                    frame_end=phase["frameEnd"],
                    velocity=phase["velocity"],
                    ball_x=phase["ballX"],
                    ball_y=phase["ballY"],
                )
            )

    # Parse server info if available
    server_info: ServerInfo | None = None
    server_json = pt_data.get("server_info_json")
    if server_json:
        server_info = ServerInfo(
            track_id=server_json["trackId"],
            confidence=server_json["confidence"],
            serve_frame=server_json["serveFrame"],
            serve_velocity=server_json["serveVelocity"],
            is_near_court=server_json["isNearCourt"],
        )

    return PlayerTrackingResult(
        positions=positions,
        frame_count=pt_data.get("frame_count") or 0,
        video_fps=pt_data.get("fps") or 30.0,
        video_width=0,  # Not stored in PlayerTrack
        video_height=0,
        processing_time_ms=pt_data.get("processing_time_ms") or 0.0,
        model_version=pt_data.get("model_version") or "yolov8n.pt",
        court_split_y=pt_data.get("court_split_y"),
        primary_track_ids=pt_data.get("primary_track_ids") or [],
        ball_phases=ball_phases,
        server_info=server_info,
        ball_positions=ball_positions,
    )


def load_labeled_rallies(
    video_id: str | None = None,
    rally_id: str | None = None,
    ball_gt_only: bool = False,
) -> list[TrackingEvaluationRally]:
    """Load rallies with ground truth labels from the database.

    Args:
        video_id: Filter by video ID (optional).
        rally_id: Filter by specific rally ID (optional).
        ball_gt_only: If True, only load rallies from videos with validated
            ball tracking ground truth (see VALID_BALL_GT_VIDEOS).

    Returns:
        List of TrackingEvaluationRally with ground truth and predictions.
    """
    # Build query based on filters
    where_clauses = ["pt.ground_truth_json IS NOT NULL"]
    params: list[str] = []

    if video_id:
        where_clauses.append("r.video_id = %s")
        params.append(video_id)

    if rally_id:
        where_clauses.append("r.id = %s")
        params.append(rally_id)

    where_sql = " AND ".join(where_clauses)

    query = f"""
        SELECT
            r.id as rally_id,
            r.video_id,
            r.start_ms,
            r.end_ms,
            v.fps as video_fps,
            v.width as video_width,
            v.height as video_height,
            pt.ground_truth_json,
            pt.positions_json,
            pt.raw_positions_json,
            pt.frame_count,
            pt.fps as pt_fps,
            pt.processing_time_ms,
            pt.model_version,
            pt.court_split_y,
            pt.primary_track_ids,
            pt.ball_phases_json,
            pt.server_info_json,
            pt.ball_positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE {where_sql}
        ORDER BY r.video_id, r.start_ms
    """

    results: list[TrackingEvaluationRally] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

            for row in rows:
                (
                    rally_id_val,
                    video_id_val,
                    start_ms,
                    end_ms,
                    video_fps,
                    video_width,
                    video_height,
                    ground_truth_json,
                    positions_json,
                    raw_positions_json,
                    frame_count,
                    pt_fps,
                    processing_time_ms,
                    model_version,
                    court_split_y,
                    primary_track_ids,
                    ball_phases_json,
                    server_info_json,
                    ball_positions_json,
                ) = row

                # Calculate frame offset to convert GT absolute frames to rally-relative
                # GT frames are stored as absolute video frames at Label Studio's 30fps rate
                # Predictions are rally-relative at tracking FPS (pt_fps, may be 60fps)
                tracking_fps = cast(float, pt_fps) if pt_fps else 30.0
                label_studio_fps = 30.0  # Label Studio always uses 30fps for frame numbers
                start_frame_at_ls_fps = int(cast(int, start_ms) / 1000 * label_studio_fps)
                fps_scale = tracking_fps / label_studio_fps  # e.g., 2.0 for 60fps tracking

                # Parse ground truth (with type cast, frame offset at LS fps, and fps scale)
                gt_json_typed = cast(dict[str, Any] | None, ground_truth_json)
                gt = _parse_ground_truth(
                    gt_json_typed,
                    frame_offset=start_frame_at_ls_fps,
                    fps_scale=fps_scale,
                )
                if gt is None:
                    logger.warning(f"Rally {rally_id_val} has NULL ground_truth_json, skipping")
                    continue

                # Parse predictions (with type casts)
                positions_json_typed = cast(list[dict[str, Any]] | None, positions_json)
                pt_data: dict[str, Any] = {
                    "frame_count": frame_count,
                    "fps": pt_fps,
                    "processing_time_ms": processing_time_ms,
                    "model_version": model_version,
                    "court_split_y": court_split_y,
                    "primary_track_ids": primary_track_ids,
                    "ball_phases_json": ball_phases_json,
                    "server_info_json": server_info_json,
                    "ball_positions_json": ball_positions_json,
                }
                predictions = _parse_predictions(positions_json_typed, pt_data)

                # Parse raw positions for parameter tuning
                raw_positions: list[PlayerPosition] | None = None
                raw_json = cast(list[dict[str, Any]] | None, raw_positions_json)
                if raw_json:
                    raw_positions = [
                        PlayerPosition(
                            frame_number=p["frameNumber"],
                            track_id=p["trackId"],
                            x=p["x"],
                            y=p["y"],
                            width=p["width"],
                            height=p["height"],
                            confidence=p["confidence"],
                        )
                        for p in raw_json
                    ]

                results.append(
                    TrackingEvaluationRally(
                        rally_id=str(rally_id_val),
                        video_id=str(video_id_val),
                        start_ms=cast(int, start_ms),
                        end_ms=cast(int, end_ms),
                        ground_truth=gt,
                        predictions=predictions,
                        video_fps=cast(float, video_fps) if video_fps else 30.0,
                        video_width=cast(int, video_width) if video_width else 1920,
                        video_height=cast(int, video_height) if video_height else 1080,
                        raw_positions=raw_positions,
                    )
                )

    # Filter to only valid ball GT videos if requested
    if ball_gt_only:
        original_count = len(results)
        results = [r for r in results if r.video_id in VALID_BALL_GT_VIDEOS]
        if original_count > len(results):
            logger.info(
                f"Filtered {original_count - len(results)} rallies with "
                f"invalid ball ground truth"
            )

    logger.info(f"Loaded {len(results)} rallies with ground truth labels")
    return results


def get_video_path(video_id: str) -> Path | None:
    """Get the local path for a video by downloading from S3 if needed.

    Args:
        video_id: The video ID from the database.

    Returns:
        Path to the local video file, or None if video not found.
    """
    from rallycut.evaluation.video_resolver import VideoResolver

    query = """
        SELECT s3_key, content_hash
        FROM videos
        WHERE id = %s
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            row = cur.fetchone()

            if row is None:
                logger.warning(f"Video {video_id} not found in database")
                return None

            s3_key, content_hash = row
            if not s3_key or not content_hash:
                logger.warning(f"Video {video_id} missing s3_key or content_hash")
                return None

            # Use VideoResolver to download/cache the video
            resolver = VideoResolver()
            try:
                video_path = resolver.resolve(cast(str, s3_key), cast(str, content_hash))
                return video_path
            except Exception as e:
                logger.error(f"Failed to download video {video_id}: {e}")
                return None
