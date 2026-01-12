"""Load ground truth rallies from database."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rallycut.evaluation.db import get_connection


@dataclass
class GroundTruthRally:
    """A rally from the database (either manual or ML-detected)."""

    id: str
    video_id: str
    start_ms: int
    end_ms: int
    confidence: float | None = None  # None = manually tagged

    @property
    def start_seconds(self) -> float:
        """Start time in seconds."""
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        """End time in seconds."""
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return (self.end_ms - self.start_ms) / 1000.0

    @property
    def is_manual(self) -> bool:
        """True if this rally was manually tagged (no ML confidence)."""
        return self.confidence is None


@dataclass
class EvaluationVideo:
    """A video with its ground truth and ML-detected rallies."""

    id: str
    filename: str
    s3_key: str
    content_hash: str
    duration_ms: int | None
    ground_truth_rallies: list[GroundTruthRally] = field(default_factory=list)
    ml_detected_rallies: list[GroundTruthRally] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float | None:
        """Duration in seconds."""
        return self.duration_ms / 1000.0 if self.duration_ms else None


def load_evaluation_videos(
    video_ids: list[str] | None = None,
    require_ground_truth: bool = True,
) -> list[EvaluationVideo]:
    """Load videos with both ground truth and ML rallies from database.

    Ground truth: rallies WHERE confidence IS NULL (manually tagged)
    ML detected: rallies WHERE confidence IS NOT NULL

    Args:
        video_ids: Optional list of video IDs to filter. If None, loads all.
        require_ground_truth: If True, only return videos that have ground truth rallies.

    Returns:
        List of EvaluationVideo with ground_truth_rallies and ml_detected_rallies populated.
    """
    with get_connection() as conn:
        # Query videos with rallies
        # Note: Table names in PostgreSQL are lowercase (videos, rallies)
        query = """
            SELECT
                v.id, v.filename, v.s3_key, v.content_hash, v.duration_ms,
                r.id as rally_id, r.start_ms, r.end_ms, r.confidence
            FROM videos v
            LEFT JOIN rallies r ON r.video_id = v.id
            WHERE v.deleted_at IS NULL
            AND v.status = 'DETECTED'
        """
        params: list[object] = []

        if video_ids:
            query += " AND v.id = ANY(%s)"
            params.append(video_ids)

        query += " ORDER BY v.id, r.start_ms"

        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    # Group by video
    videos_map: dict[str, EvaluationVideo] = {}

    for row in rows:
        # Unpack and cast row values from database
        # Row type is tuple[Any, ...] from psycopg
        r: tuple[Any, ...] = row
        vid = str(r[0])
        filename = str(r[1])
        s3_key = str(r[2])
        content_hash = str(r[3])
        duration_ms: int | None = int(r[4]) if r[4] is not None else None
        rally_id: Any = r[5]  # Can be None from LEFT JOIN
        start_ms_raw: Any = r[6]
        end_ms_raw: Any = r[7]
        confidence_raw: Any = r[8]

        if vid not in videos_map:
            videos_map[vid] = EvaluationVideo(
                id=vid,
                filename=filename,
                s3_key=s3_key,
                content_hash=content_hash,
                duration_ms=duration_ms,
                ground_truth_rallies=[],
                ml_detected_rallies=[],
            )

        # Skip if no rally (LEFT JOIN can return NULL rally)
        if rally_id is None:
            continue

        rally = GroundTruthRally(
            id=str(rally_id),
            video_id=vid,
            start_ms=int(start_ms_raw),
            end_ms=int(end_ms_raw),
            confidence=float(confidence_raw) if confidence_raw is not None else None,
        )

        if confidence_raw is None:
            # Manual rally = ground truth
            videos_map[vid].ground_truth_rallies.append(rally)
        else:
            # ML-detected rally
            videos_map[vid].ml_detected_rallies.append(rally)

    # Filter to videos that have ground truth if required
    if require_ground_truth:
        return [v for v in videos_map.values() if v.ground_truth_rallies]

    return list(videos_map.values())


def get_evaluation_video_ids() -> list[str]:
    """Get IDs of all videos that have ground truth rallies.

    Useful for quick listing without loading all rally data.
    """
    with get_connection() as conn:
        query = """
            SELECT DISTINCT v.id
            FROM videos v
            INNER JOIN rallies r ON r.video_id = v.id
            WHERE v.deleted_at IS NULL
            AND v.status = 'DETECTED'
            AND r.confidence IS NULL
        """
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [str(row[0]) for row in rows]
