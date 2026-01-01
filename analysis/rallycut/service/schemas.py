"""Pydantic models for the RallyCut detection API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SegmentType(str, Enum):
    """Classification of detected video segments."""

    RALLY = "rally"  # PLAY or SERVICE state - active gameplay
    DEAD_TIME = "dead_time"  # NO_PLAY state - between rallies


class DetectedSegment(BaseModel):
    """A detected video segment."""

    segment_id: int = Field(description="Sequential segment identifier (1-indexed)")
    segment_type: SegmentType = Field(description="Type of segment (rally or dead_time)")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    start_frame: int = Field(description="Start frame number")
    end_frame: int = Field(description="End frame number")
    duration: float = Field(description="Duration in seconds")


class VideoMetadata(BaseModel):
    """Source video metadata."""

    source_url: str = Field(description="Original video URL")
    duration: float = Field(description="Total duration in seconds")
    fps: float = Field(description="Frames per second")
    width: int = Field(description="Video width in pixels")
    height: int = Field(description="Video height in pixels")
    frame_count: int = Field(description="Total number of frames")


class MatchStatistics(BaseModel):
    """Aggregated match statistics from detection."""

    total_segments: int = Field(description="Total number of rally segments detected")
    rally_count: int = Field(description="Number of rallies (same as total_segments)")
    total_play_duration: float = Field(description="Total play time in seconds")
    total_dead_time: float = Field(description="Total dead time in seconds")
    play_percentage: float = Field(
        ge=0.0, le=100.0, description="Percentage of video that is active play"
    )
    avg_rally_duration: float = Field(description="Average rally duration in seconds")
    longest_rally_duration: float = Field(description="Longest rally duration in seconds")
    shortest_rally_duration: float = Field(description="Shortest rally duration in seconds")


class DetectionConfig(BaseModel):
    """Optional configuration overrides for detection."""

    min_play_duration: float = Field(
        default=5.0, ge=0.0, description="Minimum rally duration to include (seconds)"
    )
    padding_seconds: float = Field(
        default=1.0, ge=0.0, description="Padding before/after each rally (seconds)"
    )
    min_gap_seconds: float = Field(
        default=3.0,
        ge=0.0,
        description="Minimum NO_PLAY gap before ending a rally (seconds)",
    )
    use_proxy: bool = Field(
        default=True, description="Use 480p proxy for faster ML analysis"
    )
    stride: int = Field(
        default=32, ge=1, description="Frame sampling interval for analysis"
    )


class DetectionRequest(BaseModel):
    """API request for rally detection."""

    video_url: str = Field(description="Video URL (S3/GCS presigned or public HTTPS)")
    config: DetectionConfig | None = Field(
        default=None, description="Optional detection configuration overrides"
    )
    callback_url: str | None = Field(
        default=None, description="Webhook URL for async completion notification"
    )
    job_id: str | None = Field(
        default=None, description="Client-provided job ID (auto-generated if not provided)"
    )


class DetectionResponse(BaseModel):
    """API response with detection results."""

    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status: completed, failed, or processing")
    video: VideoMetadata | None = Field(
        default=None, description="Source video metadata"
    )
    segments: list[DetectedSegment] = Field(
        default_factory=list, description="Detected rally segments"
    )
    statistics: MatchStatistics | None = Field(
        default=None, description="Aggregated match statistics"
    )
    processing_time_seconds: float = Field(
        default=0.0, description="Total processing time in seconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp (UTC)"
    )
    error: str | None = Field(default=None, description="Error message if failed")


class JobStatus(BaseModel):
    """Status response for async job polling."""

    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Job status: pending, processing, completed, failed")
    progress: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Progress (0.0-1.0)"
    )
    result_url: str | None = Field(
        default=None, description="URL to fetch results when complete"
    )
    error: str | None = Field(default=None, description="Error message if failed")
