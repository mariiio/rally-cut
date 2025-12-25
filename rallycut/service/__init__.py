"""RallyCut cloud detection service."""

from rallycut.service.detection import DetectionService
from rallycut.service.schemas import (
    DetectedSegment,
    DetectionConfig,
    DetectionRequest,
    DetectionResponse,
    JobStatus,
    MatchStatistics,
    SegmentType,
    VideoMetadata,
)

__all__ = [
    "DetectionConfig",
    "DetectionRequest",
    "DetectionResponse",
    "DetectedSegment",
    "DetectionService",
    "JobStatus",
    "MatchStatistics",
    "SegmentType",
    "VideoMetadata",
]
