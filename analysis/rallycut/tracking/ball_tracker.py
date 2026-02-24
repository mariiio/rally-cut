"""
Ball tracking data structures and factory function.

The ball tracking system uses WASB HRNet (see wasb_model.py) as its sole model.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default (and only) ball tracking model.
# Fine-tuned WASB HRNet: 86.4% match, 29.2px error on beach volleyball GT.
DEFAULT_BALL_MODEL = "wasb"


def get_available_ball_models() -> list[str]:
    """Return list of available ball tracking model IDs."""
    return ["wasb"]


@dataclass
class BallPosition:
    """Single ball detection result."""

    frame_number: int
    x: float  # Normalized 0-1 (relative to video width)
    y: float  # Normalized 0-1 (relative to video height)
    confidence: float  # Detection confidence 0-1
    motion_energy: float = 0.0  # Motion energy at ball position (0-1)

    def to_dict(self) -> dict:
        d = {
            "frameNumber": self.frame_number,
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
        }
        if self.motion_energy > 0:
            d["motionEnergy"] = self.motion_energy
        return d


@dataclass
class BallTrackingResult:
    """Complete ball tracking result for a video segment."""

    positions: list[BallPosition] = field(default_factory=list)
    frame_count: int = 0
    video_fps: float = 30.0
    video_width: int = 0
    video_height: int = 0
    processing_time_ms: float = 0.0
    model_version: str = "wasb"
    filtering_enabled: bool = False
    raw_positions: list[BallPosition] | None = None  # Before filtering (debug)

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with ball detected (confidence > 0.5)."""
        if self.frame_count == 0:
            return 0.0
        detected = sum(1 for p in self.positions if p.confidence > 0.5)
        return detected / self.frame_count

    def to_dict(self) -> dict:
        result = {
            "positions": [p.to_dict() for p in self.positions],
            "frameCount": self.frame_count,
            "videoFps": self.video_fps,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
            "detectionRate": self.detection_rate,
            "processingTimeMs": self.processing_time_ms,
            "modelVersion": self.model_version,
            "filteringEnabled": self.filtering_enabled,
        }
        if self.raw_positions is not None:
            result["rawPositions"] = [p.to_dict() for p in self.raw_positions]
        return result

    def to_json(self, path: Path) -> None:
        """Write result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "BallTrackingResult":
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            BallPosition(
                frame_number=p["frameNumber"],
                x=p["x"],
                y=p["y"],
                confidence=p["confidence"],
                motion_energy=p.get("motionEnergy", 0.0),
            )
            for p in data.get("positions", [])
        ]

        raw_positions = None
        if "rawPositions" in data:
            raw_positions = [
                BallPosition(
                    frame_number=p["frameNumber"],
                    x=p["x"],
                    y=p["y"],
                    confidence=p["confidence"],
                    motion_energy=p.get("motionEnergy", 0.0),
                )
                for p in data["rawPositions"]
            ]

        return cls(
            positions=positions,
            frame_count=data.get("frameCount", 0),
            video_fps=data.get("videoFps", 30.0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
            processing_time_ms=data.get("processingTimeMs", 0.0),
            model_version=data.get("modelVersion", "wasb"),
            filtering_enabled=data.get("filteringEnabled", False),
            raw_positions=raw_positions,
        )


def create_ball_tracker(
    model: str = DEFAULT_BALL_MODEL,
    **kwargs: Any,
) -> Any:
    """Factory function to create a ball tracker.

    Args:
        model: Model identifier. Only 'wasb' is supported.
        **kwargs: Additional keyword arguments passed to WASBBallTracker.
            Supported: device, threshold, weights_path.

    Returns:
        A WASBBallTracker instance.
    """
    if model == "wasb":
        from rallycut.tracking.wasb_model import WASBBallTracker

        return WASBBallTracker(
            weights_path=kwargs.get("weights_path"),
            device=kwargs.get("device"),
            threshold=kwargs.get("threshold", 0.3),
        )
    else:
        available = ", ".join(get_available_ball_models())
        raise ValueError(f"Unknown ball model '{model}'. Available: {available}")
