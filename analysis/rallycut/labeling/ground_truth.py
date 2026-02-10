"""Ground truth data structures for tracking evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GroundTruthPosition:
    """A single ground truth annotation."""

    frame_number: int
    track_id: int
    label: str  # "player" or "ball"
    x: float  # Normalized center x (0-1)
    y: float  # Normalized center y (0-1)
    width: float  # Normalized width (0-1)
    height: float  # Normalized height (0-1)
    confidence: float = 1.0  # Ground truth is always 1.0

    def to_dict(self) -> dict:
        return {
            "frameNumber": self.frame_number,
            "trackId": self.track_id,
            "label": self.label,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


@dataclass
class GroundTruthResult:
    """Complete ground truth annotations."""

    positions: list[GroundTruthPosition] = field(default_factory=list)
    frame_count: int = 0
    video_width: int = 0
    video_height: int = 0

    @property
    def player_positions(self) -> list[GroundTruthPosition]:
        """Get only player positions."""
        return [p for p in self.positions if p.label == "player"]

    @property
    def ball_positions(self) -> list[GroundTruthPosition]:
        """Get only ball positions."""
        return [p for p in self.positions if p.label == "ball"]

    @property
    def unique_player_tracks(self) -> set[int]:
        """Get unique player track IDs."""
        return {p.track_id for p in self.positions if p.label == "player"}

    def to_dict(self) -> dict:
        return {
            "positions": [p.to_dict() for p in self.positions],
            "frameCount": self.frame_count,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
        }

    def to_json(self, path: Path) -> None:
        """Write result to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> GroundTruthResult:
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            GroundTruthPosition(
                frame_number=p["frameNumber"],
                track_id=p["trackId"],
                label=p["label"],
                x=p["x"],
                y=p["y"],
                width=p["width"],
                height=p["height"],
                confidence=p.get("confidence", 1.0),
            )
            for p in data.get("positions", [])
        ]

        return cls(
            positions=positions,
            frame_count=data.get("frameCount", 0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
        )
