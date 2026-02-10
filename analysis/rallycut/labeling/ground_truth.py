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
    label: str  # "player", "player_1", "player_2", etc., or "ball"
    x: float  # Normalized center x (0-1)
    y: float  # Normalized center y (0-1)
    width: float  # Normalized width (0-1)
    height: float  # Normalized height (0-1)
    confidence: float = 1.0  # Ground truth is always 1.0

    @property
    def is_player(self) -> bool:
        """Check if this is a player annotation (player, player_1, etc.)."""
        return self.label == "player" or self.label.startswith("player_")

    @property
    def is_ball(self) -> bool:
        """Check if this is a ball annotation."""
        return self.label == "ball"

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
        """Get only player positions (player, player_1, player_2, etc.)."""
        return [p for p in self.positions if p.is_player]

    @property
    def ball_positions(self) -> list[GroundTruthPosition]:
        """Get only ball positions."""
        return [p for p in self.positions if p.is_ball]

    @property
    def unique_player_tracks(self) -> set[int]:
        """Get unique player track IDs."""
        return {p.track_id for p in self.positions if p.is_player}

    def interpolate(self, frame_count: int | None = None) -> GroundTruthResult:
        """Interpolate positions between keyframes for each track.

        Label Studio exports keyframes only. This method fills in positions
        between keyframes using linear interpolation, matching how Label Studio
        displays the annotations.

        Args:
            frame_count: Total frames to interpolate to. If None, uses self.frame_count.

        Returns:
            New GroundTruthResult with interpolated positions.
        """
        total_frames = frame_count if frame_count is not None else self.frame_count
        if total_frames == 0:
            return self

        # Group positions by (track_id, label)
        tracks: dict[tuple[int, str], list[GroundTruthPosition]] = {}
        for pos in self.positions:
            key = (pos.track_id, pos.label)
            if key not in tracks:
                tracks[key] = []
            tracks[key].append(pos)

        # Interpolate each track
        interpolated: list[GroundTruthPosition] = []

        for (track_id, label), keyframes in tracks.items():
            # Sort by frame number
            keyframes = sorted(keyframes, key=lambda p: p.frame_number)

            if len(keyframes) == 1:
                # Single keyframe - just add it
                interpolated.append(keyframes[0])
                continue

            # Interpolate between consecutive keyframes
            for i in range(len(keyframes) - 1):
                start = keyframes[i]
                end = keyframes[i + 1]

                # Add all frames from start to end (exclusive of end)
                for frame in range(start.frame_number, end.frame_number):
                    if frame == start.frame_number:
                        # Use exact keyframe
                        interpolated.append(start)
                    else:
                        # Linear interpolation
                        t = (frame - start.frame_number) / (end.frame_number - start.frame_number)
                        interpolated.append(
                            GroundTruthPosition(
                                frame_number=frame,
                                track_id=track_id,
                                label=label,
                                x=start.x + t * (end.x - start.x),
                                y=start.y + t * (end.y - start.y),
                                width=start.width + t * (end.width - start.width),
                                height=start.height + t * (end.height - start.height),
                                confidence=1.0,
                            )
                        )

            # Add the last keyframe
            interpolated.append(keyframes[-1])

        return GroundTruthResult(
            positions=interpolated,
            frame_count=total_frames,
            video_width=self.video_width,
            video_height=self.video_height,
        )

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
