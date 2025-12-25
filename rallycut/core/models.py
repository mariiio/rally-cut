"""Core domain models for RallyCut."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class GameState(str, Enum):
    """Classification of video segment game state."""

    SERVICE = "service"
    PLAY = "play"
    NO_PLAY = "no_play"


class ActionType(str, Enum):
    """Types of volleyball actions."""

    SERVE = "serve"
    RECEPTION = "reception"
    SET = "set"
    ATTACK = "attack"
    BLOCK = "block"
    BALL = "ball"


@dataclass
class BoundingBox:
    """Bounding box coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.y2 - self.y1


@dataclass
class VideoInfo:
    """Video file metadata."""

    path: Path
    duration: float
    fps: float
    width: int
    height: int
    frame_count: int
    codec: str = "unknown"

    @property
    def resolution(self) -> tuple[int, int]:
        """Get resolution as (width, height) tuple."""
        return (self.width, self.height)


@dataclass
class GameStateResult:
    """Result of game state classification for a frame window."""

    state: GameState
    confidence: float
    start_frame: int | None = None
    end_frame: int | None = None
    frame_idx: int | None = None

    def __post_init__(self):
        """Handle flexible initialization."""
        # If only frame_idx provided, use it for start/end
        if self.frame_idx is not None and self.start_frame is None:
            self.start_frame = self.frame_idx
        if self.frame_idx is not None and self.end_frame is None:
            self.end_frame = self.frame_idx
        # If only start/end provided, set frame_idx to start
        if self.frame_idx is None and self.start_frame is not None:
            self.frame_idx = self.start_frame

    @property
    def frame_range(self) -> tuple[int, int]:
        """Get frame range as tuple."""
        return (self.start_frame, self.end_frame)


@dataclass
class TimeSegment:
    """A time segment in the video with a game state."""

    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    state: GameState

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        return self.end_time - self.start_time

    @property
    def frame_count(self) -> int:
        """Get number of frames (inclusive)."""
        return self.end_frame - self.start_frame + 1


@dataclass
class Action:
    """A detected volleyball action."""

    action_type: ActionType
    frame_idx: int
    timestamp: float
    confidence: float
    bbox: BoundingBox | None = None

    @property
    def position(self) -> tuple[float, float] | None:
        """Get center position if bbox available."""
        if self.bbox:
            return self.bbox.center
        return None


@dataclass
class BallPosition:
    """Ball position at a specific frame."""

    frame_idx: int
    x: float
    y: float
    confidence: float
    is_predicted: bool = False


@dataclass
class Rally:
    """A rally (sequence of play from service to point end)."""

    rally_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    actions: list[Action] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Get rally duration in seconds."""
        return self.end_time - self.start_time

    @property
    def action_sequence(self) -> list[ActionType]:
        """Get ordered list of action types in rally."""
        sorted_actions = sorted(self.actions, key=lambda a: a.frame_idx)
        return [a.action_type for a in sorted_actions]


@dataclass
class ActionCount:
    """Count of actions with optional success tracking."""

    count: int
    success_count: int | None = None

    @property
    def success_rate(self) -> float | None:
        """Calculate success rate if success_count is set."""
        if self.success_count is None or self.count == 0:
            return None
        return self.success_count / self.count


@dataclass
class MatchStatistics:
    """Aggregated statistics for a match."""

    video_info: VideoInfo
    total_rallies: int
    total_duration: float
    play_duration: float
    dead_time_duration: float
    serves: ActionCount = field(default_factory=lambda: ActionCount(0))
    receptions: ActionCount = field(default_factory=lambda: ActionCount(0))
    sets: ActionCount = field(default_factory=lambda: ActionCount(0))
    attacks: ActionCount = field(default_factory=lambda: ActionCount(0))
    blocks: ActionCount = field(default_factory=lambda: ActionCount(0))
    avg_rally_duration: float = 0.0
    longest_rally_duration: float = 0.0
    shortest_rally_duration: float = 0.0
    touches_per_rally: float = 0.0

    @property
    def dead_time_percentage(self) -> float:
        """Calculate percentage of dead time."""
        if self.total_duration == 0:
            return 0.0
        return (self.dead_time_duration / self.total_duration) * 100
