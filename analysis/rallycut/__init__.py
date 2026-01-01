"""RallyCut - Beach volleyball video analysis CLI."""

__version__ = "0.1.0"

# Core exports for library usage
from rallycut.core.config import RallyCutConfig, get_config
from rallycut.core.models import GameState, TimeSegment
from rallycut.processing.cutter import VideoCutter

__all__ = [
    "GameState",
    "RallyCutConfig",
    "TimeSegment",
    "VideoCutter",
    "get_config",
]
