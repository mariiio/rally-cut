"""RallyCut - Beach volleyball video analysis CLI."""

__version__ = "0.1.0"

# Core exports for library usage
from rallycut.core.config import RallyCutConfig, get_config
from rallycut.core.models import GameState, TimeSegment
from rallycut.processing.cutter import VideoCutter
from rallycut.processing.highlights import HighlightGenerator, HighlightScorer

__all__ = [
    "GameState",
    "HighlightGenerator",
    "HighlightScorer",
    "RallyCutConfig",
    "TimeSegment",
    "VideoCutter",
    "get_config",
]
