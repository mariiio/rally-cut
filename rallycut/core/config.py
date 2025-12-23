"""Configuration management for RallyCut."""

from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _detect_device() -> str:
    """Detect best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _find_local_weights(relative_path: str) -> Optional[Path]:
    """Find local weights relative to project root."""
    # Try relative to cwd
    cwd_path = Path.cwd() / relative_path
    if cwd_path.exists():
        return cwd_path

    # Try relative to this file's location (rallycut/core/config.py -> project root)
    project_root = Path(__file__).parent.parent.parent
    root_path = project_root / relative_path
    if root_path.exists():
        return root_path

    return None


class RallyCutConfig(BaseSettings):
    """Configuration settings for RallyCut."""

    # Model paths
    model_cache_dir: Path = Field(
        default_factory=lambda: Path(user_cache_dir("rallycut")) / "models"
    )

    # VideoMAE settings (default to local weights if available)
    videomae_model_path: Optional[Path] = Field(
        default_factory=lambda: _find_local_weights("weights/videomae/game_state_classifier")
    )
    videomae_window_size: int = 16
    videomae_stride: int = 8

    # YOLO settings (default to local weights if available)
    action_detector_path: Optional[Path] = Field(
        default_factory=lambda: _find_local_weights("weights/yolov8/action_detector/best.pt")
    )
    ball_detector_path: Optional[Path] = Field(
        default_factory=lambda: _find_local_weights("weights/yolov8/ball_detector/best.pt")
    )
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.45

    # Processing settings
    chunk_duration: float = 300.0  # 5 minutes in seconds
    batch_size: int = 8
    min_play_duration: float = 2.0  # Minimum rally duration to keep
    padding_seconds: float = 1.0  # Padding before/after play segments

    # Device settings
    device: str = Field(default_factory=_detect_device)

    model_config = SettingsConfigDict(env_prefix="RALLYCUT_")


# Global config instance
_config: Optional[RallyCutConfig] = None


def get_config() -> RallyCutConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = RallyCutConfig()
    return _config


def set_config(config: RallyCutConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
