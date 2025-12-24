"""Configuration management for RallyCut."""

from pathlib import Path
from typing import Optional

import yaml
from platformdirs import user_cache_dir
from pydantic import BaseModel, Field
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


def get_recommended_batch_size(device: str = "auto") -> int:
    """Get recommended batch size based on device capabilities.

    For CUDA devices, scales batch size based on available VRAM:
    - 8GB+ VRAM: batch_size=32
    - 12GB+ VRAM: batch_size=48
    - 16GB+ VRAM: batch_size=64

    For MPS/CPU, returns conservative default of 16.
    """
    if device == "auto":
        device = _detect_device()

    if device != "cuda":
        return 16  # Conservative default for MPS/CPU

    try:
        import torch
        if not torch.cuda.is_available():
            return 16

        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024**3)

        # Scale batch size based on VRAM
        # VideoMAE uses ~0.5GB per batch of 16 at FP16
        if vram_gb >= 16:
            return 64
        elif vram_gb >= 12:
            return 48
        elif vram_gb >= 8:
            return 32
        else:
            return 16
    except Exception:
        return 16


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


# =============================================================================
# Nested Configuration Classes
# =============================================================================


class MotionConfig(BaseModel):
    """Motion detection configuration."""

    analysis_size: tuple[int, int] = (320, 180)
    high_threshold: float = 0.08
    low_threshold: float = 0.04
    window_size: int = 5


class GameStateConfig(BaseModel):
    """VideoMAE game state classifier configuration."""

    window_size: int = 16
    analysis_size: tuple[int, int] = (224, 224)
    stride: int = 32  # Optimized: was 8, now 2.5x faster with same accuracy
    batch_size: int = 8


class TwoPassConfig(BaseModel):
    """Two-pass analyzer configuration."""

    motion_stride: int = 32
    ml_stride: int = 8
    motion_padding_seconds: float = 3.0
    boundary_seconds: float = 2.0
    # Motion thresholds optimized via parameter sweep for accurate rally detection
    motion_high_threshold: float = 0.04
    motion_low_threshold: float = 0.02


class BallTrackingConfig(BaseModel):
    """Ball tracking configuration."""

    confidence_threshold: float = 0.3
    max_missing_frames: int = 10
    edge_margin: int = 80
    # Kalman filter parameters
    kalman_measurement_noise: float = 10.0
    kalman_process_noise: float = 5.0


class TrajectoryConfig(BaseModel):
    """Trajectory processing configuration."""

    max_gap_frames: int = 15
    smooth_sigma: float = 1.5
    trail_length: int = 15


class OverlayConfig(BaseModel):
    """Overlay rendering configuration."""

    ball_color: tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    ball_radius: int = 12
    trail_color: tuple[int, int, int] = (0, 200, 255)  # Orange (BGR)
    trail_max_radius: int = 8
    trail_min_radius: int = 2
    predicted_color: tuple[int, int, int] = (100, 100, 255)  # Light red (BGR)
    draw_trail_line: bool = True
    trail_line_color: tuple[int, int, int] = (0, 150, 200)
    trail_line_thickness: int = 2


class ProxyConfig(BaseModel):
    """Proxy video generation configuration."""

    enabled: bool = True
    height: int = 480
    fps: int = 30  # Normalize to 30fps for optimal ML temporal dynamics


class SegmentConfig(BaseModel):
    """Segment processing configuration."""

    min_play_duration: float = 5.0
    padding_seconds: float = 1.0
    min_gap_seconds: float = 3.0


# =============================================================================
# Main Configuration Class
# =============================================================================


class RallyCutConfig(BaseSettings):
    """Configuration settings for RallyCut."""

    # Nested configuration groups
    motion: MotionConfig = Field(default_factory=MotionConfig)
    game_state: GameStateConfig = Field(default_factory=GameStateConfig)
    two_pass: TwoPassConfig = Field(default_factory=TwoPassConfig)
    ball_tracking: BallTrackingConfig = Field(default_factory=BallTrackingConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)

    # Model paths
    model_cache_dir: Path = Field(
        default_factory=lambda: Path(user_cache_dir("rallycut")) / "models"
    )

    # VideoMAE settings (default to local weights if available)
    videomae_model_path: Optional[Path] = Field(
        default_factory=lambda: _find_local_weights("weights/videomae/game_state_classifier")
    )

    # YOLO settings (default to local weights if available)
    action_detector_path: Optional[Path] = Field(
        default_factory=lambda: _find_local_weights("weights/yolov8/action_detector/best.pt")
    )
    ball_detector_path: Optional[Path] = Field(
        default_factory=lambda: _find_local_weights("weights/yolov8/ball_detector/best.pt")
    )
    yolo_confidence: float = 0.25

    # Processing settings
    chunk_duration: float = 300.0  # 5 minutes in seconds

    # Device settings
    device: str = Field(default_factory=_detect_device)

    # Proxy cache directory
    proxy_cache_dir: Path = Field(
        default_factory=lambda: Path(user_cache_dir("rallycut")) / "proxies"
    )

    model_config = SettingsConfigDict(
        env_prefix="RALLYCUT_",
        env_nested_delimiter="__",  # Allows RALLYCUT_MOTION__HIGH_THRESHOLD
    )

    # -------------------------------------------------------------------------
    # YAML Loading
    # -------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path) -> "RallyCutConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    @classmethod
    def find_and_load(cls) -> "RallyCutConfig":
        """Find and load config from standard locations."""
        locations = [
            Path.cwd() / "rallycut.yaml",
            Path.home() / ".config" / "rallycut" / "rallycut.yaml",
        ]

        for path in locations:
            if path.exists():
                return cls.from_yaml(path)

        # Fall back to defaults + environment variables
        return cls()


# =============================================================================
# Global Config Instance
# =============================================================================

_config: Optional[RallyCutConfig] = None


def get_config() -> RallyCutConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = RallyCutConfig.find_and_load()
    return _config


def set_config(config: RallyCutConfig) -> None:
    """Set global configuration instance."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset global configuration instance (useful for testing)."""
    global _config
    _config = None
