"""Configuration management for RallyCut."""

from pathlib import Path

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


def _find_local_weights(relative_path: str) -> Path | None:
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
# Model Variants and Presets
# =============================================================================

MODEL_VARIANTS: dict[str, str] = {
    "indoor": "weights/videomae/game_state_classifier",
    "beach": "weights/videomae/beach_volleyball",
}

# Heuristics presets for each model variant
# Indoor: original values from volleyball_analytics
# Beach: optimized for beach volleyball detection
MODEL_PRESETS: dict[str, dict[str, float]] = {
    "indoor": {
        "min_play_duration": 1.0,
        "rally_continuation_seconds": 2.0,
        "boundary_confidence_threshold": 0.35,
        "min_active_density": 0.25,
    },
    "beach": {
        "min_play_duration": 0.5,
        "rally_continuation_seconds": 3.0,
        "boundary_confidence_threshold": 0.25,
        "min_active_density": 0.15,
    },
}


def get_model_path(variant: str) -> Path | None:
    """Resolve model variant to weights path.

    Args:
        variant: Model variant name ('indoor' or 'beach')

    Returns:
        Path to model weights if found, None otherwise
    """
    relative_path = MODEL_VARIANTS.get(variant)
    if relative_path:
        return _find_local_weights(relative_path)
    return None


# =============================================================================
# Nested Configuration Classes
# =============================================================================


class GameStateConfig(BaseModel):
    """VideoMAE game state classifier configuration."""

    window_size: int = 16
    analysis_size: tuple[int, int] = (224, 224)
    stride: int = 48  # Optimized: 33% faster than 32, combined with MIN_ACTIVE_WINDOWS=1
    batch_size: int = 8
    # Temporal smoothing: fix isolated classification errors with median filter
    # Disabled by default as it can remove valid sparse detections
    enable_temporal_smoothing: bool = False
    # Window size for temporal smoothing (must be odd)
    temporal_smoothing_window: int = 3


class HWAccelConfig(BaseModel):
    """Hardware acceleration configuration."""

    enabled: bool = False  # Disabled by default, enable with --hwaccel flag
    # Automatically detect best backend based on device
    auto_detect: bool = True


class ProxyConfig(BaseModel):
    """Proxy video generation configuration."""

    enabled: bool = True
    height: int = 480
    fps: int = 30  # Normalize to 30fps for optimal ML temporal dynamics


class SegmentConfig(BaseModel):
    """Segment processing configuration.

    Note: These are the default (indoor) values. Beach-specific heuristics
    are applied via MODEL_PRESETS when --model beach is selected.
    """

    # Minimum duration for a valid play segment
    min_play_duration: float = 1.0
    # Padding added before segment start
    padding_seconds: float = 2.0
    # Padding added after segment end
    padding_end_seconds: float = 3.0
    # Gap threshold for merging segments
    min_gap_seconds: float = 5.0
    # Rally continuation: keep rally active until N consecutive seconds of NO_PLAY
    rally_continuation_seconds: float = 2.0


class TrainingBackupConfig(BaseModel):
    """S3 backup configuration for training datasets.

    Uses the default AWS credential chain (separate from app MinIO).
    Set TRAINING_S3_BUCKET env var or configure in rallycut.yaml.
    """

    s3_bucket: str = ""
    s3_prefix: str = "training"
    s3_region: str = "us-east-1"


# =============================================================================
# Main Configuration Class
# =============================================================================


class RallyCutConfig(BaseSettings):
    """Configuration settings for RallyCut."""

    # Nested configuration groups
    game_state: GameStateConfig = Field(default_factory=GameStateConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    segment: SegmentConfig = Field(default_factory=SegmentConfig)
    hwaccel: HWAccelConfig = Field(default_factory=HWAccelConfig)
    training_backup: TrainingBackupConfig = Field(default_factory=TrainingBackupConfig)

    # Model paths
    model_cache_dir: Path = Field(
        default_factory=lambda: Path(user_cache_dir("rallycut")) / "models"
    )

    # VideoMAE settings (default to local weights if available)
    videomae_model_path: Path | None = Field(
        default_factory=lambda: _find_local_weights("weights/videomae/game_state_classifier")
    )

    # Device settings
    device: str = Field(default_factory=_detect_device)

    # Proxy cache directory
    proxy_cache_dir: Path = Field(
        default_factory=lambda: Path(user_cache_dir("rallycut")) / "proxies"
    )

    model_config = SettingsConfigDict(
        env_prefix="RALLYCUT_",
        env_nested_delimiter="__",  # Allows RALLYCUT_GAME_STATE__STRIDE
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

_config: RallyCutConfig | None = None


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
