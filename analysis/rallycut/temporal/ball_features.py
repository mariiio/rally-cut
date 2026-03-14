"""Ball trajectory feature extraction for temporal modeling.

Extracts per-window ball detection features from cached WASB density data,
aligned to the same window grid as VideoMAE features. These features are
concatenated with VideoMAE features to give TemporalMaxer additional signal
for detecting short rallies.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# 5 features per window: detection_rate, mean_conf, max_conf, std_conf, ramp
BALL_FEATURE_DIM = 5

DEFAULT_BALL_DENSITY_DIR = Path("training_data/ball_density")  # relative to analysis/ cwd
WASB_CONF_THRESHOLD = 0.3


def load_ball_density(
    video_id: str,
    ball_density_dir: Path = DEFAULT_BALL_DENSITY_DIR,
) -> tuple[np.ndarray, float] | None:
    """Load cached per-frame ball confidences for a video.

    Args:
        video_id: Database video ID (NPZ files are keyed by this).
        ball_density_dir: Directory containing NPZ files.

    Returns:
        (confidences, fps) or None if not cached.
    """
    npz_path = ball_density_dir / f"{video_id}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return data["confidences"].astype(np.float32), float(data["fps"])


def extract_ball_features(
    confidences: np.ndarray,
    ball_fps: float,
    feature_fps: float,
    stride: int,
    window_size: int = 16,
) -> np.ndarray:
    """Extract per-window ball features aligned to VideoMAE windows.

    Each window covers `window_size` frames at `feature_fps`. Ball confidences
    are subsampled if recorded at higher fps than feature extraction.

    Args:
        confidences: Per-frame ball detection confidences from WASB.
        ball_fps: FPS of the ball detection data.
        feature_fps: Effective FPS used by VideoMAE (typically ~30).
        stride: Window stride in frames (same as VideoMAE feature stride).
        window_size: Window size in frames (default 16).

    Returns:
        Array of shape (num_windows, BALL_FEATURE_DIM).
    """
    # Subsample ball confidences to match feature fps
    subsample = max(1, round(ball_fps / feature_fps))
    if subsample > 1:
        confidences = confidences[::subsample]

    # Compute number of windows (same formula as VideoMAE feature extraction)
    n_frames = len(confidences)
    if n_frames < window_size:
        return np.zeros((0, BALL_FEATURE_DIM), dtype=np.float32)

    num_windows = (n_frames - window_size) // stride + 1
    features = np.zeros((num_windows, BALL_FEATURE_DIM), dtype=np.float32)

    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        window = confidences[start:end]

        detected = window >= WASB_CONF_THRESHOLD

        # Feature 0: detection rate (fraction of frames with ball detected)
        features[i, 0] = detected.mean()
        # Feature 1: mean confidence
        features[i, 1] = window.mean()
        # Feature 2: max confidence
        features[i, 2] = window.max()
        # Feature 3: confidence std
        features[i, 3] = window.std()
        # Feature 4: confidence ramp (second half mean - first half mean)
        half = window_size // 2
        features[i, 4] = window[half:].mean() - window[:half].mean()

    return features


def combine_features(
    video_features: np.ndarray,
    ball_features: np.ndarray,
) -> np.ndarray:
    """Concatenate VideoMAE and ball features along feature dimension.

    Truncates to the shorter sequence length to handle rounding differences.

    Args:
        video_features: Shape (N, D).
        ball_features: Shape (M, BALL_FEATURE_DIM).

    Returns:
        Shape (min(N, M), D + BALL_FEATURE_DIM).
    """
    min_len = min(len(video_features), len(ball_features))
    return np.concatenate(
        [video_features[:min_len], ball_features[:min_len]],
        axis=1,
    ).astype(np.float32)
