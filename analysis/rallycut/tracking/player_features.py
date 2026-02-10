"""
Player appearance feature extraction for cross-rally tracking.

Extracts visual features from player detections for consistent ID assignment:
- Skin tone (HSV) - most reliable for beach volleyball
- Jersey color (HSV) - when visible
- Body proportions - height, aspect ratio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Skin detection HSV ranges (tuned for beach volleyball in various lighting)
SKIN_HSV_LOWER = np.array([0, 20, 50], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([50, 255, 255], dtype=np.uint8)

# Minimum pixels for valid color extraction
MIN_SKIN_PIXELS = 50
MIN_JERSEY_PIXELS = 30


@dataclass
class PlayerAppearanceFeatures:
    """Appearance features extracted from a single player detection."""

    track_id: int
    frame_number: int

    # Skin tone (HSV) - most reliable for beach volleyball
    # None if not enough skin pixels detected
    skin_tone_hsv: tuple[float, float, float] | None = None
    skin_pixel_count: int = 0

    # Jersey/shorts color (HSV) - may not be visible
    jersey_color_hsv: tuple[float, float, float] | None = None
    jersey_pixel_count: int = 0

    # Body proportions
    bbox_height: float = 0.0  # Normalized bbox height
    bbox_aspect_ratio: float = 0.0  # width / height


@dataclass
class PlayerAppearanceProfile:
    """Aggregated appearance profile for a player across rallies."""

    player_id: int  # Consistent 1-4 across match

    # Average skin tone (HSV)
    avg_skin_tone_hsv: tuple[float, float, float] = (0.0, 0.0, 0.0)
    skin_sample_count: int = 0

    # Average jersey color (HSV) - may be None if not reliably detected
    avg_jersey_color_hsv: tuple[float, float, float] | None = None
    jersey_sample_count: int = 0

    # Body proportions
    avg_bbox_height: float = 0.0
    height_sample_count: int = 0

    # Team assignment (0=near court, 1=far court)
    team: int = 0

    # How many rallies this player has been seen in
    rally_count: int = 0

    def update_from_features(self, features: PlayerAppearanceFeatures) -> None:
        """Update profile with new appearance features."""
        # Update skin tone (running average)
        if features.skin_tone_hsv is not None and features.skin_pixel_count >= MIN_SKIN_PIXELS:
            if self.skin_sample_count == 0:
                self.avg_skin_tone_hsv = features.skin_tone_hsv
            else:
                # Weighted average (new sample weight decreases over time)
                weight = 1.0 / (self.skin_sample_count + 1)
                h1, s1, v1 = self.avg_skin_tone_hsv
                h2, s2, v2 = features.skin_tone_hsv
                # Hue is circular, handle wrap-around
                h_new = _circular_mean(h1, h2, weight)
                s_new = s1 * (1 - weight) + s2 * weight
                v_new = v1 * (1 - weight) + v2 * weight
                self.avg_skin_tone_hsv = (h_new, s_new, v_new)
            self.skin_sample_count += 1

        # Update jersey color
        if features.jersey_color_hsv is not None and features.jersey_pixel_count >= MIN_JERSEY_PIXELS:
            if self.jersey_sample_count == 0 or self.avg_jersey_color_hsv is None:
                self.avg_jersey_color_hsv = features.jersey_color_hsv
            else:
                weight = 1.0 / (self.jersey_sample_count + 1)
                h1, s1, v1 = self.avg_jersey_color_hsv
                h2, s2, v2 = features.jersey_color_hsv
                h_new = _circular_mean(h1, h2, weight)
                s_new = s1 * (1 - weight) + s2 * weight
                v_new = v1 * (1 - weight) + v2 * weight
                self.avg_jersey_color_hsv = (h_new, s_new, v_new)
            self.jersey_sample_count += 1

        # Update height
        if features.bbox_height > 0:
            if self.height_sample_count == 0:
                self.avg_bbox_height = features.bbox_height
            else:
                weight = 1.0 / (self.height_sample_count + 1)
                self.avg_bbox_height = self.avg_bbox_height * (1 - weight) + features.bbox_height * weight
            self.height_sample_count += 1


@dataclass
class TrackAppearanceStats:
    """Aggregated appearance stats for a single track within a rally."""

    track_id: int
    features: list[PlayerAppearanceFeatures] = field(default_factory=list)

    # Computed averages
    avg_skin_tone_hsv: tuple[float, float, float] | None = None
    avg_jersey_color_hsv: tuple[float, float, float] | None = None
    avg_bbox_height: float = 0.0

    def compute_averages(self) -> None:
        """Compute average features from all samples."""
        if not self.features:
            return

        # Skin tone average
        skin_samples = [
            f.skin_tone_hsv for f in self.features
            if f.skin_tone_hsv is not None and f.skin_pixel_count >= MIN_SKIN_PIXELS
        ]
        if skin_samples:
            h_vals = [s[0] for s in skin_samples]
            s_vals = [s[1] for s in skin_samples]
            v_vals = [s[2] for s in skin_samples]
            self.avg_skin_tone_hsv = (
                float(_circular_mean_list(h_vals)),
                float(np.mean(s_vals)),
                float(np.mean(v_vals)),
            )

        # Jersey color average
        jersey_samples = [
            f.jersey_color_hsv for f in self.features
            if f.jersey_color_hsv is not None and f.jersey_pixel_count >= MIN_JERSEY_PIXELS
        ]
        if jersey_samples:
            h_vals = [s[0] for s in jersey_samples]
            s_vals = [s[1] for s in jersey_samples]
            v_vals = [s[2] for s in jersey_samples]
            self.avg_jersey_color_hsv = (
                float(_circular_mean_list(h_vals)),
                float(np.mean(s_vals)),
                float(np.mean(v_vals)),
            )

        # Height average
        heights = [f.bbox_height for f in self.features if f.bbox_height > 0]
        if heights:
            self.avg_bbox_height = float(np.mean(heights))


def _circular_mean(a: float, b: float, weight_b: float) -> float:
    """Compute weighted circular mean for hue values (0-180 range in OpenCV)."""
    # Convert to radians (0-180 -> 0-2π)
    a_rad = a * np.pi / 90
    b_rad = b * np.pi / 90

    # Weighted average in Cartesian coordinates
    x = np.cos(a_rad) * (1 - weight_b) + np.cos(b_rad) * weight_b
    y = np.sin(a_rad) * (1 - weight_b) + np.sin(b_rad) * weight_b

    # Convert back to hue
    result = np.arctan2(y, x) * 90 / np.pi
    if result < 0:
        result += 180
    return float(result)


def _circular_mean_list(values: list[float]) -> float:
    """Compute circular mean of a list of hue values."""
    if not values:
        return 0.0

    # Convert to radians
    rads = [v * np.pi / 90 for v in values]

    # Average in Cartesian coordinates
    x = np.mean([np.cos(r) for r in rads])
    y = np.mean([np.sin(r) for r in rads])

    # Convert back
    result = np.arctan2(y, x) * 90 / np.pi
    if result < 0:
        result += 180
    return float(result)


def extract_appearance_features(
    frame: NDArray[np.uint8],
    track_id: int,
    frame_number: int,
    bbox: tuple[float, float, float, float],  # (cx, cy, w, h) normalized
    frame_width: int,
    frame_height: int,
) -> PlayerAppearanceFeatures:
    """
    Extract appearance features from a player detection.

    Args:
        frame: BGR frame from video.
        track_id: Track ID for this detection.
        frame_number: Frame number in video.
        bbox: Normalized bounding box (center_x, center_y, width, height).
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.

    Returns:
        PlayerAppearanceFeatures for this detection.
    """
    cx, cy, w, h = bbox

    # Convert to pixel coordinates
    x1 = int((cx - w / 2) * frame_width)
    y1 = int((cy - h / 2) * frame_height)
    x2 = int((cx + w / 2) * frame_width)
    y2 = int((cy + h / 2) * frame_height)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_width, x2)
    y2 = min(frame_height, y2)

    features = PlayerAppearanceFeatures(
        track_id=track_id,
        frame_number=frame_number,
        bbox_height=h,
        bbox_aspect_ratio=w / h if h > 0 else 0,
    )

    # Extract player region
    if x2 <= x1 or y2 <= y1:
        return features

    player_roi = frame[y1:y2, x1:x2]
    if player_roi.size == 0:
        return features

    # Convert to HSV
    hsv = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)

    # Extract skin tone from upper body region (face/arms)
    # Use top 40% of bbox for skin detection
    upper_height = int(player_roi.shape[0] * 0.4)
    upper_hsv = hsv[:upper_height, :]

    skin_tone, skin_count = _extract_skin_tone(upper_hsv)
    features.skin_tone_hsv = skin_tone
    features.skin_pixel_count = skin_count

    # Extract jersey color from torso region (middle 30-70% of bbox)
    torso_top = int(player_roi.shape[0] * 0.3)
    torso_bottom = int(player_roi.shape[0] * 0.7)
    torso_hsv = hsv[torso_top:torso_bottom, :]

    jersey_color, jersey_count = _extract_jersey_color(torso_hsv)
    features.jersey_color_hsv = jersey_color
    features.jersey_pixel_count = jersey_count

    return features


def _extract_skin_tone(hsv_roi: np.ndarray) -> tuple[tuple[float, float, float] | None, int]:
    """
    Extract skin tone from HSV region.

    Args:
        hsv_roi: HSV image region (upper body).

    Returns:
        Tuple of (average HSV, pixel count) or (None, 0) if not enough skin.
    """
    if hsv_roi.size == 0:
        return None, 0

    # Create skin mask
    mask = cv2.inRange(hsv_roi, SKIN_HSV_LOWER, SKIN_HSV_UPPER)

    # Count skin pixels
    skin_count = int(cv2.countNonZero(mask))
    if skin_count < MIN_SKIN_PIXELS:
        return None, skin_count

    # Extract skin pixels
    skin_pixels = hsv_roi[mask > 0]
    if len(skin_pixels) == 0:
        return None, 0

    # Compute average HSV (handle hue circularity)
    h_mean = _circular_mean_list(skin_pixels[:, 0].astype(float).tolist())
    s_mean = float(np.mean(skin_pixels[:, 1]))
    v_mean = float(np.mean(skin_pixels[:, 2]))

    return (h_mean, s_mean, v_mean), skin_count


def _extract_jersey_color(hsv_roi: np.ndarray) -> tuple[tuple[float, float, float] | None, int]:
    """
    Extract dominant jersey/clothing color from HSV region.

    Args:
        hsv_roi: HSV image region (torso area).

    Returns:
        Tuple of (dominant HSV, pixel count) or (None, 0) if not enough pixels.
    """
    if hsv_roi.size == 0:
        return None, 0

    # Exclude skin pixels (jersey should be non-skin)
    skin_mask = cv2.inRange(hsv_roi, SKIN_HSV_LOWER, SKIN_HSV_UPPER)
    non_skin_mask = cv2.bitwise_not(skin_mask)

    # Also exclude very dark pixels (shadows) and very bright (highlights)
    value_channel = hsv_roi[:, :, 2]
    brightness_mask = cv2.inRange(
        value_channel,
        np.array(30, dtype=np.uint8),
        np.array(240, dtype=np.uint8),
    )

    # Combined mask
    valid_mask = cv2.bitwise_and(non_skin_mask, brightness_mask)

    pixel_count = int(cv2.countNonZero(valid_mask))
    if pixel_count < MIN_JERSEY_PIXELS:
        return None, pixel_count

    # Extract valid pixels
    valid_pixels = hsv_roi[valid_mask > 0]
    if len(valid_pixels) == 0:
        return None, 0

    # Compute mean color (works well for solid jersey colors)
    # For multi-colored jerseys, could use k-means clustering here
    pixels_float = valid_pixels.astype(np.float32)
    h_mean = _circular_mean_list(pixels_float[:, 0].tolist())
    s_mean = float(np.mean(pixels_float[:, 1]))
    v_mean = float(np.mean(pixels_float[:, 2]))

    return (h_mean, s_mean, v_mean), pixel_count


def compute_appearance_similarity(
    profile: PlayerAppearanceProfile,
    features: TrackAppearanceStats,
) -> float:
    """
    Compute similarity between a player profile and track appearance.

    Cost formula (lower = more similar):
    cost = 1.0 - (0.50 * skin_score + 0.30 * height_score + 0.20 * jersey_score)

    Args:
        profile: Stored player appearance profile.
        features: Track appearance stats from current rally.

    Returns:
        Cost (0-1, lower = more similar).
    """
    skin_score = 0.0
    height_score = 0.0
    jersey_score = 0.0

    # Skin tone similarity (most reliable)
    if profile.avg_skin_tone_hsv and features.avg_skin_tone_hsv:
        skin_score = _hsv_similarity(profile.avg_skin_tone_hsv, features.avg_skin_tone_hsv)

    # Height similarity
    if profile.avg_bbox_height > 0 and features.avg_bbox_height > 0:
        height_diff = abs(profile.avg_bbox_height - features.avg_bbox_height)
        # Normalize: 10% height difference = 0 similarity
        height_score = max(0, 1.0 - height_diff / 0.10)

    # Jersey color similarity (least reliable)
    if profile.avg_jersey_color_hsv and features.avg_jersey_color_hsv:
        jersey_score = _hsv_similarity(profile.avg_jersey_color_hsv, features.avg_jersey_color_hsv)

    # Weighted combination
    similarity = 0.50 * skin_score + 0.30 * height_score + 0.20 * jersey_score

    return 1.0 - similarity  # Return cost (lower = better match)


def _hsv_similarity(hsv1: tuple[float, float, float], hsv2: tuple[float, float, float]) -> float:
    """
    Compute similarity between two HSV colors.

    Args:
        hsv1: First HSV color (H: 0-180, S: 0-255, V: 0-255).
        hsv2: Second HSV color.

    Returns:
        Similarity score (0-1, higher = more similar).
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    # Hue similarity (circular distance)
    h_diff = min(abs(h1 - h2), 180 - abs(h1 - h2))
    h_sim = 1.0 - h_diff / 90  # 90 degrees = 0 similarity

    # Saturation similarity
    s_diff = abs(s1 - s2)
    s_sim = 1.0 - s_diff / 255

    # Value similarity
    v_diff = abs(v1 - v2)
    v_sim = 1.0 - v_diff / 255

    # Weighted combination (hue is most important for color matching)
    return 0.5 * h_sim + 0.25 * s_sim + 0.25 * v_sim
