"""
Player appearance feature extraction for cross-rally tracking.

Extracts visual features from player detections for consistent ID assignment:
- Clothing color histograms (upper body + lower body HS + V histograms)
- Dominant clothing color (median HSV of masked clothing pixels)
- Skin tone (HSV)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Skin detection HSV ranges (tuned for beach volleyball in various lighting)
SKIN_HSV_LOWER = np.array([0, 20, 50], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([50, 255, 255], dtype=np.uint8)

# Minimum pixels for valid skin color extraction
MIN_SKIN_PIXELS = 50

# HS histogram parameters (matches color_repair.py)
HS_BINS = (16, 8)  # Hue (16 bins) x Saturation (8 bins) = 128 values
HS_RANGES = [0, 180, 0, 256]  # OpenCV HSV: H=[0,180), S=[0,256)
MIN_HIST_PIXELS = 200  # Minimum pixels for reliable histogram

# V (brightness) histogram parameters
V_BINS = 16
V_RANGES = [0, 256]

# EMA decay factor for profile updates.
# 0.0 = use equal-weight running average (legacy behavior).
# >0.0 = exponential moving average where new sample gets weight=alpha.
# Effective memory window ≈ 1/alpha rallies (e.g. 0.10 ≈ 10 rally window).
# Tuned via grid search on 46 GT videos: α=0.10 → 86.7% (+1.3pp) with
# near-zero drift (early 86.9% vs late 86.6%) vs baseline drift of +4.6pp.
PROFILE_EMA_ALPHA: float = 0.10

# Tighter skin range for clothing mask — avoids removing red/orange clothing.
# Real skin: H=5-20, moderate S (40-170), moderate V (70-230).
# Red clothing: H~0-5 or S>180. Orange: H>15 + S>170.
CLOTHING_SKIN_LOWER = np.array([5, 40, 70], dtype=np.uint8)
CLOTHING_SKIN_UPPER = np.array([20, 170, 230], dtype=np.uint8)


@dataclass
class PlayerAppearanceFeatures:
    """Appearance features extracted from a single player detection."""

    track_id: int
    frame_number: int

    # Skin tone (HSV)
    # None if not enough skin pixels detected
    skin_tone_hsv: tuple[float, float, float] | None = None
    skin_pixel_count: int = 0

    # HS histograms for clothing regions
    # Upper body (20-60% of bbox): captures t-shirt/jersey if present
    upper_body_hist: np.ndarray | None = None
    # Lower body (60-100% of bbox): captures shorts/swimsuit
    lower_body_hist: np.ndarray | None = None

    # V (brightness) histograms for clothing regions
    upper_body_v_hist: np.ndarray | None = None
    lower_body_v_hist: np.ndarray | None = None

    # Dominant clothing color (median HSV of masked lower-body pixels)
    dominant_color_hsv: tuple[float, float, float] | None = None


@dataclass
class PlayerAppearanceProfile:
    """Aggregated appearance profile for a player across rallies."""

    player_id: int  # Consistent 1-4 across match

    # Average skin tone (HSV) — None until first sample
    avg_skin_tone_hsv: tuple[float, float, float] | None = None
    skin_sample_count: int = 0

    # HS histograms for clothing regions
    avg_upper_hist: np.ndarray | None = None
    upper_hist_count: int = 0
    avg_lower_hist: np.ndarray | None = None
    lower_hist_count: int = 0

    # V (brightness) histograms
    avg_upper_v_hist: np.ndarray | None = None
    upper_v_hist_count: int = 0
    avg_lower_v_hist: np.ndarray | None = None
    lower_v_hist_count: int = 0

    # Dominant clothing color (median HSV)
    avg_dominant_color_hsv: tuple[float, float, float] | None = None
    dominant_color_count: int = 0

    # Team assignment (0=near court, 1=far court)
    team: int = 0

    # How many rallies this player has been seen in
    rally_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize profile to a JSON-compatible dict."""
        d: dict[str, Any] = {
            "player_id": self.player_id,
            "team": self.team,
            "rally_count": self.rally_count,
            "skin_sample_count": self.skin_sample_count,
            "upper_hist_count": self.upper_hist_count,
            "lower_hist_count": self.lower_hist_count,
            "upper_v_hist_count": self.upper_v_hist_count,
            "lower_v_hist_count": self.lower_v_hist_count,
            "dominant_color_count": self.dominant_color_count,
        }
        if self.avg_skin_tone_hsv is not None:
            d["avg_skin_tone_hsv"] = list(self.avg_skin_tone_hsv)
        if self.avg_upper_hist is not None:
            d["avg_upper_hist"] = self.avg_upper_hist.flatten().tolist()
        if self.avg_lower_hist is not None:
            d["avg_lower_hist"] = self.avg_lower_hist.flatten().tolist()
        if self.avg_upper_v_hist is not None:
            d["avg_upper_v_hist"] = self.avg_upper_v_hist.flatten().tolist()
        if self.avg_lower_v_hist is not None:
            d["avg_lower_v_hist"] = self.avg_lower_v_hist.flatten().tolist()
        if self.avg_dominant_color_hsv is not None:
            d["avg_dominant_color_hsv"] = list(self.avg_dominant_color_hsv)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PlayerAppearanceProfile:
        """Deserialize profile from a dict."""
        profile = cls(
            player_id=d["player_id"],
            team=d.get("team", 0),
            rally_count=d.get("rally_count", 0),
            skin_sample_count=d.get("skin_sample_count", 0),
            upper_hist_count=d.get("upper_hist_count", 0),
            lower_hist_count=d.get("lower_hist_count", 0),
            upper_v_hist_count=d.get("upper_v_hist_count", 0),
            lower_v_hist_count=d.get("lower_v_hist_count", 0),
            dominant_color_count=d.get("dominant_color_count", 0),
        )
        if "avg_skin_tone_hsv" in d and d["avg_skin_tone_hsv"] is not None:
            hsv = d["avg_skin_tone_hsv"]
            profile.avg_skin_tone_hsv = (float(hsv[0]), float(hsv[1]), float(hsv[2]))
        if "avg_upper_hist" in d and d["avg_upper_hist"] is not None:
            profile.avg_upper_hist = np.array(
                d["avg_upper_hist"], dtype=np.float32
            ).reshape(HS_BINS)
        if "avg_lower_hist" in d and d["avg_lower_hist"] is not None:
            profile.avg_lower_hist = np.array(
                d["avg_lower_hist"], dtype=np.float32
            ).reshape(HS_BINS)
        if "avg_upper_v_hist" in d and d["avg_upper_v_hist"] is not None:
            profile.avg_upper_v_hist = np.array(
                d["avg_upper_v_hist"], dtype=np.float32
            ).reshape((V_BINS,))
        if "avg_lower_v_hist" in d and d["avg_lower_v_hist"] is not None:
            profile.avg_lower_v_hist = np.array(
                d["avg_lower_v_hist"], dtype=np.float32
            ).reshape((V_BINS,))
        if "avg_dominant_color_hsv" in d and d["avg_dominant_color_hsv"] is not None:
            dc = d["avg_dominant_color_hsv"]
            profile.avg_dominant_color_hsv = (float(dc[0]), float(dc[1]), float(dc[2]))
        return profile

    def _ema_weight(self, sample_count: int) -> float:
        """Compute update weight for a new sample.

        With EMA (PROFILE_EMA_ALPHA > 0), returns a fixed weight so recent
        samples dominate. With alpha=0 (legacy), returns 1/(count+1) for
        equal-weight running average.
        """
        if PROFILE_EMA_ALPHA > 0:
            # First sample always gets weight=1 (initialization)
            if sample_count == 0:
                return 1.0
            return PROFILE_EMA_ALPHA
        return 1.0 / (sample_count + 1)

    def update_from_features(self, features: PlayerAppearanceFeatures) -> None:
        """Update profile with new appearance features."""
        # Update skin tone
        if features.skin_tone_hsv is not None and features.skin_pixel_count >= MIN_SKIN_PIXELS:
            if self.avg_skin_tone_hsv is None:
                self.avg_skin_tone_hsv = features.skin_tone_hsv
            else:
                weight = self._ema_weight(self.skin_sample_count)
                h1, s1, v1 = self.avg_skin_tone_hsv
                h2, s2, v2 = features.skin_tone_hsv
                h_new = _circular_mean(h1, h2, weight)
                s_new = s1 * (1 - weight) + s2 * weight
                v_new = v1 * (1 - weight) + v2 * weight
                self.avg_skin_tone_hsv = (h_new, s_new, v_new)
            self.skin_sample_count += 1

        # Update upper body histogram
        if features.upper_body_hist is not None:
            if self.avg_upper_hist is None:
                self.avg_upper_hist = features.upper_body_hist.copy()
            else:
                weight = self._ema_weight(self.upper_hist_count)
                self.avg_upper_hist = (
                    self.avg_upper_hist * (1 - weight)
                    + features.upper_body_hist * weight
                )
            self.upper_hist_count += 1

        # Update lower body histogram
        if features.lower_body_hist is not None:
            if self.avg_lower_hist is None:
                self.avg_lower_hist = features.lower_body_hist.copy()
            else:
                weight = self._ema_weight(self.lower_hist_count)
                self.avg_lower_hist = (
                    self.avg_lower_hist * (1 - weight)
                    + features.lower_body_hist * weight
                )
            self.lower_hist_count += 1

        # Update upper body V histogram
        if features.upper_body_v_hist is not None:
            if self.avg_upper_v_hist is None:
                self.avg_upper_v_hist = features.upper_body_v_hist.copy()
            else:
                weight = self._ema_weight(self.upper_v_hist_count)
                self.avg_upper_v_hist = (
                    self.avg_upper_v_hist * (1 - weight)
                    + features.upper_body_v_hist * weight
                )
            self.upper_v_hist_count += 1

        # Update lower body V histogram
        if features.lower_body_v_hist is not None:
            if self.avg_lower_v_hist is None:
                self.avg_lower_v_hist = features.lower_body_v_hist.copy()
            else:
                weight = self._ema_weight(self.lower_v_hist_count)
                self.avg_lower_v_hist = (
                    self.avg_lower_v_hist * (1 - weight)
                    + features.lower_body_v_hist * weight
                )
            self.lower_v_hist_count += 1

        # Update dominant color
        if features.dominant_color_hsv is not None:
            if self.avg_dominant_color_hsv is None:
                self.avg_dominant_color_hsv = features.dominant_color_hsv
            else:
                weight = self._ema_weight(self.dominant_color_count)
                h1, s1, v1 = self.avg_dominant_color_hsv
                h2, s2, v2 = features.dominant_color_hsv
                h_new = _circular_mean(h1, h2, weight)
                s_new = s1 * (1 - weight) + s2 * weight
                v_new = v1 * (1 - weight) + v2 * weight
                self.avg_dominant_color_hsv = (h_new, s_new, v_new)
            self.dominant_color_count += 1


@dataclass
class TrackAppearanceStats:
    """Aggregated appearance stats for a single track within a rally."""

    track_id: int
    features: list[PlayerAppearanceFeatures] = field(default_factory=list)

    # Computed averages
    avg_skin_tone_hsv: tuple[float, float, float] | None = None
    avg_upper_hist: np.ndarray | None = None
    avg_lower_hist: np.ndarray | None = None
    avg_upper_v_hist: np.ndarray | None = None
    avg_lower_v_hist: np.ndarray | None = None
    avg_dominant_color_hsv: tuple[float, float, float] | None = None

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

        # Upper body histogram average (each input already L1-normalized)
        upper_hists = [
            f.upper_body_hist for f in self.features
            if f.upper_body_hist is not None
        ]
        if upper_hists:
            self.avg_upper_hist = np.mean(upper_hists, axis=0).astype(np.float32)

        # Lower body histogram average (each input already L1-normalized)
        lower_hists = [
            f.lower_body_hist for f in self.features
            if f.lower_body_hist is not None
        ]
        if lower_hists:
            self.avg_lower_hist = np.mean(lower_hists, axis=0).astype(np.float32)

        # Upper body V histogram average
        upper_v_hists = [
            f.upper_body_v_hist for f in self.features
            if f.upper_body_v_hist is not None
        ]
        if upper_v_hists:
            self.avg_upper_v_hist = np.mean(upper_v_hists, axis=0).astype(np.float32)

        # Lower body V histogram average
        lower_v_hists = [
            f.lower_body_v_hist for f in self.features
            if f.lower_body_v_hist is not None
        ]
        if lower_v_hists:
            self.avg_lower_v_hist = np.mean(lower_v_hists, axis=0).astype(np.float32)

        # Dominant clothing color average
        dc_samples = [
            f.dominant_color_hsv for f in self.features
            if f.dominant_color_hsv is not None
        ]
        if dc_samples:
            h_vals = [s[0] for s in dc_samples]
            s_vals = [s[1] for s in dc_samples]
            v_vals = [s[2] for s in dc_samples]
            self.avg_dominant_color_hsv = (
                float(_circular_mean_list(h_vals)),
                float(np.mean(s_vals)),
                float(np.mean(v_vals)),
            )


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
    )

    # Extract player region
    if x2 <= x1 or y2 <= y1:
        return features

    player_roi = frame[y1:y2, x1:x2]
    if player_roi.size == 0:
        return features

    bbox_h_px = y2 - y1

    # Convert to HSV
    hsv = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)

    # Extract skin tone from upper body region (face/arms)
    # Use top 40% of bbox for skin detection
    upper_height = int(bbox_h_px * 0.4)
    upper_hsv = hsv[:upper_height, :]

    skin_tone, skin_count = _extract_skin_tone(upper_hsv)
    features.skin_tone_hsv = skin_tone
    features.skin_pixel_count = skin_count

    # Build skin+sand mask to exclude from clothing histograms.
    # Without this, legs and beach sand dominate all histograms,
    # making all players look nearly identical (0.87-0.90 similarity).
    clothing_mask = _build_clothing_mask(hsv)

    # Extract HS + V histograms for clothing regions
    # Upper body: 20-55% of bbox (t-shirt/jersey)
    upper_top = int(bbox_h_px * 0.2)
    upper_bottom = int(bbox_h_px * 0.55)
    upper_hsv_roi = hsv[upper_top:upper_bottom, :]
    upper_mask = clothing_mask[upper_top:upper_bottom, :]
    features.upper_body_hist = _extract_hs_histogram(upper_hsv_roi, mask=upper_mask)
    features.upper_body_v_hist = _extract_v_histogram(upper_hsv_roi, mask=upper_mask)

    # Lower body: 50-78% of bbox (shorts/swimsuit, avoids legs below)
    lower_top = int(bbox_h_px * 0.50)
    lower_bottom = int(bbox_h_px * 0.78)
    lower_hsv_roi = hsv[lower_top:lower_bottom, :]
    lower_mask = clothing_mask[lower_top:lower_bottom, :]
    features.lower_body_hist = _extract_hs_histogram(lower_hsv_roi, mask=lower_mask)
    features.lower_body_v_hist = _extract_v_histogram(lower_hsv_roi, mask=lower_mask)

    # Dominant clothing color: median HSV of masked lower-body pixels
    features.dominant_color_hsv = _extract_dominant_color(lower_hsv_roi, lower_mask)

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


def _build_clothing_mask(hsv_roi: np.ndarray) -> np.ndarray:
    """Build a mask isolating clothing pixels from a player bbox.

    Three-stage filtering:
    1. Spatial: central 50% of width (torso/shorts, avoids background edges)
    2. Sand exclusion: warm-hue + low-saturation pixels (conservative range)
    3. Skin exclusion: tighter HSV range than general skin detection to
       avoid removing red/orange clothing

    Args:
        hsv_roi: HSV image of the full player bbox.

    Returns:
        Binary mask (255 = include, 0 = exclude) matching hsv_roi shape[:2].
    """
    h, w = hsv_roi.shape[:2]

    # Spatial mask: central 50% of width (clothing region, avoids edges)
    margin = w // 4
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, margin:w - margin] = 255

    # Conservative sand filter: only exclude very obvious sand pixels.
    # Sand: warm hue (H=8-30), low saturation (S<50), bright (V>120).
    # This is narrower than the full skin range to avoid catching
    # light-colored clothing (white shorts: H~29, S~47, V~205).
    sand_lower = np.array([8, 10, 120], dtype=np.uint8)
    sand_upper = np.array([30, 50, 255], dtype=np.uint8)
    sand_mask = cv2.inRange(hsv_roi, sand_lower, sand_upper)
    not_sand = cv2.bitwise_not(sand_mask)
    mask = np.asarray(cv2.bitwise_and(mask, not_sand), dtype=np.uint8)

    # Skin exclusion: remove skin pixels that dominate shirtless upper body
    # and thigh areas. Uses tighter range than SKIN_HSV_LOWER/UPPER to
    # avoid removing red/orange clothing.
    skin_mask = cv2.inRange(hsv_roi, CLOTHING_SKIN_LOWER, CLOTHING_SKIN_UPPER)
    not_skin = cv2.bitwise_not(skin_mask)
    mask = np.asarray(cv2.bitwise_and(mask, not_skin), dtype=np.uint8)

    return mask


def _extract_hs_histogram(
    hsv_roi: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract normalized HS histogram from an HSV region.

    Args:
        hsv_roi: HSV image region.
        mask: Optional binary mask (255 = include pixel). If provided,
            only masked pixels contribute to the histogram.

    Returns:
        Normalized HS histogram (float32, sum=1), or None if too few pixels.
    """
    if hsv_roi.size == 0:
        return None

    if mask is not None:
        pixel_count = int(cv2.countNonZero(mask))
    else:
        pixel_count = hsv_roi.shape[0] * hsv_roi.shape[1]

    if pixel_count < MIN_HIST_PIXELS:
        return None

    hist = cv2.calcHist(
        [hsv_roi], [0, 1], mask, list(HS_BINS), HS_RANGES,
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32)


def _extract_v_histogram(
    hsv_roi: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Extract normalized V (brightness) histogram from an HSV region.

    Args:
        hsv_roi: HSV image region.
        mask: Optional binary mask (255 = include pixel).

    Returns:
        Normalized V histogram (float32, sum=1), or None if too few pixels.
    """
    if hsv_roi.size == 0:
        return None

    if mask is not None:
        pixel_count = int(cv2.countNonZero(mask))
    else:
        pixel_count = hsv_roi.shape[0] * hsv_roi.shape[1]

    if pixel_count < MIN_HIST_PIXELS:
        return None

    hist = cv2.calcHist(
        [hsv_roi], [2], mask, [V_BINS], V_RANGES,
    )
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.astype(np.float32).flatten()


def _extract_dominant_color(
    hsv_roi: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[float, float, float] | None:
    """Extract dominant clothing color from masked pixels.

    Uses circular mean for hue (handles wraparound) and median for S/V.

    Args:
        hsv_roi: HSV image region.
        mask: Binary mask (255 = include pixel).

    Returns:
        (H, S, V) tuple or None if too few pixels.
    """
    if hsv_roi.size == 0:
        return None

    if mask is not None:
        pixels = hsv_roi[mask > 0]
    else:
        pixels = hsv_roi.reshape(-1, 3)

    if len(pixels) < MIN_HIST_PIXELS:
        return None

    h_avg = _circular_mean_list(pixels[:, 0].astype(float).tolist())
    s_median = float(np.median(pixels[:, 1]))
    v_median = float(np.median(pixels[:, 2]))
    return (h_avg, s_median, v_median)


def _histogram_similarity(
    hist1: np.ndarray | None, hist2: np.ndarray | None,
) -> float | None:
    """Bhattacharyya coefficient between two histograms (HS or V).

    Returns:
        Similarity 0-1 (higher = more similar), or None if either is missing.
    """
    if hist1 is None or hist2 is None:
        return None
    # HISTCMP_BHATTACHARYYA returns distance (0 = identical, 1 = no overlap)
    dist = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - dist)


_WEIGHT_LOWER_HIST = 0.35
_WEIGHT_LOWER_V_HIST = 0.15
_WEIGHT_UPPER_HIST = 0.15
_WEIGHT_UPPER_V_HIST = 0.10
_WEIGHT_SKIN = 0.10
_WEIGHT_DOMINANT_COLOR = 0.15


def compute_appearance_similarity(
    profile: PlayerAppearanceProfile,
    features: TrackAppearanceStats,
) -> float:
    """
    Compute similarity between a player profile and track appearance.

    Cost formula (lower = more similar):
        cost = 1.0 - weighted_similarity

    Weights:
        35% lower body HS histogram
        15% lower body V histogram
        15% upper body HS histogram
        10% upper body V histogram
        10% skin tone
        15% dominant clothing color

    Missing features are skipped; remaining weights are renormalized.

    Args:
        profile: Stored player appearance profile.
        features: Track appearance stats from current rally.

    Returns:
        Cost (0-1, lower = more similar).
    """
    scores: list[tuple[float, float]] = []  # (weight, score) pairs

    # Lower body HS histogram (shorts — most discriminative feature)
    lower_sim = _histogram_similarity(profile.avg_lower_hist, features.avg_lower_hist)
    if lower_sim is not None:
        scores.append((_WEIGHT_LOWER_HIST, lower_sim))

    # Lower body V histogram (brightness)
    lower_v_sim = _histogram_similarity(
        profile.avg_lower_v_hist, features.avg_lower_v_hist,
    )
    if lower_v_sim is not None:
        scores.append((_WEIGHT_LOWER_V_HIST, lower_v_sim))

    # Upper body HS histogram (t-shirt/jersey)
    upper_sim = _histogram_similarity(profile.avg_upper_hist, features.avg_upper_hist)
    if upper_sim is not None:
        scores.append((_WEIGHT_UPPER_HIST, upper_sim))

    # Upper body V histogram (brightness)
    upper_v_sim = _histogram_similarity(
        profile.avg_upper_v_hist, features.avg_upper_v_hist,
    )
    if upper_v_sim is not None:
        scores.append((_WEIGHT_UPPER_V_HIST, upper_v_sim))

    # Skin tone similarity
    if profile.avg_skin_tone_hsv is not None and features.avg_skin_tone_hsv is not None:
        skin_score = _hsv_similarity(
            profile.avg_skin_tone_hsv, features.avg_skin_tone_hsv,
        )
        scores.append((_WEIGHT_SKIN, skin_score))

    # Dominant clothing color similarity
    if (
        profile.avg_dominant_color_hsv is not None
        and features.avg_dominant_color_hsv is not None
    ):
        dc_score = _hsv_similarity(
            profile.avg_dominant_color_hsv, features.avg_dominant_color_hsv,
        )
        scores.append((_WEIGHT_DOMINANT_COLOR, dc_score))

    if not scores:
        return 1.0  # No features → max cost (unknown)

    # Normalize weights to sum to 1.0
    total_weight = sum(w for w, _ in scores)
    similarity = sum(w * s for w, s in scores) / total_weight

    return 1.0 - similarity  # Return cost (lower = better match)


def compute_track_similarity(
    stats_a: TrackAppearanceStats,
    stats_b: TrackAppearanceStats,
) -> float:
    """Compute similarity cost between two track appearance stats.

    Same cost formula as compute_appearance_similarity but works between
    two raw TrackAppearanceStats (no accumulated profile needed).

    Returns:
        Cost (0-1, lower = more similar).
    """
    scores: list[tuple[float, float]] = []

    lower_sim = _histogram_similarity(stats_a.avg_lower_hist, stats_b.avg_lower_hist)
    if lower_sim is not None:
        scores.append((_WEIGHT_LOWER_HIST, lower_sim))

    lower_v_sim = _histogram_similarity(
        stats_a.avg_lower_v_hist, stats_b.avg_lower_v_hist,
    )
    if lower_v_sim is not None:
        scores.append((_WEIGHT_LOWER_V_HIST, lower_v_sim))

    upper_sim = _histogram_similarity(stats_a.avg_upper_hist, stats_b.avg_upper_hist)
    if upper_sim is not None:
        scores.append((_WEIGHT_UPPER_HIST, upper_sim))

    upper_v_sim = _histogram_similarity(
        stats_a.avg_upper_v_hist, stats_b.avg_upper_v_hist,
    )
    if upper_v_sim is not None:
        scores.append((_WEIGHT_UPPER_V_HIST, upper_v_sim))

    if stats_a.avg_skin_tone_hsv is not None and stats_b.avg_skin_tone_hsv is not None:
        skin_score = _hsv_similarity(
            stats_a.avg_skin_tone_hsv, stats_b.avg_skin_tone_hsv,
        )
        scores.append((_WEIGHT_SKIN, skin_score))

    if (
        stats_a.avg_dominant_color_hsv is not None
        and stats_b.avg_dominant_color_hsv is not None
    ):
        dc_score = _hsv_similarity(
            stats_a.avg_dominant_color_hsv, stats_b.avg_dominant_color_hsv,
        )
        scores.append((_WEIGHT_DOMINANT_COLOR, dc_score))

    if not scores:
        return 1.0

    total_weight = sum(w for w, _ in scores)
    similarity = sum(w * s for w, s in scores) / total_weight
    return 1.0 - similarity


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
