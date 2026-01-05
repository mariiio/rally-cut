/**
 * Camera keyframe interpolation utilities.
 * Provides smooth transitions between camera positions using easing functions.
 */

import {
  CameraKeyframe,
  CameraState,
  CameraEasing,
  DEFAULT_CAMERA_STATE,
} from '@/types/camera';

/**
 * Easing functions for smooth keyframe transitions.
 * Each function maps t (0-1) to an eased value (0-1).
 */
const easingFunctions: Record<CameraEasing, (t: number) => number> = {
  LINEAR: (t) => t,
  EASE_IN: (t) => t * t,
  EASE_OUT: (t) => t * (2 - t),
  EASE_IN_OUT: (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
};

/**
 * Clamp a value between min and max.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Linear interpolation between two values.
 */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Get valid position range based on aspect ratio and zoom level.
 * Position [0,1] should always map to valid video content (no black areas).
 *
 * @param aspectRatio - Target aspect ratio ('ORIGINAL' for 16:9, 'VERTICAL' for 9:16)
 * @param zoom - Current zoom level (1.0 = no zoom)
 * @returns Valid position bounds { minX, maxX, minY, maxY }
 */
export function getValidPositionRange(
  aspectRatio: 'ORIGINAL' | 'VERTICAL',
  zoom: number
): { minX: number; maxX: number; minY: number; maxY: number } {
  if (aspectRatio === 'ORIGINAL') {
    // 16:9: Can only pan when zoomed in
    if (zoom <= 1) {
      return { minX: 0.5, maxX: 0.5, minY: 0.5, maxY: 0.5 };
    }
    // At zoom > 1, full range becomes available
    return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  } else {
    // VERTICAL 9:16
    if (zoom < 1) {
      // Zoomed out - video letterboxed, no panning
      return { minX: 0.5, maxX: 0.5, minY: 0.5, maxY: 0.5 };
    }
    // At zoom >= 1: full horizontal pan available
    // Vertical pan only available at zoom > 1
    const verticalPan = zoom > 1;
    return {
      minX: 0,
      maxX: 1,
      minY: verticalPan ? 0 : 0.5,
      maxY: verticalPan ? 1 : 0.5,
    };
  }
}

/**
 * Interpolate camera state at a given time offset within a rally.
 *
 * @param keyframes - Array of keyframes (should be sorted by timeOffset)
 * @param timeOffset - Current time as fraction of rally (0.0 to 1.0)
 * @returns Interpolated camera state
 */
export function interpolateCameraState(
  keyframes: CameraKeyframe[],
  timeOffset: number
): CameraState {
  // Clamp time to valid range
  const t = clamp(timeOffset, 0, 1);

  // No keyframes - return default (centered, no zoom)
  if (keyframes.length === 0) {
    return DEFAULT_CAMERA_STATE;
  }

  // Single keyframe - return its values directly
  if (keyframes.length === 1) {
    const kf = keyframes[0];
    return {
      positionX: kf.positionX,
      positionY: kf.positionY,
      zoom: kf.zoom,
    };
  }

  // Ensure keyframes are sorted
  const sorted = [...keyframes].sort((a, b) => a.timeOffset - b.timeOffset);

  // Before first keyframe - use first keyframe values
  if (t <= sorted[0].timeOffset) {
    const first = sorted[0];
    return {
      positionX: first.positionX,
      positionY: first.positionY,
      zoom: first.zoom,
    };
  }

  // After last keyframe - use last keyframe values
  if (t >= sorted[sorted.length - 1].timeOffset) {
    const last = sorted[sorted.length - 1];
    return {
      positionX: last.positionX,
      positionY: last.positionY,
      zoom: last.zoom,
    };
  }

  // Find surrounding keyframes
  let beforeIdx = 0;
  for (let i = 0; i < sorted.length - 1; i++) {
    if (sorted[i].timeOffset <= t && sorted[i + 1].timeOffset >= t) {
      beforeIdx = i;
      break;
    }
  }

  const before = sorted[beforeIdx];
  const after = sorted[beforeIdx + 1];

  // Calculate local t between these two keyframes
  const range = after.timeOffset - before.timeOffset;
  const localT = range > 0 ? (t - before.timeOffset) / range : 0;

  // Apply easing from the 'before' keyframe
  const easingFn = easingFunctions[before.easing];
  const easedT = easingFn(localT);

  // Interpolate all values
  return {
    positionX: lerp(before.positionX, after.positionX, easedT),
    positionY: lerp(before.positionY, after.positionY, easedT),
    zoom: lerp(before.zoom, after.zoom, easedT),
  };
}

/**
 * Calculate the CSS styles for the video element based on camera state.
 *
 * For VERTICAL (9:16) mode, uses object-position instead of transform.
 * object-position directly controls what part of the video is visible
 * when using objectFit: cover, which is more reliable than transforms.
 *
 * @param state - Current camera state
 * @param aspectRatio - Target aspect ratio ('ORIGINAL' or 'VERTICAL')
 * @returns CSS style object
 */
export function calculateVideoTransform(
  state: CameraState,
  aspectRatio: 'ORIGINAL' | 'VERTICAL'
): React.CSSProperties {
  const { positionX, positionY, zoom } = state;

  if (aspectRatio === 'ORIGINAL') {
    // For original 16:9 aspect ratio with zoom/pan
    // At zoom 1.0, no additional scaling (video shows full frame)
    // Position 0.5 = center, translation = 0

    // Calculate pan range: at zoom Z, we see 1/Z of the video
    // Pan range is (Z - 1) / Z of the video width/height
    const panRangeX = (zoom - 1) / zoom;
    const panRangeY = (zoom - 1) / zoom;

    // Map position (0-1) to translation
    // Position 0.5 = center = 0 translation
    // Position 0 = show left edge = translate right by half pan range
    // Position 1 = show right edge = translate left by half pan range
    const translateX = (0.5 - positionX) * panRangeX * 100;
    const translateY = (0.5 - positionY) * panRangeY * 100;

    return {
      transform: `scale(${zoom}) translate(${translateX}%, ${translateY}%)`,
      transformOrigin: 'center center',
    };
  } else {
    // For 9:16 vertical from 16:9 source (NO objectFit: cover)
    //
    // The video element is set up as:
    // - left: 50%, height: 100%, width: auto
    // - This makes a 16:9 video ~3.16x wider than the 9:16 container
    //
    // We use ONLY transform (like 16:9 does) for smooth transitions:
    // - translateX handles horizontal panning (which part of the wide video is visible)
    // - scale handles zoom
    // - translateY handles vertical panning within zoomed view
    //
    // Key insight: transform-origin is calculated so it's always at the
    // CENTER OF THE VISIBLE AREA (container center), regardless of pan position.
    // This ensures scaling feels natural and centered.

    // Calculate the excess width ratio for 16:9 video in 9:16 container
    // Video is (16/9) / (9/16) = 256/81 ≈ 3.16x wider than container
    const widthRatio = (16 / 9) / (9 / 16); // ~3.16
    const excessRatio = (widthRatio - 1) / widthRatio; // ~0.684 (68.4%)

    // Horizontal translate: centers video + pans based on positionX
    // - Base translate of -50% centers the video (since left: 50%)
    // - Pan translate shifts based on position (0 = left, 0.5 = center, 1 = right)
    const baseCenterX = -50;
    const panOffsetX = (0.5 - positionX) * excessRatio * 100;
    const translateX = baseCenterX + panOffsetX;

    // Transform origin X: positioned so it's always at container center
    // As translateX changes, originX compensates to keep absolute origin fixed
    const originX = (1 - excessRatio) / 2 * 100 + positionX * excessRatio * 100;
    // Simplified: originX = 15.8 + positionX * 68.4 (for 16:9 in 9:16)

    // Vertical translate: only available when zoomed in (like 16:9)
    const panRangeY = (zoom - 1) / zoom;
    const translateY = (0.5 - positionY) * panRangeY * 100;

    return {
      transform: `translateX(${translateX}%) translateY(${translateY}%) scale(${zoom})`,
      transformOrigin: `${originX}% 50%`,
    };
  }
}

