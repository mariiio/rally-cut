/**
 * Camera keyframe interpolation utilities.
 * Provides smooth transitions between camera positions using easing functions.
 */

import {
  CameraKeyframe,
  CameraState,
  CameraEasing,
  DEFAULT_CAMERA_STATE,
  HandheldPreset,
  HANDHELD_PRESET_CONFIG,
} from '@/types/camera';
import { applyHandheldMotion, calculateCameraVelocity } from './handheldMotion';
import {
  CameraSpringState,
  createCameraSpringState,
  stepCameraSpring,
} from './springPhysics';

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
 * Calculate minimum zoom needed to prevent black corners when rotation is applied.
 * When a video is rotated, the corners extend outside the original bounds.
 * We need to zoom in enough to ensure the rotated video covers the entire viewport.
 *
 * Formula: For a rectangle rotated by θ, the bounding box grows. To ensure the
 * rotated rectangle fully covers an axis-aligned viewport of the same aspect ratio,
 * we need: zoom = |cos(θ)| + |sin(θ)| * aspectRatio (for landscape)
 *
 * @param rotationDegrees - Rotation angle in degrees
 * @param aspectRatio - Target aspect ratio ('ORIGINAL' for 16:9, 'VERTICAL' for 9:16)
 * @returns Minimum zoom factor to prevent black corners
 */
export function getMinZoomForRotation(
  rotationDegrees: number,
  aspectRatio: 'ORIGINAL' | 'VERTICAL'
): number {
  if (rotationDegrees === 0) return 1;

  const theta = Math.abs(rotationDegrees) * Math.PI / 180;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);

  // For 16:9 source video:
  // - ORIGINAL mode: cropping 16:9 from 16:9, use aspect ratio 16/9
  // - VERTICAL mode: cropping 9:16 from 16:9, the crop is narrower so use 16/9 as well
  //   (the limiting factor is still the source video's aspect ratio)
  const r = 16 / 9;

  // The zoom needed to ensure the rotated rectangle covers the viewport
  // This is derived from the bounding box of a rotated rectangle
  return cosTheta + sinTheta * r;
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

    // At zoom > 1, extend horizontal range to allow reaching video edges.
    // The transform uses a fixed excessRatio (0.684), so at higher zoom
    // we need position values outside 0-1 to reach the actual edges.
    const widthRatio = (16 / 9) / (9 / 16); // ~3.16
    const baseExcessRatio = (widthRatio - 1) / widthRatio; // ~0.684
    const effectiveWidthRatio = widthRatio * zoom;
    const neededExcessRatio = (effectiveWidthRatio - 1) / effectiveWidthRatio;
    const extensionFactor = neededExcessRatio / baseExcessRatio;
    const halfRangeX = 0.5 * extensionFactor;

    // Vertical pan only available at zoom > 1
    const verticalPan = zoom > 1;
    return {
      minX: 0.5 - halfRangeX,
      maxX: 0.5 + halfRangeX,
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
      rotation: kf.rotation ?? 0,
    };
  }

  // Keyframes are maintained in sorted order by cameraStore.addKeyframe()
  // No need to re-sort here (this function is called 60x/second during playback)

  // Before first keyframe - use first keyframe values
  if (t <= keyframes[0].timeOffset) {
    const first = keyframes[0];
    return {
      positionX: first.positionX,
      positionY: first.positionY,
      zoom: first.zoom,
      rotation: first.rotation ?? 0,
    };
  }

  // After last keyframe - use last keyframe values
  if (t >= keyframes[keyframes.length - 1].timeOffset) {
    const last = keyframes[keyframes.length - 1];
    return {
      positionX: last.positionX,
      positionY: last.positionY,
      zoom: last.zoom,
      rotation: last.rotation ?? 0,
    };
  }

  // Find surrounding keyframes
  let beforeIdx = 0;
  for (let i = 0; i < keyframes.length - 1; i++) {
    if (keyframes[i].timeOffset <= t && keyframes[i + 1].timeOffset >= t) {
      beforeIdx = i;
      break;
    }
  }

  const before = keyframes[beforeIdx];
  const after = keyframes[beforeIdx + 1];

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
    rotation: lerp(before.rotation ?? 0, after.rotation ?? 0, easedT),
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
  const { positionX, positionY, zoom, rotation } = state;

  // Calculate rotation compensation zoom to prevent black corners
  const rotationZoom = getMinZoomForRotation(rotation, aspectRatio);
  const effectiveZoom = zoom * rotationZoom;

  if (aspectRatio === 'ORIGINAL') {
    // For original 16:9 aspect ratio with zoom/pan
    // At zoom 1.0, no additional scaling (video shows full frame)
    // Position 0.5 = center, translation = 0

    // Calculate pan range: at zoom Z, we see 1/Z of the video
    // Pan range is (Z - 1) / Z of the video width/height
    const panRangeX = (effectiveZoom - 1) / effectiveZoom;
    const panRangeY = (effectiveZoom - 1) / effectiveZoom;

    // Map position (0-1) to translation
    // Position 0.5 = center = 0 translation
    // Position 0 = show left edge = translate right by half pan range
    // Position 1 = show right edge = translate left by half pan range
    const translateX = (0.5 - positionX) * panRangeX * 100;
    const translateY = (0.5 - positionY) * panRangeY * 100;

    // Build transform string with optional rotation
    // Use 3D transforms for GPU acceleration
    const rotateStr = rotation !== 0 ? ` rotate3d(0, 0, 1, ${rotation}deg)` : '';

    return {
      transform: `scale3d(${effectiveZoom}, ${effectiveZoom}, 1) translate3d(${translateX}%, ${translateY}%, 0)${rotateStr}`,
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
    const panRangeY = (effectiveZoom - 1) / effectiveZoom;
    const translateY = (0.5 - positionY) * panRangeY * 100;

    // Build transform string with optional rotation
    // Use 3D transforms for GPU acceleration
    const rotateStr = rotation !== 0 ? ` rotate3d(0, 0, 1, ${rotation}deg)` : '';

    return {
      transform: `translate3d(${translateX}%, ${translateY}%, 0) scale3d(${effectiveZoom}, ${effectiveZoom}, 1)${rotateStr}`,
      transformOrigin: `${originX}% 50%`,
    };
  }
}

/**
 * Spring state manager for handheld motion.
 * Persists across frames for smooth physics simulation.
 */
interface HandheldState {
  springState: CameraSpringState | null;
  lastRallyId: string | null;
  lastTime: number;
  lastBaseState: CameraState | null;
}

let handheldState: HandheldState = {
  springState: null,
  lastRallyId: null,
  lastTime: 0,
  lastBaseState: null,
};

/**
 * Reset handheld state (call when seeking or switching rallies).
 */
export function resetHandheldState(): void {
  handheldState = {
    springState: null,
    lastRallyId: null,
    lastTime: 0,
    lastBaseState: null,
  };
}

/**
 * Get interpolated camera state with optional handheld motion.
 * This is the main entry point during playback when handheld is enabled.
 *
 * @param keyframes Array of camera keyframes
 * @param timeOffset Time within rally as fraction (0.0 to 1.0)
 * @param absoluteTime Absolute video time in seconds (for noise sampling)
 * @param rallyId Current rally ID (for state reset detection)
 * @param preset Handheld simulation preset
 * @returns Camera state with handheld motion applied
 */
export function getCameraStateWithHandheld(
  keyframes: CameraKeyframe[],
  timeOffset: number,
  absoluteTime: number,
  rallyId: string,
  preset: HandheldPreset
): CameraState {
  // Get base interpolated state
  const baseState = interpolateCameraState(keyframes, timeOffset);

  // Early exit if handheld is off
  if (preset === 'OFF') {
    return baseState;
  }

  const config = HANDHELD_PRESET_CONFIG[preset];

  // Calculate time delta
  const timeDelta = absoluteTime - handheldState.lastTime;

  // Reset state if rally changed or we seeked significantly (>0.5s jump)
  if (
    rallyId !== handheldState.lastRallyId ||
    Math.abs(timeDelta) > 0.5 ||
    timeDelta < 0
  ) {
    handheldState = {
      springState: null,
      lastRallyId: rallyId,
      lastTime: absoluteTime,
      lastBaseState: { ...baseState },
    };
  }

  // Track previous state for velocity calculation
  const prevState = handheldState.lastBaseState || baseState;
  handheldState.lastBaseState = { ...baseState };
  handheldState.lastTime = absoluteTime;
  handheldState.lastRallyId = rallyId;

  // Apply spring physics if enabled
  let processedState = baseState;
  if (config.springEnabled && timeDelta > 0 && timeDelta < 0.1) {
    if (!handheldState.springState) {
      handheldState.springState = createCameraSpringState({
        x: baseState.positionX,
        y: baseState.positionY,
        zoom: baseState.zoom,
      });
    }

    const springConfig = {
      stiffness: config.springStiffness,
      damping: config.springDamping,
      mass: 1,
    };

    handheldState.springState = stepCameraSpring(
      handheldState.springState,
      { x: baseState.positionX, y: baseState.positionY, zoom: baseState.zoom },
      springConfig,
      timeDelta
    );

    processedState = {
      positionX: handheldState.springState.x.position,
      positionY: handheldState.springState.y.position,
      zoom: handheldState.springState.zoom.position,
      rotation: baseState.rotation, // Rotation is not spring-animated
    };
  }

  // Calculate velocity for movement-dependent shake
  const velocity = calculateCameraVelocity(prevState, baseState, timeDelta || 0.016);

  // Apply handheld motion (wobble, breathing, movement shake)
  return applyHandheldMotion(processedState, absoluteTime, velocity, preset);
}

