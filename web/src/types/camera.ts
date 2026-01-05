/**
 * Camera tracking types for Instagram-style video editing.
 * Supports per-rally camera edits with keyframes for pan/zoom.
 */

/** Output aspect ratio options */
export type AspectRatio = 'ORIGINAL' | 'VERTICAL';

/** Easing function for keyframe interpolation */
export type CameraEasing = 'LINEAR' | 'EASE_IN' | 'EASE_OUT' | 'EASE_IN_OUT';

/** A single camera keyframe within a rally */
export interface CameraKeyframe {
  id: string;
  /** Time offset within rally (0.0 = start, 1.0 = end) */
  timeOffset: number;
  /** Horizontal position (0.0 = left, 0.5 = center, 1.0 = right) */
  positionX: number;
  /** Vertical position (0.0 = top, 0.5 = center, 1.0 = bottom) */
  positionY: number;
  /** Zoom level (1.0 = no zoom, 2.0 = 2x zoom, etc.) */
  zoom: number;
  /** Easing to apply when transitioning to next keyframe */
  easing: CameraEasing;
}

/** Keyframes organized by aspect ratio */
export type KeyframesPerAspectRatio = {
  ORIGINAL: CameraKeyframe[];
  VERTICAL: CameraKeyframe[];
};

/** Camera edit settings for a rally */
export interface RallyCameraEdit {
  /** Whether camera tracking is enabled for this rally */
  enabled: boolean;
  /** Currently active aspect ratio for editing/preview */
  aspectRatio: AspectRatio;
  /** Keyframes per aspect ratio - each ratio has its own camera movements */
  keyframes: KeyframesPerAspectRatio;
}

/** Computed camera state at any point in time (interpolated) */
export interface CameraState {
  positionX: number;
  positionY: number;
  zoom: number;
}

/** Default camera edit settings */
export const DEFAULT_CAMERA_EDIT: RallyCameraEdit = {
  enabled: false,
  aspectRatio: 'ORIGINAL',
  keyframes: {
    ORIGINAL: [],
    VERTICAL: [],
  },
};

/** Default keyframe values (centered, no zoom) */
export const DEFAULT_KEYFRAME: Omit<CameraKeyframe, 'id' | 'timeOffset'> = {
  positionX: 0.5,
  positionY: 0.5,
  zoom: 1.0,
  easing: 'EASE_IN_OUT',
};

/** Default camera state (centered, no zoom) */
export const DEFAULT_CAMERA_STATE: CameraState = {
  positionX: 0.5,
  positionY: 0.5,
  zoom: 1.0,
};

/** Zoom constraints */
export const ZOOM_MIN = 0.5;  // Allow zoom out (shows letterboxing in vertical mode)
export const ZOOM_MAX = 3.0;
export const ZOOM_STEP = 0.1;

/** Time threshold for keyframe proximity (5% of rally duration) */
export const KEYFRAME_TIME_THRESHOLD = 0.05;

/**
 * Migrate old camera edit format (single keyframes array) to new format (per-aspect-ratio).
 * Used when loading from localStorage or API that may have old format.
 */
export function migrateCameraEdit(edit: RallyCameraEdit | { keyframes: CameraKeyframe[] }): RallyCameraEdit {
  // Check if already new format
  if (edit.keyframes && !Array.isArray(edit.keyframes)) {
    return edit as RallyCameraEdit;
  }

  // Old format: keyframes is an array - migrate to new format
  const oldKeyframes = Array.isArray(edit.keyframes) ? edit.keyframes : [];
  return {
    enabled: (edit as RallyCameraEdit).enabled ?? false,
    aspectRatio: (edit as RallyCameraEdit).aspectRatio ?? 'ORIGINAL',
    keyframes: {
      ORIGINAL: oldKeyframes,
      VERTICAL: [],
    },
  };
}
