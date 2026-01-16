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
  /** Rotation in degrees (-30 to +30, clockwise positive) */
  rotation: number;
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
  rotation: number;
}

/** Global camera settings at video level (applies as base for all rallies) */
export interface GlobalCameraSettings {
  /** Base zoom level (1.0 = none) - multiplied with per-rally zoom */
  zoom: number;
  /** Base horizontal position (0.5 = center) - added to per-rally offset */
  positionX: number;
  /** Base vertical position (0.5 = center) - added to per-rally offset */
  positionY: number;
  /** Base rotation in degrees - added to per-rally rotation */
  rotation: number;
}

/** Default global camera settings (neutral, no effect) */
export const DEFAULT_GLOBAL_CAMERA: GlobalCameraSettings = {
  zoom: 1.0,
  positionX: 0.5,
  positionY: 0.5,
  rotation: 0,
};

/** Default camera edit settings */
export const DEFAULT_CAMERA_EDIT: RallyCameraEdit = {
  enabled: false,
  aspectRatio: 'ORIGINAL',
  keyframes: {
    ORIGINAL: [],
    VERTICAL: [],
  },
};

/** Default keyframe values (centered, no zoom, no rotation) */
export const DEFAULT_KEYFRAME: Omit<CameraKeyframe, 'id' | 'timeOffset'> = {
  positionX: 0.5,
  positionY: 0.5,
  zoom: 1.0,
  rotation: 0,
  easing: 'EASE_IN_OUT',
};

/** Default camera state (centered, no zoom, no rotation) */
export const DEFAULT_CAMERA_STATE: CameraState = {
  positionX: 0.5,
  positionY: 0.5,
  zoom: 1.0,
  rotation: 0,
};

/** Zoom constraints */
export const ZOOM_MIN = 0.5;  // Allow zoom out (shows letterboxing in vertical mode)
export const ZOOM_MAX = 3.0;
export const ZOOM_STEP = 0.1;

/** Rotation constraints (degrees) */
export const ROTATION_MIN = -30;
export const ROTATION_MAX = 30;
export const ROTATION_STEP = 0.5;

/** Time threshold for keyframe proximity (5% of rally duration) */
export const KEYFRAME_TIME_THRESHOLD = 0.05;

/** Handheld camera motion preset (simple on/off) */
export type HandheldPreset = 'OFF' | 'NATURAL';

/** Internal configuration for each handheld preset */
export interface HandheldPresetConfig {
  /** Overall intensity multiplier (0-1) */
  intensity: number;
  /** Pan wobble intensity (0-1) */
  panIntensity: number;
  /** Zoom wobble intensity (0-1) */
  zoomIntensity: number;
  /** Breathing effect intensity (0-1) */
  breathingIntensity: number;
  /** Enable spring physics for overshoot */
  springEnabled: boolean;
  /** Spring stiffness (higher = snappier) */
  springStiffness: number;
  /** Spring damping (higher = less bounce) */
  springDamping: number;
}

/** Preset configurations (internal, not exposed to user) */
export const HANDHELD_PRESET_CONFIG: Record<HandheldPreset, HandheldPresetConfig> = {
  OFF: {
    intensity: 0,
    panIntensity: 0,
    zoomIntensity: 0,
    breathingIntensity: 0,
    springEnabled: false,
    springStiffness: 200,
    springDamping: 20,
  },
  NATURAL: {
    intensity: 0.5,
    panIntensity: 0.5,
    zoomIntensity: 0.25,
    breathingIntensity: 0.4,
    springEnabled: false,  // Disabled: causes lag on fast transitions
    springStiffness: 400,
    springDamping: 28,
  },
};

/**
 * Ensure a keyframe has all required fields (adds rotation if missing).
 */
function migrateKeyframe(kf: Partial<CameraKeyframe> & { id: string; timeOffset: number }): CameraKeyframe {
  return {
    id: kf.id,
    timeOffset: kf.timeOffset,
    positionX: kf.positionX ?? 0.5,
    positionY: kf.positionY ?? 0.5,
    zoom: kf.zoom ?? 1.0,
    rotation: kf.rotation ?? 0,
    easing: kf.easing ?? 'EASE_IN_OUT',
  };
}

/**
 * Migrate old camera edit format (single keyframes array) to new format (per-aspect-ratio).
 * Also ensures all keyframes have the rotation field.
 * Used when loading from localStorage or API that may have old format.
 */
export function migrateCameraEdit(edit: RallyCameraEdit | { keyframes: CameraKeyframe[] }): RallyCameraEdit {
  // Check if already new format
  if (edit.keyframes && !Array.isArray(edit.keyframes)) {
    const typedEdit = edit as RallyCameraEdit;
    // Ensure all keyframes have rotation field
    return {
      ...typedEdit,
      keyframes: {
        ORIGINAL: (typedEdit.keyframes.ORIGINAL ?? []).map(migrateKeyframe),
        VERTICAL: (typedEdit.keyframes.VERTICAL ?? []).map(migrateKeyframe),
      },
    };
  }

  // Old format: keyframes is an array - migrate to new format
  const oldKeyframes = Array.isArray(edit.keyframes) ? edit.keyframes : [];
  return {
    enabled: (edit as RallyCameraEdit).enabled ?? false,
    aspectRatio: (edit as RallyCameraEdit).aspectRatio ?? 'ORIGINAL',
    keyframes: {
      ORIGINAL: oldKeyframes.map(migrateKeyframe),
      VERTICAL: [],
    },
  };
}
