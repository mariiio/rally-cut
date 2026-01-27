import type { CameraEasing } from '@/types/camera';

/** Raw keyframe data from the API */
interface ApiKeyframe {
  id: string;
  timeOffset: number;
  positionX: number;
  positionY: number;
  zoom: number;
  rotation?: number;
  easing: CameraEasing;
}

/** Keyframe format for sync requests (no id) */
export interface SyncKeyframe {
  timeOffset: number;
  positionX: number;
  positionY: number;
  zoom: number;
  rotation: number;
  easing: CameraEasing;
}

/** Keyframe format with required id (from API responses) */
export interface MappedKeyframe extends SyncKeyframe {
  id: string;
}

/**
 * Map API response keyframes to frontend format (with id).
 * Normalizes rotation to default 0 when missing.
 */
export function mapApiKeyframes(keyframes: ApiKeyframe[]): MappedKeyframe[] {
  return keyframes.map(kf => ({
    id: kf.id,
    timeOffset: kf.timeOffset,
    positionX: kf.positionX,
    positionY: kf.positionY,
    zoom: kf.zoom,
    rotation: kf.rotation ?? 0,
    easing: kf.easing,
  }));
}

/**
 * Map frontend keyframes to sync format (no id).
 * Used when sending state to backend.
 */
export function mapSyncKeyframes(keyframes: Array<{
  timeOffset: number;
  positionX: number;
  positionY: number;
  zoom: number;
  rotation?: number;
  easing: CameraEasing;
}>): SyncKeyframe[] {
  return keyframes.map(kf => ({
    timeOffset: kf.timeOffset,
    positionX: kf.positionX,
    positionY: kf.positionY,
    zoom: kf.zoom,
    rotation: kf.rotation ?? 0,
    easing: kf.easing,
  }));
}
