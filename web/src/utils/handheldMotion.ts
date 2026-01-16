/**
 * Handheld camera motion simulation.
 * Combines noise-based wobble, breathing, and movement-dependent shake
 * to create realistic "person holding a camera" motion.
 */

import { cameraNoiseGenerator } from './noise';
import { CameraState, HandheldPreset, HandheldPresetConfig, HANDHELD_PRESET_CONFIG } from '@/types/camera';

/**
 * Maximum displacement values at 100% intensity.
 * Values are in the 0-1 normalized coordinate space.
 */
const MAX_WOBBLE = {
  position: 0.008,  // ~0.8% of frame at max
  zoom: 0.02,       // ~2% zoom variation
};

/**
 * Breathing parameters (very slow vertical drift).
 * Simulates the subtle up/down motion from breathing while filming.
 */
const BREATHING = {
  frequency: 0.2,   // Hz (one breath every 5 seconds)
  amplitude: 0.003, // Max vertical drift at 100%
};

/**
 * Velocity needed to trigger movement shake.
 * Values below this threshold don't add extra shake.
 */
const VELOCITY_THRESHOLD = 0.05;

/**
 * Apply handheld motion simulation to a camera state.
 *
 * @param baseState The clean interpolated camera state
 * @param time Current playback time in seconds (for noise sampling)
 * @param velocity Current camera velocity (for movement-dependent shake)
 * @param preset Handheld simulation preset
 * @returns Modified camera state with handheld motion applied
 */
export function applyHandheldMotion(
  baseState: CameraState,
  time: number,
  velocity: { x: number; y: number; zoom: number },
  preset: HandheldPreset
): CameraState {
  const config = HANDHELD_PRESET_CONFIG[preset];

  // Early exit for OFF preset
  if (preset === 'OFF' || config.intensity === 0) {
    return baseState;
  }

  const { intensity, panIntensity, zoomIntensity, breathingIntensity } = config;

  // 1. IDLE WOBBLE - constant subtle drift
  // Use different noise coordinates for X and Y to decorrelate them
  const noiseScale = 0.8; // Lower = slower drift
  const wobbleX =
    cameraNoiseGenerator.fbm(time * noiseScale, 0, 2) *
    MAX_WOBBLE.position *
    panIntensity *
    intensity;
  const wobbleY =
    cameraNoiseGenerator.fbm(0, time * noiseScale, 2) *
    MAX_WOBBLE.position *
    panIntensity *
    intensity;
  const wobbleZoom =
    cameraNoiseGenerator.fbm(time * noiseScale * 0.7, 100, 2) *
    MAX_WOBBLE.zoom *
    zoomIntensity *
    intensity;

  // 2. BREATHING - very slow vertical oscillation
  const breathingOffset =
    Math.sin(time * Math.PI * 2 * BREATHING.frequency) *
    BREATHING.amplitude *
    breathingIntensity *
    intensity;

  // 3. MOVEMENT-DEPENDENT SHAKE - more shake during fast pans
  // Calculate velocity magnitude
  const velocityMagnitude = Math.sqrt(velocity.x ** 2 + velocity.y ** 2);

  let moveShakeX = 0;
  let moveShakeY = 0;

  if (velocityMagnitude > VELOCITY_THRESHOLD) {
    // Scale shake based on velocity - kept subtle to avoid jitter during fast transitions
    const shakeMultiplier = Math.min((velocityMagnitude - VELOCITY_THRESHOLD) * 4, 1);

    // Higher frequency noise during movement for subtle jitter
    const moveNoiseScale = 2.5;
    moveShakeX =
      cameraNoiseGenerator.noise2D(time * moveNoiseScale, 50) *
      MAX_WOBBLE.position *
      0.3 *
      panIntensity *
      intensity *
      shakeMultiplier;
    moveShakeY =
      cameraNoiseGenerator.noise2D(50, time * moveNoiseScale) *
      MAX_WOBBLE.position *
      0.3 *
      panIntensity *
      intensity *
      shakeMultiplier;
  }

  return {
    positionX: baseState.positionX + wobbleX + moveShakeX,
    positionY: baseState.positionY + wobbleY + breathingOffset + moveShakeY,
    zoom: baseState.zoom * (1 + wobbleZoom),
    rotation: baseState.rotation, // Rotation passed through without handheld motion
  };
}

/**
 * Calculate velocity from consecutive camera states.
 * Used for movement-dependent shake calculation.
 *
 * @param prevState Previous camera state
 * @param currentState Current camera state
 * @param deltaTime Time between states in seconds
 * @returns Velocity for each dimension
 */
export function calculateCameraVelocity(
  prevState: CameraState,
  currentState: CameraState,
  deltaTime: number
): { x: number; y: number; zoom: number } {
  if (deltaTime <= 0) {
    return { x: 0, y: 0, zoom: 0 };
  }

  return {
    x: (currentState.positionX - prevState.positionX) / deltaTime,
    y: (currentState.positionY - prevState.positionY) / deltaTime,
    zoom: (currentState.zoom - prevState.zoom) / deltaTime,
  };
}

/**
 * Get preset display name for UI.
 */
export function getPresetDisplayName(preset: HandheldPreset): string {
  const names: Record<HandheldPreset, string> = {
    OFF: 'Off',
    NATURAL: 'Natural',
  };
  return names[preset];
}

/**
 * Check if handheld motion is enabled.
 */
export function isHandheldEnabled(preset: HandheldPreset): boolean {
  return preset !== 'OFF';
}

/**
 * Get preset configuration for internal use.
 */
export function getPresetConfig(preset: HandheldPreset): HandheldPresetConfig {
  return HANDHELD_PRESET_CONFIG[preset];
}
