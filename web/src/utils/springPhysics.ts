/**
 * Spring physics simulation for realistic camera overshoot/settle effects.
 * Based on damped harmonic oscillator model.
 */

export interface SpringState {
  position: number;
  velocity: number;
}

export interface SpringConfig {
  stiffness: number;  // Spring constant (k) - higher = snappier
  damping: number;    // Damping coefficient (c) - higher = less bounce
  mass: number;       // Mass (usually 1)
}

/**
 * Perform a single spring step with fixed timestep.
 */
function springStepInternal(
  state: SpringState,
  target: number,
  config: SpringConfig,
  dt: number
): SpringState {
  const { stiffness, damping, mass } = config;

  // F = -k(x - target) - c*v
  const displacement = state.position - target;
  const springForce = -stiffness * displacement;
  const dampingForce = -damping * state.velocity;
  const acceleration = (springForce + dampingForce) / mass;

  // Semi-implicit Euler (velocity first, then position)
  const newVelocity = state.velocity + acceleration * dt;
  const newPosition = state.position + newVelocity * dt;

  return { position: newPosition, velocity: newVelocity };
}

/**
 * Step the spring simulation forward by deltaTime seconds.
 * Uses sub-stepping for stability and smooth tracking of fast-moving targets.
 *
 * @param state Current spring state
 * @param target Target position
 * @param config Spring configuration
 * @param dt Delta time in seconds
 * @returns New spring state
 */
export function stepSpring(
  state: SpringState,
  target: number,
  config: SpringConfig,
  dt: number
): SpringState {
  // Sub-step for stability and smooth fast transitions
  // Fixed timestep of ~8ms (120fps internal simulation)
  const FIXED_STEP = 0.008;
  const maxSteps = 8; // Cap at ~64ms to prevent runaway

  const clampedDt = Math.min(dt, FIXED_STEP * maxSteps);
  const numSteps = Math.ceil(clampedDt / FIXED_STEP);
  const stepDt = clampedDt / numSteps;

  let currentState = state;
  for (let i = 0; i < numSteps; i++) {
    currentState = springStepInternal(currentState, target, config, stepDt);
  }

  return currentState;
}

/**
 * Check if spring has settled (close to target with low velocity).
 */
export function isSpringSettled(
  state: SpringState,
  target: number,
  positionThreshold: number = 0.0001,
  velocityThreshold: number = 0.001
): boolean {
  return (
    Math.abs(state.position - target) < positionThreshold &&
    Math.abs(state.velocity) < velocityThreshold
  );
}

/**
 * Multi-dimensional spring state for camera (x, y, zoom).
 */
export interface CameraSpringState {
  x: SpringState;
  y: SpringState;
  zoom: SpringState;
}

/**
 * Create initial camera spring state.
 */
export function createCameraSpringState(initial: {
  x: number;
  y: number;
  zoom: number;
}): CameraSpringState {
  return {
    x: { position: initial.x, velocity: 0 },
    y: { position: initial.y, velocity: 0 },
    zoom: { position: initial.zoom, velocity: 0 },
  };
}

/**
 * Step all camera spring dimensions.
 * Zoom uses softer spring (half stiffness) for smoother feel.
 */
export function stepCameraSpring(
  state: CameraSpringState,
  target: { x: number; y: number; zoom: number },
  config: SpringConfig,
  dt: number
): CameraSpringState {
  // Zoom is softer (half stiffness, more damping) for smoother feel
  const zoomConfig = {
    ...config,
    stiffness: config.stiffness * 0.5,
    damping: config.damping * 1.2,
  };

  return {
    x: stepSpring(state.x, target.x, config, dt),
    y: stepSpring(state.y, target.y, config, dt),
    zoom: stepSpring(state.zoom, target.zoom, zoomConfig, dt),
  };
}

/**
 * Check if all camera spring dimensions have settled.
 */
export function isCameraSpringSettled(
  state: CameraSpringState,
  target: { x: number; y: number; zoom: number }
): boolean {
  return (
    isSpringSettled(state.x, target.x) &&
    isSpringSettled(state.y, target.y) &&
    isSpringSettled(state.zoom, target.zoom, 0.001) // Looser threshold for zoom
  );
}
