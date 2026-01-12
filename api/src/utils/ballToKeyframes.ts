/**
 * Ball tracking to camera keyframes conversion algorithm.
 *
 * Uses simple moving average smoothing (matching the reference implementation
 * from fast-volleyball-tracking-inference) for stable camera tracking.
 */

import type { AspectRatio, CameraEasing } from '@prisma/client';

// ============================================================================
// Types
// ============================================================================

export interface BallPosition {
  frameNumber: number;
  x: number; // Normalized 0-1
  y: number; // Normalized 0-1
  confidence: number;
}

export interface CameraKeyframe {
  timeOffset: number; // 0-1 within rally
  positionX: number; // 0-1 camera center
  positionY: number; // 0-1 camera center
  zoom: number; // 1.0-3.0
  easing: CameraEasing;
}

export interface TrackingQuality {
  coverage: number;
  averageConfidence: number;
  isUsable: boolean;
  recommendation: string;
}

export interface BallToKeyframesResult {
  keyframes: CameraKeyframe[];
  quality: TrackingQuality;
}

// ============================================================================
// Configuration
// ============================================================================

export interface BallToKeyframesConfig {
  // Minimum confidence to accept a detection
  minConfidence: number;
  // Moving average window size in frames (used as fallback)
  smoothingWindow: number;
  // Keyframe interval in seconds (used as max spacing for smart keyframes)
  keyframeInterval: number;
  // Margin to keep ball visible (0-0.5)
  marginX: number;
  marginY: number;
  // Base zoom level
  baseZoom: number;
  // Minimum coverage to consider tracking usable
  minCoverage: number;
  // Camera lag factor (0 = instant follow, 0.95 = very slow follow)
  // Higher values = smoother but slower camera response
  cameraLag: number;
  // Maximum jump distance (normalized 0-1) to accept a detection
  // Detections that jump further than this from recent average are rejected as outliers
  maxJumpDistance: number;
  // Number of frames to use for outlier detection baseline
  outlierWindowFrames: number;
  // Exponential smoothing alpha (0-1): higher = more responsive, lower = smoother
  exponentialAlpha: number;
  // Number of frames to look ahead for camera anticipation
  lookaheadFrames: number;
  // Weight for lookahead blending (0-1): 0 = no lookahead, 1 = full lookahead
  lookaheadWeight: number;
  // Smart keyframe placement settings
  minKeyframeSpacing: number; // Minimum seconds between keyframes
  maxKeyframeSpacing: number; // Maximum seconds between keyframes (fill gaps)
  directionChangeThreshold: number; // Angle in degrees to trigger keyframe (0-180)
  positionChangeThreshold: number; // Position change (0-1) to trigger keyframe
  // Timing sync offset in SECONDS: delays ball positions to align with actual video playback
  // Positive = tracking was ahead, delay it to match video
  // This compensates for ML model temporal context and video decode latency
  // Using seconds (not frames) so it works consistently across all FPS values
  syncOffsetSeconds: number;
  // Physics-based filtering: max velocity per frame (normalized 0-1)
  // Ball can't teleport - reject detections exceeding this velocity
  maxVelocityPerFrame: number;
  // Minimum segment length (frames) to consider reliable
  minSegmentFrames: number;
  // Maximum gap (frames) allowed within a segment
  maxGapFrames: number;
  // Offset corrections: compensate for ML model detection center being slightly off
  // Positive Y = shift down (if tracking appears above actual ball)
  // Negative X = shift left (if tracking appears to the right of actual ball)
  yOffsetCorrection: number;
  xOffsetCorrection: number;

  // Hard limit on maximum keyframes (prevents over-tracking)
  maxKeyframes: number;
  // Maximum position change allowed between consecutive keyframes (0-1)
  // Keyframes with larger jumps are removed to ensure smooth trajectory
  maxPositionJump: number;
  // Lock vertical (Y) position to 0.5 (center) - for ORIGINAL mode
  // Prevents jerky vertical camera movement in wide 16:9 videos
  lockVertical: boolean;
}

// Tracking segment: continuous run of reliable ball detections
interface TrackingSegment {
  startFrame: number;
  endFrame: number;
  positions: BallPosition[];
  avgConfidence: number;
  avgVelocity: number;
  score: number; // Higher = more reliable
}

export const DEFAULT_CONFIG: BallToKeyframesConfig = {
  minConfidence: 0.3,
  smoothingWindow: 30,
  keyframeInterval: 3.0, // Max spacing for smart keyframes
  marginX: 0.15,
  marginY: 0.12,
  baseZoom: 1.0,
  minCoverage: 0.15,
  cameraLag: 0,
  maxJumpDistance: 0.3, // Reject detections that jump more than 30% of frame
  outlierWindowFrames: 10,
  exponentialAlpha: 0.15, // Moderate smoothing
  lookaheadFrames: 15, // 0.5s at 30fps
  lookaheadWeight: 0.4, // 40% lookahead blend
  // Smart keyframe settings
  minKeyframeSpacing: 0.5, // Min 0.5s between keyframes
  maxKeyframeSpacing: 3.0, // Force keyframe if none for 3s
  directionChangeThreshold: 45, // 45 degree direction change triggers keyframe
  positionChangeThreshold: 0.15, // 15% position change triggers keyframe
  syncOffsetSeconds: 0, // No offset by default
  // Physics-based segment filtering
  maxVelocityPerFrame: 0.15, // ~30 m/s normalized - ball can't exceed this
  minSegmentFrames: 10, // Min 10 frames for reliable segment
  maxGapFrames: 5, // Allow 5-frame gaps within segment
  // Offset corrections (calibration)
  yOffsetCorrection: 0, // No correction by default
  xOffsetCorrection: 0,
  // Keyframe limits
  maxKeyframes: 10, // Default max 10 keyframes
  maxPositionJump: 0.35, // Max 35% position change between keyframes
  lockVertical: false, // Allow Y tracking by default
};

// Original (16:9) - horizontal tracking only, minimal keyframes
// For wide aspect ratio, vertical camera movement looks jerky.
// Lock Y to center and only pan horizontally with zoom.
export const ORIGINAL_CONFIG: Partial<BallToKeyframesConfig> = {
  minConfidence: 0.6,
  smoothingWindow: 5,
  keyframeInterval: 5.0, // Max spacing between keyframes
  marginX: 0.12, // Horizontal margins for panning
  marginY: 0.0, // No Y margin (locked to center)
  baseZoom: 1.3, // Slight zoom to enable panning
  cameraLag: 0,
  maxJumpDistance: 0.3,
  outlierWindowFrames: 15,
  exponentialAlpha: 0.3,
  lookaheadFrames: 0, // No lookahead - more predictable
  lookaheadWeight: 0,
  // Keyframe placement - SPARSE for smooth panning
  minKeyframeSpacing: 3.0, // 3s minimum between keyframes
  maxKeyframeSpacing: 5.0, // 5s max gap
  directionChangeThreshold: 90,
  positionChangeThreshold: 0.30, // Larger threshold for 16:9
  syncOffsetSeconds: 0,
  // Physics-based filtering
  maxVelocityPerFrame: 0.1,
  minSegmentFrames: 25, // Need longer segments for confidence
  maxGapFrames: 5,
  // Offset corrections
  yOffsetCorrection: 0,
  xOffsetCorrection: 0,
  // NEW: Hard limits for smooth camera
  maxKeyframes: 5, // Max 5 keyframes for 16:9
  maxPositionJump: 0.30, // 30% max jump between keyframes
  lockVertical: true, // Lock Y to 0.5 (center) - no vertical movement
};

// Vertical (9:16) - crop to portrait, anchor-based camera tracking
// Places keyframes at high-confidence anchor frames where ball position is reliable.
// Camera follows ball position directly at anchors, CSS interpolation smooths between.
export const VERTICAL_CONFIG: Partial<BallToKeyframesConfig> = {
  minConfidence: 0.65, // Higher threshold for reliable anchors
  smoothingWindow: 5, // Minimal smoothing - just denoise
  keyframeInterval: 4.0, // Max spacing between keyframes
  marginX: 0.12, // Keep ball within margins
  marginY: 0.12,
  baseZoom: 1.25,
  cameraLag: 0,
  maxJumpDistance: 0.2, // Tighter outlier rejection
  outlierWindowFrames: 15,
  exponentialAlpha: 0.3, // Light smoothing - stay close to actual ball
  lookaheadFrames: 0, // No lookahead - follow ball directly
  lookaheadWeight: 0,
  // Anchor-based keyframe placement - SMOOTH settings
  minKeyframeSpacing: 2.5, // 2.5s minimum between keyframes (was 1.5s)
  maxKeyframeSpacing: 4.0, // 4s max gap
  directionChangeThreshold: 90, // Only major direction changes trigger keyframe
  positionChangeThreshold: 0.25, // Larger position change needed for keyframe
  syncOffsetSeconds: 0,
  // Physics-based filtering
  maxVelocityPerFrame: 0.08, // Tighter - ball can't teleport
  minSegmentFrames: 20, // Need 20+ frames for reliable segment (was 5)
  maxGapFrames: 3, // Tighter gaps
  // Offset corrections: tracking appears above and to the right of actual ball
  yOffsetCorrection: 0.03, // 3% down
  xOffsetCorrection: 0.0, // no X offset needed
  // NEW: Hard limits for smooth camera
  maxKeyframes: 8, // Max 8 keyframes for 9:16
  maxPositionJump: 0.25, // 25% max jump between keyframes
  lockVertical: false, // Allow Y tracking for portrait mode
};

// ============================================================================
// Outlier Filtering
// ============================================================================

/**
 * Filter out outlier detections that jump too far from the main ball trajectory.
 * Uses median (robust to outliers) instead of mean for baseline calculation.
 * Requires minimum baseline before accepting detections to establish stable trajectory.
 */
function filterOutliers(
  positions: BallPosition[],
  maxJumpDistance: number,
  windowFrames: number,
  minConfidence: number
): BallPosition[] {
  if (positions.length === 0) return [];

  // Sort by frame number
  const sorted = [...positions]
    .filter((p) => p.confidence >= minConfidence)
    .sort((a, b) => a.frameNumber - b.frameNumber);

  if (sorted.length === 0) return [];

  const result: BallPosition[] = [];
  const recentPositions: Array<{ x: number; y: number; frame: number }> = [];

  // Helper to calculate median
  const median = (values: number[]): number => {
    const s = [...values].sort((a, b) => a - b);
    const mid = Math.floor(s.length / 2);
    return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
  };

  // Minimum positions needed to establish baseline (prevents early outliers from anchoring)
  const minBaseline = 3;

  for (const pos of sorted) {
    // Remove old positions outside the window
    while (
      recentPositions.length > 0 &&
      pos.frameNumber - recentPositions[0].frame > windowFrames
    ) {
      recentPositions.shift();
    }

    if (recentPositions.length < minBaseline) {
      // Building baseline - accept high-confidence detections that are close to each other
      if (recentPositions.length === 0) {
        // First position, accept it
        result.push(pos);
        recentPositions.push({ x: pos.x, y: pos.y, frame: pos.frameNumber });
      } else {
        // Check against existing baseline positions
        const medX = median(recentPositions.map((p) => p.x));
        const medY = median(recentPositions.map((p) => p.y));
        const distance = Math.sqrt((pos.x - medX) ** 2 + (pos.y - medY) ** 2);

        // Use tighter threshold during baseline building
        if (distance <= maxJumpDistance * 0.7) {
          result.push(pos);
          recentPositions.push({ x: pos.x, y: pos.y, frame: pos.frameNumber });
        }
      }
    } else {
      // Baseline established - use median for robust outlier detection
      const medX = median(recentPositions.map((p) => p.x));
      const medY = median(recentPositions.map((p) => p.y));

      // Calculate distance from median (more robust than mean)
      const distance = Math.sqrt((pos.x - medX) ** 2 + (pos.y - medY) ** 2);

      if (distance <= maxJumpDistance) {
        // Accept this position
        result.push(pos);
        recentPositions.push({ x: pos.x, y: pos.y, frame: pos.frameNumber });
      }
      // else: reject as outlier (false positive)
    }
  }

  return result;
}

// ============================================================================
// Physics-Based Segment Filtering
// ============================================================================

/**
 * Filter positions by physics constraints - ball can't teleport.
 * Rejects detections that exceed maximum velocity from previous position.
 */
function filterByPhysics(
  positions: BallPosition[],
  maxVelocityPerFrame: number,
  minConfidence: number
): BallPosition[] {
  if (positions.length === 0) return [];

  // Sort by frame number and filter by confidence
  const sorted = [...positions]
    .filter((p) => p.confidence >= minConfidence)
    .sort((a, b) => a.frameNumber - b.frameNumber);

  if (sorted.length === 0) return [];

  const result: BallPosition[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    const prev = result[result.length - 1];
    const curr = sorted[i];
    const frameDiff = curr.frameNumber - prev.frameNumber;

    // Max distance scales with frame gap (ball can travel further over more frames)
    const maxDist = maxVelocityPerFrame * Math.max(1, frameDiff);

    const dist = Math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2);

    if (dist <= maxDist) {
      result.push(curr);
    }
    // else: rejected as physically impossible (teleport)
  }

  return result;
}

/**
 * Find continuous segments of reliable ball tracking.
 * A segment is a run of detections where:
 * - Frames are consecutive (or within maxGapFrames)
 * - Movement is physically plausible
 * - Confidence is above threshold
 */
function findReliableSegments(
  positions: BallPosition[],
  minConfidence: number,
  maxGapFrames: number,
  minSegmentFrames: number,
  maxVelocityPerFrame: number
): TrackingSegment[] {
  if (positions.length === 0) return [];

  // Sort by frame
  const sorted = [...positions]
    .filter((p) => p.confidence >= minConfidence)
    .sort((a, b) => a.frameNumber - b.frameNumber);

  if (sorted.length === 0) return [];

  const segments: TrackingSegment[] = [];
  let currentSegment: BallPosition[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    const prev = currentSegment[currentSegment.length - 1];
    const curr = sorted[i];
    const frameDiff = curr.frameNumber - prev.frameNumber;

    // Check if this detection continues the segment
    const dist = Math.sqrt((curr.x - prev.x) ** 2 + (curr.y - prev.y) ** 2);
    const maxDist = maxVelocityPerFrame * Math.max(1, frameDiff);
    const isPhysicallyPlausible = dist <= maxDist;
    const isWithinGap = frameDiff <= maxGapFrames;

    if (isPhysicallyPlausible && isWithinGap) {
      // Continue segment
      currentSegment.push(curr);
    } else {
      // End current segment, start new one
      if (currentSegment.length >= 2) {
        const segment = createSegment(currentSegment);
        if (segment.endFrame - segment.startFrame >= minSegmentFrames) {
          segments.push(segment);
        }
      }
      currentSegment = [curr];
    }
  }

  // Don't forget last segment
  if (currentSegment.length >= 2) {
    const segment = createSegment(currentSegment);
    if (segment.endFrame - segment.startFrame >= minSegmentFrames) {
      segments.push(segment);
    }
  }

  // Sort by score (longer, higher confidence = better)
  return segments.sort((a, b) => b.score - a.score);
}

/**
 * Create a segment object from a list of positions.
 */
function createSegment(positions: BallPosition[]): TrackingSegment {
  const avgConfidence =
    positions.reduce((sum, p) => sum + p.confidence, 0) / positions.length;

  // Calculate average velocity
  let totalVelocity = 0;
  for (let i = 1; i < positions.length; i++) {
    const dx = positions[i].x - positions[i - 1].x;
    const dy = positions[i].y - positions[i - 1].y;
    const frameDiff = positions[i].frameNumber - positions[i - 1].frameNumber;
    totalVelocity += Math.sqrt(dx * dx + dy * dy) / Math.max(1, frameDiff);
  }
  const avgVelocity = positions.length > 1 ? totalVelocity / (positions.length - 1) : 0;

  const startFrame = positions[0].frameNumber;
  const endFrame = positions[positions.length - 1].frameNumber;
  const length = endFrame - startFrame;

  // Score: length * confidence (longer, more confident = better)
  const score = length * avgConfidence;

  return {
    startFrame,
    endFrame,
    positions,
    avgConfidence,
    avgVelocity,
    score,
  };
}

/**
 * Build a smooth camera path from reliable segments.
 * Uses best segments as anchors, interpolates gaps smoothly.
 */
function buildCameraPathFromSegments(
  segments: TrackingSegment[],
  totalFrames: number,
  marginX: number,
  marginY: number
): Array<{ x: number; y: number }> {
  if (segments.length === 0 || totalFrames === 0) {
    // No reliable segments - default to center
    return Array(totalFrames).fill({ x: 0.5, y: 0.5 });
  }

  // Build frame -> position map from all segment positions
  const frameMap = new Map<number, { x: number; y: number }>();
  for (const segment of segments) {
    for (const pos of segment.positions) {
      // Clamp to margins
      const x = Math.max(marginX, Math.min(1 - marginX, pos.x));
      const y = Math.max(marginY, Math.min(1 - marginY, pos.y));
      frameMap.set(pos.frameNumber, { x, y });
    }
  }

  // Log frame map stats
  const frameNums = Array.from(frameMap.keys()).sort((a, b) => a - b);
  console.log(`[BALL_KEYFRAMES] Building camera path: totalFrames=${totalFrames}, frameMap has ${frameMap.size} entries`);
  if (frameNums.length > 0) {
    console.log(`  Frame range: ${frameNums[0]}-${frameNums[frameNums.length - 1]}`);
    // Sample position
    const samplePos = frameMap.get(frameNums[Math.floor(frameNums.length / 2)]);
    console.log(`  Sample position at frame ${frameNums[Math.floor(frameNums.length / 2)]}: (${samplePos?.x?.toFixed(3)}, ${samplePos?.y?.toFixed(3)})`);
  }

  // Build path with linear interpolation for gaps
  const path: Array<{ x: number; y: number }> = [];

  // Find first and last known positions for edge handling
  let firstKnownFrame = -1;
  let firstKnownPos: { x: number; y: number } | null = null;
  let lastKnownFrame = -1;
  let lastKnownPos: { x: number; y: number } | null = null;

  for (let f = 0; f < totalFrames; f++) {
    const pos = frameMap.get(f);
    if (pos) {
      if (firstKnownFrame < 0) {
        firstKnownFrame = f;
        firstKnownPos = pos;
      }
      lastKnownFrame = f;
      lastKnownPos = pos;
    }
  }

  // If no positions at all, return center
  if (!firstKnownPos || !lastKnownPos) {
    return Array(totalFrames).fill({ x: 0.5, y: 0.5 });
  }

  // Track interpolation state
  let prevKnown: { x: number; y: number; frame: number } = { ...firstKnownPos, frame: firstKnownFrame };
  let nextKnown: { x: number; y: number; frame: number } | null = null;

  for (let frame = 0; frame < totalFrames; frame++) {
    const pos = frameMap.get(frame);

    if (pos) {
      // Have position for this frame
      path.push(pos);
      prevKnown = { ...pos, frame };
      nextKnown = null; // Reset lookahead
    } else if (frame < firstKnownFrame) {
      // Before first detection - use first known position
      path.push({ x: firstKnownPos.x, y: firstKnownPos.y });
    } else if (frame > lastKnownFrame) {
      // After last detection - use last known position
      path.push({ x: lastKnownPos.x, y: lastKnownPos.y });
    } else {
      // Gap between detections - interpolate
      // Find next known position if not already found
      if (!nextKnown || nextKnown.frame <= frame) {
        nextKnown = null;
        for (let f = frame + 1; f < totalFrames; f++) {
          const np = frameMap.get(f);
          if (np) {
            nextKnown = { ...np, frame: f };
            break;
          }
        }
      }

      if (nextKnown && prevKnown.frame < frame) {
        // Interpolate between prev and next
        const range = nextKnown.frame - prevKnown.frame;
        const t = range > 0 ? (frame - prevKnown.frame) / range : 0;
        path.push({
          x: prevKnown.x + (nextKnown.x - prevKnown.x) * t,
          y: prevKnown.y + (nextKnown.y - prevKnown.y) * t,
        });
      } else {
        // Fallback - hold previous position
        path.push({ x: prevKnown.x, y: prevKnown.y });
      }
    }
  }

  return path;
}

/**
 * Build a smooth camera path from individual ball positions (simpler approach).
 * Uses all positions with interpolation for gaps.
 */
function buildCameraPathFromPositions(
  positions: BallPosition[],
  totalFrames: number,
  marginX: number,
  marginY: number
): Array<{ x: number; y: number }> {
  if (positions.length === 0 || totalFrames === 0) {
    return Array(totalFrames).fill({ x: 0.5, y: 0.5 });
  }

  // Build frame -> position map
  const frameMap = new Map<number, { x: number; y: number }>();
  for (const pos of positions) {
    // Clamp to margins
    const x = Math.max(marginX, Math.min(1 - marginX, pos.x));
    const y = Math.max(marginY, Math.min(1 - marginY, pos.y));
    // Keep highest confidence per frame
    if (!frameMap.has(pos.frameNumber)) {
      frameMap.set(pos.frameNumber, { x, y });
    }
  }

  // Log frame map stats
  const frameNums = Array.from(frameMap.keys()).sort((a, b) => a - b);
  console.log(`[BALL_KEYFRAMES] Building camera path: totalFrames=${totalFrames}, positions=${frameMap.size}`);
  if (frameNums.length > 0) {
    console.log(`  Frame range: ${frameNums[0]}-${frameNums[frameNums.length - 1]}`);
  }

  // Build path with interpolation
  const path: Array<{ x: number; y: number }> = [];

  // Find first and last known positions
  let firstKnownFrame = -1;
  let firstKnownPos: { x: number; y: number } | null = null;
  let lastKnownFrame = -1;
  let lastKnownPos: { x: number; y: number } | null = null;

  for (let f = 0; f < totalFrames; f++) {
    const pos = frameMap.get(f);
    if (pos) {
      if (firstKnownFrame < 0) {
        firstKnownFrame = f;
        firstKnownPos = pos;
      }
      lastKnownFrame = f;
      lastKnownPos = pos;
    }
  }

  if (!firstKnownPos || !lastKnownPos) {
    return Array(totalFrames).fill({ x: 0.5, y: 0.5 });
  }

  // Build path with linear interpolation
  let prevKnown: { x: number; y: number; frame: number } = { ...firstKnownPos, frame: firstKnownFrame };
  let nextKnown: { x: number; y: number; frame: number } | null = null;

  for (let frame = 0; frame < totalFrames; frame++) {
    const pos = frameMap.get(frame);

    if (pos) {
      path.push(pos);
      prevKnown = { ...pos, frame };
      nextKnown = null;
    } else if (frame < firstKnownFrame) {
      path.push({ x: firstKnownPos.x, y: firstKnownPos.y });
    } else if (frame > lastKnownFrame) {
      path.push({ x: lastKnownPos.x, y: lastKnownPos.y });
    } else {
      // Find next known position
      if (!nextKnown || nextKnown.frame <= frame) {
        nextKnown = null;
        for (let f = frame + 1; f < totalFrames; f++) {
          const np = frameMap.get(f);
          if (np) {
            nextKnown = { ...np, frame: f };
            break;
          }
        }
      }

      if (nextKnown && prevKnown.frame < frame) {
        const range = nextKnown.frame - prevKnown.frame;
        const t = range > 0 ? (frame - prevKnown.frame) / range : 0;
        path.push({
          x: prevKnown.x + (nextKnown.x - prevKnown.x) * t,
          y: prevKnown.y + (nextKnown.y - prevKnown.y) * t,
        });
      } else {
        path.push({ x: prevKnown.x, y: prevKnown.y });
      }
    }
  }

  return path;
}

// ============================================================================
// Simple Moving Average Smoothing
// (Matches reference implementation from fast-volleyball-tracking-inference)
// ============================================================================

/**
 * Apply moving average smoothing to an array of values.
 * Uses edge padding to maintain array length.
 * NOTE: This creates inherent delay (half window size). Use exponentialSmoothing for responsive tracking.
 */
function movingAverage(values: number[], windowSize: number): number[] {
  if (values.length === 0) return [];
  if (values.length === 1) return [...values];

  const window = Math.min(windowSize, values.length);
  const halfWindow = Math.floor(window / 2);

  // Pad with edge values
  const padded: number[] = [];
  for (let i = 0; i < halfWindow; i++) {
    padded.push(values[0]);
  }
  padded.push(...values);
  for (let i = 0; i < halfWindow; i++) {
    padded.push(values[values.length - 1]);
  }

  // Compute moving average
  const result: number[] = [];
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    for (let j = 0; j < window; j++) {
      sum += padded[i + j];
    }
    result.push(sum / window);
  }

  return result;
}

/**
 * Apply exponential smoothing to an array of values.
 * Unlike moving average, this has NO inherent delay - responds immediately to changes.
 *
 * @param values - Array of values to smooth
 * @param alpha - Smoothing factor (0-1): higher = more responsive, lower = smoother
 */
function exponentialSmoothing(values: number[], alpha: number): number[] {
  if (values.length === 0) return [];
  if (values.length === 1) return [...values];

  const result = [values[0]];
  for (let i = 1; i < values.length; i++) {
    // New value = alpha * current + (1-alpha) * previous smoothed
    result.push(alpha * values[i] + (1 - alpha) * result[i - 1]);
  }
  return result;
}

/**
 * Apply lookahead to smoothed positions to anticipate ball movement.
 * Blends current position with future position for camera anticipation.
 *
 * @param smoothedX - Smoothed X positions
 * @param smoothedY - Smoothed Y positions
 * @param lookaheadFrames - Number of frames to look ahead
 * @param lookaheadWeight - Blend weight (0 = current only, 1 = future only)
 */
function applyLookahead(
  smoothedX: number[],
  smoothedY: number[],
  lookaheadFrames: number,
  lookaheadWeight: number
): { x: number[]; y: number[] } {
  const resultX: number[] = [];
  const resultY: number[] = [];

  for (let i = 0; i < smoothedX.length; i++) {
    const futureIdx = Math.min(i + lookaheadFrames, smoothedX.length - 1);
    // Blend: (1-weight) * current + weight * future position
    resultX.push(smoothedX[i] * (1 - lookaheadWeight) + smoothedX[futureIdx] * lookaheadWeight);
    resultY.push(smoothedY[i] * (1 - lookaheadWeight) + smoothedY[futureIdx] * lookaheadWeight);
  }

  return { x: resultX, y: resultY };
}

/**
 * Detect smart keyframe positions based on trajectory direction changes.
 * Places keyframes at significant direction changes while respecting min/max spacing.
 *
 * @param xValues - Smoothed X positions (one per frame)
 * @param yValues - Smoothed Y positions (one per frame)
 * @param fps - Frames per second
 * @param config - Configuration with thresholds
 * @returns Array of frame indices where keyframes should be placed
 */
function detectSmartKeyframeIndices(
  xValues: number[],
  yValues: number[],
  fps: number,
  config: BallToKeyframesConfig
): number[] {
  if (xValues.length < 2) return [0];

  const minSpacingFrames = Math.round(config.minKeyframeSpacing * fps);
  const maxSpacingFrames = Math.round(config.maxKeyframeSpacing * fps);
  const directionThresholdRad = (config.directionChangeThreshold * Math.PI) / 180;

  const keyframeIndices: number[] = [0]; // Always start with first frame
  let lastKeyframeIdx = 0;
  let lastDirection: number | null = null;

  // Calculate direction at each point using a small window for stability
  const windowSize = Math.max(3, Math.round(fps * 0.1)); // 0.1s window

  for (let i = windowSize; i < xValues.length - windowSize; i++) {
    const framesSinceLastKeyframe = i - lastKeyframeIdx;

    // Skip if too close to last keyframe
    if (framesSinceLastKeyframe < minSpacingFrames) continue;

    // Calculate current direction using window average
    const dx = xValues[i + windowSize] - xValues[i - windowSize];
    const dy = yValues[i + windowSize] - yValues[i - windowSize];
    const currentDirection = Math.atan2(dy, dx);

    // Check position change from last keyframe
    const positionChange = Math.sqrt(
      (xValues[i] - xValues[lastKeyframeIdx]) ** 2 +
      (yValues[i] - yValues[lastKeyframeIdx]) ** 2
    );

    // Determine if we should place a keyframe
    let shouldAddKeyframe = false;

    // Force keyframe if max spacing exceeded
    if (framesSinceLastKeyframe >= maxSpacingFrames) {
      shouldAddKeyframe = true;
    }
    // Add keyframe on significant direction change
    else if (lastDirection !== null) {
      let angleDiff = Math.abs(currentDirection - lastDirection);
      // Normalize to 0-π range
      if (angleDiff > Math.PI) angleDiff = 2 * Math.PI - angleDiff;

      if (angleDiff >= directionThresholdRad) {
        shouldAddKeyframe = true;
      }
    }
    // Add keyframe on significant position change
    if (positionChange >= config.positionChangeThreshold) {
      shouldAddKeyframe = true;
    }

    if (shouldAddKeyframe) {
      keyframeIndices.push(i);
      lastKeyframeIdx = i;
      lastDirection = currentDirection;
    } else if (lastDirection === null) {
      // Initialize direction after first few frames
      lastDirection = currentDirection;
    }
  }

  // Always include last frame
  const lastIdx = xValues.length - 1;
  if (lastIdx > lastKeyframeIdx + minSpacingFrames / 2) {
    keyframeIndices.push(lastIdx);
  }

  return keyframeIndices;
}

/**
 * Build per-frame position array with forward-fill for missing frames.
 * Returns positions for every frame from 0 to maxFrame.
 *
 * @param syncOffsetSeconds - Positive value delays tracking (shifts frame numbers forward)
 *                            to compensate for ML model being ahead of actual video
 * @param fps - Frames per second, used to convert syncOffsetSeconds to frames
 */
function buildFramePositions(
  positions: BallPosition[],
  maxFrame: number,
  minConfidence: number,
  syncOffsetSeconds: number = 0,
  fps: number = 30
): Array<{ x: number; y: number } | null> {
  // Build frame -> position map from confident detections
  // Apply sync offset: shift frame numbers forward to delay tracking
  const syncOffsetFrames = Math.round(syncOffsetSeconds * fps);
  const frameMap = new Map<number, { x: number; y: number }>();
  for (const pos of positions) {
    if (pos.confidence >= minConfidence) {
      // Shift frame number by offset (delays tracking to match video)
      const adjustedFrame = pos.frameNumber + syncOffsetFrames;
      if (adjustedFrame >= 0 && adjustedFrame <= maxFrame) {
        // Keep highest confidence per frame
        const existing = frameMap.get(adjustedFrame);
        if (!existing) {
          frameMap.set(adjustedFrame, { x: pos.x, y: pos.y });
        }
      }
    }
  }

  // Build array with forward-fill for gaps
  const result: Array<{ x: number; y: number } | null> = [];
  let lastKnown: { x: number; y: number } | null = null;

  for (let frame = 0; frame <= maxFrame; frame++) {
    const pos = frameMap.get(frame);
    if (pos) {
      lastKnown = pos;
      result.push(pos);
    } else if (lastKnown) {
      // Forward-fill with last known position
      result.push(lastKnown);
    } else {
      result.push(null);
    }
  }

  // Backward-fill initial nulls if we have any positions
  if (lastKnown) {
    const firstKnownIdx = result.findIndex((p) => p !== null);
    if (firstKnownIdx > 0) {
      const firstKnown = result[firstKnownIdx];
      for (let i = 0; i < firstKnownIdx; i++) {
        result[i] = firstKnown;
      }
    }
  }

  return result;
}

/**
 * Clamp camera position to keep ball within margins.
 */
function clampPosition(value: number, margin: number): number {
  return Math.max(margin, Math.min(1 - margin, value));
}

// ============================================================================
// Anchor-Based Keyframe Detection
// ============================================================================

interface AnchorFrame {
  frameNumber: number;
  timeOffset: number; // 0-1 within rally
  x: number;
  y: number;
  confidence: number;
  continuityScore: number; // How many consecutive detections around this frame
}

/**
 * Find anchor frames - frames with high confidence AND consistent tracking.
 * These are reliable points where we know the ball position.
 */
function findAnchorFrames(
  positions: BallPosition[],
  totalFrames: number,
  minConfidence: number,
  maxVelocity: number,
  marginX: number = 0.15,
  marginY: number = 0.15,
  yOffsetCorrection: number = 0,
  xOffsetCorrection: number = 0
): AnchorFrame[] {
  if (positions.length === 0 || totalFrames === 0) return [];

  // Sort by frame
  const sorted = [...positions].sort((a, b) => a.frameNumber - b.frameNumber);

  // Filter out edge positions - ball shouldn't be at extreme edges (likely false positives)
  // Use margin values as threshold - positions that would be clamped are suspicious
  // Adding small buffer (0.02) to avoid accepting positions right at the edge
  const edgeX = marginX + 0.02;
  const edgeY = marginY + 0.02;
  const isValidPosition = (p: BallPosition): boolean => {
    // Reject positions too close to edges (would be clamped anyway)
    if (p.x < edgeX || p.x > (1 - edgeX)) return false;
    if (p.y < edgeY || p.y > (1 - edgeY)) return false;
    return true;
  };

  // Build frame -> position map for high confidence detections with valid positions
  const frameMap = new Map<number, BallPosition>();
  let edgeFiltered = 0;
  let confFiltered = 0;
  for (const p of sorted) {
    if (p.confidence < minConfidence) {
      confFiltered++;
      continue;
    }
    if (!isValidPosition(p)) {
      edgeFiltered++;
      continue;
    }
    if (!frameMap.has(p.frameNumber)) {
      frameMap.set(p.frameNumber, p);
    }
  }

  console.log(`[BALL_KEYFRAMES] Anchor filtering: ${sorted.length} total, ${confFiltered} low-conf, ${edgeFiltered} edge positions, ${frameMap.size} valid`);

  // Calculate continuity score for each frame and average position over window
  const anchors: AnchorFrame[] = [];

  for (const [frame, pos] of frameMap) {
    // Collect positions in continuity window (before and after)
    const windowPositions: Array<{ x: number; y: number }> = [{ x: pos.x, y: pos.y }];
    let continuityBefore = 0;
    let continuityAfter = 0;

    // Check backwards
    let lastPos = pos;
    for (let f = frame - 1; f >= 0 && continuityBefore < 15; f--) {
      const prev = frameMap.get(f);
      if (!prev) break;
      // Check velocity constraint
      const dx = lastPos.x - prev.x;
      const dy = lastPos.y - prev.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxVelocity * (lastPos === pos ? frame - f : 1)) break;
      windowPositions.push({ x: prev.x, y: prev.y });
      lastPos = prev;
      continuityBefore++;
    }

    // Check forwards
    lastPos = pos;
    for (let f = frame + 1; f < totalFrames && continuityAfter < 15; f++) {
      const next = frameMap.get(f);
      if (!next) break;
      // Check velocity constraint
      const dx = next.x - lastPos.x;
      const dy = next.y - lastPos.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist > maxVelocity * (lastPos === pos ? f - frame : 1)) break;
      windowPositions.push({ x: next.x, y: next.y });
      lastPos = next;
      continuityAfter++;
    }

    const continuityScore = continuityBefore + continuityAfter + 1;

    // Only consider frames with good continuity (not isolated detections)
    // Increased threshold from 3 to 5 for smoother tracking
    if (continuityScore >= 5) {
      // Average position over the continuity window for smoother camera
      // Apply offset corrections (compensate for ML detection being off-center)
      const avgX = windowPositions.reduce((sum, p) => sum + p.x, 0) / windowPositions.length + xOffsetCorrection;
      const avgY = windowPositions.reduce((sum, p) => sum + p.y, 0) / windowPositions.length + yOffsetCorrection;

      anchors.push({
        frameNumber: frame,
        timeOffset: frame / Math.max(1, totalFrames - 1),
        x: Math.min(1, Math.max(0, avgX)), // Clamp to valid range
        y: Math.min(1, Math.max(0, avgY)),
        confidence: pos.confidence,
        continuityScore,
      });
    }
  }

  // Sort by continuity score * confidence (best anchors first)
  anchors.sort((a, b) => (b.continuityScore * b.confidence) - (a.continuityScore * a.confidence));

  return anchors;
}

/**
 * Select keyframe positions from anchor frames.
 * Ensures good coverage with minimum/maximum spacing constraints.
 */
function selectKeyframeAnchors(
  anchors: AnchorFrame[],
  totalFrames: number,
  fps: number,
  minSpacingSeconds: number,
  maxSpacingSeconds: number
): AnchorFrame[] {
  if (anchors.length === 0) return [];

  const minSpacingFrames = Math.round(minSpacingSeconds * fps);
  const maxSpacingFrames = Math.round(maxSpacingSeconds * fps);

  // Sort anchors by frame number for processing
  const sortedByFrame = [...anchors].sort((a, b) => a.frameNumber - b.frameNumber);

  // Greedily select anchors with spacing constraints
  const selected: AnchorFrame[] = [];

  // Always start with an anchor near the beginning
  const firstThird = sortedByFrame.filter(a => a.frameNumber < totalFrames / 3);
  const bestFirst = firstThird.length > 0
    ? firstThird.reduce((best, a) => (a.continuityScore * a.confidence) > (best.continuityScore * best.confidence) ? a : best)
    : sortedByFrame[0];

  if (bestFirst) {
    selected.push(bestFirst);
  }

  // Add anchors ensuring spacing constraints
  for (const anchor of anchors) {
    // Skip if too close to any selected anchor
    const tooClose = selected.some(s =>
      Math.abs(anchor.frameNumber - s.frameNumber) < minSpacingFrames
    );
    if (tooClose) continue;

    selected.push(anchor);
  }

  // Sort by frame for final output
  selected.sort((a, b) => a.frameNumber - b.frameNumber);

  // Fill gaps where spacing exceeds maximum
  const filled: AnchorFrame[] = [];
  for (let i = 0; i < selected.length; i++) {
    filled.push(selected[i]);

    if (i < selected.length - 1) {
      const gap = selected[i + 1].frameNumber - selected[i].frameNumber;
      if (gap > maxSpacingFrames) {
        // Find best anchor in the gap
        const gapAnchors = anchors.filter(a =>
          a.frameNumber > selected[i].frameNumber + minSpacingFrames &&
          a.frameNumber < selected[i + 1].frameNumber - minSpacingFrames
        );
        if (gapAnchors.length > 0) {
          const best = gapAnchors.reduce((b, a) =>
            (a.continuityScore * a.confidence) > (b.continuityScore * b.confidence) ? a : b
          );
          filled.push(best);
        }
      }
    }
  }

  // Re-sort after filling
  filled.sort((a, b) => a.frameNumber - b.frameNumber);

  // Ensure we have start and end keyframes
  const result: AnchorFrame[] = [];

  // Add start keyframe if needed
  if (filled.length === 0 || filled[0].frameNumber > minSpacingFrames) {
    const nearStart = anchors.filter(a => a.frameNumber < maxSpacingFrames);
    if (nearStart.length > 0) {
      const best = nearStart.reduce((b, a) =>
        (a.continuityScore * a.confidence) > (b.continuityScore * b.confidence) ? a : b
      );
      result.push(best);
    }
  }

  result.push(...filled);

  // Add end keyframe if needed
  const lastFrame = result.length > 0 ? result[result.length - 1].frameNumber : 0;
  if (lastFrame < totalFrames - maxSpacingFrames) {
    const nearEnd = anchors.filter(a =>
      a.frameNumber > totalFrames - maxSpacingFrames &&
      a.frameNumber > lastFrame + minSpacingFrames
    );
    if (nearEnd.length > 0) {
      const best = nearEnd.reduce((b, a) =>
        (a.continuityScore * a.confidence) > (b.continuityScore * b.confidence) ? a : b
      );
      result.push(best);
    }
  }

  return result;
}

// ============================================================================
// Trajectory Validation and Deduplication
// ============================================================================

/**
 * Validate trajectory by removing keyframes that cause excessive position jumps.
 * Ensures smooth camera movement between keyframes.
 */
function validateTrajectory(
  keyframes: CameraKeyframe[],
  maxPositionJump: number
): CameraKeyframe[] {
  if (keyframes.length <= 2) return keyframes;

  // Always keep first keyframe
  const result: CameraKeyframe[] = [keyframes[0]];

  for (let i = 1; i < keyframes.length; i++) {
    const prev = result[result.length - 1];
    const curr = keyframes[i];

    // Calculate position jump from previous accepted keyframe
    const dx = curr.positionX - prev.positionX;
    const dy = curr.positionY - prev.positionY;
    const jump = Math.sqrt(dx * dx + dy * dy);

    if (jump <= maxPositionJump) {
      // Accept this keyframe
      result.push(curr);
    } else {
      // Skip this keyframe - creates too large a jump
      // If it's the last keyframe, adjust its position to be smoother
      const isLast = i === keyframes.length - 1;
      if (isLast) {
        // For last keyframe, clamp the position to max jump from previous
        const angle = Math.atan2(dy, dx);
        const clampedX = prev.positionX + Math.cos(angle) * maxPositionJump;
        const clampedY = prev.positionY + Math.sin(angle) * maxPositionJump;
        result.push({
          ...curr,
          positionX: clampedX,
          positionY: clampedY,
        });
      }
      // Non-last keyframes with excessive jumps are simply skipped
    }
  }

  console.log(`[BALL_KEYFRAMES] Trajectory validation: ${keyframes.length} -> ${result.length} keyframes (maxJump=${maxPositionJump})`);

  return result;
}

/**
 * Remove duplicate keyframes at the same timeOffset.
 * Keeps the keyframe with better trajectory continuity.
 */
function deduplicateKeyframes(keyframes: CameraKeyframe[]): CameraKeyframe[] {
  if (keyframes.length <= 1) return keyframes;

  // Sort by timeOffset first
  const sorted = [...keyframes].sort((a, b) => a.timeOffset - b.timeOffset);

  const result: CameraKeyframe[] = [];
  const timeOffsetMap = new Map<string, CameraKeyframe>();

  for (const kf of sorted) {
    // Round to avoid floating point issues (to 4 decimal places)
    const timeKey = kf.timeOffset.toFixed(4);

    if (!timeOffsetMap.has(timeKey)) {
      timeOffsetMap.set(timeKey, kf);
    }
    // If duplicate, keep the existing one (first encountered)
  }

  // Convert map back to sorted array
  for (const [_, kf] of Array.from(timeOffsetMap.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
    result.push(kf);
  }

  if (result.length < keyframes.length) {
    console.log(`[BALL_KEYFRAMES] Deduplication: ${keyframes.length} -> ${result.length} keyframes`);
  }

  return result;
}

/**
 * Enforce maximum keyframe limit by selecting the most important keyframes.
 * Preserves first and last, then selects evenly distributed middle keyframes.
 */
function enforceMaxKeyframes(
  keyframes: CameraKeyframe[],
  maxKeyframes: number
): CameraKeyframe[] {
  if (keyframes.length <= maxKeyframes) return keyframes;

  // Always keep first and last
  const first = keyframes[0];
  const last = keyframes[keyframes.length - 1];

  // For middle keyframes, select evenly distributed ones
  const middle = keyframes.slice(1, -1);
  const middleCount = maxKeyframes - 2;

  if (middleCount <= 0) {
    return [first, last];
  }

  // Select evenly spaced keyframes from middle
  const step = middle.length / middleCount;
  const selectedMiddle: CameraKeyframe[] = [];

  for (let i = 0; i < middleCount; i++) {
    const idx = Math.round(i * step);
    if (idx < middle.length) {
      selectedMiddle.push(middle[idx]);
    }
  }

  console.log(`[BALL_KEYFRAMES] Enforcing max keyframes: ${keyframes.length} -> ${2 + selectedMiddle.length} (max=${maxKeyframes})`);

  return [first, ...selectedMiddle, last];
}

// ============================================================================
// Main Entry Point
// ============================================================================

/**
 * Convert ball tracking positions to camera keyframes.
 *
 * Uses simple moving average smoothing (matching fast-volleyball-tracking-inference)
 * for stable, predictable camera tracking.
 */
export function convertBallTrackingToKeyframes(
  positions: BallPosition[],
  rallyDurationMs: number,
  fps: number,
  aspectRatio: AspectRatio = 'ORIGINAL',
  customConfig?: Partial<BallToKeyframesConfig>
): BallToKeyframesResult {
  // Merge configs based on aspect ratio
  let config = { ...DEFAULT_CONFIG };
  if (aspectRatio === 'VERTICAL') {
    config = { ...config, ...VERTICAL_CONFIG };
  } else if (aspectRatio === 'ORIGINAL') {
    config = { ...config, ...ORIGINAL_CONFIG };
  }
  if (customConfig) {
    config = { ...config, ...customConfig };
  }

  console.log(`[BALL_KEYFRAMES] Using config for ${aspectRatio}: minSpacing=${config.minKeyframeSpacing}s, maxSpacing=${config.maxKeyframeSpacing}s, maxKeyframes=${config.maxKeyframes}, lockVertical=${config.lockVertical}`);

  const totalFrames = Math.round((rallyDurationMs / 1000) * fps);

  // Calculate quality metrics
  const validPositions = positions.filter((p) => p.confidence >= config.minConfidence);
  const uniqueFrames = new Set(validPositions.map((p) => p.frameNumber));
  const coverage = uniqueFrames.size / Math.max(1, totalFrames);
  const avgConfidence =
    validPositions.length > 0
      ? validPositions.reduce((sum, p) => sum + p.confidence, 0) / validPositions.length
      : 0;

  const quality: TrackingQuality = {
    coverage: Math.min(1, coverage), // Cap at 1
    averageConfidence: avgConfidence,
    isUsable: coverage >= config.minCoverage,
    recommendation: getQualityRecommendation(coverage, avgConfidence),
  };

  if (!quality.isUsable || validPositions.length === 0) {
    return { keyframes: [], quality };
  }

  // For ORIGINAL aspect ratio, auto-apply zoom to enable ball-following camera
  // Without zoom, camera can't pan (full frame shown). Apply 1.3x zoom to enable tracking.
  if (aspectRatio === 'ORIGINAL' && config.baseZoom <= 1.0) {
    config = { ...config, baseZoom: 1.3 }; // Apply zoom to enable camera movement
  }

  // =========================================================================
  // ANCHOR-BASED APPROACH
  //
  // Places keyframes at high-confidence anchor frames where ball position is
  // reliable. Camera = ball position at anchors, CSS interpolation smooths between.
  //
  // Steps:
  // 1. Find anchor frames (high confidence + continuous tracking)
  // 2. Select best anchors with spacing constraints
  // 3. Generate keyframes directly from anchor positions
  // =========================================================================

  // Step 1: Find anchor frames
  const anchors = findAnchorFrames(
    positions,
    totalFrames,
    config.minConfidence,
    config.maxVelocityPerFrame,
    config.marginX,
    config.marginY,
    config.yOffsetCorrection,
    config.xOffsetCorrection
  );

  console.log(`[BALL_KEYFRAMES] Found ${anchors.length} anchor frames from ${positions.length} positions`);

  if (anchors.length === 0) {
    // Fallback: use any high-confidence positions
    const highConf = positions.filter(p => p.confidence >= config.minConfidence * 0.8);
    console.log(`[BALL_KEYFRAMES] No anchors found, using ${highConf.length} high-confidence positions as fallback`);

    if (highConf.length === 0) {
      return {
        keyframes: [],
        quality: {
          ...quality,
          isUsable: false,
          recommendation: 'No reliable ball tracking detected.',
        },
      };
    }

    // Create simple keyframes from high-confidence detections
    // Limit to maxKeyframes from the start
    const sorted = highConf.sort((a, b) => a.frameNumber - b.frameNumber);
    const targetKeyframes = Math.min(config.maxKeyframes, 6);
    const step = Math.max(1, Math.floor(sorted.length / targetKeyframes));
    let keyframes: CameraKeyframe[] = [];

    for (let i = 0; i < sorted.length; i += step) {
      const p = sorted[i];
      keyframes.push({
        timeOffset: p.frameNumber / Math.max(1, totalFrames - 1),
        positionX: clampPosition(p.x, config.marginX),
        positionY: config.lockVertical ? 0.5 : clampPosition(p.y, config.marginY),
        zoom: config.baseZoom,
        easing: i === 0 ? 'EASE_OUT' : 'EASE_IN_OUT' as const,
      });
    }

    // Ensure last keyframe
    const last = sorted[sorted.length - 1];
    if (keyframes.length === 0 || keyframes[keyframes.length - 1].timeOffset < 0.9) {
      keyframes.push({
        timeOffset: 1,
        positionX: clampPosition(last.x, config.marginX),
        positionY: config.lockVertical ? 0.5 : clampPosition(last.y, config.marginY),
        zoom: config.baseZoom,
        easing: 'EASE_IN_OUT' as const,
      });
    }

    // Sort, deduplicate, validate, and enforce limits
    keyframes.sort((a, b) => a.timeOffset - b.timeOffset);
    keyframes = deduplicateKeyframes(keyframes);
    keyframes = validateTrajectory(keyframes, config.maxPositionJump);
    keyframes = enforceMaxKeyframes(keyframes, config.maxKeyframes);

    return { keyframes, quality };
  }

  // Log best anchors
  anchors.slice(0, 5).forEach((a, i) => {
    console.log(`  Anchor ${i}: frame=${a.frameNumber} pos=(${a.x.toFixed(3)}, ${a.y.toFixed(3)}) conf=${a.confidence.toFixed(2)} cont=${a.continuityScore}`);
  });

  // Step 2: Select keyframe anchors with spacing constraints
  const selectedAnchors = selectKeyframeAnchors(
    anchors,
    totalFrames,
    fps,
    config.minKeyframeSpacing,
    config.maxKeyframeSpacing
  );

  console.log(`[BALL_KEYFRAMES] Selected ${selectedAnchors.length} anchors for keyframes`);

  if (selectedAnchors.length === 0) {
    return { keyframes: [], quality };
  }

  // Step 3: Generate keyframes directly from anchor positions
  // Camera position = ball position (with margin clamping)
  // Apply lockVertical if configured (for ORIGINAL mode)
  let keyframes: CameraKeyframe[] = selectedAnchors.map((anchor, i) => {
    const isFirst = i === 0;
    const isLast = i === selectedAnchors.length - 1;

    return {
      timeOffset: isLast ? 1 : Math.min(anchor.timeOffset, 1),
      positionX: clampPosition(anchor.x, config.marginX),
      positionY: config.lockVertical ? 0.5 : clampPosition(anchor.y, config.marginY),
      zoom: config.baseZoom,
      easing: isFirst ? 'EASE_OUT' : 'EASE_IN_OUT' as const,
    };
  });

  // Ensure we have a start keyframe at t=0 if first anchor is late
  if (keyframes.length > 0 && keyframes[0].timeOffset > 0.1) {
    // Use first anchor position for start
    keyframes.unshift({
      timeOffset: 0,
      positionX: keyframes[0].positionX,
      positionY: keyframes[0].positionY,
      zoom: config.baseZoom,
      easing: 'EASE_OUT' as const,
    });
  }

  // Ensure we have an end keyframe at t=1 if last anchor is early
  if (keyframes.length > 0 && keyframes[keyframes.length - 1].timeOffset < 0.9) {
    keyframes.push({
      timeOffset: 1,
      positionX: keyframes[keyframes.length - 1].positionX,
      positionY: keyframes[keyframes.length - 1].positionY,
      zoom: config.baseZoom,
      easing: 'EASE_IN_OUT' as const,
    });
  }

  // CRITICAL: Sort keyframes by timeOffset before further processing
  keyframes.sort((a, b) => a.timeOffset - b.timeOffset);

  // Step 4: Deduplicate keyframes (remove duplicates at same timeOffset)
  keyframes = deduplicateKeyframes(keyframes);

  // Step 5: Validate trajectory (remove keyframes with excessive position jumps)
  keyframes = validateTrajectory(keyframes, config.maxPositionJump);

  // Step 6: Enforce maximum keyframe limit
  keyframes = enforceMaxKeyframes(keyframes, config.maxKeyframes);

  // Final sort to ensure correct order
  keyframes.sort((a, b) => a.timeOffset - b.timeOffset);

  // Log keyframes for debugging
  console.log(`[BALL_KEYFRAMES] Generated ${keyframes.length} keyframes:`);
  keyframes.forEach((kf, i) => {
    console.log(`  [${i}] t=${kf.timeOffset.toFixed(3)} pos=(${kf.positionX.toFixed(3)}, ${kf.positionY.toFixed(3)}) zoom=${kf.zoom}`);
  });

  return { keyframes, quality };
}

function getQualityRecommendation(coverage: number, confidence: number): string {
  if (coverage < 0.15) {
    return 'Insufficient ball detections. Ball may be out of frame or occluded.';
  }
  if (confidence < 0.4) {
    return 'Low detection confidence. Consider manual keyframes.';
  }
  if (coverage > 0.6 && confidence > 0.6) {
    return 'Good tracking quality.';
  }
  return 'Acceptable tracking quality.';
}

// Legacy exports for compatibility
export interface BallToKeyframesConfig_Legacy {
  temporalFilter: { minConfidence: number; maxGapFrames: number; outlierVelocityThreshold: number };
  smoothing: { useKalman: boolean; processNoise: number; measurementNoise: number; emaAlpha: number };
  cameraBehavior: {
    deadZoneX: number;
    deadZoneY: number;
    lagFactor: number;
    marginX: number;
    marginY: number;
    lookaheadFrames: number;
    lookaheadWeight: number;
    verticalTrackingStrength: number;
  };
  decimation: {
    minIntervalSeconds: number;
    maxIntervalSeconds: number;
    positionThreshold: number;
    directionChangeThreshold: number;
    douglasPeuckerEpsilon: number;
  };
  zoom: {
    baseZoom: number;
    minZoom: number;
    maxZoom: number;
    zoomOnLowConfidence: boolean;
    confidenceThreshold: number;
    adaptiveZoom: boolean;
  };
  minTrackingCoverage: number;
  minConfidenceAverage: number;
}
