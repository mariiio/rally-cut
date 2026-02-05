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
  // Margin to keep ball visible (0-0.5)
  marginX: number;
  marginY: number;
  // Base zoom level
  baseZoom: number;
  // Minimum coverage to consider tracking usable
  minCoverage: number;
  // Keyframe spacing settings
  minKeyframeSpacing: number; // Minimum seconds between keyframes
  maxKeyframeSpacing: number; // Maximum seconds between keyframes (fill gaps)
  // Physics-based filtering: max velocity per frame (normalized 0-1)
  // Ball can't teleport - reject detections exceeding this velocity
  maxVelocityPerFrame: number;
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

export const DEFAULT_CONFIG: BallToKeyframesConfig = {
  minConfidence: 0.3,
  marginX: 0.15,
  marginY: 0.12,
  baseZoom: 1.0,
  minCoverage: 0.15,
  minKeyframeSpacing: 0.5, // Min 0.5s between keyframes
  maxKeyframeSpacing: 3.0, // Force keyframe if none for 3s
  maxVelocityPerFrame: 0.15, // ~30 m/s normalized - ball can't exceed this
  yOffsetCorrection: 0, // No correction by default
  xOffsetCorrection: 0,
  maxKeyframes: 10, // Default max 10 keyframes
  maxPositionJump: 0.35, // Max 35% position change between keyframes
  lockVertical: false, // Allow Y tracking by default
};

// Original (16:9) - horizontal tracking only, minimal keyframes
// For wide aspect ratio, vertical camera movement looks jerky.
// Lock Y to center and only pan horizontally with zoom.
export const ORIGINAL_CONFIG: Partial<BallToKeyframesConfig> = {
  minConfidence: 0.6,
  marginX: 0.12, // Horizontal margins for panning
  marginY: 0.0, // No Y margin (locked to center)
  baseZoom: 1.3, // Slight zoom to enable panning
  minKeyframeSpacing: 3.0, // 3s minimum between keyframes
  maxKeyframeSpacing: 5.0, // 5s max gap
  maxVelocityPerFrame: 0.1,
  yOffsetCorrection: 0,
  xOffsetCorrection: 0,
  maxKeyframes: 5, // Max 5 keyframes for 16:9
  maxPositionJump: 0.30, // 30% max jump between keyframes
  lockVertical: true, // Lock Y to 0.5 (center) - no vertical movement
};

// Vertical (9:16) - crop to portrait, anchor-based camera tracking
// Places keyframes at high-confidence anchor frames where ball position is reliable.
// Camera follows ball position directly at anchors, CSS interpolation smooths between.
export const VERTICAL_CONFIG: Partial<BallToKeyframesConfig> = {
  minConfidence: 0.65, // Higher threshold for reliable anchors
  marginX: 0.12, // Keep ball within margins
  marginY: 0.12,
  baseZoom: 1.25,
  minKeyframeSpacing: 2.5, // 2.5s minimum between keyframes
  maxKeyframeSpacing: 4.0, // 4s max gap
  maxVelocityPerFrame: 0.08, // Tighter - ball can't teleport
  yOffsetCorrection: 0.03, // 3% down (tracking appears above actual ball)
  xOffsetCorrection: 0.0, // no X offset needed
  maxKeyframes: 8, // Max 8 keyframes for 9:16
  maxPositionJump: 0.25, // 25% max jump between keyframes
  lockVertical: false, // Allow Y tracking for portrait mode
};

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
  for (const p of sorted) {
    if (p.confidence < minConfidence) {
      continue;
    }
    if (!isValidPosition(p)) {
      continue;
    }
    if (!frameMap.has(p.frameNumber)) {
      frameMap.set(p.frameNumber, p);
    }
  }

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
  for (const [_timeKey, kf] of Array.from(timeOffsetMap.entries()).sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]))) {
    result.push(kf);
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

  if (anchors.length === 0) {
    // Fallback: use any high-confidence positions
    const highConf = positions.filter(p => p.confidence >= config.minConfidence * 0.8);

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

  // Step 2: Select keyframe anchors with spacing constraints
  const selectedAnchors = selectKeyframeAnchors(
    anchors,
    totalFrames,
    fps,
    config.minKeyframeSpacing,
    config.maxKeyframeSpacing
  );

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

