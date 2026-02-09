'use client';

import { useMemo, useState, useEffect, useRef } from 'react';
import { Box } from '@mui/material';
import { usePlayerTrackingStore, PlayerPosition } from '@/stores/playerTrackingStore';

interface PlayerOverlayProps {
  rallyId: string;
  rallyStartTime: number;
  videoRef: React.RefObject<HTMLVideoElement | null>;
  containerRef: React.RefObject<HTMLDivElement | null>;
  fps?: number;
}

// Default colors for tracks before player assignment
const TRACK_COLORS = [
  '#FF6B6B', // Red
  '#4ECDC4', // Teal
  '#45B7D1', // Blue
  '#96CEB4', // Green
];

// Interpolated position for smooth rendering
interface InterpolatedPosition {
  x: number;
  y: number;
  w: number;
  h: number;
  courtX?: number;
  courtY?: number;
  confidence: number;
}

// Linear interpolation helper
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

export function PlayerOverlay({
  rallyId,
  rallyStartTime,
  videoRef,
  containerRef,
  fps: propFps = 30,
}: PlayerOverlayProps) {
  const [dimensions, setDimensions] = useState({ width: 0, height: 0, offsetX: 0, offsetY: 0 });
  // Use floating-point frame for sub-frame interpolation smoothness
  const [currentFrame, setCurrentFrame] = useState(0);
  const rafIdRef = useRef<number | null>(null);
  const lastTimeRef = useRef(-1);

  // Get fps from track data if available, otherwise use prop
  const { playerTracks } = usePlayerTrackingStore();
  const track = playerTracks[rallyId];
  const fps = track?.tracksJson?.fps ?? propFps;

  // Measure video dimensions and position within container
  // The video maintains aspect ratio, so it may be letterboxed/pillarboxed
  useEffect(() => {
    const video = videoRef.current;
    const container = containerRef.current;
    if (!video || !container) return;

    const updateDimensions = () => {
      // Get the actual rendered size of the video (accounts for object-fit)
      const videoRect = video.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();

      // Calculate video's actual display size based on object-fit: contain
      const videoAspect = video.videoWidth / video.videoHeight || 16/9;
      const containerAspect = containerRect.width / containerRect.height;

      let displayWidth: number;
      let displayHeight: number;

      if (containerAspect > videoAspect) {
        // Container is wider than video - pillarboxing (black bars on sides)
        displayHeight = containerRect.height;
        displayWidth = displayHeight * videoAspect;
      } else {
        // Container is taller than video - letterboxing (black bars on top/bottom)
        displayWidth = containerRect.width;
        displayHeight = displayWidth / videoAspect;
      }

      // Calculate offset (centering)
      const offsetX = (containerRect.width - displayWidth) / 2;
      const offsetY = (containerRect.height - displayHeight) / 2;

      setDimensions({
        width: displayWidth,
        height: displayHeight,
        offsetX,
        offsetY,
      });
    };

    updateDimensions();

    // Re-measure on resize and video metadata load
    const resizeObserver = new ResizeObserver(updateDimensions);
    resizeObserver.observe(container);
    video.addEventListener('loadedmetadata', updateDimensions);

    return () => {
      resizeObserver.disconnect();
      video.removeEventListener('loadedmetadata', updateDimensions);
    };
  }, [videoRef, containerRef]);

  // Animation loop for smooth frame updates (~60fps via RAF)
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateFrame = () => {
      const videoTime = video.currentTime;

      // Only update if time changed significantly (>1ms to avoid float noise)
      if (Math.abs(videoTime - lastTimeRef.current) > 0.001) {
        lastTimeRef.current = videoTime;
        // Use floating-point frame for smooth sub-frame interpolation
        const frame = (videoTime - rallyStartTime) * fps;
        setCurrentFrame(frame);
      }

      rafIdRef.current = requestAnimationFrame(updateFrame);
    };

    // Start animation loop
    rafIdRef.current = requestAnimationFrame(updateFrame);

    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [videoRef, rallyStartTime, fps]);

  const { width: videoDisplayWidth, height: videoDisplayHeight, offsetX, offsetY } = dimensions;
  const { showPlayerOverlay, selectedTrackId } = usePlayerTrackingStore();

  // Build per-track sorted positions for interpolation - only recomputes when track changes
  const trackPositions = useMemo(() => {
    if (!track?.tracksJson?.tracks) {
      return new Map<number, PlayerPosition[]>();
    }

    // TrackId -> sorted positions for binary search interpolation
    const positions = new Map<number, PlayerPosition[]>();

    for (const playerTrack of track.tracksJson.tracks) {
      // Sort positions by frame for binary search during interpolation
      const sortedPositions = [...playerTrack.positions].sort((a, b) => a.frame - b.frame);
      positions.set(playerTrack.trackId, sortedPositions);
    }

    return positions;
  }, [track]);

  // Interpolate positions for smooth rendering
  const currentPositions = useMemo(() => {
    const result: { trackId: number; position: InterpolatedPosition }[] = [];

    // Maximum distance from a detection to show overlay (~0.3s at 30fps)
    const maxDistanceFromDetection = 10;

    // For each track, find positions before/after current frame and interpolate
    for (const [trackId, positions] of trackPositions) {
      if (positions.length === 0) continue;

      const firstFrame = positions[0].frame;
      const lastFrame = positions[positions.length - 1].frame;

      // Hide if current frame is too far before first detection
      if (currentFrame < firstFrame - maxDistanceFromDetection) {
        continue; // Skip - player not yet visible
      }

      // Hide if current frame is too far after last detection
      if (currentFrame > lastFrame + maxDistanceFromDetection) {
        continue; // Skip - player no longer visible
      }

      // Binary search for the position just before or at currentFrame
      let lo = 0;
      let hi = positions.length - 1;

      // Handle edge case: before first frame (but within tolerance)
      if (currentFrame <= firstFrame) {
        const pos = positions[0];
        result.push({
          trackId,
          position: {
            x: pos.x,
            y: pos.y,
            w: pos.w,
            h: pos.h,
            courtX: pos.courtX,
            courtY: pos.courtY,
            confidence: pos.confidence,
          },
        });
        continue;
      }

      // Handle edge case: after last frame (but within tolerance)
      if (currentFrame >= lastFrame) {
        const pos = positions[positions.length - 1];
        result.push({
          trackId,
          position: {
            x: pos.x,
            y: pos.y,
            w: pos.w,
            h: pos.h,
            courtX: pos.courtX,
            courtY: pos.courtY,
            confidence: pos.confidence,
          },
        });
        continue;
      }

      // Binary search for the largest frame <= currentFrame
      while (lo < hi) {
        const mid = Math.ceil((lo + hi + 1) / 2);
        if (positions[mid].frame <= currentFrame) {
          lo = mid;
        } else {
          hi = mid - 1;
        }
      }

      const before = positions[lo];
      const after = positions[lo + 1];
      const gap = after.frame - before.frame;

      // Check for gap too large (player not detected for a while)
      // Hide overlay entirely if we're in the middle of a large gap
      const maxGap = 15; // ~0.5s at 30fps
      if (gap > maxGap) {
        const distToBefore = currentFrame - before.frame;
        const distToAfter = after.frame - currentFrame;

        // Only show if very close to one of the detections
        if (distToBefore <= maxDistanceFromDetection) {
          result.push({
            trackId,
            position: {
              x: before.x,
              y: before.y,
              w: before.w,
              h: before.h,
              courtX: before.courtX,
              courtY: before.courtY,
              confidence: before.confidence,
            },
          });
        } else if (distToAfter <= maxDistanceFromDetection) {
          result.push({
            trackId,
            position: {
              x: after.x,
              y: after.y,
              w: after.w,
              h: after.h,
              courtX: after.courtX,
              courtY: after.courtY,
              confidence: after.confidence,
            },
          });
        }
        // Otherwise skip - player not visible in this gap
        continue;
      }

      // Interpolate between before and after
      const t = (currentFrame - before.frame) / gap;

      result.push({
        trackId,
        position: {
          x: lerp(before.x, after.x, t),
          y: lerp(before.y, after.y, t),
          w: lerp(before.w, after.w, t),
          h: lerp(before.h, after.h, t),
          courtX:
            before.courtX !== undefined && after.courtX !== undefined
              ? lerp(before.courtX, after.courtX, t)
              : before.courtX ?? after.courtX,
          courtY:
            before.courtY !== undefined && after.courtY !== undefined
              ? lerp(before.courtY, after.courtY, t)
              : before.courtY ?? after.courtY,
          confidence: lerp(before.confidence, after.confidence, t),
        },
      });
    }

    return result;
  }, [trackPositions, currentFrame]);

  if (!showPlayerOverlay || !track?.tracksJson || currentPositions.length === 0 || videoDisplayWidth === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 5,
      }}
    >
      {currentPositions.map(({ trackId, position }) => {
        const color = TRACK_COLORS[(trackId - 1) % TRACK_COLORS.length];
        const isSelected = selectedTrackId === trackId;

        // Convert normalized coords to pixels (relative to video display area)
        // Add offset to account for letterboxing/pillarboxing
        const x = offsetX + (position.x - position.w / 2) * videoDisplayWidth;
        const y = offsetY + (position.y - position.h / 2) * videoDisplayHeight;
        const width = position.w * videoDisplayWidth;
        const height = position.h * videoDisplayHeight;

        return (
          <Box
            key={trackId}
            sx={{
              position: 'absolute',
              left: 0,
              top: 0,
              width,
              height,
              // Use transform for GPU-accelerated positioning (no layout thrashing)
              transform: `translate3d(${x}px, ${y}px, 0)`,
              willChange: 'transform, width, height',
              border: `2px solid ${color}`,
              borderRadius: 1,
              boxShadow: isSelected
                ? `0 0 0 2px white, 0 0 10px ${color}`
                : `0 2px 4px rgba(0,0,0,0.3)`,
            }}
          >
            {/* Track label */}
            <Box
              sx={{
                position: 'absolute',
                top: -24,
                left: 0,
                backgroundColor: color,
                color: 'white',
                px: 0.75,
                py: 0.25,
                borderRadius: 0.5,
                fontSize: '11px',
                fontWeight: 600,
                whiteSpace: 'nowrap',
              }}
            >
              Track {trackId}
            </Box>

            {/* Court position (if calibrated) */}
            {position.courtX !== undefined && position.courtY !== undefined && (
              <Box
                sx={{
                  position: 'absolute',
                  bottom: -18,
                  left: 0,
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  color: 'white',
                  px: 0.5,
                  py: 0.125,
                  borderRadius: 0.25,
                  fontSize: '9px',
                  fontFamily: 'monospace',
                }}
              >
                {position.courtX.toFixed(1)}m, {position.courtY.toFixed(1)}m
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
}
