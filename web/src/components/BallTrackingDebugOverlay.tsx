'use client';

import { useEffect, useState, useRef, RefObject, useMemo } from 'react';
import { Box } from '@mui/material';
import type { BallPosition } from '@/services/api';
import type { AspectRatio } from '@/types/camera';

interface BallTrackingDebugOverlayProps {
  positions: BallPosition[];
  frameCount: number;
  rallyStartTime: number;
  rallyEndTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
  // Camera state for coordinate transformation in 9:16 mode
  aspectRatio?: AspectRatio;
  cameraX?: number; // 0-1, camera center X position
  cameraY?: number; // 0-1, camera center Y position
  zoom?: number;
}

export function BallTrackingDebugOverlay({
  positions,
  frameCount,
  rallyStartTime,
  rallyEndTime,
  videoRef,
  aspectRatio = 'ORIGINAL',
  cameraX = 0.5,
  cameraY = 0.5,
  zoom = 1.0,
}: BallTrackingDebugOverlayProps) {
  const [videoTime, setVideoTime] = useState(0);
  const rafIdRef = useRef<number | undefined>(undefined);

  // Pre-calculate the absolute time for each position
  // Uses proper 0-indexed frame mapping: frame 0 → rallyStartTime, frame (N-1) → rallyEndTime
  // No sync offset needed - frame numbers directly correspond to video frames
  const positionsWithTime = useMemo(() => {
    const rallyDuration = rallyEndTime - rallyStartTime;
    // For 0-indexed frames: last frame is (frameCount - 1), maps to 100% of duration
    const maxFrame = Math.max(1, frameCount - 1);

    return positions.map((p) => ({
      ...p,
      // Frame 0 → rallyStartTime, Frame (frameCount-1) → rallyEndTime
      absoluteTime: rallyStartTime + (p.frameNumber / maxFrame) * rallyDuration,
    }));
  }, [positions, frameCount, rallyStartTime, rallyEndTime]);

  // Transform ball coordinates to match current camera view
  // Uses INVERSE of video transform from calculateVideoTransform in cameraInterpolation.ts
  // eslint-disable-next-line react-hooks/preserve-manual-memoization -- complex transform function intentionally memoized
  const transformCoordinate = useMemo(() => {
    if (aspectRatio === 'ORIGINAL') {
      // 16:9: no crop when zoom <= 1
      if (zoom <= 1) {
        return (x: number, y: number) => ({ x, y });
      }
      // At zoom > 1: video transform is scale(zoom) translate(tx%, ty%)
      // where tx = (0.5 - cameraX) * (zoom - 1) / zoom * 100
      //       ty = (0.5 - cameraY) * (zoom - 1) / zoom * 100
      // Ball at (bx, by) appears at container position:
      const panRangeX = (zoom - 1) / zoom;
      const panRangeY = (zoom - 1) / zoom;
      return (x: number, y: number) => ({
        x: 0.5 + (x - 0.5) * zoom + (0.5 - cameraX) * panRangeX,
        y: 0.5 + (y - 0.5) * zoom + (0.5 - cameraY) * panRangeY,
      });
    }

    // VERTICAL (9:16 from 16:9):
    // Video transform from calculateVideoTransform:
    //   translateX(translateX%) translateY(translateY%) scale(zoom)
    //   transformOrigin: ${originX}% 50%
    // where:
    //   widthRatio = 3.16, excessRatio = 0.684
    //   translateX = -50 + (0.5 - cameraX) * excessRatio * 100
    //   translateY = (0.5 - cameraY) * panRangeY * 100
    //   originX = 15.8 + cameraX * 68.4
    //
    // To find where ball at (bx, by) appears in container:
    // The video element spans 3.16x the container width, centered via left:50% + translateX
    // Ball position relative to video: (bx * videoWidth, by * videoHeight)
    // Need to find container coords accounting for all transforms

    const widthRatio = (16 / 9) / (9 / 16); // ~3.16

    // Horizontal: ball at video x maps to container position
    // Video spans from -1.08 to +2.08 in container coords (when centered)
    // Pan shifts this range based on cameraX
    const containerX = (x: number) => {
      // Ball at video x=0 is at video left edge
      // Ball at video x=1 is at video right edge
      // Video center (x=0.5) is panned to show cameraX
      // Container visible range: cameraX - 0.5/widthRatio/zoom to cameraX + 0.5/widthRatio/zoom
      const visibleHalfWidth = 0.5 / widthRatio / zoom;
      const visibleLeft = cameraX - visibleHalfWidth;
      const visibleRight = cameraX + visibleHalfWidth;
      // Map ball x to container 0-1
      return (x - visibleLeft) / (visibleRight - visibleLeft);
    };

    // Vertical: scale(zoom) around center, then translateY
    const panRangeY = (zoom - 1) / zoom;
    const containerY = (y: number) => {
      // Same as ORIGINAL zoom logic
      return 0.5 + (y - 0.5) * zoom + (0.5 - cameraY) * panRangeY;
    };

    return (x: number, y: number) => ({
      x: containerX(x),
      y: containerY(y),
    });
  }, [aspectRatio, cameraX, cameraY, zoom]);

  // Use RAF to read video.currentTime directly at 60fps for smooth overlay
  useEffect(() => {
    const updateTime = () => {
      const video = videoRef.current;
      if (video) {
        setVideoTime(video.currentTime);
      }
      rafIdRef.current = requestAnimationFrame(updateTime);
    };

    rafIdRef.current = requestAnimationFrame(updateTime);

    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [videoRef]);

  // Find positions near current time (show trail of last 0.5 seconds)
  const trailDuration = 0.5; // seconds
  const visiblePositions = positionsWithTime.filter((p) => {
    const timeDiff = videoTime - p.absoluteTime;
    return timeDiff >= 0 && timeDiff < trailDuration;
  });

  // Find the current position (closest to current video time)
  const currentPosition = positionsWithTime.reduce((closest, p) => {
    const closestDiff = Math.abs(closest.absoluteTime - videoTime);
    const pDiff = Math.abs(p.absoluteTime - videoTime);
    return pDiff < closestDiff ? p : closest;
  }, positionsWithTime[0]);

  const currentFrame = currentPosition?.frameNumber ?? 0;

  return (
    <Box
      sx={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 15,
      }}
    >
      {/* Trail dots */}
      {visiblePositions.map((pos, idx) => {
        const age = videoTime - pos.absoluteTime; // age in seconds
        const opacity = Math.max(0.1, 1 - age / trailDuration) * pos.confidence;
        const size = 8 + (trailDuration - age) * 10; // scale size based on age
        const isCurrent = pos.frameNumber === currentPosition?.frameNumber;

        // Transform coordinates for current camera view
        const transformed = transformCoordinate(pos.x, pos.y);

        // Skip if outside visible area (for 9:16 crop)
        if (transformed.x < 0 || transformed.x > 1 || transformed.y < 0 || transformed.y > 1) {
          return null;
        }

        return (
          <Box
            key={`${pos.frameNumber}-${idx}`}
            sx={{
              position: 'absolute',
              left: `${transformed.x * 100}%`,
              top: `${transformed.y * 100}%`,
              transform: 'translate(-50%, -50%)',
              width: size,
              height: size,
              borderRadius: '50%',
              bgcolor: isCurrent ? 'error.main' : 'warning.main',
              opacity,
              boxShadow: isCurrent ? '0 0 8px rgba(255,0,0,0.8)' : undefined,
            }}
          />
        );
      })}

      {/* Current position indicator with crosshair */}
      {currentPosition && (() => {
        const transformed = transformCoordinate(currentPosition.x, currentPosition.y);
        // Only show crosshair if position is in visible area
        if (transformed.x < 0 || transformed.x > 1 || transformed.y < 0 || transformed.y > 1) {
          return null;
        }
        return (
          <>
            <Box
              sx={{
                position: 'absolute',
                left: `${transformed.x * 100}%`,
                top: 0,
                bottom: 0,
                width: 1,
                bgcolor: 'rgba(255,255,255,0.3)',
              }}
            />
            <Box
              sx={{
                position: 'absolute',
                top: `${transformed.y * 100}%`,
                left: 0,
                right: 0,
                height: 1,
                bgcolor: 'rgba(255,255,255,0.3)',
              }}
            />
          </>
        );
      })()}

      {/* Stats overlay */}
      <Box
        sx={{
          position: 'absolute',
          top: 8,
          left: 8,
          bgcolor: 'rgba(0,0,0,0.7)',
          px: 1,
          py: 0.5,
          borderRadius: 1,
          fontSize: 11,
          fontFamily: 'monospace',
          color: 'white',
        }}
      >
        Frame: {currentFrame} / {frameCount}
        <br />
        {currentPosition && (
          <>
            Ball: ({currentPosition.x.toFixed(3)}, {currentPosition.y.toFixed(3)})
            <br />
            Confidence: {(currentPosition.confidence * 100).toFixed(0)}%
          </>
        )}
      </Box>
    </Box>
  );
}
