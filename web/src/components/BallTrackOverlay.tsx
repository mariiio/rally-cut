'use client';

import { useEffect, useState, useRef, RefObject, useMemo } from 'react';
import { Box } from '@mui/material';
import type { BallPosition } from '@/services/api';

interface BallTrackOverlayProps {
  positions: BallPosition[];
  frameCount: number;
  rallyStartTime: number;
  rallyEndTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
}

export function BallTrackOverlay({
  positions,
  frameCount,
  rallyStartTime,
  rallyEndTime,
  videoRef,
}: BallTrackOverlayProps) {
  const [videoTime, setVideoTime] = useState(0);
  const rafIdRef = useRef<number | undefined>(undefined);

  // Pre-calculate the absolute time for each position
  const positionsWithTime = useMemo(() => {
    const rallyDuration = rallyEndTime - rallyStartTime;
    const maxFrame = Math.max(1, frameCount - 1);

    return positions.map((p) => ({
      ...p,
      absoluteTime: rallyStartTime + (p.frameNumber / maxFrame) * rallyDuration,
    }));
  }, [positions, frameCount, rallyStartTime, rallyEndTime]);

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

  const trailDuration = 0.5;

  // Find positions near current time (show trail of last 0.5 seconds)
  const visiblePositions = useMemo(() => {
    return positionsWithTime.filter((p) => {
      const timeDiff = videoTime - p.absoluteTime;
      return timeDiff >= 0 && timeDiff < trailDuration;
    });
  }, [positionsWithTime, videoTime]);

  // Find the current position (closest to current video time)
  const currentPosition = useMemo(() => {
    if (positionsWithTime.length === 0) return null;
    return positionsWithTime.reduce((closest, p) => {
      const closestDiff = Math.abs(closest.absoluteTime - videoTime);
      const pDiff = Math.abs(p.absoluteTime - videoTime);
      return pDiff < closestDiff ? p : closest;
    }, positionsWithTime[0]);
  }, [positionsWithTime, videoTime]);

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
        const age = videoTime - pos.absoluteTime;
        const opacity = Math.max(0.1, 1 - age / trailDuration) * pos.confidence;
        const size = 8 + (trailDuration - age) * 10;
        const isCurrent = pos.frameNumber === currentPosition?.frameNumber;

        return (
          <Box
            key={`${pos.frameNumber}-${idx}`}
            sx={{
              position: 'absolute',
              left: `${pos.x * 100}%`,
              top: `${pos.y * 100}%`,
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
    </Box>
  );
}
