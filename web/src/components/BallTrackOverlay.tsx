'use client';

import { useEffect, useRef, RefObject, useMemo } from 'react';
import type { BallPosition } from '@/services/api';

interface BallTrackOverlayProps {
  positions: BallPosition[];
  frameCount: number;
  rallyStartTime: number;
  rallyEndTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
}

const TRAIL_DURATION = 0.5; // seconds
const MAX_TRAIL_DOTS = 15; // Maximum dots to render

export function BallTrackOverlay({
  positions,
  frameCount,
  rallyStartTime,
  rallyEndTime,
  videoRef,
}: BallTrackOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dotsRef = useRef<HTMLDivElement[]>([]);
  const rafIdRef = useRef<number | undefined>(undefined);

  // Pre-calculate the absolute time for each position (only when data changes)
  const positionsWithTime = useMemo(() => {
    const rallyDuration = rallyEndTime - rallyStartTime;
    const maxFrame = Math.max(1, frameCount - 1);

    return positions.map((p) => ({
      ...p,
      absoluteTime: rallyStartTime + (p.frameNumber / maxFrame) * rallyDuration,
    }));
  }, [positions, frameCount, rallyStartTime, rallyEndTime]);

  // Create dot elements once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Clear existing dots
    container.innerHTML = '';
    dotsRef.current = [];

    // Create pool of dot elements
    for (let i = 0; i < MAX_TRAIL_DOTS; i++) {
      const dot = document.createElement('div');
      dot.style.cssText = `
        position: absolute;
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        will-change: transform, opacity, width, height;
        display: none;
      `;
      container.appendChild(dot);
      dotsRef.current.push(dot);
    }
  }, []);

  // Animation loop - direct DOM manipulation, no React re-renders
  useEffect(() => {
    const updatePositions = () => {
      const video = videoRef.current;
      if (!video || dotsRef.current.length === 0) {
        rafIdRef.current = requestAnimationFrame(updatePositions);
        return;
      }

      const videoTime = video.currentTime;

      // Find visible positions (within trail duration)
      const visible: typeof positionsWithTime = [];
      for (const p of positionsWithTime) {
        const timeDiff = videoTime - p.absoluteTime;
        if (timeDiff >= 0 && timeDiff < TRAIL_DURATION) {
          visible.push(p);
        }
      }

      // Find current position (closest to video time)
      let currentFrameNumber = -1;
      if (visible.length > 0) {
        let closest = visible[0];
        for (const p of visible) {
          if (Math.abs(p.absoluteTime - videoTime) < Math.abs(closest.absoluteTime - videoTime)) {
            closest = p;
          }
        }
        currentFrameNumber = closest.frameNumber;
      }

      // Update dot elements directly
      const dots = dotsRef.current;
      for (let i = 0; i < dots.length; i++) {
        const dot = dots[i];
        const pos = visible[i];

        if (!pos) {
          dot.style.display = 'none';
          continue;
        }

        const age = videoTime - pos.absoluteTime;
        const opacity = Math.max(0.1, 1 - age / TRAIL_DURATION) * pos.confidence;
        const size = 8 + (TRAIL_DURATION - age) * 10;
        const isCurrent = pos.frameNumber === currentFrameNumber;

        dot.style.display = 'block';
        dot.style.left = `${pos.x * 100}%`;
        dot.style.top = `${pos.y * 100}%`;
        dot.style.width = `${size}px`;
        dot.style.height = `${size}px`;
        dot.style.opacity = String(opacity);
        dot.style.backgroundColor = isCurrent ? '#d32f2f' : '#ed6c02'; // error.main : warning.main
        dot.style.boxShadow = isCurrent ? '0 0 8px rgba(255,0,0,0.8)' : 'none';
      }

      rafIdRef.current = requestAnimationFrame(updatePositions);
    };

    rafIdRef.current = requestAnimationFrame(updatePositions);

    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
  }, [videoRef, positionsWithTime]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        zIndex: 15,
      }}
    />
  );
}
