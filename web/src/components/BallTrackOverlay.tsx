'use client';

import { useEffect, useRef, RefObject, useMemo } from 'react';
import type { BallPosition } from '@/services/api';

const BALL_COLOR = '#FFC107'; // Amber - consistent trail color

interface BallTrackOverlayProps {
  positions: BallPosition[];
  fps: number;
  rallyStartTime: number;
  videoRef: RefObject<HTMLVideoElement | null>;
}

const TRAIL_DURATION = 0.5; // seconds
const MAX_TRAIL_DOTS = 15; // Maximum dots to render

export function BallTrackOverlay({
  positions,
  fps,
  rallyStartTime,
  videoRef,
}: BallTrackOverlayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const dotsRef = useRef<HTMLDivElement[]>([]);

  // Pre-calculate the absolute time for each position (only when data changes)
  const positionsWithTime = useMemo(() => {
    // frameNumber is SEGMENT-RELATIVE (0-based) because tracking runs on an
    // extracted video segment. Convert to absolute video time using fps directly
    // to avoid timing drift when segment frame count doesn't exactly match
    // fps * duration.
    return positions.map((p) => ({
      ...p,
      absoluteTime: rallyStartTime + p.frameNumber / fps,
    }));
  }, [positions, fps, rallyStartTime]);

  // Create dot elements once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.innerHTML = '';
    dotsRef.current = [];

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

  // Animation loop — uses requestVideoFrameCallback for frame-accurate sync.
  // RVFC fires when a video frame is presented to the compositor, providing
  // the exact mediaTime of the displayed frame (vs RAF which can lag 1-2 frames).
  useEffect(() => {
    const video = videoRef.current;
    if (!video || dotsRef.current.length === 0) return;

    let rvfcId: number;

    const render = (videoTime: number) => {
      // Find visible positions (within trail duration)
      const visible: typeof positionsWithTime = [];
      for (const p of positionsWithTime) {
        const timeDiff = videoTime - p.absoluteTime;
        if (timeDiff >= 0 && timeDiff < TRAIL_DURATION) {
          visible.push(p);
        }
      }

      // Find current position (closest to video time within trail)
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

      // At high fps (e.g. 60fps), the trail window contains more positions than
      // MAX_TRAIL_DOTS. Show the most recent ones so the "current" dot (nearest
      // to videoTime) is always rendered — older trail dots are the ones dropped.
      const displayStart = Math.max(0, visible.length - dotsRef.current.length);
      const dots = dotsRef.current;
      for (let i = 0; i < dots.length; i++) {
        const dot = dots[i];
        const pos = visible[displayStart + i];

        if (!pos) {
          dot.style.display = 'none';
          continue;
        }

        const age = videoTime - pos.absoluteTime;
        const isCurrent = pos.frameNumber === currentFrameNumber;
        const opacity = Math.max(0.4, 1 - age / TRAIL_DURATION) * Math.max(0.6, pos.confidence);
        const size = isCurrent ? 20 : 12 + (TRAIL_DURATION - age) * 12;

        dot.style.display = 'block';
        dot.style.left = `${pos.x * 100}%`;
        dot.style.top = `${pos.y * 100}%`;
        dot.style.width = `${size}px`;
        dot.style.height = `${size}px`;
        dot.style.opacity = String(opacity);
        dot.style.backgroundColor = BALL_COLOR;
        dot.style.boxShadow = isCurrent
          ? `0 0 15px ${BALL_COLOR}, 0 0 30px ${BALL_COLOR}`
          : `0 0 8px ${BALL_COLOR}`;
        dot.style.border = isCurrent ? '3px solid white' : '1px solid rgba(255,255,255,0.5)';
      }
    };

    const onFrame = (_now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) => {
      render(metadata.mediaTime);
      rvfcId = video.requestVideoFrameCallback(onFrame);
    };

    render(video.currentTime);
    rvfcId = video.requestVideoFrameCallback(onFrame);

    return () => {
      video.cancelVideoFrameCallback(rvfcId);
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
